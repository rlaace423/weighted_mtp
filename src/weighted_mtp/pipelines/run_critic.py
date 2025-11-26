"""Critic Pre-training Runner (Stage 1)

독립 실행:
    python -m weighted_mtp.pipelines.run_critic --config configs/critic/critic.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    cleanup_s3_checkpoints,
    compute_classification_metrics_from_counts,
    compute_critic_classification_counts,
    compute_gradient_clip_stats,
    compute_gradient_norm,
    create_param_groups,
    create_scheduler,
    get_model_size,
    get_system_info,
    s3_upload_executor,
    save_checkpoint,
    shutdown_s3_executor,
)
from weighted_mtp.runtime import (
    init_distributed,
    setup_environment,
    is_main_process,
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalars,
    barrier,
)
from weighted_mtp.value_weighting.td_weighting import compute_td_targets


# ============================================================================
# Pairwise Loss 및 메트릭 함수
# ============================================================================


def pairwise_ranking_loss(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry Pairwise Ranking Loss

    P(pos > neg) = sigmoid(V_pos - V_neg)
    Loss = -log(sigmoid(V_pos - V_neg))

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        mask_pos: [batch, seq] valid token mask for positive
        mask_neg: [batch, seq] valid token mask for negative

    Returns:
        Scalar loss
    """
    # 시퀀스 평균 value 계산
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    # Pairwise ranking loss: -log(sigmoid(v_pos - v_neg))
    return -torch.nn.functional.logsigmoid(v_pos_mean - v_neg_mean).mean()


def compute_pairwise_accuracy(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> dict[str, float]:
    """Pairwise Accuracy 및 관련 메트릭 계산

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        mask_pos: [batch, seq] valid token mask for positive
        mask_neg: [batch, seq] valid token mask for negative

    Returns:
        {pairwise_accuracy, mean_pos, mean_neg, margin}
    """
    # 시퀀스 평균 value 계산
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    # V(correct) > V(incorrect)인 쌍의 비율
    correct_pairs = (v_pos_mean > v_neg_mean).float().sum()
    total_pairs = v_pos_mean.size(0)
    pairwise_accuracy = (correct_pairs / total_pairs).item()

    # 평균 값
    mean_pos = v_pos_mean.mean().item()
    mean_neg = v_neg_mean.mean().item()
    margin = mean_pos - mean_neg

    return {
        "pairwise_accuracy": pairwise_accuracy,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "margin": margin,
        "correct_pairs": correct_pairs.item(),
        "total_pairs": total_pairs,
    }


def load_adapter(config: dict, device: torch.device) -> MetaLlamaMTPAdapter:
    """Adapter 로드

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        MetaLlamaMTPAdapter 인스턴스
    """
    # value_head_type 설정 (기본값: mlp)
    value_head_type = getattr(config.training, "value_head_type", "mlp")
    # dropout 설정 (기본값: 0.0)
    dropout = getattr(config.training, "dropout", 0.0)

    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=device,
        dtype=config.models.policy.dtype,
        value_head_type=value_head_type,
        dropout=dropout,
    )
    return adapter




def validate_critic(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
    gamma: float = 1.0,
    lam: float = 0.0,
    value_head_type: str = "mlp",
    return_raw_counts: bool = False,
) -> dict[str, float]:
    """Validation 수행

    Args:
        adapter: Adapter
        dataloader: Validation DataLoader
        device: 디바이스
        gamma: TD discount factor
        lam: GAE lambda
        value_head_type: "linear" (BCE) 또는 "mlp" (MSE)
        return_raw_counts: True이면 raw counts 반환 (분산학습용 aggregation)

    Returns:
        Validation metrics (또는 return_raw_counts=True이면 raw counts 포함)
    """
    adapter.eval()

    total_loss = 0.0
    n_batches = 0

    # 분류 메트릭용 누적 변수 (micro-average)
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_correct_sum = 0.0
    total_correct_count = 0.0
    total_incorrect_sum = 0.0
    total_incorrect_count = 0.0

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            is_correct = batch["is_correct"].to(device)

            # 2. is_correct → rewards 변환 (모델 dtype 일치)
            rewards = is_correct.to(model_dtype)

            # 3. Forward (Value head만)
            outputs = adapter(input_ids, attention_mask, return_value_logits=True)
            value_logits = outputs["value_logits"]

            # 4. TD target 계산
            td_targets = compute_td_targets(
                value_logits=value_logits,
                rewards=rewards,
                attention_mask=attention_mask,
                gamma=gamma,
                lam=lam,
            )

            # Mask padded tokens AND instruction tokens (labels != -100)
            valid_label_mask = (labels != -100).unsqueeze(-1).to(model_dtype)

            # 5. Value loss 계산 (value_head_type에 따라 분기)
            if value_head_type == "sigmoid":
                # BCE loss (sigmoid 출력)
                loss_per_token = torch.nn.functional.binary_cross_entropy(
                    value_logits, td_targets, reduction="none"
                )
            else:
                # MSE loss (linear, mlp)
                loss_per_token = torch.nn.functional.mse_loss(
                    value_logits, td_targets, reduction="none"
                )

            masked_loss = loss_per_token * valid_label_mask
            value_loss = masked_loss.sum() / (valid_label_mask.sum() + 1e-8)

            total_loss += value_loss.item()

            # 6. 분류 메트릭 count 누적 (micro-average)
            valid_mask_2d = (labels != -100).to(model_dtype)
            cls_counts = compute_critic_classification_counts(
                value_logits=value_logits,
                is_correct=is_correct,
                attention_mask=valid_mask_2d,
            )
            total_tp += cls_counts["tp"]
            total_fp += cls_counts["fp"]
            total_fn += cls_counts["fn"]
            total_correct_sum += cls_counts["correct_sum"]
            total_correct_count += cls_counts["correct_count"]
            total_incorrect_sum += cls_counts["incorrect_sum"]
            total_incorrect_count += cls_counts["incorrect_count"]

            n_batches += 1

    # 평균 loss 계산
    avg_loss = total_loss / n_batches

    # 분류 메트릭 최종 계산 (micro-average)
    cls_metrics = compute_classification_metrics_from_counts(
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        correct_sum=total_correct_sum,
        correct_count=total_correct_count,
        incorrect_sum=total_incorrect_sum,
        incorrect_count=total_incorrect_count,
    )

    metrics = {
        "val_loss": avg_loss,
        "val_pred_gap": cls_metrics["pred_gap"],
        "val_mean_correct": cls_metrics["mean_correct"],
        "val_mean_incorrect": cls_metrics["mean_incorrect"],
        "val_precision": cls_metrics["precision"],
        "val_recall": cls_metrics["recall"],
        "val_f1": cls_metrics["f1"],
    }

    # 분산학습용: raw counts 포함 반환
    if return_raw_counts:
        metrics["_raw_counts"] = {
            "loss_sum": total_loss,
            "n_batches": n_batches,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "correct_sum": total_correct_sum,
            "correct_count": total_correct_count,
            "incorrect_sum": total_incorrect_sum,
            "incorrect_count": total_incorrect_count,
        }

    return metrics


def validate_critic_pairwise(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
    return_raw_counts: bool = False,
) -> dict[str, float]:
    """Pairwise Validation 수행

    Args:
        adapter: Adapter
        dataloader: Validation DataLoader (pairwise format)
        device: 디바이스
        return_raw_counts: True이면 raw counts 반환 (분산학습용 aggregation)

    Returns:
        Validation metrics (pairwise_accuracy, mean_pos, mean_neg, margin, loss)
    """
    adapter.eval()

    total_loss = 0.0
    n_batches = 0
    total_correct_pairs = 0.0
    total_pairs = 0
    total_mean_pos = 0.0
    total_mean_neg = 0.0

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

    with torch.no_grad():
        for batch in dataloader:
            # Pairwise batch 구조
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)
            neg_labels = batch["neg_labels"].to(device)

            # Forward
            pos_outputs = adapter(pos_input_ids, pos_attention_mask, return_value_logits=True)
            neg_outputs = adapter(neg_input_ids, neg_attention_mask, return_value_logits=True)

            pos_value_logits = pos_outputs["value_logits"]
            neg_value_logits = neg_outputs["value_logits"]

            # Mask (labels != -100)
            pos_mask = (pos_labels != -100).to(model_dtype)
            neg_mask = (neg_labels != -100).to(model_dtype)

            # Loss 계산
            loss = pairwise_ranking_loss(
                v_pos=pos_value_logits,
                v_neg=neg_value_logits,
                mask_pos=pos_mask,
                mask_neg=neg_mask,
            )

            # Pairwise accuracy 계산
            pairwise_metrics = compute_pairwise_accuracy(
                v_pos=pos_value_logits,
                v_neg=neg_value_logits,
                mask_pos=pos_mask,
                mask_neg=neg_mask,
            )

            total_loss += loss.item()
            total_correct_pairs += pairwise_metrics["correct_pairs"]
            total_pairs += pairwise_metrics["total_pairs"]
            total_mean_pos += pairwise_metrics["mean_pos"]
            total_mean_neg += pairwise_metrics["mean_neg"]
            n_batches += 1

    # 평균 계산
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    avg_pairwise_accuracy = total_correct_pairs / total_pairs if total_pairs > 0 else 0.0
    avg_mean_pos = total_mean_pos / n_batches if n_batches > 0 else 0.0
    avg_mean_neg = total_mean_neg / n_batches if n_batches > 0 else 0.0
    avg_margin = avg_mean_pos - avg_mean_neg

    metrics = {
        "val_loss": avg_loss,
        "val_pairwise_accuracy": avg_pairwise_accuracy,
        "val_mean_pos": avg_mean_pos,
        "val_mean_neg": avg_mean_neg,
        "val_margin": avg_margin,
    }

    # 분산학습용: raw counts 포함 반환
    if return_raw_counts:
        metrics["_raw_counts"] = {
            "loss_sum": total_loss,
            "n_batches": n_batches,
            "correct_pairs": total_correct_pairs,
            "total_pairs": total_pairs,
            "mean_pos_sum": total_mean_pos,
            "mean_neg_sum": total_mean_neg,
        }

    return metrics


def run_critic_training(config: DictConfig) -> tuple[dict[str, float], str]:
    """Critic pre-training 실행

    Args:
        config: 완전한 config 객체 (OmegaConf DictConfig)

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 0. 환경변수 로드 (MLflow credentials 등)
    ensure_env_loaded()

    # 2. Distributed 초기화 (torchrun 환경인 경우)
    if "RANK" in os.environ:
        rank, world_size = init_distributed()
    else:
        rank, world_size = 0, 1

    # 3. 로깅 설정 (rank 정보 포함)
    logger = setup_logging("CRITIC", level=config.logging.level, rank=rank)

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    logger.info("=== Critic Pre-training (Stage 1) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 5. MLflow 초기화 (Rank 0만, experiment 이름이 있는 경우만)
    use_mlflow = bool(config.mlflow.experiment)
    use_s3_upload = config.checkpoint.get("s3_upload", True) and use_mlflow
    mlflow_run_id = None  # S3 업로드 시 스레드 안전을 위해 명시적 run_id 저장
    if is_main_process() and use_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: "true" for tag in config.experiment.tags},
        )
        # Config 로깅
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        mlflow_run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {mlflow_run_id}")

    # 6. Resource 로딩
    adapter = load_adapter(config, device)

    # Transformer trunk freeze 설정
    num_unfrozen = config.training.get("num_unfrozen_layers", 0)
    n_layers = len(adapter.transformer.layers)

    if num_unfrozen > 0:
        # 마지막 N개 블록만 학습
        logger.info(f"Unfreezing last {num_unfrozen} transformer blocks (out of {n_layers})")

        # 전체 frozen
        for param in adapter.transformer.parameters():
            param.requires_grad = False

        # 마지막 N개 블록 unfreeze
        for layer in adapter.transformer.layers[-num_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True

        # final norm도 unfreeze (마지막 블록 출력에 영향)
        for param in adapter.transformer.norm.parameters():
            param.requires_grad = True
    else:
        # 기존 동작: value head만 학습
        logger.info("Freezing transformer trunk (training value head only)")
        for param in adapter.transformer.parameters():
            param.requires_grad = False

    # Value head는 항상 학습
    for param in adapter.value_head.parameters():
        param.requires_grad = True

    adapter = wrap_model_fsdp(
        adapter,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )
    tokenizer = load_tokenizer_from_config(config)

    # Model size 로깅
    model_size_local = get_model_size(unwrap_model(adapter))

    # FSDP FULL_SHARD 시 world_size를 곱해 실제 전체 파라미터 수 계산
    sharding_strategy = config.distributed.fsdp.sharding_strategy
    if sharding_strategy == "FULL_SHARD" and world_size > 1:
        model_size = {
            "total_params": model_size_local["total_params"] * world_size,
            "trainable_params": model_size_local["trainable_params"] * world_size,
            "non_trainable_params": model_size_local["non_trainable_params"] * world_size,
        }
    else:
        model_size = model_size_local

    # Trainable params breakdown 계산 (FSDP sharding 고려)
    trainable_breakdown_local = {
        "value_head": sum(p.numel() for p in adapter.value_head.parameters() if p.requires_grad),
        "trunk_blocks": sum(
            p.numel() for layer in adapter.transformer.layers[-num_unfrozen:]
            for p in layer.parameters() if p.requires_grad
        ) if num_unfrozen > 0 else 0,
        "norm": sum(
            p.numel() for p in adapter.transformer.norm.parameters() if p.requires_grad
        ) if num_unfrozen > 0 else 0,
    }

    if sharding_strategy == "FULL_SHARD" and world_size > 1:
        trainable_breakdown = {
            "value_head": trainable_breakdown_local["value_head"] * world_size,
            "trunk_blocks": trainable_breakdown_local["trunk_blocks"] * world_size,
            "norm": trainable_breakdown_local["norm"] * world_size,
        }
    else:
        trainable_breakdown = trainable_breakdown_local

    if is_main_process():
        if use_mlflow:
            mlflow.log_params(
                {
                    "model_total_params": model_size["total_params"],
                    "model_trainable_params": model_size["trainable_params"],
                    "model_non_trainable_params": model_size["non_trainable_params"],
                    "model_num_unfrozen_layers": num_unfrozen,
                    "model_trainable_value_head": trainable_breakdown["value_head"],
                    "model_trainable_trunk_blocks": trainable_breakdown["trunk_blocks"],
                    "model_trainable_norm": trainable_breakdown["norm"],
                }
            )
        logger.info(
            f"Model size: {model_size['trainable_params']:,} trainable / "
            f"{model_size['total_params']:,} total params"
        )
        logger.info(
            f"Trainable breakdown - value_head: {trainable_breakdown['value_head']:,}, "
            f"trunk_blocks: {trainable_breakdown['trunk_blocks']:,}, "
            f"norm: {trainable_breakdown['norm']:,}"
        )

        # System info 로깅
        system_info = get_system_info()
        if use_mlflow:
            mlflow.log_params(
                {
                    "system_cpu_count": system_info["cpu_count"],
                    "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
                }
            )

    # GPU monitor 초기화 (모든 rank에서 필요)
    gpu_monitor = GPUMonitor(device)

    # 5. Dataset & DataLoader 생성
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    # sampling_config를 dict로 변환
    sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
    use_pairwise = sampling_config.get("use_pairwise", False)
    logger.info(f"Pairwise 모드: {use_pairwise}")

    train_loader = create_dataloader(
        dataset_path=config.dataset.train,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        sampling_config=sampling_config,
        seed=config.data_sampling.seed,
        shuffle=True,
    )

    # Validation용 sampling_config (val_n_samples 적용)
    val_sampling_config = sampling_config.copy()
    val_sampling_config["n_samples"] = config.data_sampling.val_n_samples

    val_loader = create_dataloader(
        dataset_path=config.dataset.validation,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        sampling_config=val_sampling_config,
        seed=config.data_sampling.seed,
        shuffle=False,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Dataset statistics 로깅
    if is_main_process() and use_mlflow:
        mlflow.log_params(
            {
                "dataset_train_samples": len(train_loader.dataset),
                "dataset_val_samples": len(val_loader.dataset),
                "dataset_train_batches": len(train_loader),
                "dataset_val_batches": len(val_loader),
            }
        )

    # 6. Optimizer (param groups: trunk/value_head 분리)
    # learning_rate fallback (하위 호환성)
    default_lr = config.training.get("learning_rate", 1e-5)
    trunk_lr = config.training.get("trunk_learning_rate", default_lr)
    value_head_lr = config.training.get("value_head_learning_rate", default_lr)

    # FSDP wrapping 후에도 원본 모델 구조에 접근하여 param groups 생성
    param_groups = create_param_groups(
        adapter=unwrap_model(adapter),
        trunk_lr=trunk_lr,
        value_head_lr=value_head_lr,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # 7. Training loop
    best_val_loss = float("inf")
    global_step = 0

    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = config.training.n_epochs
    save_checkpoint_every = config.checkpoint.save_checkpoint_every

    # Fractional epoch 처리
    total_batches = len(train_loader)
    batches_to_run = int(total_batches * n_epochs)

    # Gradient accumulation 초기화
    accumulation_counter = 0
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    # Optimization steps 계산
    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_optimization_steps}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")

    # Learning rate scheduler 생성
    lr_scheduler_config = config.training.get("lr_scheduler", {})
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_optimization_steps,
        scheduler_type=lr_scheduler_config.get("type", "constant"),
        warmup_ratio=lr_scheduler_config.get("warmup_ratio", 0.05),
        min_lr_ratio=lr_scheduler_config.get("min_lr_ratio", 0.0),
    )

    current_epoch = 0.0
    batch_count = 0
    next_checkpoint_epoch = save_checkpoint_every
    train_loss_avg = 0.0  # 초기화 (0 batch 케이스 대응)

    # Throughput tracker 초기화 (모든 rank에서 필요)
    throughput_tracker = ThroughputTracker()

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

    # Optimizer 초기화 (gradient accumulation을 위해 while loop 시작 전)
    optimizer.zero_grad()

    # 7. Training loop (모든 rank 실행)
    while batch_count < batches_to_run:
        # Train 1 epoch (또는 checkpoint 경계까지)
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        # Throughput tracking 시작
        throughput_tracker.start_epoch()

        # DataLoader에서 필요한 만큼만 사용
        epoch_train_loader = iter(train_loader)
        period_metrics_sum = {"train_loss": 0.0}
        period_batches = 0

        # Train 분류 메트릭용 누적 변수 (micro-average)
        train_tp = 0.0
        train_fp = 0.0
        train_fn = 0.0
        train_correct_sum = 0.0
        train_correct_count = 0.0
        train_incorrect_sum = 0.0
        train_incorrect_count = 0.0

        # Pairwise 메트릭용 변수
        train_correct_pairs = 0.0
        train_total_pairs = 0
        train_mean_pos_sum = 0.0
        train_mean_neg_sum = 0.0

        # Pairwise 모드 여부
        is_pairwise = use_pairwise

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                # DataLoader 재시작
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련
            adapter.train()

            # Pairwise 모드와 Pointwise 모드 분기
            if is_pairwise:
                # Pairwise batch 구조
                pos_input_ids = batch["pos_input_ids"].to(device)
                pos_attention_mask = batch["pos_attention_mask"].to(device)
                pos_labels = batch["pos_labels"].to(device)
                neg_input_ids = batch["neg_input_ids"].to(device)
                neg_attention_mask = batch["neg_attention_mask"].to(device)
                neg_labels = batch["neg_labels"].to(device)

                # Forward
                pos_outputs = adapter(pos_input_ids, pos_attention_mask, return_value_logits=True)
                neg_outputs = adapter(neg_input_ids, neg_attention_mask, return_value_logits=True)

                pos_value_logits = pos_outputs["value_logits"]
                neg_value_logits = neg_outputs["value_logits"]

                # Mask (labels != -100)
                pos_mask = (pos_labels != -100).to(model_dtype)
                neg_mask = (neg_labels != -100).to(model_dtype)

                # Pairwise ranking loss
                value_loss = pairwise_ranking_loss(
                    v_pos=pos_value_logits,
                    v_neg=neg_value_logits,
                    mask_pos=pos_mask,
                    mask_neg=neg_mask,
                )

                # Pairwise 메트릭 누적
                pairwise_metrics = compute_pairwise_accuracy(
                    v_pos=pos_value_logits,
                    v_neg=neg_value_logits,
                    mask_pos=pos_mask,
                    mask_neg=neg_mask,
                )
                train_correct_pairs += pairwise_metrics["correct_pairs"]
                train_total_pairs += pairwise_metrics["total_pairs"]
                train_mean_pos_sum += pairwise_metrics["mean_pos"]
                train_mean_neg_sum += pairwise_metrics["mean_neg"]

                # Throughput용 변수
                batch_size_actual = pos_input_ids.size(0) * 2  # pos + neg
                n_tokens = pos_attention_mask.sum().item() + neg_attention_mask.sum().item()
                value_logits = pos_value_logits  # 로깅용
                valid_label_mask = pos_mask.unsqueeze(-1)  # 로깅용 placeholder
                is_correct = torch.ones(pos_input_ids.size(0), device=device)  # placeholder

            else:
                # 기존 Pointwise 모드
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                is_correct = batch["is_correct"].to(device)

                # 모델 dtype 일치
                rewards = is_correct.to(model_dtype)
                outputs = adapter(input_ids, attention_mask, return_value_logits=True)
                value_logits = outputs["value_logits"]

                batch_size, seq_len, _ = value_logits.shape

                # TD target 계산 (MC 방식 삭제, TD Learning 적용)
                td_targets = compute_td_targets(
                    value_logits=value_logits,
                    rewards=rewards,
                    attention_mask=attention_mask,
                    gamma=config.training.gamma,
                    lam=getattr(config.training, "lam", 0.0),
                )

                # Mask padded tokens AND instruction tokens (labels != -100)
                valid_label_mask = (labels != -100).unsqueeze(-1).to(model_dtype)
                attn_mask_expanded = attention_mask.unsqueeze(-1).to(model_dtype)
                loss_mask = valid_label_mask * attn_mask_expanded

                # Value loss 계산 (value_head_type에 따라 분기)
                value_head_type = config.training.value_head_type
                if value_head_type == "sigmoid":
                    # BCE loss (sigmoid 출력)
                    loss_per_token = torch.nn.functional.binary_cross_entropy(
                        value_logits, td_targets, reduction="none"
                    )
                else:
                    # MSE loss (linear, mlp)
                    loss_per_token = torch.nn.functional.mse_loss(
                        value_logits, td_targets, reduction="none"
                    )

                masked_loss = loss_per_token * loss_mask
                value_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

                # Throughput용 변수
                batch_size_actual = input_ids.size(0)
                n_tokens = attention_mask.sum().item()

            # Loss scaling (gradient accumulation 적용)
            scaled_loss = value_loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Throughput tracking (batch 단위)
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            # Period metrics 누적 (batch 단위)
            period_metrics_sum["train_loss"] += value_loss.item()

            # Train 분류 메트릭 count 누적 (micro-average) - Pointwise 모드에서만
            if not is_pairwise:
                valid_mask_2d = (labels != -100).to(model_dtype)
                cls_counts = compute_critic_classification_counts(
                    value_logits=value_logits,
                    is_correct=is_correct,
                    attention_mask=valid_mask_2d,
                )
                train_tp += cls_counts["tp"]
                train_fp += cls_counts["fp"]
                train_fn += cls_counts["fn"]
                train_correct_sum += cls_counts["correct_sum"]
                train_correct_count += cls_counts["correct_count"]
                train_incorrect_sum += cls_counts["incorrect_sum"]
                train_incorrect_count += cls_counts["incorrect_count"]

            # Optimizer step (accumulation 완료 시에만)
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping (누적된 gradient에 적용)
                if config.training.max_grad_norm > 0:
                    # DEBUG: max_grad_norm 값 확인
                    if global_step == 0:
                        logger.info(f"DEBUG: config.training.max_grad_norm = {config.training.max_grad_norm}")

                    grad_clip_stats = compute_gradient_clip_stats(
                        adapter,
                        config.training.max_grad_norm,
                    )
                else:
                    grad_norm_dict = compute_gradient_norm(adapter)
                    grad_clip_stats = {
                        "grad_norm_pre_clip": grad_norm_dict["grad_norm"],
                        "grad_norm_post_clip": grad_norm_dict["grad_norm"],
                        "grad_clip_ratio": 1.0,
                    }

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                accumulation_counter = 0

                # Step-level logging (optimizer step 시에만)
                if global_step % config.training.log_interval == 0:
                    # GPU metrics
                    gpu_metrics = gpu_monitor.get_metrics()

                    # Metric aggregation (분산 환경)
                    avg_grad_norm_post = grad_clip_stats["grad_norm_post_clip"]

                    # Value head LR 가져오기 (param_groups[1]이 value_head)
                    value_head_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]

                    if is_pairwise:
                        # Pairwise 모드: pairwise accuracy, mean_pos, mean_neg
                        batch_pairwise_acc = pairwise_metrics["pairwise_accuracy"]
                        batch_mean_pos = pairwise_metrics["mean_pos"]
                        batch_mean_neg = pairwise_metrics["mean_neg"]

                        reduced = all_reduce_scalars({
                            "loss": value_loss.item(),
                            "pairwise_accuracy": batch_pairwise_acc,
                            "mean_pos": batch_mean_pos,
                            "mean_neg": batch_mean_neg,
                        })
                        avg_loss = reduced["loss"]
                        avg_pairwise_acc = reduced["pairwise_accuracy"]
                        avg_mean_pos = reduced["mean_pos"]
                        avg_mean_neg = reduced["mean_neg"]

                        if is_main_process():
                            if use_mlflow:
                                mlflow.log_metrics(
                                    {
                                        "train/loss": avg_loss,
                                        "train/grad_norm": avg_grad_norm_post,
                                        "train/learning_rate": value_head_lr,
                                        "train/pairwise_accuracy": avg_pairwise_acc,
                                        "value/mean_pos": avg_mean_pos,
                                        "value/mean_neg": avg_mean_neg,
                                        "value/margin": avg_mean_pos - avg_mean_neg,
                                        "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                        "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                                    },
                                    step=global_step,
                                )
                        logger.info(
                            f"Step {global_step}/{total_optimization_steps}, "
                            f"Loss: {avg_loss:.4f}, "
                            f"Pairwise Acc: {avg_pairwise_acc:.3f}, "
                            f"Margin: {avg_mean_pos - avg_mean_neg:.4f}, "
                            f"LR: {value_head_lr:.2e}"
                        )
                    else:
                        # Pointwise 모드: mean_correct, mean_incorrect
                        # Value prediction std 계산 (유효 토큰만)
                        valid_values = value_logits[valid_label_mask.bool()]
                        value_std = valid_values.std().item() if valid_values.numel() > 0 else 0.0

                        # 배치 단위 mean_correct/mean_incorrect 계산
                        batch_correct_mask = is_correct.bool()
                        batch_incorrect_mask = ~batch_correct_mask
                        valid_mask_2d_step = valid_label_mask.squeeze(-1).bool()

                        # correct 시퀀스 내 토큰 평균
                        correct_token_mask = batch_correct_mask.view(-1, 1).expand_as(valid_mask_2d_step) & valid_mask_2d_step
                        batch_mean_correct = value_logits.squeeze(-1)[correct_token_mask].mean().item() if correct_token_mask.any() else 0.0

                        # incorrect 시퀀스 내 토큰 평균
                        incorrect_token_mask = batch_incorrect_mask.view(-1, 1).expand_as(valid_mask_2d_step) & valid_mask_2d_step
                        batch_mean_incorrect = value_logits.squeeze(-1)[incorrect_token_mask].mean().item() if incorrect_token_mask.any() else 0.0

                        # Loss, std, mean_correct/incorrect all_reduce (1회 통신)
                        reduced = all_reduce_scalars({
                            "loss": value_loss.item(),
                            "value_std": value_std,
                            "mean_correct": batch_mean_correct,
                            "mean_incorrect": batch_mean_incorrect,
                        })
                        avg_loss = reduced["loss"]
                        avg_value_std = reduced["value_std"]
                        avg_mean_correct = reduced["mean_correct"]
                        avg_mean_incorrect = reduced["mean_incorrect"]

                        if is_main_process():
                            if use_mlflow:
                                mlflow.log_metrics(
                                    {
                                        "train/loss": avg_loss,
                                        "train/grad_norm": avg_grad_norm_post,
                                        "train/learning_rate": value_head_lr,
                                        "value/std": avg_value_std,
                                        "value/mean_correct": avg_mean_correct,
                                        "value/mean_incorrect": avg_mean_incorrect,
                                        "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                        "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                                    },
                                    step=global_step,
                                )
                        logger.info(
                            f"Step {global_step}/{total_optimization_steps}, "
                            f"Loss: {avg_loss:.4f}, "
                            f"Grad Norm: {avg_grad_norm_post:.4f}, "
                            f"LR: {value_head_lr:.2e}"
                        )

        # Period loop 종료

        # Incomplete accumulation 처리 (validation 전)
        if accumulation_counter > 0:
            logger.info(f"Processing incomplete accumulation ({accumulation_counter} batches before validation)")

            # Gradient clipping
            if config.training.max_grad_norm > 0:
                grad_clip_stats = compute_gradient_clip_stats(
                    adapter,
                    config.training.max_grad_norm,
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0

        # Epoch 경계 도달
        current_epoch = batch_count / total_batches

        # Period-level metrics 계산
        train_loss_avg = period_metrics_sum["train_loss"] / period_batches

        # Throughput metrics 계산
        throughput_metrics = throughput_tracker.get_epoch_metrics()

        # GPU metrics (epoch-level)
        gpu_metrics_epoch = gpu_monitor.get_metrics()

        if is_pairwise:
            # Pairwise 모드: pairwise accuracy 기반 메트릭
            reduced_train_pairwise = all_reduce_scalars({
                "train_loss": train_loss_avg,
                "train_correct_pairs": train_correct_pairs,
                "train_total_pairs": train_total_pairs,
                "train_mean_pos_sum": train_mean_pos_sum,
                "train_mean_neg_sum": train_mean_neg_sum,
            }, op="sum")

            world_sz = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            train_loss_avg = reduced_train_pairwise["train_loss"] / max(1, world_sz)
            train_pairwise_acc = reduced_train_pairwise["train_correct_pairs"] / max(1, reduced_train_pairwise["train_total_pairs"])
            train_mean_pos = reduced_train_pairwise["train_mean_pos_sum"] / max(1, period_batches * world_sz)
            train_mean_neg = reduced_train_pairwise["train_mean_neg_sum"] / max(1, period_batches * world_sz)
            train_margin = train_mean_pos - train_mean_neg

            logger.info(
                f"Epoch {current_epoch:.2f} 도달 - "
                f"Train Loss: {train_loss_avg:.4f}, "
                f"Pairwise Acc: {train_pairwise_acc:.3f}, "
                f"Margin: {train_margin:.4f}"
            )

            # Validation (Pairwise)
            logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

            val_metrics = validate_critic_pairwise(
                adapter=adapter,
                dataloader=val_loader,
                device=device,
                return_raw_counts=True,
            )

            # Validation raw counts all_reduce
            raw_counts = val_metrics["_raw_counts"]
            reduced_val_counts = all_reduce_scalars({
                "loss_sum": raw_counts["loss_sum"],
                "n_batches": raw_counts["n_batches"],
                "correct_pairs": raw_counts["correct_pairs"],
                "total_pairs": raw_counts["total_pairs"],
                "mean_pos_sum": raw_counts["mean_pos_sum"],
                "mean_neg_sum": raw_counts["mean_neg_sum"],
            }, op="sum")

            avg_val_loss = reduced_val_counts["loss_sum"] / max(1, reduced_val_counts["n_batches"])
            avg_val_pairwise_acc = reduced_val_counts["correct_pairs"] / max(1, reduced_val_counts["total_pairs"])
            avg_val_mean_pos = reduced_val_counts["mean_pos_sum"] / max(1, reduced_val_counts["n_batches"])
            avg_val_mean_neg = reduced_val_counts["mean_neg_sum"] / max(1, reduced_val_counts["n_batches"])
            avg_val_margin = avg_val_mean_pos - avg_val_mean_neg

            # Epoch-level 로깅 (Pairwise)
            if is_main_process():
                if use_mlflow:
                    mlflow.log_metrics(
                        {
                            # Train metrics
                            "train/epoch_loss": train_loss_avg,
                            "train/pairwise_accuracy": train_pairwise_acc,
                            "train/mean_pos": train_mean_pos,
                            "train/mean_neg": train_mean_neg,
                            "train/margin": train_margin,
                            # Validation metrics
                            "val/loss": avg_val_loss,
                            "val/pairwise_accuracy": avg_val_pairwise_acc,
                            "val/mean_pos": avg_val_mean_pos,
                            "val/mean_neg": avg_val_mean_neg,
                            "val/margin": avg_val_margin,
                            # Performance metrics
                            "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                            "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                            "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                            "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                        },
                        step=global_step,
                    )

            logger.info(
                f"Validation - Loss: {avg_val_loss:.4f}, "
                f"Pairwise Acc: {avg_val_pairwise_acc:.3f}, "
                f"Margin: {avg_val_margin:.4f}"
            )

            # Aggregated validation metrics (checkpoint 저장용)
            aggregated_val_metrics = {
                "val_loss": avg_val_loss,
                "val_pairwise_accuracy": avg_val_pairwise_acc,
                "val_mean_pos": avg_val_mean_pos,
                "val_mean_neg": avg_val_mean_neg,
                "val_margin": avg_val_margin,
            }

        else:
            # Pointwise 모드: classification 기반 메트릭
            reduced_train = all_reduce_scalars({
                "train_loss": train_loss_avg,
                "train_tp": train_tp,
                "train_fp": train_fp,
                "train_fn": train_fn,
                "train_correct_sum": train_correct_sum,
                "train_correct_count": train_correct_count,
                "train_incorrect_sum": train_incorrect_sum,
                "train_incorrect_count": train_incorrect_count,
            }, op="sum")
            train_loss_avg = reduced_train["train_loss"] / max(1, torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

            train_cls_metrics = compute_classification_metrics_from_counts(
                tp=reduced_train["train_tp"],
                fp=reduced_train["train_fp"],
                fn=reduced_train["train_fn"],
                correct_sum=reduced_train["train_correct_sum"],
                correct_count=reduced_train["train_correct_count"],
                incorrect_sum=reduced_train["train_incorrect_sum"],
                incorrect_count=reduced_train["train_incorrect_count"],
            )

            logger.info(
                f"Epoch {current_epoch:.2f} 도달 - "
                f"Train Loss: {train_loss_avg:.4f}, "
                f"Pred Gap: {train_cls_metrics['pred_gap']:.4f}, "
                f"F1: {train_cls_metrics['f1']:.3f}"
            )

            # Validation (Pointwise)
            logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

            val_metrics = validate_critic(
                adapter=adapter,
                dataloader=val_loader,
                device=device,
                gamma=config.training.gamma,
                lam=getattr(config.training, "lam", 0.0),
                value_head_type=config.training.value_head_type,
                return_raw_counts=True,
            )

            raw_counts = val_metrics["_raw_counts"]
            reduced_val_counts = all_reduce_scalars({
                "loss_sum": raw_counts["loss_sum"],
                "n_batches": raw_counts["n_batches"],
                "tp": raw_counts["tp"],
                "fp": raw_counts["fp"],
                "fn": raw_counts["fn"],
                "correct_sum": raw_counts["correct_sum"],
                "correct_count": raw_counts["correct_count"],
                "incorrect_sum": raw_counts["incorrect_sum"],
                "incorrect_count": raw_counts["incorrect_count"],
            }, op="sum")

            avg_val_loss = reduced_val_counts["loss_sum"] / max(1, reduced_val_counts["n_batches"])
            val_cls_metrics = compute_classification_metrics_from_counts(
                tp=reduced_val_counts["tp"],
                fp=reduced_val_counts["fp"],
                fn=reduced_val_counts["fn"],
                correct_sum=reduced_val_counts["correct_sum"],
                correct_count=reduced_val_counts["correct_count"],
                incorrect_sum=reduced_val_counts["incorrect_sum"],
                incorrect_count=reduced_val_counts["incorrect_count"],
            )
            avg_val_pred_gap = val_cls_metrics["pred_gap"]
            avg_val_mean_correct = val_cls_metrics["mean_correct"]
            avg_val_mean_incorrect = val_cls_metrics["mean_incorrect"]
            avg_val_precision = val_cls_metrics["precision"]
            avg_val_recall = val_cls_metrics["recall"]
            avg_val_f1 = val_cls_metrics["f1"]

            # Epoch-level 로깅 (Pointwise)
            if is_main_process():
                if use_mlflow:
                    mlflow.log_metrics(
                        {
                            # Train metrics
                            "train/epoch_loss": train_loss_avg,
                            "train/mean_correct": train_cls_metrics["mean_correct"],
                            "train/mean_incorrect": train_cls_metrics["mean_incorrect"],
                            "train/precision": train_cls_metrics["precision"],
                            "train/recall": train_cls_metrics["recall"],
                            "train/f1": train_cls_metrics["f1"],
                            # Validation metrics
                            "val/loss": avg_val_loss,
                            "val/mean_correct": avg_val_mean_correct,
                            "val/mean_incorrect": avg_val_mean_incorrect,
                            "val/precision": avg_val_precision,
                            "val/recall": avg_val_recall,
                            "val/f1": avg_val_f1,
                            # Performance metrics
                            "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                            "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                            "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                            "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                        },
                        step=global_step,
                    )

            logger.info(
                f"Validation - Loss: {avg_val_loss:.4f}, "
                f"Pred Gap: {avg_val_pred_gap:.4f}, "
                f"F1: {avg_val_f1:.3f}"
            )

            # Aggregated validation metrics (checkpoint 저장용)
            aggregated_val_metrics = {
                "val_loss": avg_val_loss,
                "val_pred_gap": avg_val_pred_gap,
                "val_mean_correct": avg_val_mean_correct,
                "val_mean_incorrect": avg_val_mean_incorrect,
                "val_precision": avg_val_precision,
                "val_recall": avg_val_recall,
                "val_f1": avg_val_f1,
            }

        # Checkpoint 저장 (validation loss 개선 시만)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={"train_loss": train_loss_avg},
                val_metrics=aggregated_val_metrics,
                checkpoint_path=checkpoint_path,
                config={"model": {"path": config.models.policy.path}},
                s3_upload=use_s3_upload,
                mlflow_run_id=mlflow_run_id,
            )

            # 모든 GPU가 checkpoint 저장 완료까지 대기
            barrier()

            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

                # 오래된 checkpoint 정리
                if config.checkpoint.save_total_limit:
                    cleanup_old_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        save_total_limit=config.checkpoint.save_total_limit,
                    )

                    # S3 정리 (비동기)
                    if use_s3_upload:
                        s3_upload_executor.submit(
                            cleanup_s3_checkpoints,
                            experiment_id=mlflow.active_run().info.experiment_id,
                            run_id=mlflow.active_run().info.run_id,
                            save_total_limit=config.checkpoint.save_total_limit,
                        )
        else:
            logger.info(f"Validation loss did not improve ({avg_val_loss:.4f} >= {best_val_loss:.4f}), skipping checkpoint save")

        # 다음 checkpoint 경계 설정
        next_checkpoint_epoch += save_checkpoint_every

    # 8. Final checkpoint
    # is_pairwise를 while loop 밖에서 재정의 (final checkpoint용)
    is_pairwise_final = use_pairwise

    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        # 최종 validation 실행
        logger.info("--- Final Validation ---")

        if is_pairwise_final:
            # Pairwise 모드
            final_val_raw = validate_critic_pairwise(
                adapter=adapter,
                dataloader=val_loader,
                device=device,
                return_raw_counts=True,
            )

            final_raw_counts = final_val_raw["_raw_counts"]
            reduced_final_counts = all_reduce_scalars({
                "loss_sum": final_raw_counts["loss_sum"],
                "n_batches": final_raw_counts["n_batches"],
                "correct_pairs": final_raw_counts["correct_pairs"],
                "total_pairs": final_raw_counts["total_pairs"],
                "mean_pos_sum": final_raw_counts["mean_pos_sum"],
                "mean_neg_sum": final_raw_counts["mean_neg_sum"],
            }, op="sum")

            final_avg_loss = reduced_final_counts["loss_sum"] / max(1, reduced_final_counts["n_batches"])
            final_pairwise_acc = reduced_final_counts["correct_pairs"] / max(1, reduced_final_counts["total_pairs"])
            final_mean_pos = reduced_final_counts["mean_pos_sum"] / max(1, reduced_final_counts["n_batches"])
            final_mean_neg = reduced_final_counts["mean_neg_sum"] / max(1, reduced_final_counts["n_batches"])
            final_margin = final_mean_pos - final_mean_neg

            final_val_metrics = {
                "val_loss": final_avg_loss,
                "val_pairwise_accuracy": final_pairwise_acc,
                "val_mean_pos": final_mean_pos,
                "val_mean_neg": final_mean_neg,
                "val_margin": final_margin,
            }

        else:
            # Pointwise 모드
            final_val_raw = validate_critic(
                adapter=adapter,
                dataloader=val_loader,
                device=device,
                gamma=config.training.gamma,
                lam=getattr(config.training, "lam", 0.0),
                value_head_type=config.training.value_head_type,
                return_raw_counts=True,
            )

            final_raw_counts = final_val_raw["_raw_counts"]
            reduced_final_counts = all_reduce_scalars({
                "loss_sum": final_raw_counts["loss_sum"],
                "n_batches": final_raw_counts["n_batches"],
                "tp": final_raw_counts["tp"],
                "fp": final_raw_counts["fp"],
                "fn": final_raw_counts["fn"],
                "correct_sum": final_raw_counts["correct_sum"],
                "correct_count": final_raw_counts["correct_count"],
                "incorrect_sum": final_raw_counts["incorrect_sum"],
                "incorrect_count": final_raw_counts["incorrect_count"],
            }, op="sum")

            final_avg_loss = reduced_final_counts["loss_sum"] / max(1, reduced_final_counts["n_batches"])
            final_cls_metrics = compute_classification_metrics_from_counts(
                tp=reduced_final_counts["tp"],
                fp=reduced_final_counts["fp"],
                fn=reduced_final_counts["fn"],
                correct_sum=reduced_final_counts["correct_sum"],
                correct_count=reduced_final_counts["correct_count"],
                incorrect_sum=reduced_final_counts["incorrect_sum"],
                incorrect_count=reduced_final_counts["incorrect_count"],
            )

            final_val_metrics = {
                "val_loss": final_avg_loss,
                "val_pred_gap": final_cls_metrics["pred_gap"],
                "val_mean_correct": final_cls_metrics["mean_correct"],
                "val_mean_incorrect": final_cls_metrics["mean_incorrect"],
                "val_precision": final_cls_metrics["precision"],
                "val_recall": final_cls_metrics["recall"],
                "val_f1": final_cls_metrics["f1"],
            }

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=current_epoch,
            train_metrics={"train_loss": train_loss_avg},
            val_metrics=final_val_metrics,
            checkpoint_path=final_path,
            config={"model": {"path": config.models.policy.path}},
            s3_upload=use_s3_upload,
            mlflow_run_id=mlflow_run_id,
        )

        # 모든 GPU가 final checkpoint 저장 완료까지 대기
        barrier()

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 9. 모든 S3 업로드 완료 대기 및 MLflow 종료
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Critic pre-training 완료! Latest checkpoint: {latest_checkpoint_path}")

    # final_val_metrics가 정의되지 않은 경우 마지막 aggregated_val_metrics 사용
    final_metrics = final_val_metrics if config.checkpoint.save_final else aggregated_val_metrics
    return final_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critic Pre-training (Stage 1)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/critic/critic.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Config override (e.g., --override experiment.name=test)",
    )
    args = parser.parse_args()

    # Config 로드
    config = OmegaConf.load(args.config)

    # Override 적용
    if args.overrides:
        from weighted_mtp.utils.config_utils import apply_overrides
        config = apply_overrides(config, args.overrides)

    run_critic_training(config)
