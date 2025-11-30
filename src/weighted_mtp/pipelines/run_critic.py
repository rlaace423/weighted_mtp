"""Critic Pre-training Runner (독립 Value Model)

독립 ValueModel (HuggingFace 기반)을 학습하는 파이프라인.
Policy Model과 완전 분리된 별도 모델 사용.

독립 실행:
    python -m weighted_mtp.pipelines.run_critic --config configs/production/critic_mlp.yaml
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
from weighted_mtp.models.value_model import ValueModel
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    cleanup_s3_checkpoints,
    compute_gradient_clip_stats,
    compute_gradient_norm,
    compute_mc_value_loss,
    compute_pairwise_accuracy,
    create_scheduler,
    get_model_size,
    get_system_info,
    pairwise_ranking_loss,
    s3_upload_executor,
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


def load_value_model(
    config: DictConfig,
    device: torch.device,
) -> ValueModel:
    """독립 Value Model 로드

    Args:
        config: 설정 (models.value_model 포함)
        device: 디바이스

    Returns:
        ValueModel 인스턴스
    """
    # Value head 설정 (training.value_head에서 읽기)
    value_head_config = config.training.get("value_head", {})
    value_head_type = value_head_config.get("type", "mlp")
    dropout = value_head_config.get("dropout", 0.0)

    # LoRA 설정 (learning_rate, weight_decay 제외하고 전달)
    use_lora = getattr(config.training, "use_lora", False)
    lora_config = None
    if use_lora and hasattr(config.training, "lora"):
        lora_full = OmegaConf.to_container(config.training.lora, resolve=True)
        # LoRA 모델 설정만 추출 (학습 하이퍼파라미터 제외)
        lora_config = {
            k: v for k, v in lora_full.items()
            if k not in ("learning_rate", "weight_decay")
        }

    # Value Model 로드
    value_model = ValueModel.from_pretrained(
        model_path=config.models.value_model.path,
        value_head_type=value_head_type,
        dropout=dropout,
        device=str(device),
        dtype=config.models.value_model.dtype,
        use_lora=use_lora,
        lora_config=lora_config,
    )

    return value_model


def save_value_model_checkpoint(
    value_model: ValueModel,
    optimizer: torch.optim.Optimizer,
    epoch: float,
    train_metrics: dict,
    val_metrics: dict,
    checkpoint_path: Path,
    config: DictConfig = None,
) -> None:
    """Value Model checkpoint 저장

    LoRA 모드: checkpoint_type="hf_lora"로 저장 (LoRA weights + value head만)
    Full 모드: checkpoint_type="full"로 저장 (전체 backbone + value head)

    Args:
        value_model: ValueModel 인스턴스
        optimizer: Optimizer
        epoch: 현재 epoch
        train_metrics: Train metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
        config: Config (OmegaConf DictConfig)
    """
    # FSDP unwrap
    unwrapped = unwrap_model(value_model)

    # LoRA 모드 확인
    use_lora = getattr(unwrapped, "lora_enabled", False)

    if use_lora:
        # LoRA checkpoint (경량)
        from weighted_mtp.models.lora import get_hf_lora_state_dict

        checkpoint = {
            "checkpoint_type": "hf_lora",
            "lora_state_dict": get_hf_lora_state_dict(unwrapped.backbone),
            "value_head_state_dict": unwrapped.value_head.state_dict(),
            "lora_config": unwrapped.lora_config,
            "base_model_path": config.models.value_model.path if config else None,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(config, resolve=True) if config else None,
        }
    else:
        # Full checkpoint
        checkpoint = {
            "checkpoint_type": "full",
            "epoch": epoch,
            "backbone_state_dict": unwrapped.backbone.state_dict(),
            "value_head_state_dict": unwrapped.value_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(config, resolve=True) if config else None,
        }

    torch.save(checkpoint, checkpoint_path)


def validate_critic(
    value_model: ValueModel,
    dataloader: DataLoader,
    device: torch.device,
    use_pairwise: bool = True,
    use_mc_mse: bool = False,
    pairwise_coef: float = 1.0,
    mc_mse_coef: float = 1.0,
    return_raw_counts: bool = False,
) -> dict[str, float]:
    """Pairwise Validation 수행

    Args:
        value_model: ValueModel 인스턴스
        dataloader: Validation DataLoader (pairwise format)
        device: 디바이스
        use_pairwise: Pairwise ranking loss 사용 여부
        use_mc_mse: MC tokenwise MSE loss 사용 여부
        pairwise_coef: Pairwise loss 계수
        mc_mse_coef: MC MSE loss 계수
        return_raw_counts: True이면 raw counts 반환 (분산학습용 aggregation)

    Returns:
        Validation metrics (pairwise_accuracy, mean_pos, mean_neg, margin, loss)
    """
    value_model.eval()

    total_loss = 0.0
    n_batches = 0
    total_correct_pairs = 0.0
    total_pairs = 0
    total_mean_pos = 0.0
    total_mean_neg = 0.0

    # 모델 dtype 감지
    model_dtype = next(value_model.parameters()).dtype

    with torch.no_grad():
        for batch in dataloader:
            # Pairwise batch 구조
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)
            neg_labels = batch["neg_labels"].to(device)

            # Batched Forward: pos+neg concat하여 1회 forward
            batch_size = pos_input_ids.size(0)
            combined_input_ids = torch.cat([pos_input_ids, neg_input_ids], dim=0)
            combined_attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)

            combined_value_logits = value_model(combined_input_ids, combined_attention_mask)

            pos_value_logits = combined_value_logits[:batch_size]
            neg_value_logits = combined_value_logits[batch_size:]

            # 학습 대상 토큰 마스크 (labels != -100)
            pos_loss_mask = (pos_labels != -100)
            neg_loss_mask = (neg_labels != -100)

            # Value Loss 계산 (조건부)
            value_loss = torch.tensor(0.0, device=device, dtype=model_dtype)

            if use_pairwise:
                pairwise_loss_val = pairwise_ranking_loss(
                    v_pos=pos_value_logits,
                    v_neg=neg_value_logits,
                    mask_pos=pos_loss_mask,
                    mask_neg=neg_loss_mask,
                )
                value_loss = value_loss + pairwise_coef * pairwise_loss_val

            if use_mc_mse:
                pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)
                pos_mc_loss = compute_mc_value_loss(
                    pos_value_logits, pos_rewards, pos_attention_mask, pos_loss_mask
                )
                neg_rewards = torch.zeros(neg_input_ids.size(0), device=device, dtype=model_dtype)
                neg_mc_loss = compute_mc_value_loss(
                    neg_value_logits, neg_rewards, neg_attention_mask, neg_loss_mask
                )
                mc_mse_loss_val = (pos_mc_loss + neg_mc_loss) / 2
                value_loss = value_loss + mc_mse_coef * mc_mse_loss_val

            # Pairwise accuracy 계산 (항상 계산 - 메트릭용)
            pairwise_metrics = compute_pairwise_accuracy(
                v_pos=pos_value_logits,
                v_neg=neg_value_logits,
                mask_pos=pos_loss_mask,
                mask_neg=neg_loss_mask,
            )

            total_loss += value_loss.item()
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
    """Critic pre-training 실행 (독립 Value Model)

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

    logger.info("=== Critic Pre-training (독립 Value Model) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 5. MLflow 초기화 (Rank 0만, experiment 이름이 있는 경우만)
    use_mlflow = bool(config.mlflow.experiment)
    use_s3_upload = config.checkpoint.get("s3_upload", True) and use_mlflow
    mlflow_run_id = None
    if is_main_process() and use_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: "true" for tag in config.experiment.tags},
        )
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        mlflow_run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {mlflow_run_id}")

    # 6. Value Model 로드
    value_model = load_value_model(config, device)
    logger.info(f"Value Model loaded: {config.models.value_model.path}")

    # LoRA 설정 확인
    use_lora = getattr(config.training, "use_lora", False)

    # Backbone freeze 설정: use_lora=True면 LoRA가 freeze 제어 (원본 frozen, LoRA만 학습)
    if use_lora:
        logger.info("LoRA mode: backbone frozen, training LoRA + value head")
    else:
        backbone_frozen = getattr(config.training, "backbone_frozen", True)
        if backbone_frozen:
            value_model.freeze_backbone()
            logger.info("Backbone frozen: training value head only")
        else:
            logger.info("Backbone unfrozen: training entire model")

    # Model size 로깅 (FSDP wrapping 전에 계산)
    model_size = get_model_size(value_model)

    # Trainable params breakdown (FSDP 전에 계산, 로깅용)
    trainable_breakdown = {
        "value_head": sum(p.numel() for p in value_model.value_head.parameters() if p.requires_grad),
        "backbone": sum(p.numel() for p in value_model.backbone.parameters() if p.requires_grad),
    }

    # 파라미터 개수 저장 (FSDP wrapping 전에 저장, 로깅용)
    # FSDP 후에는 sharded 상태라 numel()이 다르게 나올 수 있음
    if use_lora:
        lora_param_count = sum(p.numel() for p in value_model.backbone.parameters() if p.requires_grad)
        value_head_param_count = sum(p.numel() for p in value_model.value_head.parameters())
    else:
        backbone_frozen = getattr(config.training, "backbone_frozen", True)
        if backbone_frozen:
            lora_param_count = 0
            value_head_param_count = sum(p.numel() for p in value_model.value_head.parameters() if p.requires_grad)
        else:
            lora_param_count = 0
            value_head_param_count = sum(p.numel() for p in value_model.parameters())

    # FSDP wrapping
    value_model = wrap_model_fsdp(
        value_model,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )

    # FSDP wrapping 후에 named_parameters()로 param 리스트 구성 (optimizer용)
    # use_orig_params=True이므로 원본 파라미터 이름 구조 유지됨
    if use_lora:
        # backbone의 requires_grad=True 파라미터 (LoRA adapters)
        lora_params = [p for n, p in value_model.named_parameters()
                       if 'value_head' not in n and p.requires_grad]
        # value_head 파라미터 (모두 학습)
        value_head_params = [p for n, p in value_model.named_parameters()
                             if 'value_head' in n]
    else:
        backbone_frozen = getattr(config.training, "backbone_frozen", True)
        if backbone_frozen:
            lora_params = []
            value_head_params = [p for n, p in value_model.named_parameters()
                                 if 'value_head' in n and p.requires_grad]
        else:
            lora_params = []
            value_head_params = list(value_model.parameters())

    tokenizer = load_tokenizer_from_config(config)

    if is_main_process():
        if use_mlflow:
            # backbone_frozen은 use_lora=False일 때만 적용
            backbone_frozen_effective = not use_lora and getattr(config.training, "backbone_frozen", True)
            mlflow.log_params(
                {
                    "model_total_params": model_size["total_params"],
                    "model_trainable_params": model_size["trainable_params"],
                    "model_non_trainable_params": model_size["non_trainable_params"],
                    "model_trainable_value_head": trainable_breakdown["value_head"],
                    "model_trainable_backbone": trainable_breakdown["backbone"],
                    "use_lora": use_lora,
                    "backbone_frozen": backbone_frozen_effective,
                }
            )
        logger.info(
            f"Model size: {model_size['trainable_params']:,} trainable / "
            f"{model_size['total_params']:,} total params"
        )
        logger.info(
            f"Trainable breakdown - value_head: {trainable_breakdown['value_head']:,}, "
            f"backbone: {trainable_breakdown['backbone']:,}"
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

    # GPU monitor 초기화
    gpu_monitor = GPUMonitor(device)

    # 5. Dataset & DataLoader 생성
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    # sampling_config를 dict로 변환 (pairwise 모드 강제)
    sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
    sampling_config["use_pairwise"] = True  # value head 학습은 항상 pairwise
    logger.info("Pairwise 모드 (value head 학습)")

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

    # 6. Optimizer (LoRA/Value Head 하이퍼파라미터 분리)
    # lora_params, value_head_params는 FSDP wrapping 전에 저장됨 (line 361-373)
    lora_config_full = config.training.get("lora", {})
    value_head_config = config.training.get("value_head", {})

    lora_lr = lora_config_full.get("learning_rate", 1e-4)
    lora_weight_decay = lora_config_full.get("weight_decay", 0.01)
    value_head_lr = value_head_config.get("learning_rate", 5e-4)
    value_head_weight_decay = value_head_config.get("weight_decay", 0.01)

    # Parameter groups 구성 (FSDP wrapping 후 named_parameters()로 가져온 파라미터 사용)
    if use_lora:
        param_groups = [
            {
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": lora_weight_decay,
                "name": "lora",
            },
            {
                "params": value_head_params,
                "lr": value_head_lr,
                "weight_decay": value_head_weight_decay,
                "name": "value_head",
            },
        ]

        # 로깅은 FSDP 전에 저장한 param count 사용 (FSDP 후에는 sharded 상태)
        logger.info(f"LoRA params: {lora_param_count:,}, LR={lora_lr}, WD={lora_weight_decay}")
        logger.info(f"Value Head params: {value_head_param_count:,}, LR={value_head_lr}, WD={value_head_weight_decay}")
    else:
        param_groups = [
            {
                "params": value_head_params,
                "lr": value_head_lr,
                "weight_decay": value_head_weight_decay,
                "name": "value_head",
            }
        ]

        logger.info(f"Value Head params: {value_head_param_count:,}, LR={value_head_lr}, WD={value_head_weight_decay}")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lora_lr,  # defaults["lr"]용 (scheduler 로깅에 사용)
        betas=(0.9, 0.95),
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
    train_loss_avg = 0.0

    # Throughput tracker 초기화
    throughput_tracker = ThroughputTracker()

    # 모델 dtype 감지
    model_dtype = next(value_model.parameters()).dtype

    # value_loss 설정
    value_loss_config = config.training.get("value_loss", {})
    use_pairwise = value_loss_config.get("use_pairwise", True)
    use_mc_mse = value_loss_config.get("use_mc_mse", False)
    pairwise_coef = value_loss_config.get("pairwise_coef", 1.0)
    mc_mse_coef = value_loss_config.get("mc_mse_coef", 1.0)
    logger.info(f"Value loss: use_pairwise={use_pairwise}, use_mc_mse={use_mc_mse}")

    # Gradient clipping
    max_grad_norm = config.training.get("max_grad_norm", 1.0)
    logger.info(f"Gradient clipping: max_grad_norm={max_grad_norm}")

    # Optimizer 초기화
    optimizer.zero_grad()

    # 7. Training loop
    while batch_count < batches_to_run:
        # Train until checkpoint boundary
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

        # Pairwise 메트릭용 변수
        train_correct_pairs = 0.0
        train_total_pairs = 0
        train_mean_pos_sum = 0.0
        train_mean_neg_sum = 0.0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련
            value_model.train()

            # Pairwise batch 구조
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)
            neg_labels = batch["neg_labels"].to(device)

            # Batched Forward: pos+neg concat
            batch_size = pos_input_ids.size(0)
            combined_input_ids = torch.cat([pos_input_ids, neg_input_ids], dim=0)
            combined_attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)

            combined_value_logits = value_model(combined_input_ids, combined_attention_mask)

            pos_value_logits = combined_value_logits[:batch_size]
            neg_value_logits = combined_value_logits[batch_size:]

            # 학습 대상 토큰 마스크
            pos_loss_mask = (pos_labels != -100)
            neg_loss_mask = (neg_labels != -100)

            # Value Loss 계산
            value_loss = torch.tensor(0.0, device=device, dtype=model_dtype)

            if use_pairwise:
                pairwise_loss_val = pairwise_ranking_loss(
                    v_pos=pos_value_logits,
                    v_neg=neg_value_logits,
                    mask_pos=pos_loss_mask,
                    mask_neg=neg_loss_mask,
                )
                value_loss = value_loss + pairwise_coef * pairwise_loss_val

            if use_mc_mse:
                pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)
                pos_mc_loss = compute_mc_value_loss(
                    pos_value_logits, pos_rewards, pos_attention_mask, pos_loss_mask
                )
                neg_rewards = torch.zeros(neg_input_ids.size(0), device=device, dtype=model_dtype)
                neg_mc_loss = compute_mc_value_loss(
                    neg_value_logits, neg_rewards, neg_attention_mask, neg_loss_mask
                )
                mc_mse_loss_val = (pos_mc_loss + neg_mc_loss) / 2
                value_loss = value_loss + mc_mse_coef * mc_mse_loss_val

            # Pairwise 메트릭 누적
            pairwise_metrics = compute_pairwise_accuracy(
                v_pos=pos_value_logits,
                v_neg=neg_value_logits,
                mask_pos=pos_loss_mask,
                mask_neg=neg_loss_mask,
            )
            train_correct_pairs += pairwise_metrics["correct_pairs"]
            train_total_pairs += pairwise_metrics["total_pairs"]
            train_mean_pos_sum += pairwise_metrics["mean_pos"]
            train_mean_neg_sum += pairwise_metrics["mean_neg"]

            # Throughput용 변수
            batch_size_actual = pos_input_ids.size(0) * 2
            n_tokens = pos_attention_mask.sum().item() + neg_attention_mask.sum().item()

            # Loss scaling
            scaled_loss = value_loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            throughput_tracker.update(batch_size_actual, int(n_tokens))
            period_metrics_sum["train_loss"] += value_loss.item()

            # Optimizer step
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping
                if max_grad_norm > 0:
                    grad_clip_stats = compute_gradient_clip_stats(value_model, max_grad_norm)
                else:
                    grad_norm_dict = compute_gradient_norm(value_model)
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

                # Step-level logging
                if global_step % config.training.log_interval == 0:
                    gpu_metrics = gpu_monitor.get_metrics()

                    # 분리된 LR 추출
                    lora_current_lr = optimizer.param_groups[0]["lr"]
                    value_head_current_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lora_current_lr

                    batch_pairwise_acc = pairwise_metrics["pairwise_accuracy"]
                    batch_mean_pos = pairwise_metrics["mean_pos"]
                    batch_mean_neg = pairwise_metrics["mean_neg"]

                    reduced = all_reduce_scalars({
                        "loss": value_loss.item(),
                        "pairwise_accuracy": batch_pairwise_acc,
                        "mean_pos": batch_mean_pos,
                        "mean_neg": batch_mean_neg,
                    })

                    if is_main_process():
                        if use_mlflow:
                            mlflow.log_metrics(
                                {
                                    "train/loss": reduced["loss"],
                                    "train/grad_norm": grad_clip_stats["grad_norm_post_clip"],
                                    "train/lora_lr": lora_current_lr,
                                    "train/value_head_lr": value_head_current_lr,
                                    "train/pairwise_accuracy": reduced["pairwise_accuracy"],
                                    "value/mean_pos": reduced["mean_pos"],
                                    "value/mean_neg": reduced["mean_neg"],
                                    "value/margin": reduced["mean_pos"] - reduced["mean_neg"],
                                    "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                },
                                step=global_step,
                            )
                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"Loss: {reduced['loss']:.4f}, "
                        f"Pairwise Acc: {reduced['pairwise_accuracy']:.3f}, "
                        f"Margin: {reduced['mean_pos'] - reduced['mean_neg']:.4f}, "
                        f"LoRA LR: {lora_current_lr:.2e}, VH LR: {value_head_current_lr:.2e}"
                    )

        # Period 종료

        # Incomplete accumulation 처리
        if accumulation_counter > 0:
            logger.info(f"Processing incomplete accumulation ({accumulation_counter} batches)")
            if max_grad_norm > 0:
                compute_gradient_clip_stats(value_model, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0

        current_epoch = batch_count / total_batches

        # Period metrics
        train_loss_avg = period_metrics_sum["train_loss"] / period_batches
        throughput_metrics = throughput_tracker.get_epoch_metrics()
        gpu_metrics_epoch = gpu_monitor.get_metrics()

        # Pairwise 메트릭 aggregation
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
            f"Epoch {current_epoch:.2f} - "
            f"Train Loss: {train_loss_avg:.4f}, "
            f"Pairwise Acc: {train_pairwise_acc:.3f}, "
            f"Margin: {train_margin:.4f}"
        )

        # Validation
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_critic(
            value_model=value_model,
            dataloader=val_loader,
            device=device,
            use_pairwise=use_pairwise,
            use_mc_mse=use_mc_mse,
            pairwise_coef=pairwise_coef,
            mc_mse_coef=mc_mse_coef,
            return_raw_counts=True,
        )

        # Validation aggregation
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

        # Epoch-level 로깅
        if is_main_process():
            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train/epoch_loss": train_loss_avg,
                        "train/epoch_pairwise_accuracy": train_pairwise_acc,
                        "train/epoch_margin": train_margin,
                        "val/loss": avg_val_loss,
                        "val/pairwise_accuracy": avg_val_pairwise_acc,
                        "val/margin": avg_val_margin,
                        "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                        "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                        "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                    },
                    step=global_step,
                )

        logger.info(
            f"Validation - Loss: {avg_val_loss:.4f}, "
            f"Pairwise Acc: {avg_val_pairwise_acc:.3f}, "
            f"Margin: {avg_val_margin:.4f}"
        )

        # Aggregated validation metrics
        aggregated_val_metrics = {
            "val_loss": avg_val_loss,
            "val_pairwise_accuracy": avg_val_pairwise_acc,
            "val_mean_pos": avg_val_mean_pos,
            "val_mean_neg": avg_val_mean_neg,
            "val_margin": avg_val_margin,
        }

        # Checkpoint 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

            save_value_model_checkpoint(
                value_model=value_model,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={"train_loss": train_loss_avg},
                val_metrics=aggregated_val_metrics,
                checkpoint_path=checkpoint_path,
                config=config,
            )

            barrier()

            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

                if config.checkpoint.save_total_limit:
                    cleanup_old_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        save_total_limit=config.checkpoint.save_total_limit,
                    )
        else:
            logger.info(f"Validation loss did not improve ({avg_val_loss:.4f} >= {best_val_loss:.4f})")

        next_checkpoint_epoch += save_checkpoint_every

    # 8. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        logger.info("--- Final Validation ---")

        final_val_raw = validate_critic(
            value_model=value_model,
            dataloader=val_loader,
            device=device,
            use_pairwise=use_pairwise,
            use_mc_mse=use_mc_mse,
            pairwise_coef=pairwise_coef,
            mc_mse_coef=mc_mse_coef,
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

        save_value_model_checkpoint(
            value_model=value_model,
            optimizer=optimizer,
            epoch=current_epoch,
            train_metrics={"train_loss": train_loss_avg},
            val_metrics=final_val_metrics,
            checkpoint_path=final_path,
            config=config,
        )

        barrier()

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 9. 종료
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Critic pre-training 완료! Latest checkpoint: {latest_checkpoint_path}")

    final_metrics = final_val_metrics if config.checkpoint.save_final else aggregated_val_metrics
    return final_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critic Pre-training (독립 Value Model)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/production/critic_mlp.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Config override (e.g., --override experiment.name=test)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.overrides:
        from weighted_mtp.utils.config_utils import apply_overrides
        config = apply_overrides(config, args.overrides)

    run_critic_training(config)
