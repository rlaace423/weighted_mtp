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
    compute_gradient_clip_stats,
    compute_gradient_norm,
    compute_pairwise_accuracy,
    create_param_groups,
    create_scheduler,
    get_model_size,
    get_system_info,
    pairwise_ranking_loss,
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

            # Batched Forward: pos+neg concat하여 1회 forward
            batch_size = pos_input_ids.size(0)
            combined_input_ids = torch.cat([pos_input_ids, neg_input_ids], dim=0)
            combined_attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)

            combined_outputs = adapter(combined_input_ids, combined_attention_mask, return_value_logits=True)
            combined_value_logits = combined_outputs["value_logits"]

            pos_value_logits = combined_value_logits[:batch_size]
            neg_value_logits = combined_value_logits[batch_size:]

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

    # trunk 완전 frozen 여부 (메모리 최적화용)
    trunk_frozen = (num_unfrozen == 0)
    if trunk_frozen:
        logger.info("Trunk frozen optimization enabled (no_grad for trunk forward)")

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

        # Pairwise 메트릭용 변수
        train_correct_pairs = 0.0
        train_total_pairs = 0
        train_mean_pos_sum = 0.0
        train_mean_neg_sum = 0.0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                # DataLoader 재시작
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련
            adapter.train()

            # Pairwise batch 구조
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)
            neg_labels = batch["neg_labels"].to(device)

            # Batched Forward: pos+neg concat하여 1회 forward (2배 속도 향상)
            batch_size = pos_input_ids.size(0)
            combined_input_ids = torch.cat([pos_input_ids, neg_input_ids], dim=0)
            combined_attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)

            combined_outputs = adapter(combined_input_ids, combined_attention_mask, return_value_logits=True, trunk_frozen=trunk_frozen)
            combined_value_logits = combined_outputs["value_logits"]

            pos_value_logits = combined_value_logits[:batch_size]
            neg_value_logits = combined_value_logits[batch_size:]

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

                    # Pairwise 메트릭 로깅
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

        # Pairwise 메트릭 계산
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

        # Validation
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_critic(
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

        # Epoch-level 로깅
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
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        # 최종 validation 실행
        logger.info("--- Final Validation ---")

        final_val_raw = validate_critic(
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
