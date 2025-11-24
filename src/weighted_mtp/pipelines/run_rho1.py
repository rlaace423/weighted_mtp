"""Rho-1 WMTP Runner

독립 실행:
    python -m weighted_mtp.pipelines.run_rho1 --config configs/rho1/rho1.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModelForCausalLM

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data.dataloader import create_dataloader, get_difficulty_config
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    cleanup_s3_checkpoints,
    compute_gradient_clip_stats,
    compute_weight_statistics,
    create_scheduler,
    get_model_size,
    get_system_info,
    s3_upload_executor,
    save_checkpoint,
    shutdown_s3_executor,
    upload_to_s3_async,
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
from weighted_mtp.value_weighting.rho1_weighting import (
    compute_mtp_selective_weights,
    compute_rho1_stats,
)


def load_adapter(config: dict, device: torch.device) -> MetaLlamaMTPAdapter:
    """Adapter 로드

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        MetaLlamaMTPAdapter 인스턴스
    """
    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=device,
        dtype=config.models.policy.dtype,
        initialize_value_head=False,  # Rho-1은 Value Head 불필요
    )
    return adapter


def load_reference_model(config: dict, device: torch.device) -> nn.Module:
    """Reference model 로드 (HuggingFace LlamaForCausalLM)

    Rho-1에서 Reference 모델은 NTP loss 계산용으로만 사용되므로
    HuggingFace 모델을 직접 로드합니다.

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        Reference model (eval mode, HuggingFace LlamaForCausalLM)
    """
    # dtype 변환
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(config.models.reference.dtype, torch.bfloat16)

    # HuggingFace 모델 로드
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.models.reference.path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    # Eval mode (gradient 불필요)
    ref_model.eval()

    # Gradient 계산 비활성화
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model




def validate_rho1(
    adapter: MetaLlamaMTPAdapter,
    ref_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_percent: float,
) -> dict[str, float]:
    """Validation 수행 (Rho-1)

    Args:
        adapter: Adapter (FSDP-wrapped 가능)
        ref_model: Reference model (HuggingFace LlamaForCausalLM, eval mode)
        dataloader: Validation DataLoader
        device: 디바이스
        k_percent: Top-k selection ratio (0~1)

    Returns:
        Validation metrics (FSDP 환경에서는 all-reduce 적용됨)
    """
    adapter.eval()
    ref_model.eval()

    total_weighted_ce_loss = 0.0
    total_excess_loss = 0.0
    n_batches = 0

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 2. Reference forward (HuggingFace 모델)
            ref_outputs = ref_model(input_ids)
            ref_logits = ref_outputs.logits  # [batch, seq, vocab]

            # 3. Policy forward (MTP만)
            policy_logits = adapter(input_ids)

            batch_size, seq_len, n_future, vocab_size = policy_logits.shape

            # 4. MTP selective weights (per-head binary selection)
            weights, selection_stats = compute_mtp_selective_weights(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                labels=labels,
                attention_mask=attention_mask,
                k_percent=k_percent,
            )

            # 5. Weighted CE loss (per-head)
            batch_weighted_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                if valid_len <= 0:
                    continue

                policy_logits_k = policy_logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, :valid_len, k - 1]  # Per-head weights
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    policy_logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                )

                # labels=-100인 토큰은 loss=0, mask에서도 제외
                valid_label_mask_k = (labels_k != -100).float()
                combined_mask_k = mask_k.to(model_dtype) * valid_label_mask_k

                # 모델 dtype 일치
                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * combined_mask_k.reshape(-1)

                mask_sum_k = combined_mask_k.sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # 6. Metrics 수집
            total_weighted_ce_loss += weighted_ce_loss.item()
            total_excess_loss += selection_stats.get('head_1_excess_mean', 0.0)
            n_batches += 1

    # 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches
    avg_excess_loss = total_excess_loss / n_batches

    # Validation metrics aggregation (DDP) - 1회 통신
    reduced_val = all_reduce_scalars({
        "weighted_ce_loss": avg_weighted_ce_loss,
        "excess_loss": avg_excess_loss,
    })
    avg_weighted_ce_loss = reduced_val["weighted_ce_loss"]
    avg_excess_loss = reduced_val["excess_loss"]

    metrics = {
        "val_weighted_ce_loss": avg_weighted_ce_loss,
        "val_excess_loss": avg_excess_loss,
        "val_loss": avg_weighted_ce_loss,  # Best tracking용
    }

    return metrics


def run_rho1_training(config: DictConfig) -> tuple[dict[str, float], str]:
    """Rho-1 WMTP 실행

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
    logger = setup_logging("RHO1", level=config.logging.level, rank=rank)

    logger.info("=== Rho-1 WMTP (Reference-based Weighting) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 5. MLflow 초기화 (Rank 0만)
    use_mlflow = bool(config.mlflow.experiment)
    use_s3_upload = config.checkpoint.get("s3_upload", True) and use_mlflow
    if is_main_process() and use_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: "true" for tag in config.experiment.tags},
        )
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # 6. Resource 로딩
    logger.info(f"Loading reference model: {config.models.reference.name}")
    adapter = load_adapter(config, device)
    ref_model = load_reference_model(config, device)
    tokenizer = load_tokenizer_from_config(config)
    logger.info("✓ Reference model loaded successfully")

    # 7. DDP wrapping (adapter만 - reference는 frozen inference용)
    adapter = wrap_model_fsdp(
        adapter,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )

    # Model size + System info 로깅 (Rank 0만)
    if is_main_process() and use_mlflow:
        model_size_local = get_model_size(unwrap_model(adapter))

        # FSDP FULL_SHARD 시 world_size를 곱해 실제 전체 파라미터 수 계산
        sharding_strategy = config.distributed.fsdp.sharding_strategy
        if sharding_strategy == "FULL_SHARD" and world_size > 1:
            model_size = {
                "total_params": model_size_local["total_params"] * world_size,
                "trainable_params": model_size_local["trainable_params"] * world_size,
            }
        else:
            model_size = model_size_local

        mlflow.log_params(
            {
                "model_total_params": model_size["total_params"],
                "model_trainable_params": model_size["trainable_params"],
            }
        )
        system_info = get_system_info()
        mlflow.log_params(
            {
                "system_cpu_count": system_info["cpu_count"],
                "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
            }
        )

    # GPU monitor 초기화
    gpu_monitor = GPUMonitor(device)
    throughput_tracker = ThroughputTracker()

    # 8. Dataset & DataLoader 생성
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    # Difficulty 설정 추출 (정적 weights 또는 curriculum)
    difficulty_weights, difficulty_bins = get_difficulty_config(config)
    if difficulty_weights:
        logger.info(f"Difficulty-based sampling: bins={difficulty_bins}, weights={difficulty_weights}")

    train_loader = create_dataloader(
        dataset_path=config.dataset.train,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        n_samples=config.data_sampling.n_samples,
        auto_data_balancing=config.data_sampling.auto_data_balancing,
        correct_ratio=config.data_sampling.correct_ratio,
        difficulty_weights=difficulty_weights,
        difficulty_bins=difficulty_bins,
        seed=config.data_sampling.seed,
        shuffle=True,
    )

    val_loader = create_dataloader(
        dataset_path=config.dataset.validation,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        n_samples=config.data_sampling.val_n_samples,
        auto_data_balancing=config.data_sampling.auto_data_balancing,
        correct_ratio=config.data_sampling.correct_ratio,
        difficulty_weights=None,
        difficulty_bins=None,
        seed=config.data_sampling.seed,
        shuffle=False,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # 6. Optimizer (MTP heads만 - Value head 없음) - Meta MTP 논문 설정
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # 7. Training setup
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
    logger.info(f"Top-k selection ratio: {config.training.k_percent}")

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

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

    # 8. Training loop
    optimizer.zero_grad()

    while batch_count < batches_to_run:
        # Checkpoint 경계까지 훈련
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        # DataLoader에서 필요한 만큼만 사용
        epoch_train_loader = iter(train_loader)
        period_metrics_sum = {"weighted_ce_loss": 0.0, "excess_loss": 0.0}
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                # DataLoader 재시작
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련 (Rho-1 로직)
            adapter.train()
            ref_model.eval()  # Reference는 항상 eval

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Reference forward (no grad, HuggingFace 모델)
            with torch.no_grad():
                ref_outputs = ref_model(input_ids)
                ref_logits = ref_outputs.logits  # [batch, seq, vocab]

            # Policy forward (MTP만)
            policy_logits = adapter(input_ids)

            batch_size, seq_len, n_future, vocab_size = policy_logits.shape

            # MTP selective weights (per-head binary selection)
            weights, selection_stats = compute_mtp_selective_weights(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                labels=labels,
                attention_mask=attention_mask,
                k_percent=config.training.k_percent,
            )

            # Weighted CE loss (per-head)
            batch_weighted_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                if valid_len <= 0:
                    continue

                policy_logits_k = policy_logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, :valid_len, k - 1]  # Per-head weights
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    policy_logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                )

                # labels=-100인 토큰은 loss=0, mask에서도 제외
                valid_label_mask_k = (labels_k != -100).float()
                combined_mask_k = mask_k.to(model_dtype) * valid_label_mask_k

                # 모델 dtype 일치
                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * combined_mask_k.reshape(-1)

                mask_sum_k = combined_mask_k.sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # Loss scaling (gradient accumulation 적용)
            scaled_loss = weighted_ce_loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Period metrics 누적 (batch 단위)
            period_metrics_sum["weighted_ce_loss"] += weighted_ce_loss.item()
            period_metrics_sum["excess_loss"] += selection_stats.get('head_1_excess_mean', 0.0)

            # Optimizer step (accumulation 완료 시에만)
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping (누적된 gradient에 적용)
                if config.training.max_grad_norm > 0:
                    grad_clip_stats = compute_gradient_clip_stats(
                        adapter,
                        config.training.max_grad_norm,
                    )
                else:
                    from weighted_mtp.utils.metrics_utils import compute_gradient_norm

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

            # Step-level 로깅 (optimizer step 시에만)
            if global_step % config.training.log_interval == 0 and accumulation_counter == 0:
                # GPU metrics
                gpu_metrics = gpu_monitor.get_metrics()

                # Weight distribution statistics (padding 제외)
                weight_dist_stats = compute_weight_statistics(weights, attention_mask)

                # Metric aggregation (분산 환경)
                # grad_clip_stats는 clip_grad_norm_이 이미 전역 값을 반환하므로 all_reduce 불필요
                avg_grad_norm_post = grad_clip_stats["grad_norm_post_clip"]
                avg_grad_norm_pre = grad_clip_stats["grad_norm_pre_clip"]
                avg_grad_clip_ratio = grad_clip_stats["grad_clip_ratio"]

                # 모든 로컬 메트릭 배치 all_reduce (1회 통신)
                reduced = all_reduce_scalars({
                    "weighted_ce": weighted_ce_loss.item(),
                    "selection_ratio": selection_stats['selection_ratio'],
                    "head_0_ratio": selection_stats['head_0_ratio'],
                    "head_1_ratio": selection_stats.get('head_1_ratio', 0.0),
                    "head_2_ratio": selection_stats.get('head_2_ratio', 0.0),
                    "head_3_ratio": selection_stats.get('head_3_ratio', 0.0),
                    "weight_mean": weight_dist_stats["weight_mean"],
                    "weight_std": weight_dist_stats["weight_std"],
                    "weight_min": weight_dist_stats["weight_min"],
                    "weight_max": weight_dist_stats["weight_max"],
                    "weight_entropy": weight_dist_stats["weight_entropy"],
                })
                avg_weighted_ce = reduced["weighted_ce"]
                avg_selection_ratio = reduced["selection_ratio"]

                if is_main_process():
                    if use_mlflow:
                        mlflow.log_metrics(
                            {
                                "train/weighted_ce_loss": avg_weighted_ce,
                                "train/selection_ratio": avg_selection_ratio,
                                "train/head_0_ratio": reduced["head_0_ratio"],
                                "train/head_1_ratio": reduced["head_1_ratio"],
                                "train/head_2_ratio": reduced["head_2_ratio"],
                                "train/head_3_ratio": reduced["head_3_ratio"],
                                "train/grad_norm": avg_grad_norm_post,
                                "train/grad_norm_pre_clip": avg_grad_norm_pre,
                                "train/grad_clip_ratio": avg_grad_clip_ratio,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "weight/mean": reduced["weight_mean"],
                                "weight/std": reduced["weight_std"],
                                "weight/min": reduced["weight_min"],
                                "weight/max": reduced["weight_max"],
                                "weight/entropy": reduced["weight_entropy"],
                                "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                            },
                            step=global_step,
                        )
                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"Loss: {avg_weighted_ce:.4f}, "
                        f"Grad Norm: {avg_grad_norm_post:.4f} (Clip: {avg_grad_clip_ratio:.2f}), "
                        f"Selection: {avg_selection_ratio:.1%}"
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
        train_weighted_ce_avg = period_metrics_sum["weighted_ce_loss"] / period_batches
        train_excess_avg = period_metrics_sum["excess_loss"] / period_batches

        logger.info(
            f"Epoch {current_epoch:.2f} 도달 - "
            f"Train Weighted CE: {train_weighted_ce_avg:.4f}"
        )

        # Validation 실행 (epoch 경계에서)
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_rho1(
            adapter=adapter,
            ref_model=ref_model,
            dataloader=val_loader,
            device=device,
            k_percent=config.training.k_percent,
        )

        # Epoch-level 로깅 (Rank 0만)
        if is_main_process():
            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train/epoch_weighted_ce_loss": train_weighted_ce_avg,
                        "train/epoch_excess_loss": train_excess_avg,
                        "val/weighted_ce_loss": val_metrics["val_weighted_ce_loss"],
                        "val/excess_loss": val_metrics["val_excess_loss"],
                    },
                    step=global_step,
                )

            logger.info(
                f"Validation - Weighted CE: {val_metrics['val_weighted_ce_loss']:.4f}, "
                f"Excess Loss: {val_metrics['val_excess_loss']:.4f}"
            )

        # Checkpoint 저장 (validation loss 개선 시만)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={
                    "train_weighted_ce_loss": train_weighted_ce_avg,
                    "train_excess_loss": train_excess_avg,
                },
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_path,
                config={"model": {"path": config.models.policy.path}},
            )

            # 모든 GPU가 checkpoint 저장 완료까지 대기
            barrier()

            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

                # S3 업로드 (비동기)
                if use_s3_upload:
                    s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, use_s3_upload)

                # 오래된 checkpoint 정리
                if config.checkpoint.get("save_total_limit"):
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
            logger.info(f"Validation loss did not improve ({val_metrics['val_loss']:.4f} >= {best_val_loss:.4f}), skipping checkpoint save")

        # 다음 checkpoint 경계 설정
        next_checkpoint_epoch += save_checkpoint_every

    # 9. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        # 최종 validation 실행
        logger.info("--- Final Validation ---")
        final_val_metrics = validate_rho1(
            adapter=adapter,
            ref_model=ref_model,
            dataloader=val_loader,
            device=device,
            k_percent=config.training.k_percent,
        )

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=current_epoch,
            train_metrics={
                "train_weighted_ce_loss": train_weighted_ce_avg,
                "train_excess_loss": train_excess_avg,
            },
            val_metrics=final_val_metrics,
            checkpoint_path=final_path,
            config={"model": {"path": config.models.policy.path}},
        )

        # 모든 GPU가 final checkpoint 저장 완료까지 대기
        barrier()

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 10. 모든 S3 업로드 완료 대기 및 MLflow 종료
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"🎉 Rho-1 WMTP 완료! Latest checkpoint: {latest_checkpoint_path}")

    return final_val_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rho-1 WMTP (Reference-based Weighting)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/rho1/rho1.yaml)",
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

    run_rho1_training(config)
