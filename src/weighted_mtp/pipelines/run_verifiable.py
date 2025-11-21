"""Verifiable WMTP Runner (Stage 2)

독립 실행:
    python -m weighted_mtp.pipelines.run_verifiable --config configs/verifiable/verifiable.yaml
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
from transformers import AutoTokenizer

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
    compute_value_function_stats,
    compute_weight_statistics,
    get_model_size,
    get_system_info,
    load_critic_checkpoint,
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
    all_reduce_scalar,
    barrier,
)
from weighted_mtp.value_weighting.td_weighting import (
    build_weights,
    compute_td_errors,
    compute_td_stats,
    compute_weight_stats,
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
    )
    return adapter




def get_curriculum_weights(
    current_epoch: float,
    curriculum_schedule: list[dict],
) -> dict[str, float]:
    """Curriculum schedule에서 현재 epoch에 맞는 difficulty_weights 추출

    Args:
        current_epoch: 현재 epoch (0.0 ~ n_epochs)
        curriculum_schedule: Config의 curriculum_schedule

    Returns:
        difficulty_weights (예: {"low": 0.7, "medium": 0.3, "high": 0.0})
    """
    for schedule in curriculum_schedule:
        epoch_range = schedule["epoch_range"]
        if epoch_range[0] <= current_epoch < epoch_range[1]:
            return schedule["difficulty_weights"]

    # 마지막 schedule 반환 (현재 epoch이 범위 밖인 경우)
    return curriculum_schedule[-1]["difficulty_weights"]


def validate_verifiable(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    value_coef: float,
    loss_type: str,
    weight_clip_min: float,
    weight_clip_max: float,
) -> dict[str, float]:
    """Validation 수행 (Stage 2)

    Args:
        adapter: Adapter
        dataloader: Validation DataLoader
        device: 디바이스
        beta: TD error weighting 계수
        value_coef: Value loss 계수
        loss_type: 손실 함수 타입
        weight_clip_min: Weight 최소값
        weight_clip_max: Weight 최대값

    Returns:
        Validation metrics
    """
    adapter.eval()

    total_weighted_ce_loss = 0.0
    total_value_loss = 0.0
    total_loss_sum = 0.0
    n_batches = 0

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

            # 3. Forward (MTP + Value)
            outputs = adapter(
                input_ids,
                attention_mask,
                return_value_logits=True,
                return_hidden_states=True
            )
            logits = outputs["logits"]  # [batch, seq, n_future, vocab]
            value_logits = outputs["value_logits"]  # [batch, seq, 1]

            batch_size, seq_len, n_future, vocab_size = logits.shape

            # 4. TD error 계산
            td_errors = compute_td_errors(
                value_logits=value_logits,
                rewards=rewards,
                attention_mask=attention_mask,
                gamma=1.0,
            )

            # 5. Weight 산출
            weights = build_weights(
                td_errors=td_errors,
                beta=beta,
                min_weight=weight_clip_min,
                max_weight=weight_clip_max,
            )

            # 6. Weighted CE loss
            batch_weighted_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                logits_k = logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, k - 1 : k - 1 + valid_len]
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                )

                # 모델 dtype 일치
                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.to(model_dtype).reshape(-1)

                mask_sum_k = mask_k.to(model_dtype).sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # 7. Value loss (모델 dtype 일치)
            value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
            
            # Mask padded tokens AND instruction tokens (labels != -100)
            valid_label_mask = (labels != -100).unsqueeze(-1).to(model_dtype)
            loss_mask = valid_label_mask

            if loss_type == "mse":
                loss_per_token = F.mse_loss(value_logits, value_targets, reduction="none")
            elif loss_type == "huber":
                loss_per_token = F.smooth_l1_loss(value_logits, value_targets, reduction="none")
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            masked_value_loss = loss_per_token * loss_mask
            value_loss = masked_value_loss.sum() / (loss_mask.sum() + 1e-8)

            # 8. Total loss
            total_loss = weighted_ce_loss + value_coef * value_loss

            # 9. Metrics 수집
            total_weighted_ce_loss += weighted_ce_loss.item()
            total_value_loss += value_loss.item()
            total_loss_sum += total_loss.item()
            n_batches += 1

    # 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches
    avg_value_loss = total_value_loss / n_batches
    avg_total_loss = total_loss_sum / n_batches

    metrics = {
        "val_weighted_ce_loss": avg_weighted_ce_loss,
        "val_value_loss": avg_value_loss,
        "val_loss": avg_total_loss,
    }

    return metrics


def run_verifiable_training(
    config: DictConfig
) -> tuple[dict[str, float], str]:
    """Verifiable WMTP 실행 (Stage 2)

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
    logger = setup_logging("VERIFIABLE", level=config.logging.level, rank=rank)

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    logger.info("=== Verifiable WMTP (Stage 2) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 4. Critic checkpoint 검증
    if not config.experiment.get("critic_checkpoint"):
        raise ValueError(
            "critic_checkpoint가 필요합니다!\n"
            "Config에 experiment.critic_checkpoint를 명시하세요."
        )

    logger.info(f"Critic checkpoint: {config.experiment.critic_checkpoint}")

    # 5. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 6. MLflow 초기화 (Rank 0만, experiment 이름이 있는 경우만)
    use_mlflow = bool(config.mlflow.experiment)
    if is_main_process() and use_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: "true" for tag in config.experiment.tags},
        )
        # Config 로깅
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        mlflow.log_param("critic_checkpoint", config.experiment.critic_checkpoint)
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # 7. Resource 로딩
    adapter = load_adapter(config, device)
    tokenizer = load_tokenizer_from_config(config)

    # Model size + System info 로깅
    model_size = get_model_size(adapter)
    logger.info(
        f"Model size: {model_size['trainable_params']:,} trainable / "
        f"{model_size['total_params']:,} total params"
    )
    if is_main_process() and use_mlflow:
        mlflow.log_params(
            {
                "model_total_params": model_size["total_params"],
                "model_trainable_params": model_size["trainable_params"],
            }
        )

    system_info = get_system_info()
    if is_main_process() and use_mlflow:
        mlflow.log_params(
            {
                "system_cpu_count": system_info["cpu_count"],
                "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
            }
        )

    # GPU monitor 초기화
    gpu_monitor = GPUMonitor(device)
    throughput_tracker = ThroughputTracker()

    # 8. Critic checkpoint 로드 (Value head 초기화)
    logger.info(f"Loading critic checkpoint: {config.experiment.critic_checkpoint}")
    load_critic_checkpoint(
        checkpoint_path=config.experiment.critic_checkpoint,
        adapter=adapter,
        device=device,
    )
    logger.info("✓ Critic checkpoint loaded successfully")

    # 9. DDP wrapping (critic checkpoint 로드 후)
    adapter = wrap_model_fsdp(
        adapter,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
    )

    # 10. Optimizer (전체 파라미터) - Meta MTP 논문 설정
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # 11. Training setup
    best_val_loss = float("inf")
    global_step = 0

    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = config.training.n_epochs
    save_checkpoint_every = config.checkpoint.save_checkpoint_every

    # Curriculum learning 설정
    use_curriculum = config.data_sampling.get("curriculum_learning", False)
    curriculum_schedule = config.data_sampling.get("curriculum_schedule", [])
    difficulty_bins = config.data_sampling.get("difficulty_bins", None)

    if use_curriculum:
        logger.info("Curriculum Learning: Enabled")
        logger.info(f"Difficulty bins: {difficulty_bins}")
    else:
        logger.info("Curriculum Learning: Disabled")

    # 초기 DataLoader 생성 (epoch 0.0 기준)
    if use_curriculum and curriculum_schedule:
        initial_weights = get_curriculum_weights(0.0, curriculum_schedule)
    else:
        initial_weights = None

    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    train_loader = create_dataloader(
        dataset_path=config.dataset.train,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        n_samples=config.data_sampling.n_samples,
        balance_correct=config.data_sampling.balance_correct,
        correct_ratio=config.data_sampling.correct_ratio,
        difficulty_weights=initial_weights,
        difficulty_bins=difficulty_bins,
        seed=config.data_sampling.seed,
        shuffle=True,
    )

    # Validation 샘플 수: train의 5% 또는 최소 100개
    val_n_samples = max(100, config.data_sampling.n_samples // 20)

    val_loader = create_dataloader(
        dataset_path=config.dataset.validation,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        n_samples=val_n_samples,
        balance_correct=config.data_sampling.balance_correct,
        correct_ratio=config.data_sampling.correct_ratio,
        difficulty_weights=initial_weights if use_curriculum else None,
        difficulty_bins=difficulty_bins,
        seed=config.data_sampling.seed,
        shuffle=False,
    )

    # Fractional epoch 처리
    total_batches = len(train_loader)
    batches_to_run = int(total_batches * n_epochs)

    # Gradient accumulation 초기화
    accumulation_counter = 0
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    # Optimization steps 계산
    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_optimization_steps}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")

    current_epoch = 0.0
    batch_count = 0
    next_checkpoint_epoch = save_checkpoint_every

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

    # Curriculum learning: 이전 weights 추적 (중복 DataLoader 생성 방지)
    prev_curriculum_weights = initial_weights

    # FSDP 워밍업: 첫 forward에서 all-gather 동기화 문제 방지
    logger.info("FSDP warmup forward pass 시작...")
    adapter.eval()
    with torch.no_grad():
        # 더미 입력으로 FSDP parameter sharding 동기화
        dummy_batch_size = 1
        dummy_seq_len = 16
        dummy_input = torch.ones(dummy_batch_size, dummy_seq_len, dtype=torch.long, device=device)
        dummy_mask = torch.ones(dummy_batch_size, dummy_seq_len, dtype=torch.long, device=device)
        try:
            _ = adapter(dummy_input, dummy_mask, return_value_logits=True)
            logger.info("FSDP warmup forward pass 완료")
        except Exception as e:
            logger.warning(f"FSDP warmup 중 예외 발생 (무시): {e}")

    # 모든 rank가 워밍업 완료까지 동기화
    barrier()
    logger.info("모든 rank FSDP 워밍업 동기화 완료")

    # 9. Training loop
    optimizer.zero_grad()

    while batch_count < batches_to_run:
        # Checkpoint 경계까지 훈련
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        # Curriculum learning: 현재 epoch에 맞는 difficulty_weights 계산
        if use_curriculum and curriculum_schedule:
            current_weights = get_curriculum_weights(current_epoch, curriculum_schedule)
            logger.info(f"Curriculum weights: {current_weights}")

            # weights가 변경된 경우에만 DataLoader 재생성
            if current_weights != prev_curriculum_weights:
                logger.info("Curriculum weights 변경 감지, DataLoader 재생성")
                train_loader = create_dataloader(
                    dataset_path=config.dataset.train,
                    tokenizer=tokenizer,
                    batch_size=config.training.batch_size,
                    max_length=config.dataset.max_length,
                    n_samples=config.data_sampling.n_samples,
                    balance_correct=config.data_sampling.balance_correct,
                    correct_ratio=config.data_sampling.correct_ratio,
                    difficulty_weights=current_weights,
                    difficulty_bins=difficulty_bins,
                    seed=config.data_sampling.seed + int(current_epoch * 1000),
                    shuffle=True,
                )
                prev_curriculum_weights = current_weights

                # DataLoader 재생성 후 동기화
                barrier()
            else:
                logger.info("Curriculum weights 동일, 기존 DataLoader 재사용")

        # DataLoader에서 필요한 만큼만 사용
        epoch_train_loader = iter(train_loader)
        period_metrics_sum = {
            "weighted_ce_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
        }
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                # DataLoader 재시작
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련 (Stage 2 로직)
            adapter.train()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            is_correct = batch["is_correct"].to(device)

            # 모델 dtype 일치
            rewards = is_correct.to(model_dtype)

            # Forward (MTP + Value)
            outputs = adapter(
                input_ids,
                attention_mask,
                return_value_logits=True,
                return_hidden_states=True
            )
            logits = outputs["logits"]
            value_logits = outputs["value_logits"]

            batch_size, seq_len, n_future, vocab_size = logits.shape

            # TD error 계산
            td_errors = compute_td_errors(
                value_logits=value_logits,
                rewards=rewards,
                attention_mask=attention_mask,
                gamma=1.0,
            )

            # Weight 산출
            weights = build_weights(
                td_errors=td_errors,
                beta=config.training.beta,
                min_weight=config.training.weight_clip_min,
                max_weight=config.training.weight_clip_max,
            )

            # Weighted CE loss
            batch_weighted_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                logits_k = logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, k - 1 : k - 1 + valid_len]
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                )

                # 모델 dtype 일치
                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.to(model_dtype).reshape(-1)

                mask_sum_k = mask_k.to(model_dtype).sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # Value loss (Continual Learning, 모델 dtype 일치)
            value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
            
            # Mask padded tokens AND instruction tokens (labels != -100)
            valid_label_mask = (labels != -100).unsqueeze(-1).to(model_dtype)
            loss_mask = valid_label_mask

            if config.training.loss_type == "mse":
                loss_per_token = F.mse_loss(value_logits, value_targets, reduction="none")
            elif config.training.loss_type == "huber":
                loss_per_token = F.smooth_l1_loss(
                    value_logits, value_targets, reduction="none"
                )
            else:
                raise ValueError(f"Unknown loss_type: {config.training.loss_type}")

            masked_value_loss = loss_per_token * loss_mask
            value_loss = masked_value_loss.sum() / (loss_mask.sum() + 1e-8)

            # Total loss
            total_loss = weighted_ce_loss + config.training.value_coef * value_loss

            # Loss scaling (gradient accumulation 적용)
            scaled_loss = total_loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Throughput tracking (batch 단위)
            batch_size_actual = input_ids.size(0)
            n_tokens = attention_mask.sum().item()
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            # Period metrics 누적 (batch 단위)
            period_metrics_sum["weighted_ce_loss"] += weighted_ce_loss.item()
            period_metrics_sum["value_loss"] += value_loss.item()
            period_metrics_sum["total_loss"] += total_loss.item()

            # Optimizer step (accumulation 완료 시에만)
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping (누적된 gradient에 적용)
                if config.training.max_grad_norm > 0:
                    params_with_grad = [p for group in optimizer.param_groups for p in group["params"]]
                    grad_clip_stats = compute_gradient_clip_stats(
                        params_with_grad,
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
                optimizer.zero_grad()

                global_step += 1
                accumulation_counter = 0

            # Step-level 로깅 (optimizer step 시에만)
            if global_step % config.training.log_interval == 0 and accumulation_counter == 0:
                # TD error/weight stats
                td_stats = compute_td_stats(td_errors)
                weight_stats = compute_weight_stats(weights)
                gpu_metrics = gpu_monitor.get_metrics()

                # Value function statistics
                value_func_stats = compute_value_function_stats(
                    values=value_logits.squeeze(-1),
                    returns=value_targets.squeeze(-1),
                )

                # Weight distribution statistics
                weight_dist_stats = compute_weight_statistics(weights)

                # Metric aggregation (DDP)
                avg_weighted_ce = all_reduce_scalar(weighted_ce_loss.item())
                avg_value_loss = all_reduce_scalar(value_loss.item())
                avg_total_loss = all_reduce_scalar(total_loss.item())
                avg_grad_norm_post = all_reduce_scalar(grad_clip_stats["grad_norm_post_clip"])
                avg_grad_norm_pre = all_reduce_scalar(grad_clip_stats["grad_norm_pre_clip"])
                avg_grad_clip_ratio = all_reduce_scalar(grad_clip_stats["grad_clip_ratio"])
                avg_td_mean = all_reduce_scalar(td_stats["td_mean"])
                avg_weight_mean = all_reduce_scalar(weight_stats["weight_mean"])
                avg_value_mse = all_reduce_scalar(value_func_stats["value_mse"])
                avg_value_mean = all_reduce_scalar(value_func_stats["value_mean"])
                avg_value_std = all_reduce_scalar(value_func_stats["value_std"])

                if is_main_process():
                    if use_mlflow:
                        mlflow.log_metrics(
                            {
                                "train/weighted_ce_loss": avg_weighted_ce,
                                "train/value_loss": avg_value_loss,
                                "train/total_loss": avg_total_loss,
                                "train/grad_norm": avg_grad_norm_post,
                                "train/grad_norm_pre_clip": avg_grad_norm_pre,
                                "train/grad_clip_ratio": avg_grad_clip_ratio,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "train/td_mean": avg_td_mean,
                                "train/weight_mean": avg_weight_mean,
                                "value/mse": avg_value_mse,
                                "value/mean_prediction": avg_value_mean,
                                "value/std_prediction": avg_value_std,
                                "weight/mean": weight_dist_stats["weight_mean"],
                                "weight/std": weight_dist_stats["weight_std"],
                                "weight/min": weight_dist_stats["weight_min"],
                                "weight/max": weight_dist_stats["weight_max"],
                                "weight/entropy": weight_dist_stats["weight_entropy"],
                                "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                            },
                            step=global_step,
                        )
                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"Loss: {avg_total_loss:.4f}, "
                        f"Grad Norm: {avg_grad_norm_post:.4f} (Clip: {avg_grad_clip_ratio:.2f})"
                    )

        # Period loop 종료

        # Incomplete accumulation 처리 (validation 전)
        if accumulation_counter > 0:
            logger.info(f"Processing incomplete accumulation ({accumulation_counter} batches before validation)")

            # Gradient clipping
            if config.training.max_grad_norm > 0:
                params_with_grad = [p for group in optimizer.param_groups for p in group["params"]]
                grad_clip_stats = compute_gradient_clip_stats(
                    params_with_grad,
                    config.training.max_grad_norm,
                )

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0

        # Epoch 경계 도달
        current_epoch = batch_count / total_batches

        # Period-level metrics 계산 & aggregation
        train_weighted_ce_avg = period_metrics_sum["weighted_ce_loss"] / period_batches
        train_value_avg = period_metrics_sum["value_loss"] / period_batches
        train_total_avg = period_metrics_sum["total_loss"] / period_batches

        train_weighted_ce_avg = all_reduce_scalar(train_weighted_ce_avg)
        train_value_avg = all_reduce_scalar(train_value_avg)
        train_total_avg = all_reduce_scalar(train_total_avg)

        logger.info(
            f"Epoch {current_epoch:.2f} 도달 - "
            f"Train Total Loss: {train_total_avg:.4f}"
        )

        # Validation 실행 (epoch 경계에서)
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_verifiable(
            adapter=adapter,
            dataloader=val_loader,
            device=device,
            beta=config.training.beta,
            value_coef=config.training.value_coef,
            loss_type=config.training.loss_type,
            weight_clip_min=config.training.weight_clip_min,
            weight_clip_max=config.training.weight_clip_max,
        )

        # Validation metrics aggregation
        avg_val_total = all_reduce_scalar(val_metrics["val_loss"])
        avg_val_weighted_ce = all_reduce_scalar(val_metrics["val_weighted_ce_loss"])
        avg_val_value = all_reduce_scalar(val_metrics["val_value_loss"])

        # Epoch-level 로깅
        if is_main_process() and use_mlflow:
            mlflow.log_metrics(
                {
                    "train/epoch_total_loss": train_total_avg,
                    "train/epoch_weighted_ce_loss": train_weighted_ce_avg,
                    "train/epoch_value_loss": train_value_avg,
                    "val/total_loss": avg_val_total,
                    "val/weighted_ce_loss": avg_val_weighted_ce,
                    "val/value_loss": avg_val_value,
                },
                step=int(current_epoch * 100),
            )

        logger.info(
            f"Validation - Total Loss: {avg_val_total:.4f}, "
            f"Weighted CE: {avg_val_weighted_ce:.4f}, "
            f"Value: {avg_val_value:.4f}"
        )

        # Checkpoint 저장 (validation loss 개선 시만)
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={
                    "train_total_loss": train_total_avg,
                    "train_weighted_ce_loss": train_weighted_ce_avg,
                    "train_value_loss": train_value_avg,
                },
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_path,
            )

            # 모든 GPU가 checkpoint 저장 완료까지 대기
            barrier()

            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

            # S3 업로드 (비동기)
            if is_main_process() and use_mlflow:
                s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, use_mlflow)

            # 오래된 checkpoint 정리 (최대 3개 유지)
            if config.checkpoint.get("save_total_limit"):
                cleanup_old_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    save_total_limit=config.checkpoint.save_total_limit,
                )

                # S3 정리 (비동기)
                if is_main_process() and use_mlflow:
                    s3_upload_executor.submit(
                        cleanup_s3_checkpoints,
                        experiment_id=mlflow.active_run().info.experiment_id,
                        run_id=mlflow.active_run().info.run_id,
                        save_total_limit=config.checkpoint.save_total_limit,
                    )
        else:
            logger.info(f"Validation loss did not improve ({avg_val_total:.4f} >= {best_val_loss:.4f}), skipping checkpoint save")

        # 다음 checkpoint 경계 설정
        next_checkpoint_epoch += save_checkpoint_every

    # 10. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        # 최종 validation 실행
        logger.info("--- Final Validation ---")
        final_val_metrics = validate_verifiable(
            adapter=adapter,
            dataloader=val_loader,
            device=device,
            beta=config.training.beta,
            value_coef=config.training.value_coef,
            loss_type=config.training.loss_type,
            weight_clip_min=config.training.weight_clip_min,
            weight_clip_max=config.training.weight_clip_max,
        )

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=current_epoch,
            train_metrics={
                "train_total_loss": train_total_avg,
                "train_weighted_ce_loss": train_weighted_ce_avg,
                "train_value_loss": train_value_avg,
            },
            val_metrics=final_val_metrics,
            checkpoint_path=final_path,
        )

        # 모든 GPU가 final checkpoint 저장 완료까지 대기
        barrier()

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 11. 모든 S3 업로드 완료 대기 및 MLflow 종료
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Verifiable WMTP 완료! Latest checkpoint: {latest_checkpoint_path}")

    return final_val_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifiable WMTP (Stage 2)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/verifiable/verifiable.yaml)",
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

    run_verifiable_training(config)
