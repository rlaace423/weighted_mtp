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
    compute_gradient_norm,
    compute_value_function_stats,
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
    all_reduce_scalar,
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
    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=device,
        dtype=config.models.policy.dtype,
    )
    return adapter




def validate_critic(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
    loss_type: str,
) -> dict[str, float]:
    """Validation 수행

    Args:
        adapter: Adapter
        dataloader: Validation DataLoader
        device: 디바이스
        loss_type: 손실 함수 타입

    Returns:
        Validation metrics
    """
    adapter.eval()

    total_loss = 0.0
    total_value_var = 0.0
    total_target_var = 0.0
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

            # 3. Forward (Value head만)
            outputs = adapter(input_ids, attention_mask, return_value_logits=True)
            value_logits = outputs["value_logits"]

            batch_size, seq_len, _ = value_logits.shape

            # 4. Value target 생성
            value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)

            # Mask padded tokens AND instruction tokens (labels != -100)
            # labels != -100 means it's a target token (output)
            valid_label_mask = (labels != -100).unsqueeze(-1).to(model_dtype)
            loss_mask = valid_label_mask

            # 5. Value loss 계산
            if loss_type == "mse":
                loss_per_token = torch.nn.functional.mse_loss(
                    value_logits, value_targets, reduction="none"
                )
            elif loss_type == "huber":
                loss_per_token = torch.nn.functional.smooth_l1_loss(
                    value_logits, value_targets, reduction="none"
                )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            # Masked loss
            masked_loss = loss_per_token * loss_mask
            value_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

            # 6. Metrics 수집
            total_loss += value_loss.item()
            total_value_var += value_logits.var().item()
            total_target_var += value_targets.var().item()
            n_batches += 1

    # 평균 metrics 계산
    avg_loss = total_loss / n_batches
    avg_target_var = total_target_var / n_batches

    # Value explained variance 계산
    if avg_target_var > 1e-8:
        explained_var = 1.0 - (avg_loss / avg_target_var)
    else:
        explained_var = 0.0

    metrics = {
        "val_loss": avg_loss,
        "val_explained_variance": explained_var,
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
    if is_main_process() and use_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: "true" for tag in config.experiment.tags},
        )
        # Config 로깅
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # 6. Resource 로딩
    adapter = load_adapter(config, device)

    # Transformer trunk freeze (value head만 학습)
    logger.info("Freezing transformer trunk (training value head only)")
    for param in adapter.transformer.parameters():
        param.requires_grad = False

    adapter = wrap_model_fsdp(
        adapter,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
    )
    tokenizer = load_tokenizer_from_config(config)

    # Model size 로깅
    model_size = get_model_size(unwrap_model(adapter))
    if is_main_process():
        if use_mlflow:
            mlflow.log_params(
                {
                    "model_total_params": model_size["total_params"],
                    "model_trainable_params": model_size["trainable_params"],
                    "model_non_trainable_params": model_size["non_trainable_params"],
                }
            )
        logger.info(
            f"Model size: {model_size['trainable_params']:,} trainable / "
            f"{model_size['total_params']:,} total params"
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

        train_loader = create_dataloader(
            dataset_path=config.dataset.train,
            tokenizer=tokenizer,
            batch_size=config.training.batch_size,
            max_length=config.dataset.max_length,
            n_samples=config.data_sampling.n_samples,
            balance_correct=config.data_sampling.balance_correct,
            correct_ratio=config.data_sampling.correct_ratio,
            difficulty_weights=None,
            difficulty_bins=None,
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
            difficulty_weights=None,
            difficulty_bins=None,
            seed=config.data_sampling.seed,
            shuffle=False,
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

        # Dataset statistics 로깅
        if use_mlflow:
            mlflow.log_params(
                {
                    "dataset_train_samples": len(train_loader.dataset),
                    "dataset_val_samples": len(val_loader.dataset),
                    "dataset_train_batches": len(train_loader),
                    "dataset_val_batches": len(val_loader),
                }
            )

        # 6. Optimizer (Value head only) - Meta MTP 논문 설정
        optimizer = torch.optim.AdamW(
            adapter.value_head.parameters(),
            lr=config.training.learning_rate,
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

        current_epoch = 0.0
        batch_count = 0
        next_checkpoint_epoch = save_checkpoint_every

        # Throughput tracker 초기화
        throughput_tracker = ThroughputTracker()

        # 모델 dtype 감지
        model_dtype = next(adapter.parameters()).dtype

        # Optimizer 초기화 (gradient accumulation을 위해 while loop 시작 전)
        optimizer.zero_grad()

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

            for _ in range(batches_this_period):
                try:
                    batch = next(epoch_train_loader)
                except StopIteration:
                    # DataLoader 재시작
                    epoch_train_loader = iter(train_loader)
                    batch = next(epoch_train_loader)

                # 1 batch 훈련
                adapter.train()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                is_correct = batch["is_correct"].to(device)

                # 모델 dtype 일치
                rewards = is_correct.to(model_dtype)
                outputs = adapter(input_ids, attention_mask, return_value_logits=True)
                value_logits = outputs["value_logits"]

                batch_size, seq_len, _ = value_logits.shape
                value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
                # Mask padded tokens AND instruction tokens (labels != -100)
                # labels != -100 means it's a target token (output)
                valid_label_mask = (labels != -100).unsqueeze(-1).to(model_dtype)
                loss_mask = valid_label_mask

                if config.training.loss_type == "mse":
                    loss_per_token = torch.nn.functional.mse_loss(
                        value_logits, value_targets, reduction="none"
                    )
                elif config.training.loss_type == "huber":
                    loss_per_token = torch.nn.functional.smooth_l1_loss(
                        value_logits, value_targets, reduction="none"
                    )
                else:
                    raise ValueError(f"Unknown loss_type: {config.training.loss_type}")

                masked_loss = loss_per_token * loss_mask
                value_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

                # Loss scaling (gradient accumulation 적용)
                scaled_loss = value_loss / gradient_accumulation_steps
                scaled_loss.backward()

                accumulation_counter += 1
                batch_count += 1
                period_batches += 1

                # Throughput tracking (batch 단위)
                batch_size_actual = input_ids.size(0)
                n_tokens = attention_mask.sum().item()
                throughput_tracker.update(batch_size_actual, int(n_tokens))

                # Period metrics 누적 (batch 단위)
                period_metrics_sum["train_loss"] += value_loss.item()

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

                    # Step-level logging (optimizer step 시에만)
                    if global_step % config.training.log_interval == 0:
                        # GPU metrics
                        gpu_metrics = gpu_monitor.get_metrics()

                        # Value function statistics
                        value_stats = compute_value_function_stats(
                            values=value_logits.squeeze(-1),
                            returns=value_targets.squeeze(-1),
                        )

                        # Metric aggregation (DDP)
                        avg_loss = all_reduce_scalar(value_loss.item())
                        avg_grad_norm_post = all_reduce_scalar(grad_clip_stats["grad_norm_post_clip"])
                        avg_grad_norm_pre = all_reduce_scalar(grad_clip_stats["grad_norm_pre_clip"])
                        avg_grad_clip_ratio = all_reduce_scalar(grad_clip_stats["grad_clip_ratio"])
                        avg_value_mean = all_reduce_scalar(value_stats["value_mean"])
                        avg_value_std = all_reduce_scalar(value_stats["value_std"])

                        if is_main_process():
                            if use_mlflow:
                                mlflow.log_metrics(
                                    {
                                        "train/loss": avg_loss,
                                        "train/grad_norm": avg_grad_norm_post,
                                        "train/grad_norm_pre_clip": avg_grad_norm_pre,
                                        "train/grad_clip_ratio": avg_grad_clip_ratio,
                                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                                        "value/mean_prediction": avg_value_mean,
                                        "value/std_prediction": avg_value_std,
                                        "value/mse": value_stats["value_mse"],
                                        "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                        "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                                    },
                                    step=global_step,
                                )
                        logger.info(
                            f"Step {global_step}/{total_optimization_steps}, "
                            f"Loss: {avg_loss:.4f}, "
                            f"Grad Norm: {avg_grad_norm_post:.4f} (Clip Ratio: {avg_grad_clip_ratio:.2f})"
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

            # Period-level metrics 계산
            train_loss_avg = period_metrics_sum["train_loss"] / period_batches
            train_loss_avg = all_reduce_scalar(train_loss_avg)

            logger.info(
                f"Epoch {current_epoch:.2f} 도달 - "
                f"Train Loss: {train_loss_avg:.4f}"
            )

            # Throughput metrics 계산
            throughput_metrics = throughput_tracker.get_epoch_metrics()

            # Validation 실행 (epoch 경계에서)
            logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

            val_metrics = validate_critic(
                adapter=adapter,
                dataloader=val_loader,
                device=device,
                loss_type=config.training.loss_type,
            )

            # Validation metrics aggregation
            avg_val_loss = all_reduce_scalar(val_metrics["val_loss"])
            avg_val_explained_var = all_reduce_scalar(val_metrics["val_explained_variance"])

            # GPU metrics (epoch-level)
            gpu_metrics_epoch = gpu_monitor.get_metrics()

            # Epoch-level 로깅
            if is_main_process():
                if use_mlflow:
                    mlflow.log_metrics(
                        {
                            "train/epoch_loss": train_loss_avg,
                            "val/loss": avg_val_loss,
                            "val/explained_variance": avg_val_explained_var,
                        "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                        "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                        "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                        "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                    },
                    step=int(current_epoch * 100),  # Epoch을 정수로 변환 (0.5 -> 50)
                    )

            logger.info(
                f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                f"Explained Variance: {val_metrics['val_explained_variance']:.4f}"
            )

            # Checkpoint 저장 (validation loss 개선 시만)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

                save_checkpoint(
                    adapter=adapter,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    train_metrics={"train_loss": train_loss_avg},
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
                if config.checkpoint.save_total_limit:
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
                logger.info(f"Validation loss did not improve ({avg_val_loss:.4f} >= {best_val_loss:.4f}), skipping checkpoint save")

            # 다음 checkpoint 경계 설정
            next_checkpoint_epoch += save_checkpoint_every

        # 8. Final checkpoint
        if config.checkpoint.save_final:
            final_path = checkpoint_dir / "checkpoint_final.pt"

            # 최종 validation 실행
            logger.info("--- Final Validation ---")
            final_val_metrics = validate_critic(
                adapter=adapter,
                dataloader=val_loader,
                device=device,
                loss_type=config.training.loss_type,
            )

            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={"train_loss": train_loss_avg},
                val_metrics=final_val_metrics,
                checkpoint_path=final_path,
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

    # final_val_metrics가 정의되지 않은 경우 마지막 val_metrics 사용
    final_metrics = final_val_metrics if config.checkpoint.save_final else val_metrics
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
