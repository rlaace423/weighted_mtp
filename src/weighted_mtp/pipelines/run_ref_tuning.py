"""Reference Model Fine-tuning Runner (HuggingFace NTP)

Rho-1에서 사용할 Reference Model을 codecontests 도메인에 적응시키는 파이프라인
HuggingFace LlamaForCausalLM 기반 표준 NTP (Next Token Prediction) 학습

독립 실행:
    python -m weighted_mtp.pipelines.run_ref_tuning --config configs/ref-tuning/ref_tuning.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    compute_gradient_clip_stats,
    create_scheduler,
    get_model_size,
    get_system_info,
    save_hf_checkpoint,
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


def load_hf_model(config: DictConfig, device: torch.device) -> torch.nn.Module:
    """HuggingFace 모델 로드 (학습 가능 상태)

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        HuggingFace LlamaForCausalLM (train mode)
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.models.policy.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.models.policy.path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.train()
    return model


def load_hf_tokenizer(config: DictConfig) -> AutoTokenizer:
    """HuggingFace 토크나이저 로드

    Args:
        config: 모델 설정

    Returns:
        AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.policy.path,
        use_fast=True,
    )

    # LLaMA 토크나이저는 pad_token이 없음
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def compute_ntp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """표준 NTP Cross Entropy Loss 계산

    Args:
        logits: [batch, seq, vocab] HuggingFace 출력
        labels: [batch, seq] 정답 토큰 (output 영역만 유효, -100은 무시)
        attention_mask: [batch, seq]

    Returns:
        평균 CE loss (output 토큰만)
    """
    vocab_size = logits.size(-1)

    # Shift: 현재 토큰으로 다음 토큰 예측
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # CE loss 계산 (ignore_index=-100)
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )

    # Output 토큰만 평균 (labels != -100이고 attention_mask == 1)
    valid_mask = (shift_labels != -100).float() * shift_mask.float()
    masked_loss = ce_loss * valid_mask.view(-1)

    valid_count = valid_mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    else:
        return masked_loss.sum()


def validate_ref_tuning(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validation 수행 (NTP)

    Args:
        model: HuggingFace 모델
        dataloader: Validation DataLoader
        device: 디바이스

    Returns:
        Validation metrics
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward (use_cache=False: 학습 시 KV 캐시 불필요, activation checkpointing 호환)
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits

            # NTP loss 계산
            loss = compute_ntp_loss(logits, labels, attention_mask)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches

    metrics = {
        "val_loss": avg_loss,
    }

    return metrics


def run_ref_tuning_training(config: DictConfig) -> tuple[dict[str, float], str]:
    """Reference Model Fine-tuning 실행

    Args:
        config: 완전한 config 객체 (OmegaConf DictConfig)

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 0. 환경변수 로드
    ensure_env_loaded()

    # 1. Distributed 초기화
    if "RANK" in os.environ:
        rank, world_size = init_distributed()
    else:
        rank, world_size = 0, 1

    # 2. 로깅 설정
    logger = setup_logging("REF-TUNING", level=config.logging.level, rank=rank)

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    logger.info("=== Reference Model Fine-tuning (HuggingFace NTP) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 3. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 4. MLflow 초기화 (Rank 0만)
    use_mlflow = bool(config.mlflow.experiment)
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

    # 5. 모델 및 토크나이저 로드
    logger.info(f"Loading model: {config.models.policy.path}")
    model = load_hf_model(config, device)
    tokenizer = load_hf_tokenizer(config)
    logger.info("Model and tokenizer loaded successfully")

    # 6. FSDP wrapping
    model = wrap_model_fsdp(
        model,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )

    # Model size 로깅
    model_size_local = get_model_size(unwrap_model(model))
    sharding_strategy = config.distributed.fsdp.sharding_strategy
    if sharding_strategy == "FULL_SHARD" and world_size > 1:
        model_size = {
            "total_params": model_size_local["total_params"] * world_size,
            "trainable_params": model_size_local["trainable_params"] * world_size,
        }
    else:
        model_size = model_size_local

    if is_main_process():
        if use_mlflow:
            mlflow.log_params({
                "model_total_params": model_size["total_params"],
                "model_trainable_params": model_size["trainable_params"],
            })
        logger.info(
            f"Model size: {model_size['trainable_params']:,} trainable / "
            f"{model_size['total_params']:,} total params"
        )
        system_info = get_system_info()
        if use_mlflow:
            mlflow.log_params({
                "system_cpu_count": system_info["cpu_count"],
                "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
            })

    # GPU monitor 초기화
    gpu_monitor = GPUMonitor(device)
    throughput_tracker = ThroughputTracker()

    # 7. Dataset & DataLoader 생성
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
    sampling_method = sampling_config.get("sampling_method")
    logger.info(f"Sampling method: {sampling_method}")

    train_loader = create_dataloader(
        dataset_path=config.dataset.train,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        sampling_config=sampling_config,
        seed=config.data_sampling.seed,
        shuffle=True,
    )

    # Validation용 sampling_config
    val_sampling_config = sampling_config.copy()
    if sampling_method == "difficulty":
        val_sampling_config["difficulty"] = val_sampling_config.get("difficulty", {}).copy()
        val_sampling_config["difficulty"]["n_samples"] = config.data_sampling.val_n_samples

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

    if is_main_process() and use_mlflow:
        mlflow.log_params({
            "dataset_train_samples": len(train_loader.dataset),
            "dataset_val_samples": len(val_loader.dataset),
        })

    # 8. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # 9. Training setup
    best_val_loss = float("inf")
    global_step = 0

    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = config.training.n_epochs
    save_checkpoint_every = config.checkpoint.save_checkpoint_every

    total_batches = len(train_loader)
    batches_to_run = int(total_batches * n_epochs)

    accumulation_counter = 0
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_optimization_steps}")

    # Learning rate scheduler
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

    # 10. Training loop
    optimizer.zero_grad()

    if "RANK" in os.environ:
        barrier()

    while batch_count < batches_to_run:
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        throughput_tracker.start_epoch()

        epoch_train_loader = iter(train_loader)
        period_loss_sum = 0.0
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            model.train()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward (use_cache=False: 학습 시 KV 캐시 불필요, activation checkpointing 호환)
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits

            # NTP loss 계산
            ce_loss = compute_ntp_loss(logits, labels, attention_mask)

            # Loss scaling
            scaled_loss = ce_loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Throughput tracking
            batch_size_actual = input_ids.size(0)
            n_tokens = attention_mask.sum().item()
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            period_loss_sum += ce_loss.item()

            # Optimizer step
            if accumulation_counter >= gradient_accumulation_steps:
                if config.training.max_grad_norm > 0:
                    grad_clip_stats = compute_gradient_clip_stats(
                        model,
                        config.training.max_grad_norm,
                    )
                else:
                    from weighted_mtp.utils.metrics_utils import compute_gradient_norm
                    grad_norm_dict = compute_gradient_norm(model)
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

            # Step-level 로깅
            if global_step % config.training.log_interval == 0 and accumulation_counter == 0:
                gpu_metrics = gpu_monitor.get_metrics()

                avg_grad_norm_pre = grad_clip_stats["grad_norm_pre_clip"]
                avg_grad_norm_post = grad_clip_stats["grad_norm_post_clip"]
                avg_grad_clip_ratio = grad_clip_stats["grad_clip_ratio"]

                reduced = all_reduce_scalars({"loss": ce_loss.item()})
                avg_loss = reduced["loss"]

                if is_main_process():
                    if use_mlflow:
                        mlflow.log_metrics({
                            "train/loss": avg_loss,
                            "train/grad_norm": avg_grad_norm_post,
                            "train/grad_norm_pre_clip": avg_grad_norm_pre,
                            "train/grad_clip_ratio": avg_grad_clip_ratio,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                            "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                        }, step=global_step)
                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"Loss: {avg_loss:.4f}, "
                        f"Grad Norm: {avg_grad_norm_post:.4f} (Clip: {avg_grad_clip_ratio:.2f})"
                    )

        # Incomplete accumulation 처리
        if accumulation_counter > 0:
            logger.info(f"Processing incomplete accumulation ({accumulation_counter} batches)")
            if config.training.max_grad_norm > 0:
                grad_clip_stats = compute_gradient_clip_stats(model, config.training.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0

        # Epoch 경계 도달
        current_epoch = batch_count / total_batches
        if period_batches > 0:
            train_loss_avg = period_loss_sum / period_batches
            reduced_period = all_reduce_scalars({"train_loss": train_loss_avg})
            train_loss_avg = reduced_period["train_loss"]

        logger.info(f"Epoch {current_epoch:.2f} - Train Loss: {train_loss_avg:.4f}")

        # Validation 실행
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_ref_tuning(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        reduced_val = all_reduce_scalars({"val_loss": val_metrics["val_loss"]})
        avg_val_loss = reduced_val["val_loss"]

        gpu_metrics_epoch = gpu_monitor.get_metrics()
        throughput_metrics = throughput_tracker.get_epoch_metrics()

        if is_main_process():
            if use_mlflow:
                mlflow.log_metrics({
                    "train/epoch_loss": train_loss_avg,
                    "val/loss": avg_val_loss,
                    "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                    "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                    "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                    "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                }, step=global_step)

        logger.info(f"Validation - Loss: {avg_val_loss:.4f}")

        # Checkpoint 저장 (HuggingFace 형식)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}"

            save_hf_checkpoint(
                model=model,
                tokenizer=tokenizer,
                save_dir=checkpoint_path,
                epoch=current_epoch,
                val_metrics={"val_loss": avg_val_loss},
            )

            barrier()

            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")
        else:
            logger.info(f"Validation loss did not improve ({avg_val_loss:.4f} >= {best_val_loss:.4f})")

        next_checkpoint_epoch += save_checkpoint_every

    # 11. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final"

        logger.info("--- Final Validation ---")
        final_val_metrics = validate_ref_tuning(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        reduced_final = all_reduce_scalars({"val_loss": final_val_metrics["val_loss"]})
        final_val_metrics["val_loss"] = reduced_final["val_loss"]

        save_hf_checkpoint(
            model=model,
            tokenizer=tokenizer,
            save_dir=final_path,
            epoch=current_epoch,
            val_metrics=final_val_metrics,
        )

        barrier()

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 12. MLflow 종료
    if is_main_process() and use_mlflow:
        mlflow.end_run()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Reference Model Fine-tuning 완료! Latest checkpoint: {latest_checkpoint_path}")

    final_metrics = final_val_metrics if config.checkpoint.save_final else val_metrics
    return final_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference Model Fine-tuning (HuggingFace NTP)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/ref-tuning/ref_tuning.yaml)",
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

    run_ref_tuning_training(config)
