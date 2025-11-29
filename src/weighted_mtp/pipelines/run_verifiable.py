"""Verifiable WMTP Runner (TD Weighting with Independent Value Model)

Policy Model (MTP)과 Value Model (독립)을 분리하여 사용.
Value Model은 Critic에서 학습된 checkpoint 로드, eval only.

독립 실행:
    python -m weighted_mtp.pipelines.run_verifiable --config configs/production/verifiable_pairwise.yaml
"""

import argparse
import os
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.value_model import ValueModel
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    compute_gradient_clip_stats,
    compute_mtp_ce_loss,
    compute_value_function_stats,
    compute_weight_statistics,
    create_scheduler,
    get_model_size,
    get_system_info,
    save_checkpoint,
    save_lora_checkpoint,
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
from weighted_mtp.value_weighting.td_weighting import (
    build_weights,
    compute_td_errors,
    compute_td_stats,
)
from weighted_mtp.value_weighting.td_stats_ema import TDStatsEMA


def load_policy_model(
    config: DictConfig,
    device: torch.device,
) -> MetaLlamaMTPAdapter:
    """Policy Model 로드 (순수 MTP, value_head 없음)
    
    Args:
        config: 설정 (models.policy 포함)
        device: 디바이스
    
    Returns:
        MetaLlamaMTPAdapter 인스턴스
    """
    # LoRA 설정
    use_lora = getattr(config.training, "use_lora", False)
    lora_config = None
    if use_lora:
        lora_section = config.training.get("lora", {})
        lora_config = {
            "rank": lora_section.get("rank", 8),
            "alpha": lora_section.get("alpha", 16.0),
            "dropout": lora_section.get("dropout", 0.0),
            "target_modules": lora_section.get(
                "target_modules", ["wq", "wk", "wv", "wo"]
            ),
        }
    
    policy_model = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=str(device),
        dtype=config.models.policy.dtype,
        use_lora=use_lora,
        lora_config=lora_config,
    )
    
    return policy_model


def load_value_model(
    config: DictConfig,
    device: torch.device,
) -> ValueModel:
    """Value Model 로드 (Critic checkpoint에서, eval only)
    
    Args:
        config: 설정 (models.value_model 포함)
        device: 디바이스
    
    Returns:
        ValueModel 인스턴스 (eval mode, frozen)
    """
    value_model = ValueModel.from_checkpoint(
        checkpoint_path=config.models.value_model.checkpoint_path,
        device=str(device),
    )
    
    # Eval only 모드 (전체 frozen, gradient 비활성화)
    value_model.eval_mode()
    
    return value_model


def validate_verifiable(
    policy_model: MetaLlamaMTPAdapter,
    value_model: ValueModel,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    weight_clip_min: float,
    weight_clip_max: float,
) -> dict[str, float]:
    """Validation 수행 (TD Weighting only)

    Args:
        policy_model: Policy Model (MTP)
        value_model: Value Model (eval only)
        dataloader: Validation DataLoader
        device: 디바이스
        beta: TD error weighting 계수
        weight_clip_min: Weight 최소값
        weight_clip_max: Weight 최대값

    Returns:
        Validation metrics (weighted_ce_loss, unweighted_ce_loss)
    """
    policy_model.eval()

    total_weighted_ce_loss = 0.0
    total_unweighted_ce_loss = 0.0
    n_batches = 0

    # 모델 dtype 감지
    model_dtype = next(policy_model.parameters()).dtype

    with torch.no_grad():
        for batch in dataloader:
            # Positive sample만 사용 (TD weighting용)
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)

            pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)

            # Value Model forward (TD error 계산용)
            value_logits = value_model(pos_input_ids, pos_attention_mask)

            # Policy Model forward (MTP logits)
            policy_outputs = policy_model(pos_input_ids, pos_attention_mask)
            pos_logits = policy_outputs  # [batch, seq, n_future_tokens, vocab]

            # 학습 대상 토큰 마스크 (labels != -100)
            pos_loss_mask = (pos_labels != -100)

            # TD error 계산
            td_errors = compute_td_errors(
                value_logits=value_logits,
                rewards=pos_rewards,
                loss_mask=pos_loss_mask,
                gamma=1.0,
            )
            weights = build_weights(
                td_errors=td_errors,
                loss_mask=pos_loss_mask,
                beta=beta,
                min_weight=weight_clip_min,
                max_weight=weight_clip_max,
            )

            # Policy Loss (TD Weighted)
            ce_losses = compute_mtp_ce_loss(
                logits=pos_logits,
                labels=pos_labels,
                attention_mask=pos_attention_mask,
                weights=weights,
            )
            weighted_ce_loss = ce_losses["weighted_ce_loss"]
            unweighted_ce_loss = ce_losses["unweighted_ce_loss"]

            # Metrics 수집
            total_weighted_ce_loss += weighted_ce_loss.item()
            total_unweighted_ce_loss += unweighted_ce_loss.item()
            n_batches += 1

    # 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches if n_batches > 0 else 0.0
    avg_unweighted_ce_loss = total_unweighted_ce_loss / n_batches if n_batches > 0 else 0.0

    metrics = {
        "val_weighted_ce_loss": avg_weighted_ce_loss,
        "val_unweighted_ce_loss": avg_unweighted_ce_loss,
        "val_loss": avg_unweighted_ce_loss,  # Best checkpoint 기준
    }

    return metrics


def run_verifiable_training(
    config: DictConfig
) -> tuple[dict[str, float], str]:
    """Verifiable WMTP 실행 (독립 Value Model 사용)

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
    logger = setup_logging("VERIFIABLE", level=config.logging.level, rank=rank)

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    logger.info("=== Verifiable WMTP (독립 Value Model) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 3. Environment setup
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 4. MLflow 초기화
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

    # 5. LoRA 설정 로깅
    use_lora = getattr(config.training, "use_lora", False)
    if use_lora:
        lora_section = config.training.get("lora", {})
        logger.info(f"LoRA enabled: rank={lora_section.get('rank')}, alpha={lora_section.get('alpha')}")

    # 6. 모델 로드
    # Policy Model (학습 대상)
    logger.info(f"Loading Policy Model: {config.models.policy.path}")
    policy_model = load_policy_model(config, device)
    
    # Value Model (Critic checkpoint에서, eval only)
    logger.info(f"Loading Value Model from checkpoint: {config.models.value_model.checkpoint_path}")
    value_model = load_value_model(config, device)
    logger.info("Value Model loaded in eval mode (frozen)")

    tokenizer = load_tokenizer_from_config(config)

    # Model size 로깅
    policy_size = get_model_size(policy_model)
    value_size = get_model_size(value_model)
    logger.info(
        f"Policy Model: {policy_size['trainable_params']:,} trainable / "
        f"{policy_size['total_params']:,} total params"
    )
    logger.info(
        f"Value Model: {value_size['trainable_params']:,} trainable / "
        f"{value_size['total_params']:,} total params (frozen)"
    )

    if is_main_process() and use_mlflow:
        mlflow.log_params({
            "policy_total_params": policy_size["total_params"],
            "policy_trainable_params": policy_size["trainable_params"],
            "value_total_params": value_size["total_params"],
        })

    system_info = get_system_info()
    if is_main_process() and use_mlflow:
        mlflow.log_params({
            "system_cpu_count": system_info["cpu_count"],
            "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
        })

    # GPU monitor 초기화
    gpu_monitor = GPUMonitor(device)
    throughput_tracker = ThroughputTracker()

    # 7. FSDP wrapping (Policy Model만)
    policy_model = wrap_model_fsdp(
        policy_model,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )
    
    # Value Model은 FSDP 불필요 (eval only, gradient 없음)
    value_model = value_model.to(device)

    # 8. Optimizer (Policy Model만)
    learning_rate = config.training.get("learning_rate", 1e-4)
    
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
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

    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    # sampling_config (pairwise 모드 유지, positive만 사용)
    sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
    sampling_config["use_pairwise"] = True
    logger.info("Pairwise 모드 (Positive sample만 TD weighting에 사용)")

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

    # Fractional epoch 처리
    total_batches = len(train_loader)
    batches_to_run = int(total_batches * n_epochs)

    # Gradient accumulation
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    if is_main_process() and use_mlflow:
        mlflow.log_params({
            "dataset_train_samples": len(train_loader.dataset),
            "dataset_val_samples": len(val_loader.dataset),
        })

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
    accumulation_counter = 0

    # 모델 dtype 감지
    model_dtype = next(policy_model.parameters()).dtype

    # TD EMA 통계 추적기
    td_ema_momentum = config.training.get("td_ema_momentum", 0.1)
    td_ema_warmup_steps = config.training.get("td_ema_warmup_steps", 10)
    td_ema = TDStatsEMA(
        device=device,
        momentum=td_ema_momentum,
        warmup_steps=td_ema_warmup_steps,
    )
    logger.info(f"TD EMA: momentum={td_ema_momentum}, warmup_steps={td_ema_warmup_steps}")

    # Gradient clipping
    max_grad_norm = config.training.get("max_grad_norm", 1.0)
    logger.info(f"Gradient clipping: max_grad_norm={max_grad_norm}")

    # FSDP 워밍업
    logger.info("FSDP warmup forward pass...")
    policy_model.eval()
    with torch.no_grad():
        dummy_input = torch.ones(1, 16, dtype=torch.long, device=device)
        try:
            _ = policy_model(dummy_input)
            logger.info("FSDP warmup 완료")
        except Exception as e:
            logger.warning(f"FSDP warmup 예외 (무시): {e}")

    barrier()
    logger.info("모든 rank FSDP 워밍업 동기화 완료")

    # 10. Training loop
    optimizer.zero_grad()

    while batch_count < batches_to_run:
        # Checkpoint 경계까지 훈련
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        throughput_tracker.start_epoch()

        epoch_train_loader = iter(train_loader)
        period_metrics_sum = {
            "weighted_ce_loss": 0.0,
            "unweighted_ce_loss": 0.0,
        }
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            policy_model.train()

            # Positive sample만 사용 (TD weighting용)
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)

            pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)

            # Value Model forward (TD error 계산용, no_grad)
            with torch.no_grad():
                value_logits = value_model(pos_input_ids, pos_attention_mask)

            # 학습 대상 토큰 마스크
            pos_loss_mask = (pos_labels != -100)

            # TD error 계산
            td_errors = compute_td_errors(
                value_logits=value_logits,
                rewards=pos_rewards,
                loss_mask=pos_loss_mask,
                gamma=1.0,
            )

            # Weight 산출 (EMA 기반)
            ema_mean, ema_std = td_ema.get_stats()
            weights = build_weights(
                td_errors=td_errors,
                loss_mask=pos_loss_mask,
                beta=config.training.beta,
                min_weight=config.training.weight_clip_min,
                max_weight=config.training.weight_clip_max,
                external_mean=ema_mean,
                external_std=ema_std,
            )

            # Policy Forward + Sequential Unembedding (메모리 효율적 weighted loss 계산 + backward)
            # backward()는 policy_model 내부에서 수행됨
            loss_result = policy_model(
                pos_input_ids,
                attention_mask=pos_attention_mask,
                compute_sequential_loss=True,
                labels=pos_labels,
                weights=weights,
                loss_scale=1.0 / gradient_accumulation_steps,
            )
            weighted_ce_loss = loss_result["loss"]  # detached scalar

            # unweighted_ce_loss는 Sequential 모드에서 계산하지 않음 (메모리 효율 우선)
            # logging용으로 weighted_ce_loss를 사용
            unweighted_ce_loss = weighted_ce_loss

            # EMA 통계 업데이트
            td_ema.update(td_errors, pos_loss_mask, distributed=True)

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Throughput tracking
            batch_size_actual = pos_input_ids.size(0)
            n_tokens = pos_attention_mask.sum().item()
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            # Period metrics 누적
            period_metrics_sum["weighted_ce_loss"] += weighted_ce_loss.item()
            period_metrics_sum["unweighted_ce_loss"] += unweighted_ce_loss.item()

            # Optimizer step
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping
                if max_grad_norm > 0:
                    grad_clip_stats = compute_gradient_clip_stats(policy_model, max_grad_norm)
                else:
                    grad_clip_stats = {"grad_norm_post_clip": 0.0, "grad_norm_pre_clip": 0.0}

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                accumulation_counter = 0

                # Step-level 로깅
                if global_step % config.training.log_interval == 0:
                    td_stats = compute_td_stats(td_errors, pos_loss_mask)
                    gpu_metrics = gpu_monitor.get_metrics()

                    # Value function stats (로깅용)
                    td_targets = pos_rewards.view(-1, 1, 1).expand_as(value_logits)
                    value_func_stats = compute_value_function_stats(
                        values=value_logits.squeeze(-1),
                        returns=td_targets.squeeze(-1),
                        loss_mask=pos_loss_mask,
                    )

                    weight_dist_stats = compute_weight_statistics(weights, pos_loss_mask)

                    reduced = all_reduce_scalars({
                        "weighted_ce": weighted_ce_loss.item(),
                        "unweighted_ce": unweighted_ce_loss.item(),
                        "td_mean": td_stats["td_mean"],
                        "value_mse": value_func_stats["value_mse"],
                        "weight_mean": weight_dist_stats["weight_mean"],
                    })

                    if is_main_process():
                        if use_mlflow:
                            mlflow.log_metrics({
                                "train/weighted_ce_loss": reduced["weighted_ce"],
                                "train/unweighted_ce_loss": reduced["unweighted_ce"],
                                "train/grad_norm": grad_clip_stats["grad_norm_post_clip"],
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "td/mean": reduced["td_mean"],
                                "td/ema_mean": td_ema.ema_mean.item(),
                                "td/ema_std": td_ema.ema_std.item(),
                                "value/mse": reduced["value_mse"],
                                "weight/mean": reduced["weight_mean"],
                                "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                            }, step=global_step)

                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"CE: {reduced['weighted_ce']:.4f}, "
                        f"TD: {reduced['td_mean']:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )

        # Period 종료

        # Incomplete accumulation 처리
        if accumulation_counter > 0:
            if max_grad_norm > 0:
                compute_gradient_clip_stats(policy_model, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0

        current_epoch = batch_count / total_batches

        # Period metrics
        train_weighted_ce_avg = period_metrics_sum["weighted_ce_loss"] / period_batches
        train_unweighted_ce_avg = period_metrics_sum["unweighted_ce_loss"] / period_batches
        throughput_metrics = throughput_tracker.get_epoch_metrics()
        gpu_metrics_epoch = gpu_monitor.get_metrics()

        logger.info(
            f"Epoch {current_epoch:.2f} - "
            f"Weighted CE: {train_weighted_ce_avg:.4f}, "
            f"Unweighted CE: {train_unweighted_ce_avg:.4f}"
        )

        # Validation
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_verifiable(
            policy_model=policy_model,
            value_model=value_model,
            dataloader=val_loader,
            device=device,
            beta=config.training.beta,
            weight_clip_min=config.training.weight_clip_min,
            weight_clip_max=config.training.weight_clip_max,
        )

        # Validation all_reduce
        reduced_val = all_reduce_scalars({
            "val_weighted_ce": val_metrics["val_weighted_ce_loss"],
            "val_unweighted_ce": val_metrics["val_unweighted_ce_loss"],
        })

        avg_val_weighted_ce = reduced_val["val_weighted_ce"]
        avg_val_unweighted_ce = reduced_val["val_unweighted_ce"]

        if is_main_process():
            if use_mlflow:
                mlflow.log_metrics({
                    "train/epoch_weighted_ce": train_weighted_ce_avg,
                    "train/epoch_unweighted_ce": train_unweighted_ce_avg,
                    "val/weighted_ce_loss": avg_val_weighted_ce,
                    "val/unweighted_ce_loss": avg_val_unweighted_ce,
                    "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                    "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                    "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                }, step=global_step)

        logger.info(
            f"Validation - Weighted CE: {avg_val_weighted_ce:.4f}, "
            f"Unweighted CE: {avg_val_unweighted_ce:.4f}"
        )

        aggregated_val_metrics = {
            "val_weighted_ce_loss": avg_val_weighted_ce,
            "val_unweighted_ce_loss": avg_val_unweighted_ce,
            "val_loss": avg_val_unweighted_ce,
        }

        # Checkpoint 저장
        if avg_val_unweighted_ce < best_val_loss:
            best_val_loss = avg_val_unweighted_ce
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

            # LoRA only 저장 옵션
            save_lora_only = config.checkpoint.get("save_lora_only", False) and use_lora
            
            if save_lora_only:
                save_lora_checkpoint(
                    adapter=policy_model,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    train_metrics={"train_weighted_ce": train_weighted_ce_avg},
                    val_metrics=aggregated_val_metrics,
                    checkpoint_path=checkpoint_path,
                    config={"model": {"path": config.models.policy.path}},
                )
            else:
                save_checkpoint(
                    adapter=policy_model,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    train_metrics={"train_weighted_ce": train_weighted_ce_avg},
                    val_metrics=aggregated_val_metrics,
                    checkpoint_path=checkpoint_path,
                    config={"model": {"path": config.models.policy.path}},
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
            logger.info(f"Validation loss did not improve ({avg_val_unweighted_ce:.4f} >= {best_val_loss:.4f})")

        next_checkpoint_epoch += save_checkpoint_every

    # 11. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        logger.info("--- Final Validation ---")

        final_val_metrics = validate_verifiable(
            policy_model=policy_model,
            value_model=value_model,
            dataloader=val_loader,
            device=device,
            beta=config.training.beta,
            weight_clip_min=config.training.weight_clip_min,
            weight_clip_max=config.training.weight_clip_max,
        )

        reduced_final = all_reduce_scalars({
            "val_weighted_ce": final_val_metrics["val_weighted_ce_loss"],
            "val_unweighted_ce": final_val_metrics["val_unweighted_ce_loss"],
        })

        final_aggregated = {
            "val_weighted_ce_loss": reduced_final["val_weighted_ce"],
            "val_unweighted_ce_loss": reduced_final["val_unweighted_ce"],
            "val_loss": reduced_final["val_unweighted_ce"],
        }

        save_lora_only = config.checkpoint.get("save_lora_only", False) and use_lora
        
        if save_lora_only:
            save_lora_checkpoint(
                adapter=policy_model,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={"train_weighted_ce": train_weighted_ce_avg},
                val_metrics=final_aggregated,
                checkpoint_path=final_path,
                config={"model": {"path": config.models.policy.path}},
            )
        else:
            save_checkpoint(
                adapter=policy_model,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={"train_weighted_ce": train_weighted_ce_avg},
                val_metrics=final_aggregated,
                checkpoint_path=final_path,
                config={"model": {"path": config.models.policy.path}},
            )

        barrier()

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 12. 종료
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Verifiable WMTP 완료! Latest checkpoint: {latest_checkpoint_path}")

    final_metrics = final_aggregated if config.checkpoint.save_final else aggregated_val_metrics
    return final_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifiable WMTP (독립 Value Model)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/production/verifiable_pairwise.yaml)",
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

    run_verifiable_training(config)
