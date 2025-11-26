"""Verifiable WMTP Runner (Stage 2)

독립 실행:
    python -m weighted_mtp.pipelines.run_verifiable --config configs/verifiable/verifiable.yaml
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
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    cleanup_s3_checkpoints,
    compute_gradient_clip_stats,
    compute_mtp_ce_loss,
    compute_pairwise_accuracy,
    compute_value_function_stats,
    compute_weight_statistics,
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
from weighted_mtp.value_weighting.td_weighting import (
    build_weights,
    compute_td_errors,
    compute_td_stats,
)


def validate_verifiable(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    weight_clip_min: float,
    weight_clip_max: float,
) -> dict[str, float]:
    """Validation 수행 (Stage 2) - Pairwise 전용

    Args:
        adapter: Adapter
        dataloader: Validation DataLoader (pairwise format)
        device: 디바이스
        beta: TD error weighting 계수
        weight_clip_min: Weight 최소값
        weight_clip_max: Weight 최대값

    Returns:
        Validation metrics (weighted_ce_loss, value_loss, pairwise_accuracy, margin)
    """
    adapter.eval()

    total_weighted_ce_loss = 0.0
    total_unweighted_ce_loss = 0.0
    total_value_loss = 0.0
    total_loss_sum = 0.0
    n_batches = 0

    # Pairwise metrics
    total_pairwise_accuracy = 0.0
    total_margin = 0.0

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

            pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)

            # Forward (Positive)
            pos_outputs = adapter(
                pos_input_ids,
                pos_attention_mask,
                return_value_logits=True,
                return_hidden_states=True
            )
            pos_logits = pos_outputs["logits"]
            pos_value_logits = pos_outputs["value_logits"]

            # Forward (Negative)
            neg_outputs = adapter(
                neg_input_ids,
                neg_attention_mask,
                return_value_logits=True,
                return_hidden_states=False
            )
            neg_value_logits = neg_outputs["value_logits"]

            n_future = pos_logits.shape[2]

            # TD error (Positive만 - Policy weighting용)
            # Response 토큰 마스크 (labels != -100)
            pos_response_mask = (pos_labels != -100).long()

            td_errors = compute_td_errors(
                value_logits=pos_value_logits,
                rewards=pos_rewards,
                attention_mask=pos_attention_mask,
                gamma=1.0,
            )
            weights = build_weights(
                td_errors=td_errors,
                attention_mask=pos_response_mask,
                beta=beta,
                min_weight=weight_clip_min,
                max_weight=weight_clip_max,
            )

            # Policy Loss (Positive만, TD Weighted) - 메모리 최적화된 유틸리티 사용
            ce_losses = compute_mtp_ce_loss(
                logits=pos_logits,
                labels=pos_labels,
                attention_mask=pos_attention_mask,
                weights=weights,
                n_future=n_future,
            )
            weighted_ce_loss = ce_losses["weighted_ce_loss"]
            unweighted_ce_loss = ce_losses["unweighted_ce_loss"]

            # Pairwise Ranking Loss (value head 학습)
            pos_mask = (pos_labels != -100).to(model_dtype)
            neg_mask = (neg_labels != -100).to(model_dtype)
            value_loss = pairwise_ranking_loss(pos_value_logits, neg_value_logits, pos_mask, neg_mask)

            # Pairwise accuracy
            pairwise_metrics = compute_pairwise_accuracy(pos_value_logits, neg_value_logits, pos_mask, neg_mask)
            total_pairwise_accuracy += pairwise_metrics["pairwise_accuracy"]
            total_margin += pairwise_metrics["margin"]

            # Total Loss (trunk: CE, value_head: pairwise)
            total_loss = weighted_ce_loss + value_loss

            # Metrics 수집
            total_weighted_ce_loss += weighted_ce_loss.item()
            total_unweighted_ce_loss += unweighted_ce_loss.item()
            total_value_loss += value_loss.item()
            total_loss_sum += total_loss.item()
            n_batches += 1

    # 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches
    avg_unweighted_ce_loss = total_unweighted_ce_loss / n_batches
    avg_value_loss = total_value_loss / n_batches
    avg_total_loss = total_loss_sum / n_batches

    metrics = {
        "val_weighted_ce_loss": avg_weighted_ce_loss,
        "val_unweighted_ce_loss": avg_unweighted_ce_loss,
        "val_value_loss": avg_value_loss,
        "val_loss": avg_total_loss,
        "val_pairwise_accuracy": total_pairwise_accuracy / n_batches,
        "val_margin": total_margin / n_batches,
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

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 6. MLflow 초기화 (Rank 0만, experiment 이름이 있는 경우만)
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

    # 5. 모델 로드 (pretrained 또는 checkpoint)
    logger.info(f"Loading model: {config.models.policy.path}")
    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=device,
        dtype=config.models.policy.dtype,
    )
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

    # 6. FSDP wrapping
    adapter = wrap_model_fsdp(
        adapter,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )

    # 10. Optimizer (param groups: trunk/value_head 분리)
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

    # 11. Training setup
    best_val_loss = float("inf")
    global_step = 0

    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = config.training.n_epochs
    save_checkpoint_every = config.checkpoint.save_checkpoint_every

    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    # sampling_config를 dict로 변환 (항상 pairwise 모드)
    sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
    sampling_config["use_pairwise"] = True
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
    if "difficulty" in val_sampling_config and val_sampling_config["difficulty"]:
        val_sampling_config["difficulty"] = val_sampling_config["difficulty"].copy()
        val_sampling_config["difficulty"]["n_samples"] = config.data_sampling.val_n_samples
    else:
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

    # Gradient accumulation 초기화
    accumulation_counter = 0
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    # Optimization steps 계산
    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

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

    # 모델 dtype 감지
    model_dtype = next(adapter.parameters()).dtype

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

        # Throughput 추적 시작
        throughput_tracker.start_epoch()

        # DataLoader에서 필요한 만큼만 사용
        epoch_train_loader = iter(train_loader)
        period_metrics_sum = {
            "weighted_ce_loss": 0.0,
            "unweighted_ce_loss": 0.0,
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

            # Pairwise 모드: Positive/Negative 데이터 추출
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_labels = batch["pos_labels"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)
            neg_labels = batch["neg_labels"].to(device)

            # Positive sample: reward=1 (correct)
            pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)

            # Forward (Positive) - Policy Loss + Value Loss 사용
            pos_outputs = adapter(
                pos_input_ids,
                pos_attention_mask,
                return_value_logits=True,
                return_hidden_states=True
            )
            pos_logits = pos_outputs["logits"]
            pos_value_logits = pos_outputs["value_logits"]

            # Pairwise loss용 value_logits (trunk gradient는 adapter에서 이미 차단됨)
            pos_value_for_ranking = pos_value_logits

            # Forward (Negative) - Value Loss만 사용 (no_grad로 메모리 절감)
            with torch.no_grad():
                neg_outputs = adapter(
                    neg_input_ids,
                    neg_attention_mask,
                    return_value_logits=True,
                    return_hidden_states=False
                )
            neg_value_logits = neg_outputs["value_logits"]

            # TD error 계산 (Positive만 - Policy weighting용)
            # Response 토큰 마스크 (labels != -100)
            pos_response_mask = (pos_labels != -100).long()

            td_errors = compute_td_errors(
                value_logits=pos_value_logits,
                rewards=pos_rewards,
                attention_mask=pos_attention_mask,
                gamma=1.0,
            )

            # Weight 산출 (Normalized IQL with Advantage Whitening)
            # Response 토큰만으로 정규화 (Instruction 제외)
            weights = build_weights(
                td_errors=td_errors,
                attention_mask=pos_response_mask,
                beta=config.training.beta,
                min_weight=config.training.weight_clip_min,
                max_weight=config.training.weight_clip_max,
            )

            # Policy Loss (Positive만, TD Weighted) - 메모리 최적화된 유틸리티 사용
            ce_losses = compute_mtp_ce_loss(
                logits=pos_logits,
                labels=pos_labels,
                attention_mask=pos_attention_mask,
                weights=weights,
            )
            weighted_ce_loss = ce_losses["weighted_ce_loss"]
            unweighted_ce_loss = ce_losses["unweighted_ce_loss"]

            # Pairwise Ranking Loss (value head만 학습, trunk gradient 차단됨)
            pos_mask = (pos_labels != -100).to(model_dtype)
            neg_mask = (neg_labels != -100).to(model_dtype)
            value_loss = pairwise_ranking_loss(pos_value_for_ranking, neg_value_logits, pos_mask, neg_mask)

            # Total Loss (trunk은 CE로만, value_head는 pairwise로만 학습)
            total_loss = weighted_ce_loss + value_loss

            # 로깅용 변수 설정
            input_ids = pos_input_ids
            attention_mask = pos_attention_mask
            labels = pos_labels
            rewards = pos_rewards
            value_logits = pos_value_logits
            valid_label_mask = pos_mask.unsqueeze(-1)
            # pairwise 모드에서는 rewards를 td_targets로 사용 (로깅용)
            td_targets = rewards.view(-1, 1, 1).expand_as(value_logits)

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
            period_metrics_sum["unweighted_ce_loss"] += unweighted_ce_loss.item()
            period_metrics_sum["value_loss"] += value_loss.item()
            period_metrics_sum["total_loss"] += total_loss.item()

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
                # Response 토큰 마스크 (통계 계산용)
                response_mask = valid_label_mask.squeeze(-1)

                # TD error stats (response 토큰만)
                td_stats = compute_td_stats(td_errors, response_mask)
                gpu_metrics = gpu_monitor.get_metrics()

                # Value function statistics (response 토큰만)
                value_func_stats = compute_value_function_stats(
                    values=value_logits.squeeze(-1),
                    returns=td_targets.squeeze(-1),
                    attention_mask=response_mask,
                )

                # Weight distribution statistics (response 토큰만)
                weight_dist_stats = compute_weight_statistics(weights, attention_mask, labels)

                # Metric aggregation (분산 환경)
                # grad_clip_stats는 clip_grad_norm_이 이미 전역 값을 반환하므로 all_reduce 불필요
                avg_grad_norm_post = grad_clip_stats["grad_norm_post_clip"]
                avg_grad_norm_pre = grad_clip_stats["grad_norm_pre_clip"]
                avg_grad_clip_ratio = grad_clip_stats["grad_clip_ratio"]

                # 모든 로컬 메트릭 배치 all_reduce (1회 통신)
                reduced = all_reduce_scalars({
                    "weighted_ce": weighted_ce_loss.item(),
                    "unweighted_ce": unweighted_ce_loss.item(),
                    "value_loss": value_loss.item(),
                    "total_loss": total_loss.item(),
                    "td_mean": td_stats["td_mean"],
                    "td_std": td_stats["td_std"],
                    "td_min": td_stats["td_min"],
                    "td_max": td_stats["td_max"],
                    "value_mse": value_func_stats["value_mse"],
                    "value_mean": value_func_stats["value_mean"],
                    "value_std": value_func_stats["value_std"],
                    "weight_mean": weight_dist_stats["weight_mean"],
                    "weight_std": weight_dist_stats["weight_std"],
                    "weight_min": weight_dist_stats["weight_min"],
                    "weight_max": weight_dist_stats["weight_max"],
                    "weight_entropy": weight_dist_stats["weight_entropy"],
                })
                avg_weighted_ce = reduced["weighted_ce"]
                avg_unweighted_ce = reduced["unweighted_ce"]
                avg_value_loss = reduced["value_loss"]
                avg_total_loss = reduced["total_loss"]
                avg_td_mean = reduced["td_mean"]
                avg_value_mse = reduced["value_mse"]
                avg_value_mean = reduced["value_mean"]
                avg_value_std = reduced["value_std"]

                if is_main_process():
                    if use_mlflow:
                        mlflow.log_metrics(
                            {
                                "train/weighted_ce_loss": avg_weighted_ce,
                                "train/unweighted_ce_loss": avg_unweighted_ce,
                                "train/value_loss": avg_value_loss,
                                "train/total_loss": avg_total_loss,
                                "train/grad_norm": avg_grad_norm_post,
                                "train/grad_norm_pre_clip": avg_grad_norm_pre,
                                "train/grad_clip_ratio": avg_grad_clip_ratio,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "td/mean": avg_td_mean,
                                "td/std": reduced["td_std"],
                                "td/min": reduced["td_min"],
                                "td/max": reduced["td_max"],
                                "value/mse": avg_value_mse,
                                "value/mean_prediction": avg_value_mean,
                                "value/std_prediction": avg_value_std,
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
                        f"Loss: {avg_total_loss:.4f}, "
                        f"Grad Norm: {avg_grad_norm_post:.4f} (Clip: {avg_grad_clip_ratio:.2f})"
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

        # Period-level metrics 계산 & aggregation (1회 통신)
        train_weighted_ce_avg = period_metrics_sum["weighted_ce_loss"] / period_batches
        train_unweighted_ce_avg = period_metrics_sum["unweighted_ce_loss"] / period_batches
        train_value_avg = period_metrics_sum["value_loss"] / period_batches
        train_total_avg = period_metrics_sum["total_loss"] / period_batches

        reduced_period = all_reduce_scalars({
            "weighted_ce": train_weighted_ce_avg,
            "unweighted_ce": train_unweighted_ce_avg,
            "value": train_value_avg,
            "total": train_total_avg,
        })
        train_weighted_ce_avg = reduced_period["weighted_ce"]
        train_unweighted_ce_avg = reduced_period["unweighted_ce"]
        train_value_avg = reduced_period["value"]
        train_total_avg = reduced_period["total"]

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
            weight_clip_min=config.training.weight_clip_min,
            weight_clip_max=config.training.weight_clip_max,
        )

        # Validation metrics aggregation (1회 통신)
        val_reduce_dict = {
            "val_total": val_metrics["val_loss"],
            "val_weighted_ce": val_metrics["val_weighted_ce_loss"],
            "val_unweighted_ce": val_metrics["val_unweighted_ce_loss"],
            "val_value": val_metrics["val_value_loss"],
            "val_pairwise_accuracy": val_metrics["val_pairwise_accuracy"],
            "val_margin": val_metrics["val_margin"],
        }
        reduced_val = all_reduce_scalars(val_reduce_dict)
        avg_val_total = reduced_val["val_total"]
        avg_val_weighted_ce = reduced_val["val_weighted_ce"]
        avg_val_unweighted_ce = reduced_val["val_unweighted_ce"]
        avg_val_value = reduced_val["val_value"]

        # Epoch-level 로깅
        if is_main_process():
            # Throughput 및 GPU 메트릭 수집
            throughput_metrics = throughput_tracker.get_epoch_metrics()
            gpu_metrics_epoch = gpu_monitor.get_metrics()

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train/epoch_total_loss": train_total_avg,
                        "train/epoch_weighted_ce_loss": train_weighted_ce_avg,
                        "train/epoch_unweighted_ce_loss": train_unweighted_ce_avg,
                        "train/epoch_value_loss": train_value_avg,
                        "val/total_loss": avg_val_total,
                        "val/weighted_ce_loss": avg_val_weighted_ce,
                        "val/unweighted_ce_loss": avg_val_unweighted_ce,
                        "val/value_loss": avg_val_value,
                        "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                        "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                        "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                        "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                    },
                    step=global_step,
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
                config={"model": {"path": config.models.policy.path}},
                s3_upload=use_s3_upload,
                mlflow_run_id=mlflow_run_id,
            )

            # 모든 GPU가 checkpoint 저장 완료까지 대기
            barrier()

            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

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
            config={"model": {"path": config.models.policy.path}},
            s3_upload=use_s3_upload,
            mlflow_run_id=mlflow_run_id,
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
