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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from weighted_mtp.data import AlpacaDataCollator, load_dataset
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.pipelines.checkpoint_utils import load_critic_checkpoint, save_checkpoint
from weighted_mtp.pipelines.metrics_utils import (
    GPUMonitor,
    ThroughputTracker,
    compute_gradient_norm,
    get_model_size,
    get_system_info,
)
from weighted_mtp.runtime import (
    init_distributed,
    setup_environment,
    is_main_process,
    wrap_model_ddp,
    unwrap_model,
    all_reduce_scalar,
)
from weighted_mtp.value_weighting.td_weighting import (
    build_weights,
    compute_td_errors,
    compute_td_stats,
    compute_weight_stats,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    )
    return adapter




def create_dataloader(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    n_samples: int | None,
    balance_correct: bool,
    correct_ratio: float,
    difficulty_weights: dict | None,
    difficulty_bins: dict | None,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader 생성 (Config-driven 샘플링)

    Args:
        dataset_path: 데이터셋 경로
        tokenizer: Tokenizer
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        n_samples: 샘플 수
        balance_correct: is_correct 균형 여부
        correct_ratio: correct 샘플 비율
        difficulty_weights: 난이도 가중치 (curriculum learning용)
        difficulty_bins: 난이도 구간 (curriculum learning용)
        seed: 시드
        shuffle: 셔플 여부

    Returns:
        DataLoader
    """
    # 데이터셋 이름 및 스플릿 추출
    # storage/datasets_v2/codecontests/processed/train.jsonl -> codecontests
    dataset_path_obj = Path(dataset_path)
    dataset_name = dataset_path_obj.parent.parent.name
    split_file = dataset_path_obj.name

    if "train" in split_file:
        split = "train"
    elif "valid" in split_file or "validation" in split_file:
        split = "validation"
    else:
        split = "test"

    # 데이터셋 로드 (Config-driven 샘플링)
    dataset = load_dataset(
        dataset_name=dataset_name,
        split=split,
        n_samples=n_samples,
        balance_correct=balance_correct,
        correct_ratio=correct_ratio,
        difficulty_weights=difficulty_weights,
        difficulty_bins=difficulty_bins,
        seed=seed,
    )

    # Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
    )

    return dataloader


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

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            is_correct = batch["is_correct"].to(device)

            # 2. is_correct → rewards 변환
            rewards = is_correct.float()

            # 3. full_forward (MTP + Value)
            outputs = adapter.full_forward(input_ids, attention_mask)
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

                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.float().reshape(-1)

                mask_sum_k = mask_k.float().sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # 7. Value loss
            value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
            loss_mask = attention_mask.unsqueeze(-1).float()

            if loss_type == "mse":
                loss_per_token = F.mse_loss(value_logits, value_targets, reduction="none")
            elif loss_type == "huber":
                loss_per_token = F.smooth_l1_loss(value_logits, value_targets, reduction="none")
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            masked_value_loss = loss_per_token * loss_mask
            value_loss = masked_value_loss.sum() / loss_mask.sum()

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


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    save_total_limit: int,
) -> None:
    """오래된 중간 checkpoint 삭제

    checkpoint_best.pt와 checkpoint_final.pt는 절대 삭제하지 않음
    checkpoint_epoch_*.pt만 save_total_limit 개수만큼 유지

    Args:
        checkpoint_dir: Checkpoint 디렉터리
        save_total_limit: 유지할 최대 개수
    """
    if not checkpoint_dir.exists():
        return

    # 중간 checkpoint 파일만 수집 (checkpoint_epoch_*.pt)
    epoch_checkpoints = sorted(
        [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: x.stat().st_mtime,  # 수정 시간 기준 정렬
    )

    # 삭제할 파일 개수 계산
    n_to_delete = len(epoch_checkpoints) - save_total_limit

    if n_to_delete > 0:
        for checkpoint_path in epoch_checkpoints[:n_to_delete]:
            logger.info(f"오래된 checkpoint 삭제: {checkpoint_path.name}")
            checkpoint_path.unlink()


def run_verifiable_training(
    config_path: str, **override_params: Any
) -> tuple[dict[str, float], str]:
    """Verifiable WMTP 실행 (Stage 2)

    Args:
        config_path: configs/verifiable/verifiable.yaml
        override_params: CLI overrides

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 1. Config 로딩 (defaults + verifiable config merge)
    defaults = OmegaConf.load("configs/defaults.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(defaults, config, override_params)

    # 2. 로깅 설정
    setup_logging(config.logging.level)

    # 3. Distributed 초기화 (torchrun 환경인 경우)
    if "RANK" in os.environ:
        rank, world_size = init_distributed()
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        rank, world_size = 0, 1
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

    # 6. MLflow 초기화 (Rank 0만)
    if is_main_process():
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: True for tag in config.experiment.tags},
        )
        # Config 로깅
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        mlflow.log_param("critic_checkpoint", config.experiment.critic_checkpoint)

    # 7. Resource 로딩
    adapter = load_adapter(config, device)
    tokenizer = load_tokenizer_from_config(config)

    # Model size + System info 로깅
    model_size = get_model_size(adapter)
    logger.info(
        f"Model size: {model_size['trainable_params']:,} trainable / "
        f"{model_size['total_params']:,} total params"
    )
    if is_main_process():
        mlflow.log_params(
            {
                "model_total_params": model_size["total_params"],
                "model_trainable_params": model_size["trainable_params"],
            }
        )

    system_info = get_system_info()
    if is_main_process():
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
    adapter = wrap_model_ddp(adapter, device)

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

    # Validation 샘플 수: train의 10% 또는 최소 100개
    val_n_samples = max(100, config.data_sampling.n_samples // 10)

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

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")

    current_epoch = 0.0
    batch_count = 0
    next_checkpoint_epoch = save_checkpoint_every

    # 9. Training loop
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

            # DataLoader 재생성 (새 difficulty_weights로 샘플링)
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
                seed=config.data_sampling.seed + int(current_epoch * 1000),  # Seed 변경
                shuffle=True,
            )

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

            rewards = is_correct.float()

            # full_forward (MTP + Value)
            outputs = adapter.full_forward(input_ids, attention_mask)
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

                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.float().reshape(-1)

                mask_sum_k = mask_k.float().sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # Value loss (Continual Learning)
            value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
            loss_mask = attention_mask.unsqueeze(-1).float()

            if config.training.loss_type == "mse":
                loss_per_token = F.mse_loss(value_logits, value_targets, reduction="none")
            elif config.training.loss_type == "huber":
                loss_per_token = F.smooth_l1_loss(
                    value_logits, value_targets, reduction="none"
                )
            else:
                raise ValueError(f"Unknown loss_type: {config.training.loss_type}")

            masked_value_loss = loss_per_token * loss_mask
            value_loss = masked_value_loss.sum() / loss_mask.sum()

            # Total loss
            total_loss = weighted_ce_loss + config.training.value_coef * value_loss

            # Backward & update
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient norm (clipping 전)
            grad_norm_dict = compute_gradient_norm(adapter)

            # Gradient clipping
            if config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    config.training.max_grad_norm,
                )

            optimizer.step()

            global_step += 1
            batch_count += 1
            period_batches += 1

            # Throughput tracking
            batch_size_actual = input_ids.size(0)
            n_tokens = attention_mask.sum().item()
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            # Metrics 누적
            period_metrics_sum["weighted_ce_loss"] += weighted_ce_loss.item()
            period_metrics_sum["value_loss"] += value_loss.item()
            period_metrics_sum["total_loss"] += total_loss.item()

            # Step-level 로깅
            if global_step % config.training.log_interval == 0:
                # TD error/weight stats
                td_stats = compute_td_stats(td_errors)
                weight_stats = compute_weight_stats(weights)
                gpu_metrics = gpu_monitor.get_metrics()

                # Metric aggregation (DDP)
                avg_weighted_ce = all_reduce_scalar(weighted_ce_loss.item())
                avg_value_loss = all_reduce_scalar(value_loss.item())
                avg_total_loss = all_reduce_scalar(total_loss.item())
                avg_grad_norm = all_reduce_scalar(grad_norm_dict["grad_norm"])
                avg_td_mean = all_reduce_scalar(td_stats["td_mean"])
                avg_weight_mean = all_reduce_scalar(weight_stats["weight_mean"])

                if is_main_process():
                    mlflow.log_metrics(
                        {
                            "train/weighted_ce_loss": avg_weighted_ce,
                            "train/value_loss": avg_value_loss,
                            "train/total_loss": avg_total_loss,
                            "train/grad_norm": avg_grad_norm,
                            "train/td_mean": avg_td_mean,
                            "train/weight_mean": avg_weight_mean,
                            "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                            "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                        },
                        step=global_step,
                    )
                logger.info(
                    f"Step {global_step}/{batches_to_run}, "
                    f"Total Loss: {avg_total_loss:.4f}, "
                    f"Weighted CE: {avg_weighted_ce:.4f}, "
                    f"Value: {avg_value_loss:.4f}"
                )

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
        if is_main_process():
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
                adapter=unwrap_model(adapter),
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

            logger.info(f"✓ Improved checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

            # 오래된 checkpoint 정리 (최대 3개 유지)
            if config.checkpoint.get("save_total_limit"):
                cleanup_old_checkpoints(
                    checkpoint_dir=checkpoint_dir,
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
            adapter=unwrap_model(adapter),
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

        logger.info(f"Final checkpoint saved: {final_path.name}")

    # 11. MLflow artifact 업로드 (Rank 0만)
    if is_main_process():
        # 최신 epoch checkpoint 업로드 (모두 validation loss 개선 시에만 저장되므로 best)
        epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if epoch_checkpoints:
            latest_checkpoint = epoch_checkpoints[-1]
            mlflow.log_artifact(str(latest_checkpoint), "checkpoints")
            logger.info(f"Latest checkpoint uploaded to MLflow: {latest_checkpoint.name}")
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
        "--critic-checkpoint",
        help="Critic checkpoint path override (optional, config takes priority)",
    )
    parser.add_argument("--run-name", help="MLflow run name override")
    parser.add_argument("--device", help="Device override (cuda/cpu/mps)")
    args = parser.parse_args()

    overrides = {}
    if args.critic_checkpoint:
        overrides["experiment.critic_checkpoint"] = args.critic_checkpoint
    if args.run_name:
        overrides["experiment.name"] = args.run_name
    if args.device:
        overrides["runtime.device"] = args.device

    run_verifiable_training(args.config, **overrides)
