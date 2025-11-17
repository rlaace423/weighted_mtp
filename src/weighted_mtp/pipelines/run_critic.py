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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from weighted_mtp.data import AlpacaDataCollator, load_dataset
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.pipelines.checkpoint_utils import save_checkpoint
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


def load_tokenizer(config: dict) -> AutoTokenizer:
    """Tokenizer 로드

    Args:
        config: 모델 설정

    Returns:
        AutoTokenizer 인스턴스
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.policy.path,
        use_fast=True,
    )

    # Padding token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_dataloader(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    stage: str,
    n_samples: int | None,
    balance_correct: bool,
    correct_ratio: float,
    difficulty_weights: dict | None,
    difficulty_bins: dict | None,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader 생성

    Args:
        dataset_path: 데이터셋 경로
        tokenizer: Tokenizer
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        stage: Stage 식별자
        n_samples: 샘플 수
        balance_correct: is_correct 균형 여부
        correct_ratio: correct 샘플 비율
        difficulty_weights: 난이도 가중치
        difficulty_bins: 난이도 구간
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

    # 데이터셋 로드
    dataset = load_dataset(
        dataset_name=dataset_name,
        split=split,
        stage=stage,
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

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            is_correct = batch["is_correct"].to(device)

            # 2. is_correct → rewards 변환
            rewards = is_correct.float()

            # 3. trunk_forward
            outputs = adapter.trunk_forward(input_ids, attention_mask)
            value_logits = outputs["value_logits"]

            batch_size, seq_len, _ = value_logits.shape

            # 4. Value target 생성
            value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)

            # Mask padded tokens
            loss_mask = attention_mask.unsqueeze(-1).float()

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
            value_loss = masked_loss.sum() / loss_mask.sum()

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


def train_epoch_critic(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    log_interval: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """Critic 1 epoch 훈련

    Args:
        adapter: Adapter
        dataloader: Training DataLoader
        optimizer: Optimizer
        device: 디바이스
        loss_type: 손실 함수 타입
        log_interval: 로깅 간격
        global_step: 전역 스텝

    Returns:
        (train_metrics, updated_global_step)
    """
    adapter.train()

    total_loss = 0.0
    total_value_var = 0.0
    total_target_var = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # 1. Batch를 device로 이동
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        is_correct = batch["is_correct"].to(device)

        # 2. is_correct → rewards 변환
        rewards = is_correct.float()

        # 3. trunk_forward
        outputs = adapter.trunk_forward(input_ids, attention_mask)
        value_logits = outputs["value_logits"]

        batch_size, seq_len, _ = value_logits.shape

        # 4. Value target 생성
        value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)

        # Mask padded tokens
        loss_mask = attention_mask.unsqueeze(-1).float()

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
        value_loss = masked_loss.sum() / loss_mask.sum()

        # 6. Backward & update
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        # 7. Metrics 수집
        total_loss += value_loss.item()
        total_value_var += value_logits.var().item()
        total_target_var += value_targets.var().item()
        n_batches += 1
        global_step += 1

        # 8. Step-level 로깅 (MLflow)
        if global_step % log_interval == 0:
            mlflow.log_metrics(
                {
                    "train/loss": value_loss.item(),
                },
                step=global_step,
            )
            logger.info(
                f"Step {global_step}, "
                f"Loss: {value_loss.item():.4f}"
            )

    # 평균 metrics 계산
    avg_loss = total_loss / n_batches
    avg_target_var = total_target_var / n_batches

    # Value explained variance 계산
    if avg_target_var > 1e-8:
        explained_var = 1.0 - (avg_loss / avg_target_var)
    else:
        explained_var = 0.0

    metrics = {
        "train_loss": avg_loss,
        "train_explained_variance": explained_var,
    }

    return metrics, global_step


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


def run_critic_training(config_path: str, **override_params: Any) -> tuple[dict[str, float], str]:
    """Critic pre-training 실행

    Args:
        config_path: configs/critic/critic.yaml
        override_params: CLI overrides

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 1. Config 로딩 (defaults + critic config merge)
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

    logger.info("=== Critic Pre-training (Stage 1) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 5. MLflow 초기화 (Rank 0만)
    if is_main_process():
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: True for tag in config.experiment.tags},
        )
        # Config 로깅
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))

    # 6. Resource 로딩
    adapter = load_adapter(config, device)
    adapter = wrap_model_ddp(adapter, device)
    tokenizer = load_tokenizer(config)

    # Model size 로깅
    model_size = get_model_size(unwrap_model(adapter))
    if is_main_process():
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
            stage="stage1",
            n_samples=config.data_sampling.n_samples,
            balance_correct=config.data_sampling.balance_correct,
            correct_ratio=config.data_sampling.correct_ratio,
            difficulty_weights=None,
            difficulty_bins=None,
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
            stage="stage1",
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

        logger.info(f"Total epochs: {n_epochs}")
        logger.info(f"Total batches to run: {batches_to_run}")
        logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")

        current_epoch = 0.0
        batch_count = 0
        next_checkpoint_epoch = save_checkpoint_every

        # Throughput tracker 초기화
        throughput_tracker = ThroughputTracker()

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
                is_correct = batch["is_correct"].to(device)

                rewards = is_correct.float()
                outputs = adapter.trunk_forward(input_ids, attention_mask)
                value_logits = outputs["value_logits"]

                batch_size, seq_len, _ = value_logits.shape
                value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
                loss_mask = attention_mask.unsqueeze(-1).float()

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
                value_loss = masked_loss.sum() / loss_mask.sum()

                optimizer.zero_grad()
                value_loss.backward()

                # Gradient norm 계산 (clipping 전)
                grad_norm_dict = compute_gradient_norm(adapter)

                optimizer.step()

                global_step += 1
                batch_count += 1
                period_batches += 1

                # Throughput tracking
                batch_size_actual = input_ids.size(0)
                n_tokens = attention_mask.sum().item()
                throughput_tracker.update(batch_size_actual, int(n_tokens))

                # Metrics 누적
                period_metrics_sum["train_loss"] += value_loss.item()

                # Step-level 로깅
                if global_step % config.training.log_interval == 0:
                    # GPU metrics
                    gpu_metrics = gpu_monitor.get_metrics()

                    # Metric aggregation (DDP)
                    avg_loss = all_reduce_scalar(value_loss.item())
                    avg_grad_norm = all_reduce_scalar(grad_norm_dict["grad_norm"])

                    if is_main_process():
                        mlflow.log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/grad_norm": avg_grad_norm,
                                "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                            },
                            step=global_step,
                        )
                    logger.info(
                        f"Step {global_step}/{batches_to_run}, "
                        f"Loss: {avg_loss:.4f}, "
                        f"Grad Norm: {avg_grad_norm:.4f}"
                    )

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
                    adapter=unwrap_model(adapter),
                    optimizer=optimizer,
                    epoch=current_epoch,
                    train_metrics={"train_loss": train_loss_avg},
                    val_metrics=val_metrics,
                    checkpoint_path=checkpoint_path,
                )

                logger.info(f"✓ Improved checkpoint saved: {checkpoint_path.name} (val_loss: {best_val_loss:.4f})")

                # 오래된 checkpoint 정리 (최대 3개 유지)
                if config.checkpoint.save_total_limit:
                    cleanup_old_checkpoints(
                        checkpoint_dir=checkpoint_dir,
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
                adapter=unwrap_model(adapter),
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={"train_loss": train_loss_avg},
                val_metrics=final_val_metrics,
                checkpoint_path=final_path,
            )

            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 9. MLflow artifact 업로드 (Rank 0만)
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

    logger.info(f"Critic pre-training 완료! Latest checkpoint: {latest_checkpoint_path}")

    return final_val_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critic Pre-training (Stage 1)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/critic/critic.yaml)",
    )
    parser.add_argument("--run-name", help="MLflow run name override")
    parser.add_argument("--device", help="Device override (cuda/cpu/mps)")
    args = parser.parse_args()

    overrides = {}
    if args.run_name:
        overrides["experiment.name"] = args.run_name
    if args.device:
        overrides["runtime.device"] = args.device

    run_critic_training(args.config, **overrides)
