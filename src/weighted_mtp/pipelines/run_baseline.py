"""Baseline MTP Runner (Uniform Weighting)

균등 가중치 기반 표준 MTP 학습 (비교 기준선)

독립 실행:
    python -m weighted_mtp.pipelines.run_baseline --config configs/baseline/baseline.yaml
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
    """Adapter 로드 (Value head 없이)

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        MetaLlamaMTPAdapter 인스턴스 (Value head 없음)
    """
    # Baseline은 Value head 불필요 (균등 가중치)
    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=device,
        initialize_value_head=False,
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
        balance_correct: is_correct 균형 여부 (Baseline은 False)
        correct_ratio: correct 샘플 비율 (Baseline은 1.0)
        seed: 시드
        shuffle: 셔플 여부

    Returns:
        DataLoader
    """
    # 데이터셋 이름 및 스플릿 추출
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
        difficulty_weights=None,  # Baseline은 curriculum learning 없음
        difficulty_bins=None,
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


def validate_baseline(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validation 수행 (Baseline - 균등 가중치)

    Args:
        adapter: Adapter (Value head 없음)
        dataloader: Validation DataLoader
        device: 디바이스

    Returns:
        Validation metrics
    """
    adapter.eval()

    total_ce_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 2. Forward (MTP만, Value head 없음)
            # Baseline은 Value head가 없으므로 transformer 직접 호출
            logits = adapter.transformer(input_ids, start_pos=0, return_all_heads=True)
            # logits: [batch, seq, n_future, vocab]

            batch_size, seq_len, n_future, vocab_size = logits.shape

            # 3. Uniform CE loss (모든 토큰 weight=1.0)
            batch_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                logits_k = logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                )

                # 균등 가중치 (weight=1.0)
                masked_ce_k = ce_loss_k * mask_k.float().reshape(-1)

                mask_sum_k = mask_k.float().sum()
                if mask_sum_k > 0:
                    batch_ce_loss += masked_ce_k.sum() / mask_sum_k

            ce_loss = batch_ce_loss / n_future

            # 4. Metrics 수집
            total_ce_loss += ce_loss.item()
            n_batches += 1

    # 평균 metrics 계산
    avg_ce_loss = total_ce_loss / n_batches

    metrics = {
        "val_loss": avg_ce_loss,
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
        key=lambda x: x.stat().st_mtime,
    )

    # 삭제할 파일 개수 계산
    n_to_delete = len(epoch_checkpoints) - save_total_limit

    if n_to_delete > 0:
        for checkpoint_path in epoch_checkpoints[:n_to_delete]:
            logger.info(f"오래된 checkpoint 삭제: {checkpoint_path.name}")
            checkpoint_path.unlink()


def run_baseline_training(
    config_path: str, **override_params: Any
) -> tuple[dict[str, float], str]:
    """Baseline MTP 실행 (균등 가중치)

    Args:
        config_path: configs/baseline/baseline.yaml
        override_params: CLI overrides

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 1. Config 로딩 (defaults + baseline config merge)
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

    logger.info("=== Baseline MTP (Uniform Weighting) ===")
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
            tags={tag: True for tag in config.experiment.tags},
        )
        # Config 로깅
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))

    # 6. Resource 로딩
    adapter = load_adapter(config, device)
    adapter = wrap_model_ddp(adapter, device)
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

    # 6. Optimizer (전체 파라미터) - Meta MTP 논문 설정
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

    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")

    current_epoch = 0.0
    batch_count = 0
    next_checkpoint_epoch = save_checkpoint_every

    # Throughput tracker 초기화
    throughput_tracker = ThroughputTracker()

    # 8. Training loop
    while batch_count < batches_to_run:
        # Checkpoint 경계까지 훈련
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        # DataLoader에서 필요한 만큼만 사용
        epoch_train_loader = iter(train_loader)
        period_loss_sum = 0.0
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                # DataLoader 재시작
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련 (Baseline 로직 - 균등 가중치)
            adapter.train()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward (MTP만, Value head 없음)
            logits = adapter.transformer(input_ids, start_pos=0, return_all_heads=True)
            # logits: [batch, seq, n_future, vocab]

            batch_size, seq_len, n_future, vocab_size = logits.shape

            # Uniform CE loss (모든 토큰 weight=1.0)
            batch_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                logits_k = logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                )

                # 균등 가중치 (weight=1.0)
                masked_ce_k = ce_loss_k * mask_k.float().reshape(-1)

                mask_sum_k = mask_k.float().sum()
                if mask_sum_k > 0:
                    batch_ce_loss += masked_ce_k.sum() / mask_sum_k

            ce_loss = batch_ce_loss / n_future

            # Backward & update
            optimizer.zero_grad()
            ce_loss.backward()

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
            period_loss_sum += ce_loss.item()

            # Step-level 로깅
            if global_step % config.training.log_interval == 0:
                gpu_metrics = gpu_monitor.get_metrics()

                # Metric aggregation (DDP)
                avg_loss = all_reduce_scalar(ce_loss.item())
                avg_grad_norm = all_reduce_scalar(grad_norm_dict["grad_norm"])

                if is_main_process():
                    if use_mlflow:
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
        train_loss_avg = period_loss_sum / period_batches
        train_loss_avg = all_reduce_scalar(train_loss_avg)

        logger.info(f"Epoch {current_epoch:.2f} 도달 - Train Loss: {train_loss_avg:.4f}")

        # Validation 실행
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")
        val_metrics = validate_baseline(
            adapter=adapter,
            dataloader=val_loader,
            device=device,
        )

        # Validation metrics aggregation
        avg_val_loss = all_reduce_scalar(val_metrics["val_loss"])

        # GPU metrics (epoch-level)
        gpu_metrics_epoch = gpu_monitor.get_metrics()

        # Throughput metrics 계산
        throughput_metrics = throughput_tracker.get_epoch_metrics()

        # Epoch-level 로깅
        if is_main_process():
            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train/epoch_loss": train_loss_avg,
                        "val/loss": avg_val_loss,
                        "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                        "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                        "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                        "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                    },
                    step=int(current_epoch * 100),
                )

        logger.info(f"Validation - Loss: {val_metrics['val_loss']:.4f}")

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

            # 오래된 checkpoint 정리
            if config.checkpoint.save_total_limit:
                cleanup_old_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    save_total_limit=config.checkpoint.save_total_limit,
                )
        else:
            logger.info(f"Validation loss did not improve ({avg_val_loss:.4f} >= {best_val_loss:.4f}), skipping checkpoint save")

        # 다음 checkpoint 경계 설정
        next_checkpoint_epoch += save_checkpoint_every

        # Throughput tracker 리셋
        throughput_tracker.reset_epoch()

    # 9. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        # 최종 validation 실행
        logger.info("--- Final Validation ---")
        final_val_metrics = validate_baseline(
            adapter=adapter,
            dataloader=val_loader,
            device=device,
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

    # 10. MLflow artifact 업로드 (Rank 0만)
    if is_main_process() and use_mlflow:
        # 최신 epoch checkpoint 업로드
        epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if epoch_checkpoints:
            latest_checkpoint = epoch_checkpoints[-1]
            mlflow.log_artifact(str(latest_checkpoint), "checkpoints")
            logger.info(f"Latest checkpoint uploaded to MLflow: {latest_checkpoint.name}")
        mlflow.end_run()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Baseline MTP 완료! Latest checkpoint: {latest_checkpoint_path}")

    # final_val_metrics가 정의되지 않은 경우 마지막 val_metrics 사용
    final_metrics = final_val_metrics if config.checkpoint.save_final else val_metrics
    return final_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline MTP (Uniform Weighting)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/baseline/baseline.yaml)",
    )
    parser.add_argument("--run-name", help="MLflow run name override")
    parser.add_argument("--device", help="Device override (cuda/cpu/mps)")
    args = parser.parse_args()

    overrides = {}
    if args.run_name:
        overrides["experiment.name"] = args.run_name
    if args.device:
        overrides["runtime.device"] = args.device

    run_baseline_training(args.config, **overrides)
