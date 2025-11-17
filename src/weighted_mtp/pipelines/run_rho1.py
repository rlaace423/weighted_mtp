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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from weighted_mtp.value_weighting.rho1_weighting import (
    build_weights,
    compute_excess_loss,
    compute_rho1_stats,
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


def load_reference_model(config: dict, device: torch.device) -> AutoModelForCausalLM:
    """Reference model 로드

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        Reference model (eval mode)
    """
    logger.info(f"Loading reference model: {config.models.reference.name}")
    logger.info(f"Path: {config.models.reference.path}")

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.models.reference.path,
        torch_dtype=getattr(torch, config.models.reference.dtype),
        device_map={"": device},
    )

    # Eval mode (gradient 불필요)
    ref_model.eval()

    # Gradient 계산 비활성화
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("✓ Reference model loaded successfully")

    return ref_model


def load_tokenizer(config: dict) -> AutoTokenizer:
    """Tokenizer 로드

    Args:
        config: 모델 설정

    Returns:
        AutoTokenizer 인스턴스
    """
    # tokenizer_shared_with 설정 확인
    if hasattr(config.models.reference, "tokenizer_shared_with"):
        # Reference와 Policy가 토크나이저 공유
        tokenizer_path = config.models.policy.path
    else:
        tokenizer_path = config.models.policy.path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
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
        difficulty_weights=None,  # Rho-1은 curriculum learning 없음
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


def validate_rho1(
    adapter: MetaLlamaMTPAdapter,
    ref_model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict[str, float]:
    """Validation 수행 (Rho-1)

    Args:
        adapter: Adapter (DDP-wrapped 가능)
        ref_model: Reference model
        dataloader: Validation DataLoader
        device: 디바이스
        temperature: Softmax temperature

    Returns:
        Validation metrics (DDP 환경에서는 all-reduce 적용됨)
    """
    # DDP unwrap for eval
    unwrapped_adapter = unwrap_model(adapter)
    unwrapped_adapter.eval()
    ref_model.eval()

    total_weighted_ce_loss = 0.0
    total_excess_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 2. Reference forward
            ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

            # 3. Policy forward (MTP)
            policy_outputs = unwrapped_adapter.full_forward(input_ids, attention_mask)
            policy_logits = policy_outputs["logits"]

            batch_size, seq_len, n_future, vocab_size = policy_logits.shape

            # 4. Excess loss 계산
            excess_loss = compute_excess_loss(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                labels=labels,
                attention_mask=attention_mask,
            )

            # 5. Rho-1 weights
            weights = build_weights(
                excess_loss=excess_loss,
                temperature=temperature,
                attention_mask=attention_mask,
            )

            # 6. Weighted CE loss (모든 H개 토큰)
            batch_weighted_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                if valid_len <= 0:
                    continue

                policy_logits_k = policy_logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, :valid_len]
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    policy_logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                )

                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.float().reshape(-1)

                mask_sum_k = mask_k.float().sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # 7. Metrics 수집
            total_weighted_ce_loss += weighted_ce_loss.item()
            total_excess_loss += excess_loss.mean().item()
            n_batches += 1

    # 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches
    avg_excess_loss = total_excess_loss / n_batches

    # Validation metrics aggregation (DDP)
    avg_weighted_ce_loss = all_reduce_scalar(avg_weighted_ce_loss)
    avg_excess_loss = all_reduce_scalar(avg_excess_loss)

    metrics = {
        "val_weighted_ce_loss": avg_weighted_ce_loss,
        "val_excess_loss": avg_excess_loss,
        "val_loss": avg_weighted_ce_loss,  # Best tracking용
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


def run_rho1_training(config_path: str, **override_params: Any) -> tuple[dict[str, float], str]:
    """Rho-1 WMTP 실행

    Args:
        config_path: configs/rho1/rho1.yaml
        override_params: CLI overrides

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 1. Config 로딩 (defaults + rho1 config merge)
    defaults = OmegaConf.load("configs/defaults.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(defaults, config, override_params)

    # 2. 로깅 설정
    setup_logging(config.logging.level)

    logger.info("=== Rho-1 WMTP (Reference-based Weighting) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    # 3. Distributed 초기화 (torchrun 환경인 경우)
    if "RANK" in os.environ:
        rank, world_size = init_distributed()
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        rank, world_size = 0, 1
        logger.info("Local training (single device)")

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
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))

    # 6. Resource 로딩
    adapter = load_adapter(config, device)
    ref_model = load_reference_model(config, device)
    tokenizer = load_tokenizer(config)

    # 7. DDP wrapping (adapter만 - reference는 frozen inference용)
    adapter = wrap_model_ddp(adapter, device)

    # Model size + System info 로깅 (Rank 0만)
    if is_main_process():
        model_size = get_model_size(unwrap_model(adapter))
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

    train_loader = create_dataloader(
        dataset_path=config.dataset.train,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        stage="stage1",  # 정답만 학습 (balance_correct=False)
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
        stage="stage1",
        n_samples=val_n_samples,
        balance_correct=config.data_sampling.balance_correct,
        correct_ratio=config.data_sampling.correct_ratio,
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

    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")
    logger.info(f"Temperature: {config.training.temperature}")

    current_epoch = 0.0
    batch_count = 0
    next_checkpoint_epoch = save_checkpoint_every

    # 8. Training loop
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

            # Reference forward (no grad)
            with torch.no_grad():
                ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits

            # Policy forward (MTP)
            policy_outputs = adapter.full_forward(input_ids, attention_mask)
            policy_logits = policy_outputs["logits"]

            batch_size, seq_len, n_future, vocab_size = policy_logits.shape

            # Excess loss 계산
            excess_loss = compute_excess_loss(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                labels=labels,
                attention_mask=attention_mask,
            )

            # Rho-1 weights
            weights = build_weights(
                excess_loss=excess_loss,
                temperature=config.training.temperature,
                attention_mask=attention_mask,
            )

            # Weighted CE loss (모든 H개 토큰)
            batch_weighted_ce_loss = 0.0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                if valid_len <= 0:
                    continue

                policy_logits_k = policy_logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, :valid_len]
                mask_k = attention_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    policy_logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                )

                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.float().reshape(-1)

                mask_sum_k = mask_k.float().sum()
                if mask_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

            weighted_ce_loss = batch_weighted_ce_loss / n_future

            # Backward & update
            optimizer.zero_grad()
            weighted_ce_loss.backward()

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

            # Metrics 누적
            period_metrics_sum["weighted_ce_loss"] += weighted_ce_loss.item()
            period_metrics_sum["excess_loss"] += excess_loss.mean().item()

            # Step-level 로깅
            if global_step % config.training.log_interval == 0:
                # Metric aggregation (DDP)
                avg_weighted_ce = all_reduce_scalar(weighted_ce_loss.item())
                avg_excess_loss = all_reduce_scalar(excess_loss.mean().item())

                if is_main_process():
                    mlflow.log_metrics(
                        {
                            "train/weighted_ce_loss": avg_weighted_ce,
                            "train/excess_loss": avg_excess_loss,
                        },
                        step=global_step,
                    )
                    logger.info(
                        f"Step {global_step}/{batches_to_run}, "
                        f"Weighted CE: {avg_weighted_ce:.4f}, "
                        f"Excess Loss: {avg_excess_loss:.4f}"
                    )

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
            temperature=config.training.temperature,
        )

        # Epoch-level 로깅 (Rank 0만)
        if is_main_process():
            mlflow.log_metrics(
                {
                    "train/epoch_weighted_ce_loss": train_weighted_ce_avg,
                    "train/epoch_excess_loss": train_excess_avg,
                    "val/weighted_ce_loss": val_metrics["val_weighted_ce_loss"],
                    "val/excess_loss": val_metrics["val_excess_loss"],
                },
                step=int(current_epoch * 100),
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
                adapter=unwrap_model(adapter),
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={
                    "train_weighted_ce_loss": train_weighted_ce_avg,
                    "train_excess_loss": train_excess_avg,
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
            temperature=config.training.temperature,
        )

        save_checkpoint(
            adapter=unwrap_model(adapter),
            optimizer=optimizer,
            epoch=current_epoch,
            train_metrics={
                "train_weighted_ce_loss": train_weighted_ce_avg,
                "train_excess_loss": train_excess_avg,
            },
            val_metrics=final_val_metrics,
            checkpoint_path=final_path,
        )

        logger.info(f"Final checkpoint saved: {final_path.name}")

    # 10. MLflow artifact 업로드 (Rank 0만)
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

    logger.info(f"🎉 Rho-1 WMTP 완료! Latest checkpoint: {latest_checkpoint_path}")

    return final_val_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rho-1 WMTP (Reference-based Weighting)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/rho1/rho1.yaml)",
    )
    parser.add_argument("--run-name", help="MLflow run name override")
    parser.add_argument("--device", help="Device override (cuda/cpu/mps)")
    args = parser.parse_args()

    overrides = {}
    if args.run_name:
        overrides["experiment.name"] = args.run_name
    if args.device:
        overrides["runtime.device"] = args.device

    run_rho1_training(args.config, **overrides)
