"""학습 파이프라인 (Stage 1/2 + 오케스트레이션)

Stage 1: Value Head Pretrain
Stage 2: Weighted Training with Critic Continual Learning
run_training_pipeline(): 전체 파이프라인 오케스트레이션
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from weighted_mtp.models.meta_mtp import MetaLlamaMTPAdapter
from weighted_mtp.value_weighting import (
    build_weights,
    compute_td_errors,
    compute_td_stats,
    compute_weight_stats,
)

logger = logging.getLogger(__name__)


def train_stage1(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Stage 1: Value Head Pretrain (Probabilistic Value Learning)

    Value head가 V(s_t) = P(Success | s_t)를 학습하도록 사전 학습합니다.

    핵심 원리:
    - 모든 토큰에 R_terminal (0 or 1)을 target으로 부여
    - 배치 학습을 통해 V(s_t) → P(Success | s_t) 자동 수렴
    - 동일 prefix가 다른 샘플에서 다른 R을 가지면 확률 추정

    예시:
      Sample 1 (R=1): [...A, B, C, D] → V(s_C) target = 1.0
      Sample 2 (R=0): [...A, B, C, X] → V(s_C) target = 0.0
      배치 학습 → V(s_C) ≈ 0.5 = P(Success | s_C)

    이후 Stage 2에서 TD error δ_t = V(s_{t+1}) - V(s_t)로
    토큰별 성공 확률 변화량(ΔP)을 측정하여 가중치 산출

    MTP output heads는 사용하지 않고 Value head만 학습합니다.

    Args:
        adapter: MetaLlamaMTPAdapter (Value head 포함)
        dataloader: Stage 1 DataLoader
            배치 형식: {
                "input_ids": [batch, seq],
                "attention_mask": [batch, seq],
                "labels": [batch, seq],
                "is_correct": [batch] - Boolean 또는 0/1
            }
        optimizer: Value head 전용 optimizer
        config: 학습 설정
            {
                "n_epochs": 0.5,
                "loss_type": "mse" or "huber",
            }
        device: torch.device

    Returns:
        metrics: {
            "stage1_loss": float,
            "value_explained_variance": float,
        }
    """
    adapter.train()

    total_loss = 0.0
    total_value_var = 0.0
    total_target_var = 0.0
    n_batches = 0

    loss_type = config.get("loss_type", "mse")

    for batch_idx, batch in enumerate(dataloader):
        # 1. Batch를 device로 이동
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        is_correct = batch["is_correct"].to(device)

        # 2. is_correct → rewards 변환 (float)
        rewards = is_correct.float()  # [batch]

        # 3. trunk_forward (MTP heads 사용 안 함)
        outputs = adapter.trunk_forward(input_ids, attention_mask)
        value_logits = outputs["value_logits"]  # [batch, seq, 1]

        batch_size, seq_len, _ = value_logits.shape

        # 4. Value target 생성 (Undiscounted Monte Carlo)
        # 모든 토큰에 동일한 R_terminal 부여 (γ=1.0)
        # rewards: [batch] → [batch, seq, 1]
        value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)

        # Mask padded tokens
        loss_mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]

        # 5. Value loss 계산 (패딩 토큰 제외)
        if loss_type == "mse":
            loss_per_token = F.mse_loss(value_logits, value_targets, reduction="none")
        elif loss_type == "huber":
            loss_per_token = F.smooth_l1_loss(value_logits, value_targets, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Masked loss (패딩 토큰 제외)
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

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Stage 1 - Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {value_loss.item():.4f}"
            )

    # 8. 평균 metrics 계산
    avg_loss = total_loss / n_batches
    avg_target_var = total_target_var / n_batches

    # Value explained variance 계산
    if avg_target_var > 1e-8:
        explained_var = 1.0 - (avg_loss / avg_target_var)
    else:
        explained_var = 0.0

    metrics = {
        "stage1_loss": avg_loss,
        "value_explained_variance": explained_var,
    }

    logger.info(
        f"Stage 1 완료 - Loss: {avg_loss:.4f}, "
        f"Explained Variance: {explained_var:.4f}"
    )

    return metrics


def train_stage2(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Stage 2: Weighted Training with Critic Continual Learning

    TD error 기반 가중치로 MTP를 학습하며, Value head를 continual learning으로 유지합니다.

    핵심 원리:
    - 모든 H개 미래 토큰에 대한 weighted CE loss 계산 (Meta MTP 2024 방식)
    - TD error 기반 exponential weighting: weight_k = exp(td_error_k / β)
    - MTP 가중치 시점 정렬: logits[t, k-1, :]의 가중치 = td_errors[t+k-1]
    - Critic Continual Learning: Value loss를 auxiliary loss로 추가

    MTP output heads와 Value head를 모두 학습합니다.

    Args:
        adapter: MetaLlamaMTPAdapter (Value head 포함)
        dataloader: Stage 2 DataLoader (curriculum learning)
            배치 형식: {
                "input_ids": [batch, seq],
                "attention_mask": [batch, seq],
                "labels": [batch, seq],
                "is_correct": [batch] - Boolean 또는 0/1
            }
        optimizer: 전체 파라미터 optimizer
        config: 학습 설정
            {
                "n_epochs": 2.5,
                "beta": 0.9,
                "value_coef": 0.5,
                "clip_range": 0.2,  # Value loss clipping (optional)
                "max_grad_norm": 0.5,
                "loss_type": "mse" or "huber",  # Value loss type
            }
        device: torch.device

    Returns:
        metrics: {
            "stage2_weighted_ce_loss": float,
            "stage2_value_loss": float,
            "stage2_total_loss": float,
            "td_mean": float,
            "weight_mean": float,
            "value_explained_variance": float,
        }
    """
    adapter.train()

    total_weighted_ce_loss = 0.0
    total_value_loss = 0.0
    total_loss_sum = 0.0
    total_td_mean = 0.0
    total_weight_mean = 0.0
    total_value_var = 0.0
    total_target_var = 0.0
    n_batches = 0

    beta = config.get("beta", 0.9)
    value_coef = config.get("value_coef", 0.5)
    max_grad_norm = config.get("max_grad_norm", 0.5)
    loss_type = config.get("loss_type", "mse")

    for batch_idx, batch in enumerate(dataloader):
        # 1. Batch를 device로 이동
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        is_correct = batch["is_correct"].to(device)

        # 2. is_correct → rewards 변환 (float)
        rewards = is_correct.float()  # [batch]

        # 3. full_forward (MTP + Value)
        outputs = adapter.full_forward(input_ids, attention_mask)
        logits = outputs["logits"]  # [batch, seq, n_future_tokens, vocab]
        value_logits = outputs["value_logits"]  # [batch, seq, 1]

        batch_size, seq_len, n_future, vocab_size = logits.shape

        # 4. TD error 계산
        td_errors = compute_td_errors(
            value_logits=value_logits,
            rewards=rewards,
            attention_mask=attention_mask,
            gamma=1.0,
        )  # [batch, seq]

        # 5. Weight 산출 (TD error → Exponential weighting)
        weights = build_weights(
            td_errors=td_errors,
            beta=beta,
            min_weight=0.1,
            max_weight=5.0,
        )  # [batch, seq]

        # 6. Weighted CE loss 계산 (모든 H개 미래 토큰 평균)
        # MTP 가중치 시점 정렬: logits[t, k-1, :]의 가중치 = weights[t+k-1]
        batch_weighted_ce_loss = 0.0

        for k in range(1, n_future + 1):
            # 시점 t에서 예측한 x_{t+k}
            # logits[t, k-1, :]: x_{t+k} 예측 logits
            # labels[t+k]: 실제 x_{t+k} 토큰
            # weights[t+k-1]: x_{t+k}의 가중치 (δ_{t+k-1})

            # 유효 시퀀스 길이 (마지막 k개 시점은 labels 부족)
            valid_len = seq_len - k

            logits_k = logits[:, :valid_len, k - 1, :]  # [batch, valid_len, vocab]
            labels_k = labels[:, k : k + valid_len]  # [batch, valid_len]
            weights_k = weights[:, k - 1 : k - 1 + valid_len]  # [batch, valid_len]
            mask_k = attention_mask[:, k : k + valid_len]  # [batch, valid_len]

            # CE loss (per-token)
            ce_loss_k = F.cross_entropy(
                logits_k.reshape(-1, vocab_size),
                labels_k.reshape(-1),
                reduction="none",
            )  # [batch*valid_len]

            # Weighted loss (masking 적용)
            weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.float().reshape(-1)

            # k번째 미래 토큰의 평균 weighted loss
            mask_sum_k = mask_k.float().sum()
            if mask_sum_k > 0:
                batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

        # H개 토큰 평균
        weighted_ce_loss = batch_weighted_ce_loss / n_future

        # 7. Value loss (Auxiliary, Continual Learning)
        # Stage 1과 동일한 방식으로 V(s_t) = P(Success | s_t) 유지
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

        # 9. Backward & update
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                max_grad_norm,
            )

        optimizer.step()

        # 10. Metrics 수집
        total_weighted_ce_loss += weighted_ce_loss.item()
        total_value_loss += value_loss.item()
        total_loss_sum += total_loss.item()

        # TD error 및 weight 통계
        td_stats = compute_td_stats(td_errors)
        weight_stats = compute_weight_stats(weights)
        total_td_mean += td_stats["td_mean"]
        total_weight_mean += weight_stats["weight_mean"]

        total_value_var += value_logits.var().item()
        total_target_var += value_targets.var().item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Stage 2 - Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Weighted CE Loss: {weighted_ce_loss.item():.4f}, "
                f"Value Loss: {value_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}"
            )

    # 11. 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches
    avg_value_loss = total_value_loss / n_batches
    avg_total_loss = total_loss_sum / n_batches
    avg_td_mean = total_td_mean / n_batches
    avg_weight_mean = total_weight_mean / n_batches
    avg_target_var = total_target_var / n_batches

    # Value explained variance 계산
    if avg_target_var > 1e-8:
        explained_var = 1.0 - (avg_value_loss / avg_target_var)
    else:
        explained_var = 0.0

    metrics = {
        "stage2_weighted_ce_loss": avg_weighted_ce_loss,
        "stage2_value_loss": avg_value_loss,
        "stage2_total_loss": avg_total_loss,
        "td_mean": avg_td_mean,
        "weight_mean": avg_weight_mean,
        "value_explained_variance": explained_var,
    }

    logger.info(
        f"Stage 2 완료 - "
        f"Weighted CE Loss: {avg_weighted_ce_loss:.4f}, "
        f"Value Loss: {avg_value_loss:.4f}, "
        f"Total Loss: {avg_total_loss:.4f}, "
        f"TD Mean: {avg_td_mean:.4f}, "
        f"Weight Mean: {avg_weight_mean:.4f}, "
        f"Explained Variance: {explained_var:.4f}"
    )

    return metrics


def run_training_pipeline(
    adapter: MetaLlamaMTPAdapter,
    stage1_dataloader: DataLoader,
    stage2_dataloader: DataLoader,
    config: dict,
    device: torch.device,
    save_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """전체 학습 파이프라인 오케스트레이션

    Stage 1 (Value Head Pretrain) → Stage 2 (Weighted Training) 순차 실행

    Args:
        adapter: MetaLlamaMTPAdapter (Value head 포함)
        stage1_dataloader: Stage 1 DataLoader
            - Balanced sampling (correct/incorrect 균형)
            - Stage 1은 분포 균형이 중요 (확률 학습)
        stage2_dataloader: Stage 2 DataLoader
            - Curriculum learning (difficulty 기반)
            - Stage 2는 점진적 난이도 상승
        config: 전체 파이프라인 설정
            {
                "stage1": {
                    "n_epochs": 0.5,
                    "loss_type": "mse",
                    "learning_rate": 1e-4,
                },
                "stage2": {
                    "n_epochs": 2.5,
                    "beta": 0.9,
                    "value_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "loss_type": "mse",
                    "learning_rate": 1e-5,
                },
                "save_checkpoint_every": 1,  # epoch마다 저장 (선택적)
            }
        device: torch.device
        save_dir: Checkpoint 저장 디렉터리 (None이면 저장 안 함)

    Returns:
        전체 파이프라인 metrics
        {
            "stage1": {"stage1_loss": float, "value_explained_variance": float},
            "stage2": {
                "stage2_weighted_ce_loss": float,
                "stage2_value_loss": float,
                "stage2_total_loss": float,
                "td_mean": float,
                "weight_mean": float,
                "value_explained_variance": float,
            },
        }

    Examples:
        >>> from pathlib import Path
        >>> from weighted_mtp.data import load_dataset, AlpacaDataCollator
        >>> from weighted_mtp.models.meta_mtp import load_adapter
        >>>
        >>> # Adapter 로딩
        >>> adapter = load_adapter("storage/models_v2/micro-mtp")
        >>> adapter = adapter.to(device)
        >>>
        >>> # Stage 1 데이터 로딩 (balanced)
        >>> stage1_dataset = load_dataset("codecontests_small", split="train", stage="stage1")
        >>> collator = AlpacaDataCollator(tokenizer, max_length=2048)
        >>> stage1_dataloader = DataLoader(stage1_dataset, batch_size=4, collate_fn=collator)
        >>>
        >>> # Stage 2 데이터 로딩 (curriculum)
        >>> stage2_dataset = load_dataset("codecontests_small", split="train", stage="stage2")
        >>> stage2_dataloader = DataLoader(stage2_dataset, batch_size=4, collate_fn=collator)
        >>>
        >>> # Config
        >>> config = {
        ...     "stage1": {"n_epochs": 0.5, "loss_type": "mse", "learning_rate": 1e-4},
        ...     "stage2": {"n_epochs": 2.5, "beta": 0.9, "value_coef": 0.5, "learning_rate": 1e-5},
        ... }
        >>>
        >>> # 전체 파이프라인 실행
        >>> metrics = run_training_pipeline(
        ...     adapter=adapter,
        ...     stage1_dataloader=stage1_dataloader,
        ...     stage2_dataloader=stage2_dataloader,
        ...     config=config,
        ...     device=device,
        ...     save_dir=Path("checkpoints"),
        ... )
    """
    logger.info("=" * 80)
    logger.info("학습 파이프라인 시작")
    logger.info("=" * 80)

    all_metrics = {}

    # Stage 1 설정 추출
    stage1_config = config.get("stage1", {})
    stage1_epochs = stage1_config.get("n_epochs", 0.5)
    stage1_lr = stage1_config.get("learning_rate", 1e-4)

    # Stage 2 설정 추출
    stage2_config = config.get("stage2", {})
    stage2_epochs = stage2_config.get("n_epochs", 2.5)
    stage2_lr = stage2_config.get("learning_rate", 1e-5)

    # Checkpoint 저장 주기
    save_checkpoint_every = config.get("save_checkpoint_every", None)

    # ========== Stage 1: Value Head Pretrain ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Stage 1: Value Head Pretrain ({stage1_epochs} epochs)")
    logger.info("=" * 80)

    # Stage 1 전용 optimizer (Value head만)
    stage1_optimizer = torch.optim.AdamW(
        adapter.value_head.parameters(),
        lr=stage1_lr,
    )

    # Epoch loop
    stage1_total_batches = len(stage1_dataloader)
    stage1_batches_to_run = int(stage1_total_batches * stage1_epochs)

    logger.info(
        f"Stage 1 - Total batches: {stage1_total_batches}, "
        f"Batches to run: {stage1_batches_to_run} ({stage1_epochs} epochs)"
    )

    # Stage 1 학습 (train_stage1은 1 epoch 전체를 도는 것이 아니라 dataloader 전체를 1회 순회)
    # n_epochs < 1.0인 경우를 처리하기 위해 배치 수 제한
    stage1_dataloader_limited = _limit_dataloader(stage1_dataloader, stage1_batches_to_run)

    stage1_metrics = train_stage1(
        adapter=adapter,
        dataloader=stage1_dataloader_limited,
        optimizer=stage1_optimizer,
        config=stage1_config,
        device=device,
    )

    all_metrics["stage1"] = stage1_metrics

    logger.info(f"Stage 1 완료 - Metrics: {stage1_metrics}")

    # Checkpoint 저장 (Stage 1 완료 후)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(
            adapter=adapter,
            optimizer=stage1_optimizer,
            stage="stage1",
            epoch=stage1_epochs,
            metrics=stage1_metrics,
            save_dir=save_dir,
        )

    # ========== Stage 2: Weighted Training ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Stage 2: Weighted Training ({stage2_epochs} epochs)")
    logger.info("=" * 80)

    # Stage 2 전용 optimizer (전체 파라미터)
    stage2_optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=stage2_lr,
    )

    # Epoch loop
    stage2_total_batches = len(stage2_dataloader)
    full_epochs = int(stage2_epochs)
    remaining_fraction = stage2_epochs - full_epochs

    logger.info(
        f"Stage 2 - Total batches: {stage2_total_batches}, "
        f"Full epochs: {full_epochs}, Remaining fraction: {remaining_fraction:.2f}"
    )

    stage2_metrics_history = []

    for epoch in range(full_epochs):
        logger.info(f"\nStage 2 - Epoch {epoch + 1}/{full_epochs}")

        stage2_metrics = train_stage2(
            adapter=adapter,
            dataloader=stage2_dataloader,
            optimizer=stage2_optimizer,
            config=stage2_config,
            device=device,
        )

        stage2_metrics_history.append(stage2_metrics)
        logger.info(f"Epoch {epoch + 1} 완료 - Metrics: {stage2_metrics}")

        # Checkpoint 저장 (epoch마다)
        if save_dir is not None and save_checkpoint_every is not None:
            if (epoch + 1) % save_checkpoint_every == 0:
                _save_checkpoint(
                    adapter=adapter,
                    optimizer=stage2_optimizer,
                    stage="stage2",
                    epoch=epoch + 1,
                    metrics=stage2_metrics,
                    save_dir=save_dir,
                )

    # Remaining fraction 처리 (예: 2.5 epoch의 0.5 부분)
    if remaining_fraction > 0:
        logger.info(f"\nStage 2 - Remaining fraction: {remaining_fraction:.2f} epoch")

        stage2_batches_remaining = int(stage2_total_batches * remaining_fraction)
        stage2_dataloader_limited = _limit_dataloader(stage2_dataloader, stage2_batches_remaining)

        stage2_metrics = train_stage2(
            adapter=adapter,
            dataloader=stage2_dataloader_limited,
            optimizer=stage2_optimizer,
            config=stage2_config,
            device=device,
        )

        stage2_metrics_history.append(stage2_metrics)
        logger.info(f"Remaining fraction 완료 - Metrics: {stage2_metrics}")

        # Final checkpoint 저장
        if save_dir is not None:
            _save_checkpoint(
                adapter=adapter,
                optimizer=stage2_optimizer,
                stage="stage2",
                epoch=stage2_epochs,
                metrics=stage2_metrics,
                save_dir=save_dir,
            )

    # Stage 2 평균 metrics 계산
    stage2_avg_metrics = _average_metrics(stage2_metrics_history)
    all_metrics["stage2"] = stage2_avg_metrics

    logger.info("")
    logger.info("=" * 80)
    logger.info("학습 파이프라인 완료")
    logger.info("=" * 80)
    logger.info(f"Stage 1 Final Metrics: {stage1_metrics}")
    logger.info(f"Stage 2 Avg Metrics: {stage2_avg_metrics}")

    return all_metrics


def _limit_dataloader(dataloader: DataLoader, max_batches: int) -> DataLoader:
    """DataLoader의 배치 수를 제한 (fractional epoch 처리)

    Args:
        dataloader: 원본 DataLoader
        max_batches: 최대 배치 수

    Returns:
        제한된 DataLoader (동일한 설정, 다른 dataset)
    """
    from torch.utils.data import Dataset  # noqa: PLC0415

    class LimitedDataset(Dataset):
        """배치 수 제한 Dataset wrapper"""

        def __init__(self, original_dataset, max_samples):
            self.original_dataset = original_dataset
            self.max_samples = min(max_samples, len(original_dataset))

        def __len__(self):
            return self.max_samples

        def __getitem__(self, idx):
            if idx >= self.max_samples:
                raise IndexError
            return self.original_dataset[idx]

    # 배치 크기 고려하여 샘플 수 계산
    batch_size = dataloader.batch_size
    max_samples = max_batches * batch_size

    limited_dataset = LimitedDataset(dataloader.dataset, max_samples)

    # 동일한 설정으로 새 DataLoader 생성
    return DataLoader(
        limited_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # 순서 유지 (curriculum learning 중요)
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers if hasattr(dataloader, "num_workers") else 0,
        pin_memory=dataloader.pin_memory if hasattr(dataloader, "pin_memory") else False,
    )


def _save_checkpoint(
    adapter: MetaLlamaMTPAdapter,
    optimizer: torch.optim.Optimizer,
    stage: str,
    epoch: float,
    metrics: dict[str, float],
    save_dir: Path,
) -> None:
    """Checkpoint 저장

    Args:
        adapter: MetaLlamaMTPAdapter
        optimizer: Optimizer
        stage: "stage1" or "stage2"
        epoch: Epoch 수 (fractional 가능)
        metrics: 학습 metrics
        save_dir: 저장 디렉터리
    """
    checkpoint_path = save_dir / f"checkpoint_{stage}_epoch_{epoch:.1f}.pt"

    checkpoint = {
        "stage": stage,
        "epoch": epoch,
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint 저장: {checkpoint_path}")


def _average_metrics(metrics_history: list[dict[str, float]]) -> dict[str, float]:
    """Metrics 평균 계산

    Args:
        metrics_history: Metrics 리스트 (epoch별)

    Returns:
        평균 metrics
    """
    if not metrics_history:
        return {}

    avg_metrics = {}
    for key in metrics_history[0].keys():
        avg_metrics[key] = sum(m[key] for m in metrics_history) / len(metrics_history)

    return avg_metrics
