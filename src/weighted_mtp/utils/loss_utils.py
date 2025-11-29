"""MTP Loss 계산 유틸리티

Multi-Token Prediction Cross-Entropy Loss 계산 (메모리 최적화)
run_verifiable, run_baseline, run_rho1에서 공통 사용
"""

import torch
import torch.nn.functional as F


def compute_head_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    head_idx: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """단일 MTP head에 대한 Cross-Entropy Loss 계산

    Sequential Unembedding에서 head별로 loss 계산 시 사용.
    head_idx에 따라 labels를 shift하여 올바른 타겟과 매칭.

    Args:
        logits: [batch, seq, vocab] 단일 head의 logits
        labels: [batch, seq] 원본 타겟 토큰 ID (-100: ignore)
        attention_mask: [batch, seq] 유효 토큰 마스크 (1: 유효, 0: padding)
        head_idx: MTP head 인덱스 (1-based). head 1은 t+1 예측, head 2는 t+2 예측
        weights: 토큰별 가중치 (None이면 균등 가중치)
            - [batch, seq]: Position-level (Verifiable TD weighting)
            - [batch, seq, n_future]: Per-head (Rho1 selective weighting)

    Returns:
        loss: scalar tensor (해당 head의 평균 CE loss)

    Note:
        - logits[:, t, :]는 position t에서의 예측
        - head_idx=k일 때, 타겟은 labels[:, k:k+valid_len]
        - valid_len = seq_len - head_idx (마지막 k개 position은 타겟 없음)
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    dtype = logits.dtype

    valid_len = seq_len - head_idx
    if valid_len <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # head_idx에 따른 slicing
    logits_slice = logits[:, :valid_len, :]                    # [batch, valid_len, vocab]
    labels_slice = labels[:, head_idx : head_idx + valid_len]  # [batch, valid_len]
    mask_slice = attention_mask[:, head_idx : head_idx + valid_len]

    # Cross-entropy loss (토큰별)
    ce_loss = F.cross_entropy(
        logits_slice.reshape(-1, vocab_size),
        labels_slice.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )  # [batch * valid_len]

    # 유효 토큰 마스크 (labels != -100 AND attention_mask == 1)
    valid_label_mask = (labels_slice != -100).to(dtype)
    combined_mask = mask_slice.to(dtype) * valid_label_mask
    combined_mask_flat = combined_mask.reshape(-1)

    mask_sum = combined_mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Weighted loss
    if weights is not None:
        # weights shape 자동 감지
        if weights.dim() == 3:
            # Per-head weights [batch, seq, n_future]: Rho1
            # head_idx는 1-based, weights의 마지막 차원은 0-indexed
            weights_slice = weights[:, :valid_len, head_idx - 1]
        else:
            # Position-level weights [batch, seq]: Verifiable
            weights_slice = weights[:, head_idx : head_idx + valid_len]
        weights_flat = weights_slice.reshape(-1)
        loss = (ce_loss * weights_flat * combined_mask_flat).sum() / mask_sum
    else:
        loss = (ce_loss * combined_mask_flat).sum() / mask_sum

    return loss


def compute_mtp_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_future: int | None = None,
) -> dict[str, torch.Tensor]:
    """MTP Cross-Entropy Loss 계산 (메모리 최적화)

    각 head별로 계산 후 즉시 중간 텐서를 삭제하여 메모리 절감.
    기존 for loop 대비 ~20-30% 메모리 절감 효과.

    Args:
        logits: [batch, seq, n_future, vocab] MTP logits
        labels: [batch, seq] 타겟 토큰 ID (-100: ignore)
        attention_mask: [batch, seq] 유효 토큰 마스크 (1: 유효, 0: padding)
        weights: [batch, seq] 토큰별 가중치 (None이면 균등 가중치 1.0)
        n_future: MTP head 수 (None이면 logits에서 추론)

    Returns:
        {
            "weighted_ce_loss": scalar tensor (weighted average across heads),
            "unweighted_ce_loss": scalar tensor (unweighted average across heads),
        }

    Note:
        - labels의 k번째 head에 대한 타겟은 labels[:, k:k+valid_len]
        - ignore_index=-100인 토큰은 loss 계산에서 제외
        - 메모리 최적화를 위해 각 head 계산 후 중간 텐서 즉시 삭제
    """
    batch_size, seq_len, n_future_dim, vocab_size = logits.shape
    n_future = n_future or n_future_dim
    device = logits.device
    dtype = logits.dtype

    # 균등 가중치 (weights=None인 경우)
    if weights is None:
        weights = torch.ones(batch_size, seq_len, device=device, dtype=dtype)

    # Loss 누적 변수 (스칼라로 유지하여 backward graph 최소화)
    weighted_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    unweighted_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    valid_head_count = 0

    for k in range(1, n_future + 1):
        valid_len = seq_len - k
        if valid_len <= 0:
            continue

        # 현재 head의 logits, labels, weights, mask 추출
        logits_k = logits[:, :valid_len, k - 1, :]  # [batch, valid_len, vocab]
        labels_k = labels[:, k : k + valid_len]      # [batch, valid_len]
        weights_k = weights[:, k : k + valid_len]    # [batch, valid_len]
        mask_k = attention_mask[:, k : k + valid_len]  # [batch, valid_len]

        # Cross-entropy loss (reduction="none"으로 토큰별 loss)
        ce_loss_k = F.cross_entropy(
            logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )  # [batch * valid_len]

        # 유효 토큰 마스크 (labels != -100)
        valid_label_mask_k = (labels_k != -100).to(dtype)
        combined_mask_k = mask_k.to(dtype) * valid_label_mask_k  # [batch, valid_len]
        combined_mask_flat = combined_mask_k.reshape(-1)  # [batch * valid_len]

        # 마스크 합계 (나누기용)
        mask_sum_k = combined_mask_k.sum()

        if mask_sum_k > 0:
            # Weighted loss 계산 후 즉시 스칼라로 축소
            weights_flat = weights_k.reshape(-1)
            weighted_sum = (ce_loss_k * weights_flat * combined_mask_flat).sum()
            unweighted_sum = (ce_loss_k * combined_mask_flat).sum()

            weighted_loss_sum = weighted_loss_sum + weighted_sum / mask_sum_k
            unweighted_loss_sum = unweighted_loss_sum + unweighted_sum / mask_sum_k
            valid_head_count += 1

        # 메모리 최적화: 중간 텐서 즉시 삭제
        del logits_k, labels_k, weights_k, mask_k
        del ce_loss_k, valid_label_mask_k, combined_mask_k, combined_mask_flat

    # Head 평균
    if valid_head_count > 0:
        weighted_ce_loss = weighted_loss_sum / valid_head_count
        unweighted_ce_loss = unweighted_loss_sum / valid_head_count
    else:
        weighted_ce_loss = torch.tensor(0.0, device=device, dtype=dtype)
        unweighted_ce_loss = torch.tensor(0.0, device=device, dtype=dtype)

    return {
        "weighted_ce_loss": weighted_ce_loss,
        "unweighted_ce_loss": unweighted_ce_loss,
    }


def compute_mtp_ce_loss_unweighted(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    n_future: int | None = None,
) -> torch.Tensor:
    """MTP Cross-Entropy Loss 계산 (가중치 없음, baseline용)

    compute_mtp_ce_loss의 경량 버전. 가중치가 필요 없는 baseline 학습에서 사용.

    Args:
        logits: [batch, seq, n_future, vocab] MTP logits
        labels: [batch, seq] 타겟 토큰 ID (-100: ignore)
        attention_mask: [batch, seq] 유효 토큰 마스크
        n_future: MTP head 수 (None이면 logits에서 추론)

    Returns:
        ce_loss: scalar tensor (average across heads)
    """
    _, seq_len, n_future_dim, vocab_size = logits.shape
    n_future = n_future or n_future_dim
    device = logits.device
    dtype = logits.dtype

    loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
    valid_head_count = 0

    for k in range(1, n_future + 1):
        valid_len = seq_len - k
        if valid_len <= 0:
            continue

        logits_k = logits[:, :valid_len, k - 1, :]
        labels_k = labels[:, k : k + valid_len]
        mask_k = attention_mask[:, k : k + valid_len]

        ce_loss_k = F.cross_entropy(
            logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )

        valid_label_mask_k = (labels_k != -100).to(dtype)
        combined_mask_k = mask_k.to(dtype) * valid_label_mask_k
        combined_mask_flat = combined_mask_k.reshape(-1)
        mask_sum_k = combined_mask_k.sum()

        if mask_sum_k > 0:
            loss_sum = loss_sum + (ce_loss_k * combined_mask_flat).sum() / mask_sum_k
            valid_head_count += 1

        del logits_k, labels_k, mask_k, ce_loss_k
        del valid_label_mask_k, combined_mask_k, combined_mask_flat

    if valid_head_count > 0:
        return loss_sum / valid_head_count
    return torch.tensor(0.0, device=device, dtype=dtype)
