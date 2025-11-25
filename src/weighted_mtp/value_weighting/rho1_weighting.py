"""Rho-1 Weighting: Reference model 기반 Excess loss weighting

Rho-1 원리:
- Reference model과 Policy model의 CE 차이를 계산 (Signed difference)
- 큰 차이를 보이는 토큰 = 학습 필요한 토큰
- Top-k binary selection (논문 방식)

MTP 확장:
- Head 0 (t+1): 무조건 학습 (NTP baseline과 동일)
- Head 1,2,3 (t+2~t+4): Batch-wise top-k selection

Reference:
- Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024)
- https://arxiv.org/abs/2404.07965
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_mtp_selective_weights(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    k_percent: float = 0.6,
    mode: str = "signed",
) -> tuple[torch.Tensor, dict]:
    """MTP-specific Rho-1 weighting: t+1 always, t+2~4 selective

    Reference inference는 이미 완료되어 ref_logits로 전달됨.
    각 head에 대해 ref_logits의 다른 slice만 사용.

    Args:
        policy_logits: [batch, seq, n_future=4, vocab] - Policy MTP logits
        ref_logits: [batch, seq, vocab] - Reference NTP logits (이미 계산된 것)
        labels: [batch, seq] - Target tokens
        attention_mask: [batch, seq] - Attention mask
        k_percent: Top k% selection ratio (0~1, default 0.6)
        mode: "signed" (Policy > Ref만 선택) / "absolute" (차이 큰 모든 토큰 선택) / "difficult" (Ref가 어려워하는 토큰 선택)

    Returns:
        weights: [batch, seq, n_future] - Binary selection mask (0 or 1)
        stats: dict - Selection statistics
    """
    batch_size, seq_len, n_future, vocab_size = policy_logits.shape
    device = policy_logits.device

    # Initialize: all zeros
    weights = torch.zeros(batch_size, seq_len, n_future, device=device)

    # Statistics
    stats = {}

    # HEAD 0,1,2,3 (t+1~t+4): 모두 Rho-1 selection
    for k in range(1, n_future + 1):  # k = 1, 2, 3, 4
        head_idx = k - 1  # 0, 1, 2, 3
        valid_len = seq_len - k

        if valid_len <= 0:
            stats[f'head_{head_idx}_count'] = 0
            stats[f'head_{head_idx}_ratio'] = 0.0
            continue

        # Timestep alignment
        policy_logits_k = policy_logits[:, :valid_len, head_idx, :]  # [batch, valid_len, vocab]
        ref_logits_k = ref_logits[:, k-1:k-1+valid_len, :]            # [batch, valid_len, vocab]
        labels_k = labels[:, k:k+valid_len]                            # [batch, valid_len]
        mask_k = attention_mask[:, k:k+valid_len]                      # [batch, valid_len]

        # Per-token CE loss (float32로 계산하여 정밀도 보장)
        # labels=-100인 토큰은 제외 (ignore_index)
        policy_ce = F.cross_entropy(
            policy_logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction='none',
            ignore_index=-100,
        ).view(batch_size, valid_len).float()

        ref_ce = F.cross_entropy(
            ref_logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction='none',
            ignore_index=-100,
        ).view(batch_size, valid_len).float()

        # Excess loss 계산 (mode에 따라)
        if mode == "absolute":
            excess_loss = torch.abs(policy_ce - ref_ce)
        elif mode == "signed":
            excess_loss = policy_ce - ref_ce
        elif mode == "difficult":
            excess_loss = ref_ce
        else:
            raise ValueError(f"Invalid rho1_mode: {mode}. Must be 'signed', 'absolute', or 'difficult'")
        excess_loss = excess_loss.float()  # [batch, valid_len], float32

        # Batch-wise top-k selection
        # labels=-100인 토큰도 제외 (instruction tokens)
        valid_label_mask = (labels_k != -100)
        valid_mask = mask_k.bool() & valid_label_mask
        valid_excess = excess_loss[valid_mask]

        if valid_excess.numel() == 0:
            stats[f'head_{head_idx}_count'] = 0
            stats[f'head_{head_idx}_ratio'] = 0.0
            continue

        # Top k% threshold (float32 텐서 사용)
        threshold = torch.quantile(valid_excess, 1 - k_percent)

        # Binary selection: 1 if >= threshold, 0 otherwise
        # valid_mask를 사용하여 labels=-100 토큰도 제외
        selected = (excess_loss >= threshold).float() * valid_mask.float()
        weights[:, :valid_len, head_idx] = selected

        # Statistics
        stats[f'head_{head_idx}_count'] = selected.sum().item()
        total_valid = valid_mask.sum().item()
        stats[f'head_{head_idx}_ce_mean'] = policy_ce[valid_mask].mean().item()
        stats[f'head_{head_idx}_excess_mean'] = valid_excess.mean().item()
        stats[f'head_{head_idx}_threshold'] = threshold.item()

    # Overall statistics
    total_possible = attention_mask.sum().item() * n_future
    total_selected = weights.sum().item()
    stats['selection_ratio'] = total_selected / (total_possible + 1e-8)
    stats['avg_heads_per_position'] = total_selected / (attention_mask.sum().item() + 1e-8)

    return weights, stats


def compute_rho1_stats(
    weights: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Rho-1 weighting 통계 계산 (Per-head binary weights)

    Args:
        weights: [batch, seq, n_future] - Binary weights (0 or 1)
        attention_mask: [batch, seq] - Attention mask (optional)

    Returns:
        stats: {
            "selection_ratio": float - 전체 선택 비율,
            "avg_heads_per_position": float - Position당 평균 head 수,
            "head_0_ratio": float,
            "head_1_ratio": float,
            "head_2_ratio": float,
            "head_3_ratio": float,
        }
    """
    batch_size, seq_len, n_future = weights.shape

    if attention_mask is not None:
        # Valid positions만 계산
        total_valid = attention_mask.sum().item()
    else:
        total_valid = batch_size * seq_len

    # Overall statistics
    total_selected = weights.sum().item()
    total_possible = total_valid * n_future

    stats = {
        "selection_ratio": total_selected / (total_possible + 1e-8),
        "avg_heads_per_position": total_selected / (total_valid + 1e-8),
    }

    # Per-head statistics
    for head_idx in range(n_future):
        head_weights = weights[:, :, head_idx]
        if attention_mask is not None:
            head_selected = (head_weights * attention_mask.float()).sum().item()
        else:
            head_selected = head_weights.sum().item()

        stats[f'head_{head_idx}_ratio'] = head_selected / (total_valid + 1e-8)
        stats[f'head_{head_idx}_count'] = head_selected

    return stats
