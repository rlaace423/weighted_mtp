"""Rho-1 Weighting: Reference model 기반 Excess loss weighting

Rho-1 원리:
- Reference model과 Policy model의 CE 차이를 계산
- 큰 차이를 보이는 토큰 = 어려운/중요한 토큰
- Softmax로 연속적 가중치 부여

Reference:
- Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024)
- https://openreview.net/forum?id=0NMzBwqaAJ
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_excess_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Policy vs Reference CE 차이 계산 (MTP 시점 정렬)

    MTP 구조:
    - Policy: 시점 t, head k → x_{t+k} 예측
    - Reference (NTP): 시점 t+k-1 → x_{t+k} 예측

    각 k에 대해 정렬된 위치의 logits를 비교하여 excess loss 계산

    Args:
        policy_logits: [batch, seq, n_future, vocab] - Policy MTP logits
        ref_logits: [batch, seq, vocab] - Reference NTP logits
        labels: [batch, seq] - Target tokens
        attention_mask: [batch, seq] - Attention mask

    Returns:
        excess_loss: [batch, seq] - H개 토큰의 평균 excess loss
    """
    batch_size, seq_len, n_future, vocab_size = policy_logits.shape

    # 각 k에 대해 excess loss 계산 후 평균
    total_excess_loss = torch.zeros(batch_size, seq_len, device=policy_logits.device)
    total_mask = torch.zeros(batch_size, seq_len, device=policy_logits.device)

    for k in range(1, n_future + 1):
        # 유효 길이 (마지막 k개 시점은 labels 부족)
        valid_len = seq_len - k

        if valid_len <= 0:
            continue

        # Policy: 시점 t, head k-1 → x_{t+k} 예측
        policy_logits_k = policy_logits[:, :valid_len, k - 1, :]  # [batch, valid_len, vocab]

        # Reference: 시점 t+k-1 → x_{t+k} 예측
        ref_logits_k = ref_logits[:, k - 1 : k - 1 + valid_len, :]  # [batch, valid_len, vocab]

        # Target: x_{t+k}
        labels_k = labels[:, k : k + valid_len]  # [batch, valid_len]

        # Mask
        mask_k = attention_mask[:, k : k + valid_len]  # [batch, valid_len]

        # CE loss 계산 (per token)
        policy_ce_k = F.cross_entropy(
            policy_logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction="none",
        )  # [batch * valid_len]
        policy_ce_k = policy_ce_k.view(batch_size, valid_len)

        ref_ce_k = F.cross_entropy(
            ref_logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction="none",
        )  # [batch * valid_len]
        ref_ce_k = ref_ce_k.view(batch_size, valid_len)

        # Excess loss: |policy_ce - ref_ce|
        excess_loss_k = torch.abs(policy_ce_k - ref_ce_k)  # [batch, valid_len]

        # Mask 적용 및 누적
        masked_excess_k = excess_loss_k * mask_k.float()

        # 해당 위치에 누적 (시점 0~valid_len-1)
        total_excess_loss[:, :valid_len] += masked_excess_k
        total_mask[:, :valid_len] += mask_k.float()

    # 평균 계산 (H개 토큰 평균)
    # total_mask > 0인 위치만 나누기
    avg_excess_loss = torch.where(
        total_mask > 0,
        total_excess_loss / total_mask,
        torch.zeros_like(total_excess_loss),
    )

    return avg_excess_loss


def build_weights(
    excess_loss: torch.Tensor,
    temperature: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rho-1 Softmax weighting

    Excess loss가 큰 토큰에 더 높은 가중치 부여

    Args:
        excess_loss: [batch, seq] - Excess loss per token
        temperature: Softmax temperature (기본 1.0)
        attention_mask: [batch, seq] - Attention mask (optional)

    Returns:
        weights: [batch, seq] - Softmax normalized weights
    """
    batch_size, seq_len = excess_loss.shape

    # Temperature scaling
    scaled_excess = excess_loss / temperature

    # Mask 처리 (padding 토큰은 -inf로 설정)
    if attention_mask is not None:
        scaled_excess = scaled_excess.masked_fill(~attention_mask.bool(), float("-inf"))

    # Softmax (sequence 차원)
    weights = F.softmax(scaled_excess, dim=-1)  # [batch, seq]

    # Mask 재적용 (safety)
    if attention_mask is not None:
        weights = weights * attention_mask.float()

    return weights


def compute_rho1_stats(
    excess_loss: torch.Tensor,
    weights: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Rho-1 weighting 통계 계산

    Args:
        excess_loss: [batch, seq] - Excess loss
        weights: [batch, seq] - Rho-1 weights
        attention_mask: [batch, seq] - Attention mask (optional)

    Returns:
        stats: {
            "excess_loss_mean": float,
            "excess_loss_std": float,
            "weight_mean": float,
            "weight_std": float,
            "weight_entropy": float,
        }
    """
    if attention_mask is not None:
        # Mask 적용
        mask = attention_mask.bool()
        excess_valid = excess_loss[mask]
        weights_valid = weights[mask]
    else:
        excess_valid = excess_loss.flatten()
        weights_valid = weights.flatten()

    # Excess loss 통계
    excess_mean = excess_valid.mean().item()
    excess_std = excess_valid.std().item()

    # Weight 통계
    weight_mean = weights_valid.mean().item()
    weight_std = weights_valid.std().item()

    # Weight entropy (정규화된 분포의 엔트로피)
    # H = -Σ p_i log(p_i)
    eps = 1e-8
    weight_entropy = -(weights_valid * torch.log(weights_valid + eps)).sum().item()

    return {
        "excess_loss_mean": excess_mean,
        "excess_loss_std": excess_std,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "weight_entropy": weight_entropy,
    }
