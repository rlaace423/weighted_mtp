"""TD Error 기반 Exponential Weighting

IQL/AWR 방식의 exponential weighting 적용
Kostrikov et al. "Offline RL with Implicit Q-Learning" (2021) 참고
"""

import torch


def build_weights(
    td_errors: torch.Tensor,
    beta: float = 0.9,
    min_weight: float = 0.1,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """TD error 기반 exponential weighting

    IQL/AWR 방식: weight = exp(advantage / β)
    WMTP 적용: weight = exp(td_error / β)

    직관:
    - Positive TD error (td > 0): weight > 1 → 중요 토큰 강화
    - Negative TD error (td < 0): weight < 1 → 비중요 토큰 down-weight
    - Incorrect 샘플: reward=0, value>0 → td<0 → weight<1 (자동 필터링)

    Args:
        td_errors: [batch, seq] TD error (compute_td_errors 출력)
        beta: Temperature parameter (낮을수록 집중도 높음, 기본 0.9)
        min_weight: 최소 가중치 (보수적 안정화, 기본 0.1)
        max_weight: 최대 가중치 (극단 방지, 기본 5.0)

    Returns:
        weights: [batch, seq] Token-level weights (clipped)

    Examples:
        >>> td_errors = torch.tensor([[0.2, -0.5, 0.1]])
        >>> weights = build_weights(td_errors, beta=0.9)
        >>> # exp(0.2 / 0.9) ≈ 1.25
        >>> # exp(-0.5 / 0.9) ≈ 0.57
        >>> # exp(0.1 / 0.9) ≈ 1.12
        >>> weights
        tensor([[1.25, 0.57, 1.12]])
    """
    # Exponential transformation: exp(td_error / beta)
    weights = torch.exp(td_errors / beta)

    # Conservative clipping: [min_weight, max_weight]
    # 극단적인 가중치를 방지하여 학습 안정성 확보
    weights = torch.clamp(weights, min=min_weight, max=max_weight)

    return weights
