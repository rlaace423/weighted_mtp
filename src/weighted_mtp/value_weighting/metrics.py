"""Value Weighting Metrics

TD error 및 Weight 분포 통계 계산
학습 모니터링 및 디버깅 지원
"""

from typing import Dict

import torch


def compute_weight_stats(weights: torch.Tensor) -> Dict[str, float]:
    """Weight 분포 통계 계산

    Args:
        weights: [batch, seq] Token weights

    Returns:
        {
            "weight_mean": float,
            "weight_std": float,
            "weight_min": float,
            "weight_max": float,
            "weight_entropy": float,  # Normalized entropy
        }

    Examples:
        >>> weights = torch.tensor([[1.2, 0.8, 1.0], [1.5, 0.5, 1.0]])
        >>> stats = compute_weight_stats(weights)
        >>> stats["weight_mean"]  # 평균
        1.0
        >>> stats["weight_entropy"]  # 엔트로피 (높을수록 균등 분포)
        0.95
    """
    # Flatten weights for global statistics
    weights_flat = weights.flatten()

    # Basic statistics
    weight_mean = weights_flat.mean().item()
    weight_std = weights_flat.std().item()
    weight_min = weights_flat.min().item()
    weight_max = weights_flat.max().item()

    # Entropy calculation
    # Normalize weights to probability distribution
    weights_normalized = weights_flat / (weights_flat.sum() + 1e-8)

    # Shannon entropy: -sum(p * log(p))
    # NaN 방지: log(0) → 0으로 처리
    entropy_terms = weights_normalized * torch.log(weights_normalized + 1e-10)
    entropy = -entropy_terms.sum().item()

    # Normalize entropy to [0, 1]
    # Maximum entropy = log(N), where N is number of elements
    max_entropy = torch.log(torch.tensor(len(weights_flat), dtype=torch.float32)).item()
    normalized_entropy = entropy / (max_entropy + 1e-8)

    return {
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "weight_min": weight_min,
        "weight_max": weight_max,
        "weight_entropy": normalized_entropy,
    }


def compute_td_stats(td_errors: torch.Tensor) -> Dict[str, float]:
    """TD error 분포 통계 계산

    Args:
        td_errors: [batch, seq] TD errors

    Returns:
        {
            "td_mean": float,
            "td_std": float,
            "td_min": float,
            "td_max": float,
        }

    Examples:
        >>> td_errors = torch.tensor([[0.2, -0.5, 0.1], [0.3, -0.3, 0.0]])
        >>> stats = compute_td_stats(td_errors)
        >>> stats["td_mean"]  # 평균
        -0.033
        >>> stats["td_std"]  # 표준편차
        0.28
    """
    # Flatten td_errors for global statistics
    td_flat = td_errors.flatten()

    return {
        "td_mean": td_flat.mean().item(),
        "td_std": td_flat.std().item(),
        "td_min": td_flat.min().item(),
        "td_max": td_flat.max().item(),
    }
