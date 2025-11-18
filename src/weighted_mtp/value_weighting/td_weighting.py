"""TD Error 기반 Weighting (Verifiable WMTP용)

Temporal Difference Learning 기반 토큰별 가중치 계산
- TD error 계산: Bootstrapping value estimation
- Exponential weighting: IQL/AWR 방식 (Kostrikov et al. 2021)
- 통계 추적: 학습 모니터링 및 디버깅

References:
- Sutton & Barto "Reinforcement Learning: An Introduction"
- Kostrikov et al. "Offline RL with Implicit Q-Learning" (2021)
"""

import torch


def compute_td_errors(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """표준 TD error 계산 (Intermediate + Terminal)

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        rewards: [batch] Binary reward (0.0: incorrect, 1.0: correct)
        attention_mask: [batch, seq] 유효 토큰 마스크 (1: 유효, 0: padding)
        gamma: 할인율 (기본 1.0, LLM RLHF 표준은 할인 없음)

    Returns:
        td_errors: [batch, seq] 토큰별 TD error
            - Intermediate (k < T): γV(s_k) - V(s_{k-1})
            - Terminal (k = T): R - V(s_{T-1})
            - Padding: 0.0 (masking 적용)

    Examples:
        >>> value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        >>> rewards = torch.tensor([1.0])  # Correct
        >>> attention_mask = torch.tensor([[1, 1, 1]])  # All valid
        >>> td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma=1.0)
        >>> # Expected:
        >>> # Intermediate (0→1): 1.0 * 0.7 - 0.5 = 0.2
        >>> # Intermediate (1→2): 1.0 * 0.9 - 0.7 = 0.2
        >>> # Terminal (2): 1.0 - 0.9 = 0.1
        >>> td_errors
        tensor([[0.2, 0.2, 0.1]])
    """
    batch_size, seq_len, _ = value_logits.shape

    # Value logits squeeze: [batch, seq, 1] → [batch, seq]
    values = value_logits.squeeze(-1)

    # Terminal indices 계산: 각 시퀀스의 마지막 유효 토큰 위치
    # attention_mask.sum(dim=1): [batch] 각 시퀀스의 유효 토큰 수
    # -1: 0-indexed로 변환
    terminal_indices = attention_mask.sum(dim=1).long() - 1

    # TD errors 초기화 (전체를 Intermediate로 계산)
    td_errors = torch.zeros_like(values)

    # Intermediate TD errors (k < T): γV(s_k) - V(s_{k-1})
    # values[:, 1:]: V(s_k) - 다음 상태 value
    # values[:, :-1]: V(s_{k-1}) - 현재 상태 value
    if seq_len > 1:
        td_errors[:, :-1] = gamma * values[:, 1:] - values[:, :-1]

    # Terminal TD error (k = T): R - V(s_{T-1})
    # Advanced indexing으로 vectorized 연산
    batch_indices = torch.arange(batch_size, device=values.device)
    values_terminal = values[batch_indices, terminal_indices]
    td_terminal = rewards - values_terminal

    # Terminal 위치에 TD terminal 값 할당
    td_errors[batch_indices, terminal_indices] = td_terminal

    # Padding 토큰 masking: attention_mask == 0인 위치는 td_error = 0
    td_errors = td_errors * attention_mask.float()

    return td_errors


def build_weights(
    td_errors: torch.Tensor,
    beta: float = 0.9,
    min_weight: float = 0.1,
    max_weight: float = 3.0,
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
        max_weight: 최대 가중치 (극단 방지, 기본 3.0)

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


def compute_td_stats(td_errors: torch.Tensor) -> dict[str, float]:
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


def compute_weight_stats(weights: torch.Tensor) -> dict[str, float]:
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
