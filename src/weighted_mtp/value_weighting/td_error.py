"""표준 TD Error 계산

Temporal Difference Learning 기반 토큰별 가치 추정
Sutton & Barto "Reinforcement Learning: An Introduction" 참고
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
