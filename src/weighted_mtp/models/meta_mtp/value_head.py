"""Value Head 구현"""

from typing import Union

import torch
from torch import nn


class LinearValueHead(nn.Module):
    """단일 Linear layer value head (MSE loss용)

    구조: hidden_size → 1

    Args:
        hidden_size: Transformer hidden dimension
        bias: Linear layer bias (기본값 False, RLHF 표준)
    """

    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_type = "linear"

        self.linear = nn.Linear(hidden_size, 1, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """RLHF 표준 초기화: zero init"""
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            value: [batch, seq, 1]
        """
        return self.linear(hidden_states)


class SigmoidValueHead(nn.Module):
    """Linear + Sigmoid value head (BCE loss용)

    구조: hidden_size → 1 → Sigmoid
    출력이 [0, 1] 확률값이므로 BCE loss와 함께 사용
    MC (gamma=1, lam=1)일 때만 사용 가능

    Args:
        hidden_size: Transformer hidden dimension
        bias: Linear layer bias (기본값 False, RLHF 표준)
    """

    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_type = "sigmoid"

        self.linear = nn.Linear(hidden_size, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """RLHF 표준 초기화: zero init"""
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            value: [batch, seq, 1] (0~1 확률값)
        """
        logits = self.linear(hidden_states)
        return self.sigmoid(logits)


class MLPValueHead(nn.Module):
    """2-layer MLP value head (DIAL style)

    DIAL 논문 기반: bottleneck MLP로 표현력 증가 + 과적합 방지
    구조: hidden_size → hidden_size//8 → hidden_size//16 → 1

    Args:
        hidden_size: Transformer hidden dimension
        bias: Linear layer bias (기본값 False, RLHF 표준)
    """

    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_type = "mlp"

        # DIAL 스타일 2-layer MLP (4096 → 512 → 256 → 1)
        hidden1 = hidden_size // 8   # 512 for 4096 dim
        hidden2 = hidden_size // 16  # 256 for 4096 dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden1, bias=bias),
            nn.GELU(),
            nn.Linear(hidden1, hidden2, bias=bias),
            nn.GELU(),
            nn.Linear(hidden2, 1, bias=bias),
        )

        self._init_weights()

    def _init_weights(self):
        """RLHF 표준 초기화: 마지막 layer zero init"""
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            value: [batch, seq, 1]
        """
        return self.mlp(hidden_states)


# Type alias
ValueHeadType = Union[LinearValueHead, SigmoidValueHead, MLPValueHead]


def create_value_head(hidden_size: int, head_type: str = "mlp") -> ValueHeadType:
    """Value head factory function

    Args:
        hidden_size: Transformer hidden dimension
        head_type: "linear", "sigmoid", 또는 "mlp"

    Returns:
        ValueHead 인스턴스
    """
    if head_type == "linear":
        return LinearValueHead(hidden_size)
    elif head_type == "sigmoid":
        return SigmoidValueHead(hidden_size)
    elif head_type == "mlp":
        return MLPValueHead(hidden_size)
    else:
        raise ValueError(f"Unknown value head type: {head_type}. Use 'linear', 'sigmoid', or 'mlp'.")


# 하위 호환성을 위한 alias
ValueHead = MLPValueHead
