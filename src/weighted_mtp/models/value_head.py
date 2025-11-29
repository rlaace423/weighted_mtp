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
    """2-layer MLP value head

    2.7B Value Model의 표현력 보완을 위한 넓은 bottleneck 구조
    구조: hidden_size → hidden_size//4 → hidden_size//8 → 1
    (2560 → 640 → 320 → 1)
    Dropout으로 과적합 방지

    Args:
        hidden_size: Transformer hidden dimension
        bias: Linear layer bias (기본값 False, RLHF 표준)
        dropout: Dropout 확률 (기본값 0.0, Pairwise 학습 시 0.3 권장)
    """

    def __init__(self, hidden_size: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_type = "mlp"

        # 넓은 bottleneck MLP (2560 → 640 → 320 → 1)
        hidden1 = hidden_size // 4   # 640 for 2560 dim
        hidden2 = hidden_size // 8   # 320 for 2560 dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden1, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
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


def create_value_head(
    hidden_size: int,
    head_type: str = "mlp",
    dropout: float = 0.0,
) -> ValueHeadType:
    """Value head factory function

    Args:
        hidden_size: Transformer hidden dimension
        head_type: "linear", "sigmoid", 또는 "mlp"
        dropout: Dropout 확률 (mlp 타입에만 적용)

    Returns:
        ValueHead 인스턴스
    """
    if head_type == "linear":
        return LinearValueHead(hidden_size)
    elif head_type == "sigmoid":
        return SigmoidValueHead(hidden_size)
    elif head_type == "mlp":
        return MLPValueHead(hidden_size, dropout=dropout)
    else:
        raise ValueError(f"Unknown value head type: {head_type}. Use 'linear', 'sigmoid', or 'mlp'.")


# 하위 호환성을 위한 alias
ValueHead = MLPValueHead
