"""Value Head 구현"""

from pathlib import Path

import torch
from torch import nn


class ValueHead(nn.Module):
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

        # RLHF 표준 초기화: 출력 layer zero init으로 초기 예측을 0에 가깝게
        self._init_weights()

    def _init_weights(self):
        """RLHF 표준 초기화: 마지막 layer zero init

        출력 layer를 0으로 초기화하여 초기 value 예측이 0에 가깝도록 함.
        이는 학습 초기 손실을 줄이고 안정적인 학습 시작을 보장.
        """
        # 마지막 Linear layer (출력층) zero init
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            hidden_states: [batch, seq, hidden_size] (Transformer norm 적용 후)

        Returns:
            value: [batch, seq, 1]
        """
        return self.mlp(hidden_states)

    def save_checkpoint(self, path: Path):
        """Value head checkpoint 저장

        Args:
            path: 저장 경로 (예: "value_head.pt")
        """
        torch.save({
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: torch.device) -> "ValueHead":
        """Value head checkpoint 로드

        Args:
            path: checkpoint 경로
            device: 로딩할 device

        Returns:
            ValueHead 인스턴스
        """
        ckpt = torch.load(path, map_location=device)
        value_head = cls(hidden_size=ckpt["hidden_size"])
        value_head.load_state_dict(ckpt["state_dict"])
        return value_head.to(device)
