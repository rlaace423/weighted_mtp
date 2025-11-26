"""LoRA Linear 레이어 구현

Low-Rank Adaptation을 적용한 Linear 레이어
원본 가중치는 frozen 상태로 유지하고, 저랭크 행렬만 학습

수식: h = xW^T + xA^T B^T * (alpha / rank)
- W: 원본 가중치 (frozen)
- A: [rank, in_features] 행렬 (학습)
- B: [out_features, rank] 행렬 (학습)
"""

from typing import Optional

import torch
from torch import nn


class LoRALinear(nn.Module):
    """LoRA가 적용된 Linear 레이어

    원본 nn.Linear를 감싸서 low-rank adaptation 수행
    FSDP와 호환되도록 설계 (TransformerBlock 내부에 위치)

    Args:
        in_features: 입력 차원
        out_features: 출력 차원
        rank: LoRA rank (저랭크 행렬 차원)
        alpha: LoRA scaling factor
        dropout: LoRA dropout 확률
        bias: 원본 Linear의 bias 존재 여부
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 원본 Linear (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 저랭크 행렬 (학습 대상)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # LoRA dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 가중치 초기화
        self._init_lora_weights()

    def _init_lora_weights(self):
        """LoRA 가중치 초기화

        A: Kaiming uniform (학습 시작점)
        B: zeros (초기 출력은 원본과 동일)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: [*, in_features] 입력 텐서

        Returns:
            [*, out_features] 출력 텐서
        """
        # 원본 Linear 출력
        base_output = self.linear(x)

        # LoRA 출력: x @ A^T @ B^T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> None:
        """LoRA 가중치를 원본에 병합 (inference 최적화)

        병합 후에는 추가 연산 없이 원본 Linear만 사용
        학습 재개가 필요하면 unmerge_weights() 호출
        """
        with torch.no_grad():
            # W' = W + B @ A * scaling
            merged_weight = self.lora_B @ self.lora_A * self.scaling
            self.linear.weight.add_(merged_weight)

            # LoRA 가중치 초기화 (병합 완료 표시)
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def unmerge_weights(self) -> None:
        """병합된 가중치 분리 (학습 재개용)

        주의: merge 전 LoRA 가중치가 보존되어 있어야 함
        현재 구현에서는 merge 시 LoRA를 0으로 초기화하므로
        unmerge는 merge 직전 상태로 복원 불가
        """
        raise NotImplementedError(
            "unmerge_weights는 현재 지원되지 않습니다. "
            "merge 전에 checkpoint를 저장하세요."
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """기존 nn.Linear에서 LoRALinear 생성

        Args:
            linear: 원본 nn.Linear
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: LoRA dropout 확률

        Returns:
            LoRALinear 인스턴스 (원본 가중치 복사됨)
        """
        has_bias = linear.bias is not None
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )

        # 원본 가중치 복사
        lora_linear.linear.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora_linear.linear.bias.data.copy_(linear.bias.data)

        # 원본과 동일 device/dtype 설정
        lora_linear = lora_linear.to(linear.weight.device)
        lora_linear = lora_linear.to(linear.weight.dtype)

        return lora_linear


def apply_lora_to_linear(
    module: nn.Module,
    target_names: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """모듈 내 특정 Linear 레이어에 LoRA 적용

    target_names에 해당하는 nn.Linear를 LoRALinear로 교체

    Args:
        module: 대상 모듈 (예: TransformerBlock)
        target_names: LoRA 적용할 Linear 이름 (예: ["wq", "wk", "wv", "wo"])
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout 확률

    Returns:
        LoRA가 적용된 모듈 (in-place 수정)

    Example:
        >>> block = TransformerBlock(layer_id=0, args=model_args)
        >>> apply_lora_to_linear(block.attention, ["wq", "wk", "wv", "wo"], rank=8)
    """
    for name in target_names:
        if not hasattr(module, name):
            continue

        original_linear = getattr(module, name)
        if not isinstance(original_linear, nn.Linear):
            continue

        lora_linear = LoRALinear.from_linear(
            linear=original_linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(module, name, lora_linear)

    return module


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """모델에서 LoRA 파라미터만 추출

    requires_grad=True인 LoRA 파라미터 (lora_A, lora_B)만 반환
    Optimizer에 전달하여 LoRA만 학습

    Args:
        model: LoRA가 적용된 모델

    Returns:
        LoRA 파라미터 리스트

    Example:
        >>> optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-4)
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            if param.requires_grad:
                lora_params.append(param)
    return lora_params


def merge_lora_weights(model: nn.Module) -> None:
    """모델 내 모든 LoRALinear의 가중치 병합

    Inference 최적화를 위해 LoRA를 원본에 병합
    병합 후에는 일반 Linear처럼 동작

    Args:
        model: LoRA가 적용된 모델
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()

