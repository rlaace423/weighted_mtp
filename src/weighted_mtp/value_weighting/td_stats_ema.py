"""TD Error EMA 통계 추적기

학습 전체에서 TD Error의 mean/std를 EMA로 추적하여
일관된 Advantage Whitening 정규화 제공.

BatchNorm의 running_mean/running_var와 동일한 원리:
- 각 batch의 통계를 EMA로 누적
- 모든 batch에서 누적된 EMA 통계로 정규화
- 분산 학습 시 all-reduce로 GPU 간 동기화
"""

import torch

from weighted_mtp.runtime import all_reduce_scalars, is_distributed


class TDStatsEMA:
    """TD Error EMA 통계 추적기

    학습 전체에서 mean/std를 EMA로 추적하여 일관된 정규화 제공.

    사용 흐름:
        1. 인스턴스 생성 (학습 시작 시 1회)
        2. get_stats()로 현재 EMA 통계 조회
        3. update()로 현재 batch 통계 반영
        4. state_dict()/load_state_dict()로 checkpoint 저장/로드

    Examples:
        >>> ema = TDStatsEMA(device)
        >>> for batch in batches:
        ...     td_errors = compute_td_errors(...)
        ...     mean, std = ema.get_stats()
        ...     weights = build_weights(..., external_mean=mean, external_std=std)
        ...     loss.backward()
        ...     ema.update(td_errors, loss_mask)
    """

    def __init__(
        self,
        device: torch.device,
        momentum: float = 0.1,
        warmup_steps: int = 10,
    ):
        """
        Args:
            device: torch.device (cuda:N, mps, cpu)
            momentum: EMA 업데이트 계수 (0.1 = 90% 이전 + 10% 현재)
            warmup_steps: 초기 warmup step 수 (이 기간 동안 점진적 초기화)
        """
        self.device = device
        self.momentum = momentum
        self.warmup_steps = warmup_steps

        # EMA 통계 (학습 전체에서 누적)
        self.ema_mean = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.ema_std = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.step_count = 0

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """현재 EMA 통계 반환

        Returns:
            (ema_mean, ema_std) 튜플
        """
        return self.ema_mean, self.ema_std

    def update(
        self,
        td_errors: torch.Tensor,
        loss_mask: torch.Tensor,
        distributed: bool = True,
    ):
        """EMA 통계 업데이트

        현재 batch의 통계를 계산하고 EMA에 반영.
        분산 학습 시 all-reduce로 GPU 간 평균 동기화.

        Args:
            td_errors: [batch, seq] TD error 텐서
            loss_mask: [batch, seq] 학습 대상 토큰 마스크 (labels != -100)
            distributed: True면 all-reduce로 GPU 간 동기화
        """
        # 현재 batch 통계 계산
        valid_td = td_errors[loss_mask.bool()].detach().float()

        # 유효 토큰이 없으면 업데이트 스킵
        if valid_td.numel() == 0:
            return

        batch_mean = valid_td.mean()
        batch_std = valid_td.std() if valid_td.numel() > 1 else torch.tensor(1.0, device=self.device)

        # 분산 학습 시 GPU 간 평균 동기화
        if distributed and is_distributed():
            reduced = all_reduce_scalars(
                {
                    "mean": batch_mean.item(),
                    "std": batch_std.item(),
                },
                op="mean",
            )
            batch_mean = torch.tensor(reduced["mean"], device=self.device, dtype=torch.float32)
            batch_std = torch.tensor(reduced["std"], device=self.device, dtype=torch.float32)

        self.step_count += 1

        # Warmup 기간: 점진적 초기화
        # - step 1: 100% batch (초기화)
        # - step N: (N-1)/N% EMA + 1/N% batch
        if self.step_count <= self.warmup_steps:
            warmup_weight = (self.step_count - 1) / self.warmup_steps
            self.ema_mean = warmup_weight * self.ema_mean + (1 - warmup_weight) * batch_mean
            self.ema_std = warmup_weight * self.ema_std + (1 - warmup_weight) * batch_std
        else:
            # 일반 EMA 업데이트: ema = (1 - momentum) * ema + momentum * batch
            self.ema_mean = (1 - self.momentum) * self.ema_mean + self.momentum * batch_mean
            self.ema_std = (1 - self.momentum) * self.ema_std + self.momentum * batch_std

    def state_dict(self) -> dict:
        """Checkpoint 저장용 state dict 반환"""
        return {
            "ema_mean": self.ema_mean.item(),
            "ema_std": self.ema_std.item(),
            "step_count": self.step_count,
            "momentum": self.momentum,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict: dict):
        """Checkpoint에서 state 복원"""
        self.ema_mean = torch.tensor(state_dict["ema_mean"], device=self.device, dtype=torch.float32)
        self.ema_std = torch.tensor(state_dict["ema_std"], device=self.device, dtype=torch.float32)
        self.step_count = state_dict["step_count"]
        # momentum, warmup_steps는 config에서 오므로 복원하지 않음

    def get_debug_stats(self) -> dict[str, float]:
        """디버깅/로깅용 상세 통계 반환"""
        return {
            "ema_mean": self.ema_mean.item(),
            "ema_std": self.ema_std.item(),
            "step_count": self.step_count,
            "is_warmup": self.step_count <= self.warmup_steps,
        }
