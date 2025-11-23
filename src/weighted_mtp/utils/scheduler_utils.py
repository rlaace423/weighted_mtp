"""Learning Rate Scheduler 유틸리티

Warmup + Cosine/Linear decay 스케줄러 생성
Parameter groups 분리 (trunk/value_head)
"""

import logging
from typing import Optional, TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LRScheduler,
)

if TYPE_CHECKING:
    from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter

logger = logging.getLogger(__name__)


def create_param_groups(
    adapter: "MetaLlamaMTPAdapter",
    trunk_lr: float,
    value_head_lr: float,
) -> list[dict]:
    """Optimizer용 parameter groups 생성 (trunk/value_head 분리)

    Args:
        adapter: MetaLlamaMTPAdapter 인스턴스
        trunk_lr: Trunk (transformer) peak learning rate
        value_head_lr: Value head peak learning rate

    Returns:
        Parameter groups list for optimizer

    Note:
        - Trunk params: requires_grad=True인 transformer parameters
        - Value head params: value_head parameters
        - Scheduler는 각 group의 lr을 기준으로 동일 비율 적용
    """
    # Value head parameters
    value_head_params = list(adapter.value_head.parameters())

    # Trunk parameters (requires_grad=True인 transformer params만)
    trunk_params = [
        p for p in adapter.transformer.parameters()
        if p.requires_grad
    ]

    # Logging
    n_trunk = sum(p.numel() for p in trunk_params)
    n_head = sum(p.numel() for p in value_head_params)

    logger.info(f"Parameter groups:")
    logger.info(f"  Trunk: {n_trunk:,} params, lr={trunk_lr:.2e}")
    logger.info(f"  Value head: {n_head:,} params, lr={value_head_lr:.2e}")

    param_groups = [
        {"params": trunk_params, "lr": trunk_lr},
        {"params": value_head_params, "lr": value_head_lr},
    ]

    return param_groups


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.0,
) -> Optional[LRScheduler]:
    """Learning rate scheduler 생성

    Warmup phase 후 main decay schedule 적용

    Args:
        optimizer: Optimizer
        total_steps: Total optimization steps
        scheduler_type: Scheduler 타입
            - "cosine": Cosine annealing (권장)
            - "linear": Linear decay
            - "constant": Scheduler 미사용
        warmup_ratio: Warmup steps ratio (0.0 ~ 1.0)
        min_lr_ratio: Minimum LR ratio relative to peak (0.0 ~ 1.0)

    Returns:
        LRScheduler 또는 None (constant인 경우)

    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        >>> scheduler = create_scheduler(
        ...     optimizer=optimizer,
        ...     total_steps=10000,
        ...     scheduler_type="cosine",
        ...     warmup_ratio=0.05,
        ...     min_lr_ratio=0.0,
        ... )
        >>> # Training loop
        >>> for step in range(total_steps):
        ...     optimizer.step()
        ...     scheduler.step()
    """
    if scheduler_type == "constant":
        logger.info("Scheduler: constant (no scheduling)")
        return None

    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = total_steps - warmup_steps

    if warmup_steps <= 0:
        warmup_steps = 1  # 최소 1 step

    if decay_steps <= 0:
        logger.warning(f"decay_steps={decay_steps} <= 0, using constant LR")
        return None

    peak_lr = optimizer.defaults["lr"]
    min_lr = peak_lr * min_lr_ratio

    logger.info(f"Scheduler: {scheduler_type}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps} ({warmup_ratio:.1%})")
    logger.info(f"  Decay steps: {decay_steps}")
    logger.info(f"  Peak LR: {peak_lr:.2e}")
    logger.info(f"  Min LR: {min_lr:.2e}")

    # Warmup scheduler (0 → peak)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,  # 거의 0에서 시작
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Main decay scheduler
    if scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=min_lr,
        )
    elif scheduler_type == "linear":
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio if min_lr_ratio > 0 else 1e-8,
            total_iters=decay_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Warmup → Main 순차 실행
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    return scheduler


def get_scheduler_state(scheduler: Optional[LRScheduler]) -> Optional[dict]:
    """Scheduler state dict 반환 (checkpoint 저장용)

    Args:
        scheduler: LRScheduler 또는 None

    Returns:
        state_dict 또는 None
    """
    if scheduler is None:
        return None
    return scheduler.state_dict()


def load_scheduler_state(
    scheduler: Optional[LRScheduler],
    state_dict: Optional[dict],
) -> None:
    """Scheduler state dict 로드 (checkpoint 복원용)

    Args:
        scheduler: LRScheduler 또는 None
        state_dict: 저장된 state_dict 또는 None
    """
    if scheduler is None or state_dict is None:
        return
    scheduler.load_state_dict(state_dict)
    logger.info("Scheduler state loaded")
