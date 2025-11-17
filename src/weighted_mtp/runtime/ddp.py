"""DDP 유틸리티

Distributed Data Parallel model wrapping, unwrapping, metric aggregation
"""

import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from weighted_mtp.runtime.distributed import is_distributed, is_main_process

logger = logging.getLogger(__name__)


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """DDP로 모델 래핑

    Distributed 환경에서만 DDP 적용, MPS/CPU local test는 skip

    Args:
        model: 원본 모델
        device: torch.device (cuda:rank, mps, cpu)
        find_unused_parameters: Unused gradient 허용 (기본 False)

    Returns:
        DDP-wrapped model (또는 원본 model if not distributed)

    Examples:
        >>> device = torch.device("cuda:0")
        >>> model = MyModel().to(device)
        >>> model = wrap_model_ddp(model, device)
    """
    if not is_distributed():
        # MPS/CPU local test 또는 single-GPU - no wrapping
        if is_main_process():
            logger.info("단일 장치 환경: DDP wrapping을 사용하지 않습니다.")
        return model

    # CUDA distributed - DDP wrapping
    device_ids = [device.index] if device.type == "cuda" else None

    wrapped_model = DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )

    if is_main_process():
        logger.info(
            f"DDP wrapping 완료: device_ids={device_ids}, "
            f"find_unused_parameters={find_unused_parameters}"
        )

    return wrapped_model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """DDP wrapper 제거하여 원본 모델 추출

    Checkpoint 저장 시 사용 (DDP wrapper 제외하고 state_dict 저장)

    Args:
        model: DDP-wrapped model (또는 원본 model)

    Returns:
        Original model

    Examples:
        >>> wrapped_model = wrap_model_ddp(model, device)
        >>> # Training...
        >>> original_model = unwrap_model(wrapped_model)
        >>> torch.save(original_model.state_dict(), "checkpoint.pt")
    """
    if isinstance(model, DDP):
        return model.module
    return model


def all_reduce_scalar(
    value: float,
    op: str = "mean",
) -> float:
    """GPU ranks 간 scalar 값 집계

    Loss, accuracy 등 metric 평균/합계 계산

    Args:
        value: 현재 rank의 scalar 값
        op: "mean" (평균) 또는 "sum" (합계)

    Returns:
        전체 ranks에서 집계된 값

    Examples:
        >>> # GPU 0: loss = 0.5, GPU 1: loss = 0.6
        >>> avg_loss = all_reduce_scalar(loss.item())  # 0.55
        >>> if is_main_process():
        ...     mlflow.log_metrics({"train/loss": avg_loss}, step=epoch)
    """
    if not is_distributed():
        return value

    # Tensor 변환 (current device에 위치)
    tensor = torch.tensor(value, device=torch.cuda.current_device())

    # All-reduce (SUM)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Mean 계산 (필요 시)
    if op == "mean":
        world_size = dist.get_world_size()
        tensor /= world_size

    return tensor.item()
