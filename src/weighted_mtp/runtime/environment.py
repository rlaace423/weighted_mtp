"""실행 환경 설정 (seed, device, dtype)

핵심 기능:
- Rank-aware seed 설정 (base_seed + rank)
- Device 할당 (cuda:{rank} 또는 cpu/mps)
- PyTorch backends 최적화
- 재현성 보장

분산학습 환경에서 각 GPU가 독립적이면서도 재현 가능한 난수를 생성합니다.
"""

import os
import random
import logging
from typing import Optional, Literal

import numpy as np
import torch

from weighted_mtp.runtime.distributed import get_rank, get_local_rank, is_main_process

logger = logging.getLogger(__name__)


def setup_seed(base_seed: int, rank: Optional[int] = None) -> int:
    """재현성을 위한 seed 설정 (rank-aware)

    분산학습 환경에서는 각 GPU가 다른 seed를 사용하여
    독립적인 난수를 생성하지만, 재현은 가능합니다.

    Args:
        base_seed: 기본 시드 (예: 42)
        rank: GPU rank (None이면 자동 감지)

    Returns:
        실제 사용된 seed (base_seed + rank)

    Examples:
        >>> # Rank 0: seed=42, Rank 1: seed=43, Rank 2: seed=44, ...
        >>> seed = setup_seed(base_seed=42)
        >>> print(f"Using seed: {seed}")
        Using seed: 42  # Rank 0인 경우
    """
    if rank is None:
        rank = get_rank()

    # 각 rank마다 다른 seed 사용
    actual_seed = base_seed + rank

    # Python random
    random.seed(actual_seed)

    # NumPy random
    np.random.seed(actual_seed)

    # PyTorch random
    torch.manual_seed(actual_seed)

    # CUDA random (GPU 사용 시)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)  # 모든 GPU

    if is_main_process():
        logger.info(f"Seed 설정 완료: base_seed={base_seed}, rank={rank}, actual_seed={actual_seed}")

    return actual_seed


def get_device(
    rank: Optional[int] = None,
    force_cpu: bool = False,
) -> torch.device:
    """적절한 device 반환 (cuda:{rank}, mps, 또는 cpu)

    분산학습 환경에서는 각 프로세스가 자신의 GPU를 사용합니다.
    로컬 환경에서는 사용 가능한 최선의 device를 반환합니다.

    Args:
        rank: GPU rank (None이면 자동 감지, local_rank 사용)
        force_cpu: CPU 강제 사용 여부

    Returns:
        torch.device 객체

    Examples:
        >>> # VESSL A100 4-GPU 환경
        >>> device = get_device()  # Rank 0 → cuda:0, Rank 1 → cuda:1, ...
        >>>
        >>> # 로컬 M3 Mac 환경
        >>> device = get_device()  # mps (Metal Performance Shaders)
        >>>
        >>> # CPU 강제
        >>> device = get_device(force_cpu=True)  # cpu
    """
    if force_cpu:
        return torch.device("cpu")

    # CUDA 사용 가능 (분산학습 또는 단일 GPU)
    if torch.cuda.is_available():
        if rank is None:
            rank = get_local_rank()  # 노드 내 local rank 사용

        # 현재 프로세스의 기본 CUDA device 설정 (barrier 데드락 방지)
        torch.cuda.set_device(rank)

        device = torch.device(f"cuda:{rank}")

        if is_main_process():
            gpu_name = torch.cuda.get_device_name(rank)
            logger.info(f"Device 설정: {device} ({gpu_name})")

        return device

    # MPS 사용 가능 (M1/M2/M3 Mac)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if is_main_process():
            logger.info(f"Device 설정: {device} (Apple Silicon)")
        return device

    # CPU fallback
    device = torch.device("cpu")
    if is_main_process():
        logger.warning("GPU/MPS를 사용할 수 없어 CPU를 사용합니다.")
    return device


def setup_torch_backends(
    cudnn_benchmark: bool = True,
    cudnn_deterministic: bool = False,
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high",
) -> None:
    """PyTorch backends 최적화 설정

    Args:
        cudnn_benchmark: cuDNN auto-tuner 활성화 (속도 향상, 약간의 비재현성)
        cudnn_deterministic: 완전한 재현성 보장 (속도 저하)
        float32_matmul_precision: float32 행렬곱 정밀도
            - "highest": 가장 정확, 느림
            - "high": 균형 (기본, A100 최적)
            - "medium": 빠름, 약간 부정확

    Examples:
        >>> # A100에서 최고 성능 (약간의 비재현성 허용)
        >>> setup_torch_backends(cudnn_benchmark=True, cudnn_deterministic=False)
        >>>
        >>> # 완전한 재현성 필요 시
        >>> setup_torch_backends(cudnn_benchmark=False, cudnn_deterministic=True)
    """
    # cuDNN 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic

        if cudnn_deterministic:
            # 완전한 재현성을 위한 추가 설정
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

    # TensorFloat-32 (TF32) 설정 (A100 native support)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = (float32_matmul_precision != "highest")

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = (float32_matmul_precision != "highest")

    # Float32 행렬곱 정밀도 설정
    torch.set_float32_matmul_precision(float32_matmul_precision)

    if is_main_process():
        logger.info(
            f"PyTorch backends 설정: "
            f"cudnn_benchmark={cudnn_benchmark}, "
            f"cudnn_deterministic={cudnn_deterministic}, "
            f"float32_matmul_precision={float32_matmul_precision}"
        )


def setup_environment(
    base_seed: int = 42,
    rank: Optional[int] = None,
    force_cpu: bool = False,
    cudnn_benchmark: bool = True,
    cudnn_deterministic: bool = False,
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high",
) -> tuple[int, torch.device]:
    """환경 전체 설정 (seed + device + backends)

    분산학습 환경 초기화 후 호출하여 각 GPU의 환경을 설정합니다.

    Args:
        base_seed: 기본 시드
        rank: GPU rank (None이면 자동 감지)
        force_cpu: CPU 강제 사용
        cudnn_benchmark: cuDNN auto-tuner 활성화
        cudnn_deterministic: 완전한 재현성 보장
        float32_matmul_precision: float32 행렬곱 정밀도

    Returns:
        (actual_seed, device) 튜플

    Examples:
        >>> # 분산학습 환경
        >>> from weighted_mtp.runtime.distributed import init_distributed
        >>> rank, world_size = init_distributed()
        >>> seed, device = setup_environment(base_seed=42)
        >>> print(f"Rank {rank}: seed={seed}, device={device}")
        Rank 0: seed=42, device=cuda:0
        >>>
        >>> # 로컬 환경
        >>> seed, device = setup_environment(base_seed=42)
        >>> print(f"seed={seed}, device={device}")
        seed=42, device=mps
    """
    # Seed 설정
    actual_seed = setup_seed(base_seed, rank)

    # Device 설정
    device = get_device(rank, force_cpu)

    # PyTorch backends 최적화
    setup_torch_backends(
        cudnn_benchmark=cudnn_benchmark,
        cudnn_deterministic=cudnn_deterministic,
        float32_matmul_precision=float32_matmul_precision,
    )

    if is_main_process():
        logger.info(f"환경 설정 완료: seed={actual_seed}, device={device}")

    return actual_seed, device


def get_gpu_memory_info(device: Optional[torch.device] = None) -> dict[str, float]:
    """GPU 메모리 사용량 조회

    Args:
        device: GPU device (None이면 현재 device)

    Returns:
        메모리 정보 딕셔너리 (MB 단위)
        - allocated: 할당된 메모리
        - reserved: 예약된 메모리
        - free: 사용 가능한 메모리

    Examples:
        >>> memory = get_gpu_memory_info()
        >>> print(f"Allocated: {memory['allocated']:.1f} MB")
        Allocated: 1234.5 MB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index if device.index is not None else 0

    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB

    # 전체 메모리
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**2  # MB
    free = total - allocated

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }
