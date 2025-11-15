"""분산학습 환경 초기화 및 유틸리티

핵심 기능:
- torch.distributed 초기화 (NCCL backend)
- DistributedSampler 생성 헬퍼
- Rank/World size 조회
- FSDP 설정 헬퍼 (Phase 6에서 사용)

VESSL A100 4-GPU 환경 기준으로 설계됨.
"""

import os
import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

logger = logging.getLogger(__name__)


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    timeout_minutes: int = 30,
) -> tuple[int, int]:
    """분산 학습 환경 초기화

    torchrun이 설정한 환경 변수(RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)를
    기반으로 torch.distributed를 초기화합니다.

    Args:
        backend: 통신 백엔드 (기본: "nccl" for GPU, "gloo" for CPU)
        init_method: 초기화 방법 (None이면 환경 변수 기반)
        timeout_minutes: 초기화 타임아웃 (분)

    Returns:
        (rank, world_size) 튜플

    Raises:
        RuntimeError: 분산 환경 변수가 설정되지 않은 경우

    Examples:
        >>> # torchrun --nproc_per_node=4 train.py 실행 후
        >>> rank, world_size = init_distributed()
        >>> print(f"Rank {rank}/{world_size}")
        Rank 0/4
    """
    # 환경 변수 확인
    required_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        raise RuntimeError(
            f"분산 환경 변수가 설정되지 않았습니다: {missing_vars}\n"
            "torchrun 또는 torch.distributed.launch를 사용하여 실행하세요."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Backend 자동 선택
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning("CUDA를 사용할 수 없어 'gloo' 백엔드로 전환합니다.")
        backend = "gloo"

    # 초기화
    if not dist.is_initialized():
        timeout = torch.distributed.default_pg_timeout
        if timeout_minutes > 0:
            timeout = torch.timedelta(minutes=timeout_minutes)

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timeout,
        )

        if is_main_process():
            logger.info(
                f"분산 환경 초기화 완료: backend={backend}, "
                f"world_size={world_size}, rank={rank}"
            )

    return rank, world_size


def get_rank() -> int:
    """현재 프로세스의 global rank 반환

    Returns:
        Rank (0 ~ world_size-1). 분산 환경이 아니면 0.

    Examples:
        >>> rank = get_rank()
        >>> if rank == 0:
        ...     print("Main process")
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """전체 프로세스 수 반환

    Returns:
        World size (분산 환경에서는 GPU 수). 분산 환경이 아니면 1.

    Examples:
        >>> world_size = get_world_size()
        >>> print(f"Training on {world_size} GPUs")
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """현재 노드 내 local rank 반환

    Returns:
        Local rank (0 ~ nproc_per_node-1). 분산 환경이 아니면 0.

    Examples:
        >>> local_rank = get_local_rank()
        >>> device = torch.device(f"cuda:{local_rank}")
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return get_rank()


def is_main_process() -> bool:
    """Main process (rank 0) 여부 확인

    Returns:
        rank == 0이면 True

    Examples:
        >>> if is_main_process():
        ...     # MLflow 로깅, 체크포인트 저장
        ...     save_checkpoint(model, "checkpoint.pt")
    """
    return get_rank() == 0


def is_distributed() -> bool:
    """분산 학습 환경 여부 확인

    Returns:
        torch.distributed가 초기화되었고 world_size > 1이면 True

    Examples:
        >>> if is_distributed():
        ...     sampler = DistributedSampler(dataset)
        ... else:
        ...     sampler = None
    """
    return dist.is_available() and dist.is_initialized() and get_world_size() > 1


def create_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
) -> Optional[DistributedSampler]:
    """DistributedSampler 생성 헬퍼

    분산 환경이면 DistributedSampler를 반환하고,
    로컬 환경이면 None을 반환합니다 (DataLoader의 shuffle 사용).

    Args:
        dataset: PyTorch Dataset
        shuffle: 샘플 셔플 여부
        seed: 랜덤 시드 (재현성)
        drop_last: 마지막 불완전한 배치 버릴지 여부

    Returns:
        DistributedSampler 또는 None (로컬 환경)

    Examples:
        >>> dataset = load_dataset("codecontests", split="train", n_samples=200000)
        >>> sampler = create_distributed_sampler(dataset, shuffle=True, seed=42)
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=8,
        ...     sampler=sampler,
        ...     shuffle=(sampler is None),  # sampler 없을 때만 shuffle
        ... )
        >>>
        >>> # Epoch 시작 시 (재현성을 위해 필수)
        >>> for epoch in range(num_epochs):
        ...     if sampler is not None:
        ...         sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """
    if not is_distributed():
        logger.info("로컬 환경: DistributedSampler를 사용하지 않습니다.")
        return None

    rank = get_rank()
    world_size = get_world_size()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    if is_main_process():
        logger.info(
            f"DistributedSampler 생성: "
            f"전체 샘플={len(dataset)}, "
            f"GPU당 샘플={len(dataset) // world_size}, "
            f"world_size={world_size}"
        )

    return sampler


def barrier():
    """모든 프로세스 동기화 (barrier)

    모든 프로세스가 이 지점에 도달할 때까지 대기합니다.

    Examples:
        >>> # Rank 0이 데이터 전처리
        >>> if is_main_process():
        ...     preprocess_data()
        >>> barrier()  # 모든 GPU가 전처리 완료까지 대기
        >>> # 이제 모든 GPU가 전처리된 데이터 사용 가능
    """
    if is_distributed():
        dist.barrier()


def cleanup_distributed():
    """분산 환경 정리

    torch.distributed 프로세스 그룹을 종료합니다.

    Examples:
        >>> try:
        ...     rank, world_size = init_distributed()
        ...     train(...)
        ... finally:
        ...     cleanup_distributed()
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        if is_main_process():
            logger.info("분산 환경 정리 완료")


def setup_fsdp_config(
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision: str = "bf16",
    activation_checkpointing: bool = True,
) -> dict:
    """FSDP 설정 딕셔너리 생성 (Phase 6에서 사용)

    Args:
        sharding_strategy: 샤딩 전략 (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD)
        cpu_offload: CPU 오프로드 여부
        mixed_precision: 혼합 정밀도 (bf16, fp16, fp32)
        activation_checkpointing: Activation checkpointing 사용 여부

    Returns:
        FSDP 설정 딕셔너리

    Note:
        실제 FSDP wrapping은 Phase 6 (모델 통합)에서 구현됩니다.
        이 함수는 설정 생성만 담당합니다.
    """
    config = {
        "sharding_strategy": sharding_strategy,
        "cpu_offload": cpu_offload,
        "mixed_precision": mixed_precision,
        "activation_checkpointing": activation_checkpointing,
    }

    if is_main_process():
        logger.info(f"FSDP 설정: {config}")

    return config
