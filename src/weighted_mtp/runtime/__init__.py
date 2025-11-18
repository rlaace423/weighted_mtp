"""런타임 환경 관리

Modules:
- distributed.py: 분산학습 초기화 (FSDP, DistributedSampler)
- environment.py: seed, device, dtype 설정 (rank-aware)
- mlflow.py: MLflow 초기화 및 로깅 (Phase 6 구현 예정)
"""

from weighted_mtp.runtime.distributed import (
    init_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    is_distributed,
    create_distributed_sampler,
    barrier,
    cleanup_distributed,
    setup_fsdp_config,
)

from weighted_mtp.runtime.environment import (
    setup_seed,
    get_device,
    setup_torch_backends,
    setup_environment,
    get_gpu_memory_info,
)

from weighted_mtp.runtime.fsdp import (
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalar,
)

__all__ = [
    # distributed.py
    "init_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "is_distributed",
    "create_distributed_sampler",
    "barrier",
    "cleanup_distributed",
    "setup_fsdp_config",
    # environment.py
    "setup_seed",
    "get_device",
    "setup_torch_backends",
    "setup_environment",
    "get_gpu_memory_info",
    # fsdp.py
    "wrap_model_fsdp",
    "unwrap_model",
    "all_reduce_scalar",
]
