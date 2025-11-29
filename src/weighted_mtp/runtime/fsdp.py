"""FSDP 유틸리티

FullyShardedDataParallel model wrapping, unwrapping, metric aggregation
"""

import functools
import logging
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from weighted_mtp.runtime.distributed import is_distributed, is_main_process

logger = logging.getLogger(__name__)


def _detect_transformer_layer_cls(model: torch.nn.Module) -> set:
    """모델 타입에 따른 Transformer Layer 클래스 감지

    Args:
        model: 원본 모델

    Returns:
        FSDP auto_wrap_policy에 사용할 transformer_layer_cls set
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    # HuggingFace LlamaForCausalLM 감지 (model.model.layers 구조)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return {LlamaDecoderLayer}

    # ValueModel 감지 (model.backbone.layers 구조, LlamaModel 기반)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        return {LlamaDecoderLayer}

    # MetaLlamaMTPAdapter (기본)
    from weighted_mtp.models.meta_mtp.transformer import TransformerBlock
    return {TransformerBlock}


def wrap_model_fsdp(
    model: torch.nn.Module,
    device: torch.device,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    ignored_modules: list[torch.nn.Module] | None = None,
) -> torch.nn.Module:
    """FSDP로 모델 래핑

    Distributed 환경에서만 FSDP 적용, MPS/CPU local test는 skip
    MetaLlamaMTPAdapter와 HuggingFace LlamaForCausalLM 모두 지원

    Args:
        model: 원본 모델
        device: torch.device (cuda:rank, mps, cpu)
        sharding_strategy: 샤딩 전략 ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD")
            - FULL_SHARD: ZeRO-3 (Model + Optimizer + Gradient 샤딩)
            - SHARD_GRAD_OP: ZeRO-2 (Optimizer + Gradient 샤딩)
            - NO_SHARD: DDP와 동일 (모델 복제)
        mixed_precision: FP16 mixed precision 사용 여부
        cpu_offload: CPU 오프로드 (메모리 부족 시)
        activation_checkpointing: Activation checkpointing 적용 여부 (메모리 절감)
        ignored_modules: FSDP wrapping에서 제외할 모듈 리스트 (개별 학습 필요 시)

    Returns:
        FSDP-wrapped model (또는 원본 model if not distributed)

    Examples:
        >>> device = torch.device("cuda:0")
        >>> model = MyModel().to(device)
        >>> model = wrap_model_fsdp(model, device, sharding_strategy="FULL_SHARD")
    """
    if not is_distributed():
        # MPS/CPU local test 또는 single-GPU - no wrapping
        if is_main_process():
            logger.info("단일 장치 환경: FSDP wrapping을 사용하지 않습니다.")
        return model

    # Sharding strategy 변환
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision 설정
    mp_policy = None
    if mixed_precision:
        # 모델의 dtype을 자동 감지 (Config에서 정의된 dtype 사용)
        model_dtype = next(model.parameters()).dtype
        mp_policy = MixedPrecision(
            param_dtype=model_dtype,
            reduce_dtype=model_dtype,
            buffer_dtype=model_dtype,
        )

    # CPU offload 설정
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    # Auto Wrap Policy 설정 (모델 타입에 따라 자동 감지)
    transformer_layer_cls = _detect_transformer_layer_cls(model)

    fsdp_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )

    # FSDP wrapping
    # use_orig_params=True: 원본 parameter 구조 유지
    # - FlatParameter 생성 오버헤드 제거 (첫 forward 속도 개선)
    # - All-gather를 작은 단위로 분산 (NCCL timeout 방지)
    # - NCCL Socket Fallback 환경에서 안정성 향상
    # - adapter() 통한 정상적인 FSDP Hook 실행으로 sharded parameter 접근 문제 해결
    wrapped_model = FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_config,
        device_id=device.index if device.type == "cuda" else None,
        sync_module_states=True,
        use_orig_params=True,
        ignored_modules=ignored_modules,
    )

    # Activation Checkpointing 적용 (FSDP wrapping 후)
    if activation_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        # 모델 타입에 따른 체크 함수 생성
        layer_cls_tuple = tuple(transformer_layer_cls)
        apply_activation_checkpointing(
            wrapped_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, layer_cls_tuple),
        )

    if is_main_process():
        logger.info(
            f"FSDP wrapping 완료: sharding_strategy={sharding_strategy}, "
            f"mixed_precision={mixed_precision}, cpu_offload={cpu_offload}, "
            f"activation_checkpointing={activation_checkpointing}"
        )

    return wrapped_model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """FSDP wrapper 제거하여 원본 모델 추출

    Checkpoint 저장 시 사용 (FSDP wrapper 제외하고 state_dict 저장)

    Args:
        model: FSDP-wrapped model (또는 원본 model)

    Returns:
        Original model

    Examples:
        >>> wrapped_model = wrap_model_fsdp(model, device)
        >>> # Training...
        >>> original_model = unwrap_model(wrapped_model)
        >>> torch.save(original_model.state_dict(), "checkpoint.pt")
    """
    if isinstance(model, FSDP):
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
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    tensor = torch.tensor(value, device=device)

    # All-reduce (SUM)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Mean 계산 (필요 시)
    if op == "mean":
        world_size = dist.get_world_size()
        tensor /= world_size

    return tensor.item()


def all_reduce_scalars(
    values: dict[str, float],
    op: str = "mean",
) -> dict[str, float]:
    """GPU ranks 간 여러 scalar 값을 한 번에 집계

    다중 all_reduce_scalar 호출 대신 단일 통신으로 효율적으로 처리

    Args:
        values: {이름: 값} 딕셔너리
        op: "mean" (평균) 또는 "sum" (합계)

    Returns:
        전체 ranks에서 집계된 값 딕셔너리

    Examples:
        >>> metrics = {"loss": 0.5, "accuracy": 0.8}
        >>> avg_metrics = all_reduce_scalars(metrics)  # 1회 통신
    """
    if not is_distributed():
        return values

    keys = list(values.keys())
    vals = [values[k] for k in keys]

    # 단일 텐서로 변환
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    tensor = torch.tensor(vals, dtype=torch.float64, device=device)

    # 단일 all-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Mean 계산 (필요 시)
    if op == "mean":
        world_size = dist.get_world_size()
        tensor /= world_size

    # 딕셔너리로 변환
    result = {k: tensor[i].item() for i, k in enumerate(keys)}
    return result
