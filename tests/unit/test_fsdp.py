"""FSDP 유틸리티 테스트

Tests:
- wrap_model_fsdp: FSDP wrapping (distributed/single-device)
- unwrap_model: FSDP wrapper 제거
- all_reduce_scalar: Metric aggregation
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from weighted_mtp.runtime import (
    is_distributed,
)
from weighted_mtp.runtime.fsdp import (
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalar,
)


class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """간단한 테스트 모델"""
    return SimpleModel()


@pytest.fixture
def device():
    """테스트용 디바이스"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def test_wrap_model_fsdp_single_device(simple_model, device):
    """단일 장치 환경에서는 FSDP wrapping을 하지 않음"""
    model = simple_model.to(device)

    # Single-device 환경에서는 원본 모델 반환
    wrapped_model = wrap_model_fsdp(model, device)

    # FSDP로 wrapping되지 않아야 함
    assert not isinstance(wrapped_model, FSDP)
    assert wrapped_model is model


def test_wrap_model_fsdp_no_shard(simple_model, device):
    """NO_SHARD strategy 테스트 (DDP와 동일)"""
    model = simple_model.to(device)

    # NO_SHARD로 wrapping 시도
    wrapped_model = wrap_model_fsdp(
        model, device, sharding_strategy="NO_SHARD"
    )

    # Single-device 환경이므로 FSDP로 wrapping되지 않아야 함
    if not is_distributed():
        assert not isinstance(wrapped_model, FSDP)
    else:
        # Distributed 환경에서는 FSDP wrapping됨
        assert isinstance(wrapped_model, FSDP)


def test_wrap_model_fsdp_full_shard(simple_model, device):
    """FULL_SHARD strategy 테스트 (메모리 샤딩)"""
    model = simple_model.to(device)

    # FULL_SHARD로 wrapping 시도
    wrapped_model = wrap_model_fsdp(
        model, device, sharding_strategy="FULL_SHARD"
    )

    # Single-device 환경이므로 FSDP로 wrapping되지 않아야 함
    if not is_distributed():
        assert not isinstance(wrapped_model, FSDP)
    else:
        # Distributed 환경에서는 FSDP wrapping됨
        assert isinstance(wrapped_model, FSDP)


def test_wrap_model_fsdp_mixed_precision(simple_model, device):
    """Mixed precision 옵션 테스트"""
    model = simple_model.to(device)

    # mixed_precision=True로 wrapping 시도
    wrapped_model = wrap_model_fsdp(
        model, device, mixed_precision=True
    )

    # Single-device 환경이므로 FSDP로 wrapping되지 않아야 함
    assert not isinstance(wrapped_model, FSDP)


def test_wrap_model_fsdp_cpu_offload(simple_model, device):
    """CPU offload 옵션 테스트"""
    model = simple_model.to(device)

    # cpu_offload=True로 wrapping 시도
    wrapped_model = wrap_model_fsdp(
        model, device, cpu_offload=True
    )

    # Single-device 환경이므로 FSDP로 wrapping되지 않아야 함
    assert not isinstance(wrapped_model, FSDP)


def test_unwrap_model_non_fsdp(simple_model):
    """FSDP가 아닌 모델은 그대로 반환"""
    unwrapped = unwrap_model(simple_model)
    assert unwrapped is simple_model


def test_unwrap_model_fsdp_wrapped(simple_model, device):
    """FSDP-wrapped 모델은 원본 모델 반환"""
    # FSDP로 수동 wrapping (테스트 목적)
    if device.type == "cuda" and is_distributed():
        model = simple_model.to(device)

        # TransformerBlock을 사용하지 않는 간단한 모델이므로
        # auto_wrap_policy 없이 wrapping
        fsdp_model = FSDP(model, device_id=device.index)

        # Unwrap 후 원본 모델과 동일해야 함
        unwrapped = unwrap_model(fsdp_model)
        assert unwrapped is model
    else:
        # FSDP가 불가능한 환경에서는 skip
        pytest.skip("Distributed environment not available")


def test_all_reduce_scalar_single_device():
    """단일 장치 환경에서는 값 그대로 반환"""
    value = 3.14
    result = all_reduce_scalar(value)

    # Single-device 환경에서는 입력값 그대로 반환
    assert result == value


def test_all_reduce_scalar_mean():
    """평균 집계 테스트"""
    value = 2.5
    result = all_reduce_scalar(value, op="mean")

    # Single-device 환경에서는 입력값 그대로 반환
    assert result == value


def test_all_reduce_scalar_sum():
    """합계 집계 테스트"""
    value = 1.0
    result = all_reduce_scalar(value, op="sum")

    # Single-device 환경에서는 입력값 그대로 반환
    assert result == value


def test_wrap_and_unwrap_model(simple_model, device):
    """wrap과 unwrap의 일관성 테스트"""
    model = simple_model.to(device)

    # Wrap 후 unwrap
    wrapped_model = wrap_model_fsdp(model, device)
    unwrapped_model = unwrap_model(wrapped_model)

    # 원본 모델과 동일해야 함
    assert unwrapped_model is model


def test_model_state_dict_after_unwrap(simple_model, device):
    """Unwrap 후 state_dict가 동일한지 테스트"""
    model = simple_model.to(device)

    # 원본 state_dict
    original_state_dict = model.state_dict()

    # Wrap 후 unwrap
    wrapped_model = wrap_model_fsdp(model, device)
    unwrapped_model = unwrap_model(wrapped_model)

    # state_dict가 동일해야 함
    unwrapped_state_dict = unwrapped_model.state_dict()

    assert set(original_state_dict.keys()) == set(unwrapped_state_dict.keys())

    for key in original_state_dict.keys():
        assert torch.equal(original_state_dict[key], unwrapped_state_dict[key])


def test_model_forward_after_wrap(simple_model, device):
    """FSDP wrapping 후에도 forward pass가 정상 동작하는지 테스트"""
    model = simple_model.to(device)
    wrapped_model = wrap_model_fsdp(model, device)

    # 테스트 입력
    x = torch.randn(2, 10, device=device)

    # Forward pass
    with torch.no_grad():
        output = wrapped_model(x)

    # 출력 shape 확인
    assert output.shape == (2, 5)


def test_all_reduce_scalar_type_preservation():
    """all_reduce_scalar이 float 타입을 반환하는지 테스트"""
    value = 42.0
    result = all_reduce_scalar(value)

    assert isinstance(result, float)


def test_wrap_model_fsdp_device_types(simple_model):
    """다양한 device 타입에서 wrap_model_fsdp 테스트"""
    # CPU
    cpu_device = torch.device("cpu")
    model_cpu = simple_model.to(cpu_device)
    wrapped_cpu = wrap_model_fsdp(model_cpu, cpu_device)
    assert not isinstance(wrapped_cpu, FSDP)

    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        model_mps = SimpleModel().to(mps_device)
        wrapped_mps = wrap_model_fsdp(model_mps, mps_device)
        assert not isinstance(wrapped_mps, FSDP)

    # CUDA (single-GPU)
    if torch.cuda.is_available() and not is_distributed():
        cuda_device = torch.device("cuda:0")
        model_cuda = SimpleModel().to(cuda_device)
        wrapped_cuda = wrap_model_fsdp(model_cuda, cuda_device)
        # Single-GPU 환경에서는 FSDP wrapping 하지 않음
        assert not isinstance(wrapped_cuda, FSDP)


def test_wrap_model_fsdp_strategy_validation(simple_model, device):
    """잘못된 sharding_strategy는 FULL_SHARD로 fallback"""
    model = simple_model.to(device)

    # 존재하지 않는 strategy 사용
    wrapped_model = wrap_model_fsdp(
        model, device, sharding_strategy="INVALID_STRATEGY"
    )

    # Single-device에서는 wrapping 안 됨
    if not is_distributed():
        assert not isinstance(wrapped_model, FSDP)
