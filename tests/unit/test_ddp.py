"""DDP 유틸리티 테스트

Tests:
- wrap_model_ddp: DDP wrapping (distributed/single-device)
- unwrap_model: DDP wrapper 제거
- all_reduce_scalar: Metric aggregation
"""

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from weighted_mtp.runtime import (
    wrap_model_ddp,
    unwrap_model,
    all_reduce_scalar,
    is_distributed,
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


def test_wrap_model_ddp_single_device(simple_model, device):
    """단일 장치 환경에서는 DDP wrapping을 하지 않음"""
    model = simple_model.to(device)

    # Single-device 환경에서는 원본 모델 반환
    wrapped_model = wrap_model_ddp(model, device)

    # DDP로 wrapping되지 않아야 함
    assert not isinstance(wrapped_model, DDP)
    assert wrapped_model is model


def test_wrap_model_ddp_find_unused_parameters(simple_model, device):
    """find_unused_parameters 옵션 테스트"""
    model = simple_model.to(device)

    # find_unused_parameters=True로 wrapping 시도
    wrapped_model = wrap_model_ddp(model, device, find_unused_parameters=True)

    # Single-device 환경이므로 DDP로 wrapping되지 않아야 함
    assert not isinstance(wrapped_model, DDP)


def test_unwrap_model_non_ddp(simple_model):
    """DDP가 아닌 모델은 그대로 반환"""
    unwrapped = unwrap_model(simple_model)
    assert unwrapped is simple_model


def test_unwrap_model_ddp_wrapped(simple_model, device):
    """DDP-wrapped 모델은 원본 모델 반환"""
    # DDP로 수동 wrapping (테스트 목적)
    if device.type == "cuda" and is_distributed():
        model = simple_model.to(device)
        ddp_model = DDP(model, device_ids=[device.index])

        # Unwrap 후 원본 모델과 동일해야 함
        unwrapped = unwrap_model(ddp_model)
        assert unwrapped is model
    else:
        # DDP가 불가능한 환경에서는 skip
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
    wrapped_model = wrap_model_ddp(model, device)
    unwrapped_model = unwrap_model(wrapped_model)

    # 원본 모델과 동일해야 함
    assert unwrapped_model is model


def test_model_state_dict_after_unwrap(simple_model, device):
    """Unwrap 후 state_dict가 동일한지 테스트"""
    model = simple_model.to(device)

    # 원본 state_dict
    original_state_dict = model.state_dict()

    # Wrap 후 unwrap
    wrapped_model = wrap_model_ddp(model, device)
    unwrapped_model = unwrap_model(wrapped_model)

    # state_dict가 동일해야 함
    unwrapped_state_dict = unwrapped_model.state_dict()

    assert set(original_state_dict.keys()) == set(unwrapped_state_dict.keys())

    for key in original_state_dict.keys():
        assert torch.equal(original_state_dict[key], unwrapped_state_dict[key])


def test_model_forward_after_wrap(simple_model, device):
    """DDP wrapping 후에도 forward pass가 정상 동작하는지 테스트"""
    model = simple_model.to(device)
    wrapped_model = wrap_model_ddp(model, device)

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


def test_wrap_model_ddp_device_types(simple_model):
    """다양한 device 타입에서 wrap_model_ddp 테스트"""
    # CPU
    cpu_device = torch.device("cpu")
    model_cpu = simple_model.to(cpu_device)
    wrapped_cpu = wrap_model_ddp(model_cpu, cpu_device)
    assert not isinstance(wrapped_cpu, DDP)

    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        model_mps = SimpleModel().to(mps_device)
        wrapped_mps = wrap_model_ddp(model_mps, mps_device)
        assert not isinstance(wrapped_mps, DDP)

    # CUDA (single-GPU)
    if torch.cuda.is_available() and not is_distributed():
        cuda_device = torch.device("cuda:0")
        model_cuda = SimpleModel().to(cuda_device)
        wrapped_cuda = wrap_model_ddp(model_cuda, cuda_device)
        # Single-GPU 환경에서는 DDP wrapping 하지 않음
        assert not isinstance(wrapped_cuda, DDP)
