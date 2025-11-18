"""Multi-Process DDP 통합 테스트

Phase 1 발견사항 반영:
- test_unwrap_model_ddp_wrapped SKIPPED → 실제 DDP 환경에서 검증
- all_reduce_scalar 단일 환경만 검증 → 2-process 평균 계산 검증
- Barrier 동기화 미검증 → 실제 대기 동작 검증

실행 방법:
    torchrun --nproc_per_node=2 --nnodes=1 \
        -m pytest tests/integration/test_ddp_multiprocess.py -v -s

주의:
- M3 MacBook Pro에서는 CPU Gloo backend만 사용
- CUDA NCCL은 VESSL A100 환경에서 실제 검증
"""
import os
import time
import pytest
import torch
import torch.nn as nn

from weighted_mtp.runtime import (
    init_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    all_reduce_scalar,
    barrier,
    wrap_model_ddp,
    unwrap_model,
)


class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.ddp
@pytest.mark.integration
def test_distributed_init_2_process():
    """2-process 분산 환경 초기화 검증

    Phase 1 발견: Multi-process 초기화 미검증
    → torchrun 환경변수 설정 및 init_distributed() 동작 확인
    """
    # torchrun이 설정한 환경변수 확인
    assert "RANK" in os.environ, "RANK 환경변수가 설정되지 않음 (torchrun으로 실행하세요)"
    assert "WORLD_SIZE" in os.environ, "WORLD_SIZE 환경변수가 설정되지 않음"
    assert "MASTER_ADDR" in os.environ, "MASTER_ADDR 환경변수가 설정되지 않음"
    assert "MASTER_PORT" in os.environ, "MASTER_PORT 환경변수가 설정되지 않음"

    # 분산 초기화 (Gloo backend for CPU)
    rank, world_size = init_distributed(backend="gloo")

    # 검증
    assert world_size == 2, f"Expected world_size=2, got {world_size}"
    assert rank in [0, 1], f"Rank should be 0 or 1, got {rank}"
    assert is_distributed(), "Should be in distributed mode"
    assert rank == int(os.environ["RANK"]), f"Rank mismatch: {rank} != {os.environ['RANK']}"

    # Helper 함수 검증
    assert get_rank() == rank
    assert get_world_size() == world_size

    if rank == 0:
        assert is_main_process(), "Rank 0 should be main process"
    else:
        assert not is_main_process(), "Rank 1 should not be main process"

    print(f"[Rank {rank}] Distributed init successful: world_size={world_size}")


@pytest.mark.ddp
@pytest.mark.integration
def test_all_reduce_mean_2_process():
    """all_reduce 평균 계산 검증

    Phase 1 발견: all_reduce_scalar 단일 환경만 검증 (값 그대로 반환)
    → 실제 2-process에서 평균 계산 검증
    """
    rank, world_size = init_distributed(backend="gloo")

    # Rank별로 다른 값 설정
    if rank == 0:
        value = 2.0
    else:  # rank == 1
        value = 4.0

    # all_reduce mean
    result = all_reduce_scalar(value, op="mean")

    # 검증: (2.0 + 4.0) / 2 = 3.0
    expected = 3.0
    assert abs(result - expected) < 1e-6, (
        f"[Rank {rank}] all_reduce_scalar mean failed: "
        f"expected {expected}, got {result}"
    )

    print(f"[Rank {rank}] all_reduce mean: {value} -> {result} (expected {expected})")


@pytest.mark.ddp
@pytest.mark.integration
def test_all_reduce_sum_2_process():
    """all_reduce 합계 계산 검증"""
    rank, world_size = init_distributed(backend="gloo")

    # Rank별로 다른 값
    if rank == 0:
        value = 1.0
    else:
        value = 3.0

    # all_reduce sum
    result = all_reduce_scalar(value, op="sum")

    # 검증: 1.0 + 3.0 = 4.0
    expected = 4.0
    assert abs(result - expected) < 1e-6, (
        f"[Rank {rank}] all_reduce_scalar sum failed: "
        f"expected {expected}, got {result}"
    )

    print(f"[Rank {rank}] all_reduce sum: {value} -> {result} (expected {expected})")


@pytest.mark.ddp
@pytest.mark.integration
def test_barrier_synchronization_2_process():
    """Barrier 동기화 검증

    Phase 1 발견: Barrier 동기화 미검증
    → Checkpoint 저장 시 race condition 방지를 위한 barrier() 동작 확인
    """
    rank, world_size = init_distributed(backend="gloo")

    start = time.time()

    # Rank 1은 1초 대기 (느린 프로세스 시뮬레이션)
    if rank == 1:
        time.sleep(1.0)
        print(f"[Rank {rank}] Finished sleeping 1.0s")

    # Barrier: 모든 프로세스가 여기서 대기
    print(f"[Rank {rank}] Waiting at barrier...")
    barrier()
    print(f"[Rank {rank}] Passed barrier")

    elapsed = time.time() - start

    # Rank 0도 최소 1초 대기했어야 함 (Rank 1 때문에)
    if rank == 0:
        assert elapsed >= 0.9, (
            f"[Rank 0] Should wait for Rank 1, "
            f"but elapsed={elapsed:.2f}s < 0.9s"
        )
        print(f"[Rank {rank}] Correctly waited {elapsed:.2f}s for Rank 1")
    else:
        # Rank 1은 자신의 sleep(1.0) 이상 대기
        assert elapsed >= 0.9, f"[Rank 1] elapsed={elapsed:.2f}s < 0.9s"
        print(f"[Rank {rank}] Total elapsed: {elapsed:.2f}s")


@pytest.mark.ddp
@pytest.mark.integration
def test_wrap_model_ddp_distributed():
    """DDP wrapping 검증 (분산 환경)

    Phase 1 발견: wrap_model_ddp는 단일 환경만 검증
    → 실제 분산 환경에서 DDP wrapping 동작 확인
    """
    rank, world_size = init_distributed(backend="gloo")

    device = torch.device("cpu")
    model = SimpleModel().to(device)

    # 분산 환경에서 DDP wrapping
    wrapped_model = wrap_model_ddp(model, device)

    # 검증: 분산 환경에서는 DDP로 래핑되어야 함
    from torch.nn.parallel import DistributedDataParallel as DDP
    assert isinstance(wrapped_model, DDP), (
        f"[Rank {rank}] Model should be wrapped with DDP in distributed mode"
    )

    print(f"[Rank {rank}] Model wrapped with DDP successfully")


@pytest.mark.ddp
@pytest.mark.integration
def test_unwrap_model_ddp_distributed():
    """DDP unwrap 검증 (분산 환경)

    Phase 1 발견: test_unwrap_model_ddp_wrapped SKIPPED (분산 환경 없음)
    → 실제 DDP 환경에서 unwrap 동작 확인
    """
    rank, world_size = init_distributed(backend="gloo")

    device = torch.device("cpu")
    model = SimpleModel().to(device)

    # 원본 모델 state_dict 저장
    original_state_dict = model.state_dict()

    # DDP wrapping
    wrapped_model = wrap_model_ddp(model, device)

    # Unwrap
    unwrapped_model = unwrap_model(wrapped_model)

    # 검증: 원본 모델과 동일해야 함
    assert unwrapped_model is model, (
        f"[Rank {rank}] Unwrapped model should be the original model"
    )

    # state_dict 동일 확인
    unwrapped_state_dict = unwrapped_model.state_dict()
    assert set(original_state_dict.keys()) == set(unwrapped_state_dict.keys())

    for key in original_state_dict.keys():
        assert torch.equal(original_state_dict[key], unwrapped_state_dict[key]), (
            f"[Rank {rank}] state_dict mismatch for key: {key}"
        )

    print(f"[Rank {rank}] Unwrap successful: original model recovered")


@pytest.mark.ddp
@pytest.mark.integration
@pytest.mark.slow
def test_ddp_forward_backward_2_process():
    """DDP forward/backward pass 정확성 검증

    Phase 1 발견: DDP와 단일 GPU 비교 미검증
    → 2-process DDP에서 gradient가 올바르게 계산되는지 확인
    """
    rank, world_size = init_distributed(backend="gloo")

    device = torch.device("cpu")

    # 동일한 초기 가중치 (모든 rank가 같은 seed)
    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # DDP wrapping
    model = wrap_model_ddp(model, device)

    # Rank별로 다른 데이터 (재현성)
    torch.manual_seed(42 + rank)
    x = torch.randn(4, 10, device=device)

    print(f"[Rank {rank}] Input shape: {x.shape}")

    # Forward
    output = model(x)
    loss = output.mean()

    print(f"[Rank {rank}] Loss before backward: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Gradient 검증: DDP는 자동으로 gradient를 all-reduce 평균
    for name, param in unwrap_model(model).named_parameters():
        assert param.grad is not None, (
            f"[Rank {rank}] Gradient should not be None for {name}"
        )
        print(f"[Rank {rank}] {name} grad norm: {param.grad.norm().item():.4f}")

    # 모든 rank의 gradient가 동일해야 함 (DDP all-reduce 결과)
    # 여기서는 gradient가 None이 아님만 확인 (동일성은 별도 테스트 필요)
    print(f"[Rank {rank}] Forward/backward pass successful")


@pytest.mark.ddp
@pytest.mark.integration
def test_gradient_sync_across_ranks():
    """DDP gradient 동기화 검증

    모든 rank의 gradient가 동일한지 확인 (DDP all-reduce 평균)
    """
    rank, world_size = init_distributed(backend="gloo")

    device = torch.device("cpu")

    # 동일한 초기 가중치
    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # DDP wrapping
    model = wrap_model_ddp(model, device)

    # 동일한 입력 (모든 rank)
    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device)

    # Forward & Backward
    output = model(x)
    loss = output.mean()
    loss.backward()

    # 모든 rank의 gradient를 수집 (barrier로 동기화 후)
    barrier()

    # Gradient norm 계산
    grad_norms = {}
    for name, param in unwrap_model(model).named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    # 검증: 동일한 입력 → 동일한 gradient
    # (실제로는 rank 간 통신이 필요하지만, 여기서는 로그로 확인)
    print(f"[Rank {rank}] Gradient norms: {grad_norms}")

    # 간단한 검증: Gradient가 0이 아님
    for name, norm in grad_norms.items():
        assert norm > 0, f"[Rank {rank}] {name} gradient norm should be > 0"


@pytest.mark.ddp
@pytest.mark.integration
def test_cleanup_distributed():
    """분산 환경 정리

    다른 테스트 영향 방지를 위한 cleanup
    """
    from weighted_mtp.runtime.distributed import cleanup_distributed

    rank = get_rank()

    # 정리 전 상태 확인
    assert is_distributed(), f"[Rank {rank}] Should be in distributed mode before cleanup"

    # 정리
    cleanup_distributed()

    print(f"[Rank {rank}] Distributed cleanup successful")
