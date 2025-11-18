"""Checkpoint 동기화 통합 테스트

Phase 2 발견사항 반영:
- Phase 2에서 barrier() 동작 검증 완료
- 이 테스트는 실제 checkpoint 저장/로드 시나리오에서 race condition 방지 검증

실행 방법:
    torchrun --nproc_per_node=2 --nnodes=1 \
        -m pytest tests/integration/test_checkpoint_sync.py -v -s

주의:
- M3 MacBook Pro에서는 CPU Gloo backend만 사용
- 실제 파이프라인과 동일한 checkpoint 저장 패턴 검증
- 공유 디렉토리 사용 (/tmp/test_checkpoint_*)
"""
import os
import time
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import shutil

from weighted_mtp.runtime import (
    init_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
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
def test_checkpoint_save_with_barrier():
    """Barrier로 checkpoint 저장 race condition 방지 검증

    실제 파이프라인 패턴:
    1. Rank 0만 checkpoint 저장
    2. barrier() 호출 (모든 rank 대기)
    3. 모든 rank가 checkpoint 접근 가능

    검증:
    - Rank 0이 저장 완료 전에 다른 rank가 접근하지 않음
    - 모든 rank가 barrier 통과 후 checkpoint 로드 가능
    """
    rank, world_size = init_distributed(backend="gloo")

    # 공유 디렉토리 (모든 rank 접근 가능)
    tmpdir = "/tmp/test_checkpoint_barrier"
    if is_main_process():
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    barrier()  # 모든 rank가 디렉토리 생성 대기

    try:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        # Rank 1은 느리게 처리 (race condition 시뮬레이션)
        if rank == 1:
            time.sleep(0.5)

        # Rank 0만 checkpoint 저장 (실제 파이프라인 패턴)
        if is_main_process():
            model = SimpleModel()
            checkpoint = {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "train_loss": 0.5,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[Rank {rank}] Checkpoint saved: {checkpoint_path.name}")

        # Barrier: 모든 프로세스가 저장 완료까지 대기
        print(f"[Rank {rank}] Waiting at barrier...")
        barrier()
        print(f"[Rank {rank}] Passed barrier")

        # 검증: 모든 프로세스가 checkpoint 접근 가능
        assert checkpoint_path.exists(), (
            f"[Rank {rank}] Checkpoint should exist after barrier"
        )

        # 로드 가능 (corruption 없음)
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["epoch"] == 1
        assert checkpoint["train_loss"] == 0.5
        assert "model_state_dict" in checkpoint

        print(f"[Rank {rank}] Checkpoint loaded successfully: epoch={checkpoint['epoch']}")

    finally:
        # Cleanup
        barrier()
        if is_main_process():
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.ddp
@pytest.mark.integration
def test_per_rank_checkpoint_save():
    """각 rank가 독립적으로 checkpoint 저장 (corruption 없음 검증)

    시나리오:
    - 각 rank가 자신만의 checkpoint 파일 저장
    - 동시 저장으로 인한 corruption 없음 검증

    검증:
    - 각 rank의 checkpoint가 올바르게 저장됨
    - 다른 rank의 checkpoint와 독립적
    """
    rank, world_size = init_distributed(backend="gloo")

    # 공유 디렉토리
    tmpdir = "/tmp/test_checkpoint_per_rank"
    if is_main_process():
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    barrier()

    try:
        checkpoint_path = Path(tmpdir) / f"checkpoint_rank_{rank}.pt"

        # 각 rank가 자신의 checkpoint 저장
        model = SimpleModel()
        torch.manual_seed(42 + rank)  # Rank별로 다른 가중치
        model.linear.weight.data.fill_(float(rank))  # Rank별로 구분 가능한 값

        checkpoint = {
            "rank": rank,
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Barrier: 모든 rank가 저장 완료까지 대기
        barrier()

        # 검증: 자신의 checkpoint 로드 가능
        loaded = torch.load(checkpoint_path)
        assert loaded["rank"] == rank
        assert "model_state_dict" in loaded

        # Weight 값 검증 (rank별로 다름)
        loaded_weight = loaded["model_state_dict"]["linear.weight"]
        assert torch.all(loaded_weight == float(rank)), (
            f"[Rank {rank}] Weight mismatch: expected {float(rank)}, "
            f"got {loaded_weight[0, 0].item()}"
        )

        print(f"[Rank {rank}] Per-rank checkpoint verified: weight={float(rank)}")

    finally:
        barrier()
        if is_main_process():
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.ddp
@pytest.mark.integration
def test_checkpoint_overwrite_safety():
    """동일 경로에 checkpoint 덮어쓰기 안전성 검증

    시나리오:
    - Rank 0만 checkpoint 저장 (실제 파이프라인 패턴)
    - 여러 번 저장 (epoch 업데이트)
    - Barrier로 동기화

    검증:
    - 마지막 저장 내용만 남음 (덮어쓰기 정상)
    - 모든 rank가 동일한 최신 checkpoint 로드
    """
    rank, world_size = init_distributed(backend="gloo")

    # 공유 디렉토리
    tmpdir = "/tmp/test_checkpoint_overwrite"
    if is_main_process():
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    barrier()

    try:
        checkpoint_path = Path(tmpdir) / "checkpoint_overwrite.pt"

        # Epoch 0 저장
        if is_main_process():
            torch.save({"epoch": 0, "loss": 1.0}, checkpoint_path)
        barrier()

        # 모든 rank가 epoch 0 확인
        ckpt = torch.load(checkpoint_path)
        assert ckpt["epoch"] == 0

        # Epoch 1 저장 (덮어쓰기)
        if is_main_process():
            torch.save({"epoch": 1, "loss": 0.5}, checkpoint_path)
        barrier()

        # 모든 rank가 epoch 1 확인 (덮어쓰기 정상)
        ckpt = torch.load(checkpoint_path)
        assert ckpt["epoch"] == 1, (
            f"[Rank {rank}] Expected epoch 1, got {ckpt['epoch']}"
        )
        assert ckpt["loss"] == 0.5

        print(f"[Rank {rank}] Overwrite verified: epoch={ckpt['epoch']}")

    finally:
        barrier()
        if is_main_process():
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.ddp
@pytest.mark.integration
def test_cleanup_distributed():
    """분산 환경 정리 (다른 테스트 영향 방지)"""
    from weighted_mtp.runtime.distributed import cleanup_distributed

    rank = get_rank()

    # 정리 전 상태 확인
    assert is_distributed(), f"[Rank {rank}] Should be in distributed mode"

    # 정리
    cleanup_distributed()

    print(f"[Rank {rank}] Checkpoint sync tests cleanup successful")
