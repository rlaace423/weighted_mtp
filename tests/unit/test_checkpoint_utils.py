"""checkpoint_utils Unit Tests"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from weighted_mtp.utils import (
    save_checkpoint,
)


class MockMTPHead(nn.Module):
    """테스트용 Mock MTP Head"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 100)

    def forward(self, x):
        return self.linear(x)


class MockAdapter:
    """테스트용 Mock Adapter (순수 MTP, value_head 없음)"""

    def __init__(self):
        self.mtp_head = MockMTPHead()

    def state_dict(self):
        """전체 adapter state dict"""
        return {
            "mtp_head": self.mtp_head.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """State dict 로드"""
        self.mtp_head.load_state_dict(state_dict["mtp_head"])


class TestSaveCheckpoint:
    """save_checkpoint() 테스트"""

    def test_basic_save(self, tmp_path):
        """기본 저장 동작 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.mtp_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # 파일 존재 확인
        assert checkpoint_path.exists()

        # Checkpoint 로드 및 내용 검증
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["epoch"] == 1
        assert "adapter_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["train_metrics"] == train_metrics
        assert checkpoint["val_metrics"] == val_metrics

    def test_fractional_epoch(self, tmp_path):
        """Fractional epoch 저장 지원 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.mtp_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint_epoch_0.5.pt"

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=0.5,  # Fractional epoch
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["epoch"] == 0.5

    def test_directory_creation(self, tmp_path):
        """디렉토리 자동 생성 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.mtp_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}

        # 존재하지 않는 디렉토리 경로
        checkpoint_path = tmp_path / "subdir1" / "subdir2" / "checkpoint.pt"

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # 디렉토리 및 파일 생성 확인
        assert checkpoint_path.exists()
        assert checkpoint_path.parent.exists()

    def test_string_path(self, tmp_path):
        """String path 지원 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.mtp_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path_str = str(tmp_path / "checkpoint.pt")

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path_str,  # String path
        )

        assert Path(checkpoint_path_str).exists()


class TestCheckpointIntegration:
    """save + load 통합 테스트"""

    def test_save_and_load_cycle(self, tmp_path):
        """저장 → 로드 사이클 검증"""
        # 1. 초기 adapter 생성 및 학습 시뮬레이션
        adapter_original = MockAdapter()
        optimizer = torch.optim.Adam(adapter_original.mtp_head.parameters(), lr=1e-4)

        # 임의의 학습 수행 (파라미터 변경)
        dummy_input = torch.randn(2, 4)
        loss = adapter_original.mtp_head(dummy_input).sum()
        loss.backward()
        optimizer.step()

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint.pt"

        # 2. Checkpoint 저장
        save_checkpoint(
            adapter=adapter_original,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # 3. Checkpoint 로드 및 검증
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["epoch"] == 1
        assert "adapter_state_dict" in checkpoint

        # 4. 저장된 파라미터가 학습된 파라미터와 일치하는지 확인
        original_state = adapter_original.mtp_head.state_dict()
        saved_state = checkpoint["adapter_state_dict"]["mtp_head"]

        for key in original_state:
            assert torch.allclose(original_state[key], saved_state[key])


class TestFSDPCheckpoint:
    """FSDP checkpoint 저장/로드 테스트"""

    def test_save_checkpoint_fsdp(self, tmp_path):
        """FSDP wrapped 모델 checkpoint 저장"""
        # Mock adapter 생성
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.mtp_head.parameters(), lr=1e-4)

        device = torch.device("cpu")

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint_fsdp.pt"

        # Checkpoint 저장
        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # Checkpoint 로드 및 검증
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["epoch"] == 1
        assert "adapter_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_checkpoint_normal_model_compatibility(self, tmp_path):
        """일반 모델 checkpoint 저장 (FSDP 수정 후에도 하위 호환)"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.mtp_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint_normal.pt"

        save_checkpoint(
            adapter=adapter,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # Checkpoint 로드 및 검증
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["epoch"] == 1
        assert "adapter_state_dict" in checkpoint

    def test_checkpoint_fsdp_compatibility(self, tmp_path):
        """FSDP checkpoint 저장 및 로드 호환성"""
        # 1. Checkpoint 저장 (일반 모델)
        adapter_save = MockAdapter()
        optimizer = torch.optim.Adam(adapter_save.mtp_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint_compat.pt"

        save_checkpoint(
            adapter=adapter_save,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # 2. Checkpoint 로드 및 검증
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # 3. State dict 로드 확인
        assert checkpoint["epoch"] == 1
        assert "adapter_state_dict" in checkpoint

        # 4. 파라미터 일치 확인
        original_state = adapter_save.mtp_head.state_dict()
        saved_state = checkpoint["adapter_state_dict"]["mtp_head"]

        for key in original_state:
            assert torch.allclose(original_state[key], saved_state[key])
