"""checkpoint_utils Unit Tests"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from weighted_mtp.pipelines.checkpoint_utils import (
    load_critic_checkpoint,
    save_checkpoint,
)


class MockValueHead(nn.Module):
    """테스트용 Mock Value Head"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


class MockAdapter:
    """테스트용 Mock Adapter"""

    def __init__(self):
        self.value_head = MockValueHead()
        self.mtp_head = nn.Linear(4, 100)  # Mock MTP head

    def state_dict(self):
        """전체 adapter state dict"""
        return {
            "value_head": self.value_head.state_dict(),
            "mtp_head": self.mtp_head.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """State dict 로드"""
        self.value_head.load_state_dict(state_dict["value_head"])
        self.mtp_head.load_state_dict(state_dict["mtp_head"])


class TestSaveCheckpoint:
    """save_checkpoint() 테스트"""

    def test_basic_save(self, tmp_path):
        """기본 저장 동작 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-4)

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
        assert "value_head_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["train_metrics"] == train_metrics
        assert checkpoint["val_metrics"] == val_metrics

    def test_fractional_epoch(self, tmp_path):
        """Fractional epoch 저장 지원 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-4)

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
        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-4)

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

    def test_value_head_state_dict_separation(self, tmp_path):
        """value_head_state_dict 분리 저장 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-4)

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

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # adapter_state_dict와 value_head_state_dict 모두 존재
        assert "adapter_state_dict" in checkpoint
        assert "value_head_state_dict" in checkpoint

        # value_head_state_dict는 value_head만 포함
        value_head_keys = checkpoint["value_head_state_dict"].keys()
        assert any("linear" in k for k in value_head_keys)

    def test_string_path(self, tmp_path):
        """String path 지원 검증"""
        adapter = MockAdapter()
        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-4)

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


class TestLoadCriticCheckpoint:
    """load_critic_checkpoint() 테스트"""

    def test_local_path_load(self, tmp_path):
        """Local path 로드 검증"""
        # Checkpoint 저장
        adapter_save = MockAdapter()
        optimizer = torch.optim.Adam(adapter_save.value_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            adapter=adapter_save,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # 새로운 adapter에 로드
        adapter_load = MockAdapter()
        device = torch.device("cpu")

        checkpoint = load_critic_checkpoint(
            checkpoint_path=str(checkpoint_path),
            adapter=adapter_load,
            device=device,
        )

        # Checkpoint 내용 검증
        assert checkpoint["epoch"] == 1
        assert checkpoint["val_metrics"]["val_loss"] == 0.3

        # Value head state dict 로드 확인
        # (파라미터가 동일한지는 state_dict 비교로 확인)
        original_state = adapter_save.value_head.state_dict()
        loaded_state = adapter_load.value_head.state_dict()

        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])

    def test_file_not_found(self):
        """존재하지 않는 파일 로드 시 에러 검증"""
        adapter = MockAdapter()
        device = torch.device("cpu")

        with pytest.raises(FileNotFoundError, match="Checkpoint 파일이 존재하지 않습니다"):
            load_critic_checkpoint(
                checkpoint_path="/nonexistent/path/checkpoint.pt",
                adapter=adapter,
                device=device,
            )

    def test_missing_value_head_state_dict(self, tmp_path):
        """value_head_state_dict 없는 checkpoint 로드 시 에러"""
        # 잘못된 형식의 checkpoint 저장
        checkpoint_path = tmp_path / "bad_checkpoint.pt"
        bad_checkpoint = {
            "epoch": 1,
            "adapter_state_dict": {"some": "data"},
            # value_head_state_dict 없음
        }
        torch.save(bad_checkpoint, checkpoint_path)

        adapter = MockAdapter()
        device = torch.device("cpu")

        with pytest.raises(KeyError, match="value_head_state_dict"):
            load_critic_checkpoint(
                checkpoint_path=str(checkpoint_path),
                adapter=adapter,
                device=device,
            )

    @patch("mlflow.artifacts.download_artifacts")
    def test_mlflow_uri_load(self, mock_download, tmp_path):
        """MLflow artifact URI 로드 검증 (mock)"""
        # Local checkpoint 준비
        adapter_save = MockAdapter()
        optimizer = torch.optim.Adam(adapter_save.value_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        local_checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            adapter=adapter_save,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=local_checkpoint_path,
        )

        # MLflow artifact download mock 설정
        mock_download.return_value = str(local_checkpoint_path)

        # MLflow URI로 로드
        adapter_load = MockAdapter()
        device = torch.device("cpu")
        mlflow_uri = "mlflow://8/abc123/artifacts/checkpoints/checkpoint_best.pt"

        checkpoint = load_critic_checkpoint(
            checkpoint_path=mlflow_uri,
            adapter=adapter_load,
            device=device,
        )

        # MLflow download_artifacts 호출 확인
        mock_download.assert_called_once_with(mlflow_uri)

        # Checkpoint 로드 확인
        assert checkpoint["epoch"] == 1
        assert checkpoint["val_metrics"]["val_loss"] == 0.3

    def test_device_mapping(self, tmp_path):
        """Device mapping 지원 검증"""
        # CPU에서 checkpoint 저장
        adapter_save = MockAdapter()
        optimizer = torch.optim.Adam(adapter_save.value_head.parameters(), lr=1e-4)

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.3}
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            adapter=adapter_save,
            optimizer=optimizer,
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
        )

        # CPU device로 로드 (CUDA 없는 환경에서도 동작)
        adapter_load = MockAdapter()
        device = torch.device("cpu")

        checkpoint = load_critic_checkpoint(
            checkpoint_path=str(checkpoint_path),
            adapter=adapter_load,
            device=device,
        )

        # 정상 로드 확인
        assert checkpoint["epoch"] == 1


class TestCheckpointIntegration:
    """save + load 통합 테스트"""

    def test_save_and_load_cycle(self, tmp_path):
        """저장 → 로드 사이클 검증"""
        # 1. 초기 adapter 생성 및 학습 시뮬레이션
        adapter_original = MockAdapter()
        optimizer = torch.optim.Adam(adapter_original.value_head.parameters(), lr=1e-4)

        # 임의의 학습 수행 (파라미터 변경)
        dummy_input = torch.randn(2, 4)
        loss = adapter_original.value_head(dummy_input).sum()
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

        # 3. 새로운 adapter 생성 (초기 상태)
        adapter_new = MockAdapter()

        # 초기 상태는 다름
        original_params = list(adapter_original.value_head.parameters())[0].data
        new_params_before = list(adapter_new.value_head.parameters())[0].data
        assert not torch.allclose(original_params, new_params_before)

        # 4. Checkpoint 로드
        device = torch.device("cpu")
        load_critic_checkpoint(
            checkpoint_path=str(checkpoint_path),
            adapter=adapter_new,
            device=device,
        )

        # 5. 로드 후 파라미터 일치 확인
        new_params_after = list(adapter_new.value_head.parameters())[0].data
        assert torch.allclose(original_params, new_params_after)
