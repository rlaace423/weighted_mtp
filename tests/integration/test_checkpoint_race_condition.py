"""Checkpoint Race Condition 통합 테스트

save_total_limit 설정 시 S3 업로드 중 로컬 checkpoint 삭제로 인한
race condition이 발생하지 않는지 검증
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from weighted_mtp.utils import (
    cleanup_old_checkpoints,
    save_checkpoint,
    upload_to_s3_async,
)
from weighted_mtp.utils import s3_utils


def test_race_condition_prevented(tmp_path):
    """save_total_limit=1 설정 시 race condition 미발생 확인

    시나리오:
    1. 첫 번째 checkpoint 저장
    2. S3 업로드 시작 (비동기)
    3. 두 번째 checkpoint 저장 (더 좋음)
    4. cleanup 즉시 실행 → 첫 번째 checkpoint 삭제
    5. S3 업로드는 예외 없이 완료되어야 함 (임시 복사본 사용)
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Dummy model 생성
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())

    # 첫 번째 checkpoint 저장
    checkpoint1 = checkpoint_dir / "checkpoint_epoch_1.00.pt"
    save_checkpoint(
        adapter=model,
        optimizer=optimizer,
        epoch=1.0,
        train_metrics={"loss": 2.5},
        val_metrics={"val_loss": 2.5},
        checkpoint_path=checkpoint1,
    )

    assert checkpoint1.exists(), "첫 번째 checkpoint가 저장되어야 함"

    upload_completed = False
    upload_error = None

    def slow_upload_mock(path, artifact_path):
        """느린 업로드 시뮬레이션 (파일 읽기)"""
        nonlocal upload_completed, upload_error
        try:
            # 업로드 시뮬레이션
            time.sleep(0.2)  # 업로드 시간 시뮬레이션
            with open(path, 'rb') as f:
                f.read()
            upload_completed = True
        except Exception as e:
            upload_error = e

    # mlflow mock
    with patch("weighted_mtp.utils.s3_utils.mlflow.log_artifact", side_effect=slow_upload_mock):

        # S3 업로드 시작 (비동기)
        upload_future = s3_utils.s3_upload_executor.submit(
            upload_to_s3_async, checkpoint1, enabled=True
        )

        # 짧은 대기 (업로드 시작 확인)
        time.sleep(0.1)

        # 두 번째 checkpoint 저장 (더 좋음)
        checkpoint2 = checkpoint_dir / "checkpoint_epoch_1.50.pt"
        save_checkpoint(
            adapter=model,
            optimizer=optimizer,
            epoch=1.5,
            train_metrics={"loss": 2.3},
            val_metrics={"val_loss": 2.3},
            checkpoint_path=checkpoint2,
        )

        assert checkpoint2.exists(), "두 번째 checkpoint가 저장되어야 함"

        # Cleanup 즉시 실행 (save_total_limit=1)
        cleanup_old_checkpoints(checkpoint_dir, save_total_limit=1)

        # 첫 번째 checkpoint가 삭제되었어야 함
        assert not checkpoint1.exists(), "첫 번째 checkpoint가 삭제되어야 함"
        assert checkpoint2.exists(), "두 번째 checkpoint는 유지되어야 함"

        # S3 업로드는 예외 없이 완료되어야 함
        try:
            upload_future.result(timeout=10)
        except Exception as e:
            pytest.fail(f"S3 upload failed: {e}")

    # 검증
    assert upload_completed, "업로드가 완료되어야 함"
    assert upload_error is None, f"업로드 에러 발생: {upload_error}"


def test_multiple_checkpoints_race_condition(tmp_path):
    """여러 checkpoint가 빠르게 생성될 때 race condition 미발생 확인"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoints = []
    upload_futures = []

    # 간단한 mock (실제 업로드 시뮬레이션)
    def simple_upload_mock(path, artifact_path):
        time.sleep(0.05)
        with open(path, 'rb') as f:
            f.read()

    # mlflow mock
    with patch("weighted_mtp.utils.s3_utils.mlflow.log_artifact", side_effect=simple_upload_mock):

        # 5개 checkpoint 빠르게 생성
        for i in range(5):
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{i+1}.00.pt"
            save_checkpoint(
                adapter=model,
                optimizer=optimizer,
                epoch=float(i+1),
                train_metrics={"loss": 2.0 - i*0.1},
                val_metrics={"val_loss": 2.0 - i*0.1},
                checkpoint_path=checkpoint_path,
            )
            checkpoints.append(checkpoint_path)

            # S3 업로드 시작 (비동기)
            future = s3_utils.s3_upload_executor.submit(
                upload_to_s3_async, checkpoint_path, enabled=True
            )
            upload_futures.append(future)

            # Cleanup (save_total_limit=2, 최신 2개만 유지)
            cleanup_old_checkpoints(checkpoint_dir, save_total_limit=2)

        # 모든 업로드 완료 대기
        for future in upload_futures:
            try:
                future.result(timeout=10)
            except Exception as e:
                pytest.fail(f"S3 upload failed: {e}")

    # 검증: 최신 2개만 남아있어야 함
    remaining = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert len(remaining) == 2, f"2개만 남아있어야 하는데 {len(remaining)}개 있음"
    assert remaining[0].name == "checkpoint_epoch_4.00.pt"
    assert remaining[1].name == "checkpoint_epoch_5.00.pt"
