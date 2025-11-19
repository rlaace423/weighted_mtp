"""S3 유틸리티 단위 테스트"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from weighted_mtp.utils import (
    cleanup_s3_checkpoints,
    shutdown_s3_executor,
    upload_to_s3_async,
)


@pytest.fixture
def temp_checkpoint(tmp_path):
    """임시 checkpoint 파일 생성"""
    checkpoint_path = tmp_path / "checkpoint_epoch_1.00.pt"
    checkpoint_path.write_text("dummy checkpoint data")
    return checkpoint_path


def test_upload_to_s3_async_disabled(temp_checkpoint):
    """MLflow 비활성화 시 업로드 스킵"""
    # MLflow 비활성화
    upload_to_s3_async(temp_checkpoint, mlflow_enabled=False)
    # 에러 없이 완료되면 성공


def test_upload_to_s3_async_success(temp_checkpoint):
    """임시 복사본을 통한 S3 업로드 성공"""
    # tempfile mock 설정
    mock_tmp_instance = MagicMock()
    mock_tmp_instance.name = "/tmp/checkpoint_upload_test"
    mock_tmp_instance.cleanup = MagicMock()

    with patch("tempfile.TemporaryDirectory", return_value=mock_tmp_instance):
        with patch("shutil.copy2") as mock_copy:
            with patch("weighted_mtp.utils.s3_utils.mlflow") as mock_mlflow:
                # 업로드 실행
                upload_to_s3_async(temp_checkpoint, mlflow_enabled=True)

                # 복사 호출 확인
                mock_copy.assert_called_once()

                # MLflow artifact 로깅 호출 확인
                mock_mlflow.log_artifact.assert_called_once()

                # cleanup 호출 확인
                mock_tmp_instance.cleanup.assert_called_once()


def test_upload_to_s3_async_failure(temp_checkpoint, caplog):
    """S3 업로드 실패 케이스"""
    with patch("weighted_mtp.utils.s3_utils.mlflow") as mock_mlflow:
        mock_mlflow.log_artifact.side_effect = Exception("S3 connection error")

        # 업로드 실행 (예외는 캐치되어야 함)
        upload_to_s3_async(temp_checkpoint, mlflow_enabled=True)

        # 에러 로그 확인
        assert "S3 upload failed" in caplog.text
        assert "S3 connection error" in caplog.text


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_success(mock_boto3):
    """S3 checkpoint 정리 성공 케이스"""
    # Mock S3 클라이언트 설정
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    # S3 list_objects_v2 응답 mock (5개 checkpoint)
    from datetime import datetime, timedelta
    base_time = datetime(2025, 1, 1)

    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_1.00.pt",
             "LastModified": base_time + timedelta(hours=1)},
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_2.00.pt",
             "LastModified": base_time + timedelta(hours=2)},
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_3.00.pt",
             "LastModified": base_time + timedelta(hours=3)},
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_4.00.pt",
             "LastModified": base_time + timedelta(hours=4)},
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_5.00.pt",
             "LastModified": base_time + timedelta(hours=5)},
        ]
    }

    # S3 정리 실행 (최대 2개 유지 -> 3개 삭제)
    cleanup_s3_checkpoints(
        experiment_id="1",
        run_id="abc123",
        save_total_limit=2,
    )

    # S3 삭제 호출 확인 (가장 오래된 3개)
    assert mock_s3.delete_object.call_count == 3

    # 삭제된 파일 확인
    deleted_keys = [call.kwargs["Key"] for call in mock_s3.delete_object.call_args_list]
    assert "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_1.00.pt" in deleted_keys
    assert "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_2.00.pt" in deleted_keys
    assert "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_3.00.pt" in deleted_keys


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_no_deletion(mock_boto3):
    """삭제할 checkpoint가 없는 경우"""
    # Mock S3 클라이언트 설정
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    # S3 list_objects_v2 응답 mock (2개만 존재)
    from datetime import datetime, timedelta
    base_time = datetime(2025, 1, 1)

    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_4.00.pt",
             "LastModified": base_time + timedelta(hours=4)},
            {"Key": "mlflow-artifacts/abc123/artifacts/checkpoints/checkpoint_epoch_5.00.pt",
             "LastModified": base_time + timedelta(hours=5)},
        ]
    }

    # S3 정리 실행 (최대 3개 유지 -> 삭제 없음)
    cleanup_s3_checkpoints(
        experiment_id="1",
        run_id="abc123",
        save_total_limit=3,
    )

    # S3 삭제 호출되지 않음
    mock_s3.delete_object.assert_not_called()


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_failure(mock_boto3, caplog):
    """S3 정리 실패 케이스"""
    # S3 클라이언트 실패 시뮬레이션
    mock_boto3.client.side_effect = Exception("S3 connection error")

    # S3 정리 실행 (예외는 캐치되어야 함)
    cleanup_s3_checkpoints(
        experiment_id="1",
        run_id="abc123",
        save_total_limit=2,
    )

    # 경고 로그 확인
    assert "S3 cleanup failed" in caplog.text
    assert "S3 connection error" in caplog.text


def test_shutdown_s3_executor():
    """S3 executor shutdown 테스트"""
    # Executor shutdown 실행
    shutdown_s3_executor()

    # 에러 없이 완료되면 성공
    # (실제로는 전역 executor가 shutdown되지만, 테스트 격리를 위해 검증 생략)


def test_upload_cleans_temp_on_error(temp_checkpoint):
    """업로드 실패 시에도 임시 디렉터리 정리 확인"""
    # tempfile mock 설정
    mock_tmp_instance = MagicMock()
    mock_tmp_instance.name = "/tmp/checkpoint_upload_test"
    mock_tmp_instance.cleanup = MagicMock()

    with patch("tempfile.TemporaryDirectory", return_value=mock_tmp_instance):
        with patch("weighted_mtp.utils.s3_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_artifact.side_effect = Exception("S3 error")

            # 업로드 실행 (예외는 캐치되어야 함)
            upload_to_s3_async(temp_checkpoint, mlflow_enabled=True)

            # cleanup 호출 확인 (에러 발생 시에도)
            mock_tmp_instance.cleanup.assert_called_once()


def test_original_file_deletable_during_upload(tmp_path):
    """업로드 중 원본 파일 삭제 가능 확인 (핵심 race condition 방지 테스트)"""
    import time
    from concurrent.futures import ThreadPoolExecutor

    # 실제 임시 checkpoint 파일 생성
    checkpoint_path = tmp_path / "checkpoint_epoch_1.00.pt"
    checkpoint_path.write_bytes(b"test checkpoint data" * 1000)

    upload_completed = False
    upload_error = None

    def slow_upload_mock(path, artifact_path):
        """업로드를 시뮬레이션하는 느린 함수"""
        nonlocal upload_completed, upload_error
        try:
            # 업로드 시뮬레이션 (파일 읽기)
            time.sleep(0.1)
            with open(path, 'rb') as f:
                f.read()
            upload_completed = True
        except Exception as e:
            upload_error = e

    # mlflow.log_artifact를 slow mock으로 교체
    with patch("weighted_mtp.utils.s3_utils.mlflow") as mock_mlflow:
        mock_mlflow.log_artifact.side_effect = slow_upload_mock

        # 백그라운드에서 업로드 시작
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(upload_to_s3_async, checkpoint_path, True)

        # 짧은 대기 (업로드 시작 확인)
        time.sleep(0.05)

        # 원본 파일 삭제 시도 (업로드 진행 중)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # 업로드 완료 대기
        future.result(timeout=5)
        executor.shutdown(wait=True)

    # 검증: 원본 파일은 삭제되었어야 함
    assert not checkpoint_path.exists(), "원본 파일이 삭제되어야 함"

    # 검증: 업로드는 성공했어야 함 (임시 복사본 사용)
    assert upload_completed, "업로드가 완료되어야 함"
    assert upload_error is None, f"업로드 에러 발생: {upload_error}"
