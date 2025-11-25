"""S3 checkpoint 관리 유틸리티

비동기 업로드 및 정리 기능 제공
"""

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# 전역 executor (파이프라인 초기화 시 생성)
s3_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")


def upload_to_s3_async(checkpoint_path: Path, run_id: str) -> None:
    """임시 복사본을 생성하여 S3에 안전하게 업로드

    원본 파일을 임시 디렉터리에 복사한 후 업로드하여
    업로드 중 원본 파일 삭제로 인한 race condition 방지

    MlflowClient를 사용하여 별도 스레드에서도 artifact 업로드 가능
    (mlflow.log_artifact()는 메인 스레드의 active_run 컨텍스트 필요)

    Args:
        checkpoint_path: 업로드할 checkpoint 경로
        run_id: MLflow run ID (스레드에서 사용하기 위해 명시적 전달)

    Note:
        임시 복사본은 업로드 완료 후 자동 삭제
        원본 파일과 완전히 독립적으로 동작
        S3에는 원본 파일명으로 저장됨
    """
    if not run_id:
        logger.warning(f"S3 upload skipped (no run_id): {checkpoint_path.name}")
        return

    tmp_dir = None
    try:
        # 임시 디렉터리 생성
        tmp_dir = tempfile.TemporaryDirectory(prefix="checkpoint_upload_")
        tmp_dir_path = Path(tmp_dir.name)

        # 원본 파일명 유지하여 복사
        tmp_checkpoint = tmp_dir_path / checkpoint_path.name
        shutil.copy2(checkpoint_path, tmp_checkpoint)
        logger.debug(f"Created temp copy for upload: {tmp_checkpoint}")

        # MlflowClient로 명시적 run_id 사용 (스레드 안전)
        client = MlflowClient()
        client.log_artifact(run_id, str(tmp_checkpoint), artifact_path="checkpoints")

        logger.info(f"S3 upload complete: {checkpoint_path.name} -> run_id={run_id}")

    except Exception as e:
        logger.error(f"S3 upload failed: {checkpoint_path.name} - {e}")

    finally:
        # 임시 디렉터리 정리
        if tmp_dir is not None:
            try:
                tmp_dir.cleanup()
            except Exception:
                pass  # cleanup 실패는 무시


def cleanup_s3_checkpoints(
    experiment_id: str,
    run_id: str,
    save_total_limit: int,
) -> None:
    """S3에서 오래된 checkpoint 삭제

    MLflow artifact store (S3)에서 checkpoint_epoch_*.pt 파일만 정리
    checkpoint_best.pt와 checkpoint_final.pt는 유지

    Args:
        experiment_id: MLflow experiment ID (사용되지 않음, 호환성 유지)
        run_id: MLflow run ID
        save_total_limit: 유지할 최대 개수
    """
    try:
        s3 = boto3.client("s3")
        bucket = "wmtp"
        # MLflow artifact URI 구조: s3://wmtp/mlflow-artifacts/{run_id}/artifacts/
        prefix = f"mlflow-artifacts/{run_id}/artifacts/checkpoints/"

        # S3에서 checkpoint 목록 직접 조회
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            return  # checkpoint가 없으면 종료

        # checkpoint_epoch_*.pt 파일만 필터링
        epoch_checkpoints = []
        for obj in response["Contents"]:
            key = obj["Key"]
            filename = key.split("/")[-1]
            if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
                epoch_checkpoints.append({"key": key, "last_modified": obj["LastModified"]})

        # 시간순 정렬 (LastModified 기준)
        epoch_checkpoints.sort(key=lambda x: x["last_modified"])

        # 삭제할 파일 개수 계산
        n_to_delete = len(epoch_checkpoints) - save_total_limit

        if n_to_delete > 0:
            for checkpoint in epoch_checkpoints[:n_to_delete]:
                s3.delete_object(Bucket=bucket, Key=checkpoint["key"])
                filename = checkpoint["key"].split("/")[-1]
                logger.info(f"S3 checkpoint deleted: {filename}")

    except Exception as e:
        logger.warning(f"S3 cleanup failed: {e}")


def shutdown_s3_executor() -> None:
    """모든 S3 업로드 완료 대기

    학습 종료 시 호출하여 모든 비동기 업로드 완료를 보장
    """
    s3_upload_executor.shutdown(wait=True)
    logger.info("All S3 uploads completed")


def reset_s3_executor() -> None:
    """S3 executor 재생성

    테스트 격리를 위해 shutdown 후 executor 재생성
    """
    global s3_upload_executor
    s3_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")
