"""pytest 공통 fixture"""

import pytest
import sys
from pathlib import Path

from weighted_mtp.utils import s3_utils


@pytest.fixture(scope="session", autouse=True)
def setup_vendor_path():
    """vendor 모듈 접근을 위한 PYTHONPATH 설정

    vendor/는 프로젝트 루트에 위치하므로 PYTHONPATH에 추가 필요
    """
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root() -> Path:
    """프로젝트 루트 경로"""
    return Path(__file__).parent.parent


@pytest.fixture
def storage_root(project_root: Path) -> Path:
    """storage/ 경로"""
    return project_root / "storage"


@pytest.fixture
def micro_model_path(storage_root: Path) -> Path:
    """Micro MTP 모델 경로"""
    return storage_root / "models/micro-mtp"


@pytest.fixture(autouse=True)
def ensure_s3_executor_available():
    """각 테스트 전에 s3_upload_executor가 사용 가능한지 확인

    이전 테스트에서 shutdown된 경우 재생성
    """
    if s3_utils.s3_upload_executor._shutdown:
        s3_utils.reset_s3_executor()
