"""pytest 공통 fixture"""

import pytest
from pathlib import Path


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
    return storage_root / "models_v2" / "micro-mtp"
