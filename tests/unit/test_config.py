"""Config 로딩 테스트"""

import pytest
from pathlib import Path
import yaml


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/production/baseline.yaml",
        "configs/production/verifiable_pairwise.yaml",
        "configs/production/rho1.yaml",
    ],
)
def test_load_stage_configs(project_root: Path, config_path: str):
    """Stage config YAML 로딩 (production 디렉터리 구조)"""
    full_path = project_root / config_path
    with open(full_path) as f:
        config = yaml.safe_load(f)

    assert "experiment" in config
    assert "dataset" in config
    assert "training" in config
