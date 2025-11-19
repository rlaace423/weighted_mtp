"""Config 로딩 테스트"""

import pytest
from pathlib import Path
import yaml


def test_load_defaults_config(project_root: Path):
    """defaults.yaml 로딩"""
    config_path = project_root / "configs" / "defaults.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "project" in config
    assert config["project"]["name"] == "weighted-mtp"
    assert config["models"]["policy"]["name"] == "meta-llama-mtp"


@pytest.mark.parametrize(
    "stage,config_name",
    [
        ("baseline", "baseline.yaml"),
        ("verifiable", "verifiable.yaml"),
        ("rho1", "rho1.yaml"),
    ],
)
def test_load_stage_configs(project_root: Path, stage: str, config_name: str):
    """Stage config YAML 로딩 (디렉터리 구조)"""
    config_path = project_root / "configs" / stage / config_name
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "experiment" in config
    assert "dataset" in config
    assert "training" in config
