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
    "recipe_name",
    [
        "recipe.baseline.yaml",
        "recipe.verifiable.yaml",
        "recipe.rho1_weighted.yaml",
    ],
)
def test_load_recipe_configs(project_root: Path, recipe_name: str):
    """Recipe YAML 로딩"""
    recipe_path = project_root / "configs" / recipe_name
    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)

    assert "experiment" in recipe
    assert "dataset" in recipe
    assert "training" in recipe
