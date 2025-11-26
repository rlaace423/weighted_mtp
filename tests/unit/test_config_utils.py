"""Config 검증 유틸리티 테스트"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf

from weighted_mtp.utils.config_utils import (
    validate_config,
    load_and_validate_config,
    ConfigValidationError,
)


class TestValidateConfigBasic:
    """기본 config 검증 테스트"""

    def test_valid_baseline_config(self, project_root: Path):
        """유효한 baseline config 검증 통과"""
        config_path = project_root / "configs" / "baseline" / "baseline_local.yaml"
        config = OmegaConf.load(config_path)

        # 검증 통과 (예외 없음)
        validate_config(config)

    def test_valid_verifiable_config(self, project_root: Path):
        """유효한 verifiable config 검증 통과"""
        config_path = project_root / "configs" / "verifiable" / "verifiable.yaml"
        config = OmegaConf.load(config_path)

        # 검증 통과 (예외 없음)
        validate_config(config)

    def test_missing_required_field(self):
        """필수 필드 누락 시 실패"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test"},
                # stage 누락
            }
        )

        with pytest.raises(ConfigValidationError, match="필수 필드 누락"):
            validate_config(config)

    def test_invalid_stage(self):
        """잘못된 stage 값"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "invalid_stage"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="잘못된 stage"):
            validate_config(config)


class TestValidateConfigValueRanges:
    """값 범위 검증 테스트"""

    def test_invalid_learning_rate_too_high(self):
        """Learning rate가 1.0 초과"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 2.0,  # 1.0 초과
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="learning_rate 범위 오류"):
            validate_config(config)

    def test_invalid_learning_rate_negative(self):
        """Learning rate가 음수"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": -0.001,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="learning_rate 범위 오류"):
            validate_config(config)

    def test_invalid_batch_size(self):
        """Batch size가 0 이하"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 0,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="batch_size는 양수여야 함"):
            validate_config(config)

    def test_invalid_n_epochs(self):
        """n_epochs가 0 이하"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": -1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="n_epochs는 양수여야 함"):
            validate_config(config)


class TestValidateConfigLogicalConsistency:
    """논리적 일관성 검증 테스트"""

    def test_batch_size_exceeds_samples(self):
        """Batch size가 샘플 수보다 큼"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 200,
                    "learning_rate": 1e-4,
                },
                "data_sampling": {
                    "n_samples": 100,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="batch_size.*n_samples"):
            validate_config(config)


class TestValidateStageSpecific:
    """Stage별 특수 검증 테스트"""

    def test_verifiable_missing_beta(self):
        """Verifiable stage에서 beta 필드 누락"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "verifiable"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    # beta 누락
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="training.beta 필드 필수"):
            validate_config(config)

    def test_verifiable_invalid_weight_clip(self):
        """Verifiable stage에서 weight_clip_min >= max"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "verifiable"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    "beta": 0.9,
                    "weight_clip_min": 5.0,
                    "weight_clip_max": 1.0,  # min > max
                },
            }
        )

        with pytest.raises(
            ConfigValidationError, match="weight_clip_min.*weight_clip_max"
        ):
            validate_config(config)

    def test_verifiable_curriculum_without_schedule(self):
        """Curriculum learning 활성화했는데 schedule 없음"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "verifiable"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    "beta": 0.9,
                },
                "data_sampling": {
                    "curriculum_learning": True,
                    # curriculum_schedule 누락
                },
            }
        )

        with pytest.raises(
            ConfigValidationError, match="curriculum_schedule 필수"
        ):
            validate_config(config)

    def test_rho1_missing_reference_model(self):
        """Rho1 stage에서 reference model 누락"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "rho1"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                # reference 누락
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(
            ConfigValidationError, match="models.reference.path 필드 필수"
        ):
            validate_config(config)



class TestLoadAndValidateConfig:
    """load_and_validate_config() 함수 테스트"""

    def test_load_valid_config(self, project_root: Path):
        """유효한 config 로드 및 검증"""
        config_path = project_root / "configs" / "baseline" / "baseline_local.yaml"
        config = load_and_validate_config(str(config_path))

        assert config.experiment.stage == "baseline"
        assert hasattr(config, "training")

    def test_load_nonexistent_config(self):
        """존재하지 않는 config 파일"""
        with pytest.raises(FileNotFoundError):
            load_and_validate_config("nonexistent_config.yaml")

    def test_load_invalid_config(self, tmp_path: Path):
        """잘못된 config 로드"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text(
            """
experiment:
  name: test
  stage: invalid_stage
"""
        )

        with pytest.raises(ConfigValidationError):
            load_and_validate_config(str(config_file))


class TestValidateConfigPathChecks:
    """경로 존재 확인 테스트 (통합 테스트)"""

    def test_nonexistent_model_path(self):
        """존재하지 않는 모델 경로 (디렉토리)"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "/nonexistent/model/path"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="모델 경로가 존재하지 않음"):
            validate_config(config)

    def test_checkpoint_path_skipped(self):
        """Checkpoint 경로는 검증 건너뜀"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "verifiable"},
                "models": {"policy": {"path": "/nonexistent/checkpoint.pt"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "trunk_learning_rate": 1e-5,
                    "value_head_learning_rate": 3e-5,
                    "beta": 0.2,
                    "value_coef": 0.5,
                    "weight_clip_min": 0.1,
                    "weight_clip_max": 3.0,
                },
                "data_sampling": {
                    "n_samples": 100,
                    "correct_ratio": 0.5,
                },
            }
        )

        # .pt 파일은 검증 건너뛰므로 "모델 경로" 에러 없음
        validate_config(config)

    def test_nonexistent_dataset_path(self):
        """존재하지 않는 데이터셋 경로"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "/nonexistent/train.jsonl",
                    "validation": "/nonexistent/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(
            ConfigValidationError, match="데이터셋 파일이 존재하지 않음"
        ):
            validate_config(config)
