"""Verifiable Pipeline Integration Test (MPS + micro-mtp)

M3 Mac MPS 환경에서 micro-mtp 모델로 verifiable 파이프라인 검증
- TD error-based token weighting
- Critic checkpoint 로드 (models.policy.path로 직접 지정)
- 20 samples, 0.02 epochs
"""

import pytest
import torch
from pathlib import Path

from omegaconf import OmegaConf
from weighted_mtp.pipelines.run_verifiable import run_verifiable_training


@pytest.mark.integration
@pytest.mark.slow
def test_verifiable_pipeline_micro_mtp():
    """Verifiable 파이프라인 end-to-end 테스트 (micro-mtp + MPS)"""

    # MPS 사용 가능 여부 확인
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    # Critic checkpoint 존재 확인 (models.policy.path로 사용)
    critic_checkpoint_path = Path("storage/checkpoints/critic/critic-pretrain-local/checkpoint_final.pt")
    if not critic_checkpoint_path.exists():
        pytest.skip(f"Critic checkpoint not found: {critic_checkpoint_path}. Run critic training first.")

    # Test config 경로
    config_path = "configs/verifiable/verifiable_local.yaml"
    assert Path(config_path).exists(), f"Config not found: {config_path}"

    # Override 파라미터 (초경량 테스트 설정)
    override_params = {
        "data_sampling": {
            "n_samples": 20,  # 20개 샘플
            "curriculum_learning": False,  # 테스트 간소화
        },
        "training": {
            "n_epochs": 1.5,  # 1.5 epoch (30 samples total)
            "batch_size": 2,
            "log_interval": 1,
        },
        "checkpoint": {
            "save_checkpoint_every": 1.5,
            "save_dir": "storage/checkpoints/verifiable/test-verifiable-integration",
        },
        "experiment": {
            "name": "test-verifiable-integration",
        },
        "mlflow": {
            "tracking_uri": "",
            "experiment": "",
        },
    }

    # Config 로드
    config = OmegaConf.load(config_path)

    # Overrides 적용
    override_config = OmegaConf.create(override_params)
    config = OmegaConf.merge(config, override_config)

    # 실행
    try:
        final_metrics, best_checkpoint_path = run_verifiable_training(config)

        # 검증
        assert final_metrics is not None, "Final metrics should not be None"
        assert "val_loss" in final_metrics, "Should have val_loss"
        assert "val_weighted_ce_loss" in final_metrics, "Should have weighted CE loss"
        assert "val_value_loss" in final_metrics, "Should have value loss"

        assert isinstance(final_metrics["val_loss"], float), "val_loss should be float"
        assert final_metrics["val_loss"] > 0, "val_loss should be positive"

        # Checkpoint 생성 확인
        if best_checkpoint_path:
            assert Path(best_checkpoint_path).exists(), f"Best checkpoint should exist: {best_checkpoint_path}"

        checkpoint_dir = Path(override_params["checkpoint"]["save_dir"])
        assert checkpoint_dir.exists(), "Checkpoint directory should exist"

        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0, "At least one checkpoint should be saved"

        print(f"\n✓ Verifiable pipeline test passed")
        print(f"  Final val_loss: {final_metrics['val_loss']:.4f}")
        print(f"  Final weighted CE loss: {final_metrics['val_weighted_ce_loss']:.4f}")
        print(f"  Final value loss: {final_metrics['val_value_loss']:.4f}")
        print(f"  Checkpoints saved: {len(checkpoints)}")
        print(f"  Best checkpoint: {best_checkpoint_path}")

    finally:
        # Cleanup: test checkpoints 삭제
        import shutil
        checkpoint_dir = Path(override_params["checkpoint"]["save_dir"])
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"  Cleaned up test checkpoints")


@pytest.mark.integration
def test_verifiable_config_validation():
    """Verifiable config 파일 유효성 검증"""
    config_path = Path("configs/verifiable/verifiable_local.yaml")
    assert config_path.exists(), f"Config not found: {config_path}"

    config = OmegaConf.load(str(config_path))

    # 필수 필드 검증
    assert hasattr(config, "experiment"), "Config should have experiment"
    assert hasattr(config, "models"), "Config should have models"
    assert hasattr(config, "dataset"), "Config should have dataset"
    assert hasattr(config, "data_sampling"), "Config should have data_sampling"
    assert hasattr(config, "training"), "Config should have training"
    assert hasattr(config, "checkpoint"), "Config should have checkpoint"
    assert hasattr(config, "runtime"), "Config should have runtime"

    # Verifiable 특화 검증
    assert config.experiment.stage == "verifiable", "Should be verifiable stage"
    assert config.data_sampling.auto_data_balancing == True, "Verifiable uses balanced sampling"
    assert config.data_sampling.correct_ratio == 0.5, "Verifiable uses 50:50 ratio"
    assert config.runtime.device == "mps", "Should use MPS for local test"

    # Training 파라미터 검증 (verifiable 특화)
    assert hasattr(config.training, "beta"), "Should have beta for TD error"
    assert hasattr(config.training, "value_coef"), "Should have value_coef"
    assert hasattr(config.training, "weight_clip_min"), "Should have weight_clip_min"
    assert hasattr(config.training, "weight_clip_max"), "Should have weight_clip_max"

    # Learning rate 검증 (trunk/value_head 분리)
    assert hasattr(config.training, "trunk_learning_rate"), "Should have trunk_learning_rate"
    assert hasattr(config.training, "value_head_learning_rate"), "Should have value_head_learning_rate"

    # 모델 경로 검증 (checkpoint 파일)
    model_path = Path(config.models.policy.path)
    # checkpoint는 critic 학습 후 생성되므로 경로 형식만 확인
    assert str(model_path).endswith(".pt"), f"Model path should be .pt checkpoint: {model_path}"

    tokenizer_path = Path(config.models.policy.tokenizer_path)
    assert tokenizer_path.exists(), f"Tokenizer path should exist: {tokenizer_path}"

    print(f"\n✓ Verifiable config validation passed")
