"""Baseline Pipeline Integration Test (MPS + micro-mtp)

M3 Mac MPS 환경에서 micro-mtp 모델로 baseline 파이프라인 검증
- Uniform weighting (비교 기준선)
- Correct-only 샘플링
- 20 samples, 0.02 epochs
"""

import pytest
import torch
from pathlib import Path

from omegaconf import OmegaConf
from weighted_mtp.pipelines.run_baseline import run_baseline_training


@pytest.mark.integration
@pytest.mark.slow
def test_baseline_pipeline_micro_mtp():
    """Baseline 파이프라인 end-to-end 테스트 (micro-mtp + MPS)"""

    # MPS 사용 가능 여부 확인
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    # Test config 경로
    config_path = "configs/baseline/baseline_local.yaml"
    assert Path(config_path).exists(), f"Config not found: {config_path}"

    # Override 파라미터 (초경량 테스트 설정)
    override_params = {
        "models": {
            "policy": {
                "dtype": "float32",  # MPS에서 float16은 cross_entropy overflow 가능
            },
        },
        "data_sampling": {
            "n_samples": 20,  # 20개 샘플
        },
        "training": {
            "n_epochs": 1.5,  # 1.5 epoch (30 samples total)
            "batch_size": 2,
            "log_interval": 1,
        },
        "checkpoint": {
            "save_checkpoint_every": 1.5,
            "save_dir": "storage/checkpoints/baseline/test-baseline-integration",
        },
        "experiment": {
            "name": "test-baseline-integration",
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
        final_metrics, best_checkpoint_path = run_baseline_training(config)

        # 검증
        assert final_metrics is not None, "Final metrics should not be None"
        assert "val_loss" in final_metrics, "Should have val_loss"
        assert isinstance(final_metrics["val_loss"], float), "val_loss should be float"
        assert final_metrics["val_loss"] > 0, "val_loss should be positive"

        # Checkpoint 생성 확인
        if best_checkpoint_path:
            assert Path(best_checkpoint_path).exists(), f"Best checkpoint should exist: {best_checkpoint_path}"

        checkpoint_dir = Path(override_params["checkpoint"]["save_dir"])
        assert checkpoint_dir.exists(), "Checkpoint directory should exist"

        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0, "At least one checkpoint should be saved"

        print(f"\n✓ Baseline pipeline test passed")
        print(f"  Final val_loss: {final_metrics['val_loss']:.4f}")
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
def test_baseline_config_validation():
    """Baseline config 파일 유효성 검증"""
    config_path = Path("configs/baseline/baseline_local.yaml")
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

    # Baseline 특화 검증
    assert config.experiment.stage == "baseline", "Should be baseline stage"
    assert config.data_sampling.correct_ratio == 1.0, "Baseline uses only correct samples"
    assert config.runtime.device == "mps", "Should use MPS for local test"

    # 모델 경로 검증
    model_path = Path(config.models.policy.path)
    assert model_path.exists(), f"Model path should exist: {model_path}"

    tokenizer_path = Path(config.models.policy.tokenizer_path)
    assert tokenizer_path.exists(), f"Tokenizer path should exist: {tokenizer_path}"

    print(f"\n✓ Baseline config validation passed")
