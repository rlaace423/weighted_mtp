"""Verifiable Pairwise Pipeline Integration Test (MPS + micro-mtp)

M3 Mac MPS 환경에서 micro-mtp 모델로 verifiable pairwise 파이프라인 검증
- TD Weighted Policy Loss (Positive sample only)
- Value Loss (Positive + Negative)
- Auxiliary Pairwise Ranking Loss
"""

import pytest
import torch
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from weighted_mtp.pipelines.run_verifiable import run_verifiable_training, validate_verifiable


@pytest.fixture
def verifiable_pairwise_test_config():
    """Verifiable Pairwise 테스트용 config 생성 (독립 Value Model 구조)"""
    config = {
        "project": {
            "name": "weighted-mtp",
            "version": "2.0.0",
        },
        "experiment": {
            "name": "test-verifiable-pairwise",
            "description": "Verifiable with separate Value Model test",
            "stage": "verifiable",
            "tags": ["verifiable", "pairwise", "test"],
        },
        "models": {
            # Policy Model (학습 대상)
            "policy": {
                "name": "micro-mtp",
                "path": "storage/models/micro-mtp",
                "tokenizer_path": "storage/models/meta-llama-mtp/tokenizer",
                "params": {
                    "dim": 512,
                    "n_layers": 4,
                    "n_heads": 8,
                    "n_future_tokens": 4,
                },
                "dtype": "float32",
            },
            # Value Model (Critic checkpoint에서 로드)
            "value_model": {
                "checkpoint_path": "storage/checkpoints/critic/test-pairwise-integration/checkpoint_final.pt",
            },
        },
        "dataset": {
            "name": "codecontests",
            "train": "storage/datasets/codecontests/processed/train.jsonl",
            "validation": "storage/datasets/codecontests/processed/valid.jsonl",
            "max_length": 512,
        },
        "data_sampling": {
            "seed": 42,
            "val_n_samples": 50,
            "use_pairwise": True,
            "n_samples": 100,
            "difficulty_bins": {
                "diff_7": [7, 7],
                "else": [8, 25],
            },
            "difficulty_weights": {
                "diff_7": 0.35,
                "else": 0.65,
            },
        },
        "training": {
            "n_epochs": 0.1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "beta": 1.0,
            "weight_clip_min": 0,
            "weight_clip_max": 3,
            "td_ema_momentum": 0.1,
            "td_ema_warmup_steps": 5,
            "log_interval": 1,
            "lr_scheduler": {
                "type": "constant",
                "warmup_ratio": 0.0,
                "min_lr_ratio": 1.0,
            },
        },
        "checkpoint": {
            "save_dir": "storage/checkpoints/verifiable/test-pairwise-integration",
            "save_checkpoint_every": 0.1,
            "save_best": True,
            "save_final": True,
            "save_total_limit": 2,
            "s3_upload": False,
        },
        "runtime": {
            "device": "mps",
            "seed": 42,
            "mixed_precision": False,
        },
        "distributed": {
            "fsdp": {
                "sharding_strategy": "NO_SHARD",
                "mixed_precision": False,
                "cpu_offload": False,
                "activation_checkpointing": False,
            },
        },
        "storage": {
            "root": "storage",
            "models_dir": "storage/models",
            "datasets_dir": "storage/datasets",
            "checkpoints_dir": "storage/checkpoints",
        },
        "mlflow": {
            "tracking_uri": "",
            "experiment": "",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        },
    }
    return OmegaConf.create(config)


@pytest.mark.integration
def test_verifiable_pairwise_loss_structure():
    """Verifiable loss 구조 테스트 (독립 Value Model 구조)

    total_loss = weighted_ce_loss
    (Value Model은 분리되어 Critic에서 학습, Verifiable은 eval만)
    """
    import torch
    import torch.nn.functional as F

    batch_size = 2
    seq_len = 64
    n_future = 4
    vocab_size = 100

    # Mock data
    pos_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
    pos_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    pos_labels[:, :10] = -100  # Instruction masked

    # Weighted policy loss (simplified)
    weighted_ce_loss = F.cross_entropy(
        pos_logits[:, :, 0, :].reshape(-1, vocab_size),
        pos_labels.reshape(-1),
        reduction="mean",
        ignore_index=-100,
    )

    # Total loss = weighted_ce_loss (Value Model 분리 후)
    total_loss = weighted_ce_loss

    # Assertions
    assert total_loss.dim() == 0, "Total loss should be scalar"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    assert total_loss.item() > 0, "Total loss should be positive"

    print(f"\n[Verifiable Loss Structure (Separate Value Model)]")
    print(f"  Weighted CE loss: {weighted_ce_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")


@pytest.mark.integration
def test_verifiable_gradient_flow():
    """Policy Model의 weighted CE loss gradient 역전파 검증"""
    import torch
    import torch.nn.functional as F

    batch_size = 2
    seq_len = 32
    vocab_size = 100

    # requires_grad=True
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :10] = -100  # Instruction masked

    # Weights from TD error (시뮬레이션)
    weights = torch.rand(batch_size, seq_len)
    weights = weights * (labels != -100).float()

    # Weighted CE loss
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    weighted_loss = (ce_per_token * weights).sum() / weights.sum().clamp(min=1e-8)

    # Backward
    weighted_loss.backward()

    assert logits.grad is not None, "logits gradient not computed"
    assert not torch.isnan(logits.grad).any(), "logits gradient has NaN"

    print(f"\n[Gradient Flow Test]")
    print(f"  logits grad norm: {logits.grad.norm().item():.4f}")


@pytest.mark.integration
def test_verifiable_pairwise_config_parsing():
    """Config에서 use_pairwise 파싱 테스트"""
    from omegaconf import OmegaConf

    # Pairwise 활성화 config
    config_pairwise = OmegaConf.create({
        "data_sampling": {
            "use_pairwise": True,
            "n_samples": 100,
        },
        "training": {
            "beta": 1.0,
            "weight_clip_min": 0,
            "weight_clip_max": 3,
        },
    })

    sampling_config = OmegaConf.to_container(config_pairwise.data_sampling, resolve=True)
    use_pairwise = sampling_config.get("use_pairwise", False)

    assert use_pairwise is True, "use_pairwise should be True"
    assert config_pairwise.training.beta == 1.0, "beta should be 1.0"

    # 기본값 테스트
    config_default = OmegaConf.create({
        "data_sampling": {
            "use_pairwise": False,
            "n_samples": 100,
        },
        "training": {},
    })

    sampling_config2 = OmegaConf.to_container(config_default.data_sampling, resolve=True)
    use_pairwise2 = sampling_config2.get("use_pairwise", False)

    assert use_pairwise2 is False, "use_pairwise should be False"

    print(f"\n[Config Parsing Test]")
    print(f"  Pairwise mode: use_pairwise={use_pairwise}")
    print(f"  Pointwise mode: use_pairwise={use_pairwise2}")


@pytest.mark.integration
def test_verifiable_pairwise_batch_structure():
    """Verifiable Pairwise 배치 구조 검증"""
    from weighted_mtp.data.dataloader import create_dataloader
    from weighted_mtp.models.tokenizer_utils import load_tokenizer

    tokenizer_path = Path("storage/models/meta-llama-mtp/tokenizer")
    if not tokenizer_path.exists():
        pytest.skip(f"Tokenizer not found: {tokenizer_path}")

    train_path = Path("storage/datasets/codecontests/processed/train.jsonl")
    if not train_path.exists():
        pytest.skip(f"Dataset not found: {train_path}")

    tokenizer = load_tokenizer(str(tokenizer_path))

    sampling_config = {
        "seed": 42,
        "use_pairwise": True,
        "n_samples": 500,  # 최종 쌍 수
        "difficulty_bins": {
            "diff_7": [7, 7],
            "else": [8, 25],
        },
        "difficulty_weights": {
            "diff_7": 0.35,
            "else": 0.65,
        },
    }

    dataloader = create_dataloader(
        dataset_path=str(train_path),
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        sampling_config=sampling_config,
        seed=42,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    # Pairwise 배치 키 확인
    pairwise_keys = [
        "pos_input_ids", "pos_attention_mask", "pos_labels",
        "neg_input_ids", "neg_attention_mask", "neg_labels",
    ]
    for key in pairwise_keys:
        assert key in batch, f"Missing pairwise key: {key}"

    # Shape 검증
    assert batch["pos_input_ids"].shape == batch["neg_input_ids"].shape
    assert batch["pos_labels"].shape == batch["neg_labels"].shape

    # Masking 검증
    pos_learn = (batch["pos_labels"] != -100).sum()
    neg_learn = (batch["neg_labels"] != -100).sum()
    assert pos_learn > 0, "Positive has no learning targets"
    assert neg_learn > 0, "Negative has no learning targets"

    print(f"\n[Pairwise Batch Structure]")
    print(f"  Shape: {batch['pos_input_ids'].shape}")
    print(f"  pos learning tokens: {pos_learn.item()}")
    print(f"  neg learning tokens: {neg_learn.item()}")


@pytest.mark.integration
@pytest.mark.slow
def test_verifiable_pairwise_pipeline_micro_mtp(verifiable_pairwise_test_config):
    """Verifiable 파이프라인 end-to-end 테스트 (micro-mtp + MPS, 독립 Value Model)"""

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    # Policy Model 경로 확인
    model_path = Path(verifiable_pairwise_test_config.models.policy.path)
    if not model_path.exists():
        pytest.skip(f"Policy Model not found: {model_path}")

    tokenizer_path = Path(verifiable_pairwise_test_config.models.policy.tokenizer_path)
    if not tokenizer_path.exists():
        pytest.skip(f"Tokenizer not found: {tokenizer_path}")

    # Value Model checkpoint 확인
    value_checkpoint_path = Path(verifiable_pairwise_test_config.models.value_model.checkpoint_path)
    if not value_checkpoint_path.exists():
        pytest.skip(f"Value Model checkpoint not found: {value_checkpoint_path}. Run critic training first.")

    train_path = Path(verifiable_pairwise_test_config.dataset.train)
    if not train_path.exists():
        pytest.skip(f"Train dataset not found: {train_path}")

    config = verifiable_pairwise_test_config
    checkpoint_dir = Path(config.checkpoint.save_dir)

    try:
        final_metrics, best_checkpoint_path = run_verifiable_training(config)

        # 기본 검증
        assert final_metrics is not None, "Final metrics should not be None"
        assert "val_loss" in final_metrics, "Should have val_loss"
        assert isinstance(final_metrics["val_loss"], float), "val_loss should be float"

        print(f"\n[Verifiable Pairwise Pipeline Test]")
        print(f"  Final val_loss: {final_metrics['val_loss']:.4f}")

        # Pairwise 메트릭 확인 (있으면)
        if "val_pairwise_accuracy" in final_metrics:
            print(f"  Pairwise accuracy: {final_metrics['val_pairwise_accuracy']:.3f}")
        if "val_margin" in final_metrics:
            print(f"  Margin: {final_metrics['val_margin']:.4f}")

    finally:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"  Cleaned up test checkpoints")
