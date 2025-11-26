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
    """Verifiable Pairwise 테스트용 config 생성"""
    config = {
        "project": {
            "name": "weighted-mtp",
            "version": "2.0.0",
        },
        "experiment": {
            "name": "test-verifiable-pairwise",
            "description": "Verifiable + Pairwise auxiliary loss test",
            "stage": "verifiable",
            "tags": ["verifiable", "pairwise", "test"],
        },
        "models": {
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
                "dtype": "float32",  # MPS 호환
            },
        },
        "dataset": {
            "name": "codecontests",
            "train": "storage/datasets/codecontests/processed/train.jsonl",
            "validation": "storage/datasets/codecontests/processed/valid.jsonl",
            "max_length": 1024,  # 충분한 길이
        },
        "data_sampling": {
            "seed": 42,
            "val_n_samples": 50,
            "use_pairwise": True,  # Pairwise 모드 활성화
            "n_samples": 200,  # 최종 쌍 수
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
            "n_epochs": 0.1,  # 빠른 테스트
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "trunk_learning_rate": 1e-5,
            "value_head_learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "beta": 0.5,
            "pairwise_coef": 0.1,  # Pairwise auxiliary loss 계수
            "weight_clip_min": 0,
            "weight_clip_max": 3,
            "log_interval": 1,
            "lr_scheduler": {
                "type": "constant",
                "warmup_ratio": 0.0,
                "min_lr_ratio": 1.0,
            },
        },
        "checkpoint": {
            "save_dir": "storage/checkpoints/verifiable/test-pairwise-integration",
            "save_checkpoint_every": 1.0,
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
    """Verifiable Pairwise loss 구조 테스트

    total_loss = weighted_ce_loss + pairwise_coef * aux_loss
    """
    import torch
    import torch.nn.functional as F
    from weighted_mtp.utils import pairwise_ranking_loss

    batch_size = 2
    seq_len = 64
    n_future = 4
    vocab_size = 100

    # Mock data
    pos_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
    pos_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    pos_labels[:, :10] = -100  # Instruction masked

    pos_value_logits = torch.randn(batch_size, seq_len, 1)
    neg_value_logits = torch.randn(batch_size, seq_len, 1)

    pos_mask = (pos_labels != -100).float()
    neg_mask = pos_mask.clone()

    # Weighted policy loss (simplified)
    policy_loss = F.cross_entropy(
        pos_logits[:, :, 0, :].reshape(-1, vocab_size),
        pos_labels.reshape(-1),
        reduction="mean",
        ignore_index=-100,
    )

    # Auxiliary pairwise loss
    aux_loss = pairwise_ranking_loss(pos_value_logits, neg_value_logits, pos_mask, neg_mask)

    # Total loss = weighted_ce_loss + pairwise_coef * aux_loss
    pairwise_coef = 0.1
    total_loss = policy_loss + pairwise_coef * aux_loss

    # Assertions
    assert total_loss.dim() == 0, "Total loss should be scalar"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    assert total_loss.item() > 0, "Total loss should be positive"

    print(f"\n[Verifiable Pairwise Loss Structure]")
    print(f"  Policy loss: {policy_loss.item():.4f}")
    print(f"  Pairwise aux loss: {aux_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")


@pytest.mark.integration
def test_verifiable_pairwise_gradient_flow():
    """Pairwise 모드에서 pairwise loss가 gradient를 역전파하는지 검증"""
    import torch
    from weighted_mtp.utils import pairwise_ranking_loss

    batch_size = 2
    seq_len = 32

    # requires_grad=True
    pos_value_logits = torch.randn(batch_size, seq_len, 1, requires_grad=True)
    neg_value_logits = torch.randn(batch_size, seq_len, 1, requires_grad=True)

    pos_mask = torch.ones(batch_size, seq_len)
    neg_mask = torch.ones(batch_size, seq_len)

    # Pairwise auxiliary loss
    pairwise_loss = pairwise_ranking_loss(pos_value_logits, neg_value_logits, pos_mask, neg_mask)

    # Backward
    pairwise_loss.backward()

    assert pos_value_logits.grad is not None, "pos_value_logits gradient not computed"
    assert neg_value_logits.grad is not None, "neg_value_logits gradient not computed"
    assert not torch.isnan(pos_value_logits.grad).any(), "pos gradient has NaN"
    assert not torch.isnan(neg_value_logits.grad).any(), "neg gradient has NaN"

    print(f"\n[Gradient Flow Test]")
    print(f"  pos_value grad norm: {pos_value_logits.grad.norm().item():.4f}")
    print(f"  neg_value grad norm: {neg_value_logits.grad.norm().item():.4f}")


@pytest.mark.integration
def test_verifiable_pairwise_config_parsing():
    """Config에서 use_pairwise와 pairwise_coef 파싱 테스트"""
    from omegaconf import OmegaConf

    # Pairwise 활성화 config
    config_pairwise = OmegaConf.create({
        "data_sampling": {
            "use_pairwise": True,
            "n_samples": 100,
        },
        "training": {
            "pairwise_coef": 0.2,
        },
    })

    sampling_config = OmegaConf.to_container(config_pairwise.data_sampling, resolve=True)
    use_pairwise = sampling_config.get("use_pairwise", False)
    pairwise_coef = getattr(config_pairwise.training, "pairwise_coef", 0.1)

    assert use_pairwise is True, "use_pairwise should be True"
    assert pairwise_coef == 0.2, f"pairwise_coef should be 0.2, got {pairwise_coef}"

    # 기본값 테스트
    config_default = OmegaConf.create({
        "data_sampling": {
            "use_pairwise": True,
            "n_samples": 100,
        },
        "training": {},
    })

    sampling_config2 = OmegaConf.to_container(config_default.data_sampling, resolve=True)
    use_pairwise2 = sampling_config2.get("use_pairwise", False)
    pairwise_coef2 = getattr(config_default.training, "pairwise_coef", 0.1)

    assert use_pairwise2 is True, "use_pairwise should be True"
    assert pairwise_coef2 == 0.1, f"pairwise_coef should default to 0.1, got {pairwise_coef2}"

    print(f"\n[Config Parsing Test]")
    print(f"  Explicit pairwise_coef: use_pairwise={use_pairwise}, pairwise_coef={pairwise_coef}")
    print(f"  Default pairwise_coef: use_pairwise={use_pairwise2}, pairwise_coef={pairwise_coef2}")


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
    """Verifiable Pairwise 파이프라인 end-to-end 테스트 (micro-mtp + MPS)"""

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    model_path = Path(verifiable_pairwise_test_config.models.policy.path)
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    tokenizer_path = Path(verifiable_pairwise_test_config.models.policy.tokenizer_path)
    if not tokenizer_path.exists():
        pytest.skip(f"Tokenizer not found: {tokenizer_path}")

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
