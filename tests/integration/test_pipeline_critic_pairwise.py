"""Critic Pairwise Pipeline Integration Test (MPS + micro-mtp)

M3 Mac MPS 환경에서 micro-mtp 모델로 pairwise critic 파이프라인 검증
- Pairwise ranking loss
- Bradley-Terry 모델 학습
- Pairwise accuracy 메트릭
"""

import pytest
import torch
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from weighted_mtp.pipelines.run_critic import run_critic_training


@pytest.fixture
def pairwise_test_config():
    """Pairwise 테스트용 config 생성"""
    config = {
        "project": {
            "name": "weighted-mtp",
            "version": "2.0.0",
        },
        "experiment": {
            "name": "test-critic-pairwise",
            "description": "Pairwise ranking loss test",
            "stage": "critic",
            "tags": ["critic", "pairwise", "test"],
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
            "max_length": 256,  # 테스트용 짧은 길이
        },
        "data_sampling": {
            "seed": 42,
            "val_n_samples": 10,
            "use_pairwise": True,
            "n_samples": 50,  # difficulty 기반 샘플링 후 pairwise 쌍 생성
            "auto_data_balancing": True,
            "correct_ratio": 0.5,
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
            "n_epochs": 1.0,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "trunk_learning_rate": 1e-5,
            "value_head_learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_unfrozen_layers": 0,
            "value_head_type": "mlp",
            "dropout": 0.1,
            "log_interval": 1,
            "gamma": 1.0,
            "lam": 1.0,
            "lr_scheduler": {
                "type": "constant",
                "warmup_ratio": 0.0,
                "min_lr_ratio": 1.0,
            },
        },
        "checkpoint": {
            "save_dir": "storage/checkpoints/critic/test-pairwise-integration",
            "save_checkpoint_every": 1.0,
            "save_best": True,
            "save_final": True,
            "save_total_limit": 2,
            "s3_upload": False,
        },
        "runtime": {
            "device": "mps",  # Mac 로컬 테스트
            "seed": 42,
            "mixed_precision": False,  # MPS는 mixed precision 제한
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
            "experiment": "",  # 테스트에서는 MLflow 비활성화
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        },
    }
    return OmegaConf.create(config)


@pytest.mark.integration
@pytest.mark.slow
def test_critic_pairwise_pipeline_micro_mtp(pairwise_test_config):
    """Critic Pairwise 파이프라인 end-to-end 테스트 (micro-mtp + MPS)"""

    # MPS 사용 가능 여부 확인
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    # 모델 경로 확인
    model_path = Path(pairwise_test_config.models.policy.path)
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    tokenizer_path = Path(pairwise_test_config.models.policy.tokenizer_path)
    if not tokenizer_path.exists():
        pytest.skip(f"Tokenizer not found: {tokenizer_path}")

    # 데이터셋 확인
    train_path = Path(pairwise_test_config.dataset.train)
    if not train_path.exists():
        pytest.skip(f"Train dataset not found: {train_path}")

    config = pairwise_test_config
    checkpoint_dir = Path(config.checkpoint.save_dir)

    try:
        # 파이프라인 실행
        final_metrics, best_checkpoint_path = run_critic_training(config)

        # 기본 검증
        assert final_metrics is not None, "Final metrics should not be None"
        assert "val_loss" in final_metrics, "Should have val_loss"
        assert isinstance(final_metrics["val_loss"], float), "val_loss should be float"
        assert final_metrics["val_loss"] > 0, "val_loss should be positive"

        # Pairwise 메트릭 검증
        assert "val_pairwise_accuracy" in final_metrics, "Should have val_pairwise_accuracy"
        pairwise_acc = final_metrics["val_pairwise_accuracy"]
        assert 0.0 <= pairwise_acc <= 1.0, f"Invalid pairwise_accuracy: {pairwise_acc}"

        assert "val_margin" in final_metrics, "Should have val_margin"

        # Checkpoint 생성 확인
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            assert len(checkpoints) > 0, "At least one checkpoint should be saved"

        print(f"\n✓ Critic Pairwise pipeline test passed")
        print(f"  Final val_loss: {final_metrics['val_loss']:.4f}")
        print(f"  Pairwise accuracy: {pairwise_acc:.3f}")
        print(f"  Margin: {final_metrics['val_margin']:.4f}")

    finally:
        # Cleanup
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"  Cleaned up test checkpoints")


@pytest.mark.integration
def test_pairwise_batch_structure():
    """Pairwise 배치 구조 검증 (DataLoader 레벨)"""
    from weighted_mtp.data.dataloader import create_dataloader
    from weighted_mtp.models.tokenizer_utils import load_tokenizer

    tokenizer_path = Path("storage/models/meta-llama-mtp/tokenizer")
    if not tokenizer_path.exists():
        pytest.skip(f"Tokenizer not found: {tokenizer_path}")

    # 표준 tokenizer 로딩 (padding_side="right" 보장)
    tokenizer = load_tokenizer(str(tokenizer_path))

    sampling_config = {
        "seed": 42,
        "use_pairwise": True,
        "n_samples": 500,  # 동일 problem_id 쌍 생성을 위해 충분한 샘플 필요
        "auto_data_balancing": True,
        "correct_ratio": 0.5,
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
        dataset_path="storage/datasets/codecontests/processed/train.jsonl",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=2048,  # codecontests 데이터는 긴 instruction을 가짐
        sampling_config=sampling_config,
        seed=42,
        shuffle=False,
    )

    # 첫 번째 배치 검증
    batch = next(iter(dataloader))

    # 필수 키 확인
    required_keys = [
        "pos_input_ids", "pos_attention_mask", "pos_labels",
        "neg_input_ids", "neg_attention_mask", "neg_labels",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"

    # Shape 검증
    assert batch["pos_input_ids"].shape[0] == 2  # batch_size
    assert batch["pos_input_ids"].shape[1] == 2048  # max_length
    assert batch["pos_input_ids"].shape == batch["neg_input_ids"].shape

    # Labels masking 검증
    pos_labels = batch["pos_labels"]
    neg_labels = batch["neg_labels"]

    # Instruction 부분은 -100
    assert (pos_labels[:, :10] == -100).all(), "pos instruction not masked"
    assert (neg_labels[:, :10] == -100).all(), "neg instruction not masked"

    # Output 부분은 일부 토큰이 학습 대상
    pos_non_masked = (pos_labels != -100).sum()
    neg_non_masked = (neg_labels != -100).sum()
    assert pos_non_masked > 0, "pos has no learning targets"
    assert neg_non_masked > 0, "neg has no learning targets"

    print(f"\n✓ Pairwise batch structure test passed")
    print(f"  Batch shape: {batch['pos_input_ids'].shape}")
    print(f"  pos learning tokens: {pos_non_masked.item()}")
    print(f"  neg learning tokens: {neg_non_masked.item()}")


@pytest.mark.integration
def test_pairwise_loss_computation():
    """Pairwise loss 계산 검증"""
    import torch
    from weighted_mtp.pipelines.run_critic import (
        pairwise_ranking_loss,
        compute_pairwise_accuracy,
    )

    batch_size = 4
    seq_len = 128

    # Mock value logits
    v_pos = torch.randn(batch_size, seq_len, 1)
    v_neg = torch.randn(batch_size, seq_len, 1)

    # Mock masks (일부 토큰만 유효)
    mask_pos = torch.zeros(batch_size, seq_len)
    mask_neg = torch.zeros(batch_size, seq_len)
    mask_pos[:, 50:100] = 1.0  # 50개 유효 토큰
    mask_neg[:, 50:100] = 1.0

    # Loss 계산
    loss = pairwise_ranking_loss(v_pos, v_neg, mask_pos, mask_neg)

    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    # Accuracy 계산
    metrics = compute_pairwise_accuracy(v_pos, v_neg, mask_pos, mask_neg)

    assert "pairwise_accuracy" in metrics
    assert "mean_pos" in metrics
    assert "mean_neg" in metrics
    assert "margin" in metrics

    acc = metrics["pairwise_accuracy"]
    assert 0.0 <= acc <= 1.0, f"Invalid accuracy: {acc}"

    print(f"\n✓ Pairwise loss computation test passed")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pairwise accuracy: {acc:.3f}")
    print(f"  Margin: {metrics['margin']:.4f}")


@pytest.mark.integration
def test_pairwise_loss_gradient():
    """Pairwise loss의 gradient 계산 가능 여부"""
    import torch
    from weighted_mtp.pipelines.run_critic import pairwise_ranking_loss

    batch_size = 2
    seq_len = 64

    # requires_grad=True로 생성
    v_pos = torch.randn(batch_size, seq_len, 1, requires_grad=True)
    v_neg = torch.randn(batch_size, seq_len, 1, requires_grad=True)

    mask_pos = torch.ones(batch_size, seq_len)
    mask_neg = torch.ones(batch_size, seq_len)

    loss = pairwise_ranking_loss(v_pos, v_neg, mask_pos, mask_neg)
    loss.backward()

    assert v_pos.grad is not None, "v_pos gradient not computed"
    assert v_neg.grad is not None, "v_neg gradient not computed"
    assert not torch.isnan(v_pos.grad).any(), "v_pos gradient has NaN"
    assert not torch.isnan(v_neg.grad).any(), "v_neg gradient has NaN"

    print(f"\n✓ Pairwise loss gradient test passed")
    print(f"  v_pos grad norm: {v_pos.grad.norm().item():.4f}")
    print(f"  v_neg grad norm: {v_neg.grad.norm().item():.4f}")


@pytest.mark.integration
def test_pairwise_expected_behavior():
    """Pairwise loss가 올바른 방향으로 학습하는지 검증

    V(pos) > V(neg)이면 loss 감소
    V(pos) < V(neg)이면 loss 증가
    """
    import torch
    from weighted_mtp.pipelines.run_critic import pairwise_ranking_loss

    batch_size = 4
    seq_len = 64
    mask = torch.ones(batch_size, seq_len)

    # Case 1: V(pos) > V(neg) - 올바른 순서
    v_pos_high = torch.ones(batch_size, seq_len, 1) * 2.0
    v_neg_low = torch.ones(batch_size, seq_len, 1) * (-2.0)
    loss_correct = pairwise_ranking_loss(v_pos_high, v_neg_low, mask, mask)

    # Case 2: V(pos) < V(neg) - 잘못된 순서
    v_pos_low = torch.ones(batch_size, seq_len, 1) * (-2.0)
    v_neg_high = torch.ones(batch_size, seq_len, 1) * 2.0
    loss_wrong = pairwise_ranking_loss(v_pos_low, v_neg_high, mask, mask)

    # 올바른 순서의 loss가 더 낮아야 함
    assert loss_correct < loss_wrong, \
        f"Expected loss_correct({loss_correct:.4f}) < loss_wrong({loss_wrong:.4f})"

    # Case 3: 동일한 값 - 중간 loss
    v_equal = torch.zeros(batch_size, seq_len, 1)
    loss_equal = pairwise_ranking_loss(v_equal, v_equal, mask, mask)

    # log(sigmoid(0)) = log(0.5) ≈ -0.693
    expected_equal_loss = -torch.log(torch.tensor(0.5))
    assert abs(loss_equal.item() - expected_equal_loss.item()) < 0.01, \
        f"V(pos)==V(neg)일 때 loss가 log(2)가 아님: {loss_equal.item()}"

    print(f"\n✓ Pairwise expected behavior test passed")
    print(f"  Loss (correct order): {loss_correct.item():.4f}")
    print(f"  Loss (wrong order): {loss_wrong.item():.4f}")
    print(f"  Loss (equal): {loss_equal.item():.4f}")
