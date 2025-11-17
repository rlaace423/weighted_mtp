"""Evaluation Pipeline Integration Test

micro-mtp 모델로 평가 파이프라인 E2E 테스트
"""

import pytest
import torch
from pathlib import Path

from weighted_mtp.pipelines.run_evaluation import run_evaluation


def create_dummy_checkpoint(checkpoint_dir: Path) -> Path:
    """테스트용 dummy checkpoint 생성

    Args:
        checkpoint_dir: Checkpoint 저장 디렉터리

    Returns:
        생성된 checkpoint 경로
    """
    from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter

    # Micro 모델 로드
    model = MetaLlamaMTPAdapter.from_pretrained(
        model_path="storage/models_v2/micro-mtp",
        device=torch.device("cpu"),
        initialize_value_head=False,
    )

    # Checkpoint 저장
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_test.pt"

    checkpoint = {
        "epoch": 1.0,
        "adapter_state_dict": model.state_dict(),
        "optimizer_state_dict": {},  # Dummy
        "train_metrics": {"train_loss": 3.0},
        "val_metrics": {"val_loss": 2.8},
        "config": {
            "model": {
                "path": "storage/models_v2/micro-mtp"
            }
        },
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.mark.integration
@pytest.mark.slow
def test_evaluation_pipeline_micro_mtp():
    """Micro 모델로 평가 파이프라인 E2E 테스트"""

    # MPS 사용 가능 여부 확인
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Dummy checkpoint 생성
    checkpoint_dir = Path("storage/checkpoints/test-evaluation-integration")
    checkpoint_path = create_dummy_checkpoint(checkpoint_dir)

    # humaneval 데이터셋 존재 확인
    dataset_path = Path("storage/datasets_v2/humaneval/processed/test.jsonl")
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")

    try:
        # 평가 실행 (샘플 5개만, 샘플당 2개 생성)
        results = run_evaluation(
            checkpoint_path=str(checkpoint_path),
            dataset_name="humaneval",
            num_samples_per_task=2,
            temperature=0.2,
            max_new_tokens=50,  # 짧게 생성
            device=device,
            mlflow_enabled=False,  # MLflow 비활성화
            max_tasks=5,  # 테스트용: 5개 태스크만 평가
        )

        # 결과 검증
        assert "pass_at_k" in results, "Should have pass_at_k metrics"
        assert "per_task" in results, "Should have per_task results"
        assert "checkpoint_metadata" in results, "Should have checkpoint metadata"

        # Pass@K 메트릭 검증
        assert "pass@1" in results["pass_at_k"], "Should have pass@1"
        assert isinstance(results["pass_at_k"]["pass@1"], float), "pass@1 should be float"
        assert 0.0 <= results["pass_at_k"]["pass@1"] <= 1.0, "pass@1 should be in [0, 1]"

        # Per-task 결과 검증
        assert len(results["per_task"]) > 0, "Should have at least one task result"
        first_task = results["per_task"][0]
        assert "task_id" in first_task, "Task should have task_id"
        assert "pass@1" in first_task, "Task should have pass@1"

        # Checkpoint metadata 검증
        assert results["checkpoint_metadata"]["epoch"] == 1.0
        assert results["checkpoint_metadata"]["val_metrics"]["val_loss"] == 2.8

        print(f"\n✓ Evaluation pipeline test passed")
        print(f"  Pass@1: {results['pass_at_k']['pass@1']:.2%}")
        print(f"  Tasks evaluated: {len(results['per_task'])}")
        print(f"  Checkpoint epoch: {results['checkpoint_metadata']['epoch']}")

    finally:
        # Cleanup: 테스트 checkpoint 삭제
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
            checkpoint_dir.rmdir()
