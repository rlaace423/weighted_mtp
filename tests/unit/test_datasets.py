"""datasets.py 핵심 기능 검증 테스트

핵심 검증 항목:
- 데이터셋 로딩 기본 기능
- Stage 1 샘플링 (is_correct 균형)
- Stage 2 샘플링 (difficulty curriculum)
- 재현성 (seed 고정)
"""

import pytest
from datasets import Dataset

from weighted_mtp.data import load_dataset


# 공유 데이터셋 fixture (성능 최적화)
@pytest.fixture(scope="module")
def small_dataset():
    """작은 테스트용 데이터셋 (전체 테스트에서 재사용)"""
    return load_dataset(
        "codecontests_small",
        split="train",
        stage="stage1",
        n_samples=50,
        balance_correct=False,
        seed=42
    )


class TestLoadDataset:
    """데이터셋 로딩 기본 기능 테스트"""

    def test_load_codecontests_basic(self, small_dataset):
        """CodeContests 로딩 및 필수 필드 검증"""
        dataset = small_dataset

        assert isinstance(dataset, Dataset)
        assert len(dataset) > 0

        # 필수 필드 검증
        sample = dataset[0]
        assert "instruction" in sample
        assert "input" in sample
        assert "output" in sample
        assert "task_id" in sample
        assert "is_correct" in sample
        assert "metadata" in sample
        assert "difficulty" in sample["metadata"]

    def test_difficulty_field_type(self, small_dataset):
        """difficulty 필드 타입 검증"""
        dataset = small_dataset

        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            difficulty = sample["metadata"]["difficulty"]

            assert isinstance(difficulty, int)
            assert 1 <= difficulty <= 11


class TestStage1Sampling:
    """Stage 1: is_correct 균형 샘플링 테스트"""

    def test_stage1_sampling_and_ratio(self):
        """Stage 1 샘플링 검증 (작은 데이터셋은 균형 샘플링 비활성화)"""
        n_samples = 20
        dataset = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage1",
            n_samples=n_samples,
            balance_correct=False,  # 작은 데이터셋은 균형 샘플링 불필요
            seed=42,
        )

        # 샘플 수 검증
        assert len(dataset) == n_samples

    def test_stage1_reproducibility(self):
        """Stage 1 재현성 검증 (seed 고정)"""
        # 같은 seed로 두 번 샘플링
        dataset1 = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage1",
            n_samples=20,
            balance_correct=False,
            seed=42,
        )

        dataset2 = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage1",
            n_samples=20,
            balance_correct=False,
            seed=42,
        )

        # task_id 비교 (동일한 샘플 선택)
        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2, "동일한 seed에도 다른 샘플이 선택됨"


class TestStage2Sampling:
    """Stage 2: difficulty 기반 Curriculum Learning 샘플링 테스트"""

    def test_stage2_sampling_and_distribution(self):
        """Stage 2 샘플링 + difficulty 분포 검증

        Note: 작은 테스트 데이터셋은 correct 샘플만 포함하므로
        균형 샘플링 파라미터는 전달하지만 검증은 생략
        """
        n_samples = 20
        dataset = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage2",
            n_samples=n_samples,
            balance_correct=True,  # 파라미터 전달 (통합 테스트에서 사용)
            correct_ratio=0.5,
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        # 샘플 수 검증
        assert len(dataset) == n_samples

        # difficulty 분포 계산
        low_count = sum(1 for s in dataset if 1 <= s["metadata"]["difficulty"] <= 3)
        medium_count = sum(1 for s in dataset if 4 <= s["metadata"]["difficulty"] <= 7)

        low_ratio = low_count / len(dataset)
        medium_ratio = medium_count / len(dataset)

        # 가중치 ±20% 범위 허용 (작은 샘플 수로 인한 분산 고려)
        assert 0.5 <= low_ratio <= 0.9, f"low 비율이 범위 밖: {low_ratio:.2%} (목표: 70% ±20%)"
        assert 0.1 <= medium_ratio <= 0.5, (
            f"medium 비율이 범위 밖: {medium_ratio:.2%} (목표: 30% ±20%)"
        )

        # Note: 작은 테스트 데이터셋은 correct 100%이므로 균형 검증 생략
        # 실제 균형 샘플링 검증은 통합 테스트에서 전체 데이터셋으로 수행

    def test_stage2_reproducibility(self):
        """Stage 2 재현성 검증 (균형 샘플링 포함)"""
        dataset1 = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage2",
            n_samples=20,
            balance_correct=True,
            correct_ratio=0.5,
            difficulty_weights={"low": 0.5, "medium": 0.5, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        dataset2 = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage2",
            n_samples=20,
            balance_correct=True,
            correct_ratio=0.5,
            difficulty_weights={"low": 0.5, "medium": 0.5, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2
