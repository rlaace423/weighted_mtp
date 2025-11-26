"""datasets.py Config-driven 샘플링 검증 테스트

핵심 검증 항목:
- 데이터셋 로딩 기본 기능
- Config-driven 샘플링 전략 자동 선택
- Correct/Incorrect 비율 기반 샘플링
- Correct-only sampling (correct_ratio=1.0)
- Difficulty-based sampling (difficulty_weights 있음)
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
        "codecontests",
        split="train",
        sampling_config={
            "n_samples": 50,
            "correct_ratio": 0.5,
        },
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
        """Difficulty 필드 타입 검증"""
        sample = small_dataset[0]
        difficulty = sample["metadata"]["difficulty"]
        assert isinstance(difficulty, int), "Difficulty는 int 타입이어야 함"
        assert 1 <= difficulty <= 11, "Difficulty는 1-11 범위"

    def test_is_correct_field_type(self, small_dataset):
        """is_correct 필드 타입 검증"""
        sample = small_dataset[0]
        assert isinstance(sample["is_correct"], bool), "is_correct는 bool 타입"


class TestCorrectRatioSampling:
    """Correct/Incorrect 비율 기반 샘플링 테스트"""

    def test_sampling_50_50(self):
        """50:50 비율 샘플링 검증"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 100,
                "correct_ratio": 0.5,
            },
            seed=42
        )

        assert len(dataset) == 100

        # correct/incorrect 카운트
        correct_count = sum(1 for sample in dataset if sample["is_correct"])

        # 50:50 비율 검증 (오차 ±10% 허용)
        ratio = correct_count / len(dataset)
        assert 0.4 <= ratio <= 0.6, f"Expected ~50%, got {ratio:.2%}"

    def test_sampling_70_30(self):
        """70:30 비율 샘플링 검증"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 100,
                "correct_ratio": 0.7,
            },
            seed=42
        )

        correct_count = sum(1 for sample in dataset if sample["is_correct"])
        ratio = correct_count / len(dataset)

        # 70:30 비율 검증 (오차 ±10% 허용)
        assert 0.6 <= ratio <= 0.8, f"Expected ~70%, got {ratio:.2%}"


class TestCorrectOnlySampling:
    """Correct-only 샘플링 테스트 (Rho-1, Baseline)"""

    def test_correct_only_sampling(self):
        """정답만 샘플링 검증"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 50,
                "correct_ratio": 1.0,
            },
            seed=42
        )

        assert len(dataset) > 0

        # 모든 샘플이 correct인지 검증
        all_correct = all(sample["is_correct"] for sample in dataset)
        assert all_correct, "correct_ratio=1.0이면 모든 샘플이 is_correct=True여야 함"


class TestDifficultyBasedSampling:
    """Difficulty-based curriculum 샘플링 테스트 (Verifiable)"""

    def test_difficulty_based_sampling_low_only(self):
        """Low difficulty만 샘플링 검증"""
        difficulty_weights = {"low": 1.0, "medium": 0.0, "high": 0.0}
        difficulty_bins = {"low": [1, 3], "medium": [4, 7], "high": [8, 11]}

        dataset = load_dataset(
            "codecontests",
            split="train",
            n_samples=50,
                        correct_ratio=0.5,
            difficulty_weights=difficulty_weights,
            difficulty_bins=difficulty_bins,
            seed=42
        )

        assert len(dataset) > 0

        # 모든 샘플이 low difficulty 범위인지 검증
        for sample in dataset:
            difficulty = sample["metadata"]["difficulty"]
            assert 1 <= difficulty <= 3, f"Expected low difficulty (1-3), got {difficulty}"

    def test_difficulty_based_sampling_mixed(self):
        """Mixed difficulty 샘플링 검증"""
        difficulty_weights = {"low": 0.7, "medium": 0.3, "high": 0.0}
        difficulty_bins = {"low": [1, 3], "medium": [4, 7], "high": [8, 11]}

        dataset = load_dataset(
            "codecontests",
            split="train",
            n_samples=100,
                        correct_ratio=0.5,
            difficulty_weights=difficulty_weights,
            difficulty_bins=difficulty_bins,
            seed=42
        )

        # Low/medium 분포 검증
        low_count = sum(
            1 for sample in dataset
            if 1 <= sample["metadata"]["difficulty"] <= 3
        )
        medium_count = sum(
            1 for sample in dataset
            if 4 <= sample["metadata"]["difficulty"] <= 7
        )

        # 70:30 비율 검증 (오차 허용)
        assert low_count > medium_count, "Low difficulty가 더 많아야 함"
        assert medium_count > 0, "Medium difficulty 샘플도 있어야 함"


class TestReproducibility:
    """재현성 테스트 (동일 seed → 동일 샘플)"""

    def test_same_seed_same_samples(self):
        """동일 seed로 두 번 로딩 시 동일한 샘플 반환"""
        dataset1 = load_dataset(
            "codecontests",
            split="train",
            n_samples=20,
                        correct_ratio=0.5,
            seed=42
        )

        dataset2 = load_dataset(
            "codecontests",
            split="train",
            n_samples=20,
                        correct_ratio=0.5,
            seed=42
        )

        # task_id 비교 (샘플 순서까지 동일해야 함)
        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2, "동일 seed면 동일한 샘플 순서여야 함"

    def test_different_seed_different_samples(self):
        """다른 seed로 로딩 시 다른 샘플 반환"""
        dataset1 = load_dataset(
            "codecontests",
            split="train",
            n_samples=20,
                        correct_ratio=0.5,
            seed=42
        )

        dataset2 = load_dataset(
            "codecontests",
            split="train",
            n_samples=20,
                        correct_ratio=0.5,
            seed=999
        )

        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 != ids2, "다른 seed면 다른 샘플이어야 함"


class TestSplits:
    """스플릿별 로딩 테스트"""

    def test_load_train_split(self):
        """Train 스플릿 로딩"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            n_samples=10,
                        seed=42
        )
        assert len(dataset) == 10

    def test_load_validation_split(self):
        """Validation 스플릿 로딩"""
        dataset = load_dataset(
            "codecontests",
            split="validation",
            n_samples=10,
                        seed=42
        )
        assert len(dataset) == 10

    def test_load_test_split(self):
        """Test 스플릿 로딩"""
        dataset = load_dataset(
            "codecontests",
            split="test",
            n_samples=10,
                        seed=42
        )
        assert len(dataset) == 10
