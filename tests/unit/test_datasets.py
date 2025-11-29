"""datasets.py Config-driven 샘플링 검증 테스트

핵심 검증 항목:
- 데이터셋 로딩 기본 기능
- Unique pair 샘플링 (correct_idx, incorrect_idx 모두 unique)
- Difficulty-based sampling (difficulty_weights, difficulty_bins 필수)
- max_pairs_per_problem 적용
- 재현성 (seed 고정)
"""

import pytest
from datasets import Dataset

from weighted_mtp.data import load_dataset


# 기본 샘플링 설정 (모든 테스트에서 재사용)
DEFAULT_SAMPLING_CONFIG = {
    "n_samples": 50,
    "use_pairwise": False,
    "max_pairs_per_problem": 20,
    "difficulty_bins": {"all": [0, 25]},
    "difficulty_weights": {"all": 1.0},
}


# 공유 데이터셋 fixture (성능 최적화)
@pytest.fixture(scope="module")
def small_dataset():
    """작은 테스트용 데이터셋 (전체 테스트에서 재사용)"""
    return load_dataset(
        "codecontests",
        split="train",
        sampling_config=DEFAULT_SAMPLING_CONFIG,
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
        assert 0 <= difficulty <= 25, "Difficulty는 0-25 범위 (CodeContests)"

    def test_is_correct_field_type(self, small_dataset):
        """is_correct 필드 타입 검증"""
        sample = small_dataset[0]
        assert isinstance(sample["is_correct"], bool), "is_correct는 bool 타입"


class TestUniquePairSampling:
    """Unique pair 샘플링 통합 테스트"""

    def test_pointwise_all_correct(self):
        """Pointwise 모드: 모든 샘플이 correct (unique pair에서 correct만 추출)"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 100,
                "use_pairwise": False,
                "max_pairs_per_problem": 20,
                "difficulty_bins": {"all": [0, 25]},
                "difficulty_weights": {"all": 1.0},
            },
            seed=42
        )

        assert len(dataset) == 100

        # 모든 샘플이 correct인지 검증
        all_correct = all(sample["is_correct"] for sample in dataset)
        assert all_correct, "Pointwise 모드는 correct 샘플만 반환해야 함"

    def test_pairwise_returns_pairs(self):
        """Pairwise 모드: correct_output, incorrect_output 쌍 반환"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 50,
                "use_pairwise": True,
                "max_pairs_per_problem": 20,
                "difficulty_bins": {"all": [0, 25]},
                "difficulty_weights": {"all": 1.0},
            },
            seed=42
        )

        assert len(dataset) == 50

        # 필수 필드 검증
        sample = dataset[0]
        assert "correct_output" in sample, "Pairwise 모드는 correct_output 필드 필요"
        assert "incorrect_output" in sample, "Pairwise 모드는 incorrect_output 필드 필요"


class TestDifficultyBasedSampling:
    """Difficulty-based curriculum 샘플링 테스트"""

    def test_difficulty_based_sampling_diff7_only(self):
        """diff_7 difficulty만 샘플링 검증"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 50,
                "use_pairwise": False,
                "max_pairs_per_problem": 20,
                "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
                "difficulty_weights": {"diff_7": 1.0, "else": 0.0},
            },
            seed=42
        )

        assert len(dataset) > 0

        # 모든 샘플이 diff_7 범위인지 검증
        for sample in dataset:
            difficulty = sample["metadata"]["difficulty"]
            assert difficulty == 7, f"Expected difficulty 7, got {difficulty}"

    def test_difficulty_based_sampling_mixed(self):
        """Mixed difficulty 샘플링 검증 (diff_7 + else)"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": 100,
                "use_pairwise": False,
                "max_pairs_per_problem": 20,
                "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
                "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
            },
            seed=42
        )

        # diff_7/else 분포 검증
        diff_7_count = sum(
            1 for sample in dataset
            if sample["metadata"]["difficulty"] == 7
        )
        else_count = sum(
            1 for sample in dataset
            if 8 <= sample["metadata"]["difficulty"] <= 25
        )

        # diff_7=35%, else=65% 비율 검증 (오차 허용)
        assert diff_7_count < else_count, "else가 더 많아야 함 (65%)"
        assert diff_7_count > 0, "diff_7 샘플도 있어야 함 (35%)"


class TestReproducibility:
    """재현성 테스트 (동일 seed → 동일 샘플)"""

    def test_same_seed_same_samples(self):
        """동일 seed로 두 번 로딩 시 동일한 샘플 반환"""
        sampling_config = {
            "n_samples": 20,
            "use_pairwise": False,
            "max_pairs_per_problem": 20,
            "difficulty_bins": {"all": [0, 25]},
            "difficulty_weights": {"all": 1.0},
        }

        dataset1 = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42
        )

        dataset2 = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42
        )

        # task_id 비교 (샘플 순서까지 동일해야 함)
        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2, "동일 seed면 동일한 샘플 순서여야 함"

    def test_different_seed_different_samples(self):
        """다른 seed로 로딩 시 다른 샘플 반환"""
        sampling_config = {
            "n_samples": 20,
            "use_pairwise": False,
            "max_pairs_per_problem": 20,
            "difficulty_bins": {"all": [0, 25]},
            "difficulty_weights": {"all": 1.0},
        }

        dataset1 = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42
        )

        dataset2 = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config,
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
            sampling_config={
                "n_samples": 10,
                "use_pairwise": False,
                "max_pairs_per_problem": 20,
                "difficulty_bins": {"all": [0, 25]},
                "difficulty_weights": {"all": 1.0},
            },
            seed=42
        )
        assert len(dataset) == 10

    def test_load_validation_split(self):
        """Validation 스플릿 로딩"""
        dataset = load_dataset(
            "codecontests",
            split="validation",
            sampling_config={
                "n_samples": 10,
                "use_pairwise": False,
                "max_pairs_per_problem": 20,
                "difficulty_bins": {"all": [0, 25]},
                "difficulty_weights": {"all": 1.0},
            },
            seed=42
        )
        assert len(dataset) == 10

    def test_load_test_split_uses_evaluation_loader(self):
        """Test 스플릿은 load_evaluation_dataset() 사용

        test 스플릿은 problem_index_map이 없으므로
        load_evaluation_dataset()로 전체 로드해야 함
        """
        from weighted_mtp.data.datasets import load_evaluation_dataset

        dataset = load_evaluation_dataset("codecontests", split="test")
        assert len(dataset) > 0


class TestSampleUniquePairs:
    """_sample_unique_pairs() 함수 단위 테스트

    검증 항목:
    - 모든 correct_idx unique
    - 모든 incorrect_idx unique
    - 모든 pair unique
    - n_samples 정확히 달성 (또는 에러)
    """

    @pytest.fixture
    def mock_problem_index_map(self):
        """시뮬레이션용 problem_index_map 생성"""
        # 10개 problem, 각각 correct 5개, incorrect 5개
        problem_map = {}
        base_idx = 0
        for i in range(10):
            pid = f"problem_{i}"
            problem_map[pid] = {
                "difficulty": i % 3,  # 0, 1, 2 순환
                "correct_indices": list(range(base_idx, base_idx + 5)),
                "incorrect_indices": list(range(base_idx + 100, base_idx + 105)),
            }
            base_idx += 5
        return problem_map

    def test_all_correct_idx_unique(self, mock_problem_index_map):
        """모든 correct_idx가 unique한지 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs

        pairs = _sample_unique_pairs(
            problem_index_map=mock_problem_index_map,
            n_samples=30,
            difficulty_weights={"low": 0.5, "mid": 0.5},
            difficulty_bins={"low": [0, 1], "mid": [2, 2]},
            seed=42,
            max_pairs_per_problem=10,
        )

        correct_indices = [p["correct_idx"] for p in pairs]
        assert len(correct_indices) == len(set(correct_indices)), "correct_idx에 중복 존재"

    def test_all_incorrect_idx_unique(self, mock_problem_index_map):
        """모든 incorrect_idx가 unique한지 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs

        pairs = _sample_unique_pairs(
            problem_index_map=mock_problem_index_map,
            n_samples=30,
            difficulty_weights={"low": 0.5, "mid": 0.5},
            difficulty_bins={"low": [0, 1], "mid": [2, 2]},
            seed=42,
            max_pairs_per_problem=10,
        )

        incorrect_indices = [p["incorrect_idx"] for p in pairs]
        assert len(incorrect_indices) == len(set(incorrect_indices)), "incorrect_idx에 중복 존재"

    def test_all_pairs_unique(self, mock_problem_index_map):
        """모든 (correct, incorrect) pair가 unique한지 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs

        pairs = _sample_unique_pairs(
            problem_index_map=mock_problem_index_map,
            n_samples=30,
            difficulty_weights={"low": 0.5, "mid": 0.5},
            difficulty_bins={"low": [0, 1], "mid": [2, 2]},
            seed=42,
            max_pairs_per_problem=10,
        )

        pair_tuples = [(p["correct_idx"], p["incorrect_idx"]) for p in pairs]
        assert len(pair_tuples) == len(set(pair_tuples)), "pair에 중복 존재"

    def test_exact_n_samples(self, mock_problem_index_map):
        """요청한 n_samples 정확히 반환하는지 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs

        n_samples = 25
        pairs = _sample_unique_pairs(
            problem_index_map=mock_problem_index_map,
            n_samples=n_samples,
            difficulty_weights={"low": 0.5, "mid": 0.5},
            difficulty_bins={"low": [0, 1], "mid": [2, 2]},
            seed=42,
            max_pairs_per_problem=10,
        )

        assert len(pairs) == n_samples, f"Expected {n_samples}, got {len(pairs)}"

    def test_max_pairs_per_problem_enforced(self, mock_problem_index_map):
        """max_pairs_per_problem 제한이 적용되는지 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs
        from collections import Counter

        # max_pairs_per_problem=3: low(7 problems×3)=21, mid(3 problems×3)=9
        # weight 50/50 분배 시: low_target=9, mid_target=9 → 18개 가용
        pairs = _sample_unique_pairs(
            problem_index_map=mock_problem_index_map,
            n_samples=18,  # 가용 범위 내
            difficulty_weights={"low": 0.5, "mid": 0.5},
            difficulty_bins={"low": [0, 1], "mid": [2, 2]},
            seed=42,
            max_pairs_per_problem=3,  # 낮은 cap
        )

        problem_counts = Counter(p["problem_id"] for p in pairs)
        max_count = max(problem_counts.values())
        assert max_count <= 3, f"max_pairs_per_problem=3인데 {max_count}개 쌍 발생"

    def test_insufficient_data_raises_error(self, mock_problem_index_map):
        """데이터 부족 시 ValueError 발생 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs

        # 50개 unique pair밖에 생성 못함 (10 problems × 5 pairs)
        with pytest.raises(ValueError, match="데이터 부족"):
            _sample_unique_pairs(
                problem_index_map=mock_problem_index_map,
                n_samples=100,  # 가용 50개보다 큰 값
                difficulty_weights={"low": 0.5, "mid": 0.5},
                difficulty_bins={"low": [0, 1], "mid": [2, 2]},
                seed=42,
                max_pairs_per_problem=5,
            )

    def test_reproducibility_with_same_seed(self, mock_problem_index_map):
        """동일 seed로 동일한 결과 반환 검증"""
        from weighted_mtp.data.datasets import _sample_unique_pairs

        config = {
            "problem_index_map": mock_problem_index_map,
            "n_samples": 20,
            "difficulty_weights": {"low": 0.5, "mid": 0.5},
            "difficulty_bins": {"low": [0, 1], "mid": [2, 2]},
            "seed": 42,
            "max_pairs_per_problem": 10,
        }

        pairs1 = _sample_unique_pairs(**config)
        pairs2 = _sample_unique_pairs(**config)

        assert pairs1 == pairs2, "동일 seed면 동일한 결과여야 함"

    def test_performance_large_scale(self):
        """대규모 샘플링 성능 테스트 (100,000 samples < 2초)"""
        import time
        from weighted_mtp.data.datasets import _sample_unique_pairs

        # 대규모 mock 생성: 10,000 problems, 각각 correct 20개, incorrect 20개
        large_map = {}
        base_idx = 0
        for i in range(10000):
            pid = f"problem_{i}"
            large_map[pid] = {
                "difficulty": i % 10,  # 0-9 difficulty
                "correct_indices": list(range(base_idx, base_idx + 20)),
                "incorrect_indices": list(range(base_idx + 1000000, base_idx + 1000020)),
            }
            base_idx += 20

        start = time.time()
        pairs = _sample_unique_pairs(
            problem_index_map=large_map,
            n_samples=100000,
            difficulty_weights={"all": 1.0},
            difficulty_bins={"all": [0, 25]},
            seed=42,
            max_pairs_per_problem=20,
        )
        elapsed = time.time() - start

        assert len(pairs) == 100000, f"Expected 100000, got {len(pairs)}"
        assert elapsed < 2.0, f"Expected <2s, got {elapsed:.2f}s"

        # unique 검증
        correct_indices = [p["correct_idx"] for p in pairs]
        incorrect_indices = [p["incorrect_idx"] for p in pairs]
        assert len(correct_indices) == len(set(correct_indices)), "대규모에서 correct_idx 중복"
        assert len(incorrect_indices) == len(set(incorrect_indices)), "대규모에서 incorrect_idx 중복"
