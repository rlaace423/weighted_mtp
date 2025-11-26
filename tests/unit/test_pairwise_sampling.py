"""Pairwise 샘플링 기능 검증 테스트

핵심 검증 항목:
- use_pairwise=True 시 동일 problem_id 내에서 (correct, incorrect) 쌍 생성
- Difficulty 기반 인덱스 계산 후 Pairwise 포맷 변환
- Rank-aware 분산 샘플링
"""

import pytest
import json
from pathlib import Path
from collections import Counter

from weighted_mtp.data.datasets import (
    _create_pairs_from_indices,
    _read_jsonl_pairwise,
    load_dataset,
)


@pytest.fixture(scope="module")
def train_metadata():
    """Train metadata 로딩"""
    metadata_path = Path("storage/datasets/codecontests/processed/train_metadata.json")
    if not metadata_path.exists():
        pytest.skip(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        data = json.load(f)

    return data["metadata"]


class TestCreatePairsFromIndices:
    """_create_pairs_from_indices() 함수 테스트"""

    def _get_balanced_indices(self, train_metadata, n_samples=200):
        """테스트용: correct/incorrect 균형 인덱스 생성"""
        import random
        random.seed(42)

        correct_indices = [i for i, m in enumerate(train_metadata) if m.get("is_correct")]
        incorrect_indices = [i for i, m in enumerate(train_metadata) if not m.get("is_correct")]

        n_each = n_samples // 2
        selected = random.sample(correct_indices, min(n_each, len(correct_indices)))
        selected += random.sample(incorrect_indices, min(n_each, len(incorrect_indices)))
        random.shuffle(selected)
        return selected

    def test_basic_pair_generation(self, train_metadata):
        """기본 쌍 생성 검증"""
        indices = self._get_balanced_indices(train_metadata, n_samples=200)

        pairs = _create_pairs_from_indices(
            indices=indices,
            metadata=train_metadata,
            seed=42,
        )

        assert len(pairs) > 0, "쌍이 생성되지 않음"

        # 각 쌍의 구조 검증
        for pair in pairs:
            assert "correct_idx" in pair
            assert "incorrect_idx" in pair
            assert isinstance(pair["correct_idx"], int)
            assert isinstance(pair["incorrect_idx"], int)

    def test_same_problem_id_pairs(self, train_metadata):
        """동일 problem_id 쌍인지 검증"""
        indices = self._get_balanced_indices(train_metadata, n_samples=200)

        pairs = _create_pairs_from_indices(
            indices=indices,
            metadata=train_metadata,
            seed=42,
        )

        mismatched = []
        for pair in pairs:
            correct_problem = train_metadata[pair["correct_idx"]]["problem_id"]
            incorrect_problem = train_metadata[pair["incorrect_idx"]]["problem_id"]

            if correct_problem != incorrect_problem:
                mismatched.append((correct_problem, incorrect_problem))

        assert len(mismatched) == 0, \
            f"Problem ID 불일치 쌍 발견: {mismatched[:5]}..."

    def test_correct_incorrect_labels(self, train_metadata):
        """Correct/Incorrect 레이블 검증"""
        indices = self._get_balanced_indices(train_metadata, n_samples=200)

        pairs = _create_pairs_from_indices(
            indices=indices,
            metadata=train_metadata,
            seed=42,
        )

        for pair in pairs:
            correct_label = train_metadata[pair["correct_idx"]]["is_correct"]
            incorrect_label = train_metadata[pair["incorrect_idx"]]["is_correct"]

            assert correct_label is True, \
                f"correct_idx가 is_correct=False: idx={pair['correct_idx']}"
            assert incorrect_label is False, \
                f"incorrect_idx가 is_correct=True: idx={pair['incorrect_idx']}"

    def test_deterministic_with_seed(self, train_metadata):
        """동일 seed는 동일 결과"""
        indices = self._get_balanced_indices(train_metadata, n_samples=200)

        pairs1 = _create_pairs_from_indices(indices, train_metadata, seed=123)
        pairs2 = _create_pairs_from_indices(indices, train_metadata, seed=123)

        for p1, p2 in zip(pairs1, pairs2):
            assert p1["correct_idx"] == p2["correct_idx"]
            assert p1["incorrect_idx"] == p2["incorrect_idx"]

    def test_different_seeds_different_results(self, train_metadata):
        """다른 seed는 다른 순서"""
        # 더 많은 샘플로 충분한 쌍 생성
        indices = self._get_balanced_indices(train_metadata, n_samples=500)

        pairs1 = _create_pairs_from_indices(indices, train_metadata, seed=111)
        pairs2 = _create_pairs_from_indices(indices, train_metadata, seed=222)

        # 쌍의 개수는 동일해야 함
        assert len(pairs1) == len(pairs2), "동일 인덱스에서 쌍 개수가 다름"

        # 쌍이 충분히 있어야 shuffle 테스트 의미 있음
        if len(pairs1) < 5:
            pytest.skip(f"쌍이 너무 적음 ({len(pairs1)}개) - shuffle 테스트 불가")

        # 순서가 달라야 함 (전체 리스트 비교)
        pairs1_tuples = [(p["correct_idx"], p["incorrect_idx"]) for p in pairs1]
        pairs2_tuples = [(p["correct_idx"], p["incorrect_idx"]) for p in pairs2]

        assert pairs1_tuples != pairs2_tuples, "다른 seed인데 순서가 동일"

    def test_problem_diversity(self, train_metadata):
        """다양한 problem_id에서 쌍 생성"""
        indices = self._get_balanced_indices(train_metadata, n_samples=500)

        pairs = _create_pairs_from_indices(
            indices=indices,
            metadata=train_metadata,
            seed=42,
        )

        problem_ids = set()
        for pair in pairs:
            problem_ids.add(train_metadata[pair["correct_idx"]]["problem_id"])

        # 최소 5개 이상의 다른 problem에서 샘플링
        assert len(problem_ids) >= 5, \
            f"Problem 다양성 부족: {len(problem_ids)} problems"


class TestReadJsonlPairwise:
    """_read_jsonl_pairwise() 함수 테스트"""

    def _get_balanced_indices(self, train_metadata, n_samples=200):
        """테스트용: correct/incorrect 균형 인덱스 생성"""
        import random
        random.seed(42)

        correct_indices = [i for i, m in enumerate(train_metadata) if m.get("is_correct")]
        incorrect_indices = [i for i, m in enumerate(train_metadata) if not m.get("is_correct")]

        n_each = n_samples // 2
        selected = random.sample(correct_indices, min(n_each, len(correct_indices)))
        selected += random.sample(incorrect_indices, min(n_each, len(incorrect_indices)))
        random.shuffle(selected)
        return selected

    def test_read_pairwise_data(self, train_metadata):
        """Pairwise 데이터 읽기 검증"""
        # 인덱스 생성 후 쌍 생성
        indices = self._get_balanced_indices(train_metadata, n_samples=200)
        pairs = _create_pairs_from_indices(indices, train_metadata, seed=42)

        jsonl_path = Path("storage/datasets/codecontests/processed/train.jsonl")
        if not jsonl_path.exists():
            pytest.skip(f"JSONL not found: {jsonl_path}")

        # 데이터 읽기 (최대 10쌍)
        pairs_to_read = pairs[:10]
        data = _read_jsonl_pairwise(jsonl_path, pairs_to_read)

        assert len(data) == len(pairs_to_read)

        # 각 항목 구조 검증
        for item in data:
            assert "instruction" in item
            assert "input" in item
            assert "correct_output" in item
            assert "incorrect_output" in item

            # 실제 내용이 있는지 확인
            assert len(item["instruction"]) > 0
            assert len(item["correct_output"]) > 0
            assert len(item["incorrect_output"]) > 0

    def test_correct_incorrect_different(self, train_metadata):
        """correct_output과 incorrect_output이 다른지 확인"""
        indices = self._get_balanced_indices(train_metadata, n_samples=200)
        pairs = _create_pairs_from_indices(indices, train_metadata, seed=42)

        jsonl_path = Path("storage/datasets/codecontests/processed/train.jsonl")
        if not jsonl_path.exists():
            pytest.skip(f"JSONL not found: {jsonl_path}")

        pairs_to_read = pairs[:20]
        data = _read_jsonl_pairwise(jsonl_path, pairs_to_read)

        same_output_count = 0
        for item in data:
            if item["correct_output"] == item["incorrect_output"]:
                same_output_count += 1

        # 동일한 output인 쌍이 너무 많으면 안 됨
        assert same_output_count < len(data) * 0.1, \
            f"correct_output == incorrect_output인 쌍이 너무 많음: {same_output_count}/{len(data)}"


class TestPairwiseDatasetLoading:
    """load_dataset() pairwise 모드 테스트"""

    def _get_pairwise_sampling_config(self, n_samples: int) -> dict:
        """Pairwise 모드용 sampling_config 생성"""
        return {
            "seed": 42,
            "use_pairwise": True,
            "n_samples": n_samples,
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

    def test_load_pairwise_dataset(self):
        """Pairwise 데이터셋 로딩"""
        # n_samples=200으로 설정하면 correct/incorrect 각 100개
        # 동일 problem_id 내 쌍 생성으로 실제 쌍 수는 더 적을 수 있음
        sampling_config = self._get_pairwise_sampling_config(n_samples=200)

        dataset = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
            rank=0,
            world_size=1,
        )

        # 쌍이 생성되었는지 확인 (정확한 개수는 데이터 분포에 따라 다름)
        assert len(dataset) > 0, "Pairwise 데이터셋이 비어있음"

        # 첫 번째 항목 검증
        item = dataset[0]
        assert "instruction" in item
        assert "input" in item
        assert "correct_output" in item
        assert "incorrect_output" in item

    def test_distributed_pairwise_sampling(self):
        """분산 환경 pairwise 샘플링 (rank별 분할)"""
        sampling_config = self._get_pairwise_sampling_config(n_samples=500)

        # Rank 0, World Size 2
        dataset_rank0 = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
            rank=0,
            world_size=2,
        )

        # Rank 1, World Size 2
        dataset_rank1 = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
            rank=1,
            world_size=2,
        )

        # 각 rank가 대략 절반씩
        assert len(dataset_rank0) > 0
        assert len(dataset_rank1) > 0

        # 두 rank의 합이 전체와 유사해야 함
        total_pairs = len(dataset_rank0) + len(dataset_rank1)

        # 단일 프로세스로 로드한 전체 개수와 비교
        dataset_full = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
            rank=0,
            world_size=1,
        )

        # 분산 처리 결과와 단일 처리 결과가 유사해야 함
        assert abs(total_pairs - len(dataset_full)) <= 1, \
            f"분산 처리 합계({total_pairs})와 단일 처리({len(dataset_full)})가 다름"

        # 중복 없는지 확인 (correct_output으로 비교)
        outputs_0 = {d["correct_output"][:100] for d in dataset_rank0}
        outputs_1 = {d["correct_output"][:100] for d in dataset_rank1}

        # 완전히 중복 없을 필요는 없지만, 적어도 일부는 달라야 함
        overlap = len(outputs_0 & outputs_1)
        assert overlap < len(outputs_0), \
            "Rank간 데이터가 완전히 동일함 - 분산 샘플링 문제"
