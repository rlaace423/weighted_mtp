"""Pairwise 샘플링 기능 검증 테스트

핵심 검증 항목:
- use_pairwise=True 시 동일 problem_id 내에서 (correct, incorrect) 쌍 생성
- Difficulty 기반 인덱스 계산 후 Pairwise 포맷 변환
- Rank-aware 분산 샘플링
"""

import pytest

from weighted_mtp.data.datasets import load_dataset


class TestPairwiseDatasetLoading:
    """load_dataset() pairwise 모드 테스트"""

    def _get_pairwise_sampling_config(self, n_samples: int) -> dict:
        """Pairwise 모드용 sampling_config 생성"""
        return {
            "seed": 42,
            "use_pairwise": True,
            "n_samples": n_samples,
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
        sampling_config = self._get_pairwise_sampling_config(n_samples=200)

        dataset = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
            rank=0,
            world_size=1,
        )

        # 쌍이 생성되었는지 확인
        assert len(dataset) > 0, "Pairwise 데이터셋이 비어있음"

        # 첫 번째 항목 검증
        item = dataset[0]
        assert "instruction" in item
        assert "input" in item
        assert "correct_output" in item
        assert "incorrect_output" in item

    def test_pairwise_unique_outputs(self):
        """Pairwise 모드에서 correct/incorrect output이 다른지 검증"""
        sampling_config = self._get_pairwise_sampling_config(n_samples=50)

        dataset = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
        )

        # correct_output이 모두 다른지 검증 (unique pair 보장)
        correct_outputs = [d["correct_output"][:200] for d in dataset]
        assert len(correct_outputs) == len(set(correct_outputs)), \
            "correct_output이 중복됨 - unique pair 위반"

        # incorrect_output이 모두 다른지 검증
        incorrect_outputs = [d["incorrect_output"][:200] for d in dataset]
        assert len(incorrect_outputs) == len(set(incorrect_outputs)), \
            "incorrect_output이 중복됨 - unique pair 위반"

    def test_pairwise_correct_incorrect_different(self):
        """Pairwise 쌍에서 correct와 incorrect가 다른지 검증"""
        sampling_config = self._get_pairwise_sampling_config(n_samples=50)

        dataset = load_dataset(
            dataset_name="codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42,
        )

        for item in dataset:
            # correct_output과 incorrect_output이 달라야 함
            assert item["correct_output"] != item["incorrect_output"], \
                "correct_output과 incorrect_output이 동일함"

    def test_distributed_pairwise_sampling(self):
        """분산 환경 pairwise 샘플링 (rank별 분할)"""
        sampling_config = self._get_pairwise_sampling_config(n_samples=200)

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

        # 각 rank가 데이터를 가짐
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
