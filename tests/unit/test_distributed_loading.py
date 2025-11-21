"""분산 데이터 로딩 단위 테스트 (M3 MacBook Pro 호환)

M3 환경에서 안전하게 실행 가능한 CPU 기반 단위 테스트.
실제 multiprocessing 없이 로직만 검증하여 100% 안정성 보장.
"""

import pytest
from pathlib import Path
from weighted_mtp.data.datasets import load_dataset


class TestDistributedLoading:
    """분산 데이터 로딩 로직 검증 (CPU 전용)"""

    @pytest.mark.parametrize("world_size,rank", [
        (1, 0),  # 단일 프로세스
        (2, 0), (2, 1),  # 2 GPU
        (4, 0), (4, 1), (4, 2), (4, 3),  # 4 GPU
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),  # 8 GPU
    ])
    def test_rank_aware_sampling_coverage(self, world_size, rank):
        """각 rank가 올바른 샘플 수를 로드하는지 검증"""
        total_samples = 20

        # 데이터 로드
        dataset = load_dataset(
            dataset_name="mbpp",
            split="train",
            n_samples=total_samples,
            auto_data_balancing=False,
            correct_ratio=1.0,
            rank=rank,
            world_size=world_size,
            seed=42,
        )

        # 기대 샘플 수 계산
        base_per_rank = total_samples // world_size
        remainder = total_samples % world_size
        expected = base_per_rank + (1 if rank < remainder else 0)

        # 검증
        assert len(dataset) == expected, (
            f"Rank {rank}/{world_size}: "
            f"expected {expected} samples, got {len(dataset)}"
        )

    def test_rank_aware_no_overlap(self):
        """모든 rank의 데이터가 중복 없이 분산되는지 검증"""
        world_size = 4
        total_samples = 20
        all_task_ids = []

        # 모든 rank의 데이터 수집
        for rank in range(world_size):
            dataset = load_dataset(
                dataset_name="mbpp",
                split="train",
                n_samples=total_samples,
                rank=rank,
                world_size=world_size,
                seed=42,
            )

            # task_id 수집
            task_ids = [sample["task_id"] for sample in dataset]
            all_task_ids.extend(task_ids)

        # 중복 검증
        unique_ids = set(all_task_ids)
        assert len(all_task_ids) == len(unique_ids), (
            f"Found {len(all_task_ids) - len(unique_ids)} duplicate samples across ranks"
        )

    def test_rank_aware_full_coverage(self):
        """모든 rank의 데이터를 합치면 100% 커버하는지 검증"""
        world_size = 4
        total_samples = 20
        all_task_ids = set()

        # 모든 rank의 데이터 수집
        for rank in range(world_size):
            dataset = load_dataset(
                dataset_name="mbpp",
                split="train",
                n_samples=total_samples,
                rank=rank,
                world_size=world_size,
                seed=42,
            )

            # task_id 수집
            task_ids = {sample["task_id"] for sample in dataset}
            all_task_ids.update(task_ids)

        # 전체 커버리지 검증
        assert len(all_task_ids) == total_samples, (
            f"Coverage: {len(all_task_ids)}/{total_samples} "
            f"({len(all_task_ids)/total_samples*100:.1f}%)"
        )

    def test_rank_aware_deterministic(self):
        """같은 seed로 여러 번 로드 시 동일한 결과 반환하는지 검증"""
        world_size = 4
        rank = 1
        total_samples = 20

        # 첫 번째 로드
        dataset1 = load_dataset(
            dataset_name="mbpp",
            split="train",
            n_samples=total_samples,
            rank=rank,
            world_size=world_size,
            seed=42,
        )
        task_ids1 = [sample["task_id"] for sample in dataset1]

        # 두 번째 로드 (같은 seed)
        dataset2 = load_dataset(
            dataset_name="mbpp",
            split="train",
            n_samples=total_samples,
            rank=rank,
            world_size=world_size,
            seed=42,
        )
        task_ids2 = [sample["task_id"] for sample in dataset2]

        # 재현성 검증
        assert task_ids1 == task_ids2, (
            "Same seed should produce identical results"
        )

    @pytest.mark.parametrize("world_size", [2, 4, 8])
    def test_rank_aware_balanced_distribution(self, world_size):
        """각 rank가 균등하게 데이터를 분배받는지 검증"""
        total_samples = 20
        sample_counts = []

        for rank in range(world_size):
            dataset = load_dataset(
                dataset_name="mbpp",
                split="train",
                n_samples=total_samples,
                rank=rank,
                world_size=world_size,
                seed=42,
            )
            sample_counts.append(len(dataset))

        # 최대 차이가 1 이하인지 검증 (균등 분배)
        max_diff = max(sample_counts) - min(sample_counts)
        assert max_diff <= 1, (
            f"Unbalanced distribution: {sample_counts}, max_diff={max_diff}"
        )
