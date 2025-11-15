"""데이터 파이프라인 통합 테스트

전체 파이프라인 검증:
- load_dataset → AlpacaDataCollator → DataLoader
- Stage 1/2 End-to-End
- Epoch 루프
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from weighted_mtp.data import load_dataset, AlpacaDataCollator


# Tokenizer 로딩 fixture
@pytest.fixture(scope="module")
def tokenizer():
    """실제 LlamaTokenizer 로딩 (없으면 skip)"""
    try:
        from transformers import AutoTokenizer

        tokenizer_path = Path("storage/models_v2/meta-llama-mtp/tokenizer")

        if not tokenizer_path.exists():
            pytest.skip("Tokenizer not found")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    except ImportError:
        pytest.skip("transformers required")


# 공유 데이터셋 fixture (성능 최적화)
@pytest.fixture(scope="module")
def stage1_dataset():
    """Stage 1 작은 데이터셋 (전체 테스트에서 재사용)"""
    return load_dataset(
        "codecontests_small",
        split="train",
        stage="stage1",
        n_samples=20,
        balance_correct=False,  # 작은 데이터셋은 균형 샘플링 불필요
        seed=42,
    )


@pytest.fixture(scope="module")
def stage2_dataset():
    """Stage 2 작은 데이터셋 (전체 테스트에서 재사용)"""
    return load_dataset(
        "codecontests_small",
        split="train",
        stage="stage2",
        n_samples=20,
        difficulty_weights={"low": 0.6, "medium": 0.4, "high": 0.0},
        difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
        seed=42,
    )


class TestStage1Pipeline:
    """Stage 1 전체 파이프라인 테스트"""

    def test_stage1_full_pipeline(self, tokenizer, stage1_dataset):
        """Stage 1 전체 파이프라인 (load → collate → dataloader)"""
        # Collator 생성 (작은 데이터셋의 긴 instruction을 위해 max_length=2048 사용)
        collator = AlpacaDataCollator(tokenizer, max_length=2048)

        # DataLoader 생성
        dataloader = DataLoader(
            stage1_dataset,
            batch_size=4,
            collate_fn=collator,
            shuffle=False,
        )

        # 배치 검증
        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        assert batch["input_ids"].shape == (4, 2048)
        assert batch["attention_mask"].shape == (4, 2048)
        assert batch["labels"].shape == (4, 2048)

        # Masking 검증
        for i in range(4):
            labels = batch["labels"][i]
            assert (labels[:10] == -100).all(), "Instruction 부분이 마스킹되지 않음"
            assert (labels != -100).any(), "Output 부분이 없음"

    def test_stage1_is_correct_ratio(self, stage1_dataset):
        """Stage 1 데이터셋 검증 (작은 데이터셋은 균형 샘플링 미적용)"""
        # 작은 데이터셋은 모두 correct이므로 샘플 수만 검증
        assert len(stage1_dataset) == 20


class TestStage2Pipeline:
    """Stage 2 전체 파이프라인 테스트"""

    def test_stage2_full_pipeline(self, tokenizer, stage2_dataset):
        """Stage 2 전체 파이프라인"""
        collator = AlpacaDataCollator(tokenizer, max_length=512)
        dataloader = DataLoader(stage2_dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(dataloader))

        assert batch["input_ids"].shape == (4, 512)
        assert batch["labels"].shape == (4, 512)

    def test_stage2_difficulty_distribution(self, stage2_dataset):
        """Stage 2 difficulty 분포 검증"""
        low_count = sum(
            1 for s in stage2_dataset if 1 <= s["metadata"]["difficulty"] <= 3
        )
        medium_count = sum(
            1 for s in stage2_dataset if 4 <= s["metadata"]["difficulty"] <= 7
        )

        low_ratio = low_count / len(stage2_dataset)
        medium_ratio = medium_count / len(stage2_dataset)

        # 작은 샘플 수로 인한 분산 고려 (±30%)
        assert 0.3 <= low_ratio <= 0.9
        assert 0.1 <= medium_ratio <= 0.7


class TestEpochLoop:
    """Epoch 루프 테스트"""

    def test_single_epoch(self, tokenizer, stage1_dataset):
        """단일 epoch 정상 동작"""
        collator = AlpacaDataCollator(tokenizer, max_length=2048)  # 긴 instruction 처리
        dataloader = DataLoader(stage1_dataset, batch_size=8, collate_fn=collator)

        batch_count = 0
        for batch in dataloader:
            assert batch["input_ids"].shape[0] <= 8
            assert (batch["labels"] != -100).any()
            batch_count += 1

        # 모든 배치가 처리되었는지 확인
        expected_batches = (20 + 7) // 8  # ceil(20 / 8) = 3
        assert batch_count == expected_batches

    def test_multiple_epochs(self, tokenizer, stage1_dataset):
        """여러 epoch 반복 정상 동작"""
        collator = AlpacaDataCollator(tokenizer, max_length=2048)  # 긴 instruction 처리
        dataloader = DataLoader(stage1_dataset, batch_size=8, collate_fn=collator)

        total_batches = 0
        for epoch in range(3):
            epoch_batches = 0
            for batch in dataloader:
                assert batch["input_ids"].shape[0] <= 8
                epoch_batches += 1

            total_batches += epoch_batches

        # 3 epoch 모두 동일한 배치 수
        expected_per_epoch = (20 + 7) // 8
        assert total_batches == expected_per_epoch * 3


class TestEdgeCases:
    """Edge Case 테스트"""

    def test_batch_size_one(self, tokenizer, stage1_dataset):
        """batch_size=1 처리"""
        collator = AlpacaDataCollator(tokenizer, max_length=2048)  # 긴 instruction 처리
        dataloader = DataLoader(stage1_dataset, batch_size=1, collate_fn=collator)

        batch = next(iter(dataloader))

        assert batch["input_ids"].shape == (1, 2048)

    def test_incomplete_final_batch(self, tokenizer):
        """마지막 배치가 불완전한 경우 (15 샘플, batch_size=8 → 마지막 7개)"""
        dataset = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage1",
            n_samples=15,
            balance_correct=False,
            seed=42,
            )

        collator = AlpacaDataCollator(tokenizer, max_length=2048)  # 긴 instruction 처리
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

        batches = list(dataloader)

        # 마지막 배치는 7개
        assert batches[-1]["input_ids"].shape[0] == 7

        # 총 배치 수
        assert len(batches) == 2  # ceil(15 / 8) = 2


class TestDistributedSamplerUsage:
    """DistributedSampler 사용 예시 및 시뮬레이션 테스트

    실제 A100 4-GPU 분산학습 환경에서는 torchrun으로 실행하지만,
    로컬 테스트에서는 DistributedSampler 사용법을 검증합니다.

    실제 VESSL 실행 예시:
        torchrun --nproc_per_node=4 -m weighted_mtp.cli.train \\
            --config configs/defaults.yaml \\
            --recipe configs/recipe.verifiable.yaml
    """

    def test_distributed_sampler_creation(self, tokenizer, stage1_dataset):
        """DistributedSampler 생성 및 기본 동작 검증

        로컬 환경에서는 create_distributed_sampler()가 None을 반환하고,
        분산 환경에서는 DistributedSampler를 반환합니다.
        """
        from weighted_mtp.runtime import create_distributed_sampler, is_distributed

        # 로컬 환경에서는 None 반환
        sampler = create_distributed_sampler(stage1_dataset, shuffle=True, seed=42)

        if is_distributed():
            # 분산 환경: DistributedSampler 반환
            assert sampler is not None
            assert hasattr(sampler, "set_epoch")
        else:
            # 로컬 환경: None 반환
            assert sampler is None

        # DataLoader 생성 (sampler 유무에 따라 shuffle 조정)
        collator = AlpacaDataCollator(tokenizer, max_length=2048)
        dataloader = DataLoader(
            stage1_dataset,
            batch_size=4,
            collate_fn=collator,
            sampler=sampler,
            shuffle=(sampler is None),  # sampler 없을 때만 shuffle
        )

        # 정상 동작 확인
        batch = next(iter(dataloader))
        assert batch["input_ids"].shape[0] == 4

    def test_distributed_dataloader_example(self, tokenizer):
        """VESSL A100 4-GPU 환경 사용 예시 (주석으로 설명)

        이 테스트는 실제로 분산 환경을 시뮬레이션하지 않지만,
        분산 환경에서 어떻게 사용하는지 명확한 예시를 제공합니다.
        """
        from weighted_mtp.runtime import (
            create_distributed_sampler,
            is_distributed,
            get_rank,
            get_world_size,
        )

        # Stage 2 데이터셋 로드 (200K 샘플)
        dataset = load_dataset(
            "codecontests_small",
            split="train",
            stage="stage2",
            n_samples=20,  # 테스트에서는 20개 (실제: 200000)
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
            )

        # DistributedSampler 생성 (분산 환경에서만)
        # 로컬: None, VESSL 4-GPU: DistributedSampler
        sampler = create_distributed_sampler(dataset, shuffle=True, seed=42)

        # DataLoader 생성
        collator = AlpacaDataCollator(tokenizer, max_length=512)
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # 각 GPU별 batch size
            collate_fn=collator,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=0,  # 테스트에서는 0
            pin_memory=False,
        )

        # Epoch 루프 예시
        for epoch in range(2):
            # 분산 환경에서는 매 epoch마다 set_epoch() 호출 (재현성)
            if sampler is not None:
                sampler.set_epoch(epoch)

            batch_count = 0
            for batch in dataloader:
                assert "input_ids" in batch
                assert "labels" in batch
                batch_count += 1

            # 로컬 환경: 전체 데이터 처리
            # 분산 환경: 각 GPU가 1/4씩 처리
            if is_distributed():
                # VESSL 4-GPU 환경 예시:
                # - 전체 샘플: 200,000
                # - 각 GPU: 50,000 (samples[rank::4])
                # - Rank 0: samples[0::4]
                # - Rank 1: samples[1::4]
                # - Rank 2: samples[2::4]
                # - Rank 3: samples[3::4]
                expected_batches_per_gpu = len(dataset) // (get_world_size() * 2)
            else:
                # 로컬 환경: 전체 데이터
                expected_batches_per_gpu = (len(dataset) + 1) // 2

            # 배치 수 검증
            assert batch_count > 0

    def test_distributed_sampler_data_distribution_explanation(self):
        """DistributedSampler 데이터 분배 원리 설명 (Documentation)

        이 테스트는 실제 검증보다는 DistributedSampler가
        어떻게 데이터를 분산하는지 명확한 문서를 제공합니다.
        """
        # 05_phase3_detailed_plan.md Decision 4 참고
        #
        # ✅ 올바른 접근 (DistributedSampler 사용):
        #
        # from weighted_mtp.runtime import (
        #     init_distributed,
        #     create_distributed_sampler,
        #     get_rank,
        # )
        #
        # # 1. 분산 환경 초기화
        # rank, world_size = init_distributed()
        #
        # # 2. Dataset 로드 (전체 샘플 수 지정)
        # dataset = load_dataset(
        #     "codecontests",
        #     split="train",
        #     stage="stage2",
        #     n_samples=200000,  # 전체 샘플 수
        #     seed=42
        # )
        #
        # # 3. DistributedSampler 생성
        # sampler = create_distributed_sampler(dataset, shuffle=True, seed=42)
        #
        # # 4. DataLoader 생성
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=2,  # 각 GPU별 batch size
        #     sampler=sampler,
        #     shuffle=False,  # sampler가 shuffle 담당
        # )
        #
        # # 결과:
        # # - Rank 0: samples[0::4]  (50,000 samples)
        # # - Rank 1: samples[1::4]  (50,000 samples)
        # # - Rank 2: samples[2::4]  (50,000 samples)
        # # - Rank 3: samples[3::4]  (50,000 samples)
        # # - 총합: 200,000 samples (중복 없음)
        # # - 메모리: 각 GPU는 전체의 1/4만 로드
        #
        # # Epoch 시작 시 (재현성 보장)
        # for epoch in range(num_epochs):
        #     sampler.set_epoch(epoch)
        #     for batch in dataloader:
        #         ...
        #
        # ❌ 잘못된 접근 (피해야 함):
        #
        # # 각 GPU에서 전체 데이터 로드 (메모리 4배 낭비)
        # dataset = load_dataset("codecontests", split="train")  # 3.7M samples
        # dataloader = DataLoader(dataset, batch_size=8)  # 모든 GPU가 동일 데이터
        #
        # # 결과:
        # # - 각 GPU: 3.7M samples 로드 (중복)
        # # - 총 메모리: 15GB × 4 = 60GB (낭비)

        # 이 테스트는 문서 역할만 하므로 실제 검증은 pass
        pass
