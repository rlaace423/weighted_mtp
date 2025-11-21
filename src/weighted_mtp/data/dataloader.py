"""DataLoader 생성 (분산 학습 지원)

4개 파이프라인의 중복 create_dataloader() 로직을 통합.
메타데이터 기반 Rank-aware 샘플링으로 각 GPU가 필요한 데이터만 로드.
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from weighted_mtp.data.datasets import load_dataset
from weighted_mtp.data.collators import AlpacaDataCollator
from weighted_mtp.runtime.distributed import get_rank, get_world_size


def create_dataloader(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    n_samples: int,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader 생성 (분산 학습 지원)

    메타데이터 기반으로 데이터셋을 효율적으로 로딩.
    분산 환경에서는 Rank-aware 샘플링으로 각 GPU가 필요한 데이터만 로드.

    Args:
        dataset_path: 데이터셋 경로 (예: storage/datasets/codecontests/processed/train.jsonl)
        tokenizer: Tokenizer
        batch_size: 배치 크기 (per GPU)
        max_length: 최대 시퀀스 길이
        n_samples: 전체 샘플 수
        balance_correct: is_correct 균형 샘플링 여부
        correct_ratio: correct 샘플 비율 (0.5 = 50:50)
        difficulty_weights: 난이도별 가중치 (curriculum learning용, Optional)
        difficulty_bins: 난이도 구간 정의 (Optional)
        seed: 랜덤 시드
        shuffle: 셔플 여부

    Returns:
        DataLoader

    Examples:
        >>> # Baseline: 정답만 100,000개
        >>> loader = create_dataloader(
        ...     "storage/datasets/codecontests/processed/train.jsonl",
        ...     tokenizer, batch_size=4, max_length=2048,
        ...     n_samples=100000, balance_correct=False, correct_ratio=1.0
        ... )
        >>>
        >>> # Verifiable: Curriculum learning
        >>> loader = create_dataloader(
        ...     "storage/datasets/codecontests/processed/train.jsonl",
        ...     tokenizer, batch_size=4, max_length=2048,
        ...     n_samples=100000, balance_correct=True, correct_ratio=0.5,
        ...     difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
        ...     difficulty_bins={"low": [1,3], "medium": [4,7], "high": [8,11]}
        ... )
    """
    # 데이터셋 이름 및 스플릿 추출
    dataset_path_obj = Path(dataset_path)
    dataset_name = dataset_path_obj.parent.parent.name
    split_file = dataset_path_obj.name

    if "train" in split_file:
        split = "train"
    elif "valid" in split_file or "validation" in split_file:
        split = "validation"
    else:
        split = "test"

    # 분산 환경 정보 자동 추출
    rank = get_rank()
    world_size = get_world_size()

    # Rank-aware 데이터셋 로드
    dataset = load_dataset(
        dataset_name=dataset_name,
        split=split,
        n_samples=n_samples,
        balance_correct=balance_correct,
        correct_ratio=correct_ratio,
        difficulty_weights=difficulty_weights,
        difficulty_bins=difficulty_bins,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    return dataloader
