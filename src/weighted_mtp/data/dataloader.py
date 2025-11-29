"""DataLoader 생성 (분산 학습 지원)

4개 파이프라인의 중복 create_dataloader() 로직을 통합.
메타데이터 기반 Rank-aware 샘플링으로 각 GPU가 필요한 데이터만 로드.
"""

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from weighted_mtp.data.datasets import load_dataset
from weighted_mtp.data.collators import AlpacaDataCollator, PairwiseDataCollator
from weighted_mtp.runtime.distributed import get_rank, get_world_size


def get_curriculum_weights(
    current_epoch: float,
    curriculum_schedule: list[dict],
) -> dict[str, float]:
    """Curriculum schedule에서 현재 epoch에 맞는 difficulty_weights 추출

    Args:
        current_epoch: 현재 epoch (0.0 ~ n_epochs)
        curriculum_schedule: Config의 curriculum_schedule

    Returns:
        difficulty_weights (예: {"low": 0.7, "medium": 0.3, "high": 0.0})
    """
    for schedule in curriculum_schedule:
        epoch_range = schedule["epoch_range"]
        if epoch_range[0] <= current_epoch < epoch_range[1]:
            return dict(schedule["difficulty_weights"])

    # 마지막 schedule 반환 (현재 epoch이 범위 밖인 경우)
    return dict(curriculum_schedule[-1]["difficulty_weights"])


def get_difficulty_config(
    config: DictConfig,
    current_epoch: float = 0.0,
) -> tuple[Optional[dict], Optional[dict]]:
    """Config에서 difficulty 설정 추출

    우선순위:
    1. curriculum_schedule 있고 curriculum_learning=true → epoch별 동적 weights
    2. difficulty_weights 있음 → 정적 weights
    3. 둘 다 없음 → None (기존 동작 유지)

    Args:
        config: 전체 config (OmegaConf DictConfig)
        current_epoch: 현재 epoch (curriculum용)

    Returns:
        (difficulty_weights, difficulty_bins) - 둘 다 None일 수 있음

    Examples:
        >>> # 정적 weights 사용
        >>> weights, bins = get_difficulty_config(config)
        >>>
        >>> # curriculum learning (epoch별 동적 weights)
        >>> weights, bins = get_difficulty_config(config, current_epoch=1.5)
    """
    data_sampling = config.data_sampling

    # difficulty_bins 추출
    difficulty_bins = data_sampling.get("difficulty_bins", None)
    if difficulty_bins:
        difficulty_bins = dict(difficulty_bins)

    # 1순위: curriculum_schedule (동적)
    use_curriculum = data_sampling.get("curriculum_learning", False)
    curriculum_schedule = data_sampling.get("curriculum_schedule", None)

    if use_curriculum and curriculum_schedule:
        difficulty_weights = get_curriculum_weights(current_epoch, list(curriculum_schedule))
    # 2순위: difficulty_weights (정적)
    elif data_sampling.get("difficulty_weights", None):
        difficulty_weights = dict(data_sampling.difficulty_weights)
    else:
        difficulty_weights = None

    return difficulty_weights, difficulty_bins


def create_dataloader(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    sampling_config: dict,
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
        sampling_config: 샘플링 설정 딕셔너리
            - use_pairwise: Pairwise 포맷 사용 여부 (기본: False)
            - n_samples: 샘플링할 총 샘플 수
            - difficulty_weights: 난이도별 가중치 (필수)
            - difficulty_bins: 난이도 구간 정의 (필수)
            - max_pairs_per_problem: problem당 최대 샘플 수
        seed: 랜덤 시드
        shuffle: 셔플 여부

    Returns:
        DataLoader

    Examples:
        >>> # Correct-only 방식 (Baseline, Rho-1, Verifiable)
        >>> sampling_config = {
        ...     "use_pairwise": False,
        ...     "n_samples": 100000,
        ...     "difficulty_bins": {"all": [0, 25]},
        ...     "difficulty_weights": {"all": 1.0}
        ... }
        >>> loader = create_dataloader(path, tokenizer, 4, 2048, sampling_config)
        >>>
        >>> # Pairwise 방식 (Critic)
        >>> sampling_config = {
        ...     "use_pairwise": True,
        ...     "n_samples": 100000,
        ...     "difficulty_bins": {"all": [0, 25]},
        ...     "difficulty_weights": {"all": 1.0}
        ... }
        >>> loader = create_dataloader(path, tokenizer, 16, 2048, sampling_config)
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
        sampling_config=sampling_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Collator 선택 (use_pairwise에 따라 분기)
    use_pairwise = sampling_config.get("use_pairwise", False)
    if use_pairwise:
        collator = PairwiseDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )
    else:
        collator = AlpacaDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # DataLoader 생성
    # 분산 환경에서 drop_last=True로 모든 rank의 배치 수 동일하게 유지 (FSDP 데드락 방지)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        drop_last=(world_size > 1),
    )

    return dataloader
