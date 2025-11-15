"""JSONL → HuggingFace Dataset 로딩 (메타데이터 기반)

메타데이터 기반 효율적 로딩:
- 전체 데이터 로드 없이 메타데이터만으로 샘플 선택
- Stage 1: is_correct 균형 샘플링 (50:50)
- Stage 2: difficulty 기반 Curriculum Learning
- 메모리 사용량 99% 절감
"""

from pathlib import Path
from typing import Literal, Optional
import logging
import random
import json

from datasets import Dataset, DatasetDict
import numpy as np

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: Literal["codecontests", "mbpp", "humaneval",
                          "codecontests_small", "mbpp_small", "humaneval_small"],
    split: Optional[str] = None,
    stage: Optional[Literal["stage1", "stage2"]] = None,
    n_samples: Optional[int] = None,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
) -> Dataset | DatasetDict:
    """JSONL 파일을 메타데이터 기반으로 효율적 로딩

    메타데이터 파일을 먼저 읽어 필요한 샘플의 인덱스를 계산한 후,
    JSONL 파일에서 해당 라인만 선택적으로 읽습니다.

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval, *_small)
        split: 데이터 스플릿 (train, validation, test). None이면 모든 스플릿 로딩
        stage: 샘플링 전략 (stage1: is_correct 균형, stage2: difficulty curriculum)
        n_samples: 샘플링할 샘플 수
        balance_correct: is_correct 균형 샘플링 여부
        correct_ratio: correct 샘플 비율 (기본 0.5)
        difficulty_weights: 난이도별 가중치 (Stage 2 전용)
        difficulty_bins: 난이도 구간 정의 (Stage 2 전용)
        seed: 랜덤 시드

    Returns:
        Dataset 또는 DatasetDict (split이 None인 경우)

    Examples:
        >>> # Stage 1: is_correct 균형 샘플링
        >>> dataset = load_dataset(
        ...     "codecontests",
        ...     split="train",
        ...     stage="stage1",
        ...     n_samples=50000,
        ...     balance_correct=True,
        ...     seed=42
        ... )
        >>>
        >>> # Stage 2: difficulty curriculum
        >>> dataset = load_dataset(
        ...     "codecontests",
        ...     split="train",
        ...     stage="stage2",
        ...     n_samples=200000,
        ...     difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
        ...     difficulty_bins={"low": [1,3], "medium": [4,7], "high": [8,11]},
        ...     seed=42
        ... )
    """
    # JSONL 경로 해석
    data_files = _get_dataset_paths(dataset_name)

    logger.info(f"데이터셋 로딩 시작: {dataset_name}")

    if split is not None:
        # 단일 스플릿 로딩
        if split not in data_files:
            raise ValueError(
                f"스플릿 '{split}'이 존재하지 않습니다. "
                f"가능한 스플릿: {list(data_files.keys())}"
            )

        # 메타데이터 기반 로딩 (샘플링이 필요한 경우)
        if stage is not None and n_samples is not None:
            metadata = _load_metadata(dataset_name, split)

            if metadata is None:
                raise FileNotFoundError(
                    f"메타데이터 파일이 존재하지 않습니다: {dataset_name}/{split}\n"
                    f"먼저 'python scripts/extract_metadata.py --dataset {dataset_name.replace('_small', '')} "
                    f"--split {split}' 를 실행하세요."
                )

            logger.info(f"메타데이터 기반 샘플링 시작: {dataset_name}/{split}")

            # 1. 메타데이터로 인덱스 계산
            indices = _compute_sampling_indices_from_metadata(
                metadata=metadata,
                stage=stage,
                n_samples=n_samples,
                balance_correct=balance_correct,
                correct_ratio=correct_ratio,
                difficulty_weights=difficulty_weights,
                difficulty_bins=difficulty_bins,
                seed=seed,
            )

            # 2. 해당 인덱스의 라인만 JSONL에서 읽기
            jsonl_path = Path(data_files[split])
            samples = _read_jsonl_by_indices(jsonl_path, indices)

            # 3. HuggingFace Dataset으로 변환
            dataset = Dataset.from_list(samples)

            logger.info(
                f"메타데이터 기반 로딩 완료: {len(dataset)} 샘플 "
                f"(메모리 효율적)"
            )

            return dataset
        else:
            # 샘플링 없이 전체 로딩 (메타데이터 불필요)
            raise NotImplementedError(
                "샘플링 없는 전체 로딩은 지원하지 않습니다. "
                "stage와 n_samples를 지정하여 메타데이터 기반 로딩을 사용하세요."
            )
    else:
        # 모든 스플릿 로딩 (DatasetDict)
        raise NotImplementedError(
            "전체 스플릿 로딩(DatasetDict)은 지원하지 않습니다. "
            "split을 명시적으로 지정하세요."
        )


def _get_dataset_paths(dataset_name: str) -> dict[str, str]:
    """데이터셋 이름으로 JSONL 파일 경로 해석

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval, *_small)

    Returns:
        스플릿별 JSONL 파일 경로 딕셔너리
    """
    # 작은 데이터셋인지 확인
    if dataset_name.endswith("_small"):
        base_dir = Path("storage/datasets_local_small")
        dataset_dir = base_dir / dataset_name
    else:
        base_dir = Path("storage/datasets_v2")
        dataset_dir = base_dir / dataset_name / "processed"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리가 존재하지 않습니다: {dataset_dir}")

    # 표준 스플릿 이름 매핑
    if dataset_name.endswith("_small"):
        split_mappings = {
            "train": ["train_small.jsonl"],
            "validation": ["valid_small.jsonl", "validation_small.jsonl"],
            "test": ["test_small.jsonl"],
        }
    else:
        split_mappings = {
            "train": ["train.jsonl"],
            "validation": ["valid.jsonl", "validation.jsonl"],
            "test": ["test.jsonl"],
        }

    data_files = {}

    for split_name, candidates in split_mappings.items():
        for candidate in candidates:
            file_path = dataset_dir / candidate
            if file_path.exists():
                data_files[split_name] = str(file_path)
                break

    if not data_files:
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_dir}")

    return data_files


def _load_metadata(
    dataset_name: str,
    split: str,
) -> Optional[list[dict]]:
    """메타데이터 파일 로드

    Args:
        dataset_name: 데이터셋 이름
        split: 스플릿 이름

    Returns:
        메타데이터 리스트 또는 None (파일이 없는 경우)
    """
    # 데이터셋 디렉터리 결정
    if dataset_name.endswith("_small"):
        base_dir = Path("storage/datasets_local_small")
        dataset_dir = base_dir / dataset_name
        suffix = "_small"
    else:
        base_dir = Path("storage/datasets_v2")
        dataset_dir = base_dir / dataset_name / "processed"
        suffix = ""

    if not dataset_dir.exists():
        logger.error(f"데이터셋 디렉터리가 존재하지 않습니다: {dataset_dir}")
        return None

    # 메타데이터 파일 경로
    # validation → validation 또는 valid
    if split == "validation":
        candidates = [
            dataset_dir / f"validation{suffix}_metadata.json",
            dataset_dir / f"valid{suffix}_metadata.json",
        ]
    else:
        candidates = [dataset_dir / f"{split}{suffix}_metadata.json"]

    metadata_path = None
    for candidate in candidates:
        if candidate.exists():
            metadata_path = candidate
            break

    if metadata_path is None:
        logger.error(f"메타데이터 파일을 찾을 수 없습니다: {dataset_dir}/{split}")
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", [])
        logger.info(f"메타데이터 로드 완료: {len(metadata):,} 샘플 ({metadata_path.name})")

        return metadata
    except Exception as e:
        logger.error(f"메타데이터 로드 실패: {e}")
        return None


def _compute_sampling_indices_from_metadata(
    metadata: list[dict],
    stage: Literal["stage1", "stage2"],
    n_samples: int,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
) -> list[int]:
    """메타데이터 기반으로 샘플링 인덱스 계산

    전체 데이터를 로드하지 않고 메타데이터만으로 샘플 인덱스를 계산합니다.

    Args:
        metadata: 메타데이터 리스트
        stage: 샘플링 전략
        n_samples: 샘플링할 샘플 수
        balance_correct: is_correct 균형 샘플링 여부
        correct_ratio: correct 샘플 비율
        difficulty_weights: 난이도별 가중치
        difficulty_bins: 난이도 구간 정의
        seed: 랜덤 시드

    Returns:
        선택된 인덱스 리스트
    """
    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(metadata)

    if stage == "stage1":
        # Stage 1: is_correct 균형 샘플링
        if not balance_correct:
            # 균형 샘플링 없이 랜덤 샘플링
            indices = random.sample(range(total_samples), min(n_samples, total_samples))
            return indices

        # is_correct 필드 존재 확인
        if not metadata[0].get("is_correct") is not None:
            logger.warning("is_correct 필드가 없습니다. 랜덤 샘플링으로 전환합니다.")
            indices = random.sample(range(total_samples), min(n_samples, total_samples))
            return indices

        # correct/incorrect 인덱스 분리
        correct_indices = []
        incorrect_indices = []

        for idx, meta in enumerate(metadata):
            if meta.get("is_correct"):
                correct_indices.append(idx)
            else:
                incorrect_indices.append(idx)

        logger.info(
            f"메타데이터 분포: correct={len(correct_indices):,}, "
            f"incorrect={len(incorrect_indices):,}"
        )

        # 샘플 수 계산
        n_correct = int(n_samples * correct_ratio)
        n_incorrect = n_samples - n_correct

        # 각 그룹에서 샘플링
        selected_correct = random.sample(
            correct_indices, min(n_correct, len(correct_indices))
        )
        selected_incorrect = random.sample(
            incorrect_indices, min(n_incorrect, len(incorrect_indices))
        )

        # 병합 및 섞기
        selected_indices = selected_correct + selected_incorrect
        random.shuffle(selected_indices)

        actual_ratio = len(selected_correct) / len(selected_indices)
        logger.info(
            f"Stage 1 인덱스 계산 완료: {len(selected_indices)} 인덱스 "
            f"(correct: {len(selected_correct)}, incorrect: {len(selected_incorrect)}, "
            f"비율: {actual_ratio:.2%})"
        )

        return selected_indices

    elif stage == "stage2":
        # Stage 2: difficulty 기반 Curriculum Learning
        if difficulty_weights is None or difficulty_bins is None:
            logger.warning("difficulty_weights 또는 difficulty_bins가 없습니다. 랜덤 샘플링으로 전환합니다.")
            indices = random.sample(range(total_samples), min(n_samples, total_samples))
            return indices

        # difficulty 필드 존재 확인
        if "difficulty" not in metadata[0]:
            logger.warning("difficulty 필드가 없습니다. 랜덤 샘플링으로 전환합니다.")
            indices = random.sample(range(total_samples), min(n_samples, total_samples))
            return indices

        # 난이도 구간별 인덱스 분리
        if balance_correct:
            # is_correct 필드 존재 확인
            if "is_correct" not in metadata[0]:
                logger.warning("is_correct 필드가 없어 균형 샘플링을 비활성화합니다.")
                balance_correct = False

        if balance_correct:
            # 각 bin별로 correct/incorrect 인덱스 분리
            bin_indices = {
                bin_name: {"correct": [], "incorrect": []}
                for bin_name in difficulty_bins.keys()
            }

            for idx, meta in enumerate(metadata):
                difficulty = meta.get("difficulty")
                is_correct = meta.get("is_correct")

                if difficulty is None:
                    continue

                # 해당 난이도가 어느 bin에 속하는지 판단
                for bin_name, (min_diff, max_diff) in difficulty_bins.items():
                    if min_diff <= difficulty <= max_diff:
                        if is_correct:
                            bin_indices[bin_name]["correct"].append(idx)
                        else:
                            bin_indices[bin_name]["incorrect"].append(idx)
                        break

            # 각 bin별 샘플 수 로그
            for bin_name, indices_dict in bin_indices.items():
                n_correct = len(indices_dict["correct"])
                n_incorrect = len(indices_dict["incorrect"])
                logger.info(
                    f"{bin_name} bin: correct={n_correct:,}, incorrect={n_incorrect:,}"
                )
        else:
            # 균형 샘플링 없이 난이도만 분리
            bin_indices = {bin_name: [] for bin_name in difficulty_bins.keys()}

            for idx, meta in enumerate(metadata):
                difficulty = meta.get("difficulty")

                if difficulty is None:
                    continue

                # 해당 난이도가 어느 bin에 속하는지 판단
                for bin_name, (min_diff, max_diff) in difficulty_bins.items():
                    if min_diff <= difficulty <= max_diff:
                        bin_indices[bin_name].append(idx)
                        break

            # 각 bin별 샘플 수 로그
            bin_stats = ", ".join(
                [f"{name}: {len(indices):,}" for name, indices in bin_indices.items()]
            )
            logger.info(f"난이도 분포: {bin_stats}")

        # 가중치에 따라 각 bin에서 샘플링
        selected_indices = []

        for bin_name, weight in difficulty_weights.items():
            if weight <= 0:
                continue

            bin_n_samples = int(n_samples * weight)

            if balance_correct:
                # 균형 샘플링
                indices_dict = bin_indices.get(bin_name, {"correct": [], "incorrect": []})
                correct_indices = indices_dict["correct"]
                incorrect_indices = indices_dict["incorrect"]

                if len(correct_indices) == 0 and len(incorrect_indices) == 0:
                    logger.warning(f"{bin_name} bin에 샘플이 없습니다.")
                    continue

                # 목표 샘플 수 계산
                n_correct_target = int(bin_n_samples * correct_ratio)
                n_incorrect_target = bin_n_samples - n_correct_target

                # 각 그룹에서 샘플링
                n_correct_actual = min(n_correct_target, len(correct_indices))
                n_incorrect_actual = min(n_incorrect_target, len(incorrect_indices))

                # 샘플 부족 시 보충
                if n_correct_actual + n_incorrect_actual < bin_n_samples:
                    shortage = bin_n_samples - (n_correct_actual + n_incorrect_actual)
                    if n_correct_actual < n_correct_target and len(incorrect_indices) > n_incorrect_actual:
                        n_incorrect_actual = min(
                            n_incorrect_actual + shortage, len(incorrect_indices)
                        )
                    elif n_incorrect_actual < n_incorrect_target and len(correct_indices) > n_correct_actual:
                        n_correct_actual = min(
                            n_correct_actual + shortage, len(correct_indices)
                        )

                # 샘플링 수행
                sampled_correct = (
                    random.sample(correct_indices, n_correct_actual)
                    if n_correct_actual > 0
                    else []
                )
                sampled_incorrect = (
                    random.sample(incorrect_indices, n_incorrect_actual)
                    if n_incorrect_actual > 0
                    else []
                )

                selected_indices.extend(sampled_correct)
                selected_indices.extend(sampled_incorrect)

                actual_ratio = (
                    n_correct_actual / (n_correct_actual + n_incorrect_actual)
                    if (n_correct_actual + n_incorrect_actual) > 0
                    else 0
                )
                logger.info(
                    f"{bin_name} bin: correct={n_correct_actual}, incorrect={n_incorrect_actual}, "
                    f"total={n_correct_actual + n_incorrect_actual} (목표: {bin_n_samples}, 비율: {actual_ratio:.2%})"
                )
            else:
                # 균형 샘플링 없이 랜덤 샘플링
                available_indices = bin_indices.get(bin_name, [])

                if len(available_indices) == 0:
                    logger.warning(f"{bin_name} bin에 샘플이 없습니다.")
                    continue

                # 샘플링
                sampled = random.sample(
                    available_indices, min(bin_n_samples, len(available_indices))
                )
                selected_indices.extend(sampled)

                logger.info(f"{bin_name} bin에서 {len(sampled)} 인덱스 선택 (목표: {bin_n_samples})")

        # 샘플 수가 부족하면 랜덤으로 추가
        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            all_indices = list(range(total_samples))
            available = [idx for idx in all_indices if idx not in selected_indices]

            if available:
                additional = random.sample(available, min(remaining, len(available)))
                selected_indices.extend(additional)
                logger.warning(f"샘플 부족으로 {len(additional)} 인덱스 추가 (랜덤)")

        # 섞기
        random.shuffle(selected_indices)

        # 정확히 n_samples 개수 맞추기
        selected_indices = selected_indices[:n_samples]

        logger.info(f"Stage 2 인덱스 계산 완료: {len(selected_indices)} 인덱스")

        return selected_indices

    else:
        raise ValueError(f"지원하지 않는 stage: {stage}")


def _read_jsonl_by_indices(
    jsonl_path: Path,
    indices: list[int],
) -> list[dict]:
    """JSONL 파일에서 특정 인덱스의 라인만 읽기

    전체 파일을 메모리에 로드하지 않고 필요한 라인만 읽습니다.

    Args:
        jsonl_path: JSONL 파일 경로
        indices: 읽을 라인 인덱스 리스트

    Returns:
        선택된 샘플 리스트
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 파일이 존재하지 않습니다: {jsonl_path}")

    # 인덱스를 정렬하여 순차적으로 읽기 (성능 최적화)
    sorted_indices = sorted(enumerate(indices), key=lambda x: x[1])
    target_indices = {idx: original_pos for original_pos, idx in sorted_indices}

    samples = [None] * len(indices)
    current_line = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx in target_indices:
                try:
                    sample = json.loads(line.strip())
                    original_pos = target_indices[line_idx]
                    samples[original_pos] = sample
                except json.JSONDecodeError as e:
                    logger.warning(f"라인 {line_idx} 파싱 오류: {e}")
                    continue

                current_line += 1

                # 모든 목표 샘플을 찾았으면 종료
                if current_line >= len(indices):
                    break

    # None 제거 (파싱 실패한 샘플)
    samples = [s for s in samples if s is not None]

    logger.info(f"JSONL 읽기 완료: {len(samples)} 샘플 로드")

    return samples
