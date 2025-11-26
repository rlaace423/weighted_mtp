"""JSONL → HuggingFace Dataset 로딩 (메타데이터 기반)

메타데이터 기반 효율적 로딩:
- 전체 데이터 로드 없이 메타데이터만으로 샘플 선택
- Difficulty 기반 샘플링 (bins, weights, correct_ratio 등)
- use_pairwise 옵션으로 Pairwise 포맷 변환 지원
- 메모리 사용량 99% 절감
"""

from pathlib import Path
from typing import Optional
from collections import defaultdict
import logging
import random
import json

from datasets import Dataset
import numpy as np

logger = logging.getLogger(__name__)

# 메타데이터 전역 캐시 (DataLoader 재생성 시 중복 로딩 방지)
_metadata_cache: dict[str, list[dict]] = {}


def load_dataset(
    dataset_name: str,
    split: str,
    sampling_config: dict,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> Dataset:
    """JSONL 파일을 메타데이터 기반으로 효율적 로딩 (Rank-aware 분산)

    메타데이터 파일을 먼저 읽어 필요한 샘플의 인덱스를 계산한 후,
    JSONL 파일에서 해당 라인만 선택적으로 읽습니다.

    분산 환경에서 각 GPU가 자기 담당 샘플만 로드합니다.
    재현성을 위해 모든 rank가 동일한 시드로 전체 인덱스를 계산한 후,
    rank::world_size 패턴으로 서브셋을 선택합니다.

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval)
        split: 데이터 스플릿 (train, validation, test)
        sampling_config: 샘플링 설정 딕셔너리
            - use_pairwise: Pairwise 포맷 변환 여부 (기본: False)
            - n_samples: 샘플링할 총 샘플 수
            - auto_data_balancing: correct/incorrect 균형 샘플링
            - correct_ratio: correct 샘플 비율
            - difficulty_weights: 난이도별 가중치
            - difficulty_bins: 난이도 구간 정의
        seed: 랜덤 시드
        rank: 현재 프로세스의 global rank (기본: 0)
        world_size: 전체 프로세스 수 (기본: 1)

    Returns:
        Dataset (분산 환경에서는 1/world_size 크기)
    """
    # 메타데이터 로드
    metadata = _load_metadata(dataset_name, split)

    if metadata is None:
        raise FileNotFoundError(
            f"메타데이터 파일이 존재하지 않습니다: {dataset_name}/{split}\n"
            f"먼저 'python scripts/extract_metadata.py --dataset {dataset_name.replace('_small', '')} "
            f"--split {split}' 를 실행하세요."
        )

    logger.info(f"메타데이터 기반 샘플링 시작: {dataset_name}/{split}")

    # use_pairwise 옵션 확인
    use_pairwise = sampling_config.get("use_pairwise", False)

    # 1. Difficulty 기반 인덱스 계산 (모든 rank 동일, 재현성 보장)
    all_indices = _compute_sampling_indices(
        metadata=metadata,
        sampling_config=sampling_config,
        seed=seed,
    )

    # 2. JSONL 경로 확인
    data_files = _get_dataset_paths(dataset_name)
    if split not in data_files:
        raise ValueError(
            f"스플릿 '{split}'이 존재하지 않습니다. "
            f"가능한 스플릿: {list(data_files.keys())}"
        )
    jsonl_path = Path(data_files[split])

    # 3. use_pairwise에 따라 로딩 방식 결정
    if use_pairwise:
        # Pairwise 모드: 선택된 인덱스 내에서 (correct, incorrect) 쌍 생성
        all_pairs = _create_pairs_from_indices(
            indices=all_indices,
            metadata=metadata,
            seed=seed,
        )

        # Rank 담당 쌍 필터링
        if world_size > 1:
            rank_pairs = all_pairs[rank::world_size]
            logger.info(
                f"[Rank {rank}/{world_size}] 전체 {len(all_pairs):,} 쌍 중 "
                f"{len(rank_pairs):,} 쌍 로드 (분산 학습)"
            )
        else:
            rank_pairs = all_pairs
            logger.info(f"Pairwise 샘플링 완료: {len(rank_pairs):,} 쌍 (로컬 환경)")

        # JSONL에서 쌍 데이터 읽기
        samples = _read_jsonl_pairwise(jsonl_path, rank_pairs)

    else:
        # Pointwise 모드: 기존 방식
        if world_size > 1:
            rank_indices = all_indices[rank::world_size]
            logger.info(
                f"[Rank {rank}/{world_size}] 전체 {len(all_indices):,} 샘플 중 "
                f"{len(rank_indices):,} 샘플 로드 (분산 학습)"
            )
        else:
            rank_indices = all_indices
            logger.info(f"메타데이터 기반 샘플링 완료: {len(rank_indices):,} 인덱스 (로컬 환경)")

        # JSONL에서 인덱스 기반 데이터 읽기
        samples = _read_jsonl_by_indices(jsonl_path, rank_indices)

    # 4. HuggingFace Dataset으로 변환
    dataset = Dataset.from_list(samples)

    mode_str = "Pairwise" if use_pairwise else "Pointwise"
    logger.info(f"데이터셋 로드 완료 ({mode_str}): {len(dataset):,} 샘플")

    return dataset


def _get_dataset_paths(dataset_name: str) -> dict[str, str]:
    """데이터셋 이름으로 JSONL 파일 경로 해석

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval)

    Returns:
        스플릿별 JSONL 파일 경로 딕셔너리
    """
    base_dir = Path("storage/datasets")
    dataset_dir = base_dir / dataset_name / "processed"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리가 존재하지 않습니다: {dataset_dir}")

    # 표준 스플릿 이름 매핑
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
    """메타데이터 파일 로드 (캐싱 지원)

    동일한 데이터셋/스플릿의 메타데이터는 캐시에서 재사용하여
    DataLoader 재생성 시 중복 로딩을 방지합니다.

    Args:
        dataset_name: 데이터셋 이름
        split: 스플릿 이름

    Returns:
        메타데이터 리스트 또는 None (파일이 없는 경우)
    """
    # 캐시 키 생성
    cache_key = f"{dataset_name}/{split}"

    # 캐시에서 조회
    if cache_key in _metadata_cache:
        cached_metadata = _metadata_cache[cache_key]
        logger.info(f"메타데이터 캐시 히트: {len(cached_metadata):,} 샘플 ({cache_key})")
        return cached_metadata

    base_dir = Path("storage/datasets")
    dataset_dir = base_dir / dataset_name / "processed"

    if not dataset_dir.exists():
        logger.error(f"데이터셋 디렉터리가 존재하지 않습니다: {dataset_dir}")
        return None

    # 메타데이터 파일 경로 (validation → validation 또는 valid)
    if split == "validation":
        candidates = [
            dataset_dir / "validation_metadata.json",
            dataset_dir / "valid_metadata.json",
        ]
    else:
        candidates = [dataset_dir / f"{split}_metadata.json"]

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

        # 캐시에 저장
        _metadata_cache[cache_key] = metadata
        logger.info(f"메타데이터 캐시 저장: {cache_key}")

        return metadata
    except Exception as e:
        logger.error(f"메타데이터 로드 실패: {e}")
        return None


def _compute_sampling_indices(
    metadata: list[dict],
    sampling_config: dict,
    seed: int,
) -> list[int]:
    """메타데이터 기반으로 샘플링 인덱스 계산

    전체 데이터를 로드하지 않고 메타데이터만으로 샘플 인덱스를 계산합니다.

    Args:
        metadata: 메타데이터 리스트
        sampling_config: 샘플링 설정 딕셔너리
        seed: 랜덤 시드

    Returns:
        선택된 인덱스 리스트
    """
    random.seed(seed)
    np.random.seed(seed)

    # 샘플링 설정 추출
    n_samples = sampling_config.get("n_samples", len(metadata))
    auto_data_balancing = sampling_config.get("auto_data_balancing", False)
    correct_ratio = sampling_config.get("correct_ratio", 0.5)
    difficulty_weights = sampling_config.get("difficulty_weights")
    difficulty_bins = sampling_config.get("difficulty_bins")

    # Difficulty-based sampling
    if difficulty_weights is not None and difficulty_bins is not None:
        logger.info("샘플링 전략: Difficulty-based curriculum learning")
        return _sample_by_difficulty(
            metadata, n_samples, difficulty_weights, difficulty_bins,
            auto_data_balancing, correct_ratio, seed
        )

    # Balanced correct/incorrect sampling
    if auto_data_balancing:
        logger.info(f"샘플링 전략: Balanced sampling (correct_ratio={correct_ratio})")
        return _sample_balanced(
            metadata, n_samples, correct_ratio, seed
        )

    # Correct-only sampling
    if correct_ratio == 1.0:
        logger.info("샘플링 전략: Correct-only sampling")
        return _sample_correct_only(
            metadata, n_samples, seed
        )

    # Random sampling (fallback)
    logger.info("샘플링 전략: Random sampling")
    total_samples = len(metadata)
    indices = random.sample(range(total_samples), min(n_samples, total_samples))
    logger.info(f"Random 샘플링 완료: {len(indices)} 인덱스")
    return indices


def _sample_by_difficulty(
    metadata: list[dict],
    n_samples: int,
    difficulty_weights: dict,
    difficulty_bins: dict,
    auto_data_balancing: bool,
    correct_ratio: float,
    seed: int,
) -> list[int]:
    """Difficulty-based curriculum learning 샘플링

    난이도 구간별 가중치에 따라 샘플링하며, 선택적으로 correct/incorrect 균형도 유지합니다.

    Args:
        metadata: 메타데이터 리스트
        n_samples: 샘플링할 샘플 수
        difficulty_weights: 난이도별 가중치 (예: {"low": 0.7, "medium": 0.3, "high": 0.0})
        difficulty_bins: 난이도 구간 (예: {"low": [1,3], "medium": [4,7], "high": [8,11]})
        auto_data_balancing: is_correct 균형 샘플링 여부
        correct_ratio: correct 샘플 비율
        seed: 랜덤 시드

    Returns:
        선택된 인덱스 리스트
    """
    random.seed(seed)

    # difficulty 필드 존재 확인
    if "difficulty" not in metadata[0]:
        logger.warning("difficulty 필드가 없습니다. 랜덤 샘플링으로 전환합니다.")
        return random.sample(range(len(metadata)), min(n_samples, len(metadata)))

    # is_correct 필드 확인 (auto_data_balancing=True인 경우)
    if auto_data_balancing and "is_correct" not in metadata[0]:
        logger.warning("is_correct 필드가 없어 균형 샘플링을 비활성화합니다.")
        auto_data_balancing = False

    # === 샘플링 계획 로깅 ===
    bins_str = ", ".join([f"{name}[{r[0]}-{r[1]}]" for name, r in difficulty_bins.items()])
    weights_str = ", ".join([f"{name}={w:.2f}" for name, w in difficulty_weights.items() if w > 0])
    logger.info(f"=== 샘플링 계획 ===")
    logger.info(f"목표: {n_samples:,}개, correct_ratio: {correct_ratio}")
    logger.info(f"difficulty_bins: {bins_str}")
    logger.info(f"weights: {weights_str}")

    # 난이도 구간별 인덱스 분리 (correct/incorrect 구분)
    bin_indices = {
        bin_name: {"correct": [], "incorrect": []}
        for bin_name in difficulty_bins.keys()
    }

    for idx, meta in enumerate(metadata):
        difficulty = meta.get("difficulty")
        is_correct = meta.get("is_correct")

        if difficulty is None:
            continue

        for bin_name, (min_diff, max_diff) in difficulty_bins.items():
            if min_diff <= difficulty <= max_diff:
                if is_correct:
                    bin_indices[bin_name]["correct"].append(idx)
                else:
                    bin_indices[bin_name]["incorrect"].append(idx)
                break

    # === 가용 데이터 로깅 ===
    logger.info(f"=== 가용 데이터 ===")
    for bin_name in difficulty_bins.keys():
        indices_dict = bin_indices.get(bin_name, {"correct": [], "incorrect": []})
        n_correct = len(indices_dict["correct"])
        n_incorrect = len(indices_dict["incorrect"])
        logger.info(f"{bin_name}: C={n_correct:,}, I={n_incorrect:,}")

    # 가중치에 따라 각 bin에서 샘플링
    selected_indices = []
    sampling_results = {}  # 결과 수집용

    for bin_name, weight in difficulty_weights.items():
        if weight <= 0:
            continue

        bin_n_samples = int(n_samples * weight)

        if auto_data_balancing:
            # 균형 샘플링
            indices_dict = bin_indices.get(bin_name, {"correct": [], "incorrect": []})
            correct_indices = indices_dict["correct"]
            incorrect_indices = indices_dict["incorrect"]

            if len(correct_indices) == 0 and len(incorrect_indices) == 0:
                sampling_results[bin_name] = {"target": bin_n_samples, "correct": 0, "incorrect": 0}
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

            sampling_results[bin_name] = {
                "target": bin_n_samples,
                "correct": n_correct_actual,
                "incorrect": n_incorrect_actual
            }
        else:
            # auto_data_balancing = false: correct_ratio에 따라 필터링, 보충 없음
            indices_dict = bin_indices.get(bin_name, {"correct": [], "incorrect": []})
            correct_indices = indices_dict["correct"]
            incorrect_indices = indices_dict["incorrect"]

            # correct_ratio에 따라 사용할 인덱스 결정
            if correct_ratio == 1.0:
                available_indices = correct_indices
            elif correct_ratio == 0.0:
                available_indices = incorrect_indices
            else:
                # 0 < ratio < 1: 비율에 맞게 샘플링 (보충 없음)
                n_correct_target = int(bin_n_samples * correct_ratio)
                n_incorrect_target = bin_n_samples - n_correct_target

                n_correct_actual = min(n_correct_target, len(correct_indices))
                n_incorrect_actual = min(n_incorrect_target, len(incorrect_indices))

                sampled_correct = random.sample(correct_indices, n_correct_actual) if n_correct_actual > 0 else []
                sampled_incorrect = random.sample(incorrect_indices, n_incorrect_actual) if n_incorrect_actual > 0 else []

                selected_indices.extend(sampled_correct)
                selected_indices.extend(sampled_incorrect)

                sampling_results[bin_name] = {
                    "target": bin_n_samples,
                    "correct": n_correct_actual,
                    "incorrect": n_incorrect_actual
                }
                continue

            if len(available_indices) == 0:
                sampling_results[bin_name] = {"target": bin_n_samples, "actual": 0}
                continue

            # 샘플링
            sampled = random.sample(
                available_indices, min(bin_n_samples, len(available_indices))
            )
            selected_indices.extend(sampled)

            sampling_results[bin_name] = {"target": bin_n_samples, "actual": len(sampled)}

    # Bin 간 보충: auto_data_balancing = true일 때만 수행
    if auto_data_balancing and len(selected_indices) < n_samples:
        shortage = n_samples - len(selected_indices)
        selected_set = set(selected_indices)

        # 가중치 높은 순으로 bin 정렬하여 보충 시도
        sorted_bins = sorted(
            [(name, weight) for name, weight in difficulty_weights.items() if weight > 0],
            key=lambda x: x[1],
            reverse=True
        )

        supplemented = 0
        for bin_name, _ in sorted_bins:
            if supplemented >= shortage:
                break

            indices_dict = bin_indices.get(bin_name, {"correct": [], "incorrect": []})
            # 이미 선택되지 않은 인덱스에서 보충
            available_c = [i for i in indices_dict["correct"] if i not in selected_set]
            available_i = [i for i in indices_dict["incorrect"] if i not in selected_set]
            available = available_c + available_i

            if available:
                to_add = min(shortage - supplemented, len(available))
                additional = random.sample(available, to_add)
                selected_indices.extend(additional)
                selected_set.update(additional)
                supplemented += to_add

                if to_add > 0:
                    # 결과에 보충 정보 추가
                    if bin_name in sampling_results:
                        if "supplemented" not in sampling_results[bin_name]:
                            sampling_results[bin_name]["supplemented"] = 0
                        sampling_results[bin_name]["supplemented"] += to_add

    # 섞기
    random.shuffle(selected_indices)

    # === 샘플링 결과 로깅 ===
    logger.info(f"=== 샘플링 결과 ===")

    total_correct = 0
    total_incorrect = 0
    total_actual = 0

    for bin_name, result in sampling_results.items():
        target = result["target"]
        supplemented = result.get("supplemented", 0)

        if "correct" in result:
            correct = result["correct"]
            incorrect = result["incorrect"]
            actual = correct + incorrect + supplemented
            total_correct += correct
            total_incorrect += incorrect
            ratio_str = f"C:{correct:,}, I:{incorrect:,}"
        else:
            actual = result["actual"] + supplemented
            ratio_str = f"{actual:,}"

        if supplemented > 0:
            ratio_str += f", 보충:{supplemented:,}"

        total_actual += actual
        pct = (actual / target * 100) if target > 0 else 0
        logger.info(f"{bin_name}: 목표={target:,}, 실제={actual:,} ({pct:.0f}%) [{ratio_str}]")

    # 최종 summary
    final_total = len(selected_indices)
    if auto_data_balancing:
        final_ratio = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
        logger.info(f"--- 합계: {final_total:,}개, Correct: {total_correct:,} ({final_ratio:.1%})")
    else:
        logger.info(f"--- 합계: {final_total:,}개")

    # 부족 시 에러 발생
    if final_total < n_samples:
        shortage = n_samples - final_total
        raise ValueError(
            f"데이터 부족: {shortage:,}개 부족. "
            f"요청: {n_samples:,}, 가용: {final_total:,}. "
            f"n_samples를 {final_total:,} 이하로 설정하세요."
        )

    return selected_indices


def _sample_balanced(
    metadata: list[dict],
    n_samples: int,
    correct_ratio: float,
    seed: int,
) -> list[int]:
    """Balanced correct/incorrect 샘플링

    is_correct 필드에 따라 correct/incorrect 샘플을 지정 비율로 샘플링합니다.

    Args:
        metadata: 메타데이터 리스트
        n_samples: 샘플링할 샘플 수
        correct_ratio: correct 샘플 비율 (예: 0.5 → 50:50)
        seed: 랜덤 시드

    Returns:
        선택된 인덱스 리스트
    """
    random.seed(seed)

    # is_correct 필드 존재 확인
    if "is_correct" not in metadata[0]:
        logger.warning("is_correct 필드가 없습니다. 랜덤 샘플링으로 전환합니다.")
        return random.sample(range(len(metadata)), min(n_samples, len(metadata)))

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

    actual_ratio = len(selected_correct) / len(selected_indices) if selected_indices else 0
    logger.info(
        f"Balanced 샘플링 완료: {len(selected_indices)} 인덱스 "
        f"(correct: {len(selected_correct)}, incorrect: {len(selected_incorrect)}, "
        f"비율: {actual_ratio:.2%})"
    )

    return selected_indices


def _sample_correct_only(
    metadata: list[dict],
    n_samples: int,
    seed: int,
) -> list[int]:
    """Correct-only 샘플링

    is_correct=True인 샘플만 필터링하여 샘플링합니다.

    Args:
        metadata: 메타데이터 리스트
        n_samples: 샘플링할 샘플 수
        seed: 랜덤 시드

    Returns:
        선택된 인덱스 리스트
    """
    random.seed(seed)

    # is_correct 필드 확인
    if "is_correct" not in metadata[0]:
        logger.warning("is_correct 필드가 없습니다. 전체 샘플에서 랜덤 샘플링합니다.")
        return random.sample(range(len(metadata)), min(n_samples, len(metadata)))

    # 정답 샘플만 필터링
    correct_indices = [i for i, m in enumerate(metadata) if m.get("is_correct", False)]

    logger.info(f"메타데이터 분포: correct={len(correct_indices):,}, total={len(metadata):,}")

    # 정답 샘플에서 랜덤 샘플링
    selected_indices = random.sample(correct_indices, min(n_samples, len(correct_indices)))
    random.shuffle(selected_indices)

    logger.info(f"Correct-only 샘플링 완료: {len(selected_indices)} 인덱스 (정답만)")

    return selected_indices


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


def load_evaluation_dataset(
    dataset_name: str,
    split: str = "test",
) -> Dataset:
    """평가용 데이터셋 로드 (전체 샘플, 샘플링 없음)

    벤치마크 평가를 위해 데이터셋 전체를 로드합니다.
    학습용 load_dataset()과 달리 샘플링/필터링 없이 전체 데이터를 반환합니다.

    Args:
        dataset_name: 데이터셋 이름 (humaneval, mbpp, codecontests)
        split: 데이터 스플릿 (test, validation)

    Returns:
        Dataset (HuggingFace Dataset 형식)

    Examples:
        >>> dataset = load_evaluation_dataset("humaneval", split="test")
        >>> print(len(dataset))
        164
        >>> print(dataset[0].keys())
        dict_keys(['instruction', 'input', 'output', 'task_id', 'metadata'])
    """
    # 데이터셋 경로 구성
    dataset_dir = Path("storage/datasets") / dataset_name / "processed"
    jsonl_path = dataset_dir / f"{split}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"평가 데이터셋 파일이 존재하지 않습니다: {jsonl_path}\n"
            f"먼저 데이터셋 준비를 완료하세요."
        )

    # JSONL 파일 전체 읽기
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"라인 {line_idx} 파싱 오류: {e}")
                continue

    logger.info(f"평가 데이터셋 로드 완료: {dataset_name}/{split} ({len(samples)} 샘플)")

    # HuggingFace Dataset으로 변환
    return Dataset.from_list(samples)


# ============================================================================
# Pairwise 포맷 변환 함수들
# ============================================================================


def _create_pairs_from_indices(
    indices: list[int],
    metadata: list[dict],
    seed: int,
) -> list[dict]:
    """Difficulty 샘플링된 인덱스를 (correct, incorrect) 쌍으로 변환

    동일 problem_id 내에서만 쌍을 생성합니다.

    Args:
        indices: difficulty 샘플링으로 선택된 인덱스
        metadata: 전체 메타데이터
        seed: 랜덤 시드

    Returns:
        [{"correct_idx": int, "incorrect_idx": int}, ...] 리스트

    Raises:
        ValueError: 쌍 생성이 불가능한 경우
    """
    random.seed(seed)

    # 필수 필드 확인
    if len(metadata) == 0:
        raise ValueError("메타데이터가 비어있습니다.")

    if "problem_id" not in metadata[0]:
        raise ValueError(
            "problem_id 필드가 메타데이터에 없습니다. "
            "Pairwise 모드에는 problem_id가 필수입니다."
        )

    if "is_correct" not in metadata[0]:
        raise ValueError(
            "is_correct 필드가 메타데이터에 없습니다. "
            "Pairwise 모드에는 is_correct가 필수입니다."
        )

    # 선택된 인덱스를 problem_id별로 그룹핑
    indices_set = set(indices)
    problem_groups = defaultdict(lambda: {"correct": [], "incorrect": []})

    for idx in indices:
        meta = metadata[idx]
        pid = meta.get("problem_id")
        is_correct = meta.get("is_correct")

        if pid is None:
            continue

        if is_correct:
            problem_groups[pid]["correct"].append(idx)
        else:
            problem_groups[pid]["incorrect"].append(idx)

    # correct와 incorrect가 모두 있는 problem에서 쌍 생성
    pairs = []
    valid_problems = 0

    for pid, groups in problem_groups.items():
        correct_list = groups["correct"]
        incorrect_list = groups["incorrect"]

        if len(correct_list) > 0 and len(incorrect_list) > 0:
            valid_problems += 1
            # 모든 조합 생성
            for c_idx in correct_list:
                for i_idx in incorrect_list:
                    pairs.append({"correct_idx": c_idx, "incorrect_idx": i_idx})

    logger.info(
        f"Pairwise 쌍 생성: {len(indices):,}개 인덱스 → "
        f"{valid_problems:,}개 문제에서 {len(pairs):,}개 쌍 생성"
    )

    # 쌍이 없으면 에러
    if len(pairs) == 0:
        # 디버깅 정보 수집
        total_correct = sum(len(g["correct"]) for g in problem_groups.values())
        total_incorrect = sum(len(g["incorrect"]) for g in problem_groups.values())

        raise ValueError(
            f"Pairwise 쌍 생성 실패: 동일 problem_id 내에 correct와 incorrect가 "
            f"모두 있는 문제가 없습니다.\n"
            f"선택된 인덱스 분포: correct={total_correct:,}, incorrect={total_incorrect:,}\n"
            f"correct_ratio를 0.5에 가깝게 설정하거나 n_samples를 늘려보세요."
        )

    # 섞기
    random.shuffle(pairs)

    return pairs


def _read_jsonl_pairwise(
    jsonl_path: Path,
    pairs: list[dict],
) -> list[dict]:
    """Pairwise 샘플 로딩

    Args:
        jsonl_path: JSONL 파일 경로
        pairs: [{"correct_idx": int, "incorrect_idx": int}, ...] 리스트

    Returns:
        [{instruction, input, correct_output, incorrect_output}, ...] 리스트

    Raises:
        ValueError: 요청된 쌍 수와 로드된 쌍 수가 다른 경우
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 파일이 존재하지 않습니다: {jsonl_path}")

    # 필요한 모든 인덱스 수집
    all_indices = set()
    for pair in pairs:
        all_indices.add(pair["correct_idx"])
        all_indices.add(pair["incorrect_idx"])

    logger.info(f"Pairwise 로딩: {len(pairs):,} 쌍, {len(all_indices):,} 유니크 인덱스")

    # 인덱스별 샘플 로드
    idx_to_sample = {}
    target_set = set(all_indices)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx in target_set:
                try:
                    sample = json.loads(line.strip())
                    idx_to_sample[line_idx] = sample
                except json.JSONDecodeError as e:
                    logger.warning(f"라인 {line_idx} 파싱 오류: {e}")
                    continue

                # 모든 목표 샘플을 찾았으면 종료
                if len(idx_to_sample) >= len(all_indices):
                    break

    # 쌍 구성
    result = []
    skipped = 0
    for pair in pairs:
        correct_idx = pair["correct_idx"]
        incorrect_idx = pair["incorrect_idx"]

        if correct_idx not in idx_to_sample or incorrect_idx not in idx_to_sample:
            skipped += 1
            continue

        correct_sample = idx_to_sample[correct_idx]
        incorrect_sample = idx_to_sample[incorrect_idx]

        result.append({
            "instruction": correct_sample["instruction"],
            "input": correct_sample.get("input", ""),
            "correct_output": correct_sample["output"],
            "incorrect_output": incorrect_sample["output"],
        })

    if skipped > 0:
        logger.warning(f"Pairwise 로딩 중 {skipped:,}개 쌍 스킵 (인덱스 누락)")

    # 요청된 쌍 수와 로드된 쌍 수 검증
    if len(result) < len(pairs):
        missing = len(pairs) - len(result)
        raise ValueError(
            f"Pairwise 로딩 실패: {missing:,}개 쌍 누락. "
            f"요청: {len(pairs):,}, 로드: {len(result):,}. "
            f"JSONL 파일에 누락된 인덱스가 있습니다."
        )

    logger.info(f"Pairwise JSONL 읽기 완료: {len(result):,} 쌍 로드")

    return result
