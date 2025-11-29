"""JSONL → HuggingFace Dataset 로딩 (메타데이터 기반)

메타데이터 기반 효율적 로딩:
- 전체 데이터 로드 없이 메타데이터만으로 샘플 선택
- Difficulty 기반 가중 샘플링 (bins, weights 필수)
- use_pairwise 옵션으로 Pairwise 포맷 변환 지원
- Unique pair 보장 (correct_idx, incorrect_idx 모두 고유)
"""

from pathlib import Path
from typing import Optional
import logging
import random
import json

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    split: str,
    sampling_config: dict,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> Dataset:
    """JSONL 파일을 메타데이터 기반으로 효율적 로딩 (Rank-aware 분산)

    Problem-level 쌍 샘플링을 지원합니다.
    use_pairwise=true일 때 n_samples는 최종 쌍 수를 의미합니다.

    분산 환경에서 각 GPU가 자기 담당 샘플만 로드합니다.
    재현성을 위해 모든 rank가 동일한 시드로 전체 샘플을 계산한 후,
    rank::world_size 패턴으로 서브셋을 선택합니다.

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval)
        split: 데이터 스플릿 (train, validation, test)
        sampling_config: 샘플링 설정 딕셔너리
            - use_pairwise: Pairwise 포맷 변환 여부 (기본: False)
            - n_samples: 샘플링할 샘플/쌍 수
            - difficulty_weights: 난이도별 가중치
            - difficulty_bins: 난이도 구간 정의
        seed: 랜덤 시드
        rank: 현재 프로세스의 global rank (기본: 0)
        world_size: 전체 프로세스 수 (기본: 1)

    Returns:
        Dataset (분산 환경에서는 1/world_size 크기)
    """
    logger.info(f"데이터셋 로딩 시작: {dataset_name}/{split}")

    # 샘플링 설정 추출
    use_pairwise = sampling_config.get("use_pairwise", False)
    n_samples = sampling_config.get("n_samples", 100000)
    difficulty_weights = sampling_config.get("difficulty_weights")
    difficulty_bins = sampling_config.get("difficulty_bins")
    max_pairs_per_problem = sampling_config.get("max_pairs_per_problem")  # problem당 cap

    # JSONL 경로 확인
    data_files = _get_dataset_paths(dataset_name)
    if split not in data_files:
        raise ValueError(
            f"스플릿 '{split}'이 존재하지 않습니다. "
            f"가능한 스플릿: {list(data_files.keys())}"
        )
    jsonl_path = Path(data_files[split])

    # difficulty_bins + difficulty_weights 필수
    if not (difficulty_weights and difficulty_bins):
        raise ValueError(
            "difficulty_bins와 difficulty_weights는 필수입니다. "
            "sampling_config에 두 값을 모두 설정하세요."
        )

    problem_index_map = _load_problem_index_map(dataset_name, split)

    if not problem_index_map:
        raise FileNotFoundError(
            f"problem_index_map이 메타데이터에 없습니다: {dataset_name}/{split}\n"
            f"먼저 'python scripts/create_storage/extract_metadata.py' 를 실행하세요."
        )

    # Unique pair 샘플링 (공통 로직)
    all_pairs = _sample_unique_pairs(
        problem_index_map=problem_index_map,
        n_samples=n_samples,
        difficulty_weights=difficulty_weights,
        difficulty_bins=difficulty_bins,
        seed=seed,
        max_pairs_per_problem=max_pairs_per_problem,
    )

    if use_pairwise:
        # Pairwise 모드: correct + incorrect 쌍 그대로 사용
        if world_size > 1:
            rank_pairs = all_pairs[rank::world_size]
            logger.info(
                f"[Rank {rank}/{world_size}] 전체 {len(all_pairs):,} 쌍 중 "
                f"{len(rank_pairs):,} 쌍 로드 (분산 학습)"
            )
        else:
            rank_pairs = all_pairs
            logger.info(f"Pairwise 샘플링 완료: {len(rank_pairs):,} 쌍 (로컬 환경)")

        samples = _read_jsonl_pairwise(jsonl_path, rank_pairs)
    else:
        # Pointwise 모드: unique pair에서 correct 샘플만 추출 (이미 unique 보장)
        correct_indices = [p["correct_idx"] for p in all_pairs]

        if world_size > 1:
            rank_indices = correct_indices[rank::world_size]
            logger.info(
                f"[Rank {rank}/{world_size}] 전체 {len(correct_indices):,} 샘플 중 "
                f"{len(rank_indices):,} 샘플 로드 (분산 학습)"
            )
        else:
            rank_indices = correct_indices
            logger.info(f"Pointwise 샘플링 완료: {len(rank_indices):,} unique correct 샘플")

        samples = _read_jsonl_by_indices(jsonl_path, rank_indices)

    # HuggingFace Dataset으로 변환
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


# problem_index_map 전역 캐시
_problem_index_map_cache: dict[str, dict] = {}


def _load_problem_index_map(
    dataset_name: str,
    split: str,
) -> dict[str, dict]:
    """메타데이터에서 problem_index_map 로드 (캐싱 지원)

    Args:
        dataset_name: 데이터셋 이름
        split: 스플릿 이름

    Returns:
        {problem_id: {difficulty, correct_indices, incorrect_indices}}
    """
    cache_key = f"{dataset_name}/{split}"

    if cache_key in _problem_index_map_cache:
        cached = _problem_index_map_cache[cache_key]
        logger.info(f"problem_index_map 캐시 히트: {len(cached):,} problems ({cache_key})")
        return cached

    base_dir = Path("storage/datasets")
    dataset_dir = base_dir / dataset_name / "processed"

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
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        problem_index_map = data.get("problem_index_map", {})

        if not problem_index_map:
            logger.warning(
                f"problem_index_map이 메타데이터에 없습니다. "
                f"extract_metadata.py를 다시 실행하세요."
            )
            return {}

        _problem_index_map_cache[cache_key] = problem_index_map
        logger.info(f"problem_index_map 로드 완료: {len(problem_index_map):,} problems")

        return problem_index_map
    except Exception as e:
        logger.error(f"problem_index_map 로드 실패: {e}")
        return {}


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
# Unique Pair 샘플링
# ============================================================================


def _sample_unique_pairs(
    problem_index_map: dict[str, dict],
    n_samples: int,
    difficulty_weights: dict,
    difficulty_bins: dict,
    seed: int,
    max_pairs_per_problem: Optional[int] = None,
) -> list[dict]:
    """Difficulty 기반 unique pair 샘플링

    각 correct/incorrect 인덱스가 최대 1번만 사용되도록 보장.
    problem당 max_pairs_per_problem개로 제한하여 다양성 확보.

    알고리즘 (O(n) 복잡도):
    1. 각 bin에서 problem별 correct/incorrect 리스트 셔플
    2. min(correct수, incorrect수, max_pairs_per_problem) 개수만큼 1:1 매핑
    3. difficulty weight 비례로 bin별 할당
    4. 전체 셔플 후 n_samples개 반환

    Args:
        problem_index_map: {problem_id: {difficulty, correct_indices, incorrect_indices}}
        n_samples: 샘플링할 쌍 수 (= correct sample 수)
        difficulty_weights: 난이도별 가중치
        difficulty_bins: 난이도 구간
        seed: 랜덤 시드
        max_pairs_per_problem: problem당 최대 쌍 수 (None이면 무제한)

    Returns:
        [{"correct_idx": int, "incorrect_idx": int, "problem_id": str}, ...]

    Raises:
        ValueError: 가용 쌍이 n_samples보다 적은 경우
    """
    random.seed(seed)

    # 1. Difficulty bin별로 problem 분류
    bin_problems: dict[str, list[dict]] = {bin_name: [] for bin_name in difficulty_bins}

    for pid, info in problem_index_map.items():
        difficulty = info.get("difficulty")
        correct_indices = info.get("correct_indices", [])
        incorrect_indices = info.get("incorrect_indices", [])

        # 쌍 생성 가능한 problem만 포함
        if len(correct_indices) == 0 or len(incorrect_indices) == 0:
            continue

        if difficulty is None:
            continue

        # 해당 difficulty bin 찾기
        for bin_name, bin_range in difficulty_bins.items():
            min_d, max_d = bin_range[0], bin_range[1]
            if min_d <= difficulty <= max_d:
                bin_problems[bin_name].append({
                    "problem_id": pid,
                    "correct_indices": list(correct_indices),
                    "incorrect_indices": list(incorrect_indices),
                })
                break

    # 2. 각 problem에서 unique pair 생성 (preshuffle + 1:1 매핑)
    bin_pairs: dict[str, list[dict]] = {bin_name: [] for bin_name in difficulty_bins}

    for bin_name, problems in bin_problems.items():
        for p in problems:
            correct_list = p["correct_indices"]
            incorrect_list = p["incorrect_indices"]

            # 셔플
            random.shuffle(correct_list)
            random.shuffle(incorrect_list)

            # 1:1 unique 매핑
            n_available = min(len(correct_list), len(incorrect_list))
            if max_pairs_per_problem:
                n_available = min(n_available, max_pairs_per_problem)

            for i in range(n_available):
                bin_pairs[bin_name].append({
                    "correct_idx": correct_list[i],
                    "incorrect_idx": incorrect_list[i],
                    "problem_id": p["problem_id"],
                })

    # 3. 가용 데이터 로깅
    logger.info("=== Unique Pair 샘플링 ===")
    if max_pairs_per_problem:
        logger.info(f"max_pairs_per_problem: {max_pairs_per_problem}")

    total_available = 0
    for bin_name in difficulty_bins:
        n_pairs = len(bin_pairs.get(bin_name, []))
        n_problems = len(bin_problems.get(bin_name, []))
        total_available += n_pairs
        logger.info(f"{bin_name}: {n_problems:,} problems, {n_pairs:,} unique 쌍")

    logger.info(f"전체 가용: {total_available:,} unique 쌍")

    # 4. 비어있는 bin 감지 및 weight 재분배
    valid_weights = {}
    for bin_name, weight in difficulty_weights.items():
        if weight <= 0:
            continue
        if len(bin_pairs.get(bin_name, [])) == 0:
            logger.warning(f"Bin '{bin_name}'에 유효한 쌍이 없습니다. weight 재분배됨.")
        else:
            valid_weights[bin_name] = weight

    # Weight 정규화
    if valid_weights:
        total_weight = sum(valid_weights.values())
        normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
    else:
        raise ValueError("모든 bin에 유효한 데이터가 없습니다.")

    # 5. Difficulty weight에 따라 bin별 샘플 수 할당 (나머지 분배)
    bin_targets = {}
    total_assigned = 0
    for bin_name, weight in normalized_weights.items():
        target = int(n_samples * weight)
        bin_targets[bin_name] = target
        total_assigned += target

    # 나머지 분배 (가용 데이터가 많은 bin 우선)
    remainder = n_samples - total_assigned
    if remainder > 0:
        sorted_bins = sorted(
            normalized_weights.keys(),
            key=lambda b: len(bin_pairs.get(b, [])),
            reverse=True
        )
        for i in range(remainder):
            bin_targets[sorted_bins[i % len(sorted_bins)]] += 1

    # 6. 할당량에 따라 추출
    selected_pairs = []
    sampling_results = {}

    for bin_name, bin_target in bin_targets.items():
        available_pairs = bin_pairs.get(bin_name, [])

        # 셔플 후 할당량만큼 추출
        random.shuffle(available_pairs)
        actual = min(bin_target, len(available_pairs))
        selected_pairs.extend(available_pairs[:actual])

        sampling_results[bin_name] = {"target": bin_target, "actual": actual}

    # 7. 샘플링 결과 로깅
    logger.info("=== 샘플링 결과 ===")
    total_actual = 0
    for bin_name, result in sampling_results.items():
        target = result["target"]
        actual = result["actual"]
        total_actual += actual
        pct = (actual / target * 100) if target > 0 else 0
        logger.info(f"{bin_name}: 목표={target:,}, 실제={actual:,} ({pct:.0f}%)")

    logger.info(f"--- 합계: {len(selected_pairs):,}개 unique 쌍")

    # 8. 최종 셔플
    random.shuffle(selected_pairs)

    # 9. 부족 시 에러
    if len(selected_pairs) < n_samples:
        shortage = n_samples - len(selected_pairs)
        raise ValueError(
            f"데이터 부족: {shortage:,}개 쌍 부족. "
            f"요청: {n_samples:,}, 가용: {len(selected_pairs):,}. "
            f"n_samples를 {len(selected_pairs):,} 이하로 설정하거나 "
            f"max_pairs_per_problem을 늘리세요."
        )

    return selected_pairs[:n_samples]


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
