# Problem-level 쌍 기반 샘플링 구조 개선 계획

## 배경: 기존 계획의 한계

### 관련 문서
- `use_pairwise_restructure_plan.md`: `sampling_method` → `use_pairwise` 구조 통일
- `pairwise_pipeline_integration_plan.md`: verifiable 파이프라인 pairwise 통합

### 기존 접근 방식의 문제
```
[기존 흐름]
1. n_samples=100000, correct_ratio=0.5 설정
2. Difficulty 샘플링 → 5만 correct + 5만 incorrect 인덱스 선택
3. 선택된 인덱스에서 problem_id로 쌍 매칭
4. 문제: 쌍 수가 n_samples와 불일치 (문제별 분포에 따라 가변)
```

**핵심 문제**: `n_samples`가 "인덱스 수"를 의미하여, 최종 쌍 수를 보장할 수 없음

---

## 새로운 설계: Problem-level 쌍 샘플링

### 핵심 원칙
1. **n_samples = 최종 쌍 개수**로 정의 일관성 유지
2. **Problem 단위로 쌍을 먼저 구성**한 후 difficulty 기반 샘플링
3. **메타데이터에 problem_index_map 추가**하여 효율적 샘플링

### 변경된 흐름
```
[새로운 흐름]
1. 메타데이터에서 problem_index_map 로드
   - {problem_id: {difficulty, correct_indices, incorrect_indices}}
2. Problem별 가용 쌍 풀 구성
   - 쌍 수 = len(correct) × len(incorrect)
3. Difficulty 기반으로 n_samples개 쌍 샘플링
   - difficulty_weights에 따라 bin별 할당
4. use_pairwise=true → 쌍 반환
   use_pairwise=false → correct만 반환
```

### 결과 보장
| 설정 | use_pairwise=true | use_pairwise=false |
|------|-------------------|---------------------|
| n_samples: 100000 | 10만 쌍 | 10만 correct 샘플 |
| difficulty_weights | 각 bin에서 비율대로 | 각 bin에서 비율대로 |

---

## 메타데이터 구조 확장

### 현재 구조
```json
{
  "metadata": [
    {"is_correct": true, "difficulty": 7, "problem_id": "abc123"},
    {"is_correct": false, "difficulty": 7, "problem_id": "abc123"},
    ...
  ],
  "stats": {...}
}
```

### 확장 구조
```json
{
  "metadata": [...],
  "problem_index_map": {
    "abc123": {
      "difficulty": 7,
      "correct_indices": [0, 5, 12, 45],
      "incorrect_indices": [3, 8, 15, 22]
    },
    "def456": {
      "difficulty": 9,
      "correct_indices": [1, 6],
      "incorrect_indices": [4, 9, 10, 18]
    }
  },
  "stats": {...}
}
```

### 장점
- 런타임에 problem_id 그룹핑 불필요
- 쌍 생성 시 O(1) 조회
- 메모리 효율적 (인덱스만 저장)

---

## Phase별 구현 계획

### Phase 1: 메타데이터 확장 (extract_metadata.py)

**목표**: problem_index_map 생성 로직 추가

**파일**: `scripts/create_storage/extract_metadata.py`

#### Step 1.1: extract_metadata_from_jsonl() 수정

```python
def extract_metadata_from_jsonl(jsonl_path: Path, output_path: Path) -> dict:
    """JSONL 파일에서 메타데이터 추출 (problem_index_map 포함)"""

    metadata_list = []
    problem_index_map = {}  # 신규 추가

    for idx, line in enumerate(f):
        item = json.loads(line.strip())

        # 기존: 개별 메타데이터 추출
        meta = {
            "is_correct": item.get("is_correct"),
            "difficulty": item.get("metadata", {}).get("difficulty"),
            "problem_id": item.get("metadata", {}).get("problem_id"),
        }
        metadata_list.append(meta)

        # 신규: problem_index_map 구성
        pid = meta.get("problem_id")
        if pid:
            if pid not in problem_index_map:
                problem_index_map[pid] = {
                    "difficulty": meta.get("difficulty"),
                    "correct_indices": [],
                    "incorrect_indices": [],
                }

            if meta.get("is_correct"):
                problem_index_map[pid]["correct_indices"].append(idx)
            else:
                problem_index_map[pid]["incorrect_indices"].append(idx)

    # 출력 데이터
    output_data = {
        "metadata": metadata_list,
        "problem_index_map": problem_index_map,  # 신규 추가
        "stats": stats,
    }
```

#### Step 1.2: 통계 정보 확장

```python
stats = {
    "total": len(metadata_list),
    "correct": n_correct,
    "incorrect": n_incorrect,
    "n_problems": len(problem_index_map),
    "n_valid_problems": sum(
        1 for p in problem_index_map.values()
        if len(p["correct_indices"]) > 0 and len(p["incorrect_indices"]) > 0
    ),
    "total_possible_pairs": sum(
        len(p["correct_indices"]) * len(p["incorrect_indices"])
        for p in problem_index_map.values()
    ),
    "difficulty_dist": {...},
}
```

---

### Phase 2: datasets.py 핵심 로직 재구조화

**목표**: Problem-level 쌍 샘플링 구현

**파일**: `src/weighted_mtp/data/datasets.py`

#### Step 2.1: _load_problem_index_map() 함수 추가

```python
def _load_problem_index_map(
    dataset_name: str,
    split: str,
) -> dict[str, dict]:
    """메타데이터에서 problem_index_map 로드

    Returns:
        {problem_id: {difficulty, correct_indices, incorrect_indices}}
    """
    # 메타데이터 로드 (기존 함수 재사용)
    metadata_path = _get_metadata_path(dataset_name, split)

    with open(metadata_path, "r") as f:
        data = json.load(f)

    return data.get("problem_index_map", {})
```

#### Step 2.2: _sample_pairs_by_difficulty() 함수 신규 생성

```python
def _sample_pairs_by_difficulty(
    problem_index_map: dict[str, dict],
    n_samples: int,
    difficulty_weights: dict,
    difficulty_bins: dict,
    seed: int,
) -> list[dict]:
    """Problem-level 쌍을 difficulty 기반으로 샘플링

    Reservoir sampling으로 메모리 효율적 구현

    Args:
        problem_index_map: {problem_id: {difficulty, correct_indices, incorrect_indices}}
        n_samples: 샘플링할 쌍 수
        difficulty_weights: 난이도별 가중치
        difficulty_bins: 난이도 구간
        seed: 랜덤 시드

    Returns:
        [{"correct_idx": int, "incorrect_idx": int, "problem_id": str}, ...]
    """
    random.seed(seed)

    # 1. Difficulty bin별로 problem 분류
    bin_problems = {bin_name: [] for bin_name in difficulty_bins}

    for pid, info in problem_index_map.items():
        difficulty = info["difficulty"]
        correct_indices = info["correct_indices"]
        incorrect_indices = info["incorrect_indices"]

        # 쌍 생성 가능한 problem만 포함
        if len(correct_indices) == 0 or len(incorrect_indices) == 0:
            continue

        # 해당 difficulty bin 찾기
        for bin_name, (min_d, max_d) in difficulty_bins.items():
            if min_d <= difficulty <= max_d:
                # 가용 쌍 수 계산
                n_pairs = len(correct_indices) * len(incorrect_indices)
                bin_problems[bin_name].append({
                    "problem_id": pid,
                    "correct_indices": correct_indices,
                    "incorrect_indices": incorrect_indices,
                    "n_pairs": n_pairs,
                })
                break

    # 2. Difficulty weight에 따라 bin별 샘플 수 할당
    selected_pairs = []

    for bin_name, weight in difficulty_weights.items():
        if weight <= 0:
            continue

        bin_n_samples = int(n_samples * weight)
        problems = bin_problems.get(bin_name, [])

        if not problems:
            logger.warning(f"Bin '{bin_name}'에 유효한 problem이 없습니다.")
            continue

        # 3. Reservoir sampling으로 쌍 추출
        bin_pairs = _reservoir_sample_pairs(
            problems=problems,
            n_samples=bin_n_samples,
            seed=seed + hash(bin_name),
        )

        selected_pairs.extend(bin_pairs)

    # 섞기
    random.shuffle(selected_pairs)

    # 부족 시 에러
    if len(selected_pairs) < n_samples:
        raise ValueError(
            f"데이터 부족: {n_samples - len(selected_pairs)}개 쌍 부족. "
            f"요청: {n_samples}, 가용: {len(selected_pairs)}"
        )

    return selected_pairs[:n_samples]
```

#### Step 2.3: _reservoir_sample_pairs() 함수 추가

```python
def _reservoir_sample_pairs(
    problems: list[dict],
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Problem 리스트에서 쌍을 reservoir sampling

    메모리 효율: 모든 조합을 생성하지 않고 스트리밍 샘플링

    Args:
        problems: [{problem_id, correct_indices, incorrect_indices, n_pairs}, ...]
        n_samples: 샘플링할 쌍 수
        seed: 랜덤 시드

    Returns:
        [{"correct_idx": int, "incorrect_idx": int, "problem_id": str}, ...]
    """
    random.seed(seed)

    # 총 가용 쌍 수 계산
    total_pairs = sum(p["n_pairs"] for p in problems)

    if total_pairs <= n_samples:
        # 전체 쌍이 목표보다 적으면 모두 반환
        all_pairs = []
        for p in problems:
            for c_idx in p["correct_indices"]:
                for i_idx in p["incorrect_indices"]:
                    all_pairs.append({
                        "correct_idx": c_idx,
                        "incorrect_idx": i_idx,
                        "problem_id": p["problem_id"],
                    })
        return all_pairs

    # Reservoir sampling
    reservoir = []
    seen = 0

    for p in problems:
        for c_idx in p["correct_indices"]:
            for i_idx in p["incorrect_indices"]:
                seen += 1

                if len(reservoir) < n_samples:
                    reservoir.append({
                        "correct_idx": c_idx,
                        "incorrect_idx": i_idx,
                        "problem_id": p["problem_id"],
                    })
                else:
                    # 확률적 교체
                    j = random.randint(0, seen - 1)
                    if j < n_samples:
                        reservoir[j] = {
                            "correct_idx": c_idx,
                            "incorrect_idx": i_idx,
                            "problem_id": p["problem_id"],
                        }

    return reservoir
```

#### Step 2.4: load_dataset() 함수 수정

```python
def load_dataset(
    dataset_name: str,
    split: str,
    sampling_config: dict,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> Dataset:
    """JSONL 데이터셋 로딩 (Problem-level 쌍 샘플링 지원)"""

    use_pairwise = sampling_config.get("use_pairwise", False)
    n_samples = sampling_config.get("n_samples", 100000)
    difficulty_weights = sampling_config.get("difficulty_weights")
    difficulty_bins = sampling_config.get("difficulty_bins")

    # JSONL 경로
    jsonl_path = _get_jsonl_path(dataset_name, split)

    if use_pairwise or difficulty_weights:
        # Problem-level 쌍 기반 샘플링
        problem_index_map = _load_problem_index_map(dataset_name, split)

        all_pairs = _sample_pairs_by_difficulty(
            problem_index_map=problem_index_map,
            n_samples=n_samples,
            difficulty_weights=difficulty_weights,
            difficulty_bins=difficulty_bins,
            seed=seed,
        )

        # 분산 처리
        if world_size > 1:
            rank_pairs = all_pairs[rank::world_size]
        else:
            rank_pairs = all_pairs

        if use_pairwise:
            # Pairwise 모드: 쌍 데이터 로드
            samples = _read_jsonl_pairwise(jsonl_path, rank_pairs)
        else:
            # Non-pairwise 모드: correct만 로드
            correct_indices = [p["correct_idx"] for p in rank_pairs]
            samples = _read_jsonl_by_indices(jsonl_path, correct_indices)

    else:
        # 기존 방식 (difficulty 설정 없는 경우)
        metadata = _load_metadata(dataset_name, split)
        all_indices = _compute_sampling_indices(metadata, sampling_config, seed)

        if world_size > 1:
            rank_indices = all_indices[rank::world_size]
        else:
            rank_indices = all_indices

        samples = _read_jsonl_by_indices(jsonl_path, rank_indices)

    return Dataset.from_list(samples)
```

---

### Phase 3: 삭제 대상 (기존 중복 코드)

**원칙 4에 따라 하위 호환성 없이 완전 삭제**

#### datasets.py 삭제 목록

| 함수/코드 | 위치 | 이유 |
|-----------|------|------|
| `_create_pairs_from_indices()` | L601-691 | 새로운 `_sample_pairs_by_difficulty()`로 대체 |
| `_sample_by_difficulty()` 내 correct/incorrect 분리 로직 | L291-450 | Problem-level 샘플링으로 통합 |

#### 삭제하지 않는 코드

| 함수 | 이유 |
|------|------|
| `_read_jsonl_pairwise()` | 쌍 데이터 로딩에 재사용 |
| `_read_jsonl_by_indices()` | 인덱스 기반 로딩에 재사용 |
| `_load_metadata()` | 기존 호환성 유지 (difficulty 없는 경우) |

---

### Phase 4: Config 업데이트

**파일**: `configs/critic/critic_pairwise.yaml`

```yaml
data_sampling:
  seed: 42
  val_n_samples: 2000
  use_pairwise: true

  n_samples: 100000  # 최종 쌍 수 (10만 쌍 보장)

  difficulty_bins:
    diff_7: [7, 7]
    else: [8, 25]
  difficulty_weights:
    diff_7: 0.35    # 35,000 쌍
    else: 0.65      # 65,000 쌍

  # correct_ratio는 pairwise에서 무시됨 (항상 1:1)
```

---

### Phase 5: 테스트 코드 업데이트

#### Step 5.1: Unit Test - Problem-level 샘플링

**파일**: `tests/unit/test_problem_level_sampling.py`

```python
def test_sample_pairs_by_difficulty_returns_exact_n_samples():
    """n_samples 정확히 반환하는지 검증"""
    problem_index_map = {
        "p1": {"difficulty": 7, "correct_indices": [0, 1], "incorrect_indices": [2, 3]},
        "p2": {"difficulty": 9, "correct_indices": [4], "incorrect_indices": [5, 6, 7]},
    }

    pairs = _sample_pairs_by_difficulty(
        problem_index_map=problem_index_map,
        n_samples=5,
        difficulty_weights={"diff_7": 0.5, "else": 0.5},
        difficulty_bins={"diff_7": [7, 7], "else": [8, 25]},
        seed=42,
    )

    assert len(pairs) == 5


def test_sample_pairs_by_difficulty_raises_on_insufficient_data():
    """데이터 부족 시 에러 발생 검증"""
    problem_index_map = {
        "p1": {"difficulty": 7, "correct_indices": [0], "incorrect_indices": [1]},
    }

    with pytest.raises(ValueError, match="데이터 부족"):
        _sample_pairs_by_difficulty(
            problem_index_map=problem_index_map,
            n_samples=100,  # 가용 쌍: 1개
            difficulty_weights={"diff_7": 1.0},
            difficulty_bins={"diff_7": [7, 7]},
            seed=42,
        )
```

#### Step 5.2: Integration Test - 전체 파이프라인

**파일**: `tests/integration/test_pipeline_critic_pairwise.py` (수정)

```python
def test_pairwise_dataloader_returns_exact_samples():
    """DataLoader가 설정된 n_samples만큼 쌍 반환"""
    config = {
        "use_pairwise": True,
        "n_samples": 100,
        "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
        "difficulty_weights": {"diff_7": 0.5, "else": 0.5},
    }

    dataloader = create_dataloader(
        dataset_path="storage/datasets/codecontests/processed/train.jsonl",
        tokenizer=tokenizer,
        batch_size=10,
        max_length=512,
        sampling_config=config,
    )

    total_samples = sum(len(batch["pos_input_ids"]) for batch in dataloader)
    assert total_samples == 100
```

---

## 구현 순서

| 순서 | Phase | 파일 | 의존성 |
|------|-------|------|--------|
| 1 | Phase 1 | extract_metadata.py | 없음 |
| 2 | - | 메타데이터 재생성 실행 | Phase 1 |
| 3 | Phase 2 | datasets.py | Phase 1, 2 |
| 4 | Phase 3 | datasets.py (삭제) | Phase 2 |
| 5 | Phase 4 | configs/*.yaml | Phase 2 |
| 6 | Phase 5 | tests/*.py | Phase 4 |

---

## 검증 체크리스트

- [ ] 메타데이터에 problem_index_map 정상 생성
- [ ] `_sample_pairs_by_difficulty()`가 정확히 n_samples개 쌍 반환
- [ ] difficulty_weights 비율대로 bin별 샘플 분배
- [ ] use_pairwise=false 시 correct만 반환
- [ ] 분산 환경에서 각 rank가 1/world_size 샘플 로드
- [ ] 데이터 부족 시 명확한 에러 메시지

---

## 예상 영향도

| 항목 | 영향 |
|------|------|
| Breaking Change | Config의 correct_ratio가 pairwise에서 무시됨 |
| 성능 | Reservoir sampling으로 메모리 효율 개선 |
| 일관성 | n_samples가 최종 샘플 수로 보장됨 |
| 유지보수 | 중복 함수 제거로 코드 단순화 |

---

## 승인 요청

위 계획에 대해 검토 및 승인 부탁드립니다.

- [ ] Phase 1: extract_metadata.py - problem_index_map 추가
- [ ] Phase 2: datasets.py - Problem-level 쌍 샘플링 구현
- [ ] Phase 3: 기존 중복 코드 삭제
- [ ] Phase 4: Config 파일 업데이트
- [ ] Phase 5: 테스트 코드 작성/수정
