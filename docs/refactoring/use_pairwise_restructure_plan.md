# use_pairwise 옵션 재구조화 계획

## 문제 정의

현재 `sampling_method: "pairwise"`가 `sampling_method: "difficulty"`와 별개의 샘플링 방식으로 구현되어 있음.
이로 인해 Pairwise 학습 시 다른 실험들과 **데이터 분포가 달라지는 문제** 발생.

### 현재 구조 (문제점)
```yaml
# Pairwise 모드 - difficulty 기반 샘플링과 완전 다른 데이터 분포
data_sampling:
  sampling_method: "pairwise"  # difficulty와 배타적
  pairwise:
    n_pairs: 50000  # difficulty 설정 무시됨
```

### 목표 구조
```yaml
# Difficulty 기반 샘플링 + Pairwise 포맷 옵션
data_sampling:
  seed: 42
  val_n_samples: 1000
  use_pairwise: true  # 옵션으로 변경

  n_samples: 100000
  auto_data_balancing: true
  correct_ratio: 0.5
  bins:
    diff_7: [7, 7]
    else: [8, 25]
  weights:
    diff_7: 0.35
    else: 0.65
```

---

## 변경 범위 분석

### 영향받는 파일

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `src/weighted_mtp/data/datasets.py` | 핵심 수정 | sampling_method 분기 제거, use_pairwise 로직 통합 |
| `src/weighted_mtp/data/dataloader.py` | 수정 | use_pairwise 기반 collator 선택 |
| `src/weighted_mtp/pipelines/run_critic.py` | 수정 | use_pairwise 플래그 읽기 |
| `configs/critic/critic_pairwise.yaml` | 재작성 | 새 구조로 변경 |
| `configs/critic/critic_mlp.yaml` | 수정 | use_pairwise: false 추가 |
| `tests/unit/test_pairwise_*.py` | 수정 | 새 구조에 맞게 테스트 업데이트 |
| `tests/integration/test_pipeline_critic_pairwise.py` | 수정 | 새 구조에 맞게 테스트 업데이트 |

### 삭제 대상 (중복/불필요 코드)

1. `datasets.py`의 `sampling_method` 분기 로직
2. `datasets.py`의 `_load_pairwise_dataset()` 독립 함수 → difficulty 샘플링 내부로 통합
3. `run_critic.py`의 `is_pairwise = (sampling_method == "pairwise")` 패턴
4. Config의 `sampling_method` 필드 및 `pairwise` 독립 섹션

---

## 단계별 수정 계획

### Phase 1: datasets.py 핵심 로직 재구조화

**목표**: difficulty 기반 샘플링 후 pairwise 포맷 변환

#### Step 1.1: load_dataset() 함수 시그니처 및 분기 로직 수정

**변경 전** (`datasets.py:28-86`):
```python
def load_dataset(..., sampling_config: dict, ...):
    # sampling_method 분기
    sampling_method = sampling_config.get("sampling_method")
    if sampling_method == "pairwise":
        return _load_pairwise_dataset(...)  # 완전 별개 로직

    # difficulty 로직...
```

**변경 후**:
```python
def load_dataset(..., sampling_config: dict, ...):
    # 1. Difficulty 기반 인덱스 계산 (항상 실행)
    all_indices = _compute_sampling_indices_from_metadata(...)

    # 2. Rank 분산 처리
    rank_indices = all_indices[rank::world_size]

    # 3. use_pairwise에 따라 로딩 방식 결정
    use_pairwise = sampling_config.get("use_pairwise", False)
    if use_pairwise:
        samples = _load_as_pairwise(jsonl_path, rank_indices, metadata)
    else:
        samples = _read_jsonl_by_indices(jsonl_path, rank_indices)

    return Dataset.from_list(samples)
```

#### Step 1.2: _load_as_pairwise() 함수 신규 생성

기존 `_load_pairwise_dataset()`, `_sample_pairwise()`, `_read_jsonl_pairwise()` 로직을 통합.

**핵심 변경점**:
- 기존: 메타데이터에서 직접 (correct, incorrect) 쌍 생성
- 변경: difficulty 기반으로 선택된 인덱스 내에서 쌍 생성

```python
def _load_as_pairwise(
    jsonl_path: Path,
    indices: list[int],  # difficulty 샘플링으로 선택된 인덱스
    metadata: list[dict],
) -> list[dict]:
    """Difficulty 샘플링된 인덱스를 pairwise 포맷으로 변환

    Args:
        jsonl_path: JSONL 파일 경로
        indices: difficulty 샘플링으로 선택된 인덱스
        metadata: 전체 메타데이터

    Returns:
        [{instruction, input, correct_output, incorrect_output}, ...] 리스트
    """
    # 1. 선택된 인덱스를 problem_id별로 그룹핑
    # 2. 각 problem_id 내에서 (correct, incorrect) 쌍 생성
    # 3. JSONL에서 해당 쌍의 데이터 로딩
```

#### Step 1.3: 삭제할 함수들

- `_load_pairwise_dataset()`: `_load_as_pairwise()`로 대체
- `_sample_pairwise()`: 로직 통합 후 삭제

**주의**: `_read_jsonl_pairwise()`는 `_load_as_pairwise()` 내부에서 재사용 가능, 별도 함수로 유지

#### Step 1.4: _compute_sampling_indices_from_metadata() 수정

**삭제할 코드**:
```python
sampling_method = sampling_config.get("sampling_method")

# Difficulty 방식: 난이도 기반 샘플링
if sampling_method == "difficulty":
    # ...
else:
    raise ValueError(f"잘못된 sampling_method: {sampling_method}")
```

**변경 후**:
```python
# sampling_method 분기 제거 - difficulty만 존재
difficulty_config = sampling_config.get("difficulty", {})
# 이하 기존 difficulty 로직...
```

---

### Phase 2: dataloader.py 수정

**목표**: use_pairwise 플래그 기반 collator 선택

#### Step 2.1: create_dataloader() 수정

**변경 전** (`dataloader.py:158-169`):
```python
# Collator 선택 (sampling_method에 따라 분기)
sampling_method = sampling_config.get("sampling_method")
if sampling_method == "pairwise":
    collator = PairwiseDataCollator(...)
else:
    collator = AlpacaDataCollator(...)
```

**변경 후**:
```python
# Collator 선택 (use_pairwise에 따라 분기)
use_pairwise = sampling_config.get("use_pairwise", False)
if use_pairwise:
    collator = PairwiseDataCollator(...)
else:
    collator = AlpacaDataCollator(...)
```

---

### Phase 3: run_critic.py 수정

**목표**: use_pairwise 플래그 기반 학습 분기

#### Step 3.1: sampling_method → use_pairwise 변수명 변경

**변경 전** (`run_critic.py:567-569`):
```python
sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
sampling_method = sampling_config.get("sampling_method")
logger.info(f"샘플링 방식: {sampling_method}")
```

**변경 후**:
```python
sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
use_pairwise = sampling_config.get("use_pairwise", False)
logger.info(f"Pairwise 모드: {use_pairwise}")
```

#### Step 3.2: is_pairwise 플래그 변경

**변경 전** (`run_critic.py:717`, 1277):
```python
is_pairwise = (sampling_method == "pairwise")
# ...
is_pairwise_final = (sampling_method == "pairwise")
```

**변경 후**:
```python
is_pairwise = use_pairwise
# ...
is_pairwise_final = use_pairwise
```

#### Step 3.3: validation sampling_config 구성 수정

**변경 전** (`run_critic.py:582-588`):
```python
val_sampling_config = sampling_config.copy()
if sampling_method == "difficulty":
    val_sampling_config["difficulty"] = val_sampling_config.get("difficulty", {}).copy()
    val_sampling_config["difficulty"]["n_samples"] = config.data_sampling.val_n_samples
elif sampling_method == "pairwise":
    val_sampling_config["pairwise"] = val_sampling_config.get("pairwise", {}).copy()
    val_sampling_config["pairwise"]["n_pairs"] = config.data_sampling.val_n_samples
```

**변경 후**:
```python
val_sampling_config = sampling_config.copy()
# difficulty 설정에 val_n_samples 적용 (pairwise 모드와 무관)
if "difficulty" in val_sampling_config:
    val_sampling_config["difficulty"] = val_sampling_config["difficulty"].copy()
    val_sampling_config["difficulty"]["n_samples"] = config.data_sampling.val_n_samples
else:
    val_sampling_config["n_samples"] = config.data_sampling.val_n_samples
```

---

### Phase 4: Config 파일 재구조화

#### Step 4.1: critic_pairwise.yaml 재작성

**변경 전**:
```yaml
data_sampling:
  sampling_method: "pairwise"
  seed: 42
  val_n_samples: 2000
  pairwise:
    n_pairs: 50000
```

**변경 후**:
```yaml
data_sampling:
  seed: 42
  val_n_samples: 2000
  use_pairwise: true

  n_samples: 100000  # difficulty 기반 총 샘플 수
  auto_data_balancing: true
  correct_ratio: 0.5

  difficulty_bins:
    low: [1, 6]
    medium: [7, 10]
    high: [11, 25]
  difficulty_weights:
    low: 0.3
    medium: 0.5
    high: 0.2
```

#### Step 4.2: critic_mlp.yaml 수정

**추가**:
```yaml
data_sampling:
  use_pairwise: false  # 명시적 설정
  # 기존 difficulty 설정 유지
```

---

### Phase 5: 테스트 코드 업데이트

#### Step 5.1: test_pairwise_sampling.py 수정

- `sampling_method: "pairwise"` → `use_pairwise: true` 변경
- `pairwise.n_pairs` → `difficulty.n_samples` + `use_pairwise: true`

#### Step 5.2: test_pairwise_collator.py

- 변경 없음 (collator 자체는 동일)

#### Step 5.3: test_pipeline_critic_pairwise.py 수정

- Config fixture에서 새 구조 적용

---

## 삭제 목록 (하위 호환성 코드 완전 제거)

### datasets.py
```python
# 삭제 1: sampling_method 분기 (line 75-86)
sampling_method = sampling_config.get("sampling_method")
if sampling_method == "pairwise":
    return _load_pairwise_dataset(...)

# 삭제 2: _load_pairwise_dataset() 전체 함수 (line 776-836)
def _load_pairwise_dataset(...): ...

# 삭제 3: sampling_method 검증 (line 251, 291-292)
sampling_method = sampling_config.get("sampling_method")
# ...
else:
    raise ValueError(f"잘못된 sampling_method: {sampling_method}")
```

### dataloader.py
```python
# 삭제: sampling_method 변수 (line 159)
sampling_method = sampling_config.get("sampling_method")
```

### run_critic.py
```python
# 삭제: sampling_method 변수 및 관련 분기 로직
sampling_method = sampling_config.get("sampling_method")
# ...
if sampling_method == "difficulty":
elif sampling_method == "pairwise":
```

### Config
```yaml
# 삭제: sampling_method 필드
sampling_method: "pairwise"  # 또는 "difficulty"

# 삭제: pairwise 독립 섹션
pairwise:
  n_pairs: 50000
```

---

## 구현 순서

1. **Phase 1**: datasets.py 핵심 로직 (의존성 없음)
2. **Phase 2**: dataloader.py (Phase 1 완료 후)
3. **Phase 3**: run_critic.py (Phase 1, 2 완료 후)
4. **Phase 4**: Config 파일 (Phase 1-3 완료 후)
5. **Phase 5**: 테스트 코드 (Phase 4 완료 후)

## 예상 영향도

- **Breaking Change**: Config 구조 변경으로 기존 `critic_pairwise.yaml` 호환 불가
- **데이터 분포 통일**: Pairwise/Pointwise 실험 간 동일한 difficulty 기반 데이터 분포 보장
- **코드 단순화**: sampling_method 분기 제거로 유지보수성 향상

---

## 검증 계획

1. **Unit Test**: `test_pairwise_sampling.py` - difficulty 샘플링 + pairwise 포맷 변환
2. **Unit Test**: `test_pairwise_collator.py` - collator 동작 (기존 유지)
3. **Integration Test**: `test_pipeline_critic_pairwise.py` - 전체 파이프라인
4. **Manual Test**: micro-mtp 모델로 실제 학습 수행

---

## 승인 요청

위 계획에 대해 검토 및 승인 부탁드립니다.

- [ ] Phase 1: datasets.py 핵심 로직 재구조화
- [ ] Phase 2: dataloader.py 수정
- [ ] Phase 3: run_critic.py 수정
- [ ] Phase 4: Config 파일 재구조화
- [ ] Phase 5: 테스트 코드 업데이트
