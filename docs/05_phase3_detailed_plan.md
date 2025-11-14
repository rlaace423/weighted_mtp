# Phase 3: 데이터 파이프라인 구현 가이드

## 문서 개요

본 문서는 **Phase 3: 데이터 파이프라인 구현**을 위한 실행 가이드입니다. 구체적인 코드보다는 **설계 의도, 구현 요구사항, 검증 기준**에 집중하여 구현자가 맥락을 이해하고 자율적으로 구현할 수 있도록 합니다.

**버전**: v2.0 (2025-11-14)
**선행 조건**: Phase 1 (storage 준비), Phase 2 (코드 스켈레톤) 완료
**목표**: Meta LLaMA MTP 학습을 위한 메모리 효율적 데이터 파이프라인 구축

---

## Part 1: 개요 및 맥락

### 1.1 Phase 3의 위치와 목적

Phase 3는 **데이터 → 모델** 연결의 핵심 구간입니다.

```
Phase 1 (storage)  →  Phase 2 (skeleton)  →  [Phase 3 (data)]  →  Phase 4 (model)
     준비된 JSONL          코드 구조              파이프라인         학습 실행
```

**핵심 질문**: 어떻게 3.7M 개의 샘플을 효율적으로 학습에 활용할 것인가?

### 1.2 핵심 혁신: Stage별 차별화 샘플링

**문제 인식**:
- 전체 데이터셋(3.7M) 로딩: 메모리 ~15GB, 학습 시간 수십 시간
- Stage 1 (Value Head Pretrain): correct/incorrect 구분 학습만 필요
- Stage 2 (Weighted Training): 쉬운 문제부터 학습하는 것이 TD error 안정화에 유리

**해결책**: Stage별로 목적에 맞는 데이터만 선별적 로딩

| Stage | 목적 | 필요 데이터 | 샘플 크기 | 메모리 |
|-------|------|-------------|-----------|--------|
| **전체 로딩** | - | 전체 | 3.7M | ~15GB |
| **Stage 1** | Value head가 correct/incorrect 구분 학습 | is_correct 균형 (50:50) | 10-50K | ~200MB |
| **Stage 2** | 쉬운 문제부터 학습하여 TD error 안정화 | Difficulty 기반 Curriculum | 100-500K | ~800MB |

**효율 개선**: 메모리 **18.75~75배**, 학습 시간 **94~98% 단축**

### 1.3 기대 효과

1. **메모리 효율**: GPU 메모리를 모델과 gradient에 집중
2. **학습 안정성**: Curriculum Learning으로 TD error 폭주 방지
3. **실험 속도**: 빠른 iteration으로 hyperparameter 탐색 가능
4. **재현성**: seed 고정으로 실험 재현 보장

---

## Part 2: 데이터 구조 이해

### 2.1 실제 데이터 검증 결과

Phase 3 착수 전, 실제 데이터 구조를 정밀 분석했습니다. (2025-11-14 검증 완료)

#### CodeContests (학습용)
```json
{
  "instruction": "문제 설명",
  "input": "테스트 케이스 예시",
  "output": "Python 솔루션 코드",
  "task_id": "problem_correct_0",
  "is_correct": true,  // ✅ 존재 확인
  "metadata": {
    "source": "code_contests",
    "difficulty": 7,   // ✅ 존재 확인 (1-11, 낮을수록 쉬움)
    "has_tests": true
  }
}
```

**핵심 발견**:
- ✅ `is_correct` 필드 존재 → Stage 1 균형 샘플링 가능
- ✅ `difficulty` 필드 존재 → Stage 2 Curriculum Learning 가능
- ✅ 실제 분포: diff=7 (86.7%), diff=2 (6.4%), diff=1 (4.4%), diff=11 (2.1%), diff=6 (0.4%)

**샘플 수**:
- Train: 3,691,981 (correct: 1,754,404 / incorrect: 1,937,577)
- Valid: 14,725 (correct: 8,184 / incorrect: 6,541)
- Test: 14,851 (correct: 8,038 / incorrect: 6,813)

#### MBPP & HumanEval (평가용)
```json
{
  "instruction": "...",
  "output": "...",
  "task_id": "...",
  "metadata": {
    "source": "mbpp",
    "test_list": ["assert ..."],  // MBPP만
    "test": "...",                 // HumanEval만
    "has_tests": true
  }
}
```

**핵심 차이**:
- ❌ `is_correct` 없음 (correct 솔루션만 포함)
- ❌ `difficulty` 없음
- ✅ Test cases로 평가 가능
- **용도**: 평가 전용 (학습 사용 불가)

### 2.2 데이터 스키마 요약

| 필드 | CodeContests | MBPP | HumanEval | 용도 |
|------|--------------|------|-----------|------|
| `is_correct` | ✅ boolean | ❌ | ❌ | Stage 1 균형 샘플링 |
| `difficulty` | ✅ int (1-11) | ❌ | ❌ | Stage 2 Curriculum |
| `test_list` | ❌ | ✅ array | ❌ | 평가 |
| `test` | ❌ | ❌ | ✅ string | 평가 |

**결론**: **CodeContests만 학습용, MBPP/HumanEval은 평가 전용**

---

## Part 3: 핵심 설계 결정

### 3.1 Decision 1: Stage별 샘플링 전략

**문제**: 전체 데이터는 불필요하고 비효율적

**해결책**: Stage마다 목적에 맞는 데이터만 샘플링

#### Stage 1: is_correct 균형 샘플링

**Rationale**:
- Value head는 "correct와 incorrect를 구분"하는 법을 학습해야 함
- 한쪽으로 편향되면 학습 실패 (예: 모든 샘플을 correct로 예측)
- 균형잡힌 데이터로 binary classification 능력 확보

**요구사항**:
- correct : incorrect = 50 : 50 (±10% 허용)
- 전체 난이도 균등 샘플링 (난이도 편향 방지)
- 샘플 크기: 10,000 ~ 50,000
- 재현성: seed=42 고정

**기대 효과**:
- Value head가 빠르게 수렴 (10K면 충분)
- 메모리 200MB 이하

#### Stage 2: Difficulty 기반 Curriculum Learning

**Rationale**:
- TD error는 난이도 높은 문제에서 불안정 (value 예측 어려움)
- 쉬운 문제부터 학습하면 value function이 점진적으로 개선
- 어려운 문제는 value function이 안정화된 후 학습

**Curriculum 전략**:

| Epoch 구간 | Low (1-3) | Medium (4-7) | High (8-11) | 목적 |
|-------------|-----------|--------------|-------------|------|
| 초반 (0-30%) | 70% | 30% | 0% | 기초 학습, TD error 안정화 |
| 중반 (30-70%) | 30% | 60% | 10% | 점진적 난이도 증가 |
| 후반 (70-100%) | 10% | 50% | 40% | 고난이도 문제 집중 |

**요구사항**:
- 난이도 구간(bins) 정의: `{"low": [1,3], "medium": [4,7], "high": [8,11]}`
- Epoch 진행에 따라 가중치 동적 변경
- 샘플 크기: 100,000 ~ 500,000
- is_correct 혼합: TD error weighting이 자동 필터링 (incorrect → 낮은 weight)

**기대 효과**:
- TD error 분산 감소
- 수렴 속도 향상
- 메모리 800MB 이하

### 3.2 Decision 2: HuggingFace Dataset 사용

**문제**: 3.7M JSONL을 어떻게 효율적으로 로드할 것인가?

**대안 비교**:

| 방식 | 장점 | 단점 |
|------|------|------|
| 직접 JSONL 읽기 | 단순, 제어 용이 | 메모리 비효율, 캐싱 없음 |
| Pandas DataFrame | 분석 편리 | 메모리 과다, PyTorch 통합 어려움 |
| **HuggingFace Dataset** | 캐싱, 메모리 효율, PyTorch 통합 | 초기 학습 곡선 |

**Decision**: HuggingFace Dataset

**Rationale**:
1. **자동 캐싱**: 한 번 로드하면 디스크에 캐시 (재실행 시 빠름)
2. **메모리 효율**: 전체를 메모리에 올리지 않음
3. **PyTorch 통합**: DataLoader와 자연스럽게 연결
4. **Filter/Map 지원**: 샘플링, 전처리 파이프라인 구성 용이

### 3.3 Decision 3: Loss Masking 전략

**문제**: Alpaca 형식에서 무엇을 학습 대상으로 할 것인가?

**결정**: **Instruction/Input은 제외, Output만 학습**

**Rationale**:
1. **학습 목표 명확화**: 모델이 "문제를 해결하는 코드"만 생성하도록
2. **Gradient 집중**: Instruction/Input 복원에 gradient 낭비 방지
3. **표준 SFT 관행**: HuggingFace TRL, Alpaca 등 표준 방식

**구현 요구사항**:
- Instruction 토큰: `labels = -100` (attention은 유지, loss만 제외)
- Input 토큰: `labels = -100`
- Output 토큰: `labels = token_ids` (실제 학습 대상)
- Padding 토큰: `labels = -100`
- PyTorch CrossEntropyLoss는 -100을 자동 무시

---

## Part 4: 구현 가이드

### 4.1 Step 1: 데이터 로딩 및 샘플링 (`datasets.py`)

#### 목표
JSONL 파일을 HuggingFace Dataset으로 로드하고, Stage별 샘플링 전략을 적용합니다.

#### 핵심 기능

**1. load_dataset() 함수**

```python
def load_dataset(
    dataset_name: Literal["codecontests", "mbpp", "humaneval"],
    split: Optional[str] = None,
    # Stage별 샘플링 파라미터
    stage: Optional[Literal["stage1", "stage2"]] = None,
    n_samples: Optional[int] = None,
    balance_correct: bool = False,
    difficulty_weights: Optional[dict] = None,
    seed: int = 42,
) -> Dataset | DatasetDict:
    """JSONL → HuggingFace Dataset 로딩 + Stage별 샘플링"""
```

**책임**:
- JSONL 파일 경로 해석 (codecontests → `storage/datasets_v2/codecontests/processed/`)
- HuggingFace `load_dataset("json", ...)` 호출
- Stage 파라미터에 따라 샘플링 적용
- 재현성을 위한 seed 고정

**2. apply_stage_sampling() 함수**

```python
def apply_stage_sampling(
    dataset: Dataset,
    stage: Literal["stage1", "stage2"],
    n_samples: int,
    **sampling_config
) -> Dataset:
    """Stage별 샘플링 로직 적용"""
```

**Stage 1 로직**:
1. `is_correct==True` 샘플 인덱스 추출
2. `is_correct==False` 샘플 인덱스 추출
3. 각각에서 `n_samples * correct_ratio` 만큼 랜덤 샘플링
4. 병합 후 섞기 (shuffle)
5. `dataset.select(indices)` 반환

**Stage 2 로직**:
1. difficulty_bins와 difficulty_weights 파싱
2. 각 bin별로 해당 난이도 샘플 인덱스 추출
   예: low [1-3] → `sample["metadata"]["difficulty"] in [1,2,3]`
3. 각 bin에서 `n_samples * weight` 만큼 랜덤 샘플링
4. 병합 후 섞기
5. 정확히 n_samples 개수 맞추기

#### 요구사항

| 항목 | Stage 1 | Stage 2 |
|------|---------|---------|
| 샘플 크기 | 10,000 ~ 50,000 | 100,000 ~ 500,000 |
| 필터링 기준 | is_correct 균형 (50:50 ±10%) | difficulty 가중치 (±15%) |
| 난이도 처리 | 전체 균등 | Curriculum (초반 70% low → 후반 40% high) |
| 재현성 | seed=42 고정 | seed=42 고정 |
| 적용 대상 | CodeContests만 | CodeContests만 |

#### 검증 기준

**기능 검증**:
- [ ] load_dataset("codecontests", split="train") 성공
- [ ] is_correct 필드가 boolean으로 파싱됨
- [ ] difficulty 필드가 integer (1-11)로 파싱됨
- [ ] Stage 1: correct 샘플 비율 40-60%
- [ ] Stage 2: difficulty 분포가 가중치 ±15% 이내
- [ ] seed 고정 시 동일한 샘플 선택

**성능 검증**:
- [ ] Stage 1 (50K): 메모리 <300MB
- [ ] Stage 2 (200K): 메모리 <1GB
- [ ] 로딩 속도: >100 samples/sec

### 4.2 Step 2: Loss Masking Collator (`collators.py`)

#### 목표
Alpaca 형식 데이터를 토큰화하고, Instruction/Input은 loss 계산에서 제외합니다.

#### 핵심 기능

**AlpacaDataCollator 클래스**

```python
@dataclass
class AlpacaDataCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    n_future_tokens: int = 4

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """배치를 토큰화하고 loss masking 적용"""
```

**Masking 로직**:

```
텍스트 구조:
|<BOS>|<instruction>              |<input>     |<output>        |<PAD>|
  1     문제 설명 토큰들...        예시 토큰     솔루션 토큰들...   0

labels:
|-100 |-100 -100 -100 ... -100    |-100 -100   |tok tok tok ... |-100|
       ↑ instruction 제외           ↑ input 제외  ↑ output만 학습  ↑ pad 제외
```

**구현 전략**:
1. Instruction 텍스트만 별도 토큰화 → 길이 `len_inst` 계산
2. Input 텍스트만 별도 토큰화 → 길이 `len_input` 계산
3. 전체 `instruction + input + output` 토큰화 → `input_ids`
4. `labels = input_ids.clone()`
5. `labels[0 : 1+len_inst+len_input] = -100` (BOS + inst + input 마스킹)
6. `labels[attention_mask == 0] = -100` (padding 마스킹)

#### 요구사항

| 항목 | 값 |
|------|-----|
| Max length | 2048 (Phase 1에서 필터링 완료) |
| Padding | "max_length" 또는 "longest" |
| BOS/EOS | Tokenizer 자동 처리 |
| Masking value | -100 |
| MTP 지원 | n_future_tokens=4 |

#### 검증 기준

**Masking 경계 검증**:
- [ ] BOS 토큰: labels = -100
- [ ] Instruction 영역: labels = -100
- [ ] Input 영역: labels = -100
- [ ] Output 영역: labels = token_ids (not -100)
- [ ] Padding 영역: labels = -100
- [ ] attention_mask: 모든 토큰 1 (padding 제외)

**수치 검증**:
```python
# 단일 샘플 테스트
sample = {"instruction": "Add two numbers.", "input": "", "output": "def add(a,b): return a+b"}
batch = collator([sample])
labels = batch["labels"][0]

# Instruction 부분은 -100
assert (labels[:10] == -100).all()

# Output 부분은 token ID
assert (labels[-50:][labels[-50:] != -100]).numel() > 0
```

### 4.3 Step 3: 통합 및 검증

#### 통합 파이프라인

```python
# Stage 1 예시
tokenizer = LlamaTokenizer.from_pretrained("storage/models_v2/meta-llama-mtp/tokenizer")
dataset = load_dataset(
    "codecontests",
    split="train",
    stage="stage1",
    n_samples=50000,
    balance_correct=True,
    seed=42
)
collator = AlpacaDataCollator(tokenizer, max_length=2048)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

# 배치 생성 테스트
batch = next(iter(dataloader))
assert batch["input_ids"].shape == (8, 2048)
assert batch["labels"].shape == (8, 2048)
```

#### End-to-End 검증

**시나리오 1: Stage 1 파이프라인**
1. load_dataset() → 50K 샘플, is_correct 균형
2. AlpacaDataCollator → masking 적용
3. DataLoader → 배치 생성 (batch_size=8)
4. 검증: correct 비율, masking 경계, shape

**시나리오 2: Stage 2 파이프라인**
1. load_dataset() → 200K 샘플, difficulty 가중치
2. AlpacaDataCollator → masking 적용
3. DataLoader → 배치 생성
4. 검증: difficulty 분포, masking 경계

**시나리오 3: Epoch 루프**
```python
for epoch in range(3):
    for batch in dataloader:
        assert batch["input_ids"].shape[0] <= 8
        assert (batch["labels"] != -100).any()  # output 영역 존재
```

---

## Part 5: 검증 및 위험 관리

### 5.1 3-Tier 검증 체계

#### Tier 1: 기능 검증 (Functional Validation)

**데이터 로딩**:
- [ ] codecontests/mbpp/humaneval 모두 로딩 성공
- [ ] train/valid/test split 접근 가능
- [ ] is_correct (CodeContests): boolean 파싱
- [ ] difficulty (CodeContests): integer (1-11) 파싱
- [ ] test_list (MBPP): array 파싱
- [ ] test (HumanEval): string 파싱

**Stage별 샘플링**:
- [ ] Stage 1: 샘플 수 = n_samples ±1%
- [ ] Stage 1: correct 비율 = 50% ±10%
- [ ] Stage 2: 샘플 수 = n_samples ±1%
- [ ] Stage 2: difficulty 분포 = 가중치 ±15%
- [ ] seed 고정 시 재현성 100%

**Loss Masking**:
- [ ] BOS: labels = -100
- [ ] Instruction: labels = -100
- [ ] Input: labels = -100
- [ ] Output: labels = token_ids
- [ ] Padding: labels = -100
- [ ] attention_mask: 전체 context 포함

#### Tier 2: 품질 검증 (Quality Validation)

**성능 목표**:

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| Stage 1 메모리 | <300MB | `torch.cuda.memory_allocated()` |
| Stage 2 메모리 | <1GB | `torch.cuda.memory_allocated()` |
| 로딩 속도 | >100 samples/sec | `time.time()` |
| DataLoader throughput | >50 batches/sec | Epoch 시간 측정 |

**코드 품질**:
- [ ] Ruff linting 통과
- [ ] Black formatting 통과
- [ ] Type hints 100% (mypy 경고 0)
- [ ] Docstring 100% (Args, Returns, Examples)

**테스트 커버리지**:
- [ ] Unit tests: datasets.py >80%
- [ ] Unit tests: collators.py >80%
- [ ] Integration tests: 전체 파이프라인 >70%

#### Tier 3: 통합 검증 (Integration Validation)

**Stage 1 End-to-End**:
```bash
pytest tests/integration/test_stage1_pipeline.py -v
```
- DataLoader → Collator → Batch 검증
- is_correct 분포 확인
- Masking 경계 확인
- 3 epoch 동안 정상 동작

**Stage 2 End-to-End**:
```bash
pytest tests/integration/test_stage2_pipeline.py -v
```
- Curriculum schedule 적용 (초반/중반/후반)
- difficulty 분포 변화 확인
- TD error 계산 연동 (Phase 4에서 검증)

### 5.2 위험 관리 매트릭스

#### 고위험 (High Impact, High Probability)

**Risk 1: difficulty 필드 파싱 오류**
- **영향**: Curriculum Learning 완전 실패
- **확률**: Low (이미 검증 완료)
- **완화 전략**:
  - schema.json 검증 완료 (difficulty: integer 1-11)
  - 1000 샘플 분포 분석 완료 (diff=7: 86.7%)
  - Unit test에서 difficulty 파싱 검증
- **대비책**: difficulty 없으면 랜덤 샘플링으로 fallback

**Risk 2: Masking 경계 계산 오류**
- **영향**: 학습 실패 (instruction 학습 또는 output 제외)
- **확률**: Medium
- **완화 전략**:
  - 단위 테스트로 여러 샘플 수동 검증
  - Instruction/Input 길이 별도 계산 후 적용
  - 경계 전후 토큰 로깅하여 육안 확인
- **대비책**: 문제 발견 시 즉시 수정, 학습 재시작

#### 중위험 (Medium Impact, Medium Probability)

**Risk 3: 메모리 부족 (OOM)**
- **영향**: 학습 중단
- **확률**: Low (샘플 크기 제한됨)
- **완화 전략**:
  - Stage 1: 50K max (메모리 <300MB)
  - Stage 2: 500K max (메모리 <1GB)
  - batch_size 동적 조정
- **대비책**: n_samples 감소 또는 streaming 모드

**Risk 4: 샘플링 분포 편향**
- **영향**: 성능 소폭 저하
- **확률**: Medium
- **완화 전략**:
  - seed 고정으로 재현성 보장
  - 실제 분포 로깅 및 모니터링
  - 가중치 ±15% 오차 허용
- **대비책**: 가중치 조정 재실험

#### 저위험 (Low Impact)

**Risk 5: HuggingFace Dataset 캐싱 이슈**
- **영향**: 로딩 시간 증가
- **확률**: Low
- **완화**: 캐시 디렉터리 명시, 권한 확인
- **대비책**: 캐시 삭제 후 재로딩

**Risk 6: Tokenizer 불일치**
- **영향**: 토큰 ID 오류
- **확률**: Very Low (Meta tokenizer 고정)
- **완화**: tokenizer.model 경로 하드코딩
- **대비책**: 토큰화 결과 수동 검증

### 5.3 문제 해결 가이드

**증상**: Stage 1에서 correct 비율이 50%가 아님
- **원인**: is_correct 필드 파싱 오류 또는 샘플 부족
- **해결**: `print(dataset[0])` 확인, is_correct 타입 검증

**증상**: Stage 2에서 difficulty 분포가 가중치와 다름
- **원인**: difficulty bins 정의 오류 또는 샘플 부족
- **해결**: `Counter([s["metadata"]["difficulty"] for s in dataset])` 확인

**증상**: Masking 후 loss가 학습되지 않음
- **원인**: Output 영역도 -100으로 마스킹됨
- **해결**: `(batch["labels"] != -100).sum()` 확인, >0이어야 함

**증상**: DataLoader에서 OOM
- **원인**: batch_size 또는 max_length 과다
- **해결**: batch_size 감소, max_length 확인 (2048 이하)

---

## Part 6: 완료 기준 및 다음 단계

### 6.1 Phase 3 완료 체크리스트

#### 코드 완성
- [ ] `src/weighted_mtp/data/datasets.py` 구현
  - load_dataset() 함수
  - apply_stage_sampling() 함수
  - get_dataset_config() 함수
- [ ] `src/weighted_mtp/data/collators.py` 구현
  - AlpacaDataCollator 클래스
- [ ] `src/weighted_mtp/data/transforms.py` 구현 (선택)
  - Padding, truncation utilities

#### 테스트 완성
- [ ] `tests/unit/test_datasets.py`
  - test_load_single_split()
  - test_stage1_sampling()
  - test_stage2_sampling()
  - test_difficulty_field()
- [ ] `tests/unit/test_collators.py`
  - test_alpaca_collator_masking()
  - test_masking_boundaries()
- [ ] `tests/integration/test_data_pipeline.py`
  - test_stage1_end_to_end()
  - test_stage2_end_to_end()

#### 검증 완료
- [ ] Tier 1 (기능): 모든 체크리스트 통과
- [ ] Tier 2 (품질): 성능 목표 달성
- [ ] Tier 3 (통합): End-to-end 테스트 통과
- [ ] 3 epoch 정상 동작 확인

#### 문서화
- [ ] Docstring 100% (Args, Returns, Examples)
- [ ] `src/weighted_mtp/data/__init__.py` public API export
- [ ] Phase 3 완료 보고서 작성

### 6.2 Phase 4 착수 조건

Phase 3 완료 후, 다음 조건을 만족해야 Phase 4 (Meta Adapter 통합)로 진행:

✅ **필수 조건**:
1. DataLoader가 올바른 형식의 배치 생성 (`input_ids`, `attention_mask`, `labels`)
2. Loss masking이 정확히 작동 (unit test 검증)
3. Stage 1/2 샘플링이 요구사항 충족 (분포 검증)
4. `vendor/meta_llama/` 모듈 import 가능
5. `storage/models_v2/meta-llama-mtp/` 모델 자산 준비됨

✅ **권장 조건**:
1. Integration test 100% 통과
2. 메모리 사용량 목표 달성 (<1GB for Stage 2)
3. Code quality 기준 충족 (linting, formatting, type hints)

### 6.3 예상 소요 시간

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| datasets.py 구현 | 4-6시간 | Stage 샘플링 로직 포함 |
| collators.py 구현 | 3-4시간 | Masking 로직 |
| Unit tests 작성 | 3-4시간 | datasets + collators |
| Integration tests | 2-3시간 | End-to-end |
| 검증 및 디버깅 | 2-3시간 | 3-tier 검증 |
| 문서화 | 1-2시간 | Docstring, 보고서 |
| **합계** | **15-22시간** | 약 2-3일 |

---

## 부록

### A. 용어 정리

| 용어 | 정의 |
|------|------|
| **Curriculum Learning** | 쉬운 문제부터 어려운 문제로 점진적으로 학습 난이도를 증가시키는 전략 |
| **Loss Masking** | 특정 토큰의 labels를 -100으로 설정하여 loss 계산에서 제외하는 기법 |
| **Stage별 샘플링** | 학습 단계(Stage)마다 목적에 맞는 데이터를 선별적으로 로딩하는 전략 |
| **is_correct 균형** | Correct와 Incorrect 샘플을 50:50 비율로 샘플링 |
| **difficulty bins** | 난이도 값을 구간(low/medium/high)으로 그룹화 |

### B. 참고 자료

**내부 문서**:
- `docs/00_ideal_structure.md`: 전체 아키텍처, Stage별 샘플링 전략
- `docs/02_implementation_plan.md`: Phase 3 요구사항
- `storage/datasets_v2/*/schema.json`: 데이터 스키마 정의

**외부 레퍼런스**:
- [HuggingFace Datasets](https://huggingface.co/docs/datasets): Dataset API 문서
- [Alpaca Training](https://github.com/tatsu-lab/stanford_alpaca): Loss masking 참고
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html): DataLoader 사용법

### C. Stage별 샘플링 설정 예시

**config/defaults.yaml**:
```yaml
data:
  sampling:
    stage1:
      n_samples: 50000
      balance_correct: true
      correct_ratio: 0.5
      seed: 42
    stage2:
      n_samples: 200000
      difficulty_bins:
        low: [1, 3]
        medium: [4, 7]
        high: [8, 11]
      curriculum_schedule:
        - epoch_range: [0.0, 0.3]
          weights: {low: 0.7, medium: 0.3, high: 0.0}
        - epoch_range: [0.3, 0.7]
          weights: {low: 0.3, medium: 0.6, high: 0.1}
        - epoch_range: [0.7, 1.0]
          weights: {low: 0.1, medium: 0.5, high: 0.4}
      seed: 42
```

---

**문서 종료**

이 가이드는 Phase 3 구현을 위한 **방향성과 요구사항**을 제공합니다. 구체적 구현 디테일은 구현자의 판단에 맡기되, 핵심 설계 결정과 검증 기준은 반드시 준수해야 합니다.

Phase 3 완료 시, 이 문서와 실제 구현의 차이점을 `docs/phase3_completion_report.md`에 기록하여 다음 Phase의 참고 자료로 활용합니다.
