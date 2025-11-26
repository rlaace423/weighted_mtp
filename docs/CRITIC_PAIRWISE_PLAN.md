# Critic 일반화 성능 확보 계획서

Verifiable Critic의 Validation 성능 붕괴(Shortcut Learning) 현상을 해결하기 위한 수정 계획.

---

## 1. 문제 진단

### 1.1 현상

`critic_linear.yaml` 기반 실험 결과 (`mlruns_nipa/869427667672898361/4eeb155d0a244757a04f87a3bb1fe40c`):

| 지표 | Train | Validation | 비고 |
|------|-------|------------|------|
| mean_correct | 0.559 | 0.568 | - |
| mean_incorrect | 0.339 | 0.554 | - |
| **Gap** | **0.220** | **0.014** | Train에서만 분리 |
| F1 | 0.66 | 0.56 | Val이 랜덤 수준 |

### 1.2 원인 분석

```
Train unique instructions: 10,481개
Validation unique instructions: 95개
Overlap: 3개 (3.2%)
```

**핵심 문제**: 모델이 `Instruction ID → Score` 매핑을 암기 (Shortcut Learning)
- Train에서 본 instruction에 대해서는 correct/incorrect 구분 가능
- Validation의 새로운 instruction에서는 일반화 실패

### 1.3 데이터 특성

Train 데이터에서 **6,950개 instruction**이 correct와 incorrect를 모두 보유:
- 같은 문제에 대해 정답/오답 코드가 모두 존재
- 이 특성을 활용한 Pairwise 학습이 가능

---

## 2. 해결 전략

### 2.1 핵심 아이디어

**"같은 Instruction 내에서 코드의 상대적 품질을 비교하게 만든다"**

| 구분 | AS-IS (Pointwise) | TO-BE (Pairwise) |
|------|-------------------|------------------|
| 입력 | `(inst, output, is_correct)` | `(inst, correct_out, incorrect_out)` |
| 학습 대상 | 절대적 점수 예측 | 상대적 우열 판단 |
| Loss | `BCE(V(x), label)` | `-log(sigmoid(V_pos - V_neg))` |
| 문제점 | instruction 암기 | - |
| 효과 | - | 코드 품질 차이만 학습 |

### 2.2 변경 사항 요약

| 구분 | 변경 전 | 변경 후 |
|------|---------|---------|
| 모델 구조 | MLP (no dropout) | MLP + Dropout(0.3) |
| 학습 대상 | Backbone + Head | Backbone Freeze + Head Only |
| Loss 함수 | MSE/BCE (Absolute) | Pairwise Ranking (Relative) |
| 데이터 | Pointwise 샘플 | Triplet (Inst, Correct, Incorrect) |
| 평가 지표 | F1, Pred Gap | Pairwise Accuracy |

---

## 3. 수정 대상 파일

### 3.1 신규/수정

| 파일 | 변경 유형 | 변경 내용 |
|------|----------|----------|
| `models/meta_mtp/value_head.py` | 수정 | MLPValueHead에 dropout 추가 |
| `data/collators.py` | 추가 | `PairwiseDataCollator` 클래스 |
| `data/datasets.py` | 수정 | `_sample_pairwise()` 추가, `_sample_by_problem_id()` 삭제 |
| `pipelines/run_critic.py` | 수정 | Pairwise ranking loss 지원, problems 분기 삭제 |
| `configs/critic/critic_pairwise.yaml` | 신규 | Pairwise 학습용 config |

### 3.2 삭제 대상 (problems 방식 제거)

| 파일 | 삭제 내용 |
|------|----------|
| `data/datasets.py` | `_sample_by_problem_id()` 함수 전체 삭제 |
| `data/datasets.py` | `_compute_sampling_indices_from_metadata()`에서 `problems` 분기 삭제 |
| `pipelines/run_critic.py` | `sampling_method == "problems"` 관련 분기 삭제 |
| `utils/config_utils.py` | problems 관련 validation 로직 삭제 (있는 경우) |

**삭제 근거**: `problems` 방식은 같은 problem_id 샘플을 함께 로드하지만, 최종 셔플로 쌍이 분리되어 pairwise 학습에 부적합. `pairwise` 방식이 이를 완전히 대체.

---

## 4. 세부 구현 명세

### 4.1 value_head.py - Dropout 추가

**변경 위치**: `MLPValueHead.__init__()`

```python
class MLPValueHead(nn.Module):
    """2-layer MLP value head (DIAL style)

    구조: hidden_size -> hidden_size//8 -> hidden_size//16 -> 1
    Dropout으로 instruction ID 암기 방지
    """
    def __init__(self, hidden_size: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_type = "mlp"

        hidden1 = hidden_size // 8   # 512 for 4096 dim
        hidden2 = hidden_size // 16  # 256 for 4096 dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden1, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),  # 추가
            nn.Linear(hidden1, hidden2, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),  # 추가
            nn.Linear(hidden2, 1, bias=bias),
        )

        self._init_weights()
```

**하위 호환성**: `dropout=0.0` 기본값으로 기존 동작 유지

**Factory 함수 수정**:
```python
def create_value_head(hidden_size: int, head_type: str = "mlp", dropout: float = 0.0) -> ValueHeadType:
    if head_type == "mlp":
        return MLPValueHead(hidden_size, dropout=dropout)
    # ...
```

### 4.2 collators.py - PairwiseDataCollator 추가

```python
@dataclass
class PairwiseDataCollator:
    """Pairwise Ranking 학습용 Collator

    같은 instruction의 (correct, incorrect) 쌍을 배치로 구성.
    각 쌍에서 correct 코드의 value가 incorrect보다 높도록 학습.

    Args:
        tokenizer: HuggingFace PreTrainedTokenizer
        max_length: 최대 시퀀스 길이 (기본 2048)
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """배치를 토큰화

        Args:
            batch: 샘플 리스트, 각 샘플은 다음 키 포함:
                - instruction: 문제 설명
                - input: 입력 예시
                - correct_output: 정답 코드
                - incorrect_output: 오답 코드

        Returns:
            pos_input_ids: [batch, seq] - correct samples
            pos_attention_mask: [batch, seq]
            pos_labels: [batch, seq] - loss masking용
            neg_input_ids: [batch, seq] - incorrect samples
            neg_attention_mask: [batch, seq]
            neg_labels: [batch, seq] - loss masking용
        """
        pos_input_ids = []
        pos_attention_mask = []
        pos_labels = []
        neg_input_ids = []
        neg_attention_mask = []
        neg_labels = []

        for sample in batch:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            correct_output = sample["correct_output"]
            incorrect_output = sample["incorrect_output"]

            # Positive (correct) 토큰화
            pos_tokens = self._tokenize_sample(instruction, input_text, correct_output)
            pos_input_ids.append(pos_tokens["input_ids"])
            pos_attention_mask.append(pos_tokens["attention_mask"])
            pos_labels.append(pos_tokens["labels"])

            # Negative (incorrect) 토큰화
            neg_tokens = self._tokenize_sample(instruction, input_text, incorrect_output)
            neg_input_ids.append(neg_tokens["input_ids"])
            neg_attention_mask.append(neg_tokens["attention_mask"])
            neg_labels.append(neg_tokens["labels"])

        return {
            "pos_input_ids": torch.stack(pos_input_ids),
            "pos_attention_mask": torch.stack(pos_attention_mask),
            "pos_labels": torch.stack(pos_labels),
            "neg_input_ids": torch.stack(neg_input_ids),
            "neg_attention_mask": torch.stack(neg_attention_mask),
            "neg_labels": torch.stack(neg_labels),
        }

    def _tokenize_sample(self, instruction: str, input_text: str, output: str) -> dict:
        """단일 샘플 토큰화 (AlpacaDataCollator 로직 재사용)"""
        # ... (기존 AlpacaDataCollator.__call__ 내부 로직과 동일)
```

### 4.3 datasets.py - Pairwise 샘플링 추가

**sampling_method 분기에 추가**:

```python
def _compute_sampling_indices_from_metadata(...):
    # ...

    # 3. Pairwise 방식: 같은 problem_id의 correct/incorrect 쌍
    elif sampling_method == "pairwise":
        pairwise_config = sampling_config.get("pairwise", {})
        return _sample_pairwise(
            metadata=metadata,
            n_pairs=pairwise_config.get("n_pairs", 10000),
            seed=seed,
        )
```

**핵심 함수**:

```python
def _sample_pairwise(
    metadata: list[dict],
    n_pairs: int,
    seed: int,
) -> list[dict]:
    """같은 problem_id의 (correct, incorrect) 쌍 샘플링

    Args:
        metadata: 메타데이터 (problem_id, is_correct 필드 필수)
        n_pairs: 생성할 쌍의 수
        seed: 랜덤 시드

    Returns:
        [{"correct_idx": int, "incorrect_idx": int}, ...] 리스트
    """
    random.seed(seed)

    # problem_id별 correct/incorrect 인덱스 분리
    problem_indices = defaultdict(lambda: {"correct": [], "incorrect": []})

    for idx, meta in enumerate(metadata):
        pid = meta.get("problem_id")
        is_correct = meta.get("is_correct")

        if pid is None:
            continue

        if is_correct:
            problem_indices[pid]["correct"].append(idx)
        else:
            problem_indices[pid]["incorrect"].append(idx)

    # correct와 incorrect가 모두 있는 problem만 필터링
    valid_problems = [
        pid for pid, indices in problem_indices.items()
        if len(indices["correct"]) > 0 and len(indices["incorrect"]) > 0
    ]

    logger.info(f"Pairwise 샘플링 가능 문제: {len(valid_problems)}개")

    # 쌍 생성
    pairs = []
    for pid in valid_problems:
        correct_indices = problem_indices[pid]["correct"]
        incorrect_indices = problem_indices[pid]["incorrect"]

        # 모든 조합 생성
        for c_idx in correct_indices:
            for i_idx in incorrect_indices:
                pairs.append({"correct_idx": c_idx, "incorrect_idx": i_idx})

    logger.info(f"전체 가능한 쌍: {len(pairs)}개")

    # 랜덤 샘플링
    if len(pairs) > n_pairs:
        pairs = random.sample(pairs, n_pairs)

    random.shuffle(pairs)
    logger.info(f"Pairwise 샘플링 완료: {len(pairs)}개 쌍")

    return pairs
```

**JSONL 로딩 수정** (`_read_jsonl_by_indices` 대신 pair 기반 로딩):

```python
def _read_jsonl_pairwise(
    jsonl_path: Path,
    pairs: list[dict],
) -> list[dict]:
    """Pairwise 샘플 로딩

    Args:
        jsonl_path: JSONL 파일 경로
        pairs: [{"correct_idx": int, "incorrect_idx": int}, ...] 리스트

    Returns:
        [{"instruction", "input", "correct_output", "incorrect_output"}, ...] 리스트
    """
    # 필요한 인덱스 수집
    all_indices = set()
    for pair in pairs:
        all_indices.add(pair["correct_idx"])
        all_indices.add(pair["incorrect_idx"])

    # 인덱스별 샘플 로드
    idx_to_sample = {}
    sorted_indices = sorted(all_indices)
    # ... (기존 _read_jsonl_by_indices 로직 활용)

    # 쌍 구성
    result = []
    for pair in pairs:
        correct_sample = idx_to_sample[pair["correct_idx"]]
        incorrect_sample = idx_to_sample[pair["incorrect_idx"]]

        result.append({
            "instruction": correct_sample["instruction"],
            "input": correct_sample.get("input", ""),
            "correct_output": correct_sample["output"],
            "incorrect_output": incorrect_sample["output"],
        })

    return result
```

### 4.4 run_critic.py - Pairwise Loss 지원

**Loss 함수 추가**:

```python
def pairwise_ranking_loss(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry Pairwise Ranking Loss

    P(pos > neg) = sigmoid(V_pos - V_neg)
    Loss = -log(sigmoid(V_pos - V_neg))

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        mask_pos: [batch, seq] valid token mask for positive
        mask_neg: [batch, seq] valid token mask for negative

    Returns:
        Scalar loss
    """
    # 시퀀스 평균 value 계산
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    # Pairwise ranking loss
    return -torch.nn.functional.logsigmoid(v_pos_mean - v_neg_mean).mean()
```

**Pairwise Accuracy 메트릭**:

```python
def compute_pairwise_accuracy(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> float:
    """Pairwise Accuracy 계산

    V(correct) > V(incorrect)인 쌍의 비율
    """
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    correct_pairs = (v_pos_mean > v_neg_mean).float().sum()
    total_pairs = v_pos_mean.size(0)

    return (correct_pairs / total_pairs).item()
```

**Training loop 분기**:

```python
# config에서 loss_type 확인
loss_type = config.training.get("loss_type", "pointwise")

if loss_type == "pairwise":
    # Pairwise 학습
    pos_outputs = adapter(batch["pos_input_ids"], batch["pos_attention_mask"], ...)
    neg_outputs = adapter(batch["neg_input_ids"], batch["neg_attention_mask"], ...)

    loss = pairwise_ranking_loss(
        v_pos=pos_outputs["value_logits"],
        v_neg=neg_outputs["value_logits"],
        mask_pos=(batch["pos_labels"] != -100).float(),
        mask_neg=(batch["neg_labels"] != -100).float(),
    )
else:
    # 기존 Pointwise 학습
    # ...
```

### 4.5 critic_pairwise.yaml - Config 신규 생성

```yaml
# Critic Pre-training (Stage 1) - Pairwise Ranking
# Shortcut Learning 방지를 위한 상대적 품질 비교 학습

project:
  name: weighted-mtp
  version: "2.0.0"

experiment:
  name: critic-pretrain-pairwise
  description: "Pairwise ranking loss for generalization"
  stage: critic
  tags:
    - critic
    - pairwise-ranking
    - dropout

models:
  policy:
    name: meta-llama-mtp
    path: storage/models/meta-llama-mtp
    tokenizer_path: storage/models/meta-llama-mtp/tokenizer
    params:
      dim: 4096
      n_layers: 32
      n_heads: 32
      n_future_tokens: 4
      intermediate_size: 11008
      rope_theta: 10000.0
      vocab_size: 32000
    dtype: bfloat16

dataset:
  name: codecontests
  train: storage/datasets/codecontests/processed/train.jsonl
  validation: storage/datasets/codecontests/processed/valid.jsonl
  max_length: 2048

data_sampling:
  sampling_method: "pairwise"
  seed: 42
  val_n_samples: 1000

  pairwise:
    n_pairs: 50000  # 학습할 쌍의 수

training:
  loss_type: "pairwise"  # "pointwise" 또는 "pairwise"
  n_epochs: 1.0
  batch_size: 32  # 쌍 단위이므로 실제 샘플 수는 2배
  gradient_accumulation_steps: 2
  trunk_learning_rate: 0.0  # Backbone freeze
  value_head_learning_rate: 1.0e-3
  max_grad_norm: 1.0
  num_unfrozen_layers: 0  # Backbone 완전 freeze
  value_head_type: mlp
  dropout: 0.3  # Value head dropout
  log_interval: 10

  lr_scheduler:
    type: cosine
    warmup_ratio: 0.05
    min_lr_ratio: 0.01

checkpoint:
  save_dir: storage/checkpoints/critic/${experiment.name}
  save_checkpoint_every: 0.2
  save_best: true
  save_final: true
  save_total_limit: 2
  s3_upload: true

runtime:
  device: cuda
  seed: 42
  mixed_precision: true

distributed:
  fsdp:
    sharding_strategy: FULL_SHARD
    mixed_precision: true
    cpu_offload: false
    activation_checkpointing: true

storage:
  root: storage
  models_dir: storage/models
  datasets_dir: storage/datasets
  checkpoints_dir: storage/checkpoints

mlflow:
  tracking_uri: ""
  experiment: "weighted-mtp/production"

logging:
  level: INFO
  format: "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
```

---

## 5. 구현 순서

1. **Step 1**: `datasets.py`에서 `_sample_by_problem_id()` 및 problems 분기 삭제
2. **Step 2**: `run_critic.py`에서 problems 관련 분기 삭제
3. **Step 3**: `value_head.py` 수정 - dropout 인자 추가
4. **Step 4**: `collators.py`에 `PairwiseDataCollator` 추가
5. **Step 5**: `datasets.py`에 `_sample_pairwise()` 및 `_read_jsonl_pairwise()` 추가
6. **Step 6**: `run_critic.py`에 pairwise loss 및 메트릭 추가
7. **Step 7**: `critic_pairwise.yaml` config 생성
8. **Step 8**: 로컬 테스트 실행 및 검증

**원칙**: 삭제 먼저, 추가는 나중에 (기존 코드 정리 후 새 기능 구현)

---

## 6. 기대 효과

### 6.1 목표 지표

| 지표 | 현재 | 목표 |
|------|------|------|
| Val Pairwise Accuracy | ~50% (추정) | **>65%** |
| Train-Val Gap | 0.22 vs 0.01 | **<0.05 차이** |
| 일반화 | 실패 | 새 instruction에서 분별 |

### 6.2 이론적 근거

- **Bradley-Terry Model**: 쌍대 비교에서 승률을 모델링하는 표준 방법
- **Contrastive Learning**: 같은 context 내 positive/negative 비교는 더 robust한 표현 학습 유도
- **Dropout Regularization**: 특정 뉴런에 대한 의존성 감소로 암기 방지

---

## 7. 참고

- 실험 결과: `storage/mlruns_nipa/869427667672898361/4eeb155d0a244757a04f87a3bb1fe40c`
- 기존 config: `configs/critic/critic_linear.yaml`
- 연구 배경: `docs/RESEARCH.md`
