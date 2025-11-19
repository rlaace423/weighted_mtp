# 아키텍처

Weighted MTP의 핵심 아키텍처 및 구현 결정사항.

---

## Pure PyTorch 구현

Meta LLaMA MTP 아키텍처를 Pure PyTorch로 재구현하여 FSDP 완전 호환 및 학습 가능성 확보.

### 핵심 특징

- **fairscale 제거**: `nn.Embedding`, `nn.Linear` 사용
- **Gradient 계산 가능**: `@torch.inference_mode()` 제거
- **Device-agnostic**: cuda/mps/cpu 자동 지원
- **FSDP 호환**: Safetensors 저장/로딩 지원
- **Flash Attention**: Training 시 2-4배 속도 향상

### Flash Attention 구현

```python
# transformer.py: Attention.forward()
# Training 시 Flash Attention (start_pos=0, seqlen>1)
if start_pos == 0 and seqlen > 1:
    output = F.scaled_dot_product_attention(
        xq, keys, values,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,  # Causal masking 자동 적용
    )
# Inference 시 KV cache 방식 유지
else:
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)
```

**효과**:
- Training: 2-4배 속도 향상, 메모리 30-40% 절감
- Inference: 기존 KV cache 방식 유지 (호환성)
- PyTorch 2.0+ 네이티브 지원 (별도 라이브러리 불필요)

### RoPE freqs_cis 처리

```python
# complex64 타입(safetensors 미지원)이므로 state_dict에서 제외
self.freqs_cis = precompute_freqs_cis(...)  # 일반 속성

def forward(self, tokens):
    # Runtime에 명시적 device 이동
    freqs_cis = self.freqs_cis[0:seqlen].to(tokens.device)
```

**효과**: Safetensors 저장/로딩, FSDP checkpoint, State dict 크기 감소

---

## 파이프라인 비교

| Pipeline | Value Head | 데이터 샘플링 | Weight 메커니즘 | LOC |
|----------|-----------|--------------|----------------|-----|
| **Baseline** | ❌ | 정답만 (150K) | Uniform (1.0) | 583 |
| **Critic** | ✅ 학습 대상 | 50:50 균형 (50K) | N/A (Value loss만) | 703 |
| **Verifiable** | ✅ Continual | Curriculum (150K, 50:50) | `exp(td_error/β)` | 810 |
| **Rho-1** | ❌ | 정답만 (150K) | Top-k binary | 662 |

### 공통 흐름

```python
# 1. Config 로딩 (deep merge)
config = OmegaConf.merge(defaults, experiment_config)

# 2. 분산 환경 초기화 (VESSL)
if is_distributed:
    rank, world_size = init_distributed()
    device = f"cuda:{rank}"

# 3. 모델 로딩
model = MetaLlamaMTPAdapter.from_pretrained(
    model_path,
    initialize_value_head=(stage in ["critic", "verifiable"])
)

# 4. 메타데이터 기반 데이터 로딩
dataset = load_dataset_with_metadata(...)

# 5. Training Loop
for batch in dataloader:
    outputs = model.full_forward(batch)
    loss = compute_loss(outputs, weights)
    loss.backward()
```

---

## 메타데이터 기반 로딩

99% 메모리 절감을 위한 핵심 혁신.

### 워크플로우

```python
# 1. 메타데이터만 로드 (~217MB)
metadata = json.load(open(f"{dataset}_metadata.json"))

# 2. Config 기반 샘플링 인덱스 계산
if balance_correct:  # Critic, Verifiable
    indices = balanced_sample(metadata, correct_ratio=0.5)
elif correct_ratio == 1.0:  # Baseline, Rho-1
    indices = filter_correct_samples(metadata)

# 3. JSONL에서 해당 라인만 선택적 읽기
samples = [jsonl_lines[idx] for idx in indices]
```

### 메모리 효과

- **기존**: 전체 로드 (~15GB)
- **개선**: 메타데이터(~217MB) + 필요 샘플만
  - Baseline (150K): ~0.8GB (95% 절감)
  - Verifiable (150K): ~0.8GB (95% 절감)
  - Critic (50K): ~0.4GB (97% 절감)

---

## 분산학습 구조

VESSL A100 4-GPU 환경에서 FSDP (FullyShardedDataParallel) 기반 분산학습.

### Torchrun 설정

```bash
# 4-GPU 분산학습
PYTHONPATH=src torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable.yaml
```

### 자동 환경변수

- `RANK`: 전체 프로세스 순위 (0-3)
- `WORLD_SIZE`: 전체 프로세스 개수 (4)
- `MASTER_ADDR`, `MASTER_PORT`: 통신 설정

### FSDP Wrapping

```python
# runtime/fsdp.py
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = wrap_model_fsdp(
    model,
    device,
    sharding_strategy=config.distributed.fsdp.sharding_strategy,
    mixed_precision=config.distributed.fsdp.mixed_precision,
)
# → FSDP wrapper 적용 (파이프라인별 sharding 전략)
```

### Sharding Strategies

**NO_SHARD** (Critic):
- 모델 복제 (DDP 동일)
- Value Head만 학습 → 메모리 충분
- 통신 오버헤드 최소

**FULL_SHARD** (Verifiable/Baseline/Rho-1):
- 모델/Gradient/Optimizer 샤딩
- 6.7B 전체 학습 → 메모리 75% 절감 (90GB → 30GB)
- All-gather (forward) + Reduce-scatter (backward)

### Mixed Precision (BFloat16)

```python
# FSDP Mixed Precision 설정 (모델 dtype 자동 감지)
model_dtype = next(model.parameters()).dtype  # torch.bfloat16
mp_policy = MixedPrecision(
    param_dtype=model_dtype,
    reduce_dtype=model_dtype,
    buffer_dtype=model_dtype,
)
```

**특징**:
- BFloat16 사용 (A100 네이티브 지원)
- Float16 대비 wider dynamic range (8 exp bits)
- Gradient explosion 방지 (loss scaling 불필요)
- Config에서 dtype 지정 → 모델 로딩 시 자동 적용

### 성능 최적화 효과 종합

| 최적화 | 메모리 효과 | 속도 효과 | 적용 대상 |
|--------|-----------|----------|----------|
| **FSDP FULL_SHARD** | 75% 절감 (90GB→30GB) | 6-10% 느림 | Verifiable/Baseline/Rho-1 |
| **FSDP NO_SHARD** | DDP 동일 | DDP 동일 | Critic |
| **BFloat16** | 동일 (vs Float16) | A100 최적화 | 모든 파이프라인 |
| **Flash Attention** | 30-40% 절감 | 2-4배 빠름 | Training (start_pos=0) |

**시너지 효과** (FULL_SHARD + BFloat16 + Flash Attention):
- 메모리: 90GB → 18-21GB (77% 절감)
- 속도: Flash Attention 2-4배 > FSDP 6-10% 오버헤드
- Batch size: 4 → 8-12 증가 가능

### 데이터 분산

**Rank-aware Sampling** (DistributedSampler 대신):

```python
# datasets.py: 메타데이터 기반 Rank-aware 샘플링
all_indices = _compute_sampling_indices_from_metadata(...)  # 모든 rank 동일

if world_size > 1:
    rank_indices = all_indices[rank::world_size]  # Rank별 분할

# 각 GPU가 전체의 1/4 처리 (중복 없음)
# Rank 0: samples[0::4]
# Rank 1: samples[1::4]
# ...
```

**장점**:
- 75% 메모리 절약 (각 rank가 1/4만 로드)
- 메타데이터 기반 커리큘럼 학습 지원
- 재현성 보장 (모든 rank가 동일 인덱스 계산)

### Rank 0 책임

- MLflow 로깅
- Checkpoint 저장 (FSDP Full state dict gathering)
- S3 비동기 업로드

### FSDP Checkpoint 저장

```python
# utils/checkpoint_utils.py
if isinstance(adapter, FSDP):
    with FSDP.state_dict_type(
        adapter,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        adapter_state_dict = adapter.state_dict()
else:
    adapter_state_dict = adapter.state_dict()
```

**특징**:
- FSDP는 명시적 Full state dict gathering 필요
- `rank0_only=True`로 Rank 0만 저장 (메모리 효율)
- Single-device 환경은 일반 state_dict() 사용

---

## Value Weighting

### Critic: Probabilistic Value Learning

**Value Head 학습 (MSE Loss)**:
```python
# Stage 1 (Critic Pretrain): Value Head 단독 학습
# Target: 모든 토큰에 동일한 reward (R_terminal) 부여
value_targets = rewards.unsqueeze(1).expand(batch_size, seq_len, 1)
value_loss = F.mse_loss(value_logits, value_targets)

# 학습 원리: V(s_t) → E[R | s_t] = P(Success | s_t) 자동 수렴
# - Correct 샘플: R=1.0 → V → 1.0 (성공 확률 100%)
# - Incorrect 샘플: R=0.0 → V → 0.0 (실패 확률 100%)
# - 동일 prefix에서 correct/incorrect 섞여있으면 V → 확률값
```

**특징**:
- MSE loss로 Probabilistic value 학습 (TD error 아님)
- Binary reward [0,1] → V(s_t) 자연 bounded [0,1]
- RM 불필요 (~28GB VRAM 절약)

### Verifiable: TD Error Weighting

**TD Error 기반 Weight 계산**:
```python
# 1. Value Head forward (Critic에서 학습된 weights 사용)
value_logits = model.value_head(hidden_states)

# 2. TD error 계산
# Intermediate tokens (k < T): Bootstrapping
td_error_k = gamma * V(s_k) - V(s_{k-1})
# Terminal token (k = T): Direct reward
td_error_T = R - V(s_{T-1})

# 3. Exponential weighting (IQL/AWR 방식)
weight_k = exp(td_error_k / beta)  # beta=0.9
weight_k = clamp(weight_k, min=0.1, max=5.0)

# 4. Weighted loss
weighted_ce_loss = (ce_loss * weights).mean()
```

**Critic Continual Learning** (Stage 2):
```python
# Stage2: Policy + Value 동시 학습
total_loss = weighted_ce_loss + value_coef * value_loss
# value_coef=0.5 (Stable Baselines3 표준)
```

**특징**:
- TD error는 **Weight 계산용**으로만 사용
- Value head 학습 자체는 MSE loss
- Incorrect 샘플 자동 down-weighting (td_error < 0 → weight < 1)

### Rho-1 Weighting

**Top-k Binary Selection**:
```python
# Reference loss 차이 계산
excess_loss = policy_loss - reference_loss

# Per-head binary weights
weights[:, :, 0] = 1.0  # Head 0 always
for head_idx in range(1, n_future_tokens):
    threshold = torch.quantile(excess_loss[:, :, head_idx], 1 - k_percent)
    weights[:, :, head_idx] = (excess_loss[:, :, head_idx] <= threshold).float()
```

**특징**:
- Signed difference (policy - reference)
- Head 0 항상 학습, Head 1~3 selective
- k_percent=0.6 (top 60% 선택)

---

## 주요 디렉터리

```
weighted_mtp/
├── configs/              # 계층적 config (defaults + 실험별)
│   ├── defaults.yaml     # 공통 설정
│   ├── baseline/
│   ├── critic/
│   ├── verifiable/
│   └── rho1/
├── src/weighted_mtp/
│   ├── models/meta_mtp/  # Pure PyTorch Transformer
│   ├── pipelines/        # 4개 독립 파이프라인
│   ├── data/             # 메타데이터 로딩
│   ├── runtime/          # 분산학습 (FSDP)
│   ├── utils/            # S3, checkpoint
│   └── value_weighting/  # TD error, Rho-1
├── storage/
│   ├── models/           # Safetensors 모델
│   ├── datasets/         # JSONL + 메타데이터
│   └── checkpoints/      # 학습 checkpoint
└── tests/
    ├── unit/             # 15개 단위 테스트
    └── integration/      # 5개 통합 테스트
```

---

## 참고

- **구현**: `src/weighted_mtp/models/meta_mtp/transformer.py` (358 lines)
- **메타데이터**: `src/weighted_mtp/data/datasets.py`
- **분산학습**: `src/weighted_mtp/runtime/distributed.py`
- **Value Weighting**: `src/weighted_mtp/value_weighting/`
