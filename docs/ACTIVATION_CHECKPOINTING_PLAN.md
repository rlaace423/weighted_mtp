# FSDP + Activation Checkpointing + Sequential Backward 호환성 해결 계획

## 목차
1. [배경 지식: 분산학습이란?](#1-배경-지식-분산학습이란)
2. [우리 프로젝트 파이프라인 구조](#2-우리-프로젝트-파이프라인-구조)
3. [문제 상황 분석](#3-문제-상황-분석)
4. [해결 방안](#4-해결-방안)
5. [Phase별 구현 계획](#5-phase별-구현-계획)

---

## 1. 배경 지식: 분산학습이란?

### 1.1 왜 분산학습이 필요한가?

LLM(Large Language Model)은 파라미터가 수십억 개입니다. 예를 들어:
- LLaMA 7B: 70억 개 파라미터 → 약 14GB (FP16 기준)
- 학습 시에는 gradient, optimizer state도 필요 → 약 56GB 이상

**단일 GPU(예: 80GB H100)로는 부족합니다.**

```
┌──────────────────────────────────────────────────────────┐
│                    단일 GPU 메모리 구성                    │
├──────────────────────────────────────────────────────────┤
│  Model Parameters     : 14GB                             │
│  Gradients           : 14GB                             │
│  Optimizer States    : 28GB (Adam: 2x params)           │
│  Activations         : ?GB (batch size에 비례)           │
├──────────────────────────────────────────────────────────┤
│  Total               : 56GB + α (80GB 초과 가능)         │
└──────────────────────────────────────────────────────────┘
```

### 1.2 FSDP (Fully Sharded Data Parallel)란?

FSDP는 모델을 여러 GPU에 **쪼개서(shard)** 저장하는 기술입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FSDP 개념도                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [원본 모델: Layer 1-32]                                        │
│           │                                                     │
│           ▼ FSDP FULL_SHARD                                     │
│   ┌───────────────┬───────────────┬───────────────┐             │
│   │    GPU 0      │    GPU 1      │    GPU 2      │             │
│   │  Layer 1-11   │  Layer 12-22  │  Layer 23-32  │             │
│   │  (1/3 저장)   │  (1/3 저장)   │  (1/3 저장)   │             │
│   └───────────────┴───────────────┴───────────────┘             │
│                                                                 │
│   Forward 시: 필요한 layer를 all-gather로 모아서 계산            │
│   Backward 시: gradient 계산 후 다시 shard                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 포인트:**
- 평소에는 각 GPU가 모델의 1/N만 저장 (메모리 절약)
- 계산이 필요할 때만 all-gather로 전체 파라미터를 모음
- 계산 끝나면 다시 쪼개서 저장

### 1.3 Activation Checkpointing이란?

Forward pass 중 생성되는 중간 결과(activation)도 메모리를 많이 차지합니다.

```
일반적인 Forward-Backward:
┌──────────────────────────────────────────────────────────┐
│ Forward:  Input → [Act1] → [Act2] → [Act3] → Output     │
│                     ↓        ↓        ↓                  │
│                   저장      저장      저장   (메모리 사용)│
│                                                          │
│ Backward: Output.grad → Act3 → Act2 → Act1 → Input.grad │
│                          ↑       ↑       ↑               │
│                        사용     사용     사용             │
└──────────────────────────────────────────────────────────┘

Activation Checkpointing:
┌──────────────────────────────────────────────────────────┐
│ Forward:  Input → [Act1] → [Act2] → [Act3] → Output     │
│                     X        X        X    (저장 안 함)  │
│                                                          │
│ Backward: Output.grad                                    │
│           → 다시 Forward 계산 (recompute) → Act3 사용    │
│           → 다시 Forward 계산 (recompute) → Act2 사용    │
│           → 다시 Forward 계산 (recompute) → Act1 사용    │
└──────────────────────────────────────────────────────────┘
```

**Trade-off:**
- 메모리 절약 (activation 저장 안 함)
- 계산 시간 증가 (forward를 2번 수행)

---

## 2. 우리 프로젝트 파이프라인 구조

### 2.1 MTP (Multi-Token Prediction)란?

일반 LLM은 다음 1개 토큰만 예측합니다. MTP는 **다음 4개 토큰을 동시에** 예측합니다.

```
일반 LLM (Next Token Prediction):
┌──────────────────────────────────────────────────────────┐
│ Input: "The cat sat on the"                              │
│ Output: "mat" (다음 1개만 예측)                           │
└──────────────────────────────────────────────────────────┘

MTP (Multi-Token Prediction):
┌──────────────────────────────────────────────────────────┐
│ Input: "The cat sat on the"                              │
│ Output:                                                  │
│   Head 1: "mat"     (t+1 예측)                           │
│   Head 2: "."       (t+2 예측)                           │
│   Head 3: "The"     (t+3 예측)                           │
│   Head 4: "dog"     (t+4 예측)                           │
└──────────────────────────────────────────────────────────┘
```

### 2.2 우리 모델 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    MetaLlamaMTPAdapter 구조                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Tokens [batch, seq]                                      │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │         Embedding Layer                 │                    │
│  │         [batch, seq, 4096]              │                    │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │      Trunk Layers (Layer 1-28)          │  ◀── 공유 부분     │
│  │      TransformerBlock × 28              │                    │
│  └─────────────────────────────────────────┘                    │
│       │                                                         │
│       ▼  h_trunk: [batch, seq, 4096]                            │
│       │                                                         │
│  ┌────┴────┬────────┬────────┬────────┐                         │
│  │         │        │        │        │                         │
│  ▼         ▼        ▼        ▼        ▼                         │
│ ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐                       │
│ │L29│    │L30│    │L31│    │L32│    │   │  ◀── Prediction Heads │
│ └───┘    └───┘    └───┘    └───┘    └───┘      (Head 1-4)       │
│   │        │        │        │        │                         │
│   ▼        ▼        ▼        ▼        ▼                         │
│ ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐                       │
│ │Out│    │Out│    │Out│    │Out│    │   │  ◀── Output Projection│
│ └───┘    └───┘    └───┘    └───┘    └───┘      [vocab=32000]    │
│   │        │        │        │        │                         │
│   ▼        ▼        ▼        ▼        ▼                         │
│ Head1    Head2    Head3    Head4                                │
│ (t+1)    (t+2)    (t+3)    (t+4)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Sequential Unembedding이란?

일반적인 방식은 4개 Head의 logits를 한 번에 계산합니다:
```python
# 일반 방식: 메모리 많이 사용
logits = model(input)  # [batch, seq, 4, vocab=32000]
# → batch=48, seq=2048 기준: 48 × 2048 × 4 × 32000 × 2bytes = 25GB
```

**Sequential Unembedding**은 Head별로 하나씩 처리합니다:
```python
# Sequential: 메모리 절약
for head_idx in [1, 2, 3, 4]:
    logits_k = compute_one_head(head_idx)  # [batch, seq, vocab]
    loss_k = compute_loss(logits_k)
    loss_k.backward()  # 즉시 backward → logits_k 메모리 해제
    del logits_k
```

### 2.4 3가지 Pipeline과 Weighting 방식

우리 프로젝트는 3가지 학습 방식이 있습니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline 비교                             │
├──────────────┬──────────────────────────────────────────────────┤
│   Pipeline   │              Weighting 방식                       │
├──────────────┼──────────────────────────────────────────────────┤
│              │                                                  │
│   Baseline   │  균등 가중치 (모든 토큰 weight = 1.0)             │
│              │  weights = None                                  │
│              │                                                  │
├──────────────┼──────────────────────────────────────────────────┤
│              │                                                  │
│   Rho-1      │  Per-Head Binary Selection                       │
│              │  Reference 모델과 비교해서 "학습 필요한" 토큰만 선택│
│              │  weights: [batch, seq, n_future=4]  ◀── 3D       │
│              │                                                  │
│              │  예시: Head별로 다른 토큰이 선택됨                 │
│              │  weights[0, :, 0] = [1,0,1,1,0,...]  (Head 1)    │
│              │  weights[0, :, 1] = [0,1,1,0,1,...]  (Head 2)    │
│              │                                                  │
├──────────────┼──────────────────────────────────────────────────┤
│              │                                                  │
│  Verifiable  │  Position-level TD Weighting                     │
│              │  Value Model의 TD error 기반 가중치               │
│              │  weights: [batch, seq]  ◀── 2D                   │
│              │                                                  │
│              │  예시: 위치별로 같은 가중치가 모든 Head에 적용     │
│              │  weights[0, :] = [0.5, 1.2, 0.8, 2.1, ...]       │
│              │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

### 2.5 Tensor 차원 흐름 (Sequential Unembedding)

```
입력:
  tokens: [batch=48, seq=2048]

Trunk Forward:
  h_trunk: [batch=48, seq=2048, dim=4096]

Head별 처리 (head_idx = 1, 2, 3, 4):
  ┌─────────────────────────────────────────────────────────────┐
  │ Head 1 (t+1 예측):                                          │
  │   logits_k: [48, 2048, 32000]                               │
  │   labels:   [48, 2047]  (valid_len = seq - 1)               │
  │   weights:                                                  │
  │     - Baseline:   None                                      │
  │     - Rho-1:      [48, 2047] ← weights[:, :2047, 0]         │
  │     - Verifiable: [48, 2047] ← weights[:, 1:2048]           │
  ├─────────────────────────────────────────────────────────────┤
  │ Head 2 (t+2 예측):                                          │
  │   logits_k: [48, 2048, 32000]                               │
  │   labels:   [48, 2046]  (valid_len = seq - 2)               │
  │   weights:                                                  │
  │     - Baseline:   None                                      │
  │     - Rho-1:      [48, 2046] ← weights[:, :2046, 1]         │
  │     - Verifiable: [48, 2046] ← weights[:, 2:2048]           │
  ├─────────────────────────────────────────────────────────────┤
  │ ... (Head 3, 4 동일 패턴)                                    │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. 문제 상황 분석

### 3.1 에러 메시지

```
RuntimeError: The size of tensor a (4096) must match the size of
tensor b (0) at non-singleton dimension 2
```

위치: `transformer.py:52` - `RMSNorm.forward()`의 `output * self.weight`

### 3.2 왜 이런 에러가 발생하는가?

**문제의 코드 (transformer.py:366):**
```python
for head_idx in [1, 2, 3, 4]:
    logits_k = compute_head_forward(...)
    loss_k = compute_loss(logits_k)

    # 문제 지점!
    loss_k.backward(retain_graph=(head_idx < 4))
```

**`retain_graph=True`란?**

PyTorch는 backward() 후 연산 그래프를 삭제합니다. 여러 번 backward()를 하려면 그래프를 유지해야 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    retain_graph 동작                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  retain_graph=False (기본):                                     │
│    loss.backward()                                              │
│    → gradient 계산 후 연산 그래프 삭제                           │
│    → 두 번째 backward() 불가능                                   │
│                                                                 │
│  retain_graph=True:                                             │
│    loss.backward(retain_graph=True)                             │
│    → gradient 계산 후 연산 그래프 유지                           │
│    → 두 번째 backward() 가능                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 FSDP + Checkpointing + retain_graph 충돌

**문제 발생 시나리오:**

```
시간순서 →

1. Forward (Head 1):
   ┌─────────────────────────────────────────────────────────────┐
   │ FSDP: Layer 29 파라미터를 all-gather (모든 GPU에서 모음)     │
   │ → 계산 수행                                                  │
   │ → Activation Checkpointing: 중간 결과 저장 안 함            │
   │ FSDP: Layer 29 파라미터를 다시 shard (쪼개서 저장)           │
   └─────────────────────────────────────────────────────────────┘

2. Loss 계산 (Head 1)

3. Backward (Head 1) with retain_graph=True:
   ┌─────────────────────────────────────────────────────────────┐
   │ Activation Checkpointing: Forward를 다시 계산해야 함!       │
   │                                                             │
   │ 문제: FSDP 파라미터가 이미 sharded 상태                      │
   │      → all-gather가 제대로 호출되지 않음                     │
   │      → RMSNorm.weight 크기가 0 (빈 텐서)                    │
   │      → RuntimeError 발생!                                   │
   └─────────────────────────────────────────────────────────────┘
```

**그림으로 보는 충돌:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        충돌 메커니즘                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  정상 상황 (retain_graph=False):                                │
│                                                                 │
│  Forward ──────────────────────▶ Backward                       │
│     │                               │                           │
│     │ FSDP gather                   │ FSDP gather               │
│     │ (파라미터 모음)               │ (파라미터 모음)            │
│     ▼                               ▼                           │
│   계산                            계산                          │
│     │                               │                           │
│     │ FSDP scatter                  │ FSDP scatter              │
│     ▼                               ▼                           │
│   완료                            완료                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  문제 상황 (retain_graph=True + Checkpointing):                 │
│                                                                 │
│  Forward ──────────────────────▶ Backward (1st)                 │
│     │                               │                           │
│     │ FSDP gather                   │ Checkpointing:            │
│     ▼                               │ "Forward 다시 계산해야 해"│
│   계산                              │                           │
│     │                               │ FSDP: "???"               │
│     │ FSDP scatter                  │ (gather 호출 안 됨)       │
│     ▼                               │                           │
│   완료                              ▼                           │
│                                  파라미터 크기 = 0              │
│                                  RuntimeError!                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 왜 `tensor b (0)`인가?

FSDP `FULL_SHARD`에서:
- 3개 GPU 사용 시 각 GPU는 파라미터의 1/3만 저장
- Forward 컨텍스트 외부에서는 sharded 상태
- `RMSNorm.weight`의 크기: 4096 → 4096/3 ≈ 1365 (로컬)
- recomputation 시 gather 실패 → 크기 0 (빈 텐서)

---

## 4. 해결 방안

### 4.1 핵심 아이디어

**`retain_graph=True`를 제거하고, 단일 backward로 변경**

```python
# 기존 (문제 코드)
for head_idx in [1, 2, 3, 4]:
    loss_k = compute_loss(head_idx)
    loss_k.backward(retain_graph=(head_idx < 4))  # 여러 번 backward

# 수정 후 (해결)
total_loss = 0
for head_idx in [1, 2, 3, 4]:
    loss_k = compute_loss(head_idx)
    total_loss = total_loss + loss_k  # loss만 누적

total_loss.backward()  # 한 번만 backward (loop 끝난 후)
```

### 4.2 왜 이 방법이 작동하는가?

```
┌─────────────────────────────────────────────────────────────────┐
│                      수정 후 동작 흐름                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Forward Phase (모든 Head):                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Head 1: FSDP gather → 계산 → loss_1 → FSDP scatter      │    │
│  │ Head 2: FSDP gather → 계산 → loss_2 → FSDP scatter      │    │
│  │ Head 3: FSDP gather → 계산 → loss_3 → FSDP scatter      │    │
│  │ Head 4: FSDP gather → 계산 → loss_4 → FSDP scatter      │    │
│  │                                                         │    │
│  │ total_loss = loss_1 + loss_2 + loss_3 + loss_4          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  Backward Phase (한 번만):                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ total_loss.backward()                                   │    │
│  │                                                         │    │
│  │ Checkpointing이 Forward를 recompute할 때:               │    │
│  │ → FSDP forward 컨텍스트 내에서 실행됨                   │    │
│  │ → 파라미터가 정상적으로 gather됨                        │    │
│  │ → 문제 없음!                                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 메모리 영향

**걱정:** "loss를 누적하면 메모리가 더 필요하지 않나요?"

**답:** 약간 증가하지만, Activation Checkpointing으로 상쇄됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      메모리 비교                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  기존 방식 (retain_graph + No Checkpointing):                   │
│  - 모든 activation 저장 필요                                    │
│  - 메모리: 매우 높음                                            │
│                                                                 │
│  기존 방식 (retain_graph + Checkpointing):                      │
│  - 작동 안 함 (RuntimeError)                                    │
│                                                                 │
│  수정 방식 (single backward + Checkpointing):                   │
│  - loss graph만 유지 (작음)                                     │
│  - activation은 checkpointing으로 절약                          │
│  - 메모리: 적당함                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase별 구현 계획

### Phase 1: Sequential Loss 계산 방식 수정

**목표:** `retain_graph=True` 제거

**수정 파일:** `src/weighted_mtp/models/meta_mtp/transformer.py`

**변경 내용:**

```python
# 변경 전 (line 346-377)
def forward(self, ...):
    ...
    if compute_sequential_loss:
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for head_idx, layer in enumerate(prediction_heads, start=1):
            h = layer(h_trunk, ...)
            logits_k = self.output(h)
            loss_k = compute_head_ce_loss(...)

            # 문제 지점
            scaled_loss = (loss_k * loss_scale) / n_future
            scaled_loss.backward(retain_graph=(head_idx < n_future))

            del logits_k, h
            total_loss = total_loss + loss_k.detach()

        return {"loss": total_loss / n_future, "n_heads": n_future}

# 변경 후
def forward(self, ...):
    ...
    if compute_sequential_loss:
        # loss를 누적할 변수 (gradient graph 유지)
        accumulated_loss = None

        for head_idx, layer in enumerate(prediction_heads, start=1):
            h = layer(h_trunk, ...)
            logits_k = self.output(h)
            loss_k = compute_head_ce_loss(...)

            # loss 누적 (backward는 하지 않음)
            if accumulated_loss is None:
                accumulated_loss = loss_k
            else:
                accumulated_loss = accumulated_loss + loss_k

            del logits_k, h  # 메모리 절약

        # 단일 backward (loop 종료 후)
        scaled_loss = (accumulated_loss / n_future) * loss_scale
        scaled_loss.backward()

        return {
            "loss": accumulated_loss.detach() / n_future,
            "n_heads": n_future
        }
```

### Phase 2: 메모리 최적화

**목표:** loss graph 최소화

**전략:** hidden state를 detach하여 trunk와 head gradient 분리

```python
# h_trunk에서 gradient 흐름 분리
h_trunk_for_heads = h_trunk.detach().requires_grad_(True)

# Head 처리
for head_idx, layer in enumerate(prediction_heads, start=1):
    h = layer(h_trunk_for_heads, ...)  # detached trunk 사용
    ...

# backward
scaled_loss.backward()

# trunk에 gradient 전파 (별도)
if h_trunk.requires_grad:
    h_trunk.backward(h_trunk_for_heads.grad)
```

### Phase 3: Weighting 로직 검증

**목표:** 3가지 Pipeline 모두 정상 작동 확인

**검증 항목:**

| Pipeline | weights 입력 | 슬라이싱 로직 | 기대 동작 |
|----------|-------------|--------------|----------|
| Baseline | `None` | N/A | 균등 가중치 1.0 |
| Rho-1 | `[B,S,4]` | `[:, :valid_len, head_idx-1]` | Per-head selection |
| Verifiable | `[B,S]` | `[:, head_idx:head_idx+valid_len]` | Position-level |

### Phase 4: 통합 테스트

**테스트 명령:**

```bash
# 환경 설정
export MLFLOW_URI="file:///path/to/mlruns"

# Baseline 테스트
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun \
  --nproc_per_node=3 \
  -m weighted_mtp.pipelines.run_baseline \
  --config configs/production/baseline.yaml \
  --override distributed.fsdp.activation_checkpointing=true

# Rho-1 테스트
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun \
  --nproc_per_node=3 \
  -m weighted_mtp.pipelines.run_rho1 \
  --config configs/production/rho1.yaml \
  --override distributed.fsdp.activation_checkpointing=true

# Verifiable 테스트
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun \
  --nproc_per_node=3 \
  -m weighted_mtp.pipelines.run_verifiable \
  --config configs/production/verifiable.yaml \
  --override distributed.fsdp.activation_checkpointing=true
```

### Phase 5: 메모리 프로파일링

**확인 항목:**
- Peak GPU memory
- batch_size 유지 가능 여부
- 필요 시 `gradient_accumulation_steps` 조정

---

## 6. 요약

### 문제
- FSDP + Activation Checkpointing + `retain_graph=True` 조합이 호환되지 않음
- Checkpointing의 recomputation 시 FSDP 파라미터가 gather되지 않음

### 해결
- `retain_graph=True` 제거
- 모든 head의 loss를 누적 후 **단일 backward** 수행
- Checkpointing과 FSDP가 정상적으로 협력 가능

### 결과
- Activation Checkpointing 사용 가능 → 메모리 절약
- 기존 batch_size 유지 가능
- 3개 Pipeline (Baseline, Rho-1, Verifiable) 모두 호환

---

## 7. 관련 파일

| 파일 | 역할 | 수정 여부 |
|------|------|----------|
| `transformer.py` | Sequential loss 계산 | **수정 필요** |
| `loss_utils.py` | Head별 CE loss 계산 | 검증만 |
| `fsdp.py` | FSDP + Checkpointing 설정 | 변경 없음 |
| `run_baseline.py` | Baseline pipeline | 변경 없음 |
| `run_rho1.py` | Rho-1 pipeline | 변경 없음 |
| `run_verifiable.py` | Verifiable pipeline | 변경 없음 |
