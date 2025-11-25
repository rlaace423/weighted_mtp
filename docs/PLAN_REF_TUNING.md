# ref-tuning 파이프라인 구현 계획

Rho-1 WMTP에서 사용할 Reference Model(ref-sheared-llama-2.7b)을 codecontests 도메인에 맞게 fine-tuning하는 파이프라인 구현 계획서.

---

## 1. 현재 상황 분석

### 1.1 목적

Rho-1 논문에서 Reference Model은 "어떤 토큰이 학습하기 어려운지"를 판단하는 기준으로 사용됩니다. 현재 `ref-sheared-llama-2.7b`는 일반 도메인에서 사전학습된 모델이므로, codecontests 도메인에 맞게 fine-tuning하면 더 정확한 토큰 선택 기준을 제공할 수 있습니다.

**핵심 가설**: 도메인 적응된 Reference Model이 더 의미 있는 excess loss를 산출함

### 1.2 기존 아키텍처 분석

| 항목 | 기존 파이프라인 (baseline, rho1 등) | ref-tuning (신규) |
|------|-----------------------------------|-------------------|
| **모델 구조** | MetaLlamaMTPAdapter (MTP 아키텍처) | HuggingFace LlamaForCausalLM (표준 NTP) |
| **모델 크기** | 6.7B (meta-llama-mtp) | 2.7B (sheared-llama) |
| **학습 방식** | Multi-Token Prediction | Next Token Prediction |
| **출력 형태** | [batch, seq, n_future, vocab] | [batch, seq, vocab] |
| **Value Head** | 선택적 | 불필요 |

### 1.3 ref-sheared-llama-2.7b 모델 구조

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 2560,
  "num_hidden_layers": 32,
  "num_attention_heads": 20,
  "num_key_value_heads": 20,
  "intermediate_size": 6912,
  "vocab_size": 32000,
  "max_position_embeddings": 4096
}
```

- **위치**: `storage/models/ref-sheared-llama-2.7b/raw/`
- **형식**: HuggingFace PyTorch 모델 (pytorch_model-00001-of-00002.bin 등)
- **토크나이저**: LLaMA 토크나이저 (동일 vocab_size=32000)

### 1.4 재사용 가능한 기존 구성 요소

| 구성 요소 | 재사용 여부 | 비고 |
|----------|------------|------|
| `data/dataloader.py` | O | sampling_config 기반 DataLoader 생성 |
| `data/datasets.py` | O | 메타데이터 기반 샘플링 |
| `data/collators.py` | O | AlpacaDataCollator (input_ids, attention_mask, labels) |
| `runtime/distributed.py` | O | init_distributed, barrier, all_reduce_scalars |
| `runtime/fsdp.py` | △ | FSDP wrapping (HuggingFace 모델 호환 확인 필요) |
| `utils/checkpoint_utils.py` | △ | HuggingFace 형식 저장 추가 필요 |
| `utils/metrics_utils.py` | O | gradient norm, throughput 등 |
| `models/tokenizer_utils.py` | △ | HuggingFace 토크나이저 로드 추가 |
| `core/logging.py`, `core/env.py` | O | 로깅, 환경변수 |

---

## 2. 구현 계획 (Phase별)

### Phase 1: Config 정의 (configs/ref-tuning/)

**목표**: baseline.yaml을 참고하여 ref-tuning 전용 config 생성

**파일 구조**:
```
configs/ref-tuning/
└── ref_tuning.yaml  # 단일 config 파일
```

**주요 설정 항목**:
```yaml
# 실험 메타정보
experiment:
  name: ref-tuning-codecontests
  description: "Reference model fine-tuning for Rho-1"
  stage: ref-tuning

# 모델 설정 (HuggingFace 형식)
models:
  policy:
    name: ref-sheared-llama-2.7b
    path: storage/models/ref-sheared-llama-2.7b/raw
    dtype: bfloat16
    # MTP params 불필요 (HuggingFace 자동 로드)

# 데이터 샘플링 (Rho-1과 동일: 정답만 학습)
data_sampling:
  sampling_method: "difficulty"
  difficulty:
    n_samples: 200000
    correct_ratio: 1.0  # 정답만

# 학습 설정 (2.7B 모델 최적화)
training:
  n_epochs: 2.0
  batch_size: 32  # 더 큰 배치 가능 (모델 크기 작음)
  learning_rate: 5.0e-5
```

**검증 기준**:
- Config 로드 성공 (OmegaConf)
- 경로 유효성 검증 (model path, dataset path)

---

### Phase 2: 파이프라인 구현 (pipelines/run_ref_tuning.py)

**목표**: HuggingFace LlamaForCausalLM 기반 NTP fine-tuning 파이프라인

**핵심 구현 사항**:

#### 2.1 모델 로딩 (HuggingFace 방식)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hf_model(config: dict, device: torch.device) -> nn.Module:
    """HuggingFace 모델 로드 (학습 가능 상태)"""
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(config.models.policy.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.models.policy.path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.train()  # 학습 모드
    return model
```

#### 2.2 토크나이저 로딩

```python
def load_tokenizer(config: dict) -> AutoTokenizer:
    """HuggingFace 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.policy.path,
        use_fast=True,
    )
    # LLaMA 토크나이저는 pad_token이 없음
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
```

#### 2.3 Training Loop (NTP 방식)

```python
def compute_ntp_loss(logits, labels, attention_mask):
    """표준 NTP Cross Entropy Loss 계산

    Args:
        logits: [batch, seq, vocab] HuggingFace 출력
        labels: [batch, seq] 정답 토큰 (output 영역만 유효)
        attention_mask: [batch, seq]

    Returns:
        평균 CE loss (output 토큰만)
    """
    vocab_size = logits.size(-1)

    # Shift: 현재 토큰으로 다음 토큰 예측
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # CE loss 계산 (ignore_index=-100)
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )

    # Output 토큰만 평균
    valid_mask = (shift_labels != -100).float() * shift_mask.float()
    masked_loss = ce_loss * valid_mask.view(-1)

    return masked_loss.sum() / valid_mask.sum()
```

#### 2.4 FSDP Wrapping (HuggingFace 호환)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def wrap_hf_model_fsdp(model, device, config):
    """HuggingFace 모델 FSDP wrapping"""
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=...,  # config에서
        mixed_precision=...,
        device_id=device,
        use_orig_params=True,
    )
    return model
```

#### 2.5 Checkpoint 저장 (HuggingFace 형식)

```python
def save_hf_checkpoint(model, tokenizer, checkpoint_path, config):
    """HuggingFace 형식으로 checkpoint 저장

    Rho-1에서 AutoModelForCausalLM.from_pretrained()로 로드 가능
    """
    # FSDP unwrap
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(...):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    # HuggingFace 형식 저장
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
```

**파일 구조**:
```
src/weighted_mtp/pipelines/
├── run_baseline.py      # 기존
├── run_critic.py        # 기존
├── run_verifiable.py    # 기존
├── run_rho1.py          # 기존
├── run_evaluation.py    # 기존
└── run_ref_tuning.py    # 신규
```

**검증 기준**:
- 단일 GPU 학습 동작 확인
- Loss 감소 추이 확인
- Checkpoint 저장/로드 검증

---

### Phase 3: 유틸리티 확장

**목표**: HuggingFace 모델 지원을 위한 기존 유틸리티 확장

#### 3.1 tokenizer_utils.py 확장

```python
def load_tokenizer_from_config(config) -> AutoTokenizer:
    """Config 기반 토크나이저 로드 (MTP/HuggingFace 자동 분기)"""
    # ref-tuning인 경우 HuggingFace 토크나이저 사용
    if config.experiment.stage == "ref-tuning":
        return AutoTokenizer.from_pretrained(config.models.policy.path)

    # 기존 MTP 토크나이저 로직
    return load_meta_mtp_tokenizer(config)
```

#### 3.2 fsdp.py HuggingFace 지원 추가

기존 `wrap_model_fsdp()` 함수가 MetaLlamaMTPAdapter만 지원하므로, HuggingFace 모델 지원 추가:

```python
def wrap_model_fsdp(model, device, ...):
    """FSDP wrapping (MTP/HuggingFace 자동 분기)"""
    # HuggingFace LlamaForCausalLM 감지
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return _wrap_hf_llama_fsdp(model, device, ...)

    # 기존 MetaLlamaMTPAdapter
    return _wrap_mtp_adapter_fsdp(model, device, ...)
```

#### 3.3 checkpoint_utils.py HuggingFace 저장 추가

```python
def save_hf_checkpoint(model, tokenizer, save_dir, ...):
    """HuggingFace 형식 checkpoint 저장"""
    ...
```

---

### Phase 4: 통합 테스트 및 검증

**목표**: 전체 파이프라인 통합 테스트

#### 4.1 단위 테스트

| 테스트 항목 | 검증 내용 |
|------------|----------|
| Config 로드 | OmegaConf 파싱, 경로 유효성 |
| 모델 로드 | HuggingFace 모델 정상 로드 |
| DataLoader | 기존 collator 호환성 |
| Forward/Backward | NTP loss 계산 및 gradient |
| Checkpoint | HuggingFace 형식 저장/로드 |

#### 4.2 통합 테스트

```bash
# 로컬 테스트 (소규모 데이터)
PYTHONPATH=src python -m weighted_mtp.pipelines.run_ref_tuning \
    --config configs/ref-tuning/ref_tuning.yaml \
    --override data_sampling.difficulty.n_samples=1000 \
    --override training.n_epochs=0.1
```

#### 4.3 분산 학습 테스트

```bash
# 4-GPU 분산 테스트
PYTHONPATH=src torchrun --nproc_per_node=4 \
    src/weighted_mtp/pipelines/run_ref_tuning.py \
    --config configs/ref-tuning/ref_tuning.yaml
```

---

## 3. 개발 원칙 준수 체크리스트

| 원칙 | 준수 방안 |
|------|----------|
| **원칙 1**: 앞/뒤 흐름 확인 | RESEARCH.md, ARCHITECTURE.md, 기존 파이프라인 분석 완료 |
| **원칙 2**: 기존 구조 존중 | data/, runtime/, utils/ 모듈 재사용, 일관된 config 구조 |
| **원칙 3**: 중복 제거 | 기존 유틸리티 확장 (새 파일 최소화) |
| **원칙 4**: 깨끗한 구현 | 변수명/인자명 통일 (config, device, model 등) |
| **원칙 4-1**: 호환 관계 검토 | DataLoader, Collator 인터페이스 유지 |
| **원칙 4-2**: wrapper 최소화 | 직접적인 HuggingFace API 사용 |
| **원칙 4-3**: 주석 한글 | 핵심 동작 설명만, 이모지 미사용 |
| **원칙 5**: 계획 대비 검토 | 각 Phase 완료 후 계획서 대비 검증 |
| **원칙 6**: 의존성 도구 활용 | uv 활용 (transformers 버전 관리) |

---

## 4. 예상 산출물

### 4.1 신규 파일

| 파일 | 설명 |
|------|------|
| `configs/ref-tuning/ref_tuning.yaml` | ref-tuning config |
| `src/weighted_mtp/pipelines/run_ref_tuning.py` | 메인 파이프라인 |

### 4.2 수정 파일

| 파일 | 수정 내용 |
|------|----------|
| `src/weighted_mtp/models/tokenizer_utils.py` | HuggingFace 토크나이저 지원 추가 |
| `src/weighted_mtp/runtime/fsdp.py` | HuggingFace FSDP wrapping 지원 |
| `src/weighted_mtp/utils/checkpoint_utils.py` | HuggingFace 형식 저장 추가 |
| `src/weighted_mtp/pipelines/__init__.py` | run_ref_tuning 추가 |

### 4.3 최종 checkpoint 경로

```
storage/checkpoints/ref-tuning/ref-tuning-codecontests/
├── checkpoint_epoch_0.40/  # HuggingFace 형식
│   ├── config.json
│   ├── pytorch_model.bin (또는 safetensors)
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── checkpoint_final/
```

---

## 5. Rho-1 연동 계획

ref-tuning 완료 후, Rho-1 config에서 reference model 경로 변경:

```yaml
# configs/rho1/rho1.yaml
models:
  reference:
    name: ref-sheared-llama-2.7b-tuned
    path: storage/checkpoints/ref-tuning/ref-tuning-codecontests/checkpoint_final
    dtype: bfloat16
```

---

## 6. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| 토크나이저 불일치 | Rho-1에서 토큰 정렬 실패 | vocab_size 동일 (32000) 확인, 동일 토크나이저 사용 |
| FSDP wrapping 실패 | 분산 학습 불가 | HuggingFace FSDP 공식 문서 참조, NO_SHARD fallback |
| 메모리 부족 | 학습 실패 | batch_size 조정, activation_checkpointing 활성화 |
| Checkpoint 호환성 | 로드 실패 | save_pretrained 표준 API 사용 |

---

## 7. 진행 순서 요약

```
Phase 1 (Config)
    └── configs/ref-tuning/ref_tuning.yaml 생성

Phase 2 (Pipeline)
    └── run_ref_tuning.py 구현
        ├── load_hf_model()
        ├── load_tokenizer()
        ├── compute_ntp_loss()
        ├── validate_ref_tuning()
        └── run_ref_tuning_training()

Phase 3 (Utilities)
    ├── tokenizer_utils.py 확장
    ├── fsdp.py HuggingFace 지원
    └── checkpoint_utils.py HuggingFace 저장

Phase 4 (Testing)
    ├── 단위 테스트
    ├── 로컬 통합 테스트
    └── 분산 학습 테스트
```

---

## 8. 승인 요청 사항

1. **신규 파일 생성**: `configs/ref-tuning/ref_tuning.yaml`, `run_ref_tuning.py`
2. **기존 파일 수정**: tokenizer_utils.py, fsdp.py, checkpoint_utils.py
3. **HuggingFace 형식 checkpoint**: Rho-1 연동을 위한 표준 형식 사용

구현을 진행해도 될까요?
