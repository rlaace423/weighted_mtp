# HuggingFace 2.7B 모델 LoRA 지원 개발 계획

## 개요

HuggingFace LlamaModel/LlamaForCausalLM 기반 2.7B 모델에 LoRA 지원을 추가하여,
critic_mlp와 ref_tuning 파이프라인에서 효율적인 학습을 가능하게 하고,
verifiable과 rho1에서 학습된 LoRA 파라미터를 로드하여 사용할 수 있도록 한다.

---

## 현재 구조 분석

### 파이프라인별 2.7B 모델 사용 현황

| 파이프라인 | 모델 구조 | 학습 방식 | 저장 형식 | 로드 대상 |
|-----------|----------|----------|----------|----------|
| **critic_mlp** | LlamaModel + MLP head | backbone frozen, head만 학습 | 전체 저장 | - |
| **ref_tuning** | LlamaForCausalLM | Full Fine-tuning | HuggingFace 형식 | - |
| **verifiable** | - | - | - | critic checkpoint |
| **rho1** | - | - | - | ref_tuning checkpoint |

### 문제점

1. **critic_mlp**: backbone frozen + head만 학습 → pos/neg 분리 실패 (표현력 부족)
2. **ref_tuning**: Full Fine-tuning → 메모리 비효율, 저장 용량 큼 (5GB+)
3. **저장 형식 불일치**: critic은 커스텀 .pt, ref_tuning은 HuggingFace 디렉토리

---

## 목표 구조

### 통합 LoRA 지원

```
storage/models/ref-sheared-llama-2.7b/raw/  (Base Model, 변경 없음)
     ↓
┌────────────────────────────────────────────────────────────────┐
│                    HfLoraModel (공통 모듈)                      │
│  - LlamaModel 또는 LlamaForCausalLM backbone                   │
│  - LoRA adapter 적용 (q_proj, k_proj, v_proj, o_proj, etc.)   │
│  - Optional: MLP value head                                    │
└────────────────────────────────────────────────────────────────┘
     ↓
┌─────────────────┬─────────────────┐
│   critic_mlp    │   ref_tuning    │
│  LoRA + head    │    LoRA only    │
│     학습         │      학습        │
└─────────────────┴─────────────────┘
     ↓                    ↓
┌─────────────────┬─────────────────┐
│ lora_checkpoint │ lora_checkpoint │
│ + value_head    │    (LoRA만)     │
└─────────────────┴─────────────────┘
     ↓                    ↓
┌─────────────────┬─────────────────┐
│   verifiable    │      rho1       │
│  base + LoRA    │   base + LoRA   │
│  + value_head   │   (ref model)   │
└─────────────────┴─────────────────┘
```

### 통합 Checkpoint 형식

```python
checkpoint = {
    "checkpoint_type": "hf_lora",  # 타입 식별자
    "lora_state_dict": {...},      # LoRA 파라미터 (lora_A, lora_B)
    "value_head_state_dict": {...}, # MLP head (critic만, 없으면 {})
    "lora_config": {
        "rank": 64,
        "alpha": 128.0,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", ...],
    },
    "base_model_path": "storage/models/ref-sheared-llama-2.7b/raw",
    "epoch": 1.5,
    "train_metrics": {...},
    "val_metrics": {...},
    "config": {...},  # 전체 config (재현성)
}
```

---

## Phase별 개발 계획

### Phase 1: 기존 LoRA 모듈 확장 (코드 재사용)

**파일:** `src/weighted_mtp/models/lora/lora_linear.py` (기존 파일에 함수 추가)

#### 1.1 기존 코드 재사용 분석

기존 `lora_linear.py`에 이미 범용적인 LoRA 구현이 존재:

```python
# 기존 코드 (재사용)
class LoRALinear(nn.Module):
    """nn.Linear를 감싸서 low-rank adaptation 수행"""

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank, alpha, dropout):
        """아무 nn.Linear → LoRALinear 변환"""
        ...

def apply_lora_to_linear(module: nn.Module, target_names: list[str], ...):
    """모듈 내 특정 nn.Linear를 LoRALinear로 교체"""
    ...

def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """모델에서 LoRA 파라미터만 추출"""
    ...
```

**이 함수들은 HuggingFace 모델에도 그대로 적용 가능** (nn.Linear 기반이므로)

#### 1.2 HuggingFace 전용 wrapper 함수 추가

```python
# lora_linear.py에 추가

# HuggingFace Llama 모델의 Linear 레이어 이름
HF_ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
HF_MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]

DEFAULT_HF_LORA_CONFIG = {
    "rank": 64,
    "alpha": 128.0,
    "dropout": 0.05,
    "target_modules": HF_ATTENTION_TARGETS + HF_MLP_TARGETS,
}


def apply_lora_to_hf_model(
    model: nn.Module,
    lora_config: dict = None,
) -> nn.Module:
    """HuggingFace Llama 모델에 LoRA 적용

    기존 apply_lora_to_linear 함수를 HuggingFace 구조에 맞게 호출

    Args:
        model: LlamaModel 또는 LlamaForCausalLM
        lora_config: LoRA 설정 (rank, alpha, dropout, target_modules)

    Returns:
        LoRA가 적용된 모델 (원본 weights frozen, LoRA만 학습 가능)
    """
    config = {**DEFAULT_HF_LORA_CONFIG}
    if lora_config:
        config.update(lora_config)

    rank = config["rank"]
    alpha = config["alpha"]
    dropout = config["dropout"]
    target_modules = config["target_modules"]

    # HuggingFace 모델 구조 감지
    # LlamaForCausalLM: model.model.layers
    # LlamaModel: model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("지원하지 않는 모델 구조입니다.")

    # Attention 타겟
    attn_targets = [t for t in target_modules if t in HF_ATTENTION_TARGETS]
    # MLP 타겟
    mlp_targets = [t for t in target_modules if t in HF_MLP_TARGETS]

    # 각 layer에 LoRA 적용 (기존 함수 재사용)
    for layer in layers:
        if attn_targets:
            apply_lora_to_linear(
                module=layer.self_attn,
                target_names=attn_targets,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        if mlp_targets:
            apply_lora_to_linear(
                module=layer.mlp,
                target_names=mlp_targets,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    # 원본 weights frozen (LoRA 파라미터만 학습 가능)
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    return model


def get_hf_lora_state_dict(model: nn.Module) -> dict:
    """HuggingFace 모델에서 LoRA 파라미터만 추출

    기존 get_lora_parameters와 유사하지만 state_dict 형태로 반환
    """
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_hf_lora_state_dict(model: nn.Module, lora_state_dict: dict) -> None:
    """LoRA 파라미터를 HuggingFace 모델에 로드"""
    current_state = model.state_dict()
    for k, v in lora_state_dict.items():
        if k in current_state:
            current_state[k] = v
    model.load_state_dict(current_state)
```

#### 1.3 __init__.py 수정

```python
# models/lora/__init__.py 수정
from .lora_linear import (
    LoRALinear,
    apply_lora_to_linear,
    get_lora_parameters,
    merge_lora_weights,
    # HuggingFace 전용 추가
    apply_lora_to_hf_model,
    get_hf_lora_state_dict,
    load_hf_lora_state_dict,
    DEFAULT_HF_LORA_CONFIG,
)
```

---

### Phase 2: ValueModel LoRA 지원 (critic_mlp용)

**파일:** `src/weighted_mtp/models/value_model.py`

#### 2.1 apply_lora 메서드 추가

```python
class ValueModel(nn.Module):
    def __init__(self, ...):
        ...
        self.lora_enabled = False
        self.lora_config = None

    def apply_lora(self, lora_config: dict = None) -> None:
        """Backbone에 LoRA 적용

        Args:
            lora_config: LoRA 설정 (None이면 기본값 사용)
        """
        from weighted_mtp.models.hf_lora import apply_lora_to_hf_model, DEFAULT_HF_LORA_CONFIG

        config = {**DEFAULT_HF_LORA_CONFIG}
        if lora_config:
            config.update(lora_config)

        apply_lora_to_hf_model(self.backbone, config)

        self.lora_enabled = True
        self.lora_config = config

    def get_trainable_parameters(self) -> list:
        """학습 가능한 파라미터 반환 (LoRA + value_head)"""
        return [p for p in self.parameters() if p.requires_grad]
```

#### 2.2 from_pretrained 수정

```python
@classmethod
def from_pretrained(
    cls,
    model_path: str,
    value_head_type: str = "mlp",
    dropout: float = 0.0,
    device: str = "cuda",
    dtype: str = "bfloat16",
    use_lora: bool = False,
    lora_config: dict = None,
) -> "ValueModel":
    ...
    model = cls(backbone, value_head, config)

    if use_lora:
        model.apply_lora(lora_config)

    return model
```

#### 2.3 from_checkpoint 수정 (LoRA checkpoint 로드)

```python
@classmethod
def from_checkpoint(
    cls,
    checkpoint_path: str,
    device: str = "cuda",
) -> "ValueModel":
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_type = checkpoint.get("checkpoint_type", "full")

    if checkpoint_type == "hf_lora":
        # Base model 로드 + LoRA 적용
        base_model_path = checkpoint["base_model_path"]
        lora_config = checkpoint["lora_config"]

        model = cls.from_pretrained(
            model_path=base_model_path,
            use_lora=True,
            lora_config=lora_config,
            ...
        )

        # LoRA weights 로드
        load_hf_lora_state_dict(model.backbone, checkpoint["lora_state_dict"])

        # Value head 로드
        if checkpoint.get("value_head_state_dict"):
            model.value_head.load_state_dict(checkpoint["value_head_state_dict"])
    else:
        # 기존 full checkpoint 로드 (하위 호환)
        ...

    return model
```

---

### Phase 3: run_critic.py 수정

**파일:** `src/weighted_mtp/pipelines/run_critic.py`

#### 3.1 LoRA 설정 처리

```python
def run_critic_training(config: DictConfig):
    ...
    # LoRA 설정 확인
    use_lora = getattr(config.training, "use_lora", False)
    lora_config = OmegaConf.to_container(config.training.lora) if use_lora else None

    # Value Model 로드
    value_model = ValueModel.from_pretrained(
        model_path=config.models.value_model.path,
        value_head_type=config.training.value_head_type,
        dropout=config.training.dropout,
        device=str(device),
        dtype=config.models.value_model.dtype,
        use_lora=use_lora,
        lora_config=lora_config,
    )

    # use_lora=True면 backbone_frozen 무시 (LoRA가 freeze 제어)
    if not use_lora:
        backbone_frozen = getattr(config.training, "backbone_frozen", True)
        if backbone_frozen:
            value_model.freeze_backbone()
```

#### 3.2 Checkpoint 저장 수정

```python
def save_value_model_checkpoint(..., use_lora: bool = False):
    unwrapped = unwrap_model(value_model)

    if use_lora:
        # LoRA checkpoint (경량)
        from weighted_mtp.models.hf_lora import get_hf_lora_state_dict

        checkpoint = {
            "checkpoint_type": "hf_lora",
            "lora_state_dict": get_hf_lora_state_dict(unwrapped.backbone),
            "value_head_state_dict": unwrapped.value_head.state_dict(),
            "lora_config": unwrapped.lora_config,
            "base_model_path": config.models.value_model.path,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(config, resolve=True),
        }
    else:
        # 기존 full checkpoint
        checkpoint = {
            "checkpoint_type": "full",
            "backbone_state_dict": unwrapped.backbone.state_dict(),
            "value_head_state_dict": unwrapped.value_head.state_dict(),
            ...
        }

    torch.save(checkpoint, checkpoint_path)
```

---

### Phase 4: run_ref_tuning.py LoRA 지원

**파일:** `src/weighted_mtp/pipelines/run_ref_tuning.py`

#### 4.1 LoRA 적용

```python
def load_hf_model(config: DictConfig, device: torch.device) -> torch.nn.Module:
    ...
    model = AutoModelForCausalLM.from_pretrained(...)

    use_lora = getattr(config.training, "use_lora", False)
    if use_lora:
        from weighted_mtp.models.hf_lora import apply_lora_to_hf_model
        lora_config = OmegaConf.to_container(config.training.lora)
        apply_lora_to_hf_model(model, lora_config)

    return model
```

#### 4.2 LoRA Checkpoint 저장

```python
def save_ref_lora_checkpoint(model, checkpoint_path, config, ...):
    """ref_tuning LoRA checkpoint 저장 (value_head 없음)"""
    from weighted_mtp.models.hf_lora import get_hf_lora_state_dict

    checkpoint = {
        "checkpoint_type": "hf_lora",
        "lora_state_dict": get_hf_lora_state_dict(model),
        "value_head_state_dict": {},  # ref_tuning은 value_head 없음
        "lora_config": OmegaConf.to_container(config.training.lora),
        "base_model_path": config.models.policy.path,
        "epoch": epoch,
        ...
    }
    torch.save(checkpoint, checkpoint_path)
```

---

### Phase 5: run_rho1.py Reference 모델 로드 수정

**파일:** `src/weighted_mtp/pipelines/run_rho1.py`

#### 5.1 LoRA Reference 모델 로드

```python
def load_reference_model(config: dict, device: torch.device) -> nn.Module:
    ref_config = config.models.reference
    checkpoint_path = ref_config.get("checkpoint_path")

    if checkpoint_path and checkpoint_path.endswith(".pt"):
        # LoRA checkpoint에서 로드
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if checkpoint.get("checkpoint_type") == "hf_lora":
            from weighted_mtp.models.hf_lora import apply_lora_to_hf_model, load_hf_lora_state_dict

            # Base model 로드
            ref_model = AutoModelForCausalLM.from_pretrained(
                checkpoint["base_model_path"],
                torch_dtype=dtype,
            ).to(device)

            # LoRA 적용 및 weights 로드
            apply_lora_to_hf_model(ref_model, checkpoint["lora_config"])
            load_hf_lora_state_dict(ref_model, checkpoint["lora_state_dict"])

            # LoRA merge (inference 최적화)
            merge_hf_lora_weights(ref_model)
    else:
        # 기존 HuggingFace 디렉토리에서 로드
        ref_model = AutoModelForCausalLM.from_pretrained(ref_config.path, ...)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model
```

---

### Phase 6: Config 수정

#### 6.1 critic_mlp.yaml

```yaml
training:
  # 기존 backbone_frozen은 use_lora=true 시 무시됨
  backbone_frozen: true

  # LoRA 설정 추가
  use_lora: true
  lora:
    rank: 64
    alpha: 128.0
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

  # Value head 설정
  value_head_type: mlp
  dropout: 0.2

checkpoint:
  save_lora_only: true  # LoRA + value_head만 저장
```

#### 6.2 ref_tuning.yaml

```yaml
training:
  use_lora: true
  lora:
    rank: 64
    alpha: 128.0
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

checkpoint:
  save_lora_only: true  # LoRA만 저장 (value_head 없음)
```

#### 6.3 rho1.yaml

```yaml
models:
  reference:
    name: ref-sheared-llama-2.7b
    # LoRA checkpoint 경로로 변경
    checkpoint_path: storage/checkpoints/ref-tuning/lora-ref-tuning/checkpoint_final.pt
    # base_model_path는 checkpoint 내부에 포함됨
    dtype: bfloat16
```

#### 6.4 verifiable.yaml (변경 없음)

```yaml
models:
  value_model:
    # critic LoRA checkpoint (자동으로 hf_lora 타입 감지)
    checkpoint_path: storage/checkpoints/critic/lora-critic-pretrain/checkpoint_final.pt
```

---

## 파일 변경 요약

| Phase | 파일 | 변경 내용 |
|-------|------|----------|
| 1 | `models/lora/lora_linear.py` | 수정: HF 전용 함수 추가 (기존 LoRALinear 재사용) |
| 1 | `models/lora/__init__.py` | 수정: 새 함수 export 추가 |
| 2 | `models/value_model.py` | 수정: apply_lora, from_pretrained, from_checkpoint |
| 3 | `pipelines/run_critic.py` | 수정: LoRA 처리, checkpoint 저장 |
| 4 | `pipelines/run_ref_tuning.py` | 수정: LoRA 적용, checkpoint 저장 |
| 5 | `pipelines/run_rho1.py` | 수정: LoRA reference 모델 로드 |
| 6 | `configs/production/critic_mlp.yaml` | 수정: lora 설정 추가 |
| 6 | `configs/production/ref_tuning.yaml` | 수정: lora 설정 추가 |
| 6 | `configs/production/rho1.yaml` | 수정: checkpoint_path 방식 |

### 코드 재사용 현황

| 기존 코드 | 재사용 여부 | 용도 |
|----------|------------|------|
| `LoRALinear` 클래스 | 100% 재사용 | nn.Linear → LoRALinear 변환 |
| `LoRALinear.from_linear()` | 100% 재사용 | 원본 weights 복사하여 LoRA 생성 |
| `apply_lora_to_linear()` | 100% 재사용 | 모듈 내 Linear 교체 |
| `get_lora_parameters()` | 100% 재사용 | LoRA 파라미터 추출 |
| `merge_lora_weights()` | 100% 재사용 | inference 최적화 |

---

## 하위 호환성

| 기존 방식 | 호환 여부 | 처리 방법 |
|----------|----------|----------|
| critic full checkpoint | O | checkpoint_type="full" 자동 감지 |
| ref_tuning HuggingFace 디렉토리 | O | checkpoint_path 없으면 기존 path 사용 |
| verifiable 기존 로드 | O | from_checkpoint에서 타입 자동 감지 |

---

## 예상 효과

| 항목 | 기존 | LoRA 적용 후 |
|------|------|-------------|
| critic checkpoint 크기 | ~5GB (전체) | ~200MB (LoRA+head) |
| ref_tuning checkpoint | ~5GB (HF 디렉토리) | ~100MB (LoRA만) |
| critic 학습 표현력 | 낮음 (head만) | 높음 (LoRA+head) |
| ref_tuning 메모리 | Full FT | LoRA만 학습 |

---

## 테스트 계획

1. **Unit Test**: hf_lora_utils 함수 테스트
2. **Integration Test**:
   - critic_mlp LoRA 학습 → checkpoint 저장 → verifiable 로드
   - ref_tuning LoRA 학습 → checkpoint 저장 → rho1 로드
3. **E2E Test**: 전체 파이프라인 실행 (local config)
