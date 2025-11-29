# 독립 Value Model 아키텍처 전면 리팩토링 계획

기존 "trunk + value_head" 구조를 **완전 폐기**하고, InstructGPT 스타일의 **완전히 독립된 Value Model**로 전환한다.

---

## 핵심 변경 사항

### 폐기 대상 (완전 제거)

1. **`MetaLlamaMTPAdapter.value_head`**: trunk에 붙어있는 value head 완전 제거
2. **`value_head.py`의 Policy Model 연동**: Policy Model에서 value head 관련 코드 제거
3. **Verifiable의 value_loss**: Policy 학습에서 value loss 완전 제거
4. **trunk_frozen 개념**: 별도 모델이므로 불필요

### 신규 구현

1. **독립 Value Model**: 별도 backbone (2.7B 등) + value head
2. **HuggingFace 기반**: `LlamaForSequenceClassification` 또는 커스텀 구조
3. **완전 분리된 학습/추론**: Critic에서 학습, Verifiable에서 eval only 로드

---

## 현재 구조 vs 목표 구조

### 현재 구조 (폐기 예정)

```
┌─────────────────────────────────────────┐
│       MetaLlamaMTPAdapter               │
│  ┌─────────────────────────────────┐    │
│  │  Transformer (7B MTP Trunk)     │    │
│  └─────────────────────────────────┘    │
│           ↓              ↓              │
│    MTP Heads (4)    Value Head (MLP)    │  ← 폐기!
└─────────────────────────────────────────┘
```

**문제점:**
- Value head가 MTP trunk의 hidden state에 의존
- trunk_frozen=false 시 gradient 간섭
- 동일 backbone 공유로 인한 목적 충돌
- Value Model 경량화 불가능

### 목표 구조 (완전 분리)

```
┌──────────────────────┐     ┌──────────────────────┐
│     Policy Model     │     │     Value Model      │
│        (7B)          │     │     (2.7B 별도)      │
│  ┌────────────────┐  │     │  ┌────────────────┐  │
│  │ MTP Trunk      │  │     │  │ HF LlamaModel  │  │
│  │ + LoRA         │  │     │  │ (독립 weights) │  │
│  └────────────────┘  │     │  └────────────────┘  │
│         ↓            │     │         ↓            │
│   MTP Heads (4)      │     │   Value Head (MLP)   │
│                      │     │                      │
│   value_head 없음    │     │   독립된 모델        │
└──────────────────────┘     └──────────────────────┘
      CE Loss 학습               Critic에서 학습
      (Weighted)                 Verifiable: Eval Only
```

**장점:**
- **완전 분리**: Policy와 Value가 서로 다른 모델
- **경량화**: Value Model에 더 작은 모델 사용 가능 (2.7B)
- **간섭 제거**: gradient 충돌 완전 제거
- **유연성**: Value Model 아키텍처 자유롭게 선택 가능

---

## 개발 원칙

모든 수정 시 아래 조건을 반드시 따를 것:

1. **[원칙 1]** 추가하려는 기능의 앞/뒤 흐름을 직접 열어 확인 후, 분석하여 현재 구조를 먼저 파악할 것
2. **[원칙 2]** 앞선 구조를 존중하여 필수적인 로직이 누락되지 않도록 하는 동시에, 유사한 기능을 하는 기존 중복 메서드나 파일, utils 등이 불필요하게 반복되지 않도록, 기능 추가 시 섬세하게 분석하고 일관된 흐름을 만들 것
3. **[원칙 3]** 앞선 코드의 구현이 중복되거나, 잘못된 경우 기존 잘못된/중복된 구조를 삭제를 하고 새로 만들어야 함. 최선의 방안을 검토하여 반드시 승인받을 것
4. **[원칙 4]** 앞선 구조 삭제 및 새로운 코드 생성이 결정되면, 하위 호환성을 고려하지말아야 함. 불필요한 중복이나 fallback을 절대 놔두지 말고, 기존 코드를 전격적으로 삭제하고 새로운 코드를 깨끗하게 만들 것
   - **[원칙 4-1]** 앞/뒤 클래스와 메서드 호환 관계를 검토하고, 인자와 변수명을 통일성 있게 네이밍하고, 호출하고, 응답할 것
   - **[원칙 4-2]** 코드 작성 시 단순 wrapper 형태의 비공개 메서드를 반복하지 말 것. 내부 메서드들을 과도하게 계층적으로 만들어서는 안 됨
   - **[원칙 4-3]** 주석 작성 시, 한글로 작성해야 하며 이모지는 써서는 안됨. 이번 수정에만 국한된 불필요한 주석 (ex. phase1, version2.0 등)은 작성하지 말 것. 주석은 코드 동작에 대한 핵심 설명만
5. **[원칙 5]** 요청된 범위 개발 완료 후에는, 다음으로 바로 넘어가지 말고 계획서와 비교해 방향성이 맞는지 검토하고 구현 결과를 계획과 비교해 설명할 것. 성과를 과장하지 말고 계획서와 비교해 객관적으로 기술할 것
   - **[원칙 5-1]** 구현 결과 검토 중, 요청된 개발 범위가 아니더라도 검토 내용 안에 중요한 오류나 개선 포인트가 함께 포함된 경우엔 성과 기술 시에 번외로 알려줄 것
6. **[원칙 6]** 현재 프로젝트의 패키지 의존성 도구(uv, poetry, conda 등)를 반드시 활용하고, 의존성/환경 문제는 코드가 아니라 의존성/환경 문제로서 해결할 것

---

## Phase 1: 독립 Value Model 구현

### 목표
HuggingFace 기반의 완전히 독립된 Value Model을 구현한다. 기존 MTP Transformer와 무관한 별도 모델.

### 작업 내용

#### 1.1 Value Model 클래스 생성

**파일**: `src/weighted_mtp/models/value_model.py`

```python
"""독립 Value Model

HuggingFace LlamaModel 기반의 완전 독립된 Value Model.
Policy Model(MTP)과 완전히 분리되어 별도 backbone 사용.

Critic 파이프라인에서 학습, Verifiable에서 eval only로 사용.
"""

import torch
from torch import nn
from transformers import LlamaModel, LlamaConfig
from pathlib import Path


class ValueModel(nn.Module):
    """독립 Value Model
    
    HuggingFace LlamaModel + Value Head 구조.
    Policy Model과 완전히 독립된 별도 모델.
    
    Args:
        backbone: HuggingFace LlamaModel
        value_head: nn.Module (Linear 또는 MLP)
        config: LlamaConfig
    """
    
    def __init__(
        self,
        backbone: LlamaModel,
        value_head: nn.Module,
        config: LlamaConfig,
    ):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head
        self.config = config
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        value_head_type: str = "mlp",
        dropout: float = 0.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> "ValueModel":
        """HuggingFace pretrained 모델에서 Value Model 생성
        
        Args:
            model_path: HuggingFace 모델 경로 (예: storage/models/ref-sheared-llama-2.7b)
            value_head_type: "linear" 또는 "mlp"
            dropout: MLP dropout
            device: 디바이스
            dtype: 데이터 타입
        
        Returns:
            ValueModel 인스턴스
        """
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        
        # HuggingFace LlamaModel 로드
        config = LlamaConfig.from_pretrained(model_path)
        backbone = LlamaModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        
        # Value Head 생성
        hidden_size = config.hidden_size
        value_head = cls._create_value_head(hidden_size, value_head_type, dropout)
        value_head = value_head.to(device=device, dtype=torch_dtype)
        
        return cls(backbone, value_head, config)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "ValueModel":
        """Critic checkpoint에서 Value Model 로드
        
        Args:
            checkpoint_path: checkpoint 파일 경로
            device: 디바이스
        
        Returns:
            ValueModel 인스턴스 (eval mode)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Config에서 모델 경로 추출
        config_dict = checkpoint.get("config", {})
        model_path = config_dict.get("models", {}).get("value_model", {}).get("path")
        value_head_type = config_dict.get("training", {}).get("value_head_type", "mlp")
        dropout = config_dict.get("training", {}).get("dropout", 0.0)
        dtype = config_dict.get("models", {}).get("value_model", {}).get("dtype", "bfloat16")
        
        # 모델 생성
        model = cls.from_pretrained(
            model_path=model_path,
            value_head_type=value_head_type,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # State dict 로드
        model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        model.value_head.load_state_dict(checkpoint["value_head_state_dict"])
        
        return model
    
    @staticmethod
    def _create_value_head(
        hidden_size: int,
        head_type: str,
        dropout: float,
    ) -> nn.Module:
        """Value Head 생성"""
        if head_type == "linear":
            return nn.Linear(hidden_size, 1, bias=False)
        elif head_type == "mlp":
            hidden1 = hidden_size // 8
            hidden2 = hidden_size // 16
            return nn.Sequential(
                nn.Linear(hidden_size, hidden1, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1, bias=False),
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq]
        
        Returns:
            value_logits: [batch, seq, 1]
        """
        # Backbone forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # Value head
        value_logits = self.value_head(hidden_states)  # [batch, seq, 1]
        
        return value_logits
    
    def freeze_backbone(self):
        """Backbone frozen (value head만 학습)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def eval_mode(self):
        """Eval only 모드 (전체 frozen)"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
```

#### 1.2 디렉토리 구조 변경

```
src/weighted_mtp/models/
├── meta_mtp/
│   ├── adapter.py          # Policy Model (value_head 제거!)
│   ├── transformer.py      # MTP Transformer
│   ├── value_head.py       # 삭제 또는 deprecated
│   └── ...
├── value_model.py          # 독립 Value Model (신규, 단일 파일)
└── __init__.py
```

#### 1.3 Value Head 파일 처리

**`meta_mtp/value_head.py`**: 
- Policy Model에서 더 이상 사용하지 않음
- `ValueModel` 내부에 직접 구현하거나 import
- 기존 파일은 deprecated 처리 후 삭제

### 완료 기준

- [ ] `ValueModel` 클래스 구현 (HuggingFace 기반)
- [ ] `from_pretrained()` 구현 (2.7B 등 경량 모델 지원)
- [ ] `from_checkpoint()` 구현
- [ ] `freeze_backbone()`, `eval_mode()` 구현
- [ ] 단위 테스트 통과

---

## Phase 2: Policy Model에서 Value Head 완전 제거

### 목표
`MetaLlamaMTPAdapter`에서 value_head 관련 코드를 완전히 제거하여 순수 MTP 모델로 만든다.

### 작업 내용

#### 2.1 MetaLlamaMTPAdapter 수정

**파일**: `src/weighted_mtp/models/meta_mtp/adapter.py`

**제거 대상:**

```python
# 제거할 코드들

# 1. __init__에서 value_head 제거
self.value_head = value_head  # 삭제

# 2. from_pretrained에서 value_head 관련 파라미터 제거
initialize_value_head: bool = True,  # 삭제
value_head_type: str = "mlp",        # 삭제
dropout: float = 0.0,                # 삭제

# 3. value_head 초기화 코드 제거
if initialize_value_head:
    value_head = create_value_head(...)  # 삭제

# 4. forward에서 value_logits 반환 제거
return outputs, value_logits  # → return outputs

# 5. trunk_forward 메서드 제거 (더 이상 필요 없음)
def trunk_forward(self, ...):  # 전체 삭제
```

**변경 후 구조:**

```python
class MetaLlamaMTPAdapter(nn.Module):
    """MTP Policy Model Adapter
    
    순수 MTP 모델. Value head 없음.
    Value function은 별도 ValueModel에서 처리.
    """
    
    def __init__(
        self,
        transformer: Transformer,
        model_args: ModelArgs,
        lora_enabled: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.model_args = model_args
        self.lora_enabled = lora_enabled
        # value_head 없음!
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass (MTP logits만 반환)"""
        outputs = self.transformer(input_ids, attention_mask)
        return outputs  # value_logits 없음!
```

#### 2.2 value_head.py 처리

**파일**: `src/weighted_mtp/models/meta_mtp/value_head.py`

**옵션 A: 삭제**
- `ValueModel`에서 value head를 내부 구현
- 기존 파일 완전 삭제

**옵션 B: 이동**
- `ValueModel`이 import할 수 있도록 `models/` 레벨로 이동
- `models/value_head.py`로 이동

**권장: 옵션 A (삭제)**
- `ValueModel` 내부에 직접 구현하여 의존성 단순화

### 완료 기준

- [ ] `MetaLlamaMTPAdapter`에서 value_head 완전 제거
- [ ] `trunk_forward()` 메서드 제거
- [ ] `from_pretrained()`에서 value_head 관련 파라미터 제거
- [ ] `value_head.py` 삭제 또는 이동
- [ ] 기존 테스트 수정 및 통과

---

## Phase 3: Critic 파이프라인 전면 수정

### 목표
Critic 파이프라인을 독립 `ValueModel`을 학습하는 구조로 완전히 재작성한다.

### 작업 내용

#### 3.1 run_critic.py 전면 수정

**변경 전:**
```python
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
adapter = MetaLlamaMTPAdapter.from_pretrained(...)
# trunk + value_head 함께 사용
```

**변경 후:**
```python
from weighted_mtp.models.value_model import ValueModel

# 독립 Value Model 로드 (HuggingFace 기반, 예: 2.7B)
value_model = ValueModel.from_pretrained(
    model_path=config.models.value_model.path,  # ref-sheared-llama-2.7b
    value_head_type=config.training.value_head_type,
    dropout=config.training.dropout,
    device=device,
    dtype=config.models.value_model.dtype,
)

# Backbone frozen 옵션
if config.training.backbone_frozen:
    value_model.freeze_backbone()
```

#### 3.2 Checkpoint 저장 구조

**새로운 저장 함수:**

```python
def save_value_model_checkpoint(
    value_model: ValueModel,
    optimizer: torch.optim.Optimizer,
    epoch: float,
    train_metrics: dict,
    val_metrics: dict,
    checkpoint_path: Path,
    config: dict = None,
) -> None:
    """Value Model 전체 저장
    
    저장 내용:
    - backbone_state_dict: HuggingFace LlamaModel weights
    - value_head_state_dict: Value head weights
    - optimizer_state_dict
    - config (모델 경로 포함, 로드 시 필요)
    """
    checkpoint = {
        "epoch": epoch,
        "backbone_state_dict": value_model.backbone.state_dict(),
        "value_head_state_dict": value_model.value_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": OmegaConf.to_container(config) if config else None,
    }
    torch.save(checkpoint, checkpoint_path)
```

#### 3.3 Config 구조 변경

**파일**: `configs/production/critic_mlp.yaml`

```yaml
# Critic Pre-training (독립 Value Model)

# 모델 설정
models:
  # Value Model (독립, HuggingFace 기반)
  value_model:
    name: ref-sheared-llama-2.7b
    path: storage/models/ref-sheared-llama-2.7b/raw  # HuggingFace 형식
    dtype: bfloat16
  
  # Policy 관련 설정 제거 (Critic은 Value Model만 학습)
  # policy: 제거!

# 학습 설정
training:
  backbone_frozen: true   # Backbone frozen (value head만 학습)
  value_head_type: mlp
  dropout: 0.3
  learning_rate: 1.0e-4
  
  # trunk_frozen: 제거 (별도 모델이므로 불필요)
  # trunk_learning_rate: 제거
```

#### 3.4 학습 루프 단순화

Value Model만 학습하므로 단순해짐:

```python
# Forward
value_logits = value_model(input_ids, attention_mask)

# Loss (MC MSE 또는 Pairwise)
value_loss = compute_value_loss(value_logits, rewards, loss_mask)

# Backward
value_loss.backward()
optimizer.step()
```

### 완료 기준

- [ ] `run_critic.py`에서 `ValueModel` 사용
- [ ] `save_value_model_checkpoint()` 구현
- [ ] Config 구조 변경 (`models.value_model`)
- [ ] 학습 루프 단순화
- [ ] 기존 테스트 수정 및 통과

---

## Phase 4: Verifiable 파이프라인 전면 수정

### 목표
Verifiable 파이프라인에서 Policy Model (value_head 없음)과 Value Model (독립)을 분리하여 사용한다.

### 작업 내용

#### 4.1 run_verifiable.py 전면 수정

**변경 전:**
```python
adapter = MetaLlamaMTPAdapter.from_pretrained(
    model_path=config.models.policy.path,
    value_head_type=value_head_type,
    use_lora=use_lora,
    ...
)
# 동일 adapter에서 policy_logits, value_logits 모두 사용
```

**변경 후:**
```python
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.value_model import ValueModel

# 1. Policy Model 로드 (value_head 없음, LoRA 적용)
policy_model = MetaLlamaMTPAdapter.from_pretrained(
    model_path=config.models.policy.path,
    use_lora=use_lora,
    lora_config=lora_config,
    # value_head 관련 파라미터 없음!
)

# 2. Value Model 로드 (Critic checkpoint에서, eval only)
value_model = ValueModel.from_checkpoint(
    checkpoint_path=config.models.value_model.checkpoint_path,
    device=device,
)
value_model.eval_mode()  # 전체 frozen
```

#### 4.2 Forward Pass 완전 분리

```python
# TD Error 계산 (Value Model 사용, no_grad)
with torch.no_grad():
    value_logits = value_model(pos_input_ids, pos_attention_mask)

td_errors = compute_td_errors(
    value_logits=value_logits,
    rewards=pos_rewards,
    loss_mask=pos_loss_mask,
    gamma=1.0,
)
weights = build_weights(td_errors, ...)

# Policy Loss (Policy Model 사용, gradient 활성화)
policy_outputs = policy_model(pos_input_ids, pos_attention_mask)
policy_logits = policy_outputs.logits  # MTP logits만

ce_loss = compute_mtp_ce_loss(
    logits=policy_logits,
    labels=pos_labels,
    weights=weights,
    ...
)

# Total Loss (CE만, value_loss 없음!)
total_loss = ce_loss
total_loss.backward()
```

#### 4.3 Value Loss 완전 제거

**제거 대상:**

```python
# 모두 제거
value_loss = compute_mc_value_loss(...)
total_loss = weighted_ce_loss + value_loss_coef * value_loss

# Negative sample의 value 계산도 제거
neg_value_logits = ...  # 제거
```

Value Model은 Critic에서 이미 학습 완료. Verifiable에서는 eval only.

#### 4.4 Config 구조 변경

**파일**: `configs/production/verifiable_pairwise.yaml`

```yaml
# Verifiable WMTP (TD Weighting)

# 모델 설정
models:
  # Policy Model (학습 대상, value_head 없음)
  policy:
    name: meta-llama-mtp
    path: storage/models/meta-llama-mtp
    tokenizer_path: storage/models/meta-llama-mtp/tokenizer
    params:
      dim: 4096
      n_layers: 32
      n_heads: 32
      n_future_tokens: 4
    dtype: bfloat16
  
  # Value Model (Critic checkpoint에서 로드, eval only)
  value_model:
    checkpoint_path: storage/checkpoints/critic/critic-pretrain-mlp-final/checkpoint_final.pt
    # HuggingFace 모델 경로는 checkpoint 내 config에 포함

# 학습 설정
training:
  # Policy 학습
  learning_rate: 1.0e-4
  use_lora: true
  lora:
    rank: 64
    alpha: 128.0
    dropout: 0.05
    target_modules: [wq, wk, wv, wo, w1, w2, w3]
  
  # TD error weighting
  beta: 0.9
  weight_clip_min: 0.1
  weight_clip_max: 3
  
  # 제거된 설정들:
  # trunk_frozen: 제거 (별도 모델)
  # value_head_type: 제거 (별도 모델)
  # value_head_learning_rate: 제거
  # value_loss_coef: 제거
  # value_loss: 제거
```

### 완료 기준

- [ ] Policy Model과 Value Model 완전 분리 로드
- [ ] Value Model eval only 설정
- [ ] TD error 계산에 독립 Value Model 사용
- [ ] value_loss 완전 제거
- [ ] Config 구조 변경
- [ ] 통합 테스트 통과

---

## Phase 5: 코드 정리 및 테스트

### 목표
기존 코드 완전 정리, 테스트 업데이트, 문서화.

### 작업 내용

#### 5.1 삭제 대상 코드

| 파일 | 삭제 내용 |
|------|----------|
| `meta_mtp/adapter.py` | `value_head`, `trunk_forward()`, 관련 파라미터 |
| `meta_mtp/value_head.py` | 전체 파일 삭제 |
| `run_verifiable.py` | `value_loss` 계산, negative value 처리 |
| `run_critic.py` | `MetaLlamaMTPAdapter` 사용 부분 |
| `checkpoint_utils.py` | 기존 adapter 저장 로직 정리 |

#### 5.2 Config 정리

| Config 파일 | 변경 내용 |
|-------------|----------|
| `critic_mlp.yaml` | `models.policy` → `models.value_model` |
| `verifiable_pairwise.yaml` | `value_model.checkpoint_path` 추가, value 관련 설정 제거 |

#### 5.3 테스트 업데이트

**신규 테스트:**
- `tests/unit/test_value_model.py`: `ValueModel` 단위 테스트

**수정 테스트:**
- `tests/unit/test_adapter.py`: value_head 관련 테스트 제거
- `tests/integration/test_pipeline_critic.py`: `ValueModel` 사용
- `tests/integration/test_pipeline_verifiable.py`: 분리 로드 테스트

### 완료 기준

- [ ] 삭제 대상 코드 완전 제거
- [ ] 모든 테스트 통과
- [ ] 문서화 완료

---

## 구현 순서 요약

| Phase | 작업 | 핵심 변경 | 의존성 |
|-------|------|----------|--------|
| **Phase 1** | ValueModel 구현 | HuggingFace 기반 독립 모델 | 없음 |
| **Phase 2** | Policy에서 value_head 제거 | MetaLlamaMTPAdapter 정리 | 없음 |
| **Phase 3** | Critic 파이프라인 | ValueModel 학습 구조 | Phase 1 |
| **Phase 4** | Verifiable 파이프라인 | 분리 로드, value_loss 제거 | Phase 1, 2, 3 |
| **Phase 5** | 정리 및 테스트 | 삭제, 문서화 | Phase 1~4 |

---

## 메모리 및 성능 분석

### 메모리 사용량

| 구분 | 현재 구조 | 목표 구조 |
|------|----------|----------|
| **Policy Model** | 7B + LoRA + Value Head | 7B + LoRA |
| **Value Model** | 공유 (0GB 추가) | **2.7B (독립)** |
| **총 메모리** | ~14GB | ~14GB + ~5GB = ~19GB |

### 최적화 방안

1. **Value Model 경량화**
   - 2.7B 사용 (7B 대신)
   - 향후 더 작은 모델 (1.3B) 가능

2. **Eval Only 최적화**
   - `torch.no_grad()` 사용
   - Gradient 메모리 0
   - Activation checkpointing 불필요

3. **FSDP 적용**
   - Policy Model: FSDP (학습용)
   - Value Model: 단순 로드 (eval only, FSDP 불필요)

### 학습 시간

| 파이프라인 | 현재 | 목표 |
|------------|------|------|
| **Critic** | 7B backbone | 2.7B backbone (더 빠름) |
| **Verifiable** | Policy + Value 동시 | Policy만 (더 빠름) |

---

## 참고 자료

- **InstructGPT (Ouyang et al., 2022)**: 별도 Value Model 사용
- **TRL PPOTrainer**: HuggingFace RLHF 구현
- **PPO (Schulman et al., 2017)**: Separate network 논의
- **Sheared LLaMA**: 2.7B 경량 모델
