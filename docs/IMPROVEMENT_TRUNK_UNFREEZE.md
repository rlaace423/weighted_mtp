# Trunk 상위 블록 Unfreeze 개선 계획

## 배경

현재 Critic 파이프라인은 모든 Transformer trunk 파라미터를 frozen하고 value head(4096→1 단일 Linear)만 학습함. 이는 frozen representation 위에서 linear probe를 수행하는 것과 동일하여 표현력 부족 문제 발생.

**현재 문제점**:
- value/mean_prediction ≈ 0.4766 (거의 상수)
- value/std_prediction ≈ 0.084 (매우 낮음)
- value/mse ≈ 0.2578 (상수 0.65 모델보다 낮은 성능)

**RLHF 표준 레시피**: Value head는 단일 Linear로 유지하되, trunk 상위 레이어를 unfreeze하여 representation 자체가 reward 신호를 반영하도록 함.

---

## 현재 구조 분석

### Freeze 로직 위치
```python
# src/weighted_mtp/pipelines/run_critic.py:213-220
for param in adapter.transformer.parameters():
    param.requires_grad = False

for param in adapter.value_head.parameters():
    param.requires_grad = True
```

### Transformer 구조
```python
# src/weighted_mtp/models/meta_mtp/transformer.py
class Transformer(nn.Module):
    self.tok_embeddings = nn.Embedding(...)
    self.layers = nn.ModuleList()  # TransformerBlock 리스트
    self.norm = RMSNorm(...)
    # MTP heads...
```

### 영향 범위
- `run_critic.py`: Stage 1 (Value head pretrain)
- `run_verifiable.py`: Stage 2 (현재 freeze 없음, 전체 학습)

---

## 개선 계획

### Phase 1: Config 확장

**목표**: `num_unfrozen_layers` 설정 추가

**수정 파일**:
- `configs/critic/critic_local.yaml`
- `configs/critic/critic_1gpu.yaml`
- `configs/critic/critic_4gpu.yaml`

**추가 설정**:
```yaml
training:
  # 기존 설정...
  num_unfrozen_layers: 4  # 마지막 N개 TransformerBlock unfreeze (0=value head만)
```

**기본값**: 4 (meta-llama-mtp는 32개 레이어, 마지막 4개 = 12.5%)

---

### Phase 2: Freeze 로직 수정

**목표**: 마지막 N개 블록 선택적 unfreeze

**수정 파일**: `src/weighted_mtp/pipelines/run_critic.py`

**현재 코드** (213-220행):
```python
logger.info("Freezing transformer trunk (training value head only)")
for param in adapter.transformer.parameters():
    param.requires_grad = False

for param in adapter.value_head.parameters():
    param.requires_grad = True
```

**수정 후 코드**:
```python
num_unfrozen = config.training.get("num_unfrozen_layers", 0)
n_layers = len(adapter.transformer.layers)

if num_unfrozen > 0:
    # 마지막 N개 블록만 학습
    logger.info(f"Unfreezing last {num_unfrozen} transformer blocks (out of {n_layers})")

    # 전체 frozen
    for param in adapter.transformer.parameters():
        param.requires_grad = False

    # 마지막 N개 블록 unfreeze
    for layer in adapter.transformer.layers[-num_unfrozen:]:
        for param in layer.parameters():
            param.requires_grad = True

    # final norm도 unfreeze (마지막 블록 출력에 영향)
    for param in adapter.transformer.norm.parameters():
        param.requires_grad = True
else:
    # 기존 동작: value head만 학습
    logger.info("Freezing transformer trunk (training value head only)")
    for param in adapter.transformer.parameters():
        param.requires_grad = False

# Value head는 항상 학습
for param in adapter.value_head.parameters():
    param.requires_grad = True
```

**설계 결정**:
- `num_unfrozen_layers=0`: 기존 동작 (value head만)
- `num_unfrozen_layers=4`: 마지막 4개 블록 + final norm + value head
- final norm 포함 이유: 마지막 블록 출력이 norm을 거쳐 value head로 전달됨

---

### Phase 3: FSDP 호환성 확인

**현재 FSDP 설정** (run_critic.py:222-230):
```python
adapter = wrap_model_fsdp(
    adapter,
    device,
    sharding_strategy=config.distributed.fsdp.sharding_strategy,
    mixed_precision=config.distributed.fsdp.mixed_precision,
    cpu_offload=config.distributed.fsdp.cpu_offload,
    activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    ignored_modules=[adapter.value_head],
)
```

**검토 사항**:
1. `ignored_modules`: value head만 제외 중 - 수정 불필요
2. `use_orig_params=True`: 기존 설정 유지 (optimizer 호환성)
3. Trainable params filtering: 이미 `requires_grad=True`만 선택

**수정 필요 여부**: 없음 - 현재 FSDP 설정이 이미 호환됨

**이유**:
- FSDP는 `requires_grad` 상태를 존중
- `ignored_modules`는 value head를 FSDP wrapping에서 제외하여 별도 관리
- optimizer는 `[p for p in adapter.parameters() if p.requires_grad]`로 필터링

---

### Phase 4: Optimizer 설정 검토

**현재 설정** (run_critic.py:315-323):
```python
trainable_params = [p for p in adapter.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=config.training.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)
```

**고려 사항**:
- 파라미터 수 증가로 메모리 사용량 증가
- Learning rate 조정 필요 가능성 (trunk 레이어는 더 작은 lr 권장)

**선택적 개선**: Layer-wise Learning Rate Decay (LLRD)
```yaml
training:
  learning_rate: 1.0e-5
  lr_decay_factor: 0.9  # 레이어별 lr 감쇠 (선택)
```

**Phase 4에서는 기본 설정만 적용**, LLRD는 추후 성능 개선 필요 시 추가

---

### Phase 5: 로깅 및 모니터링 개선

**추가 메트릭**:
```python
# Trainable params 상세 로깅
trainable_by_component = {
    "value_head": sum(p.numel() for p in adapter.value_head.parameters() if p.requires_grad),
    "trunk_blocks": sum(
        p.numel() for layer in adapter.transformer.layers[-num_unfrozen:]
        for p in layer.parameters() if p.requires_grad
    ),
    "norm": sum(p.numel() for p in adapter.transformer.norm.parameters() if p.requires_grad),
}
logger.info(f"Trainable params breakdown: {trainable_by_component}")
```

**MLflow 로깅**:
```python
mlflow.log_params({
    "model_num_unfrozen_layers": num_unfrozen,
    "model_trainable_trunk_params": trainable_by_component["trunk_blocks"],
})
```

---

## 예상 파라미터 수 변화

### Meta LLaMA MTP (6.7B, 32 layers)

| 설정 | Trainable Params | 비율 |
|------|------------------|------|
| `num_unfrozen_layers=0` | ~4K (value head만) | 0.00006% |
| `num_unfrozen_layers=4` | ~850M | 12.7% |
| `num_unfrozen_layers=8` | ~1.7B | 25.4% |

### 메모리 영향
- 4개 블록 unfreeze 시 gradient 메모리 ~3-4GB 추가 예상
- Optimizer state 추가 ~6-8GB
- 총 추가 메모리: ~10-12GB

**권장**: A100 80GB에서 4개 블록 unfreeze 문제없음

---

## 테스트 계획

### 단위 테스트
```python
# tests/unit/test_freeze_logic.py
def test_num_unfrozen_layers_zero():
    """num_unfrozen_layers=0일 때 기존 동작 유지"""

def test_num_unfrozen_layers_four():
    """num_unfrozen_layers=4일 때 마지막 4개 블록 + norm + value_head만 trainable"""

def test_num_unfrozen_layers_exceeds_total():
    """num_unfrozen_layers > n_layers일 때 전체 학습"""
```

### 통합 테스트
```bash
# 로컬 micro 모델로 테스트
PYTHONPATH=src python -m weighted_mtp.pipelines.run_critic \
  --config configs/critic/critic_local.yaml \
  --override training.num_unfrozen_layers=2
```

### 검증 항목
1. Trainable params 수가 예상과 일치하는지
2. Forward/backward 그래프가 정상 동작하는지
3. value/std가 증가하는지 (상수 예측 탈출)
4. FSDP 분산 학습에서 gradient sync 정상인지

---

## 구현 우선순위

1. **필수**: Phase 1, 2 (Config + Freeze 로직)
2. **필수**: Phase 5 일부 (Trainable params 로깅)
3. **선택**: Phase 5 전체 (상세 메트릭)

---

## 예상 결과

| 지표 | Before (value head만) | After (4 blocks + value head) |
|------|----------------------|-------------------------------|
| `value/std` | 0.084 | > 0.2 (예상) |
| `value/mse` | 0.2578 | < 0.22 (예상) |
| 수렴 속도 | 느림 | 개선 |

**근거**: Trunk 상위 레이어가 reward 신호에 맞게 representation을 조정하여, value head가 더 discriminative한 feature를 받게 됨.

---

## 롤백 계획

문제 발생 시:
```yaml
training:
  num_unfrozen_layers: 0  # 기존 동작으로 복원
```

기존 코드와 완전 호환 유지됨.
