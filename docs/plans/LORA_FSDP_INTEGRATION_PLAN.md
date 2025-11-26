# LoRA + FSDP 통합 개발 계획서

## 목표

기존 Meta LLaMA MTP 파이프라인의 trunk 부분(Transformer)에 FSDP와 호환되는 LoRA를 적용하여 메모리 효율적인 fine-tuning 지원

---

## 현재 구조 분석

### 모델 구조 (Pure PyTorch)

```
MetaLlamaMTPAdapter
├── transformer: Transformer
│   ├── tok_embeddings: nn.Embedding (vocab_size, dim)
│   ├── layers: ModuleList[TransformerBlock] (trunk layers)
│   │   └── TransformerBlock
│   │       ├── attention: Attention
│   │       │   ├── wq: nn.Linear (dim, n_heads * head_dim)  <- LoRA 타겟
│   │       │   ├── wk: nn.Linear (dim, n_kv_heads * head_dim)  <- LoRA 타겟
│   │       │   ├── wv: nn.Linear (dim, n_kv_heads * head_dim)  <- LoRA 타겟
│   │       │   └── wo: nn.Linear (n_heads * head_dim, dim)  <- LoRA 타겟
│   │       ├── feed_forward: FeedForward
│   │       │   ├── w1: nn.Linear (dim, hidden_dim)  <- LoRA 타겟 (선택)
│   │       │   ├── w2: nn.Linear (hidden_dim, dim)  <- LoRA 타겟 (선택)
│   │       │   └── w3: nn.Linear (dim, hidden_dim)  <- LoRA 타겟 (선택)
│   │       ├── attention_norm: RMSNorm
│   │       └── ffn_norm: RMSNorm
│   ├── extra_heads: ModuleList[TransformerBlock] (MTP heads)
│   ├── norm: RMSNorm
│   └── output: nn.Linear (dim, vocab_size)
└── value_head: ValueHead (Critic/Verifiable용)
```

### FSDP Wrapping 구조

```python
# fsdp.py: wrap_model_fsdp()
- TransformerBlock 단위 auto wrapping
- use_orig_params=True (원본 parameter 구조 유지)
- Mixed precision (BFloat16)
- Activation checkpointing 지원
```

### 파이프라인별 학습 파라미터

| Pipeline | Trunk | Value Head | MTP Heads |
|----------|-------|------------|-----------|
| Baseline | 전체 학습 | 없음 | 학습 |
| Critic | Frozen (또는 일부) | 학습 | Frozen |
| Verifiable | 학습 | 학습 | 학습 |
| Rho-1 | 전체 학습 | 없음 | 학습 |

---

## LoRA 적용 설계

### 핵심 결정사항

1. **Custom LoRA 구현** (Pure PyTorch)
   - 현재 프로젝트가 HuggingFace Transformers 대신 순수 PyTorch Transformer 사용
   - PEFT 라이브러리 직접 사용 불가
   - Custom `LoRALinear` 클래스 구현 필요

2. **FSDP 호환성 확보**
   - LoRA 레이어도 TransformerBlock 내부에 위치하여 함께 wrapping
   - `use_orig_params=True` 유지
   - LoRA 파라미터만 `requires_grad=True` 설정

3. **타겟 모듈 (기본 설정)**
   - Attention: `wq`, `wk`, `wv`, `wo`
   - FFN 제외 (메모리 효율 우선)

### LoRALinear 클래스 설계

```python
class LoRALinear(nn.Module):
    """LoRA가 적용된 Linear 레이어
    
    W' = W + BA 형태로 low-rank adaptation 적용
    - W: 원본 가중치 (frozen)
    - B: [out_features, rank] 행렬 (학습)
    - A: [rank, in_features] 행렬 (학습)
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        # original_linear의 가중치는 frozen
        # lora_A, lora_B는 학습 대상
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h = x @ W^T + x @ A^T @ B^T * (alpha / rank)
        pass
```

---

## Phase별 개발 계획

### Phase 1: LoRA 기반 모듈 구현

**목표**: FSDP 호환 LoRA 레이어 및 유틸리티 구현

**작업 내용**:

1. `src/weighted_mtp/models/lora/lora_linear.py` 생성
   - `LoRALinear` 클래스 구현
   - Kaiming/Xavier 초기화 (A: Kaiming uniform, B: zeros)
   - dropout 지원
   - 원본 가중치 frozen 처리

2. `src/weighted_mtp/models/lora/lora_utils.py` 생성
   - `apply_lora_to_model()`: 모델에 LoRA 적용
   - `get_lora_parameters()`: LoRA 파라미터만 추출
   - `merge_lora_weights()`: LoRA를 원본에 병합 (inference용)
   - `save_lora_checkpoint()`: LoRA 가중치만 저장
   - `load_lora_checkpoint()`: LoRA 가중치 로드

3. `src/weighted_mtp/models/lora/__init__.py` 생성
   - 모듈 export

**예상 파일 구조**:
```
src/weighted_mtp/models/
├── lora/
│   ├── __init__.py
│   ├── lora_linear.py      # LoRALinear 클래스
│   └── lora_utils.py       # 적용/저장/로드 유틸리티
└── meta_mtp/
    ├── adapter.py          # 수정: LoRA 지원 추가
    └── ...
```

**검증 기준**:
- LoRALinear이 nn.Linear와 동일한 출력 생성 (LoRA 초기화 시)
- requires_grad가 LoRA 파라미터에만 True
- FSDP wrapping 후 정상 forward/backward

---

### Phase 2: Adapter LoRA 통합

**목표**: MetaLlamaMTPAdapter에 LoRA 적용 기능 통합

**작업 내용**:

1. `adapter.py` 수정
   - `from_pretrained()` 시그니처 확장:
     ```python
     def from_pretrained(
         cls,
         model_path: str,
         device: str = "auto",
         dtype: Optional[str] = None,
         initialize_value_head: bool = True,
         value_head_type: str = "mlp",
         dropout: float = 0.0,
         # LoRA 관련 인자 추가
         use_lora: bool = False,
         lora_config: Optional[dict] = None,
     ) -> "MetaLlamaMTPAdapter":
     ```
   - LoRA 적용 시점: 모델 로드 후, FSDP wrapping 전
   - 원본 trunk 파라미터 frozen 처리

2. `_from_checkpoint()` 수정
   - LoRA checkpoint 로드 지원
   - state_dict에서 LoRA 가중치 분리 처리

3. 새 메서드 추가:
   ```python
   def get_trainable_parameters(self) -> list[nn.Parameter]:
       """학습 대상 파라미터만 반환 (LoRA 또는 전체)"""
   
   def merge_lora(self) -> None:
       """LoRA 가중치를 원본에 병합 (inference 최적화)"""
   ```

**검증 기준**:
- use_lora=True 시 trainable params 수 대폭 감소
- LoRA 적용 후 forward 출력 shape 동일
- checkpoint 저장/로드 정상 동작

---

### Phase 3: 파이프라인 통합

**목표**: 모든 파이프라인에서 LoRA 학습 지원

**작업 내용**:

1. Config 스키마 확장 (모든 파이프라인)
   ```yaml
   training:
     use_lora: false  # LoRA 사용 여부
     lora:
       rank: 8
       alpha: 16
       dropout: 0.0
       target_modules:
         - "wq"
         - "wk"
         - "wv"
         - "wo"
   ```

2. `run_baseline.py` 수정
   - LoRA config 파싱
   - `load_adapter()` 호출 시 LoRA 인자 전달
   - Optimizer에 LoRA 파라미터만 전달 (use_lora=True 시)

3. `run_critic.py` 수정
   - LoRA + Value head 조합 지원
   - trunk frozen 로직과 LoRA 통합

4. `run_verifiable.py` 수정
   - LoRA + Value head 동시 학습 지원
   - param_groups에 LoRA 그룹 추가

5. `run_rho1.py` 수정
   - LoRA 적용 후 reference 모델과 비교

**검증 기준**:
- 각 파이프라인 config에서 use_lora=true 동작
- 메모리 사용량 감소 확인
- 학습 수렴 검증

---

### Phase 4: FSDP 통합 검증

**목표**: FSDP 분산학습 환경에서 LoRA 정상 동작 검증

**작업 내용**:

1. `fsdp.py` 수정 (필요시)
   - LoRALinear가 TransformerBlock 내부에 있으므로 추가 수정 최소화
   - ignored_modules 처리 검토 (필요시)

2. Checkpoint 저장/로드 검증
   - FSDP state_dict에서 LoRA 가중치 정상 포함 확인
   - `save_checkpoint()` 수정 (LoRA 전용 저장 옵션)

3. 분산학습 테스트
   - 4-GPU torchrun 환경 테스트
   - All-reduce 동기화 검증
   - 메모리 프로파일링

**검증 기준**:
- torchrun --nproc_per_node=4 정상 동작
- Checkpoint 저장 후 재로드 시 동일 결과
- gradient all-reduce 정상 동작

---

### Phase 5: 테스트 및 문서화

**목표**: 통합 테스트 및 사용 가이드 작성

**작업 내용**:

1. 단위 테스트 추가
   - `tests/unit/test_lora_linear.py`
   - `tests/unit/test_lora_adapter.py`

2. 통합 테스트 추가
   - `tests/integration/test_pipeline_lora.py`

3. Config 예시 추가
   - `configs/baseline/baseline_lora.yaml`
   - `configs/verifiable/verifiable_lora.yaml`

4. README 업데이트
   - LoRA 사용법 섹션 추가
   - 권장 하이퍼파라미터 가이드

**검증 기준**:
- 모든 테스트 통과
- 문서 완성도

---

## 의존성 관리

### 필요 패키지

현재 `pyproject.toml`에 추가 의존성 불필요 (Pure PyTorch 구현)

### 검증 명령

```bash
# 의존성 설치 확인
uv sync

# 단위 테스트
pytest tests/unit/test_lora*.py -v

# 통합 테스트 (단일 GPU)
pytest tests/integration/test_pipeline_lora.py -v

# 분산 테스트 (4-GPU)
torchrun --nproc_per_node=4 -m pytest tests/integration/test_pipeline_lora.py -v -k ddp
```

---

## 예상 메모리 절감

| 모델 (6.7B) | Full Fine-tuning | LoRA (rank=8) | 절감률 |
|-------------|------------------|---------------|--------|
| Trainable Params | 6.7B | ~6.7M | 99.9% |
| Optimizer States | ~54GB | ~54MB | 99.9% |
| Gradient Memory | ~27GB | ~27MB | 99.9% |
| **Total (4-GPU)** | ~90GB/GPU | ~30GB/GPU | ~67% |

---

## 일정 추정

| Phase | 예상 소요 | 우선순위 |
|-------|----------|----------|
| Phase 1 | 1일 | 높음 |
| Phase 2 | 1일 | 높음 |
| Phase 3 | 2일 | 높음 |
| Phase 4 | 1일 | 중간 |
| Phase 5 | 1일 | 중간 |
| **Total** | **6일** | - |

---

## 리스크 및 대응

### 1. FSDP + LoRA 호환성 문제

**리스크**: FSDP sharding 시 LoRA 파라미터 동기화 문제

**대응**: 
- `use_orig_params=True` 유지
- LoRA 모듈이 TransformerBlock 내부에 위치하여 함께 sharding

### 2. Checkpoint 호환성

**리스크**: 기존 checkpoint와 LoRA checkpoint 혼용 시 충돌

**대응**:
- checkpoint 포맷 버전 관리
- LoRA 가중치 별도 prefix (`lora_`) 사용

### 3. 성능 저하

**리스크**: LoRA 추가 연산으로 인한 속도 저하

**대응**:
- inference 시 `merge_lora()` 호출하여 원본에 병합
- 학습 시에만 분리된 LoRA 연산 사용

---

## 승인 요청 사항

1. **Phase 1 진행 승인**: LoRA 기반 모듈 신규 생성
2. **타겟 모듈 범위**: Attention만 vs Attention+FFN
3. **기본 하이퍼파라미터**: rank=8, alpha=16

승인 후 Phase 1부터 순차 진행 예정입니다.

