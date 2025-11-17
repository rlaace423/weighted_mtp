# storage/ 자산 재구성 계획

`00_ideal_structure.md`에서 정의한 목표 아키텍처를 구현하기 위해, 기존 `storage/` 디렉터리를 최신 PPO/RLHF 베스트 프랙티스와 Pure PyTorch 파이프라인에 맞추어 재구성한다. 본 문서는 데이터/모델 자산을 어떤 형태로 변환하고 저장해야 하는지에 대한 구체적 작업 지침을 제공한다.

---

## 1. 목표 및 원칙

1. **단일 진실 소스**: `storage/`는 로컬 준비 → VESSL Storage 업로드 전에 반드시 거치는 staging 영역으로 사용한다.
2. **명확한 버전 관리**: 모델/데이터셋 모두 `v{major}.{minor}` 폴더를 두고, 변경 시 새 버전 디렉터리를 생성한다.
3. **Pure PyTorch 구현**: Meta 아키텍처를 참고하되, Pure PyTorch로 재구현(`src/models/meta_mtp/`)하여 fairscale 의존성 제거, FSDP 완전 호환, safetensors 저장 지원을 확보한다. Meta vendor 코드(`vendor/meta_llama/`)는 참고용으로만 유지하며 실제 학습에 사용하지 않는다. `storage/`에는 순수 weight/config/tokenizer/adapter 설정만 둔다.
4. **Safetensors 호환성**: RoPE freqs_cis는 complex64 타입(safetensors 미지원)이므로 `register_buffer` 대신 일반 속성으로 저장하여 state_dict에서 제외한다. Runtime에 자동 계산하여 device 이동 처리.
5. **TD error 기반 가중치 호환성**: value 추정, TD error 정규화, 토큰 가중치 계산에 필요한 메타데이터(토크나이저 ID, sequence length, reward/label 필드 등)를 JSON 스키마로 명시한다.
6. **로컬 ↔ VESSL 가시성**: 업로드 스크립트(`scripts/sync_to_vessl_storage.py`)가 그대로 사용하는 경로 규약을 정의하고, 테스트용 경량 자산을 동일 구조에 포함한다.
7. **A100 4-GPU 분산학습 호환성**: 모델 체크포인트는 FSDP(Fully Sharded Data Parallel) 형식을 지원하며, 데이터셋은 DistributedSampler를 통한 분산 로딩을 전제로 준비한다.

---

## 2. 타깃 디렉터리 구조 (요약)

```
storage/
├── datasets_v2/
│   ├── codecontests/
│   │   ├── raw/            # huggingface 원본(JSONL)
│   │   ├── processed/      # Alpaca 스타일 SFT JSONL
│   │   └── stats/          # 길이·정확도 통계
│   ├── mbpp/
│   └── humaneval/
├── models_v2/
│   ├── meta-llama-mtp/
│   │   ├── raw/            # 7B_1T_4/consolidated.pth, params.json
│   │   ├── safetensors/    # 변환된 model.safetensors, value_head.safetensors(옵션)
│   │   ├── tokenizer/      # tokenizer.model, tokenizer_config.json
│   │   ├── configs/        # params.json 복사본, meta_adapter.yaml
│   │   └── metadata.json
│   ├── ref-sheared-llama-2.7b/
│   ├── starling-rm-7b/     # 옵션
│   └── micro-mtp/
└── README.md
```

---

## 3. 모델 자산 준비 지침

### 3.1 Meta LLaMA MTP (Base, Pure PyTorch 구현)

#### Meta 원본 다운로드 (참고용)
1. **원본 번들 다운로드 (raw/)**
   ```bash
   hf download facebook/multi-token-prediction 7B_1T_4/consolidated.pth \
     --local-dir storage/models_v2/meta-llama-mtp/raw
   hf download facebook/multi-token-prediction 7B_1T_4/params.json \
     --local-dir storage/models_v2/meta-llama-mtp/raw
   hf download facebook/multi-token-prediction tokenizer.model \
     --local-dir storage/models_v2/meta-llama-mtp/tokenizer
   ```
   - `llama/*.py`는 **참고용으로만** `vendor/meta_llama/`에 복사 (아키텍처 검증용)
   - **실제 학습에는 사용하지 않음** (fairscale 의존성, @inference_mode 등 문제)

#### Pure PyTorch 재구현 기반 변환
2. **Pure PyTorch Transformer 생성 및 safetensors 저장**
   ```bash
   # Pure PyTorch Transformer로 재생성
   python - <<'PY'
   import json
   import torch
   from safetensors.torch import save_file
   from weighted_mtp.models.meta_mtp import ModelArgs, Transformer

   # params.json 읽기
   with open("storage/models_v2/meta-llama-mtp/raw/7B_1T_4/params.json") as f:
       params = json.load(f)

   # ModelArgs 생성
   model_args = ModelArgs(
       dim=params["dim"],
       n_layers=params["n_layers"],
       n_heads=params["n_heads"],
       n_kv_heads=params.get("n_kv_heads"),
       vocab_size=params["vocab_size"],
       n_future_tokens=params.get("n_future_tokens", 4),
       rope_theta=params.get("rope_theta", 10000.0),
       max_seq_len=params.get("max_position_embeddings", 2048),
       norm_eps=params.get("rms_norm_eps", 1e-5),
   )

   # Pure PyTorch Transformer 생성
   transformer = Transformer(model_args)

   # Meta 원본 weights 로드 (선택적 - 있는 경우)
   # state_dict = torch.load("storage/models_v2/meta-llama-mtp/raw/7B_1T_4/consolidated.pth", map_location="cpu")
   # transformer.load_state_dict(state_dict, strict=True)

   # Safetensors 저장 (freqs_cis는 자동 제외됨)
   save_file(
       transformer.state_dict(),
       "storage/models_v2/meta-llama-mtp/safetensors/model.safetensors",
       metadata={"dtype": "float16"}
   )
   PY

   # params.json 복사
   cp storage/models_v2/meta-llama-mtp/raw/7B_1T_4/params.json \
      storage/models_v2/meta-llama-mtp/configs/params.json
   ```

   **핵심 구현 원칙**:
   - **Pure PyTorch 구조**: `nn.Embedding`, `nn.Linear` 사용 (fairscale 제거)
   - **Gradient 계산 가능**: `@torch.inference_mode()` 제거
   - **Device-agnostic**: cuda/mps/cpu 자동 지원
   - **RoPE freqs_cis 처리**:
     ```python
     # ✅ 현재 구현 (safetensors 호환)
     # transformer.py에서
     self.freqs_cis = precompute_freqs_cis(...)  # 일반 속성 (state_dict 미포함)

     def forward(self, tokens):
         # Runtime에 명시적 device 이동
         freqs_cis = self.freqs_cis[0:seqlen].to(tokens.device)
     ```
   - **효과**:
     - ✅ Safetensors 저장/로딩 가능
     - ✅ FSDP checkpoint 저장 가능
     - ✅ State dict 크기 감소 (~256KB 절감)
     - ✅ HuggingFace Hub 배포 가능

3. **Adapter 설정 및 Value Head**
   - `configs/meta_adapter.yaml`: params.json과 동일 값 사용 (`n_future_tokens=4`, `intermediate_size=11008`, `rope_theta=10000.0`, `dtype=float16`)
   - Stage1 value head 생성 시: `safetensors/value_head.safetensors` 저장, `metadata.json.value_head` 갱신

4. **metadata.json 작성 (구조 예시)**
   ```json
   {
     "model_name": "meta-llama-mtp",
     "version": "v2.0.0",
     "implementation": "pure_pytorch",
     "n_params": 6.7e9,
     "format": "safetensors",
     "dtype": "float16",
     "files": {
       "checkpoint": "safetensors/model.safetensors",
       "tokenizer": "tokenizer/tokenizer.model",
       "params": "configs/params.json",
       "adapter": "configs/meta_adapter.yaml"
     },
     "value_head": {
       "path": null,
       "stage1_checkpoint_id": null
     },
     "tokenizer_shared_with": null,
     "source": {
       "architecture": "Meta LLaMA MTP (Pure PyTorch reimplementation)",
       "original_repo": "facebook/multi-token-prediction",
       "original_revision": "7B_1T_4",
       "note": "vendor/meta_llama/ 참고용, 실제 구현은 src/models/meta_mtp/"
     },
     "safetensors": {
       "freqs_cis_handling": "runtime_computed",
       "note": "freqs_cis는 complex64 타입(safetensors 미지원)으로 state_dict 미포함, forward 시 자동 계산"
     },
     "distributed": {
       "fsdp_compatible": true,
       "note": "단일 safetensors 파일로 저장, FSDP가 런타임에 4-GPU로 자동 샤딩"
     }
   }
   ```
   - **Pure PyTorch 구현**: `implementation: "pure_pytorch"` 필드로 명시
   - **Safetensors 호환성**: `freqs_cis_handling: "runtime_computed"` 필드로 freqs_cis 처리 방식 명시
   - **분산학습 참고**: 초기 체크포인트는 단일 `model.safetensors` 파일로 저장하며, FSDP가 학습 시작 시 4개 GPU로 파라미터를 자동 분산한다.
   - **체크포인트 저장**: 학습 중 저장되는 체크포인트는 FSDP `state_dict`를 통해 통합된 단일 파일로 저장 (Rank 0만 수행).

5. **검증 포인트**
   - Pure PyTorch Transformer 생성 성공 (fairscale 의존성 없음)
   - Forward pass shape 정확: `[batch, seq, n_future_tokens, vocab]`
   - Gradient 계산 가능 확인 (`@torch.inference_mode()` 제거됨)
   - `model.safetensors` dtype이 float16인지 확인
   - Safetensors 저장/로딩 정상 (freqs_cis 자동 제외, runtime 계산)
   - Device 이동 정상 (cuda/mps/cpu)
   - FSDP wrapping 가능
   - `params.json` ↔ `meta_adapter.yaml` ↔ `metadata.json` 간 dim/heads/rope 값 일치
   - SHA256 해시를 `safetensors/SHA256SUMS`에 기록

### 3.2 Reference / Reward 모델
- Rho-1 reference: Hugging Face에서 sharded PyTorch `.bin`을 다운로드한 뒤 safetensors로 병합, Base 토크나이저와 동일함을 `metadata.json.tokenizer_shared_with="meta-llama-mtp"`로 기록.
- Reward 모델(선택): 동일 절차로 safetensors 변환. 미사용 시 `metadata.json.status="optional"`로 표기하고 빈 디렉터리 유지.

### 3.3 Micro MTP (로컬 테스트, Pure PyTorch)
- `scripts/regenerate_micro_model.py` 실행으로 Pure PyTorch 구조로 경량 모델 생성
  - 4-layer/512-dim 버전
  - Trunk + Extra heads 분리 구조 (n_layers=4, n_future_tokens=4 → layers 1개 + extra_heads 3개)
- 산출물: `safetensors/model.safetensors` (freqs_cis 제외), `configs/config.json`, `tokenizer/`, `metadata.json(target_device="mps")`
- 검증: `uv run pytest tests/unit/test_adapter.py -k micro` 통과 (11/11 tests)

### 3.4 모델 로딩 아키텍처
**통합 로딩 메커니즘**: `MetaLlamaMTPAdapter.from_pretrained()`

모든 파이프라인은 `MetaLlamaMTPAdapter.from_pretrained()` classmethod를 통해 모델을 로딩한다.

**Signature**:
```python
MetaLlamaMTPAdapter.from_pretrained(
    model_path: str,
    device: str = "auto",
    dtype: Optional[str] = None,
    initialize_value_head: bool = True,
) -> MetaLlamaMTPAdapter
```

**로딩 절차**:
1. **Transformer 로딩**: `checkpoints.load_meta_mtp_model()` 호출
   - safetensors 파일 로드
   - Dtype 변환 (dtype 인자 또는 safetensors 기본값)
   - Device 이동 (auto/cuda/mps/cpu)
2. **ModelArgs 파싱**: params.json 또는 config.json 자동 감지
   - params.json 우선 (Meta 표준)
   - config.json fallback (HuggingFace 호환)
3. **Value Head 선택적 초기화**:
   - `initialize_value_head=True` (기본값): Critic/Verifiable Stage용
     - ValueHead 생성 및 device/dtype 일치
   - `initialize_value_head=False`: Rho-1 Stage용
     - Value Head 불필요 (MTP trunk만 사용)

**Tokenizer 공유 구조**:
- Policy 모델과 Reference 모델은 동일한 tokenizer 사용
- Tokenizer 위치: `{model_path}/tokenizer/`
- 구성 파일:
  - `tokenizer.model` (SentencePiece 모델)
  - `tokenizer_config.json` (HuggingFace 설정)
- Reference 모델도 동일 경로에서 tokenizer 로드 (중복 저장 불필요)

**Reference 모델 전략** (Rho-1 전용):
- HuggingFace `AutoModelForCausalLM.from_pretrained()` 직접 사용
- Custom wrapper 불필요 (표준 인터페이스로 충분)
- Tokenizer는 policy 모델과 공유

---

## 4. 데이터셋 준비 지침

### 4.1 공통 프로세스
1. **원본 다운로드**
   - HuggingFace datasets 라이브러리를 사용해 Parquet 형식으로 직접 로드한다.
   - CodeContests는 `name`, `description`, `public_tests`, `solutions`, `incorrect_solutions` 등의 필드를 포함한다.
   - raw/ 디렉터리는 선택 사항 (HF에서 직접 로드하므로 중간 JSONL 저장 불필요).
2. **전처리 & 변환** (`scripts/setup_datasets.py`)
   - HuggingFace dataset → Alpaca 형식 JSONL 변환
   - 작업 내용:
     - `description` → `instruction`, `public_tests` → `input` (최대 2개 예시)
     - `solutions` (correct) 및 `incorrect_solutions` 모두 처리
     - 각 솔루션마다 별도 레코드 생성: `task_id`에 `_correct_N` 또는 `_incorrect_N` 접미사 추가
     - **top-level `is_correct` 필드** 추가 (`true` 또는 `false`)
     - 길이 필터: instruction + input + output 합산이 2048 토큰 초과 시 제외
     - Python/Python3 솔루션만 포함 (언어 코드 1 또는 3)
   - train/valid/test split은 HuggingFace 기본 split 사용
3. **메타데이터 추출** (`scripts/extract_metadata.py`)
   - **목적**: 메모리 효율적 학습을 위해 `is_correct`, `difficulty` 정보만 별도 파일로 추출
   - **입력**: `processed/*.jsonl` (전체 데이터셋)
   - **출력**: `processed/*_metadata.json`
   - **구조**:
     ```json
     {
       "metadata": [
         {"is_correct": true, "difficulty": 7},
         {"is_correct": false, "difficulty": 2},
         ...
       ],
       "stats": {
         "total": 3691981,
         "correct": 1754404,
         "incorrect": 1937577,
         "difficulty_dist": {"0": 1519213, "1": 2701, ...}
       }
     }
     ```
   - **크기**: 전체 데이터(~15GB) 대비 ~217MB (99% 메모리 절감)
   - **실행**: `python scripts/extract_metadata.py --dataset codecontests --split train`
4. **통계/무결성 기록**
   - `stats/YYYY-MM-DD_summary.json`에 샘플 수, 평균 토큰 길이(`instruction`, `input`, `output`), `is_correct` 분포, 최대 길이 등을 기록.
   - `scripts/validate_datasets.py`로 schema 검증, 2048 토큰 초과 여부 검사, SHA256 로그를 수행한다.
5. **메모리 효율 학습 워크플로우**
   - **메타데이터 기반 샘플링** (핵심 혁신):
     1. 전체 데이터셋(3.7M, ~15GB)을 메모리에 로드하지 **않음**
     2. 메타데이터 파일(`*_metadata.json`, ~217MB)만 로드
     3. Config 기반으로 필요한 샘플 인덱스 계산 (Stage별 전략 적용)
     4. JSONL 파일에서 계산된 인덱스의 라인만 선택적으로 읽기
     5. HuggingFace Dataset으로 변환
   - **메모리 절감 효과**:
     - Stage 1 (50K 샘플): 메타데이터(~217MB) + 샘플(~200MB) = **~417MB** (기존 15GB 대비 97% 절감)
     - Stage 2 (200K 샘플): 메타데이터(~217MB) + 샘플(~800MB) = **~1GB** (기존 15GB 대비 93% 절감)
   - **분산학습 호환성**:
     - 메타데이터 기반으로 선택된 샘플들을 `DistributedSampler`가 4개 GPU로 자동 분할
     - Rank 0: `samples[0::4]`, Rank 1: `samples[1::4]`, Rank 2: `samples[2::4]`, Rank 3: `samples[3::4]`
     - 각 GPU는 전체의 1/4만 처리 (중복 없음)
     - Epoch 재현성: `sampler.set_epoch(epoch)` 호출

### 4.2 processed `schema.json` 예시
```json
{
  "dataset": "codecontests",
  "format": "alpaca",
  "required_fields": ["instruction", "input", "output", "task_id", "is_correct"],
  "optional_fields": ["metadata"],
  "field_types": {
    "instruction": "string",
    "input": "string",
    "output": "string",
    "task_id": "string",
    "is_correct": "boolean",
    "metadata": {
      "source": "string",
      "difficulty": "integer",
      "has_tests": "boolean"
    }
  },
  "description": "codecontests dataset in Alpaca format with correct and incorrect solutions",
  "source": "deepmind/code_contests"
}
```

### 4.3 전처리 실행 예시
```bash
# 전체 데이터셋 일괄 처리 (다운로드 + 변환 + 메타데이터 추출 + small + stats)
uv run python scripts/setup_datasets.py --datasets all --steps all

# 개별 데이터셋 처리
uv run python scripts/setup_datasets.py --datasets codecontests --steps all
uv run python scripts/setup_datasets.py --datasets mbpp --steps all
uv run python scripts/setup_datasets.py --datasets humaneval --steps all

# 단계별 실행
uv run python scripts/setup_datasets.py --datasets codecontests --steps process
uv run python scripts/setup_datasets.py --datasets codecontests --steps metadata  # 메타데이터 추출
uv run python scripts/setup_datasets.py --datasets codecontests --steps small,stats

# 메타데이터만 별도 추출
uv run python scripts/extract_metadata.py --dataset codecontests --split train
uv run python scripts/extract_metadata.py --dataset codecontests --split valid
uv run python scripts/extract_metadata.py --dataset codecontests --split test
```
> `scripts/setup_datasets.py`가 HuggingFace에서 직접 로드하여 processed, metadata, small, stats를 모두 생성한다.

---

## 5. 변환 작업 절차

### 5.1 모델 (Pure PyTorch 기반)
1. **원본 확보**: Hugging Face에서 `consolidated.pth`, `params.json`, `tokenizer.model`을 `models_v2/<model>/raw/`로 다운로드한다.
   - `vendor/meta_llama/`의 레퍼런스 코드는 **참고용**으로만 유지 (아키텍처 검증)
2. **Pure PyTorch Transformer 생성 및 파생물**
   - Pure PyTorch Transformer 생성 (`src/models/meta_mtp/`)
   - Safetensors 저장 (float16 유지, **freqs_cis 자동 제외**) → `safetensors/model.safetensors`
   - `params.json` 복사 → `configs/params.json`
   - `meta_adapter.yaml` 작성 (dim=4096, n_layers=32, n_heads=32, intermediate_size=11008, rope_theta=10000.0, n_future_tokens=4, dtype=float16)
   - Stage1 결과가 있으면 `safetensors/value_head.safetensors` 추가
3. **메타데이터 작성**
   - `metadata.json`에 `implementation: "pure_pytorch"`, `freqs_cis_handling: "runtime_computed"` 필드 추가
   - dtype, 원본 리포지토리, SHA256, tokenizer 공유 여부 기록
   - `safetensors/SHA256SUMS`에 해시 저장
4. **검증**
   - Pure PyTorch Transformer 생성 및 forward pass 정상
   - Gradient 계산 가능 확인 (`@torch.inference_mode()` 제거됨)
   - Safetensors 저장/로딩 정상 (freqs_cis runtime 계산)
   - Device 이동 정상 (cuda/mps/cpu)
   - FSDP wrapping 가능
   - 파라미터 일치 검증 (params.json ↔ meta_adapter ↔ metadata)
   - Unit tests 11/11 통과

### 5.2 데이터셋
1. **raw 정리**: CodeContests 등 원본 JSONL을 `datasets_v2/<name>/raw/`로 이동하고 SHA256 기록.
2. **전처리 실행**: `src/data/prepare.py`를 돌려 processed train/validation/test, stats, schema를 생성한다.
3. **검증 & 메타데이터 추출**
   - `scripts/validate_datasets.py`로 스키마, 길이, 중복을 검사.
   - `scripts/extract_metadata.py`로 메타데이터 파일 생성 (is_correct, difficulty만 추출, 99% 메모리 절감)
   - README에 생성 일자와 SHA256을 갱신한다.

### 5.3 README 갱신
`storage/README.md`에 다음 항목을 명시한다.
- 디렉터리 레이아웃 요약
- 버전 관리 규칙
- 업로드 스크립트 실행 방법
- 무결성 체크 명령 (`sha256sum`)
- 자주 묻는 질문(예: value head가 없을 때 처리 방법)

---

## 6. VESSL 업로드 전략

- `scripts/sync_to_vessl_storage.py`에서 다음 규약을 따른다.
  - 기본 업로드 루트: `vessl://weighted-mtp/{models,datasets}/{version}/...`
  - 업로드 전 SHA256 비교로 중복 전송 방지.
  - Micro 모델과 small 데이터셋은 `--include-local-small` 플래그로 선택 업로드.
- 업로드 후 VESSL CLI에서 디렉터리 구조를 검증:
  ```bash
  vessl storage ls weighted-mtp/models_v2/meta-llama-mtp
  ```

---

## 7. 검증 체크리스트

### 모델 (Pure PyTorch)
- [ ] Pure PyTorch Transformer 생성 성공 (fairscale 의존성 없음)
- [ ] Forward pass shape 정확: `[batch, seq, n_future_tokens, vocab]`
- [ ] Gradient 계산 가능 (`@torch.inference_mode()` 제거됨)
- [ ] Safetensors 저장/로딩 정상 (freqs_cis 자동 제외, runtime 계산)
- [ ] Device 이동 정상 (cuda/mps/cpu)
- [ ] FSDP wrapping 가능
- [ ] `meta_adapter.yaml`의 `n_future_tokens`가 recipe horizon과 일치
- [ ] `meta_adapter.yaml`의 `intermediate_size`, `rope_theta`, `dtype`가 Meta 레퍼런스(11008, 10000.0, float16)와 일치
- [ ] `tokenizer.model`과 Reference/Reward 모델 토크나이저의 vocab size가 동일
- [ ] `model.safetensors`의 dtype(float16)과 SHA256 해시 기록 완료
- [ ] (선택) value head 존재 시 shape: `[hidden_size, 1]` 확인
- [ ] `metadata.json`에 `implementation: "pure_pytorch"`, `freqs_cis_handling: "runtime_computed"`, `distributed.fsdp_compatible: true` 필드 포함 확인
- [ ] Unit tests 11/11 통과 (`pytest tests/unit/test_adapter.py`)

### 데이터
- [ ] 각 processed JSONL이 2048 토큰 이하 데이터만 포함한다.
- [ ] `is_correct` 필드가 없는 샘플이 있는 경우 Verifiable 실험에서 제외하도록 별도 태깅.
- [ ] `schema.json`과 실제 샘플 구조가 일치한다.
- [ ] 통계 파일이 최신 날짜로 업데이트되었다.
- [ ] 전체 데이터셋 샘플 수가 4의 배수가 아니어도 DistributedSampler가 자동 처리하므로 문제없음을 확인.

### 스크립트
- [ ] `scripts/prepare_local_small_model.py` 실행 후 Micro 모델 테스트 (`tests/unit/test_adapter.py`) 통과.
- [ ] `scripts/sync_to_vessl_storage.py --dry-run`으로 업로드 시뮬레이션 성공.

### 분산학습
- [ ] FSDP wrapper가 단일 safetensors 파일을 올바르게 로드하고 4-GPU로 분산하는지 로컬 테스트 수행.
- [ ] DistributedSampler가 데이터를 중복 없이 4개 GPU로 분할하는지 검증 (각 GPU별 샘플 수 확인).
- [ ] Rank 0만 체크포인트 저장 및 MLflow 로깅을 수행하는지 확인.
- [ ] torchrun으로 실행 시 모든 GPU가 동일한 `world_size=4`를 인식하는지 확인.

---

## 8. 후속 작업

1. `00_ideal_structure.md`에 명시된 테스트/체크리스트와 본 계획을 비교하여 누락 항목을 `docs/migration_notes.md`에 기록.
2. 변환 스크립트 작성 후 Git에 추가 (`src/data/prepare.py`, `scripts/*`).
3. 실제 변환을 수행하고, 변경된 자산을 VESSL Storage에 업로드.
4. 파이프라인에서 새 경로(`models_v2`, `datasets_v2`)를 사용하도록 Config 업데이트.

이 문서는 변환 실행 전 리뷰 및 승인용으로 사용하며, 실행 중 발견되는 이슈는 문서 하단에 주석으로 추가 업데이트한다.

