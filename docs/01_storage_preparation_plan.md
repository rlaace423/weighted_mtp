# storage/ 자산 재구성 계획

`00_ideal_structure.md`에서 정의한 목표 아키텍처를 구현하기 위해, 기존 `storage/` 디렉터리를 최신 PPO/RLHF 베스트 프랙티스와 Meta LLaMA MTP 네이티브 워크플로우에 맞추어 재구성한다. 본 문서는 데이터/모델 자산을 어떤 형태로 변환하고 저장해야 하는지에 대한 구체적 작업 지침을 제공한다.

---

## 1. 목표 및 원칙

1. **단일 진실 소스**: `storage/`는 로컬 준비 → VESSL Storage 업로드 전에 반드시 거치는 staging 영역으로 사용한다.
2. **명확한 버전 관리**: 모델/데이터셋 모두 `v{major}.{minor}` 폴더를 두고, 변경 시 새 버전 디렉터리를 생성한다.
3. **Meta 네이티브 존중**: Meta가 배포한 파이썬 소스는 코드베이스(`src/models/meta_mtp/`)에 두고, `storage/`에는 순수 weight/config/tokenizer/adapter 설정만 둔다.
4. **TD error 기반 가중치 호환성**: value 추정, TD error 정규화, 토큰 가중치 계산에 필요한 메타데이터(토크나이저 ID, sequence length, reward/label 필드 등)를 JSON 스키마로 명시한다.
5. **로컬 ↔ VESSL 가시성**: 업로드 스크립트(`scripts/sync_to_vessl_storage.py`)가 그대로 사용하는 경로 규약을 정의하고, 테스트용 경량 자산을 동일 구조에 포함한다.

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
├── datasets_local_small/   # 소형 검증 세트
│   └── codecontests_small/
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

### 3.1 Meta LLaMA MTP (Base)
1. **원본 번들 다운로드 (raw/)**
   ```bash
   hf download facebook/multi-token-prediction 7B_1T_4/consolidated.pth \
     --local-dir storage/models_v2/meta-llama-mtp/raw
   hf download facebook/multi-token-prediction 7B_1T_4/params.json \
     --local-dir storage/models_v2/meta-llama-mtp/raw
   hf download facebook/multi-token-prediction tokenizer.model \
     --local-dir storage/models_v2/meta-llama-mtp/tokenizer
   ```
   - `llama/*.py`는 소스 루트의 `vendor/meta_llama/`로 복사하고, `storage/`에는 남겨 두지 않는다.

2. **파생 자산 생성 (safetensors/, configs/)**
   ```bash
   python - <<'PY'
   import torch
   from safetensors.torch import save_file

   state_dict = torch.load("storage/models_v2/meta-llama-mtp/raw/7B_1T_4/consolidated.pth", map_location="cpu")
   save_file(state_dict, "storage/models_v2/meta-llama-mtp/safetensors/model.safetensors", metadata={"dtype": "float16"})
   PY
   cp storage/models_v2/meta-llama-mtp/raw/7B_1T_4/params.json \
      storage/models_v2/meta-llama-mtp/configs/params.json
   ```
   - `configs/meta_adapter.yaml`은 params.json과 동일한 값을 사용해 수동 작성(`n_future_tokens=4`, `intermediate_size=11008`, `rope_theta=10000.0`, `dtype=float16` 등).
   - Stage1 value head가 생성되면 `safetensors/value_head.safetensors`로 저장하고 `metadata.json.value_head`를 갱신한다.

3. **metadata.json 작성 (구조 예시)**
   ```json
   {
     "model_name": "meta-llama-mtp",
     "version": "v2.0.0",
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
       "repo": "facebook/multi-token-prediction",
       "revision": "7B_1T_4"
     }
   }
   ```

4. **검증 포인트**
   - `model.safetensors` dtype이 float16인지 확인 (`safe_open(...).dtype`).
   - `params.json` ↔ `meta_adapter.yaml` ↔ `metadata.json` 간 dim/heads/rope 값이 일치하는지 검증.
   - SHA256 해시를 `safetensors/SHA256SUMS`에 기록하고 README에도 업데이트.

### 3.2 Reference / Reward 모델
- Rho-1 reference: Hugging Face에서 sharded PyTorch `.bin`을 다운로드한 뒤 safetensors로 병합, Base 토크나이저와 동일함을 `metadata.json.tokenizer_shared_with="meta-llama-mtp"`로 기록.
- Reward 모델(선택): 동일 절차로 safetensors 변환. 미사용 시 `metadata.json.status="optional"`로 표기하고 빈 디렉터리 유지.

### 3.3 Micro MTP (로컬 테스트)
- `scripts/prepare_local_small_model.py` 실행으로 Base safetensors에서 4-layer/512-dim 버전을 생성.
- 산출물: `safetensors/model.safetensors`, `configs/config.json`, `tokenizer/`, `metadata.json(target_device="mps")`.
- 전용 테스트: `uv run pytest tests/unit/test_adapter.py -k micro`.

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
3. **통계/무결성 기록**  
   - `stats/YYYY-MM-DD_summary.json`에 샘플 수, 평균 토큰 길이(`instruction`, `input`, `output`), `is_correct` 분포, 최대 길이 등을 기록.  
   - `scripts/validate_datasets.py`로 schema 검증, 2048 토큰 초과 여부 검사, SHA256 로그를 수행한다.
4. **로컬 소형 세트**  
   - `head` 기반으로 `datasets_local_small/<name>_small/{train_small.jsonl,validation_small.jsonl}` 생성(≤100 / ≤32).  
   - CLI에서 `--dataset-suffix small`로 선택 가능하도록 경로를 유지한다.

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
# 전체 데이터셋 일괄 처리 (다운로드 + 변환 + small + stats)
uv run python scripts/setup_datasets.py --datasets all --steps all

# 개별 데이터셋 처리
uv run python scripts/setup_datasets.py --datasets codecontests --steps all
uv run python scripts/setup_datasets.py --datasets mbpp --steps all
uv run python scripts/setup_datasets.py --datasets humaneval --steps all

# 단계별 실행
uv run python scripts/setup_datasets.py --datasets codecontests --steps process
uv run python scripts/setup_datasets.py --datasets codecontests --steps small,stats
```
> `scripts/setup_datasets.py`가 HuggingFace에서 직접 로드하여 processed, small, stats를 모두 생성한다.

---

## 5. 변환 작업 절차

### 5.1 모델
1. **원본 확보**: Hugging Face에서 `consolidated.pth`, `params.json`, `tokenizer.model`을 `models_v2/<model>/raw/`로 다운로드한다.
2. **파생물 생성**  
   - safetensors 변환 (float16 유지) → `safetensors/model.safetensors`.  
   - `params.json` 복사 → `configs/params.json`.  
   - `meta_adapter.yaml` 작성(파라미터 값: dim=4096, n_layers=32, n_heads=32, intermediate_size=11008, rope_theta=10000.0, n_future_tokens=4, dtype=float16).  
   - Stage1 결과가 있으면 `safetensors/value_head.safetensors` 추가.
3. **메타데이터 작성**: `metadata.json`에 dtype, 원본 리포지토리, SHA256, tokenizer 공유 여부를 기록하고 `safetensors/SHA256SUMS`에 해시를 남긴다.
4. **검증**  
   - safetensors dtype 확인, 파라미터 일치 검증( params.json ↔ meta_adapter ↔ metadata ).  
   - `vendor/meta_llama/`의 파이썬 레퍼런스 코드와 버전 주석을 맞춘다.

### 5.2 데이터셋
1. **raw 정리**: CodeContests 등 원본 JSONL을 `datasets_v2/<name>/raw/`로 이동하고 SHA256 기록.
2. **전처리 실행**: `src/data/prepare.py`를 돌려 processed train/validation/test, stats, schema를 생성한다.
3. **검증 & 스몰셋**  
   - `scripts/validate_datasets.py`로 스키마, 길이, 중복을 검사.  
   - `datasets_local_small`에 소형셋 작성.  
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

### 모델
- [ ] `meta_adapter.yaml`의 `n_future_tokens`가 recipe horizon과 일치한다.
- [ ] `meta_adapter.yaml`의 `intermediate_size`, `rope_theta`, `dtype`가 Meta 레퍼런스(11008, 10000.0, float16)와 일치한다.
- [ ] `tokenizer.model`과 Reference/Reward 모델 토크나이저의 vocab size가 동일하다.
- [ ] `model.safetensors`의 dtype(float16)과 SHA256 해시를 기록 완료한다.
- [ ] (선택) value head 존재 시 shape: `[hidden_size, 1]` 확인.

### 데이터
- [ ] 각 processed JSONL이 2048 토큰 이하 데이터만 포함한다.
- [ ] `is_correct` 필드가 없는 샘플이 있는 경우 Verifiable 실험에서 제외하도록 별도 태깅.
- [ ] `schema.json`과 실제 샘플 구조가 일치한다.
- [ ] 통계 파일이 최신 날짜로 업데이트되었다.

### 스크립트
- [ ] `scripts/prepare_local_small_model.py` 실행 후 Micro 모델 테스트 (`tests/unit/test_adapter.py`) 통과.
- [ ] `scripts/sync_to_vessl_storage.py --dry-run`으로 업로드 시뮬레이션 성공.

---

## 8. 후속 작업

1. `00_ideal_structure.md`에 명시된 테스트/체크리스트와 본 계획을 비교하여 누락 항목을 `docs/migration_notes.md`에 기록.
2. 변환 스크립트 작성 후 Git에 추가 (`src/data/prepare.py`, `scripts/*`).
3. 실제 변환을 수행하고, 변경된 자산을 VESSL Storage에 업로드.
4. 파이프라인에서 새 경로(`models_v2`, `datasets_v2`)를 사용하도록 Config 업데이트.

이 문서는 변환 실행 전 리뷰 및 승인용으로 사용하며, 실행 중 발견되는 이슈는 문서 하단에 주석으로 추가 업데이트한다.

