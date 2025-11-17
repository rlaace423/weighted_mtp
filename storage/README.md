# Storage Directory

본 디렉터리는 WMTP 리팩토링 프로젝트의 모든 모델 및 데이터셋 자산을 관리합니다.

## 디렉터리 구조

```
storage/
├── models/                        # v2 형식 모델 (Phase 1 완료)
│   ├── meta-llama-mtp/              # Base model (25GB) ✅
│   ├── ref-sheared-llama-2.7b/      # Reference model (10GB) ✅
│   ├── starling-rm-7b/              # Reward model (25GB, optional) ✅
│   └── micro-mtp/                   # 로컬 테스트용 경량 모델 (50MB) ✅
├── datasets/                      # v2 형식 데이터셋 ✅
│   ├── codecontests/
│   │   ├── raw/                     # 원본 JSONL ✅
│   │   ├── processed/               # 전처리된 데이터 (schema.json 포함) ✅
│   │   └── stats/                   # 통계 파일 ✅
│   ├── mbpp/                        # ✅
│   └── humaneval/                   # ✅
└── datasets_local_small/             # 로컬 테스트용 축소 데이터셋 ✅
    ├── codecontests_small/          # ✅
    ├── mbpp_small/                  # ✅
    └── humaneval_small/             # ✅
```

## 모델 자산 (models)

### meta-llama-mtp
- **크기**: 25GB
- **형식**: safetensors
- **용도**: Base model for all experiments
- **파일**:
  - `safetensors/model.safetensors` - 모델 가중치
  - `safetensors/SHA256SUMS` - 무결성 검증
  - `tokenizer/tokenizer.model` - SentencePiece 토크나이저
  - `configs/config.json` - 모델 설정
  - `configs/meta_adapter.yaml` - Meta MTP adapter 설정
  - `metadata.json` - 버전, SHA256, 파일 경로 정보

### ref-sheared-llama-2.7b
- **크기**: 10GB
- **형식**: safetensors (sharded pytorch에서 변환)
- **용도**: Rho-1 Weighted 실험용 reference model
- **tokenizer**: meta-llama-mtp와 공유 (vocab_size: 32000)

### starling-rm-7b
- **크기**: 25GB
- **형식**: safetensors (pytorch에서 변환)
- **용도**: 선택적 Reward Model (Verifiable Critic 기본 사용 안 함)
- **상태**: optional

## 데이터셋 자산 (datasets)

### codecontests
- **샘플 수**: train 3.2M, validation 15K, test 8K
- **is_correct 필드**: ✅ 있음
- **용도**: 모든 실험 (Baseline, Verifiable Critic, Rho-1 Weighted)
- **특징**: 경쟁 프로그래밍 문제

### mbpp
- **샘플 수**: train 374, validation 90, test 500
- **is_correct 필드**: ❌ 없음
- **용도**: Baseline MTP, Rho-1 Weighted만 가능 (Verifiable Critic 불가)
- **특징**: 기본 Python 프로그래밍 문제

### humaneval
- **샘플 수**: test 164
- **is_correct 필드**: ❌ 없음
- **용도**: 평가 전용 (학습 불가)
- **특징**: test split만 존재

## 로컬 테스트용 데이터셋 (datasets_local_small)

M3 Mac 등 로컬 환경에서 빠른 테스트를 위한 축소 버전:
- `codecontests_small`: train 100개, validation 32개
- `mbpp_small`: train 100개, validation 32개 (실제는 374/90이므로 전체)
- `humaneval_small`: test 32개

## 버전 관리

**v2.0.0** (2025-11-13)
- Meta LLaMA MTP 네이티브 파이프라인 최적화
- pytorch → safetensors 변환 완료
- 모든 모델 SHA256 해시 기록
- 데이터셋 schema.json 생성

## 무결성 검증

각 모델의 safetensors 파일 무결성 확인:

```bash
# meta-llama-mtp
cd storage/models/meta-llama-mtp/safetensors
sha256sum -c SHA256SUMS

# ref-sheared-llama-2.7b
cd storage/models/ref-sheared-llama-2.7b/safetensors
sha256sum -c SHA256SUMS

# starling-rm-7b
cd storage/models/starling-rm-7b/safetensors
sha256sum -c SHA256SUMS
```

## VESSL 업로드 (Phase 8 예정)

```bash
# 프로덕션 자산 업로드
uv run python scripts/sync_to_vessl_storage.py

# 로컬 small 포함 업로드
uv run python scripts/sync_to_vessl_storage.py --include-local-small
```

## 주의사항

1. **storage/models/** (v1)는 원본 보관용으로 유지
2. **storage/datasets/** (v1)는 원본 보관용으로 유지
3. **Phase 2 이후 코드에서는 models, datasets만 참조**
4. **Meta reference 코드 경로** (Phase 2에서 변경 예정):
   - 현재: `storage/models/llama-7b-mtp/llama/` (Phase 1 완료)
   - 이동 예정: `vendor/meta_llama/` (Phase 2에서 수행)
   - 이유: 외부 의존성 명시적 분리, 업스트림 업데이트 용이
5. **파라미터 기준 통합**:
   - `meta_adapter.yaml` = 모든 파라미터의 단일 출처
   - Meta `7B_1T_4` 기준 (intermediate_size: 11008, rope_theta: 10000.0, dtype: float16)
   - 필요 시 float16 → bfloat16 변환본을 생성하고, 변환 여부를 `metadata.json`/SHA256과 함께 기록
   - Phase 2에서 `configs/defaults.yaml`에 파라미터 스냅샷 등록

## 데이터셋 스키마 (공식 규격)

모든 데이터셋은 다음 필드 규약을 따릅니다:

```json
{
  "prompt": "string (required)",
  "response": "string (required)",
  "is_correct": "boolean (optional - Baseline/Rho-1에서는 null)",
  "metadata": {
    "problem_id": "string",
    "difficulty": "string (e.g., easy, medium, hard)"
  }
}
```

**검증**:
- `src/data/prepare.py`: 전처리 및 스키마 검증
- `scripts/validate_datasets.py`: 무결성 확인 (Phase1/Phase7)
- sequence length ≤ 2048 (초과 샘플 사전 필터링)

## FAQ

### Q: value head가 없을 때는?
A: metadata.json의 `value_head.path`가 null이면 파이프라인이 자동으로 초기화합니다.

### Q: 데이터셋 재전처리 방법은?
A: Phase 2 이후 `src/data/prepare.py`를 사용합니다.

### Q: Verifiable Critic에 어떤 데이터셋을 사용하나요?
A: codecontests만 사용 가능합니다 (is_correct 필드 보유).

---

**Phase 1 완료**: 2025-11-13
**다음 단계**: Phase 2 (코드 스켈레톤 구축)
