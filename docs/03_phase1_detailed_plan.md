# Phase 1: 모델·데이터 자산 확보

## 1. 개요

**목표**: `storage/` 표준 구조로 원본 수집·변환하여 WMTP 실험 기반 마련

**핵심 성과**: 5개 모델 + 3개 데이터셋 (3.7M samples) + 메타데이터 파일 (99% 메모리 절감)

**선행 조건**: Phase 0 (프로젝트 킥오프 & 환경 정비) 완료

**주요 산출물**:
- `storage/models_v2/`: 5개 모델 (meta-llama-mtp 6.7B, ref-sheared-llama-2.7b 2.7B, starling-rm-7b 13.3B, micro-mtp, micro-ref)
- `storage/datasets_v2/`: CodeContests 3.7M, MBPP 964, HumanEval 164
- 메타데이터 파일: `*_metadata.json` (is_correct, difficulty만 포함, ~217MB)

---

## 2. 모델 자산 준비

### 2.1 Meta LLaMA MTP (Base Model)

**다운로드 소스**: facebook/multi-token-prediction (Hugging Face)

**변환 작업**:
1. Raw 다운로드: `7B_1T_4/consolidated.pth`, `params.json`, `tokenizer.model`
2. Safetensors 변환: float16 유지, freqs_cis 제외 (runtime 계산)
3. Configs 복사: `params.json` → `configs/params.json`
4. Adapter 설정: `meta_adapter.yaml` (n_future_tokens=4, rope_theta=10000.0)
5. Metadata 작성: `metadata.json` (implementation: "pure_pytorch", dtype: "float16")

**산출물**:
```
storage/models_v2/meta-llama-mtp/
├── safetensors/
│   ├── model.safetensors (25GB, float16, SHA256: a29872d0...)
│   └── SHA256SUMS
├── configs/
│   ├── params.json
│   └── meta_adapter.yaml
├── tokenizer/
│   ├── tokenizer.model
│   └── tokenizer_config.json
└── metadata.json
```

**검증**:
- ✅ Safetensors 저장/로딩 정상 (freqs_cis 자동 생성)
- ✅ Forward pass shape: `[batch, seq, n_future_tokens, vocab]`
- ✅ Gradient 계산 가능 (학습 모드)
- ✅ Device 이동: cuda/mps/cpu 자동 지원
- ✅ SHA256 검증 통과

### 2.2 Reference Model (Rho-1)

**다운로드 소스**: princeton-nlp/Sheared-LLaMA-2.7B (Hugging Face)

**변환 작업**:
1. Sharded PyTorch `.bin` 병합
2. Safetensors 변환: float16
3. Tokenizer 공유: Meta MTP와 동일 (SHA256: 9e556afd...)
4. Metadata 작성: `tokenizer_shared_with: "meta-llama-mtp"`

**산출물**:
```
storage/models_v2/ref-sheared-llama-2.7b/
├── safetensors/
│   └── model.safetensors (10.3GB, float16, SHA256: 4091b6ac...)
├── configs/
│   └── config.json
└── metadata.json
```

**검증**:
- ✅ HuggingFace `AutoModelForCausalLM` 로딩 정상
- ✅ Tokenizer 공유 확인 (동일 vocab size: 32000)

### 2.3 Reward Model (Optional)

**다운로드 소스**: berkeley-nest/Starling-RM-7B-alpha (Hugging Face)

**변환 작업**:
1. Safetensors 변환: bfloat16 유지
2. Metadata 작성: `status: "optional"`

**산출물**:
```
storage/models_v2/starling-rm-7b/
├── safetensors/
│   └── model.safetensors (25GB, bfloat16, SHA256: cd90dc78...)
└── metadata.json
```

### 2.4 Micro Models (로컬 테스트용)

**생성 방법**: `scripts/prepare_local_small_model.py` 실행

**변환 작업**:
1. Base safetensors에서 일부 레이어 슬라이싱
2. 4-layer, 512-dim 경량 모델 생성
3. Metadata 작성: `target_device: "mps"`

**산출물**:
```
storage/models_v2/micro-mtp/
├── safetensors/model.safetensors (177MB, float16)
├── configs/config.json
└── metadata.json

storage/models_v2/micro-ref/
├── safetensors/model.safetensors (177MB, float16)
└── metadata.json
```

**검증**:
- ✅ 파일 크기 <200MB
- ✅ Unit tests 11/11 통과 (`pytest tests/unit/test_adapter.py -k micro`)

---

## 3. 데이터셋 준비

### 3.1 CodeContests (학습용)

**다운로드 소스**: deepmind/code_contests (Hugging Face)

**전처리 작업** (`scripts/setup_datasets.py`):
1. HuggingFace dataset → Alpaca 형식 JSONL 변환
2. Correct + Incorrect solutions 통합 저장
3. Top-level `is_correct` 필드 추가 (boolean)
4. Metadata 추출: `difficulty` (1-11), `source`, `has_tests`
5. 길이 필터: instruction + input + output 합산 2048 토큰 이하
6. Python/Python3 솔루션만 포함

**메타데이터 추출** (`scripts/extract_metadata.py`):
- 전체 데이터(~15GB)에서 is_correct, difficulty만 추출
- 메타데이터 파일 크기: ~217MB (99% 압축)
- 용도: 메모리 효율적 학습 (Phase 3에서 사용)

**산출물**:
```
storage/datasets_v2/codecontests/processed/
├── train.jsonl (3,691,981 samples)
├── train_metadata.json (~217MB)
├── valid.jsonl (14,725 samples)
├── valid_metadata.json
├── test.jsonl (14,851 samples)
├── test_metadata.json
└── schema.json
```

**샘플 구조**:
```json
{
  "instruction": "문제 설명",
  "input": "테스트 케이스 예시",
  "output": "Python 솔루션 코드",
  "task_id": "problem_correct_0",
  "is_correct": true,
  "metadata": {
    "source": "code_contests",
    "difficulty": 7,
    "has_tests": true
  }
}
```

**메타데이터 구조**:
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
    "difficulty_dist": {"7": 3200000, "2": 236000, ...}
  }
}
```

**검증**:
- ✅ 샘플 수: Train 3,691,981 (correct: 1,754,404 / incorrect: 1,937,577)
- ✅ is_correct 필드 존재 및 boolean 타입 확인
- ✅ difficulty 필드 존재 및 1-11 범위 확인
- ✅ 메타데이터 파일 생성 완료 (train/valid/test)

### 3.2 MBPP (평가용)

**다운로드 소스**: mbpp (Hugging Face)

**전처리 작업**:
1. Alpaca 형식 변환
2. test_list 필드 포함 (평가용)
3. is_correct 필드 없음 (correct 솔루션만 포함)

**산출물**:
```
storage/datasets_v2/mbpp/processed/
├── train.jsonl (374 samples)
├── validation.jsonl (90 samples)
├── test.jsonl (500 samples)
└── schema.json
```

**검증**:
- ✅ 샘플 수: 374 + 90 + 500 = 964
- ✅ test_list 필드 존재 확인

### 3.3 HumanEval (평가용)

**다운로드 소스**: openai_humaneval (Hugging Face)

**전처리 작업**:
1. Alpaca 형식 변환
2. test 필드 포함 (평가용)
3. is_correct 필드 없음

**산출물**:
```
storage/datasets_v2/humaneval/processed/
├── test.jsonl (164 samples)
└── schema.json
```

**검증**:
- ✅ 샘플 수: 164
- ✅ test 필드 존재 확인

---

## 4. 실제 성과 요약 (2025-11-14)

### 4.1 모델 자산 (5개)

| 모델 | 크기 | dtype | SHA256 | 파라미터 |
|------|------|-------|--------|---------|
| meta-llama-mtp | 25GB | float16 | a29872d0... | 6.7B |
| ref-sheared-llama-2.7b | 10.3GB | float16 | 4091b6ac... | 2.7B |
| starling-rm-7b | 25GB | bfloat16 | cd90dc78... | 13.3B |
| micro-mtp | 177MB | float16 | - | 4-layer |
| micro-ref | 177MB | float16 | - | 4-layer |

### 4.2 데이터셋 자산 (3개)

| 데이터셋 | Train | Valid | Test | 특징 |
|---------|-------|-------|------|------|
| CodeContests | 3,691,981 | 14,725 | 14,851 | is_correct, difficulty 포함 |
| MBPP | 374 | 90 | 500 | test_list 포함 (평가용) |
| HumanEval | - | - | 164 | test 포함 (평가용) |

### 4.3 메타데이터 파일 (10개)

**생성된 메타데이터 파일**:
- codecontests: train_metadata.json, valid_metadata.json, test_metadata.json
- mbpp: train_metadata.json, validation_metadata.json, test_metadata.json
- humaneval: test_metadata.json
- (소형 데이터셋용 메타데이터 3개 추가)

**메모리 효율**:
- 전체 데이터: ~15GB
- 메타데이터: ~217MB
- **압축률: 99%**

### 4.4 핵심 혁신

**Pure PyTorch 구현 준비**:
- Meta vendor 코드를 참고용으로만 유지
- Safetensors 호환 (freqs_cis runtime 계산)
- FSDP 완전 호환

**메타데이터 기반 로딩 준비**:
- is_correct, difficulty 정보 별도 추출
- Phase 3에서 메모리 효율적 학습에 활용

---

## 5. Phase 2 착수 조건

Phase 1 완료 후, 다음 조건을 만족하여 Phase 2 (코드 스켈레톤 & Pure PyTorch 구현)로 진행 가능:

✅ **필수 조건**:
1. `storage/models_v2/meta-llama-mtp/` 모델 자산 준비 완료
2. `storage/datasets_v2/codecontests/` 데이터셋 준비 완료
3. 메타데이터 파일 생성 완료 (train/valid/test)
4. SHA256 해시 검증 통과
5. dtype(float16) 유지 확인

✅ **권장 조건**:
1. Safetensors 저장/로딩 테스트 통과
2. 토크나이저 공유 여부 확인 (Reference 모델)
3. 메타데이터 파일 구조 검증 (is_correct, difficulty 필드)

---

## 6. 참고 자료

### 6.1 내부 문서

- `docs/00_ideal_structure.md`: 전체 아키텍처 및 storage 구조 정의
- `docs/01_storage_preparation_plan.md`: 모델·데이터 자산 변환 가이드
- `docs/02_implementation_plan.md`: Phase 1 요구사항 및 실제 성과

### 6.2 스크립트

**모델 변환**:
- `scripts/prepare_local_small_model.py`: Micro 모델 생성
- `scripts/verify_mtp_model.py`: 모델 검증

**데이터셋 전처리**:
- `scripts/setup_datasets.py`: HuggingFace → Alpaca 변환
- `scripts/extract_metadata.py`: 메타데이터 추출
- `scripts/validate_datasets.py`: 데이터셋 검증

### 6.3 핵심 파일

**모델 메타데이터**:
- `storage/models_v2/meta-llama-mtp/metadata.json`
- `storage/models_v2/ref-sheared-llama-2.7b/metadata.json`

**데이터셋 메타데이터**:
- `storage/datasets_v2/codecontests/processed/train_metadata.json` (~217MB)
- `storage/datasets_v2/codecontests/processed/schema.json`

---

**Phase 1 완료** (2025-11-14)

이 문서는 Phase 1 구현 결과를 기반으로 소급 작성되었으며, 00, 01, 02 문서와의 일관성을 유지합니다.
