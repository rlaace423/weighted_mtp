# WMTP Implementation Roadmap

본 문서는 `docs/00_ideal_structure.md`와 `docs/01_storage_preparation_plan.md`에서 합의한 목표 구조를 실제 구현 순서로 풀어낸 실행 계획이다. 목적은 `docs/wmtp_research_proposal.md`에 정의된 세 가지 WMTP 실험(Baseline, Verifiable Critic, Rho-1 Weighted)이 재현 가능한 형태로 완성되도록 단계별 수행 항목, 산출물, 검증 조건을 명확히 하는 데 있다. 각 Phase는 선행 조건 충족 시 착수하며, 산출물이 승인되지 않으면 다음 단계로 넘어가지 않는다.

---

## 전체 Phase 개요

| Phase | 명칭 | 핵심 목표 | 주요 선행 조건 | 대표 산출물 |
|-------|------|------------|----------------|-------------|
| P0 | 프로젝트 킥오프 & 환경 정비 | 문서 정합성·용어·역할 정리 | 없음 | 용어 사전, 일정, 책임 매트릭스 |
| P1 | 모델·데이터 자산 확보 | `storage/` 표준 구조로 원본 수집·변환 | P0 | `models_v2/`, `datasets_v2/` v2.0.0 |
| P2 | 코드 스켈레톤 & 벤더 정리 | `vendor/meta_llama/` 및 `src/` 골격 구축 | P1 | 스켈레톤 코드, `pyproject.toml` 갱신 |
| P3 | 데이터 파이프라인 구현 | Raw → Alpaca 변환, 검증 유틸 완성 | P1 | `src/data/prepare.py`, `scripts/validate_datasets.py` |
| P4 | Meta Adapter 통합 | safetensors 로딩/forward 경로 확립 | P2 | `MetaLlamaMTPAdapter`, unit test |
| P5 | Value Weighting 모듈 | TD error, weight builder, metrics 구현 | P2 | `src/value_weighting/*`, 테스트 |
| P6 | 학습 파이프라인 Stage 0~3 | 환경 초기화·Stage1/2·MLflow 로깅 | P3, P4, P5 | `pipelines/training.py`, 통합 테스트 |
| P7 | 평가·분석 파이프라인 | Rho-1 비교 및 인퍼런스 루틴 마련 | P6 | `pipelines/evaluation.py`, Pass@K 리포트 |
| P8 | Config & CLI 체계 | defaults/recipes, CLI preset 완성 | P2, P6 | `core/config.py`, `cli/train.py`, recipes |
| P9 | 테스트 & 품질 게이트 | unit/integration/CI 파이프라인 구축 | P4~P8 | pytest suite, CI 워크플로우 |
| P10 | 배포 & 실험 운영 | VESSL 연동, MLflow/스토리지 업로드 | P1, P6, P8 | `scripts/sync_to_vessl_storage.py`, 배포 가이드 |
| P11 | 문서화 & 인수 | 문서 업데이트, 최종 체크리스트 | 전체 | 최신 문서/체인지로그/릴리즈 노트 |

---

## Phase 상세

### P0. 프로젝트 킥오프 & 환경 정비
- **목표**  
  - 모든 참여자가 설계 문서(00·01·제안서)를 이해하고 용어 정의를 통일한다.
  - 일정, 역할 분담, 브랜치 전략, 코드 리뷰 정책을 확립한다.
- **주요 활동**  
  - 문서 리뷰 & 이슈 목록화, `docs/glossary.md` 작성.  
  - 개발 환경 세팅(uv/ruff/pre-commit), 비밀키 관리 정책 확정.  
  - 작업 추적 보드 구성(Jira/Linear 등).
- **산출물**: 용어 사전, 일정표, 책임 매트릭스, 환경 세팅 가이드.  
- **검증 기준**: 결정 사항을 문서화하고 P1 착수 체크리스트 승인.

### P1. 모델·데이터 자산 확보 (`storage/` v2.0.0)
- **목표**  
  - Meta `7B_1T_4` 번들, Rho-1 reference, 소형 테스트 모델을 `storage/models_v2/` 표준 구조로 정리한다.  
  - CodeContests/MBPP/HumanEval raw 자산을 `datasets_v2/`에 수집하고 SHA256 기록을 남긴다.
- **주요 활동**  
  1. Hugging Face에서 `consolidated.pth`, `params.json`, `tokenizer.model` 다운로드 → `raw/` 저장.  
  2. safetensors 변환, `configs/params.json` 복사, `meta_adapter.yaml` 작성, `metadata.json` 갱신.  
  3. CodeContests raw JSONL 정리(`instruction/input/output/task_id/test_cases/is_correct/full_text` 유지), SHA256 계산.  
  4. `storage/README.md` 업데이트.
- **산출물**: `models_v2/` 및 `datasets_v2/` 디렉터리, SHA256 로그, 업데이트된 README.  
- **검증 기준**: dtype(float16) 유지, 토크나이저 공유 여부 기재, split 누락 없음, 체크리스트 서명.

### P2. 코드 스켈레톤 & 벤더 정리
- **목표**  
  - `vendor/meta_llama/`에 Meta 레퍼런스 코드( `llama/*.py` )를 옮기고, `src/` 모듈 골격을 생성한다.
- **주요 활동**  
  - vendor 패키지 초기화(`__init__.py`), mypy-friendly type stub 정리.  
  - `src/` 하위 디렉터리/`__init__.py` 생성, 인터페이스 스텁 작성.  
  - `pyproject.toml`, `ruff.toml`, pre-commit 훅 구성.  
  - `configs/defaults.yaml`, recipe 초안 마련(값은 placeholder 가능).
- **산출물**: 스켈레톤 코드, vendor 패키지, 기본 설정 파일.  
- **검증 기준**: `uv run python -c "from vendor.meta_llama import Transformer"` 성공, lint/format 통과.

### P3. 데이터 파이프라인 구현
- **목표**
  - HuggingFace 데이터셋을 직접 로드하여 Alpaca 형식으로 변환하고, correct/incorrect solutions를 통합 처리한다.
  - **Loss masking 구현**: instruction/input 토큰은 학습 제외, output 토큰만 학습 대상
- **주요 활동**
  - `src/data/prepare.py`: HF datasets 로드 → Alpaca 스타일 변환 (instruction/input/output, top-level is_correct)
  - `src/data/collators.py`: **Instruction/Input masking collator** 구현
    - Instruction 길이 추적 → labels에 -100 설정
    - Input 길이 추적 → labels에 -100 설정
    - Output만 실제 token ID 유지 (loss 계산)
    - attention_mask는 모든 토큰 유지 (context로 활용)
  - **Correct + Incorrect 통합**: 단일 JSONL에 is_correct 필드로 구분, task_id 접미사 추가
  - 토큰 필터링: SentencePieceProcessor로 2048 토큰 초과 샘플 제거
  - `scripts/validate_datasets.py`: 스키마, 토큰 길이, SHA256 체크
  - `datasets_local_small/` 자동 생성 스크립트
- **산출물**: 변환 스크립트, collator 모듈, 검증 스크립트, stats JSON
- **검증 기준**:
  - top-level is_correct 필드 존재
  - Collator가 instruction/input 토큰을 -100으로 마스킹
  - Output 토큰만 loss 계산 확인
  - 2048 토큰 초과 없음, schema 검증 통과

### P4. Meta Adapter 통합
- **목표**  
  - `MetaLlamaMTPAdapter`가 safetensors/params/json 조합을 로딩해 trunk/full forward를 제공하도록 구현한다.
- **주요 활동**  
  - `src/models/meta_mtp/adapter.py`: 캐시 준비, n_future_tokens 적용, extra_heads 처리.  
  - `src/models/meta_mtp/checkpoints.py`: dtype 변환, 장치 선택, value head 로딩.  
  - micro 모델을 사용한 unit test (`tests/unit/test_adapter.py`).  
  - 오류/로그 정책 정리.
- **산출물**: Adapter 모듈, 체크포인트 유틸, unit test.  
- **검증 기준**: micro 모델 trunk_forward < 2s, dtype & shape 검증, `pytest -k adapter` 통과.

### P5. Value Weighting 모듈
- **목표**  
  - Verifiable Critic/Rho-1 Weighted 실험에 필요한 TD error 계산, weight builder, metrics를 모듈화한다.
- **주요 활동**  
  - `td_error.py`: advantage 계산 (GAE, Z-score), dtype 안전성 확보.  
  - `weight_builder.py`: temperature softmax, clipping, entropy floor.  
  - `metrics.py`: TD error mean/std, weight entropy, KL 모니터링.  
  - (옵션) `regularizers.py`: critic drift 방지 로직.  
  - 단위 테스트: zero reward, extreme reward, masking 케이스.
- **산출물**: value_weighting 패키지.  
- **검증 기준**: 모든 unit test 통과, 수치 비교(참조 스크립트)에서 오차 허용 범위 내.

### P6. 학습 파이프라인 Stage 0~3
- **목표**
  - Stage0(환경) → Stage1(trunk pretrain) → Stage2(weighted training) → Stage3(logging)을 오케스트레이션한다.
  - **Stage2 Critic Continual Learning**: Value loss를 auxiliary loss로 추가하여 policy 학습 중 critic도 지속 학습
- **주요 활동**
  - `pipelines/training.py`: 환경 초기화, 데이터 로더, Stage1/2 실행, MLflow 로깅, 체크포인트 저장.
  - **Stage2 Loss 구조 구현**:
    - `total_loss = weighted_ce_loss + value_coef * value_loss`
    - Value coefficient: 0.5 (기본) 또는 1.0 (recipe 설정)
    - Value loss clipping: clip_range=0.2
    - Gradient clipping: max_grad_norm=0.5~1.0
  - `runtime/environment.py`: seed/device/dtype 설정.
  - `runtime/mlflow.py`: 실험 생성, 메트릭/아티팩트 기록.
  - 통합 테스트: micro 모델 + small 데이터로 end-to-end smoke test.
- **산출물**: 학습 파이프라인 코드, 통합 테스트(`tests/integration/test_stage*.py`).
- **검증 기준**:
  - Smoke test 성공
  - MLflow에 핵심 메트릭 기록: TD error, weight entropy, **value loss, value explained variance**
  - Stage2에서 value loss가 auxiliary loss로 추가되어 critic 지속 학습 확인

### P7. 평가·분석 파이프라인
- **목표**  
  - 학습된 모델의 Pass@K, Exact Match, Rho-1 비교 분석을 자동화한다.
- **주요 활동**  
  - `pipelines/evaluation.py`: inference 루틴, beam/nucleus 옵션, Rho-1 reference와의 loss 비교.  
  - `scripts/generate_metrics_report.py`: Pass@1/5/10, 평균 길이, 예시 코드 저장.  
  - MBPP/HumanEval 평가 스크립트 통합(채점 모듈).  
  - 결과를 MLflow/JSON으로 기록.
- **산출물**: 평가 파이프라인, 리포트 생성 스크립트.  
- **검증 기준**: baseline 모델용 평가 스크립트 문제없이 동작, 리포트 파일 생성.

### P8. Config & CLI 체계
- **목표**  
  - 사용자 진입점(`cli/train.py`)과 설정(`core/config.py`, recipes)을 완성해 실험 실행을 단순화한다.
- **주요 활동**  
  - Pydantic 기반 Config/Recipe 정의, defaults/recipes/ 환경변수 로딩.  
  - CLI 옵션(`--config`, `--recipe`, `--preset`, `--use-micro-model`).  
  - 세 실험 레시피(Baseline, Verifiable, Rho-1)의 파라미터 테이블 정리.  
  - dry-run 모드, 로깅 포맷 통일.
- **산출물**: `core/config.py`, `cli/train.py`, `configs/*.yaml`.  
- **검증 기준**: `uv run python -m weighted_mtp.cli.train --dry-run` 성공, recipe별 파이프라인 파라미터 주입 확인.

### P9. 테스트 & 품질 게이트
- **목표**  
  - 회귀 방지 체계를 구축하고 GitHub Actions 등 CI를 통해 지속적으로 검증한다.
- **주요 활동**  
  - unit/integration 테스트 확장, coverage 목표 설정.  
  - `scripts/run_smoke_tests.sh` 작성, pre-commit 훅 구성.  
  - `.github/workflows/ci.yaml`(lint, format, tests) 구축.  
  - 실패 시 triage 가이드 작성.
- **산출물**: pytest suite, smoke script, CI 워크플로우.  
- **검증 기준**: CI 100% 통과, 커버리지 목표(예: 핵심 모듈 80%+) 충족.

### P10. 배포 & 실험 운영
- **목표**  
  - VESSL/클라우드 환경에서 실험을 실행할 준비를 갖추고, 스토리지·MLflow 연동을 완성한다.
- **주요 활동**  
  - `scripts/sync_to_vessl_storage.py`: 모델/데이터 업로드 자동화.  
  - Dockerfile 또는 VESSL 기본 이미지 설정, requirements 잠금(`uv pip compile`).  
  - MLflow endpoint 검증, 자격 증명 관리.  
  - 배포 가이드, 예제 명령 작성.
- **산출물**: 동기화 스크립트, 배포 가이드, dry-run 로그.  
- **검증 기준**: `vessl run create ... --dry-run` 성공, MLflow artifact 업로드 확인.

### P11. 문서화 & 인수
- **목표**  
  - 모든 문서/체인지로그를 최신 상태로 맞추고, 최종 인수 체크리스트를 완료한다.
- **주요 활동**  
  - `docs/00_ideal_structure.md`, `docs/01_storage_preparation_plan.md` 차이 반영.  
  - `docs/migration_notes.md` 및 CHANGELOG 업데이트.  
  - 운영 가이드, Known Issues 작성.  
  - 인수 체크리스트(모델/데이터/코드/배포/문서) 전항목 확인 후 승인.
- **산출물**: 최신 문서 세트, 체인지로그, 릴리즈 노트, 인수 보고서.  
- **검증 기준**: 모든 체크 항목 ✔️, PO 승인 기록.

---

## 의존성 & 병행 전략
- **P0 → P1 → P2 → P3 → P4/P5 → P6 → P7** 순으로 핵심 경로를 이행한다.  
- P4(Value weighting)과 P5(파이프라인) 일부는 설계 확정 후 병행 가능하지만, 테스트는 Stage 통합 후에만 유효하다.  
- P8~P10은 학습 파이프라인이 안정화(P6 완료)된 뒤 바로 착수 가능하며, P9(CI)는 주요 모듈이 준비되는 시점부터 점진적으로 구축한다.
- 모든 Phase 종료 시 `migration_notes.md`에 결과와 차이를 기록해 다음 단계 착수 조건으로 활용한다.

---

## 완료 기준 요약
- **실험 실행**: CLI recipe로 Baseline/Verifiable/Rho-1 실험이 동일 파이프라인 위에서 실행되고, 평가 리포트가 자동 생성된다.
- **자산 관리**: `storage/` 구조와 metadata가 문서와 일치하며, SHA256/dtype/토크나이저 정보가 기록되어 재현성을 보장한다.
- **품질 보증**: unit/integration/CI가 모두 통과하고, MLflow/배포 스크립트가 정상 동작한다.
- **문서화**: 설계·준비·구현·운영 문서가 최신 상태로 유지되며, 제안서에 명시된 세 실험 비교 보고가 가능하다.
# WMTP 리팩토링 Phase별 구현 계획

본 문서는 `docs/00_ideal_structure.md`와 `docs/01_storage_preparation_plan.md`를 실제 코드/데이터 자산으로 구현하기 위한 상세 단계별 로드맵이다. 모든 Phase는 **철학 → 구현 단위 → 산출물 → 검증 기준**을 명확히 정의하며, 순차적으로 수행하되 병렬 가능한 작업은 명시된 조건을 만족할 때에 한해 병행한다.

---

## 0. 공통 철학 및 원칙 정리

1. **Meta 네이티브 우선**: 모델 로딩·forward는 Meta 레퍼런스 구현을 직접 호출하는 Adapter 기반 구조를 유지한다. HuggingFace 호환 계층은 제거한다.
2. **단순·명료한 파이프라인**: 3개 핵심 실험(Baseline, Verifiable Critic, Rho-1 Weighted)에 집중하고, 각 실험은 동일한 파이프라인 위에서 recipe 차이만 남긴다.
3. **TD error 기반 가중치 안정화**: GAE, Z-score, weight 정규화/클리핑 등 critic-weighted WMTP에 필요한 안정화 기법을 내장한다.
4. **storage/의 단일화**: 변환된 dataset/model은 `01_storage_preparation_plan.md` 스키마에 맞추어 staging한다. 코드베이스에는 로직만, 자산은 storage에만 둔다.
5. **검증 우선 개발**: 각 Phase는 최소 하나 이상의 자동 테스트 또는 체크리스트를 통과해야 다음 Phase로 진행할 수 있다.

---

## 1. Phase 요약

| Phase | 명칭 | 주요 목표 | 선행 조건 | 핵심 산출물 |
|-------|------|-----------|-----------|-------------|
| P0 | 킥오프 & 환경 정비 | 일정/역할 정의, 문서 정합성 확보 | 없음 | 실행 일정, 용어 사전 |
| P1 | storage 자산 변환 | 모델/데이터 v2 구조로 재편 | P0 | `storage/models_v2`, `storage/datasets_v2`, README |
| P2 | 코드 스켈레톤 구축 | 디렉터리 생성, 빈 모듈 배치 | P1 (부분 병행 가능) | `src/` 하위 기본 모듈, configs/skeleton |
| P3 | Meta Adapter 통합 | Meta 네이티브 모델 로딩 확립 | P2 | `MetaLlamaMTPAdapter` 개선, 로딩 테스트 |
| P4 | Value Weighting 모듈 | TD error 계산/가중치 생성 로직 설계 | P2 | `src/value_weighting/` 모듈, 단위 테스트 |
| P5 | 파이프라인 구현 | Stage 0~3 오케스트레이션 | P3, P4 | `pipelines/training.py`, 통합 테스트 |
| P6 | CLI & Config | `cli.train`, config/recipe 적용 | P5 (초기 버전은 병행) | CLI 엔트리, defaults.yaml, recipe 3종 |
| P7 | 테스트 & 검증 | unit/integration 테스트 세트 | P3~P6 | pytest 시나리오, GitHub Actions 워크플로우 |
| P8 | 배포 & VESSL 연동 | sync 스크립트, MLflow 연동 확인 | P1, P6 | `scripts/sync_to_vessl_storage.py`, VESSL 드라이런 |
| P9 | 문서화 & 최종 점검 | 문서 정리, 체인지로그, 인수 조건 | 전체 | docs 업데이트, 체인지로그, 승인 체크리스트 |

---

## 2. 상세 Phase별 계획

### Phase 0. 킥오프 & 환경 정비
- **목표**: 팀이 공통 철학을 공유하고, 용어/경로/도구 체계를 확정한다.
- **작업**
  1. `00_ideal_structure.md`, `01_storage_preparation_plan.md`, 본 문서를 리뷰하고 승인 기록 남김.
  2. 용어 사전 작성(`docs/glossary.md`): horizon, trunk_forward, micro model 등 정의.
  3. 개발 브랜치 전략/CI 정책 합의.
- **산출물**
  - `docs/glossary.md`
  - 프로젝트 일정 및 책임자 매트릭스(내부 공유 문서)
- **검증 기준**
  - 모든 문서가 최신 상태이며 상호 모순 없음.
  - 모든 참여자가 Phase 1 착수 조건에 동의.

### Phase 1. storage 자산 변환
- **목표**: 모델/데이터 자산을 `01_storage_preparation_plan.md` 구조로 재편.
- **선행조건**: Phase 0 완료.
- **작업**
  1. 기존 `storage/models_v1`, `storage/datasets_v1` 자산 백업.
  2. safetensors 병합, `meta_adapter.yaml`, `metadata.json` 생성 (Meta `7B_1T_4` params: `intermediate_size=11008`, `rope_theta=10000.0`, `dtype=float16` 반영).
  3. 데이터셋 전처리 스크립트 작성 및 실행 → `datasets_v2` 구축.
  4. `storage/README.md` 업데이트와 체크리스트 작성.
- **산출물**
  - `storage/models_v2/...`
  - `storage/datasets_v2/...`
  - `storage/datasets_local_small/...`
- **검증 기준**
  - 체크리스트 항목 전부 통과 (SHA256, dtype=float16, schema, stats).
  - micro 모델/데이터로 최소 smoke test 통과 (임시 스크립트 ok).

### Phase 2. 코드 스켈레톤 구축
- **목표**: 이상적 구조에 맞는 패키지/파일을 생성하고, 인터페이스 뼈대를 정의. Meta reference 코드를 vendor/로 이동.
- **선행조건**: Phase 1 (모델/데이터 구조 확인) — 단, 구조 정의는 병행 가능.
- **작업**
  1. `vendor/meta_llama/` 디렉터리 생성 및 Meta reference 코드 이동
     - `storage/models/llama-7b-mtp/llama/*` → `vendor/meta_llama/`
     - `vendor/__init__.py`, `vendor/meta_llama/__init__.py` 작성
     - import 경로 정리: `from vendor.meta_llama import Transformer, ModelArgs`
  2. `src/` 하위 디렉터리 생성 (`cli`, `core`, `data`, `models`, `value_weighting`, `pipelines`, `runtime`, `utils`).
  3. 각 모듈에 `__init__.py`와 타입 힌트 기반 인터페이스 스텁 작성.
  4. configs 디렉터리에 `defaults.yaml`, recipe 템플릿 생성. `defaults.yaml`에 모델 파라미터 스냅샷 등록.
  5. pre-commit 셋업, formatting/linting 기준 확정.
- **산출물**
  - `vendor/meta_llama/` 패키지 (Meta reference 코드)
  - 최소 구현 클래스/함수 (pass 처리) 포함된 스켈레톤 코드.
  - `pyproject.toml` 업데이트(패키지 경로 `packages = ["weighted_mtp", "vendor"]`, 의존성).
  - `configs/defaults.yaml` (모델 파라미터 스냅샷 포함)
- **검증 기준**
  - `uv run python -c "from vendor.meta_llama import Transformer; print('OK')"` 성공
  - `uv run pytest` (빈 테스트) + `uv run ruff check` 통과.
  - import 경로 모두 정상 (ModuleNotFoundError 없음).

### Phase 3. Meta Adapter 통합
- **목표**: Meta LLaMA MTP 모델을 네이티브 인터페이스로 로딩하고, trunk/full forward 경로를 확립.
- **선행조건**: Phase 2 스켈레톤, Phase 1 모델 자산.
- **작업**
  1. `src/models/meta_mtp/adapter.py` 구현: freqs_cis/causal mask 캐시, trunk/full forward.
  2. `src/models/meta_mtp/checkpoints.py` 작성: safetensors 로딩, dtype 정리.
  3. micro 모델로 adapter 단위 테스트(`tests/unit/test_adapter.py`).
  4. Stage1 value head 저장/로드 인터페이스 정의.
- **산출물**
  - 동작하는 Meta Adapter 클래스.
  - 관련 unit test.
- **검증 기준**
  - micro 모델로 trunk_forward 시간을 측정해 합리적 범위(수 초 이하).
  - Value head normalization 확인 (평균/표준편차 0≠, 1≠ 0).

### Phase 4. Value Weighting 모듈
- **목표**: TD error 기반 토큰 가중치 계산을 모듈화.
- **선행조건**: Phase 2.
- **작업**
  1. `value_weighting.td_error`: value head 출력과 보상으로 TD error 계산.
  2. `value_weighting.weight_builder`: TD error 정규화, temperature/클리핑, entropy 제약 적용.
  3. `value_weighting.metrics`: weight/TD error 통계 계산.
  4. (선택) `value_weighting.regularizers`: drift 감시, 선택적 KL 모니터링.
  5. 최신 연구(AsyPPO, PSPO, DVPO, SFPO, DPO 등)에서 차용 가능한 안정화 포인트 주석화.
- **산출물**
  - 각 모듈에 테스트 작성 (`tests/unit/test_td_error.py`, `test_weight_builder.py` 등).
  - config schema 업데이트(kl target, clip 값 등).
- **검증 기준**
  - 단위 테스트에서 edge case 통과 (zero rewards, extreme TD error 등).
  - 결과값이 참고 구현과 일치하는지 간단한 수치 검증.

### Phase 5. 파이프라인 구현
- **목표**: Stage 0~3을 아우르는 실행 흐름 구축.
- **선행조건**: Phase 3, Phase 4.
- **작업**
  1. `pipelines/training.py`에 run_training_pipeline 구현 (환경 초기화, 모델 로딩, Stage1/2, MLflow).
  2. `pipelines/evaluation.py`에 baseline 평가 스텁.
  3. runtime 모듈(환경/분산/MLflow) 초기화 코드 작성.
  4. Stage별 metrics 로깅 설계 (loss, TD error, weight stats).
- **산출물**
  - 통합 파이프라인 코드.
  - integration 테스트(`tests/integration/test_stage1_local.py`, `test_stage2_local.py`).
- **검증 기준**
  - micro 모델 + small 데이터로 Stage1, Stage2 순차 실행 성공.
  - MLflow 로컬 모드(`file://`)에서 메트릭 기록 확인.

### Phase 6. CLI & Config 시스템
- **목표**: 사용자 진입점 및 설정 체계 완성.
- **선행조건**: Phase 5 (기본 파이프라인 동작), Phase 2 (config 스켈레톤).
- **작업**
  1. `cli/train.py`: argparse, preset 옵션(`--preset local-light`, `--use-micro-model`).
  2. `core/config.py`: Pydantic 모델 정의 (Config, Recipe).
  3. `configs/defaults.yaml`, recipe 3종 작성 및 검증.
  4. `.env` 로딩(python-dotenv) 및 VESSL/MLflow 환경변수 주입 로직.
- **산출물**
  - CLI에서 세 실험을 각각 실행할 수 있는 최소 기능.
  - config validation 테스트.
- **검증 기준**
  - `uv run python -m weighted_mtp.cli.train --dry-run` 성공.
  - recipe별 파라미터가 파이프라인에 정확히 반영되는지 확인.

### Phase 7. 테스트 & 검증 체계
- **목표**: 안정적인 회귀 방지 및 품질 보증.
- **선행조건**: Phase 3~6.
- **작업**
  1. unit/integration 테스트 확장, 커버리지를 최소한 Stage1/Stage2 core에 맞춘다.
  2. GitHub Actions 워크플로우 작성(`.github/workflows/test.yaml`): 포맷팅, lint, tests.
  3. 로컬 smoke 테스트 스크립트 작성 (`scripts/run_smoke_tests.sh`).
- **산출물**
  - 테스트 스위트, CI 파이프라인.
  - 테스트 결과 리포트(coverage 등).
- **검증 기준**
  - CI 모든 단계 통과.
  - 실패 시 빠른 triage 프로세스 확립.

### Phase 8. 배포 & VESSL 연동
- **목표**: 실제 GPU 환경에서 실행 가능한 상태로 배포.
- **선행조건**: Phase 1 (자산), Phase 6 (CLI).
- **작업**
  1. `scripts/sync_to_vessl_storage.py`: storage/ 내용 업로드 자동화.
  2. Docker 이미지 빌드/테스트 (필요 시 `docker/` 폴더 추가).
  3. VESSL CLI 명령 템플릿(`docs/00_ideal_structure.md`의 명령 검증).
  4. MLflow 서버 연결 및 artifact 업로드 확인.
- **산출물**
  - 업로드 스크립트, Dockerfile/이미지 (필요 시).
  1. VESSL dry-run 로그.
- **검증 기준**
  - `vessl run create ... --dry-run` 성공.
  - MLflow에 실험 기록 생성, artifact 업로드 확인.

### Phase 9. 문서화 & 최종 점검
- **목표**: 모든 산출물을 정리하고, 인수 기준을 충족했음을 증명.
- **선행조건**: 전체 Phase.
- **작업**
  1. `docs/migration_notes.md`에 변경 사항 및 차이점 기록.
  2. `docs/00_ideal_structure.md`, `docs/01_storage_preparation_plan.md` 실제 결과 반영 업데이트.
  3. CHANGELOG 작성, 릴리즈 노트 초안.
  4. 최종 검증 체크리스트 (모델/데이터/코드/배포/문서) 점검.
- **산출물**
  - 최신 문서 세트, 체인지로그, 인수 보고서.
  - 릴리즈 태그 준비(필요 시).
- **검증 기준**
  - 모든 체크리스트 ✔️, 잔여 이슈 없음.
  - 리뷰어/PO 승인.

---

## 3. Phase 간 의존성 및 병행 전략

- **P1 ↔ P2**: 모델/데이터 구조가 확정돼야 파이프라인을 안정적으로 테스트할 수 있으나, 스켈레톤 작성은 병행 가능. 단, Phase 2 테스트에서 사용할 자산 mock이 준비돼야 한다.
- **P3와 P4**: Meta Adapter와 PPO 모듈은 상호 독립적으로 개발 가능하지만, 통합 테스트는 둘 다 준비돼야 진행 가능.
- **P6 이후**: CLI/Config 구현이 완료되기 전에도 단위 테스트는 가능하지만, 통합 검증은 CLI를 통해 수행하는 것이 바람직하다.
- **P8**: storage 자산 업로드와 CLI 안정화가 선행돼야 VESSL 연동 테스트가 의미 있다.

---

## 4. 위험 요소 및 대응

| 위험 | 설명 | 영향 | 대응 전략 |
|------|------|------|-----------|
| 모델 자산 무결성 | safetensors 병합 오류, adapter 파라미터 불일치 | 학습 실패 | SHA256 검증, meta_adapter.yaml 2중 검토 |
| 데이터 품질 | `is_correct` 누락, 2048 초과 샘플 미제거 | Verifiable 학습 실패 | 전처리 스크립트 자동화 + 테스트 |
| Weight 불안정 | TD error 폭주, weight 집중 | 학습 중단 | Z-score/temperature 조정, weight clip |
| VESSL 업로드 실패 | 경로 오타, 권한 문제 | 배포 지연 | dry-run 스크립트, Secrets 검증 |
| 일정 지연 | 병목 Phase 장기화 | 전체 일정 영향 | Phase 완성 조건 엄격히 관리, 병렬화 |

---

## 5. 모니터링 및 리포트

- **주간 리포트 템플릿** (Notion/내부 문서)
  - 진행 Phase 요약, 주요 산출물 링크
  - 위험 항목 업데이트
  - 다음 주 목표
- **메트릭**
  - 테스트 통과 수, 커버리지
  - storage 자산 변환 진행률
  - TD error/weight 모니터링 (실제 학습 시)
- **리뷰 사이클**
  - Phase마다 코드 리뷰 + 문서 리뷰 필수
  - Phase 완료 시 승인자 서명 기록 (체인지로그 또는 PR 설명)

---

## 6. 마일스톤 타임라인 (예시)

| 주차 | Phase | 예상 소요 | 비고 |
|------|-------|-----------|------|
| Week 1 | P0, P1 | 3일, 2일 | 자산 변환 완료 |
| Week 2 | P2, P3 | 3일, 2일 | 스켈레톤 + Meta Adapter |
| Week 3 | P4, P5 | 2일, 3일 | PPO 모듈 + 파이프라인 |
| Week 4 | P6, P7 | 2일, 3일 | CLI/Config + 테스트/CI |
| Week 5 | P8, P9 | 3일, 2일 | 배포 연동 + 문서화 |

*실제 일정은 자원 배분에 따라 조정 가능하며, 각 Phase 완료 시점에 상태 점검 미팅을 갖는다.*

---

## 7. Phase 완료 보고 템플릿

각 Phase 종료 시 아래 템플릿을 채워 `docs/migration_notes.md` 또는 별도 보고서에 기록한다.

```
### Phase X 완료 보고
- 일정: YYYY-MM-DD ~ YYYY-MM-DD
- 담당자: 이름
- 주요 산출물:
  - 경로/링크
- 체크리스트:
  - [x] 항목1
  - [ ] 항목2 (보류 사유)
- 발견된 이슈 / 해결방안:
- 다음 단계 착수 조건:
```

---

이 계획 문서는 리팩토링 전 과정의 기준선이다. Phase 수행 중 발견되는 변경 사항이나 위험 요소는 즉시 문서에 반영하며, 모든 결정은 문서 간 정합성을 유지하는 것을 원칙으로 한다. Phase별 승인 없이 다음 단계로 넘어가지 않도록 각 담당자는 산출물과 검증 결과를 명확히 공유해야 한다.

