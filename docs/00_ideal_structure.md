# Weighted MTP 리팩토링 이상적 구조

WMTP 리팩토링 프로젝트는 `wmtp_research_proposal.md`에 정의된 목표(간결한 3개 실험, Meta MTP 네이티브 활용, VESSL 전용 파이프라인, 로컬 M3 테스트 모드)를 달성하면서 최신 PPO/RLHF 베스트 프랙티스를 통합하는 것을 지향한다. 본 문서는 코드 스켈레톤, 디렉터리/파일 역할, 모델·데이터 아티팩트 규격을 한눈에 정리한다.

---

## 1. 프로젝트 목표 재정의

- **간결한 실험 범위**: Baseline MTP, Verifiable Critic WMTP, Rho-1 Weighted 세 실험에 집중한다.
- **Meta 네이티브 파이프라인**: Meta LLaMA MTP reference 구현을 직접 호출한다.
- **VESSL 전용 배포**: storage → VESSL Storage (S3 미사용)로 정리하고, MLflow 서버 구성은 기존 WMTP와 동일하게 재사용한다.
- **로컬 경량 모드**: M3 Mac + MPS 환경에서 tensor decoding layer를 축소한 초경량 모델을 CLI 플래그로 로드할 수 있도록 한다.
- **설정 단순화**: config YAML을 최소화하고 필수 실험 recipe만 유지한다.

---

## 2. Critic-Weighted WMTP 학습 베스트 프랙티스 (2024~2025)

- **TD Error 계산 (표준 Temporal Difference)**:
  - **공식 (표준 RL)**:
    ```python
    # Intermediate tokens (k < T): Bootstrapping
    δ_k = r_k + γV(s_k) - V(s_{k-1})
        = γV(s_k) - V(s_{k-1})  # r_k = 0

    # Terminal token (k = T): Direct reward
    δ_T = R - V(s_{T-1})  # V(terminal) = 0
    ```
  - **MTP 적용**:
    - 토큰 x_k의 가중치 = `exp(δ_k / β)` (β=0.9)
    - 상태 표기: `s_{k-1}` = 토큰 x_k 생성 전 prefix, `s_k` = 토큰 x_k 생성 후
  - **직관**: δ_k는 "토큰 x_k를 선택한 것의 marginal value"
  - **Bootstrapping 효과**:
    - Intermediate: 다음 상태 V(s_k)로 현재 토큰 기여도 추정 → 분산 감소
    - Terminal: 실제 보상 R 직접 사용 → 편향 없음
  - **V function only**: Q function 불필요, terminal state 가정
  - **정규화 불필요**: Binary reward [0,1] 환경에서 td_error 자연 bounded
  - **이론적 근거**:
    - Sutton & Barto "RL: An Introduction" - 표준 TD(0) 공식
    - TDRM (2024): Intermediate bootstrapping + Terminal direct reward
    - Policy Gradient Theorem (Sutton et al. 1999)
- **Token-level TD Weighting (표준 Bootstrapping)**:
  - **Weight 계산**: `weight = exp(td_error / β)` (β=0.9)
  - **TD error**: Intermediate는 bootstrapping, Terminal은 direct reward
  - **Weight clipping**: `min=0.1, max=5.0` (보수적 안정 장치)
  - **Value Head**: Unbounded linear (표현력 유지, RLHF 표준)
  - **No Normalization**: Binary reward [0,1] 환경에서 TD error 자연 bounded [-1,1]
  - **Sample-level filtering 효과**: Incorrect 샘플은 낮은 reward → 음수 TD error → weight < 1
  - IQL/AWR의 exponential weighting 방식 차용 (단, Q function 없이 V function만 사용)
  - Binary reward 특성을 활용하여 whitening 없이 TD error 절대값 의미 유지
  - 각 토큰이 독립적으로 가중치를 받아 suboptimal data도 안전하게 학습 가능
  - 참고: TD error weighting (outcome-based reward for LLMs)
- **Value Head 품질 관리**:
  - Meta 모델 hidden state는 `norm` 적용 후 Value Head에 전달한다.
  - Value loss 클리핑(`value_clip=0.2`)과 drift 방지용 EMA/anchor 손실을 병행한다.
- **Reward 스케일**: Binary reward [0,1] 고정 (정규화 불필요). TD error 자연 bounded [-1,1]로 배치 간 안정성 확보.
- **Critic Continual Learning** (PPO Best Practice):
  - **Stage2에서 Value Loss를 Auxiliary Loss로 추가**: Policy 학습 중 critic도 지속 학습
  - **Loss 구조**: `total_loss = weighted_ce_loss + value_coef * value_loss`
  - **Value Coefficient**: 0.5 (Stable Baselines3 표준) 또는 1.0 (HuggingFace TRL)
  - **Value Loss Clipping**: MSE 또는 Huber loss에 clipping 적용 (clip_range=0.2)
  - **Monitoring**: Value explained variance 추적 (1.0에 가까울수록 이상적)
  - **Gradient Clipping**: Global gradient norm clipping (max_grad_norm=0.5~1.0)
- **추가 모니터링**: Critic drift 감시를 위해 KL 또는 cosine distance를 선택적으로 기록할 수 있다.
- **최신 연구 참고**
  - **Implicit Q-Learning (IQL, Kostrikov et al., 2021)**: Q+V function 학습 후 Advantage(Q-V) 기반 weighting. 우리는 V function만 학습하여 TD error 기반 weighting 사용.
  - **Exponential weighting 방식**: IQL/AWR의 `exp(advantage/β)` 패턴을 차용하되, Q function 없이 TD error로 대체.
  - *AsyPPO*, *PSPO*, *DVPO*, *SFPO*, *VC-PPO* (Value-Calibrated PPO, 2025) 등은 다중 크리틱·소프트 클립핑·전역 가치 모델·value initialization 개선을 제안하며, TD error 계산 아이디어 측면에서 참고 가능하다.
  - Direct Preference Optimization(DPO), RLOO 등 critic-free 접근법은 향후 확장 연구로 문서화한다.

---

## 3. 시스템 아키텍처 개요

```
CLI (uv run python -m weighted_mtp.cli.train)
    ↓
Config Loader (yaml + .env)
    ↓
Runtime Context (seed, device, console, MLflow)
    ↓
Resource Loader
    ├─ ModelBundleLoader (Meta MTP adapter, mini model)
    ├─ DatasetRegistry (JSONL → HF Dataset)
    └─ TokenizerFactory
    ↓
Pipeline Orchestrator
    ├─ Stage 0: 모델 준비 (Adapter, Value Head init)
    ├─ Stage 1: Trunk Pretraining (선택, Verifiable 전용)
    ├─ Stage 2: TD Error 기반 WMTP Training (value inference → weight builder → weighted loss)
    └─ Stage 3: Eval & Artifact Upload
    ↓
Reports (MLflow, console, checkpoints, logs)
```

---

## 4. 디렉터리 스켈레톤과 역할

```
weighted_mtp/
├── base.md
├── docs/
│   ├── 00_ideal_structure.md              # 본 문서
│   ├── ppo_references.md                  # 외부 논문/블로그 요약
│   └── migration_notes.md                 # Legacy → New 변환 기록
├── configs/
│   ├── defaults.yaml                      # 장비/스토리지 공통 설정
│   ├── recipe.baseline.yaml               # 실험 1: Baseline MTP
│   ├── recipe.verifiable.yaml             # 실험 2: Verifiable Critic
│   └── recipe.rho1_weighted.yaml          # 실험 3: Rho-1 Weighted
├── scripts/
│   ├── prepare_local_small_model.py       # 로컬 경량 모델 생성
│   ├── sync_to_vessl_storage.py           # 모델/데이터 업로드
│   ├── validate_datasets.py               # 데이터셋 검증 (스키마, 길이)
│   └── export_mlflow_artifacts.py         # MLflow 로그 추출
├── vendor/
│   ├── __init__.py
│   └── meta_llama/                        # Meta LLaMA reference 구현
│       ├── __init__.py
│       ├── model.py                       # Transformer, ModelArgs
│       ├── generation.py                  # Llama inference
│       └── tokenizer.py                   # Tokenizer
├── src/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── train.py                       # argparse + run_pipeline
│   ├── core/
│   │   ├── config.py                      # Pydantic 모델 (Config, Recipe)
│   │   ├── logging.py                     # Console/파일 로거 초기화
│   │   └── registry.py                    # 가벼운 플러그인 맵
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py                    # JSONL 로딩, HF Dataset 캐시, Stage별 샘플링 전략
│   │   ├── collators.py                   # MTP용 data collator (instruction/input masking)
│   │   ├── transforms.py                  # 토큰 마스킹, truncation
│   │   └── prepare.py                     # 데이터셋 전처리 (스키마 검증)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── meta_mtp/
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py                 # MetaLlamaMTPAdapter
│   │   │   ├── policy.py                  # 정책 헤드 (logits)
│   │   │   └── value_head.py              # Value head 정의 및 로딩
│   │   └── checkpoints.py                 # safetensors 로딩 유틸
│   ├── value_weighting/
│   │   ├── __init__.py
│   │   ├── td_error.py                    # 표준 TD error 계산 (Intermediate: γV(s_k)-V(s_{k-1}), Terminal: R-V(s_{T-1}))
│   │   ├── weight_builder.py              # TD error 기반 가중치 산출 (exp(td_error/β), bootstrapping)
│   │   ├── regularizers.py                # 가중치 클리핑 (min/max)
│   │   └── metrics.py                     # TD error/weight 모니터링
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── training.py                    # run_training_pipeline 진입점
│   │   └── evaluation.py                  # optional eval 파이프라인
│   ├── runtime/
│   │   ├── environment.py                 # seed, torch.backends 설정
│   │   ├── distributed.py                 # FSDP/Deepspeed 옵션
│   │   └── mlflow.py                      # MLflow 초기화 및 로깅
│   └── utils/
│       ├── timers.py
│       ├── checkpointing.py
│       └── metrics.py
├── storage/
│   ├── datasets_v2/                       # 로컬 원본 (VESSL 업로드 전)
│   │   ├── codecontests/
│   │   │   ├── processed/
│   │   │   │   ├── train.jsonl
│   │   │   │   ├── valid.jsonl
│   │   │   │   └── test.jsonl
│   │   │   ├── stats/
│   │   │   └── schema.json
│   │   └── mbpp/
│   │       └── processed/
│   │           └── train.jsonl
│   ├── datasets_local_small/             # 로컬 테스트용 소형 데이터셋
│   ├── models_v2/
│   │   ├── meta-llama-mtp/
│   │   │   ├── configs/
│   │   │   │   ├── params.json
│   │   │   │   └── meta_adapter.yaml     # Adapter 설정 (n_future_tokens 등)
│   │   │   ├── safetensors/
│   │   │   │   └── model.safetensors
│   │   │   ├── tokenizer/
│   │   │   │   └── tokenizer.model
│   │   │   └── metadata.json
│   │   ├── starling-rm-7b/               # (선택) RM 모델
│   │   ├── ref-sheared-llama-2.7b/       # Rho-1 reference
│   │   ├── micro-mtp/                    # 로컬 테스트용 경량 모델
│   │   └── micro-ref/                    # 로컬 테스트용 reference 모델
│   └── README.md                         # 업로드 전 점검표
├── tests/
│   ├── unit/
│   │   ├── test_adapter.py
│   │   ├── test_td_error.py
│   │   └── test_weight_builder.py
│   ├── integration/
│   │   ├── test_stage1_local.py
│   │   └── test_stage2_local.py
│   └── fixtures/
│       └── mini_rollout.pt
└── pyproject.toml
```

### 디렉터리별 세부 역할
- `configs/`: 환경 고정값(`defaults.yaml`) + 실험 recipe만 유지. recipe에는 dataset split, horizon, reward 설정 등 실험 차이만 명시한다. `defaults.yaml`에 모델 파라미터 스냅샷 및 **Stage별 데이터 샘플링 전략** 등록.
  - **Stage별 샘플링 config 예시**:
    ```yaml
    data:
      sampling:
        stage1:
          n_samples: 50000
          balance_correct: true
          correct_ratio: 0.5
          difficulty_range: [1, 11]  # 전체 난이도
          seed: 42
        stage2:
          n_samples: 200000
          curriculum_learning: true
          difficulty_schedule:
            - epoch_range: [0.0, 0.3]    # 초반 30%
              difficulty_weights: {low: 0.7, medium: 0.3, high: 0.0}
            - epoch_range: [0.3, 0.7]    # 중반 40%
              difficulty_weights: {low: 0.3, medium: 0.6, high: 0.1}
            - epoch_range: [0.7, 1.0]    # 후반 30%
              difficulty_weights: {low: 0.1, medium: 0.5, high: 0.4}
          difficulty_bins: {low: [1, 3], medium: [4, 7], high: [8, 11]}
          seed: 42
    ```
- `vendor/meta_llama/`: Meta LLaMA reference 구현을 외부 의존성으로 명시. HuggingFace에서 직접 다운로드하여 배치. 업스트림 업데이트 시 이 디렉터리만 교체.
- `src/models/meta_mtp/`: Meta reference를 래핑하는 adapter와 value head만 포함. `from vendor.meta_llama import Transformer`로 import.
- `src/data/datasets.py`: **Stage별 샘플링 전략** 구현. JSONL에서 HuggingFace Dataset 로딩 시 `is_correct`, `difficulty` 기반 필터링/샘플링.
- `src/data/prepare.py`: 데이터셋 전처리 및 스키마 검증 (instruction, input, output, is_correct, metadata).
- `src/value_weighting/`: TD error 기반 가중치 계산 로직을 기능 단위로 분할하여 테스트 가능하도록 구성.
- `scripts/validate_datasets.py`: 데이터셋 무결성 검증 사용.
- `storage/`: 로컬 실험 리소스 원천. `models_v2/`, `datasets_v2/`, `datasets_local_small/`로 구성. VESSL Storage 업로드 전에 이 구조를 그대로 동기화한다.

---

## 5. 파이프라인 단계별 책임

| 단계 | 모듈 | 주요 작업 | 입력 | 출력 |
|------|------|-----------|------|------|
| Stage 0 | `runtime.environment` | seed, dtype, device 설정 | Config | Torch 전역 상태 |
|        | `models.meta_mtp.adapter` | Meta 모델 로딩 및 `MetaLlamaMTPAdapter` 초기화 | 모델 bundle | Adapter 인스턴스 |
| Stage 1 (옵션) | `pipelines.training.TrunkPretrainer` | trunk_forward 기반 Value Head 사전학습 | Adapter, dataset | pretrain checkpoint |
| Stage 2 | `value_weighting.td_error` | 표준 TD error 계산 (Intermediate: `γV(s_k)-V(s_{k-1})`, Terminal: `R-V(s_{T-1})`) | Adapter, dataset batch | TD error tensor |
|        | `value_weighting.weight_builder` | TD error 기반 가중치 산출 (`exp(td_error/β)`, β=0.9, bootstrapping) | TD error tensor | token weights |
|        | `trainer.wmtp` | 가중치 기반 MTP loss 계산 및 업데이트 | token weights, logits | loss, metrics |
| Stage 3 | `pipelines.training` | 평가, 체크포인트, MLflow 로깅 | Trainer state | artifacts |

---

## 6. 모델 아티팩트 규격

### 6.1 Meta LLaMA MTP (Base, facebook/multi-token-prediction/7B_1T_4)
- **Meta 배포 원본**
  - `7B_1T_4/consolidated.pth` (PyTorch state_dict)
  - `7B_1T_4/params.json` (모델 하이퍼파라미터: dim=4096, n_layers=32, n_heads=32, n_future_tokens=4, rope_theta=10000.0)
  - `tokenizer.model` (LLaMA SentencePiece)
  - `llama/{model.py,generation.py,tokenizer.py,__init__.py}` (레퍼런스 코드)
- **프로젝트 표준 파생물(storage/models_v2/meta-llama-mtp)**
  - `safetensors/model.safetensors` : `consolidated.pth`를 float16 그대로 변환한 파일
  - `configs/params.json` : 원본 `params.json`을 복사(필요 시 추가 필드 포함)
  - `configs/meta_adapter.yaml` : project adapter 설정 (`intermediate_size:11008`, `rope_theta:10000.0`, `dtype:float16`, `n_future_tokens:4` 등)
  - `tokenizer/tokenizer.model` + `tokenizer/tokenizer_config.json`
  - `metadata.json` : 버전, dtype(float16), SHA256, 변환 일자 기록
  - `llama/*.py` : Meta 레퍼런스 코드를 `vendor/meta_llama/`로 이동 후 동기화
- **검증 체크리스트**
  - 변환 전후 dtype 유지(float16) 확인
  - `meta_adapter.yaml`과 `params.json`의 dim/heads/rope 값 일치 검증
  - SHA256 기록 및 MLflow에 업로드 경로 등록

### 6.2 Reference (Rho-1) Model
- **원본 자산**: Microsoft Rho-1(예: `microsoft/rho-math-7b-v0.1`)의 sharded PyTorch `.bin`과 `tokenizer.json`.
- **파생물(storage/models_v2/ref-sheared-llama-2.7b)**
  - `safetensors/model.safetensors` (shard 병합 후 float16 변환)
  - `configs/config.json` (원본 config 복사)
  - `tokenizer/` : Base와 동일 토크나이저 사용을 `metadata.json.tokenizer_shared_with`로 명시
  - `metadata.json` : dtype/SHA256/원본 리포지토리 정보 기록

### 6.3 Reward Model (선택)
- 구조는 Reference 모델과 동일. 현재는 placeholder 상태이므로 `metadata.json`에 `"status": "optional"`을 남기고, 필요 시 safetensors 변환 절차를 동일하게 따른다.

### 6.4 Micro MTP (로컬 테스트)
- `scripts/prepare_local_small_model.py`로 Base safetensors에서 일부 레이어를 슬라이싱하여 생성.
- `safetensors/model.safetensors`, `configs/config.json`(dim 512, layers 4, vocab 32000 등), `tokenizer/`, `metadata.json(target_device:"mps")`를 저장.
- 체크리스트: 파일 크기 <50MB, dtype float16 유지, `tests/unit/test_adapter.py -k micro` 통과.

---

## 7. 데이터셋 규격

### 7.1 Raw CodeContests (HuggingFace 원본: deepmind/code_contests)
- **소스**: HuggingFace datasets 라이브러리를 통해 Parquet 형식으로 로드
- **주요 필드**
  - `name`: 문제 고유 식별자 (예: "brcktsrm")
  - `description`: 문제 설명 (자연어). 예시 입력·출력, 제약조건 포함
  - `public_tests`: `{"input": [...], "output": [...]}` 구조의 공개 테스트 케이스
  - `private_tests`: 비공개 테스트 케이스 (평가용)
  - `solutions`: `{"language": [...], "solution": [...]}` 구조의 정답 솔루션들
  - `incorrect_solutions`: `{"language": [...], "solution": [...]}` 구조의 오답 솔루션들
  - `difficulty`: 문제 난이도
  - `source`: 출처 플랫폼 (Codeforces 등)
  - 기타: `cf_contest_id`, `cf_rating`, `cf_tags` 등 메타데이터
- **처리 방식**: `scripts/setup_datasets.py`가 HuggingFace에서 직접 로드하여 Alpaca 형식으로 변환
- **길이 제약**: Meta LLaMA 토크나이저 기준 2048 토큰 이하로 필터링 (instruction + input + output 합산)
- **분할**: train/valid/test (HuggingFace 기본 split 사용)

### 7.2 Alpaca 스타일 SFT 변환 (storage/datasets_v2/codecontests/processed/*.jsonl)
- **필드**
  - `instruction`: 문제 설명 (HF `description` 필드에서 변환)
  - `input`: 공개 테스트 케이스 예시 (최대 2개, `public_tests`에서 추출)
  - `output`: Python 솔루션 코드 (correct 또는 incorrect)
  - `task_id`: 문제명 + 솔루션 타입 접미사 (예: `"brcktsrm_correct_0"`, `"brcktsrm_incorrect_1"`)
  - `is_correct`: **top-level 필드**로 솔루션 정답 여부 표시 (`true` 또는 `false`)
  - `metadata`: `{"source": "code_contests", "difficulty": <int>, "has_tests": true/false}`
    - `difficulty`: Codeforces 난이도 등급 (1~11, 낮을수록 쉬움)
    - **분포 (train 1000샘플)**: diff=7 (86.7%), diff=2 (6.4%), diff=1 (4.4%), diff=11 (2.1%), diff=6 (0.4%)
    - **활용**: Stage별 Curriculum Learning에서 난이도 기반 샘플링 전략에 사용
- **변환 로직** (`scripts/setup_datasets.py`)
  - Correct solutions: `solutions` 필드의 Python/Python3 솔루션 추출 → `is_correct: true`
  - Incorrect solutions: `incorrect_solutions` 필드의 Python/Python3 솔루션 추출 → `is_correct: false`
  - 모든 솔루션을 **단일 JSONL 파일에 통합 저장** (`processed/train.jsonl` 등)
  - task_id에 `_correct_N` / `_incorrect_N` 접미사로 구분
- **실제 샘플 수** (2025-11-14 기준):
  - **Train**: 3,691,981 samples (correct: 1,754,404 / incorrect: 1,937,577)
  - **Valid**: 14,725 samples (correct: 8,184 / incorrect: 6,541)
  - **Test**: 14,851 samples (correct: 8,038 / incorrect: 6,813)
- **Stage별 샘플링 전략 (메모리 효율 학습)**
  - **Stage 1 (Value Head Pretrain)**:
    - `is_correct` 균형 샘플링: 50% correct, 50% incorrect
    - 샘플 크기: 10,000~50,000 (전체의 0.3~1.4%)
    - Difficulty 무관: 모든 난이도 균등 샘플링
    - 목적: Value head가 correct/incorrect 구분 학습
    - 구현: `load_dataset(stage="stage1", balance_correct=True, correct_ratio=0.5, n_samples=50000)`
  - **Stage 2 (Weighted Training)**:
    - Curriculum Learning: Difficulty 기반 점진적 증가
      - 초반 epoch (0~30%): low (1-3) 70%, medium (4-7) 30%, high (8-11) 0%
      - 중반 epoch (30~70%): low 30%, medium 60%, high 10%
      - 후반 epoch (70~100%): low 10%, medium 50%, high 40%
    - 샘플 크기: 100,000~500,000 (전체의 2.7~13.5%)
    - `is_correct` 혼합: TD error weighting이 자동 필터링 (incorrect → 낮은 weight)
    - 목적: 쉬운 문제부터 학습하여 TD error 안정화, 점진적 난이도 증가
    - 구현: `load_dataset(stage="stage2", curriculum_learning=True, difficulty_schedule=[...], n_samples=200000)`
- **추가 규칙**
  - 토큰 길이 필터링: instruction + input + output 합산이 2048 토큰 초과 시 제외
  - Python/Python3 솔루션만 포함 (언어 코드 1 또는 3)
  - **Loss Masking**: instruction/input 토큰은 labels에서 -100으로 마스킹하여 loss 계산 제외
    - Instruction 토큰: labels = -100 (attention은 유지, loss만 제외)
    - Input 토큰: labels = -100
    - Output 토큰: 실제 token ID (loss 계산 대상)
    - 구현: Data collator에서 instruction/input 길이를 추적하여 자동 마스킹

### 7.3 기타 데이터셋(MBPP, HumanEval 등)
- 동일한 파이프라인(`src/data/prepare.py`)으로 `prompt/response/metadata`를 생성하되, CodeContests 전용 필드(`test_cases`, `is_correct`)가 없는 경우 빈 객체로 채운다.
- 로컬 스몰셋: `train_small.jsonl`(≤100), `validation_small.jsonl`(≤32) 제공. CLI `--dataset-suffix small`로 선택 가능.
- 모든 processed JSONL은 SHA256 및 스키마 검증을 `scripts/validate_datasets.py`에서 통과해야 한다.

---

## 8. 로컬 & VESSL 워크플로우

### 로컬 (M3 Mac, MPS)
```bash
uv sync
TOKENIZERS_PARALLELISM=false \
uv run python -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --recipe configs/recipe.verifiable.yaml \
  --preset local-light \
  --use-micro-model true
```
- `--use-micro-model`: `storage/models_v2/micro-mtp/`를 로드.
- `--preset local-light`: 배치 1, epoch 0.1, Stage 1만 실행 등 초경량 설정 적용.

### VESSL (A100 1~4장)
```bash
vessl run create \
  --cluster vessl-gcp-oregon \
  --resource a100-1gpu \
  --image ghcr.io/wooshikwon/weighted-mtp:latest \
  --name verifiable_critic_prod \
  --env-file .env.vessl \
  --command "uv run python -m weighted_mtp.cli.train \
      --config configs/defaults.yaml \
      --recipe configs/recipe.verifiable.yaml \
      --run-name verifiable_prod_001"
```
- `scripts/sync_to_vessl_storage.py`로 `storage/{models,datasets}` 업로드.
- MLflow URI/인증 정보는 `.env.vessl`에 주입.

---

## 9. 연구 의도와 구현 매핑

본 프로젝트는 세 가지 핵심 실험을 통해 Weighted MTP의 효과를 검증한다. 각 실험의 요구사항과 구현 요소를 매핑하여 일관성을 확보한다.

| 실험 | 학습 데이터 | Weight 메커니즘 | 평가 데이터 | 모델 요구사항 | 구현 모듈 |
|------|------------|----------------|------------|--------------|-----------|
| **Baseline MTP** | CodeContests (correct only) | Uniform (가중치 없음) | MBPP, HumanEval | meta-llama-mtp | `pipelines/training.py` |
| **Verifiable Critic WMTP** | CodeContests (correct + incorrect) | **TD Error Weighting**: `exp(td_error/β)` (IQL 방식 차용, V only) | MBPP, HumanEval | meta-llama-mtp + Value Head | `value_weighting/weight_builder.py`, `value_weighting/td_error.py` |
| **Rho-1 Weighted** | CodeContests (correct only) | Reference loss weighting | MBPP, HumanEval | meta-llama-mtp + ref-sheared-llama-2.7b | `value_weighting/td_error.py` (KL mode), `models/checkpoints.py` |

**데이터셋별 용도:**
- **CodeContests**: 학습용 (is_correct 필드 포함: correct/incorrect solutions)
  - Baseline: `is_correct==True`만 필터링
  - Verifiable Critic: **correct + incorrect 모두 사용** (Direct Weight로 자동 조절)
  - Rho-1: `is_correct==True`만 필터링
- **MBPP, HumanEval**: 평가용 (Pass@K 계산, is_correct 필드 없음)

**Verifiable Critic Weight 메커니즘 상세 (표준 TD Learning):**
```python
# MTP 시나리오: 시점 t에서 H개 미래 토큰 예측
# tokens: x_{t+1}, x_{t+2}, ..., x_{t+H}
# states: s_t, s_{t+1}, ..., s_{t+H}

# 1. 각 토큰의 표준 TD Error 계산
gamma = 1.0  # LLM RLHF 표준 (할인 없음)
reward = is_correct  # Binary: 1.0 or 0.0

for k in range(1, H+1):
    s_before = prefix[:t+k]  # 토큰 x_{t+k} 생성 전: s_{t+k-1}
    s_after = prefix[:t+k+1]  # 토큰 x_{t+k} 생성 후: s_{t+k}

    value_before = value_head(s_before)  # V(s_{t+k-1})

    if k < H:  # Intermediate tokens: Bootstrapping
        value_after = value_head(s_after)  # V(s_{t+k})
        r_k = 0.0  # 중간 토큰은 보상 없음
        td_error_k = r_k + gamma * value_after - value_before
        # = gamma * V(s_{t+k}) - V(s_{t+k-1})
    else:  # Terminal token (k = H): Direct reward
        # V(terminal) = 0 가정
        td_error_k = reward - value_before
        # = R - V(s_{t+H-1})

# td_error_k 특성:
# - Intermediate: γV(s_k) - V(s_{k-1}) → 다음 상태로 bootstrapping (분산 감소)
# - Terminal: R - V(s_{T-1}) → 실제 보상 직접 사용 (편향 없음)
# - Binary reward 환경에서 자연 bounded
# - 직관: "토큰 x_k를 선택한 것의 marginal value"

# 2. Exponential Weighting (표준 IQL/AWR 방식)
beta = 0.9  # Temperature parameter
weight_k = torch.exp(td_error_k / beta)

# 3. Conservative safety clipping
weight_k = torch.clamp(weight_k, min=0.1, max=5.0)

# 4. Incorrect 샘플 자동 down-weighting:
# reward=0, value>0 → td_error<0 → weight<1 (자동 필터링)
# reward=1, value<1 → td_error>0 → weight>1 (강화)
# Bootstrapping으로 value 수렴 가속 → 안정적 학습

# 핵심: 표준 TD Learning (TDRM 2024, Sutton & Barto)
# - Intermediate: Bootstrapping으로 분산 감소
# - Terminal: Direct reward로 편향 제거
# - V function만 사용 (Q function 불필요)
```
