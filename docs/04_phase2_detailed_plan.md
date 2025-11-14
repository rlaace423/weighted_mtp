# Phase 2: 코드 스켈레톤 & 벤더 정리 상세 실행 계획

본 문서는 Phase 2의 실행 시나리오를 Step-by-Step으로 정리한다. 목표는 `docs/00_ideal_structure.md`에서 정의한 목표 구조에 따라 코드베이스 기반을 구축하고, Meta LLaMA MTP 네이티브 파이프라인을 위한 vendor 패키지와 src 모듈 골격을 완성하는 것이다.

---

## Phase 1 완료 현황 요약

### ✅ 완료된 핵심 성과

**모델 자산** (storage/models_v2/):
- ✅ meta-llama-mtp (25GB, float16, SHA256: a29872d0..., n_params: 6.7B)
  - safetensors/model.safetensors
  - configs/params.json, meta_adapter.yaml
  - tokenizer/tokenizer.model
  - metadata.json (repo: facebook/multi-token-prediction, revision: 7B_1T_4)
- ✅ ref-sheared-llama-2.7b (10.3GB, float16, SHA256: 4091b6ac..., n_params: 2.7B)
  - tokenizer_shared_with: meta-llama-mtp
- ✅ starling-rm-7b (25GB, bfloat16, SHA256: cd90dc78..., n_params: 13.3B)
- ✅ micro-mtp (177MB, float16, 4 layers, 512 dim, target_device: mps)
- ✅ micro-ref (177MB, float16, 4 layers, 512 dim)

**데이터셋** (storage/datasets_v2/):
- ✅ HumanEval: 164 test samples (avg 450/180 chars)
- ✅ MBPP: 374 train + 90 validation + 500 test (avg 78/178 chars)
- ✅ CodeContests (correct + incorrect 통합):
  - **완료**: correct solutions + incorrect solutions 완전 통합
  - **실제 샘플 수**: Train **3,691,981** (correct 1,754,404 / incorrect 1,937,577)
  - **Valid**: 14,725 samples (correct 8,184 / incorrect 6,541)
  - **Test**: 14,851 samples (correct 8,038 / incorrect 6,813)
  - **top-level is_correct 필드** 포함 (true/false)
  - task_id 접미사: `_correct_N` / `_incorrect_N`
  - 토큰 필터링: instruction+input+output ≤2048 tokens
- ✅ Small 버전: 각 데이터셋당 100 train + 32 valid/test (storage/datasets_local_small/)

**인프라**:
- ✅ 3개 통합 스크립트: setup_models.py, setup_datasets.py, verify_storage.py
- ✅ metadata.json, SHA256SUMS, stats 완비
- ✅ 문서화 완료 (docs/00~03)

### ✅ Phase 1 완료 사항 및 Phase 2 진입 조건

Phase 1에서 발견된 이슈를 모두 해결하고 다음 조치를 완료했습니다:

1. **✅ 토큰 길이 필터링 구현 완료**
   - CodeContests 최대 output: 49,831자 → 2048 토큰 초과 샘플 제거
   - **해결**: `setup_datasets.py`에 SentencePieceProcessor 통합
   - **구현**: `_filter_by_token_length()` 메서드로 2048 토큰 초과 샘플 제거
   - **통계**: Stats에 토큰 통계 추가 (avg_*_tokens, max_*_tokens)
   - **재처리**: CodeContests valid split 포함하여 재처리 완료

2. **✅ Sheared-LLaMA config.json 위치 수정**
   - **해결**: `raw/config.json` → `configs/config.json` 복사 완료
   - **검증**: verify_storage.py에서 정상 확인

3. **Meta 레퍼런스 코드 준비**
   - facebook/multi-token-prediction의 llama/ 디렉터리 필요
   - **Phase 2 처리**: Step 1에서 git sparse-checkout으로 다운로드 및 vendor/ 구성

4. **✅ CodeContests validation split 수정**
   - **발견**: HuggingFace 실제 split 이름은 "valid" (not "validation")
   - **해결**: `setup_datasets.py`에서 "validation" → "valid" 수정 및 재처리

---

## Phase 2 목표 및 원칙

### 목표
1. **Meta 네이티브 파이프라인 기반 구축**: vendor/meta_llama/ 패키지에 Meta 공식 코드 배치
2. **프로젝트 구조 완성**: src/, configs/, tests/ 디렉터리 생성 및 모듈 골격 작성
3. **개발 환경 정비**: pyproject.toml, ruff, pre-commit 설정
4. **타입 안정성 확보**: 모든 인터페이스에 타입 힌트 적용 (Python 3.10+)

### 원칙 (개발 원칙 준수)
- **[원칙 1]** Meta 레퍼런스 구조를 존중하되, 프로젝트 통합을 위한 최소 변경만 수행
- **[원칙 2]** vendor/와 src/의 책임을 명확히 분리: vendor는 외부 의존성, src는 프로젝트 로직
- **[원칙 4-2]** 단순 wrapper 형태의 메서드 지양, 계층 과도하게 깊게 만들지 않음
- **[원�식 4-3]** 주석은 코드 동작 핵심만, 불필요한 phase2/version 언급 배제
- **[원칙 6]** uv를 프로젝트 의존성 도구로 사용

---

## Step별 상세 실행 계획

### Step 0. 사전 준비 및 Phase 1 완료 확인

**목표**: Phase 1 이슈 해결 확인 및 환경 점검

**작업**:
1. ✅ Sheared-LLaMA config.json 복사 (완료)
   ```bash
   # 이미 완료됨
   ls -lh storage/models_v2/ref-sheared-llama-2.7b/configs/config.json
   ```

2. ✅ CodeContests 재처리 확인 (완료)
   ```bash
   # valid split 및 토큰 필터링 적용 확인
   ls -lh storage/datasets_v2/codecontests/processed/
   cat storage/datasets_v2/codecontests/stats/*_summary.json
   ```

3. 개발 도구 설치 확인
   ```bash
   uv --version  # 0.1.x+
   python --version  # 3.10+
   ```

4. Git 작업 브랜치 생성
   ```bash
   git checkout -b phase2-skeleton
   ```

**산출물**:
- ✅ storage/models_v2/ref-sheared-llama-2.7b/configs/config.json
- ✅ storage/datasets_v2/codecontests/processed/valid.jsonl (신규)
- ✅ storage/datasets_v2/codecontests/stats/*_summary.json (토큰 통계 포함)
- phase2-skeleton 브랜치

**검증**:
```bash
# 모델 검증
uv run python scripts/verify_storage.py --check models
# ref-sheared-llama-2.7b: All checks passed! 확인

# 데이터셋 검증 (valid split 포함)
find storage/datasets_v2/codecontests/processed -name "*.jsonl"
# 예상 출력: train.jsonl, valid.jsonl, test.jsonl 확인

# 샘플 수 확인
wc -l storage/datasets_v2/codecontests/processed/*.jsonl
# train: 3,691,981 / valid: 14,725 / test: 14,851
```

---

### Step 1. Meta 레퍼런스 코드 확보 및 vendor/ 구성

**목표**: Meta LLaMA MTP 공식 구현을 vendor/meta_llama/로 통합

**작업**:
1. Meta 레퍼런스 코드 다운로드
   ```bash
   # facebook/multi-token-prediction 리포지토리의 llama/ 디렉터리 다운로드
   git clone --depth 1 --filter=blob:none --sparse \
     https://github.com/facebookresearch/multi-token-prediction.git /tmp/mtp
   cd /tmp/mtp
   git sparse-checkout set llama
   ```

2. vendor/meta_llama/ 디렉터리 생성 및 코드 복사
   ```bash
   mkdir -p vendor/meta_llama
   cp -r /tmp/mtp/llama/* vendor/meta_llama/
   ```

3. vendor/__init__.py 작성
   ```python
   """
   vendor/ - 외부 의존성 패키지

   Meta LLaMA MTP 공식 구현 등 외부 레퍼런스 코드를 배치한다.
   업스트림 업데이트 시 이 디렉터리만 교체하는 것을 원칙으로 한다.
   """
   ```

4. vendor/meta_llama/__init__.py 수정 (최소 변경)
   ```python
   # Meta 원본 import 구조 유지
   from .model import Transformer, ModelArgs
   from .generation import LLaMA
   from .tokenizer import Tokenizer

   __all__ = ["Transformer", "ModelArgs", "LLaMA", "Tokenizer"]
   ```

5. 버전 정보 기록
   ```bash
   # vendor/meta_llama/VERSION 생성
   echo "facebook/multi-token-prediction" > vendor/meta_llama/VERSION
   echo "commit: $(cd /tmp/mtp && git rev-parse HEAD)" >> vendor/meta_llama/VERSION
   echo "date: $(date -u +%Y-%m-%d)" >> vendor/meta_llama/VERSION
   ```

**산출물**:
- vendor/__init__.py
- vendor/meta_llama/ (model.py, generation.py, tokenizer.py, __init__.py)
- vendor/meta_llama/VERSION

**검증**:
```bash
# import 테스트
uv run python -c "from vendor.meta_llama import Transformer, ModelArgs; print('OK')"
```

**참고**: Meta 코드의 라이선스(LICENSE) 파일도 함께 복사하여 준수한다.

---

### Step 2. pyproject.toml 및 개발 환경 설정

**목표**: uv 기반 의존성 관리 및 개발 도구 구성

**작업**:
1. pyproject.toml 생성
   ```toml
   [project]
   name = "weighted-mtp"
   version = "0.2.0"  # Phase 2 버전
   description = "Weighted Multi-Token Prediction with Critic-based Value Weighting"
   readme = "README.md"
   requires-python = ">=3.10"
   license = {text = "MIT"}

   dependencies = [
       "torch>=2.1.0",
       "safetensors>=0.4.0",
       "transformers>=4.35.0",
       "datasets>=2.15.0",
       "numpy>=1.24.0",
       "pyyaml>=6.0",
       "pydantic>=2.5.0",
       "mlflow>=2.9.0",
       "sentencepiece>=0.1.99",
       "rich>=13.7.0",  # console logging
       "python-dotenv>=1.0.0",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=7.4.0",
       "pytest-cov>=4.1.0",
       "ruff>=0.1.8",
       "black>=23.12.0",
       "mypy>=1.7.0",
       "pre-commit>=3.6.0",
   ]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build]
   packages = ["weighted_mtp", "vendor"]

   [tool.ruff]
   line-length = 100
   target-version = "py310"
   select = ["E", "F", "I", "N", "W", "UP"]
   ignore = ["E501"]  # line-length는 black이 처리

   [tool.black]
   line-length = 100
   target-version = ["py310"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = "test_*.py"
   addopts = "-v --cov=weighted_mtp --cov-report=term-missing"

   [tool.mypy]
   python_version = "3.10"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = false  # 점진적 타입 적용
   ```

2. ruff.toml 생성 (선택, pyproject.toml에 통합 가능)
   ```toml
   # pyproject.toml의 [tool.ruff]로 통합됨
   ```

3. .pre-commit-config.yaml 생성
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.1.8
       hooks:
         - id: ruff
           args: [--fix, --exit-non-zero-on-fix]
     - repo: https://github.com/psf/black
       rev: 23.12.0
       hooks:
         - id: black
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.5.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
   ```

4. .gitignore 업데이트
   ```
   # Python
   __pycache__/
   *.py[cod]
   .venv/
   *.egg-info/

   # IDE
   .vscode/
   .idea/

   # Phase outputs
   storage/models_v2/*/raw/  # 원본 checkpoint는 gitignore
   storage/datasets_v2/*/raw/
   *.safetensors  # 대용량 모델 파일

   # MLflow
   mlruns/

   # Env
   .env
   .env.*
   ```

5. 의존성 설치
   ```bash
   uv pip install -e ".[dev]"
   pre-commit install
   ```

**산출물**:
- pyproject.toml
- .pre-commit-config.yaml
- .gitignore (업데이트)
- uv.lock (자동 생성)

**검증**:
```bash
uv run ruff check .
uv run black --check .
uv run pytest --collect-only  # 테스트 없어도 OK
```

---

### Step 3. src/ 모듈 스켈레톤 생성

**목표**: 00_ideal_structure.md의 디렉터리 구조에 따라 src/ 모듈 골격 작성

**작업**:
1. 디렉터리 생성
   ```bash
   mkdir -p src/weighted_mtp/{cli,core,data,models/meta_mtp,value_weighting,pipelines,runtime,utils}
   mkdir -p tests/{unit,integration,fixtures}
   ```

2. 최상위 src/weighted_mtp/__init__.py
   ```python
   """
   Weighted Multi-Token Prediction (WMTP) 프로젝트

   Meta LLaMA MTP 네이티브 파이프라인을 사용하는 Critic-weighted WMTP 구현.
   세 가지 핵심 실험을 지원한다:
   - Baseline MTP
   - Verifiable Critic WMTP
   - Rho-1 Weighted

   주요 모듈:
   - cli: 사용자 진입점
   - core: 설정, 로깅, 레지스트리
   - data: 데이터 로딩 및 전처리
   - models: Meta Adapter 및 체크포인트 관리
   - value_weighting: TD error 기반 토큰 가중치 계산
   - pipelines: 학습/평가 파이프라인
   - runtime: 환경 초기화, MLflow 연동
   - utils: 공통 유틸리티
   """
   __version__ = "0.2.0"
   ```

3. 각 모듈의 __init__.py 작성 (스텁)

   **src/weighted_mtp/cli/__init__.py**:
   ```python
   """CLI 진입점"""
   ```

   **src/weighted_mtp/core/__init__.py**:
   ```python
   """핵심 설정 및 레지스트리"""
   # config.py: Pydantic Config/Recipe
   # logging.py: Console/파일 로거
   # registry.py: 플러그인 맵
   ```

   **src/weighted_mtp/data/__init__.py**:
   ```python
   """데이터 로딩 및 전처리"""
   # datasets.py: JSONL → HF Dataset
   # collators.py: MTP용 data collator
   # transforms.py: 토큰 마스킹, truncation
   ```

   **src/weighted_mtp/models/__init__.py**:
   ```python
   """모델 관리"""
   ```

   **src/weighted_mtp/models/meta_mtp/__init__.py**:
   ```python
   """Meta LLaMA MTP Adapter"""
   # adapter.py: MetaLlamaMTPAdapter (trunk/full forward)
   # policy.py: 정책 헤드
   # value_head.py: Value head 정의 및 로딩
   ```

   **src/weighted_mtp/value_weighting/__init__.py**:
   ```python
   """TD error 기반 가중치 계산"""
   # td_error.py: GAE, Z-score 정규화
   # weight_builder.py: temperature softmax, clipping
   # regularizers.py: entropy floor, drift 방지
   # metrics.py: TD error/weight 모니터링
   ```

   **src/weighted_mtp/pipelines/__init__.py**:
   ```python
   """학습 및 평가 파이프라인"""
   # training.py: run_training_pipeline (Stage 0-3)
   # evaluation.py: Pass@K, Rho-1 비교
   ```

   **src/weighted_mtp/runtime/__init__.py**:
   ```python
   """런타임 환경 관리"""
   # environment.py: seed, device, dtype 설정
   # distributed.py: FSDP/Deepspeed 옵션
   # mlflow.py: MLflow 초기화 및 로깅
   ```

   **src/weighted_mtp/utils/__init__.py**:
   ```python
   """공통 유틸리티"""
   # timers.py
   # checkpointing.py
   # metrics.py
   ```

4. 기본 타입 정의 (src/weighted_mtp/core/types.py)
   ```python
   """공통 타입 정의"""
   from pathlib import Path
   from typing import TypeAlias

   PathLike: TypeAlias = str | Path
   Device: TypeAlias = str  # "cpu" | "cuda" | "mps"
   DType: TypeAlias = str   # "float16" | "bfloat16" | "float32"
   ```

**산출물**:
- src/weighted_mtp/ (8개 하위 모듈)
- tests/ (unit, integration, fixtures)
- 각 모듈의 __init__.py (docstring 포함)

**검증**:
```bash
uv run python -c "import weighted_mtp; print(weighted_mtp.__version__)"
uv run ruff check src/
```

---

### Step 4. configs/ 기본 설정 파일 작성

**목표**: defaults.yaml과 3개 recipe 템플릿 생성

**작업**:
1. configs/ 디렉터리 생성
   ```bash
   mkdir -p configs
   ```

2. configs/defaults.yaml 작성
   ```yaml
   # 공통 설정 (장비, 스토리지, 모델 파라미터 스냅샷)

   project:
     name: weighted-mtp
     version: "2.0.0"

   storage:
     root: storage
     models_dir: storage/models_v2
     datasets_dir: storage/datasets_v2
     local_small_dir: storage/datasets_local_small

   models:
     policy:
       name: meta-llama-mtp
       path: storage/models_v2/meta-llama-mtp
       params:
         dim: 4096
         n_layers: 32
         n_heads: 32
         n_future_tokens: 4
         intermediate_size: 11008
         rope_theta: 10000.0
         vocab_size: 32000
       dtype: float16

     reference:
       name: ref-sheared-llama-2.7b
       path: storage/models_v2/ref-sheared-llama-2.7b
       dtype: float16
       tokenizer_shared_with: meta-llama-mtp

     reward:
       name: starling-rm-7b
       path: storage/models_v2/starling-rm-7b
       dtype: bfloat16
       status: optional

   runtime:
     device: cuda  # cpu, cuda, mps
     seed: 42
     mixed_precision: true

   mlflow:
     tracking_uri: file://./mlruns
     experiment_name: wmtp-experiments

   logging:
     level: INFO
     format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
   ```

3. configs/recipe.baseline.yaml 작성
   ```yaml
   # 실험 1: Baseline MTP (가중치 없음)

   experiment:
     name: baseline-mtp
     description: "Standard MTP without token weighting"

   dataset:
     name: codecontests
     split:
       train: storage/datasets_v2/codecontests/processed/train.jsonl
       validation: null  # CodeContests는 validation 없음
       test: storage/datasets_v2/codecontests/processed/test.jsonl
     max_length: 2048
     use_small: false  # true로 설정 시 datasets_local_small/ 사용

   training:
     stages:
       - stage1_trunk_pretrain: false  # Baseline은 Stage1 skip
       - stage2_weighted_training: true

     stage2:
       use_weighting: false  # Baseline: 가중치 미사용
       batch_size: 4
       gradient_accumulation_steps: 4
       learning_rate: 1.0e-5
       num_epochs: 3
       warmup_steps: 100

   evaluation:
     metrics: [loss, perplexity]
     pass_at_k: [1, 5, 10]
   ```

4. configs/recipe.verifiable.yaml 작성
   ```yaml
   # 실험 2: Verifiable Critic WMTP

   experiment:
     name: verifiable-critic-wmtp
     description: "WMTP with TD error-based token weighting"

   dataset:
     name: codecontests
     split:
       train: storage/datasets_v2/codecontests/processed/train.jsonl
       validation: null
       test: storage/datasets_v2/codecontests/processed/test.jsonl
     max_length: 2048
     use_small: false

   training:
     stages:
       - stage1_trunk_pretrain: true  # Value head 사전학습
       - stage2_weighted_training: true

     stage1:
       num_epochs: 1
       batch_size: 8
       learning_rate: 5.0e-5

     stage2:
       use_weighting: true
       batch_size: 4
       gradient_accumulation_steps: 4
       learning_rate: 1.0e-5
       num_epochs: 3
       warmup_steps: 100

       # TD error 안정화
       value_weighting:
         gae_gamma: 0.99
         gae_lambda: 0.95
         td_error_normalization: zscore  # zscore | minmax | none
         weight_temperature: 1.0
         weight_clip_min: 0.1
         weight_clip_max: 5.0
         entropy_floor: 0.01

   evaluation:
     metrics: [loss, perplexity, td_error_mean, td_error_std, weight_entropy]
     pass_at_k: [1, 5, 10]
   ```

5. configs/recipe.rho1_weighted.yaml 작성
   ```yaml
   # 실험 3: Rho-1 Weighted

   experiment:
     name: rho1-weighted
     description: "Token weighting based on reference model loss difference"

   dataset:
     name: codecontests
     split:
       train: storage/datasets_v2/codecontests/processed/train.jsonl
       validation: null
       test: storage/datasets_v2/codecontests/processed/test.jsonl
     max_length: 2048
     use_small: false

   training:
     stages:
       - stage1_trunk_pretrain: false  # Rho-1은 reference loss 사용
       - stage2_weighted_training: true

     stage2:
       use_weighting: true
       batch_size: 4
       gradient_accumulation_steps: 4
       learning_rate: 1.0e-5
       num_epochs: 3
       warmup_steps: 100

       # Rho-1 weighting
       value_weighting:
         method: rho1  # rho1 | td_error
         reference_model: ref-sheared-llama-2.7b
         temperature: 1.0
         weight_clip_min: 0.1
         weight_clip_max: 5.0
         kl_monitoring: true  # KL divergence 모니터링

   evaluation:
     metrics: [loss, perplexity, reference_loss_diff, weight_entropy, kl_divergence]
     pass_at_k: [1, 5, 10]
   ```

6. configs/local-light.yaml (로컬 테스트 preset)
   ```yaml
   # 로컬 M3 Mac 경량 테스트 preset

   override:
     storage:
       models_dir: storage/models_v2

     models:
       policy:
         name: micro-mtp
         path: storage/models_v2/micro-mtp
       reference:
         name: micro-ref
         path: storage/models_v2/micro-ref

     dataset:
       use_small: true  # datasets_local_small/ 사용
       max_length: 512

     training:
       stage1:
         num_epochs: 0.1
         batch_size: 1
       stage2:
         num_epochs: 0.1
         batch_size: 1
         gradient_accumulation_steps: 1

     runtime:
       device: mps
       mixed_precision: false
   ```

**산출물**:
- configs/defaults.yaml
- configs/recipe.baseline.yaml
- configs/recipe.verifiable.yaml
- configs/recipe.rho1_weighted.yaml
- configs/local-light.yaml

**검증**:
```bash
uv run python -c "import yaml; yaml.safe_load(open('configs/defaults.yaml'))"
```

---

### Step 5. 기본 CLI 스텁 작성

**목표**: 최소 기능 CLI 진입점 구현 (dry-run 지원)

**작업**:
1. src/weighted_mtp/cli/train.py 작성
   ```python
   """학습 파이프라인 CLI 진입점"""
   import argparse
   from pathlib import Path
   import yaml
   from rich.console import Console

   console = Console()


   def load_config(config_path: Path, recipe_path: Path | None = None) -> dict:
       """설정 파일 로딩"""
       with open(config_path) as f:
           config = yaml.safe_load(f)

       if recipe_path:
           with open(recipe_path) as f:
               recipe = yaml.safe_load(f)
           # 단순 병합 (Phase 6에서 Pydantic으로 교체)
           config.update(recipe)

       return config


   def main():
       parser = argparse.ArgumentParser(
           description="Weighted MTP 학습 파이프라인"
       )
       parser.add_argument(
           "--config",
           type=Path,
           default=Path("configs/defaults.yaml"),
           help="기본 설정 파일 경로",
       )
       parser.add_argument(
           "--recipe",
           type=Path,
           help="실험 recipe 파일 (baseline, verifiable, rho1_weighted)",
       )
       parser.add_argument(
           "--preset",
           choices=["local-light"],
           help="사전 정의된 preset",
       )
       parser.add_argument(
           "--use-micro-model",
           action="store_true",
           help="Micro 모델 사용 (로컬 테스트)",
       )
       parser.add_argument(
           "--dry-run",
           action="store_true",
           help="설정만 출력하고 종료",
       )
       parser.add_argument(
           "--run-name",
           help="MLflow 실험 run 이름",
       )

       args = parser.parse_args()

       # 설정 로딩
       config = load_config(args.config, args.recipe)

       if args.preset == "local-light":
           preset_path = Path("configs/local-light.yaml")
           with open(preset_path) as f:
               preset = yaml.safe_load(f)
           config.update(preset.get("override", {}))

       if args.use_micro_model:
           config["models"]["policy"]["name"] = "micro-mtp"
           config["models"]["policy"]["path"] = "storage/models_v2/micro-mtp"

       # Dry-run
       if args.dry_run:
           console.print("[bold green]Dry-run mode: 설정 확인[/bold green]")
           console.print(yaml.dump(config, default_flow_style=False))
           return

       # TODO: Phase 6에서 실제 파이프라인 연결
       console.print("[yellow]Phase 2: 파이프라인 미구현 (스텁)[/yellow]")
       console.print(f"실험: {config.get('experiment', {}).get('name', 'N/A')}")
       console.print(f"모델: {config['models']['policy']['name']}")
       console.print(f"데이터셋: {config.get('dataset', {}).get('name', 'N/A')}")


   if __name__ == "__main__":
       main()
   ```

2. weighted_mtp 패키지를 모듈로 실행 가능하게 설정
   ```bash
   # src/weighted_mtp/__main__.py 생성
   echo 'from weighted_mtp.cli.train import main; main()' > src/weighted_mtp/__main__.py
   ```

**산출물**:
- src/weighted_mtp/cli/train.py
- src/weighted_mtp/__main__.py

**검증**:
```bash
uv run python -m weighted_mtp --dry-run
uv run python -m weighted_mtp --config configs/defaults.yaml \
  --recipe configs/recipe.baseline.yaml --dry-run
```

---

### Step 6. tests/ 기본 구조 및 스켈레톤 테스트 작성

**목표**: pytest 구조 확립 및 import 테스트

**작업**:
1. tests/conftest.py 작성
   ```python
   """pytest 공통 fixture"""
   import pytest
   from pathlib import Path


   @pytest.fixture
   def project_root() -> Path:
       """프로젝트 루트 경로"""
       return Path(__file__).parent.parent


   @pytest.fixture
   def storage_root(project_root: Path) -> Path:
       """storage/ 경로"""
       return project_root / "storage"


   @pytest.fixture
   def micro_model_path(storage_root: Path) -> Path:
       """Micro MTP 모델 경로"""
       return storage_root / "models_v2" / "micro-mtp"
   ```

2. tests/unit/test_imports.py
   ```python
   """모듈 import 테스트"""
   import pytest


   def test_import_weighted_mtp():
       """weighted_mtp 패키지 import"""
       import weighted_mtp
       assert weighted_mtp.__version__ == "0.2.0"


   def test_import_vendor_meta_llama():
       """vendor.meta_llama import"""
       from vendor.meta_llama import Transformer, ModelArgs
       assert Transformer is not None
       assert ModelArgs is not None


   def test_import_submodules():
       """하위 모듈 import"""
       import weighted_mtp.cli
       import weighted_mtp.core
       import weighted_mtp.data
       import weighted_mtp.models
       import weighted_mtp.value_weighting
       import weighted_mtp.pipelines
       import weighted_mtp.runtime
       import weighted_mtp.utils
   ```

3. tests/unit/test_config.py
   ```python
   """Config 로딩 테스트"""
   import pytest
   from pathlib import Path
   import yaml


   def test_load_defaults_config(project_root: Path):
       """defaults.yaml 로딩"""
       config_path = project_root / "configs" / "defaults.yaml"
       with open(config_path) as f:
           config = yaml.safe_load(f)

       assert "project" in config
       assert config["project"]["name"] == "weighted-mtp"
       assert config["models"]["policy"]["name"] == "meta-llama-mtp"


   @pytest.mark.parametrize("recipe_name", [
       "recipe.baseline.yaml",
       "recipe.verifiable.yaml",
       "recipe.rho1_weighted.yaml",
   ])
   def test_load_recipe_configs(project_root: Path, recipe_name: str):
       """Recipe YAML 로딩"""
       recipe_path = project_root / "configs" / recipe_name
       with open(recipe_path) as f:
           recipe = yaml.safe_load(f)

       assert "experiment" in recipe
       assert "dataset" in recipe
       assert "training" in recipe
   ```

4. tests/integration/ (placeholder)
   ```bash
   touch tests/integration/__init__.py
   touch tests/integration/test_stage1_local.py  # Phase 6에서 구현
   touch tests/integration/test_stage2_local.py  # Phase 6에서 구현
   ```

**산출물**:
- tests/conftest.py
- tests/unit/test_imports.py
- tests/unit/test_config.py
- tests/integration/ (placeholder)

**검증**:
```bash
uv run pytest tests/unit/ -v
# 모든 테스트 통과 확인
```

---

### Step 7. 문서 업데이트 및 Phase 2 검증

**목표**: 문서 정합성 확보 및 Phase 2 체크리스트 완료

**작업**:
1. README.md 생성
   ```markdown
   # Weighted Multi-Token Prediction (WMTP)

   Meta LLaMA MTP 네이티브 파이프라인을 사용하는 Critic-weighted WMTP 구현.

   ## 프로젝트 구조

   ```
   weighted_mtp/
   ├── vendor/meta_llama/  # Meta 레퍼런스 코드
   ├── src/weighted_mtp/   # 프로젝트 소스
   ├── configs/            # 설정 파일
   ├── storage/            # 모델 및 데이터 자산
   ├── tests/              # 테스트
   ├── scripts/            # 유틸리티 스크립트
   └── docs/               # 문서
   ```

   ## 빠른 시작

   ```bash
   # 의존성 설치
   uv pip install -e ".[dev]"

   # 로컬 테스트 (Micro 모델)
   uv run python -m weighted_mtp \
     --config configs/defaults.yaml \
     --recipe configs/recipe.baseline.yaml \
     --preset local-light \
     --dry-run

   # 테스트 실행
   uv run pytest tests/unit/
   ```

   ## 실험

   - **Baseline MTP**: 표준 MTP (가중치 없음)
   - **Verifiable Critic WMTP**: TD error 기반 토큰 가중치
   - **Rho-1 Weighted**: Reference 모델 loss 차이 기반 가중치

   ## 문서

   - [00_ideal_structure.md](docs/00_ideal_structure.md): 이상적 구조
   - [01_storage_preparation_plan.md](docs/01_storage_preparation_plan.md): Storage 재구성
   - [02_implementation_plan.md](docs/02_implementation_plan.md): 전체 구현 계획
   - [03_phase1_detailed_plan.md](docs/03_phase1_detailed_plan.md): Phase 1 상세
   - [04_phase2_detailed_plan.md](docs/04_phase2_detailed_plan.md): Phase 2 상세

   ## 라이선스

   MIT License
   ```

2. docs/phase2_checklist.md 생성
   ```markdown
   # Phase 2 완료 체크리스트

   ## ✅ Step 0: 사전 준비
   - [ ] Sheared-LLaMA config.json 복사 완료
   - [ ] 개발 도구 설치 확인 (uv, python 3.10+)
   - [ ] phase2-skeleton 브랜치 생성

   ## ✅ Step 1: vendor/ 구성
   - [ ] Meta 레퍼런스 코드 다운로드
   - [ ] vendor/meta_llama/ 생성
   - [ ] vendor/meta_llama/VERSION 기록
   - [ ] import 테스트 통과

   ## ✅ Step 2: pyproject.toml 설정
   - [ ] pyproject.toml 생성
   - [ ] .pre-commit-config.yaml 생성
   - [ ] .gitignore 업데이트
   - [ ] uv pip install -e ".[dev]" 성공
   - [ ] ruff, black 검사 통과

   ## ✅ Step 3: src/ 스켈레톤
   - [ ] 8개 하위 모듈 디렉터리 생성
   - [ ] 각 모듈 __init__.py 작성
   - [ ] weighted_mtp.__version__ import 성공

   ## ✅ Step 4: configs/ 작성
   - [ ] defaults.yaml 생성
   - [ ] 3개 recipe 생성 (baseline, verifiable, rho1)
   - [ ] local-light.yaml 생성
   - [ ] YAML 로딩 테스트 통과

   ## ✅ Step 5: CLI 스텁
   - [ ] cli/train.py 작성
   - [ ] __main__.py 생성
   - [ ] --dry-run 모드 동작 확인

   ## ✅ Step 6: tests/ 기본 구조
   - [ ] conftest.py 작성
   - [ ] test_imports.py 통과
   - [ ] test_config.py 통과

   ## ✅ Step 7: 문서 업데이트
   - [ ] README.md 생성
   - [ ] phase2_checklist.md 작성
   - [ ] 00~04 문서 정합성 확인

   ## ✅ Step 8: 최종 검증
   - [ ] 전체 테스트 통과 (uv run pytest)
   - [ ] linting 통과 (uv run ruff check)
   - [ ] import 모든 경로 정상
   - [ ] Git 커밋 및 PR 준비
   ```

3. docs/00_ideal_structure.md 업데이트 (필요 시)
   - Phase 2 완료 항목 표시
   - vendor/meta_llama/ 구조 반영

**산출물**:
- README.md
- docs/phase2_checklist.md
- docs/00_ideal_structure.md (업데이트)

**검증**:
- 모든 체크리스트 항목 ✅
- 문서 간 모순 없음

---

### Step 8. 최종 검증 및 Phase 3 착수 조건 확인

**목표**: Phase 2 완료 확인 및 Git 커밋

**작업**:
1. 전체 테스트 실행
   ```bash
   uv run pytest tests/ -v
   ```

2. 코드 품질 검증
   ```bash
   uv run ruff check .
   uv run black --check .
   ```

3. Import 경로 검증
   ```bash
   uv run python -c "
   import weighted_mtp
   from vendor.meta_llama import Transformer
   from weighted_mtp.cli.train import main
   print('All imports OK')
   "
   ```

4. Git 커밋
   ```bash
   git add .
   git commit -m "feat: Phase 2 완료 - 코드 스켈레톤 & vendor 정리

   - vendor/meta_llama/ 패키지 구성
   - src/weighted_mtp/ 8개 모듈 스켈레톤 생성
   - pyproject.toml, pre-commit 설정
   - configs/defaults.yaml 및 3개 recipe 작성
   - CLI --dry-run 모드 구현
   - tests/unit/ 기본 테스트 추가

   Phase 2 체크리스트 전항목 완료.
   "
   ```

**산출물**:
- phase2-skeleton 브랜치 커밋
- Phase 2 완료 보고서 (선택)

**검증**:
- [ ] pytest 전체 통과
- [ ] ruff, black 통과
- [ ] import 모든 경로 정상
- [ ] Git 커밋 완료

---

## Phase 2 완료 체크리스트 (요약)

### 코드베이스
- [ ] vendor/meta_llama/ 패키지 생성 및 Meta 코드 배치
- [ ] src/weighted_mtp/ 8개 모듈 디렉터리 및 __init__.py 작성
- [ ] pyproject.toml, .pre-commit-config.yaml, .gitignore 설정
- [ ] CLI --dry-run 모드 구현 (python -m weighted_mtp)

### 설정 파일
- [ ] configs/defaults.yaml (모델 파라미터 스냅샷 포함)
- [ ] configs/recipe.baseline.yaml
- [ ] configs/recipe.verifiable.yaml
- [ ] configs/recipe.rho1_weighted.yaml
- [ ] configs/local-light.yaml

### 테스트
- [ ] tests/conftest.py
- [ ] tests/unit/test_imports.py (통과)
- [ ] tests/unit/test_config.py (통과)
- [ ] tests/integration/ (placeholder)

### 문서
- [ ] README.md
- [ ] docs/phase2_checklist.md
- [ ] docs/04_phase2_detailed_plan.md (본 문서)

### 검증
- [ ] uv run pytest tests/ -v (전체 통과)
- [ ] uv run ruff check . (경고 없음)
- [ ] uv run python -m weighted_mtp --dry-run (정상 동작)
- [ ] Git 커밋 완료

---

## 위험 요소 및 대응

| 위험 | 영향 | 대응 전략 |
|------|------|-----------|
| Meta 레퍼런스 코드 버전 불일치 | Adapter 구현 실패 | VERSION 파일로 commit hash 기록, 필요 시 특정 commit checkout |
| pyproject.toml 의존성 충돌 | 설치 실패 | uv.lock으로 버전 고정, 최소 버전만 명시 |
| import 경로 오류 | 테스트 실패 | pyproject.toml의 packages 설정 재확인, PYTHONPATH 점검 |
| pre-commit 훅 과부하 | 개발 지연 | 핵심 검사만 활성화, --no-verify 옵션 문서화 |
| CLI --dry-run 미동작 | 설정 검증 불가 | yaml.safe_load 예외 처리, rich console 오류 표시 |

---

## Phase 3 착수 조건

다음 항목이 모두 완료되어야 Phase 3(데이터 파이프라인 구현)로 진행 가능:

1. **코드베이스 검증**:
   - [ ] vendor/meta_llama/ import 성공
   - [ ] weighted_mtp 전체 모듈 import 성공
   - [ ] pytest 전체 통과

2. **설정 파일 검증**:
   - [ ] defaults.yaml 및 3개 recipe YAML 로딩 성공
   - [ ] --dry-run 모드로 설정 병합 확인

3. **문서 정합성**:
   - [ ] 00~04 문서 간 모순 없음
   - [ ] Phase 2 체크리스트 전항목 ✅

4. **Git 관리**:
   - [ ] phase2-skeleton 브랜치 커밋
   - [ ] 코드 리뷰 준비 (필요 시)

---

## Phase 3 미리보기

**Phase 3: 데이터 파이프라인 구현**의 핵심 작업:
1. src/data/datasets.py: JSONL → HF Dataset 로딩
2. src/data/transforms.py: 토큰 길이 필터링 및 truncation (CodeContests 대응)
3. src/data/collators.py: MTP용 data collator (n_future_tokens 처리, loss masking)
4. src/data/prepare.py: 배치 전처리 파이프라인
5. tests/unit/test_data.py: 데이터 로딩 테스트

**Phase 1 발견사항 반영**:
- **Correct + Incorrect solutions 통합 처리**: 단일 JSONL에 모두 저장 (3.7M samples)
- **Top-level is_correct 필드**: true/false로 솔루션 정답 여부 표시
- **Task ID 접미사**: `_correct_N` / `_incorrect_N`으로 구분
- **토큰 필터링**: instruction+input+output ≤2048 tokens (SentencePieceProcessor 사용)
- **HuggingFace 직접 로드**: Parquet → Alpaca JSONL 변환 (raw/ 중간 저장 불필요)
- **Split 명명**: HuggingFace 원본 "valid" split 사용 (train/valid/test)
- **Loss Masking**: instruction/input 토큰은 labels=-100으로 마스킹, output만 학습
- metadata 보존 (source, difficulty, has_tests) → MLflow 로깅

**Phase 5+ 미리보기 - Stage2 Loss 구조**:
- **Critic Continual Learning**: Stage2에서 value loss를 auxiliary loss로 추가
- **Loss 구조**: `total_loss = weighted_ce_loss + value_coef * value_loss`
- **Value Coefficient**: 0.5 (Stable Baselines3) or 1.0 (HuggingFace TRL)
- **Value Loss Clipping**: clip_range=0.2로 안정화
- **Monitoring**: Value explained variance, TD error mean/std 추적

---

## 참고 자료

- [00_ideal_structure.md](./00_ideal_structure.md): 디렉터리 구조 및 파이프라인 설계
- [01_storage_preparation_plan.md](./01_storage_preparation_plan.md): Storage 자산 규격
- [02_implementation_plan.md](./02_implementation_plan.md): 전체 Phase 개요
- [03_phase1_detailed_plan.md](./03_phase1_detailed_plan.md): Phase 1 실행 결과
- Meta LLaMA MTP: https://github.com/facebookresearch/multi-token-prediction
- uv 문서: https://github.com/astral-sh/uv

---

**이 문서는 Phase 2 실행 전 리뷰 및 승인용으로 사용하며, 실행 중 발견되는 변경 사항은 즉시 반영한다.**
