# Phase 6: Config & CLI 체계 구현 가이드

## 문서 개요

본 문서는 **Phase 6: Config & CLI 체계 구현**을 위한 실행 가이드입니다. Phase 5에서 구현된 학습 파이프라인을 사용자가 CLI로 실행 가능하도록 만들고, 3가지 실험(Baseline, Verifiable Critic, Rho-1 Weighted)을 설정 기반으로 자동 실행하는 시스템을 구축합니다.

**버전**: v1.0 (2025-11-16)
**선행 조건**: Phase 5 (학습 파이프라인) 완료
**목표**: CLI 진입점 → Config 로딩 → 파이프라인 실행 → MLflow 로깅

---

## Part 1: 개요 및 맥락

### 1.1 Phase 6의 위치와 목적

Phase 6는 **사용자 진입점**과 **파이프라인 연결**의 최종 구간입니다.

```
Phase 5 (pipeline)  →  [Phase 6 (config/CLI)]  →  사용자 실행
  run_training_pipeline()     CLI argparse + Config           실험 자동화
```

**핵심 질문**: 어떻게 사용자가 간단한 명령어로 3가지 실험을 실행할 수 있게 만들 것인가?

### 1.2 현재 상태 분석

**기존 구현 현황** (`cli/train.py`):
- ✅ `deep_merge()`: Config 중첩 병합 구현
- ✅ `load_config()`: YAML 로딩 구현
- ✅ argparse 기본 골격 (--config, --recipe, --preset, --use-micro-model, --dry-run, --run-name)
- ❌ 파이프라인 연결 없음 (TODO 주석만 존재)

**기존 구현 현황** (`pipelines/training.py`):
- ✅ `run_training_pipeline()`: Stage 1/2 오케스트레이션 완료
- ✅ Fractional epochs 지원 (0.5, 2.5)
- ✅ Checkpoint 저장, Metrics 수집
- ❌ CLI와 연결 없음

**기존 Config 파일**:
- ✅ `configs/defaults.yaml`: 공통 설정 존재
- ✅ `configs/recipe.*.yaml`: 3개 실험 레시피 존재
- ❌ Config 구조와 파이프라인 입력 불일치

### 1.3 Phase 6 핵심 목표

**목표 1: CLI 진입점 완성**
- `uv run python -m weighted_mtp.cli.train` 명령으로 학습 실행 가능
- --recipe 옵션으로 3가지 실험 자동 선택

**목표 2: Config → Pipeline 연결**
- YAML 설정을 `run_training_pipeline()` 함수 인자로 변환
- 모델, 데이터, 학습 파라미터 자동 매핑

**목표 3: 환경별 실행 모드**
- 로컬 모드: `--preset local-light` + `--use-micro-model`
- 분산 모드: FSDP 4-GPU (VESSL 환경)

**목표 4: MLflow 연동**
- Experiment 자동 생성, Run 이름 설정
- Metrics/Artifacts 자동 로깅

---

## Part 2: 문제 분석 및 설계

### 2.1 현재 상태의 Gap 분석

#### Gap 1: Config 구조 불일치

**문제**: `defaults.yaml`과 `run_training_pipeline()` 입력 형식 불일치

**defaults.yaml 구조**:
```yaml
training:
  stage1:
    n_epochs: 0.5
    learning_rate: 1.0e-4
    loss_type: mse
  stage2:
    n_epochs: 2.5
    beta: 0.9
    value_coef: 0.5
```

**run_training_pipeline() 요구 형식**:
```python
config = {
    "stage1": {
        "n_epochs": 0.5,
        "loss_type": "mse",
        "learning_rate": 1e-4,
    },
    "stage2": {
        "n_epochs": 2.5,
        "beta": 0.9,
        "value_coef": 0.5,
        "learning_rate": 1e-5,
    },
}
```

**해결 방안**: Config 추출 함수 구현 (`_extract_training_config()`)

#### Gap 2: 모델/데이터 로딩 누락

**문제**: CLI에서 모델/데이터 로딩 로직 없음

**필요 단계**:
1. Config에서 모델 경로 추출
2. `MetaLlamaMTPAdapter` 로딩
3. Tokenizer 로딩
4. Stage별 Dataset 로딩 (stage1/stage2 샘플링 전략)
5. DataLoader 생성 (collator 포함)

**해결 방안**: Resource 로딩 함수 구현 (`_load_resources()`)

#### Gap 3: 분산학습 환경 초기화 누락

**문제**: CLI에서 분산학습 환경 설정 없음

**필요 단계**:
1. torch.distributed 초기화 (NCCL backend)
2. Rank/World size 설정
3. Device 할당 (cuda:{rank})
4. Seed 설정 (base_seed + rank)

**해결 방안**: 분산학습 초기화 함수 (`runtime/distributed.py`에서 이미 구현됨)

#### Gap 4: MLflow 연동 누락

**문제**: CLI에서 MLflow 초기화 없음

**필요 단계**:
1. MLflow experiment 생성/로딩
2. Run 시작 (run_name 설정)
3. Config 로깅 (params)
4. Metrics 로깅 (실시간)
5. Checkpoint 업로드 (artifacts)

**해결 방안**: MLflow 초기화 함수 (`runtime/mlflow.py` 구현 필요)

### 2.2 설계 원칙

**원칙 1: 기존 구조 존중**
- Phase 5 `run_training_pipeline()` 인터페이스 변경 금지
- Config deep_merge 방식 유지
- Data collator (AlpacaDataCollator) 재사용

**원칙 2: 점진적 구현**
- Step 1: 로컬 단일 GPU 모드 먼저 구현
- Step 2: 분산학습 모드 추가
- Step 3: MLflow 연동

**원칙 3: 실험 독립성**
- Baseline, Verifiable, Rho-1 각각 독립 실행 가능
- Recipe 파일만 바꾸면 자동 전환

**원칙 4: 에러 처리 강화**
- Config validation (필수 필드 확인)
- 모델/데이터 경로 존재 확인
- Device 호환성 확인

---

## Part 3: Step별 구현 계획

### Step 1: Config 추출 및 검증 함수 구현

#### 목표
YAML config를 `run_training_pipeline()` 입력 형식으로 변환합니다.

#### 구현 파일
- `src/weighted_mtp/cli/train.py`

#### 핵심 함수

**1) _extract_training_config()**
```python
def _extract_training_config(config: dict) -> dict:
    """YAML config에서 학습 파라미터 추출

    Args:
        config: 전체 YAML config (defaults + recipe 병합)

    Returns:
        run_training_pipeline() 입력 형식 config
        {
            "stage1": {"n_epochs": float, "loss_type": str, "learning_rate": float},
            "stage2": {"n_epochs": float, "beta": float, "value_coef": float, ...},
            "save_checkpoint_every": int,
        }

    Raises:
        ValueError: 필수 필드 누락 시
    """
    training_cfg = config.get("training", {})

    # Stage 1 config
    stage1_cfg = training_cfg.get("stage1", {})
    stage1 = {
        "n_epochs": stage1_cfg.get("n_epochs", 0.5),
        "loss_type": stage1_cfg.get("loss_type", "mse"),
        "learning_rate": stage1_cfg.get("learning_rate", 1e-4),
    }

    # Stage 2 config
    stage2_cfg = training_cfg.get("stage2", {})
    stage2 = {
        "n_epochs": stage2_cfg.get("n_epochs", 2.5),
        "beta": stage2_cfg.get("beta", 0.9),
        "value_coef": stage2_cfg.get("value_coef", 0.5),
        "max_grad_norm": stage2_cfg.get("max_grad_norm", 0.5),
        "loss_type": stage2_cfg.get("loss_type", "mse"),
        "learning_rate": stage2_cfg.get("learning_rate", 1e-5),
        "weight_clip_min": stage2_cfg.get("weight_clip_min", 0.1),
        "weight_clip_max": stage2_cfg.get("weight_clip_max", 5.0),
    }

    return {
        "stage1": stage1,
        "stage2": stage2,
        "save_checkpoint_every": training_cfg.get("save_checkpoint_every", 1),
    }
```

**2) _validate_config()**
```python
def _validate_config(config: dict) -> None:
    """Config 필수 필드 검증

    Args:
        config: 전체 YAML config

    Raises:
        ValueError: 필수 필드 누락 또는 형식 오류
    """
    # 필수 섹션 확인
    required_sections = ["project", "models", "dataset", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config 필수 섹션 누락: {section}")

    # 모델 경로 확인
    policy_path = Path(config["models"]["policy"]["path"])
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy 모델 경로 없음: {policy_path}")

    # 데이터셋 경로 확인
    dataset_split = config["dataset"].get("split", {})
    train_path = Path(dataset_split.get("train", ""))
    if not train_path.exists():
        raise FileNotFoundError(f"학습 데이터셋 경로 없음: {train_path}")

    # Stage 설정 확인
    training = config.get("training", {})
    if "stage1" not in training or "stage2" not in training:
        raise ValueError("Config에 stage1 또는 stage2 설정 누락")
```

#### 검증 기준
- [ ] defaults.yaml 로딩 후 추출 성공
- [ ] recipe 병합 후 추출 성공
- [ ] 필수 필드 누락 시 ValueError
- [ ] 모델/데이터 경로 미존재 시 FileNotFoundError

---

### Step 2: Resource 로딩 함수 구현

#### 목표
Config를 기반으로 모델, 토크나이저, 데이터셋, DataLoader를 로딩합니다.

#### 구현 파일
- `src/weighted_mtp/cli/train.py`

#### 핵심 함수

**1) _load_model()**
```python
def _load_model(config: dict, device: torch.device) -> MetaLlamaMTPAdapter:
    """모델 로딩 (Adapter + Value Head)

    Args:
        config: 전체 YAML config
        device: torch.device

    Returns:
        MetaLlamaMTPAdapter (device로 이동됨)
    """
    from weighted_mtp.models.meta_mtp import load_adapter

    model_path = Path(config["models"]["policy"]["path"])

    logger.info(f"모델 로딩 시작: {model_path}")
    adapter = load_adapter(model_path, device=device)

    logger.info(f"모델 로딩 완료: {adapter.model.params}")
    return adapter
```

**2) _load_tokenizer()**
```python
def _load_tokenizer(config: dict):
    """토크나이저 로딩

    Args:
        config: 전체 YAML config

    Returns:
        Tokenizer (transformers 또는 SentencePiece)
    """
    from transformers import AutoTokenizer

    tokenizer_path = Path(config["models"]["policy"]["path"]) / "tokenizer"

    logger.info(f"토크나이저 로딩 시작: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return tokenizer
```

**3) _load_datasets()**
```python
def _load_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """Stage별 데이터셋 로딩

    Args:
        config: 전체 YAML config

    Returns:
        (stage1_dataset, stage2_dataset)
    """
    from weighted_mtp.data import load_dataset

    dataset_name = config["dataset"]["name"]

    # Stage 1 샘플링 전략 (is_correct 균형)
    data_config = config.get("data", {}).get("sampling", {})
    stage1_config = data_config.get("stage1", {})

    logger.info("Stage 1 데이터셋 로딩 (is_correct 균형)")
    stage1_dataset = load_dataset(
        dataset_name=dataset_name,
        split="train",
        stage="stage1",
        n_samples=stage1_config.get("n_samples", 30000),
        balance_correct=stage1_config.get("balance_correct", True),
        correct_ratio=stage1_config.get("correct_ratio", 0.5),
        seed=stage1_config.get("seed", 42),
    )

    # Stage 2 샘플링 전략 (curriculum learning)
    stage2_config = data_config.get("stage2", {})

    logger.info("Stage 2 데이터셋 로딩 (curriculum learning)")
    # TODO: Curriculum learning은 epoch 진행에 따라 동적 샘플링 필요
    # 현재는 전체 샘플만 로드 (curriculum은 training loop에서 처리)
    stage2_dataset = load_dataset(
        dataset_name=dataset_name,
        split="train",
        stage="stage2",
        n_samples=stage2_config.get("n_samples", 100000),
        balance_correct=stage2_config.get("balance_correct", True),
        correct_ratio=stage2_config.get("correct_ratio", 0.5),
        seed=stage2_config.get("seed", 42),
    )

    return stage1_dataset, stage2_dataset
```

**4) _create_dataloaders()**
```python
def _create_dataloaders(
    stage1_dataset: Dataset,
    stage2_dataset: Dataset,
    tokenizer,
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader 생성

    Args:
        stage1_dataset: Stage 1 Dataset
        stage2_dataset: Stage 2 Dataset
        tokenizer: Tokenizer
        config: 전체 YAML config

    Returns:
        (stage1_dataloader, stage2_dataloader)
    """
    from torch.utils.data import DataLoader
    from weighted_mtp.data import AlpacaDataCollator

    max_length = config["dataset"].get("max_length", 2048)

    # Collator 생성
    collator = AlpacaDataCollator(tokenizer=tokenizer, max_length=max_length)

    # Stage별 batch size
    stage1_batch_size = config["training"]["stage1"].get("batch_size", 8)
    stage2_batch_size = config["training"]["stage2"].get("batch_size", 4)

    stage1_dataloader = DataLoader(
        stage1_dataset,
        batch_size=stage1_batch_size,
        collate_fn=collator,
        shuffle=True,
    )

    stage2_dataloader = DataLoader(
        stage2_dataset,
        batch_size=stage2_batch_size,
        collate_fn=collator,
        shuffle=True,
    )

    return stage1_dataloader, stage2_dataloader
```

#### 검증 기준
- [ ] micro-mtp 모델 로딩 성공
- [ ] 토크나이저 로딩 성공
- [ ] Stage 1 데이터셋 로딩 성공 (is_correct 균형 확인)
- [ ] Stage 2 데이터셋 로딩 성공
- [ ] DataLoader 생성 성공 (collator 적용 확인)

---

### Step 3: MLflow 초기화 모듈 구현 (WMTP EC2 + S3 재사용)

#### 목표
**WMTP 프로젝트의 MLflow + S3 구성을 그대로 재사용**하여 실험 추적 시스템을 구현합니다.

#### 기존 WMTP 구성 확인

**EC2 MLflow Tracking Server**:
- URI: `http://13.50.240.176` (Port 80, Basic Auth)
- 인증: 환경변수 `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` 필요
- 설정: 프로젝트 루트의 `.env` 파일에 설정 (`.env.example` 참고)

**S3 Artifact Storage**:
- Bucket: `s3://wmtp/mlflow-artifacts`
- AWS 인증: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
- Config: `configs/config.s3.yaml` 참고

**기존 구현 재사용**:
- `/Users/wesley/Desktop/wooshikwon/wmtp/src/utils/monitoring/mlflow.py`의 `MLflowManager` 클래스
- Basic Auth 자동 주입 (`_maybe_inject_basic_auth()`)
- S3 artifact location 자동 설정
- Flatten config logging 지원

#### 구현 파일
- `src/weighted_mtp/runtime/mlflow.py` (신규 생성, WMTP `MLflowManager` 재사용)

#### 핵심 구현 (WMTP 코드 기반)

**MLflowManager 클래스** (WMTP에서 복사):
```python
"""MLflow 실험 추적 관리자 (WMTP EC2 + S3 재사용)

기존 WMTP 프로젝트의 MLflow 구성을 그대로 사용:
- EC2 MLflow Server: http://13.50.240.176 (Basic Auth)
- S3 Artifacts: s3://wmtp/mlflow-artifacts
- 환경변수 기반 인증
"""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from rich.console import Console

console = Console()


class MLflowManager:
    """MLflow 작업 관리자 (EC2 Tracking Server + S3 Artifacts)

    WMTP 기존 구성 재사용:
    - Tracking: EC2 서버 (http://13.50.240.176)
    - Artifacts: S3 (s3://wmtp/mlflow-artifacts)
    - 인증: 환경변수 Basic Auth
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "default",
        s3_artifacts: str | None = None,
    ):
        """MLflow 관리자 초기화

        Args:
            tracking_uri: MLflow 추적 서버 URI
                - 원격: "http://13.50.240.176" (EC2)
                - 로컬: "file://./mlruns"
            experiment_name: 실험 이름
                - 권장: "weighted-mtp/production"
            s3_artifacts: Artifact 저장 S3 경로
                - "s3://wmtp/mlflow-artifacts"
        """
        self.tracking_uri = self._maybe_inject_basic_auth(tracking_uri)
        self.experiment_name = experiment_name
        self.s3_artifacts = s3_artifacts

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Initialize client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Set or create experiment with S3 artifact location
        self.experiment_id = self._setup_experiment(experiment_name, s3_artifacts)
        self.run: mlflow.ActiveRun | None = None

    def _maybe_inject_basic_auth(self, uri: str) -> str:
        """HTTP(S) URI에 Basic Auth 자격증명 주입

        환경변수 MLFLOW_TRACKING_USERNAME/PASSWORD가 있으면
        URI에 user:pass@host 형식으로 주입합니다.
        """
        try:
            if not uri.startswith(("http://", "https://")):
                return uri

            parsed = urlparse(uri)
            if "@" in parsed.netloc:
                return uri  # 이미 크리덴셜 포함

            username = os.getenv("MLFLOW_TRACKING_USERNAME")
            password = os.getenv("MLFLOW_TRACKING_PASSWORD")
            if not username or not password:
                return uri

            # user:pass@host[:port]
            if ":" in parsed.netloc:
                host, port = parsed.netloc.split(":", 1)
                netloc = f"{username}:{password}@{host}:{port}"
            else:
                netloc = f"{username}:{password}@{parsed.netloc}"

            injected = parsed._replace(netloc=netloc)
            return urlunparse(injected)
        except Exception:
            return uri

    def _setup_experiment(
        self, experiment_name: str, artifact_location: str | None = None
    ) -> str:
        """Experiment 설정 또는 생성 (S3 artifact location 포함)"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                mlflow.set_experiment(experiment_name)
                console.print(
                    f"[green]Using existing experiment: {experiment_name}[/green]"
                )
                if artifact_location:
                    console.print(
                        f"[blue]Artifacts location: {artifact_location}[/blue]"
                    )
                return experiment.experiment_id
            else:
                # Create new experiment with S3 artifact_location
                experiment_id = self.client.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
                mlflow.set_experiment(experiment_name)
                console.print(
                    f"[green]Created new experiment: {experiment_name}[/green]"
                )
                if artifact_location:
                    console.print(
                        f"[blue]Artifacts will be stored in: {artifact_location}[/blue]"
                    )
                return experiment_id
        except Exception as e:
            msg = str(e)
            if isinstance(e, MlflowException) or "401" in msg or "Unauthorized" in msg:
                raise RuntimeError(
                    "MLflow 인증 실패: EC2 서버 접근 불가. "
                    "환경변수 MLFLOW_TRACKING_USERNAME/PASSWORD 확인 필요. "
                    f"현재 URI: {self.tracking_uri}"
                ) from e
            console.print(f"[yellow]Warning: Failed to setup experiment: {e}[/yellow]")
            mlflow.set_experiment("Default")
            return "0"

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> mlflow.ActiveRun:
        """MLflow run 시작"""
        self.run = mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=tags,
        )
        assert self.run is not None
        console.print(f"[green]Started MLflow run: {self.run.info.run_id}[/green]")
        return self.run

    def end_run(self, status: str = "FINISHED") -> None:
        """MLflow run 종료"""
        if self.run:
            mlflow.end_run(status=status)
            console.print(f"[green]Ended MLflow run with status: {status}[/green]")
            self.run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Parameters 로깅 (flatten)"""
        if not self.run:
            return

        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            mlflow.log_param(key, str(value))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Metrics 로깅"""
        if not self.run:
            return

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Artifact 로깅 (S3 자동 업로드)"""
        if not self.run:
            return

        local_path = Path(local_path)
        if local_path.is_file():
            mlflow.log_artifact(str(local_path), artifact_path)
        elif local_path.is_dir():
            mlflow.log_artifacts(str(local_path), artifact_path)

    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Nested dict를 flatten"""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def create_mlflow_manager(config: dict) -> MLflowManager:
    """Config에서 MLflow 관리자 생성

    WMTP 구성 재사용:
    - tracking_uri: "http://13.50.240.176"
    - experiment: "weighted-mtp/production"
    - s3_artifacts: "s3://wmtp/mlflow-artifacts"
    """
    mlflow_config = config.get("mlflow", {})

    return MLflowManager(
        tracking_uri=mlflow_config.get("tracking_uri", "file://./mlruns"),
        experiment_name=mlflow_config.get("experiment", "default"),
        s3_artifacts=mlflow_config.get("s3_artifacts"),
    )
```

#### defaults.yaml에 MLflow 설정 추가

```yaml
# MLflow 실험 추적 설정 (WMTP EC2 + S3 재사용)
mlflow:
  tracking_uri: "http://13.50.240.176"  # EC2 MLflow Server (Basic Auth)
  experiment: "weighted-mtp/production"
  s3_artifacts: "s3://wmtp/mlflow-artifacts"  # S3 Artifact Storage
```

#### 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 환경변수를 설정합니다:

```bash
# .env 파일 예시 (.env.example 참고)

# AWS S3 Credentials (MLflow Artifact Storage)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-west-2

# MLflow EC2 Server Authentication (Basic Auth)
MLFLOW_TRACKING_USERNAME=your_mlflow_username
MLFLOW_TRACKING_PASSWORD=your_mlflow_password

# Hugging Face 인증 (모델 다운로드용)
HF_TOKEN=your_huggingface_token_here
```

**주의**: `.env` 파일은 `.gitignore`에 포함되어 Git에 커밋되지 않습니다. 실제 credential 값은 `.env.example`을 복사하여 `.env`에 설정하세요.

#### 검증 기준
- [ ] MLflow EC2 서버 연결 성공 (Basic Auth)
- [ ] Experiment 생성/로드 성공
- [ ] S3 artifact location 설정 확인
- [ ] Run 시작/종료 성공
- [ ] Params 로깅 성공 (flatten)
- [ ] Metrics 로깅 성공
- [ ] Checkpoint S3 업로드 성공

---

### Step 4: CLI main() 함수 완성

#### 목표
모든 구성 요소를 연결하여 실제 학습을 실행합니다.

#### 구현 파일
- `src/weighted_mtp/cli/train.py`

#### 핵심 로직

**main() 함수 개선**:
```python
def main():
    """CLI 진입점"""
    # 1. Argument parsing (기존 유지)
    parser = argparse.ArgumentParser(description="Weighted MTP 학습 파이프라인")
    # ... (기존 argparse 코드)
    args = parser.parse_args()

    # 2. Config 로딩 (기존 deep_merge 사용)
    config = load_config(args.config, args.recipe)

    # 3. Preset 적용
    if args.preset == "local-light":
        preset_path = Path("configs/local-light.yaml")
        with open(preset_path) as f:
            preset = yaml.safe_load(f)
        config = deep_merge(config, preset.get("override", {}))

    # 4. Micro 모델 오버라이드
    if args.use_micro_model:
        config["models"]["policy"]["name"] = "micro-mtp"
        config["models"]["policy"]["path"] = "storage/models_v2/micro-mtp"

    # 5. Dry-run 모드
    if args.dry_run:
        console.print("[bold green]Dry-run mode: 설정 확인[/bold green]")
        console.print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        return

    # 6. Config 검증
    _validate_config(config)

    # 7. Logging 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 8. 환경 초기화 (분산학습 포함)
    from weighted_mtp.runtime.environment import setup_environment

    device = setup_environment(config)

    # 9. MLflow 초기화 (Rank 0만, WMTP EC2 + S3)
    from weighted_mtp.runtime.mlflow import create_mlflow_manager
    from weighted_mtp.runtime.distributed import is_main_process

    mlflow_manager = None
    if is_main_process():
        logger.info("MLflow 초기화 (EC2 + S3)")
        mlflow_manager = create_mlflow_manager(config)

        # Run 시작
        run_name = args.run_name
        if run_name is None:
            # 자동 run_name 생성
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = config.get("experiment", {}).get("name", "unknown")
            run_name = f"{exp_name}_{timestamp}"

        mlflow_manager.start_run(run_name=run_name)

        # Config params 로깅
        mlflow_manager.log_params(config)

    try:
        # 10. Resource 로딩
        logger.info("Resource 로딩 시작")

        adapter = _load_model(config, device)
        tokenizer = _load_tokenizer(config)
        stage1_dataset, stage2_dataset = _load_datasets(config)
        stage1_dataloader, stage2_dataloader = _create_dataloaders(
            stage1_dataset, stage2_dataset, tokenizer, config
        )

        logger.info("Resource 로딩 완료")

        # 11. Training config 추출
        training_config = _extract_training_config(config)

        # 12. 파이프라인 실행
        from weighted_mtp.pipelines.training import run_training_pipeline

        logger.info("학습 파이프라인 시작")

        save_dir = Path("checkpoints") / config.get("experiment", {}).get("name", "default")
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics = run_training_pipeline(
            adapter=adapter,
            stage1_dataloader=stage1_dataloader,
            stage2_dataloader=stage2_dataloader,
            config=training_config,
            device=device,
            save_dir=save_dir,
        )

        logger.info("학습 파이프라인 완료")

        # 13. 최종 결과 출력
        console.print("\n[bold green]학습 완료![/bold green]")
        console.print(f"Stage 1 Loss: {metrics['stage1']['stage1_loss']:.4f}")
        console.print(f"Stage 2 Total Loss: {metrics['stage2']['stage2_total_loss']:.4f}")

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}", exc_info=True)
        raise

    finally:
        # 14. MLflow 종료 (Rank 0만)
        if mlflow_manager is not None:
            mlflow_manager.end_run()
```

#### 검증 기준
- [ ] `--dry-run` 모드 동작 확인
- [ ] `--use-micro-model` 모드로 로컬 학습 성공
- [ ] `--recipe configs/recipe.verifiable.yaml` 실행 성공
- [ ] MLflow에 metrics 로깅 확인
- [ ] Checkpoint 저장 확인

---

### Step 5: 분산학습 지원 추가

#### 목표
A100 4-GPU 환경에서 FSDP 기반 분산학습을 지원합니다.

#### 구현 파일
- `src/weighted_mtp/cli/train.py` (분기 로직 추가)
- `src/weighted_mtp/runtime/environment.py` (이미 구현됨)

#### 핵심 수정

**분산학습 자동 감지**:
```python
def main():
    # ... (기존 코드)

    # 8. 환경 초기화 (분산학습 자동 감지)
    from weighted_mtp.runtime.environment import setup_environment
    from weighted_mtp.runtime.distributed import is_distributed, get_rank

    device = setup_environment(config)

    # 분산학습 여부 확인
    if is_distributed():
        rank = get_rank()
        logger.info(f"분산학습 모드: Rank {rank}")

        # DistributedSampler 사용
        # TODO: DataLoader 생성 시 sampler 추가
    else:
        logger.info("단일 GPU 모드")

    # ... (나머지 코드)
```

**DistributedSampler 적용**:
```python
def _create_dataloaders(
    stage1_dataset: Dataset,
    stage2_dataset: Dataset,
    tokenizer,
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader 생성 (분산학습 지원)"""
    from torch.utils.data import DataLoader, DistributedSampler
    from weighted_mtp.data import AlpacaDataCollator
    from weighted_mtp.runtime.distributed import is_distributed

    # ... (collator, batch_size 설정)

    # DistributedSampler 생성 (분산학습 시)
    if is_distributed():
        stage1_sampler = DistributedSampler(stage1_dataset, shuffle=True)
        stage2_sampler = DistributedSampler(stage2_dataset, shuffle=True)
        stage1_shuffle = False  # Sampler가 shuffle 담당
        stage2_shuffle = False
    else:
        stage1_sampler = None
        stage2_sampler = None
        stage1_shuffle = True
        stage2_shuffle = True

    stage1_dataloader = DataLoader(
        stage1_dataset,
        batch_size=stage1_batch_size,
        collate_fn=collator,
        shuffle=stage1_shuffle,
        sampler=stage1_sampler,
    )

    stage2_dataloader = DataLoader(
        stage2_dataset,
        batch_size=stage2_batch_size,
        collate_fn=collator,
        shuffle=stage2_shuffle,
        sampler=stage2_sampler,
    )

    return stage1_dataloader, stage2_dataloader
```

#### VESSL 실행 예시

**로컬 테스트**:
```bash
uv run python -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --recipe configs/recipe.verifiable.yaml \
  --preset local-light \
  --use-micro-model \
  --run-name test_run_001
```

**VESSL 분산학습**:
```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --recipe configs/recipe.verifiable.yaml \
  --run-name vessl_prod_001
```

#### 검증 기준
- [ ] 단일 GPU 모드 정상 동작
- [ ] DistributedSampler 적용 확인 (분산 환경)
- [ ] 4-GPU 환경에서 데이터 중복 없이 분할 확인
- [ ] Rank 0만 MLflow 로깅 확인

---

### Step 6: Recipe 파일 검증 및 보완

#### 목표
3가지 실험 recipe 파일이 파이프라인과 정확히 매핑되도록 검증합니다.

#### 검증 항목

**recipe.baseline.yaml**:
- [ ] experiment.name 존재
- [ ] training.stage1 설정 존재
- [ ] training.stage2.use_weighting = false 확인
- [ ] dataset 경로 정확

**recipe.verifiable.yaml**:
- [ ] experiment.name 존재
- [ ] training.stage2.use_weighting = true 확인
- [ ] training.stage2.beta, value_coef 존재
- [ ] dataset 경로 정확

**recipe.rho1_weighted.yaml**:
- [ ] experiment.name 존재
- [ ] training.stage2.weighting_method = "rho1" 확인
- [ ] models.reference 경로 존재
- [ ] dataset 경로 정확

#### 필요한 수정

현재 `recipe.verifiable.yaml`의 문제:
- `training.stage1.num_epochs` → `training.stage1.n_epochs` (통일)
- `training.stage2.num_epochs` → `training.stage2.n_epochs` (통일)

**수정 예시**:
```yaml
training:
  stage1:
    n_epochs: 0.5  # ← num_epochs에서 변경
    batch_size: 8
    learning_rate: 5.0e-5

  stage2:
    n_epochs: 2.5  # ← num_epochs에서 변경
    batch_size: 4
```

---

## Part 4: 검증 및 테스트

### 4.1 Unit Test

**테스트 파일**: `tests/unit/test_cli_train.py` (신규 생성)

**테스트 항목**:
```python
class TestConfigExtraction:
    """Config 추출 함수 테스트"""

    def test_extract_training_config(self):
        """training config 추출 정확성"""
        config = {
            "training": {
                "stage1": {"n_epochs": 0.5, "loss_type": "mse", "learning_rate": 1e-4},
                "stage2": {"n_epochs": 2.5, "beta": 0.9, "value_coef": 0.5},
            }
        }

        result = _extract_training_config(config)

        assert result["stage1"]["n_epochs"] == 0.5
        assert result["stage2"]["beta"] == 0.9

    def test_validate_config_missing_section(self):
        """필수 섹션 누락 시 ValueError"""
        config = {"project": {}}  # models, dataset 누락

        with pytest.raises(ValueError):
            _validate_config(config)

    def test_validate_config_missing_path(self):
        """모델 경로 없을 시 FileNotFoundError"""
        config = {
            "project": {},
            "models": {"policy": {"path": "/nonexistent"}},
            "dataset": {},
            "training": {},
        }

        with pytest.raises(FileNotFoundError):
            _validate_config(config)
```

### 4.2 Integration Test

**테스트 파일**: `tests/integration/test_cli_integration.py` (신규 생성)

**테스트 항목**:
```python
class TestCLIIntegration:
    """CLI end-to-end 통합 테스트"""

    def test_dry_run_mode(self):
        """--dry-run 모드 동작 확인"""
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "weighted_mtp.cli.train",
                "--config", "configs/defaults.yaml",
                "--recipe", "configs/recipe.verifiable.yaml",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Dry-run mode" in result.stdout

    def test_micro_model_training(self):
        """Micro 모델로 실제 학습 실행"""
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "weighted_mtp.cli.train",
                "--config", "configs/defaults.yaml",
                "--preset", "local-light",
                "--use-micro-model",
                "--run-name", "test_integration",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10분 timeout
        )

        assert result.returncode == 0
        assert "학습 완료" in result.stdout

        # Checkpoint 저장 확인
        checkpoint_dir = Path("checkpoints/verifiable-critic-wmtp")
        assert checkpoint_dir.exists()
```

### 4.3 Smoke Test

**테스트 스크립트**: `scripts/smoke_test_cli.sh` (신규 생성)

```bash
#!/bin/bash
set -e

echo "=== CLI Smoke Test 시작 ==="

# 1. Dry-run 테스트
echo "[1/3] Dry-run 모드 테스트"
uv run python -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --recipe configs/recipe.verifiable.yaml \
  --dry-run

# 2. Micro 모델 학습 테스트
echo "[2/3] Micro 모델 학습 테스트"
uv run python -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --preset local-light \
  --use-micro-model \
  --run-name smoke_test

# 3. MLflow 확인
echo "[3/3] MLflow 로그 확인"
if [ ! -d "mlruns" ]; then
  echo "❌ MLflow 디렉터리 없음"
  exit 1
fi

echo "✅ CLI Smoke Test 통과"
```

---

## Part 5: 완료 기준 및 다음 단계

### 5.1 Phase 6 완료 체크리스트

#### 코드 완성
- [ ] `_extract_training_config()` 구현
- [ ] `_validate_config()` 구현
- [ ] `_load_model()` 구현
- [ ] `_load_tokenizer()` 구현
- [ ] `_load_datasets()` 구현
- [ ] `_create_dataloaders()` 구현 (분산 지원)
- [ ] `src/weighted_mtp/runtime/mlflow.py` 구현
  - init_mlflow(), log_metrics(), finish_mlflow()
- [ ] `cli/train.py` main() 함수 완성
  - 파이프라인 연결
  - MLflow 연동

#### Config 정리
- [ ] `recipe.verifiable.yaml` 필드명 통일 (num_epochs → n_epochs)
- [ ] `recipe.baseline.yaml` 검증
- [ ] `recipe.rho1_weighted.yaml` 검증

#### 테스트 완성
- [ ] `tests/unit/test_cli_train.py` 작성
  - test_extract_training_config()
  - test_validate_config()
- [ ] `tests/integration/test_cli_integration.py` 작성
  - test_dry_run_mode()
  - test_micro_model_training()
- [ ] `scripts/smoke_test_cli.sh` 작성

#### 검증 완료
- [ ] Dry-run 모드 동작 확인
- [ ] Micro 모델 로컬 학습 성공 (<2.5시간)
- [ ] MLflow 로깅 확인 (metrics, params)
- [ ] Checkpoint 저장 확인
- [ ] 3가지 recipe 모두 실행 가능 확인
- [ ] 분산학습 모드 동작 확인 (VESSL 환경)

#### 문서화
- [ ] 본 문서 (`08_phase6_detailed_plan.md`) 소급 업데이트
- [ ] CLI 사용 예시 추가 (README 또는 별도 문서)
- [ ] VESSL 실행 가이드 작성

### 5.2 Phase 7 착수 조건

Phase 6 완료 후, 다음 조건을 만족해야 Phase 7 (평가 파이프라인)로 진행:

**필수 조건**:
1. CLI로 3가지 실험 모두 실행 가능
2. Micro 모델 로컬 학습 성공
3. MLflow 로깅 정상 동작
4. Checkpoint 저장/로드 정상 동작
5. Integration test 통과

**권장 조건**:
1. VESSL 환경에서 분산학습 검증 (4-GPU)
2. Production 모델 (7B) 로딩 검증
3. Smoke test 자동화 (CI)

### 5.3 예상 소요 시간

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| Step 1: Config 추출/검증 | 3-4시간 | 함수 구현 + 단위 테스트 |
| Step 2: Resource 로딩 | 4-6시간 | 모델/데이터/DataLoader |
| Step 3: MLflow 모듈 | 3-4시간 | 초기화 + 로깅 |
| Step 4: CLI main() 완성 | 4-6시간 | 전체 연결 + 에러 처리 |
| Step 5: 분산학습 지원 | 2-3시간 | DistributedSampler 적용 |
| Step 6: Recipe 검증 | 2-3시간 | 3개 파일 수정 |
| 통합 테스트 및 디버깅 | 4-6시간 | End-to-end 검증 |
| 문서화 | 2-3시간 | 본 문서 + 사용 가이드 |
| **합계** | **24-35시간** | 약 3-4.5일 |

### 5.4 Phase 7 Preview

**Phase 7: 평가 파이프라인** (다음 단계)

핵심 구현:
1. `pipelines/evaluation.py`: MBPP, HumanEval 평가
2. Pass@K 계산 (K=1,5,10)
3. Inference 루틴 (beam search, nucleus sampling)
4. Rho-1 reference와의 loss 비교
5. 평가 리포트 자동 생성

**Phase 6와의 연계**:
- Phase 6 CLI → `--eval-only` 플래그 추가
- Checkpoint 로드 → 평가 실행
- MLflow에 평가 결과 자동 로깅

---

## 부록

### A. 디렉터리 구조 (Phase 6 추가)

```
weighted_mtp/
├── src/weighted_mtp/
│   ├── cli/
│   │   └── train.py           # ← Phase 6: main() 완성, resource 로딩 추가
│   ├── runtime/
│   │   ├── mlflow.py          # ← Phase 6: 신규 생성
│   │   ├── environment.py     # (Phase 3에서 구현됨)
│   │   └── distributed.py     # (Phase 3에서 구현됨)
│   └── ...
├── configs/
│   ├── defaults.yaml
│   ├── recipe.baseline.yaml
│   ├── recipe.verifiable.yaml # ← Phase 6: 필드명 수정
│   ├── recipe.rho1_weighted.yaml
│   └── local-light.yaml
├── tests/
│   ├── unit/
│   │   └── test_cli_train.py  # ← Phase 6: 신규 생성
│   └── integration/
│       └── test_cli_integration.py  # ← Phase 6: 신규 생성
├── scripts/
│   └── smoke_test_cli.sh      # ← Phase 6: 신규 생성
└── checkpoints/               # ← Phase 6: 자동 생성 (저장 디렉터리)
```

### B. 개발원칙 준수 체크리스트

**[원칙 1] 앞/뒤 흐름 분석**:
- [x] Phase 5 `run_training_pipeline()` 인터페이스 확인
- [x] Phase 3 `load_dataset()`, `AlpacaDataCollator` 확인
- [x] Phase 4 `load_adapter()` 확인
- [x] Phase 3 `runtime/distributed.py`, `runtime/environment.py` 확인

**[원칙 2] 기존 구조 존중**:
- [x] `run_training_pipeline()` 인터페이스 변경 없음
- [x] Config deep_merge 방식 유지
- [x] 기존 Data collator 재사용
- [x] 분산학습 모듈 재사용

**[원칙 3] 전격적 변경 승인**:
- [ ] 새로운 접근 시 사용자 승인 획득
- [ ] 기존 계획과 차이 발생 시 문서화

**[원칙 4] 하위 호환성 고려 없음**:
- [ ] 주석: 한글, 이모지 없음, 코드 동작 핵심만
- [ ] 로깅: 한글, 이모지 없음
- [ ] 변수명: 통일성 있게 네이밍 (n_epochs, not num_epochs)

**[원칙 5] 계획서와 비교**:
- [ ] Phase 6 완료 후 본 문서 소급 업데이트
- [ ] 차이점 객관적 기술
- [ ] 성과 과장 없음

**[원칙 6] 패키지 의존성 도구 활용**:
- [x] uv로 의존성 관리
- [x] pytest 실행 시 `uv run pytest` 사용

### C. Config 필드명 통일 규칙

**통일된 필드명** (Phase 6 적용):
- `n_epochs` (not `num_epochs`)
- `learning_rate` (not `lr`)
- `batch_size`
- `gradient_accumulation_steps`

**Recipe 파일 수정 예시**:
```yaml
# Before (일관성 없음)
training:
  stage1:
    num_epochs: 1        # ← 수정 필요
  stage2:
    num_epochs: 3        # ← 수정 필요

# After (통일됨)
training:
  stage1:
    n_epochs: 0.5        # ← 통일
  stage2:
    n_epochs: 2.5        # ← 통일
```

### D. MLflow 로깅 항목

**Params (Config)**:
```python
{
    "project.name": "weighted-mtp",
    "experiment.name": "verifiable-critic-wmtp",
    "models.policy.name": "meta-llama-mtp",
    "training.stage1.n_epochs": 0.5,
    "training.stage2.n_epochs": 2.5,
    "training.stage2.beta": 0.9,
    "training.stage2.value_coef": 0.5,
}
```

**Metrics (실시간)**:
```python
{
    "stage1_loss": float,
    "stage1_value_explained_variance": float,
    "stage2_weighted_ce_loss": float,
    "stage2_value_loss": float,
    "stage2_total_loss": float,
    "td_mean": float,
    "td_std": float,
    "weight_mean": float,
    "weight_entropy": float,
    "value_explained_variance": float,
}
```

**Artifacts (최종)**:
```
checkpoints/
  └── verifiable-critic-wmtp/
      ├── checkpoint_epoch_0.pt
      ├── checkpoint_epoch_1.pt
      └── final_metrics.json
```

---

**문서 종료**

본 문서는 Phase 6 **상세 계획**을 정리한 초안입니다. 구현 과정에서 실제 상태를 반영하여 소급 업데이트할 예정입니다.

**핵심 목표 요약**:
1. CLI main() 함수 완성 (파이프라인 연결)
2. Config 추출 및 검증 로직 구현
3. Resource 로딩 자동화 (모델, 데이터, DataLoader)
4. MLflow 초기화 및 로깅 구현
5. 분산학습 지원 (DistributedSampler)
6. 3가지 실험 recipe 검증 및 실행 확인
