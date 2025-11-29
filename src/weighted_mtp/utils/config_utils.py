"""Config 검증 및 override 유틸리티

OmegaConf를 사용하되, 필수 필드 및 값 범위를 검증한다.
CLI override를 통한 계층적 필드 수정을 지원한다.
"""

from pathlib import Path
from typing import List
from omegaconf import DictConfig, OmegaConf


class ConfigValidationError(Exception):
    """Config 검증 실패 시 발생하는 예외"""

    pass


def apply_overrides(config: DictConfig, overrides: List[str]) -> DictConfig:
    """CLI override를 config에 적용

    Args:
        config: 기본 config 객체
        overrides: "key=value" 형태의 override 리스트
                  예: ["experiment.name=test", "training.batch_size=8"]

    Returns:
        Override가 적용된 config 객체

    Raises:
        ValueError: Override 형식이 잘못된 경우

    Examples:
        >>> config = OmegaConf.load("config.yaml")
        >>> overrides = ["experiment.name=test", "training.batch_size=8"]
        >>> config = apply_overrides(config, overrides)
    """
    if not overrides:
        return config

    # Override 형식 검증
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"잘못된 override 형식: {override}. "
                f"'key=value' 형태여야 합니다 (예: experiment.name=test)"
            )

    # OmegaConf의 dotlist 기능 사용
    try:
        override_config = OmegaConf.from_dotlist(overrides)
        merged_config = OmegaConf.merge(config, override_config)
        return merged_config
    except Exception as e:
        raise ValueError(f"Override 적용 실패: {e}")


def parse_override_value(value_str: str) -> any:
    """Override value를 적절한 타입으로 변환

    Args:
        value_str: 문자열 형태의 값

    Returns:
        타입이 추론된 값 (int, float, bool, str)

    Note:
        OmegaConf.from_dotlist()가 자동으로 타입 추론하므로
        이 함수는 필요 시에만 사용
    """
    # Bool
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # String
    return value_str


def validate_config(config: DictConfig) -> None:
    """Config 파일 검증

    Args:
        config: OmegaConf DictConfig 객체

    Raises:
        ConfigValidationError: 검증 실패 시

    검증 항목:
        - 필수 필드 존재 확인
        - 값 범위 검증 (learning_rate > 0 등)
        - 경로 존재 확인 (모델, 데이터셋)
        - Stage별 특수 필드 검증
        - 논리적 일관성 (batch_size <= n_samples 등)
    """
    errors: List[str] = []

    # 1. 필수 필드 확인
    required_fields = [
        "experiment.name",
        "experiment.stage",
        "models.policy.path",
        "dataset.name",
        "dataset.train",
        "dataset.validation",
        "training.n_epochs",
        "training.batch_size",
    ]

    for field in required_fields:
        if not _has_nested_field(config, field):
            errors.append(f"필수 필드 누락: {field}")

    # Learning rate 필드 검증 (learning_rate 또는 trunk/value_head LR)
    if hasattr(config, "training"):
        has_lr = hasattr(config.training, "learning_rate")
        has_separate_lr = (
            hasattr(config.training, "trunk_learning_rate") and
            hasattr(config.training, "value_head_learning_rate")
        )
        if not has_lr and not has_separate_lr:
            errors.append(
                "필수 필드 누락: training.learning_rate 또는 "
                "training.trunk_learning_rate + training.value_head_learning_rate"
            )

    # 2. Stage 값 검증
    if hasattr(config, "experiment") and hasattr(config.experiment, "stage"):
        valid_stages = ["baseline", "critic", "verifiable", "rho1", "ref-tuning"]
        stage = config.experiment.stage
        if stage not in valid_stages:
            errors.append(f"잘못된 stage: {stage}. 유효 값: {valid_stages}")
    else:
        stage = None

    # 3. Learning rate 범위 검증
    if hasattr(config, "training"):
        # 단일 learning_rate
        if hasattr(config.training, "learning_rate"):
            lr = config.training.learning_rate
            if lr <= 0 or lr > 1.0:
                errors.append(f"learning_rate 범위 오류: {lr} (0 < lr <= 1.0)")
        # 분리된 trunk/value_head LR
        if hasattr(config.training, "trunk_learning_rate"):
            lr = config.training.trunk_learning_rate
            if lr <= 0 or lr > 1.0:
                errors.append(f"trunk_learning_rate 범위 오류: {lr} (0 < lr <= 1.0)")
        if hasattr(config.training, "value_head_learning_rate"):
            lr = config.training.value_head_learning_rate
            if lr <= 0 or lr > 1.0:
                errors.append(f"value_head_learning_rate 범위 오류: {lr} (0 < lr <= 1.0)")

    # 4. Batch size 검증
    if hasattr(config, "training") and hasattr(config.training, "batch_size"):
        batch_size = config.training.batch_size
        if batch_size <= 0:
            errors.append(f"batch_size는 양수여야 함: {batch_size}")

    # 5. Epochs 검증
    if hasattr(config, "training") and hasattr(config.training, "n_epochs"):
        n_epochs = config.training.n_epochs
        if n_epochs <= 0:
            errors.append(f"n_epochs는 양수여야 함: {n_epochs}")

    # 6. Gradient accumulation steps 검증
    if hasattr(config, "training") and hasattr(
        config.training, "gradient_accumulation_steps"
    ):
        gas = config.training.gradient_accumulation_steps
        if gas <= 0:
            errors.append(f"gradient_accumulation_steps는 양수여야 함: {gas}")

    # 7. Max grad norm 검증
    if hasattr(config, "training") and hasattr(config.training, "max_grad_norm"):
        mgn = config.training.max_grad_norm
        if mgn <= 0:
            errors.append(f"max_grad_norm은 양수여야 함: {mgn}")

    # 8. 모델 경로 존재 확인 (.pt checkpoint는 학습 후 생성되므로 제외)
    if hasattr(config, "models") and hasattr(config.models, "policy"):
        model_path = Path(config.models.policy.path)
        if not str(model_path).endswith(".pt") and not model_path.exists():
            errors.append(f"모델 경로가 존재하지 않음: {model_path}")

    # 9. 데이터셋 경로 존재 확인
    if hasattr(config, "dataset"):
        if hasattr(config.dataset, "train"):
            train_path = Path(config.dataset.train)
            if not train_path.exists():
                errors.append(f"학습 데이터셋 파일이 존재하지 않음: {train_path}")

        if hasattr(config.dataset, "validation"):
            valid_path = Path(config.dataset.validation)
            if not valid_path.exists():
                errors.append(f"검증 데이터셋 파일이 존재하지 않음: {valid_path}")

    # 10. Stage별 특수 검증
    if stage == "verifiable":
        _validate_verifiable_stage(config, errors)
    elif stage == "rho1":
        _validate_rho1_stage(config, errors)
    elif stage == "critic":
        _validate_critic_stage(config, errors)

    # 11. data_sampling 구조 검증
    _validate_data_sampling(config, errors)

    # 12. 논리적 일관성 검증 (batch_size vs 샘플 수)
    if hasattr(config, "training") and hasattr(config.training, "batch_size"):
        batch_size = config.training.batch_size
        if hasattr(config, "data_sampling"):
            ds = config.data_sampling
            if hasattr(ds, "n_samples"):
                n_samples = ds.n_samples
                if batch_size > n_samples:
                    errors.append(f"batch_size({batch_size})가 n_samples({n_samples})보다 큼")

    # 에러 보고
    if errors:
        raise ConfigValidationError(
            f"Config 검증 실패 ({len(errors)}개 오류):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _validate_verifiable_stage(config: DictConfig, errors: List[str]) -> None:
    """Verifiable stage 전용 검증

    Args:
        config: Config 객체
        errors: 에러 목록 (mutable)
    """
    # Beta 필드 확인
    if not (hasattr(config, "training") and hasattr(config.training, "beta")):
        errors.append("verifiable stage는 training.beta 필드 필수")

    # Weight clip 검증
    if hasattr(config, "training"):
        if hasattr(config.training, "weight_clip_min") and hasattr(
            config.training, "weight_clip_max"
        ):
            min_w = config.training.weight_clip_min
            max_w = config.training.weight_clip_max
            if min_w >= max_w:
                errors.append(
                    f"weight_clip_min({min_w})이 weight_clip_max({max_w})보다 크거나 같음"
                )

    # Curriculum learning 검증
    if hasattr(config, "data_sampling"):
        if hasattr(config.data_sampling, "curriculum_learning"):
            if config.data_sampling.curriculum_learning:
                if not hasattr(config.data_sampling, "curriculum_schedule"):
                    errors.append(
                        "curriculum_learning=true인 경우 curriculum_schedule 필수"
                    )

    # Critic checkpoint 검증 (선택 필드이므로 경고만)
    # 실제 학습 시에는 존재해야 하지만, config 검증 시점에는 아직 없을 수 있음


def _validate_rho1_stage(config: DictConfig, errors: List[str]) -> None:
    """Rho1 stage 전용 검증

    Args:
        config: Config 객체
        errors: 에러 목록 (mutable)
    """
    # Reference model 필드 확인
    if not (
        hasattr(config, "models")
        and hasattr(config.models, "reference")
        and hasattr(config.models.reference, "path")
    ):
        errors.append("rho1 stage는 models.reference.path 필드 필수")
    else:
        ref_path = Path(config.models.reference.path)
        if not ref_path.exists():
            errors.append(f"reference 모델 경로가 존재하지 않음: {ref_path}")

    # Temperature 검증
    if hasattr(config, "training") and hasattr(config.training, "temperature"):
        temp = config.training.temperature
        if temp <= 0:
            errors.append(f"temperature는 양수여야 함: {temp}")


def _validate_critic_stage(config: DictConfig, errors: List[str]) -> None:
    """Critic stage 전용 검증

    Args:
        config: Config 객체
        errors: 에러 목록 (mutable)
    """
    if not hasattr(config, "training"):
        return

    # sigmoid value_head_type은 MC (gamma=1, lam=1)일 때만 사용 가능
    value_head_type = getattr(config.training, "value_head_type", "mlp")
    if value_head_type == "sigmoid":
        gamma = getattr(config.training, "gamma", 1.0)
        lam = getattr(config.training, "lam", 1.0)

        if gamma != 1.0 or lam != 1.0:
            errors.append(
                f"value_head_type='sigmoid'는 MC (gamma=1.0, lam=1.0)일 때만 사용 가능. "
                f"현재: gamma={gamma}, lam={lam}. "
                f"BCE loss는 target이 0~1 범위여야 하므로 GAE 사용 불가."
            )


def _validate_data_sampling(config: DictConfig, errors: List[str]) -> None:
    """data_sampling 섹션 검증

    Difficulty 기반 샘플링 설정을 검증한다.
    use_pairwise 옵션은 샘플링 후 pairwise 포맷 변환 여부를 결정한다.

    Args:
        config: Config 객체
        errors: 에러 목록 (mutable)
    """
    if not hasattr(config, "data_sampling"):
        return

    ds = config.data_sampling

    # n_samples 검증
    if hasattr(ds, "n_samples"):
        if ds.n_samples <= 0:
            errors.append(f"n_samples는 양수여야 함: {ds.n_samples}")

    # difficulty_weights와 difficulty_bins 일관성
    has_weights = hasattr(ds, "difficulty_weights")
    has_bins = hasattr(ds, "difficulty_bins")
    if has_weights and not has_bins:
        errors.append("difficulty_weights 사용 시 difficulty_bins도 필수")
    if has_bins and not has_weights:
        errors.append("difficulty_bins 사용 시 difficulty_weights도 필수")


def _has_nested_field(config: DictConfig, field_path: str) -> bool:
    """중첩 필드 존재 확인

    Args:
        config: OmegaConf DictConfig
        field_path: 점으로 구분된 필드 경로 (예: 'experiment.name')

    Returns:
        필드 존재 여부
    """
    keys = field_path.split(".")
    current = config
    for key in keys:
        if not hasattr(current, key):
            return False
        current = getattr(current, key)
    return True


def load_and_validate_config(config_path: str) -> DictConfig:
    """Config 로드 및 검증

    Args:
        config_path: Config 파일 경로

    Returns:
        검증된 DictConfig 객체

    Raises:
        ConfigValidationError: 검증 실패 시
        FileNotFoundError: Config 파일이 없는 경우
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")

    # Config 로드
    config = OmegaConf.load(config_file)

    # 검증
    validate_config(config)

    return config
