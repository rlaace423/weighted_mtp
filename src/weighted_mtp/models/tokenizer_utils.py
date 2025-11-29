"""Tokenizer 로딩 유틸리티

전체 파이프라인의 tokenizer 로딩을 통합 관리.
padding_side, pad_token 등 일관된 설정 보장.
"""

from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    """경로 기반 tokenizer 로드

    Args:
        tokenizer_path: tokenizer 경로 (디렉터리 또는 HuggingFace 모델명)

    Returns:
        AutoTokenizer 인스턴스

    Examples:
        >>> tokenizer = load_tokenizer("storage/models/meta-llama-mtp/tokenizer")
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
    )

    # Padding 설정 (학습용 표준)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def load_tokenizer_from_config(config) -> AutoTokenizer:
    """Config 기반 tokenizer 로드

    우선순위:
    1. config.models.policy.tokenizer_path (Policy Model 있는 경우)
    2. config.models.value_model.tokenizer_path (Value Model만 있는 경우)
    3. 해당 모델의 path (fallback)

    Args:
        config: OmegaConf config 객체

    Returns:
        AutoTokenizer 인스턴스

    Examples:
        >>> config = OmegaConf.load("configs/production/baseline.yaml")
        >>> tokenizer = load_tokenizer_from_config(config)
    """
    # Policy Model 있는 경우 (Baseline, Rho1, NTP, Verifiable)
    if hasattr(config.models, "policy"):
        tokenizer_path = getattr(
            config.models.policy,
            "tokenizer_path",
            config.models.policy.path  # fallback
        )
    # Value Model만 있는 경우 (Critic)
    elif hasattr(config.models, "value_model"):
        tokenizer_path = getattr(
            config.models.value_model,
            "tokenizer_path",
            config.models.value_model.path  # fallback
        )
    else:
        raise ValueError("Config에 models.policy 또는 models.value_model이 없습니다")

    return load_tokenizer(tokenizer_path)


def resolve_tokenizer_path(model_path: str) -> str:
    """모델 경로에서 tokenizer 경로 해석

    Args:
        model_path: 모델 경로 (예: storage/models/meta-llama-mtp)

    Returns:
        tokenizer 경로

    Examples:
        >>> path = resolve_tokenizer_path("storage/models/meta-llama-mtp")
        >>> # returns "storage/models/meta-llama-mtp/tokenizer"
    """
    model_path_obj = Path(model_path)

    # micro-mtp는 config.json이 RAG로 오인되므로 meta-llama-mtp tokenizer 사용
    if model_path_obj.name == "micro-mtp":
        return str(Path("storage/models/meta-llama-mtp/tokenizer"))

    # tokenizer 서브디렉터리 우선
    tokenizer_path = model_path_obj / "tokenizer"
    if tokenizer_path.exists():
        return str(tokenizer_path)

    # Fallback: 모델 경로 직접 사용
    return str(model_path_obj)
