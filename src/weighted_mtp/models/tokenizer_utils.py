"""Tokenizer 로딩 유틸리티

중복 제거를 위한 통합 tokenizer 로딩 함수
"""

from transformers import AutoTokenizer


def load_tokenizer_from_config(config: dict) -> AutoTokenizer:
    """Config 기반 통합 tokenizer 로딩

    우선순위:
    1. config.models.policy.tokenizer_path (명시적 지정)
    2. config.models.policy.path (기본값)

    Args:
        config: OmegaConf config 객체

    Returns:
        AutoTokenizer 인스턴스

    Examples:
        >>> config = OmegaConf.load("configs/critic/critic_local.yaml")
        >>> tokenizer = load_tokenizer_from_config(config)
    """
    # tokenizer_path 명시적 지정 확인 (우선)
    tokenizer_path = getattr(
        config.models.policy,
        "tokenizer_path",
        config.models.policy.path  # fallback
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),  # 명시적 문자열 변환
        use_fast=True,
    )

    # Padding token 설정 (LLaMA 표준)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
