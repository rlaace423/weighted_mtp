"""모델 checkpoint 로딩 유틸리티"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from .transformer import ModelArgs, Transformer

logger = logging.getLogger(__name__)


def _get_device(device: str) -> torch.device:
    """적절한 device 반환

    Args:
        device: "cuda", "mps", "cpu", "auto"

    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def load_meta_mtp_model(
    model_dir: Path,
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
    params_override: Optional[dict] = None,
) -> Transformer:
    """Meta MTP 모델 통합 로딩

    Args:
        model_dir: storage/models/meta-llama-mtp/ 경로
        device: "cuda", "mps", "cpu", "auto"
        dtype: torch.float16 등 (None이면 params.json 기준)
        params_override: ModelArgs 오버라이드 설정
            - n_future_tokens 변경은 지원하지 않음 (MTP/NTP 구조 불일치)
            - 다른 파라미터 (max_seq_len 등)는 변경 가능

    Returns:
        Transformer 인스턴스

    Raises:
        FileNotFoundError: params.json 또는 safetensors 파일 없음
        RuntimeError: safetensors 키 불일치
        ValueError: n_future_tokens 변경 시도 시
    """
    model_dir = Path(model_dir)

    # 1. params.json 또는 config.json 로드 → ModelArgs
    params_path = model_dir / "configs/params.json"
    config_path = model_dir / "configs/config.json"

    if params_path.exists():
        with open(params_path) as f:
            params_dict = json.load(f)
    elif config_path.exists():
        # config.json 사용 (micro 모델용)
        with open(config_path) as f:
            config_dict = json.load(f)
        # config.json 형식을 ModelArgs로 변환
        params_dict = {
            "dim": config_dict.get("hidden_size", config_dict.get("dim")),
            "n_layers": config_dict.get("num_hidden_layers", config_dict.get("n_layers")),
            "n_heads": config_dict.get("num_attention_heads", config_dict.get("n_heads")),
            "n_kv_heads": config_dict.get("num_key_value_heads", config_dict.get("n_kv_heads")),
            "vocab_size": config_dict.get("vocab_size"),
            "n_future_tokens": config_dict.get("n_future_tokens", 1),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "max_seq_len": config_dict.get("max_position_embeddings", 2048),
            "norm_eps": config_dict.get("rms_norm_eps", 1e-5),
        }
    else:
        raise FileNotFoundError(f"Neither params.json nor config.json found in {model_dir}/configs/")

    # params_override 적용 (n_future_tokens 변경은 차단)
    if params_override:
        # n_future_tokens 변경 시도 감지
        ckpt_n_future = params_dict.get("n_future_tokens", 1)
        override_n_future = params_override.get("n_future_tokens")

        if override_n_future is not None and override_n_future != ckpt_n_future:
            raise ValueError(
                f"n_future_tokens 변경 불가: checkpoint={ckpt_n_future}, override={override_n_future}. "
                f"MTP extra_heads는 미래 토큰 예측용으로 학습되어 NTP layer로 재사용 불가. "
                f"NTP가 필요하면 순수 LLaMA2 checkpoint 사용 권장."
            )

        # n_future_tokens 외의 파라미터만 업데이트
        valid_keys = ModelArgs.__dataclass_fields__.keys()
        filtered_override = {
            k: v for k, v in params_override.items()
            if k in valid_keys and k != "n_future_tokens"
        }
        if filtered_override:
            params_dict.update(filtered_override)
            logger.info(f"params_override 적용: {filtered_override}")

    model_args = ModelArgs(**params_dict)

    # 2. safetensors 로드
    ckpt_path = model_dir / "safetensors/model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"safetensors not found: {ckpt_path}")

    state_dict = load_file(str(ckpt_path))

    # 3. Transformer 생성
    transformer = Transformer(model_args)

    # 4. state_dict 로드 (strict=True로 키 검증)
    # freqs_cis는 일반 속성이므로 state_dict에 포함되지 않음
    try:
        transformer.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"State dict keys mismatch: {e}")

    # 5. Device 이동
    device_obj = _get_device(device)
    transformer = transformer.to(device_obj)

    # 6. dtype 설정 (선택적)
    if dtype is not None:
        transformer = transformer.to(dtype)
    elif "float16" in str(ckpt_path) or any("float16" in str(v) for v in params_dict.values()):
        # params.json에 dtype 정보가 있으면 사용
        transformer = transformer.to(torch.float16)

    return transformer
