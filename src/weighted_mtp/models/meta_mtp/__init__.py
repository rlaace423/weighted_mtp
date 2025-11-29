"""Meta LLaMA MTP Adapter

순수 PyTorch 구현 (MTP Policy Model):
- transformer.py: Meta 구조 참고, fairscale 제거, FSDP 호환
- checkpoints.py: safetensors 로딩
- adapter.py: MTP forward wrapper (value head 없음)

Value Head는 models/value_head.py에서 정의 (독립 ValueModel에서 사용)
"""

from .adapter import MetaLlamaMTPAdapter
from .checkpoints import load_meta_mtp_model
from .transformer import ModelArgs, Transformer

# Value Head는 상위 모듈에서 re-export (ValueModel에서 사용)
from ..value_head import (
    LinearValueHead,
    MLPValueHead,
    ValueHead,
    ValueHeadType,
    create_value_head,
)

__all__ = [
    "MetaLlamaMTPAdapter",
    "load_meta_mtp_model",
    "ModelArgs",
    "Transformer",
    "LinearValueHead",
    "MLPValueHead",
    "ValueHead",
    "ValueHeadType",
    "create_value_head",
]
