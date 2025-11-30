"""LoRA (Low-Rank Adaptation) 모듈

FSDP 호환 LoRA 구현:
- LoRALinear: nn.Linear를 감싸는 LoRA 레이어
- apply_lora_to_linear: 기존 Linear에 LoRA 적용
- get_lora_parameters: 학습 대상 LoRA 파라미터 추출
- apply_lora_to_hf_model: HuggingFace Llama 모델에 LoRA 적용
"""

from .lora_linear import (
    LoRALinear,
    apply_lora_to_linear,
    get_lora_parameters,
    merge_lora_weights,
    # HuggingFace 전용
    apply_lora_to_hf_model,
    get_hf_lora_state_dict,
    load_hf_lora_state_dict,
    DEFAULT_HF_LORA_CONFIG,
    HF_ATTENTION_TARGETS,
    HF_MLP_TARGETS,
)

__all__ = [
    "LoRALinear",
    "apply_lora_to_linear",
    "get_lora_parameters",
    "merge_lora_weights",
    # HuggingFace 전용
    "apply_lora_to_hf_model",
    "get_hf_lora_state_dict",
    "load_hf_lora_state_dict",
    "DEFAULT_HF_LORA_CONFIG",
    "HF_ATTENTION_TARGETS",
    "HF_MLP_TARGETS",
]

