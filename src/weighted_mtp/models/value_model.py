"""독립 Value Model

HuggingFace LlamaModel 기반의 완전 독립된 Value Model.
Policy Model(MTP)과 완전히 분리되어 별도 backbone 사용.

Critic 파이프라인에서 학습, Verifiable에서 eval only로 사용.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import LlamaModel, LlamaConfig

from .value_head import create_value_head, ValueHeadType


class ValueModel(nn.Module):
    """독립 Value Model
    
    HuggingFace LlamaModel + Value Head 구조.
    Policy Model과 완전히 독립된 별도 모델.
    
    Args:
        backbone: HuggingFace LlamaModel
        value_head: Value Head (Linear, MLP 등)
        config: LlamaConfig
    """
    
    def __init__(
        self,
        backbone: LlamaModel,
        value_head: ValueHeadType,
        config: LlamaConfig,
    ):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head
        self.config = config
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        value_head_type: str = "mlp",
        dropout: float = 0.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> "ValueModel":
        """HuggingFace pretrained 모델에서 Value Model 생성
        
        Args:
            model_path: HuggingFace 모델 경로 (예: storage/models/ref-sheared-llama-2.7b/raw)
            value_head_type: "linear", "sigmoid", 또는 "mlp"
            dropout: MLP dropout (mlp 타입에만 적용)
            device: 디바이스 ("cuda", "cpu" 등)
            dtype: 데이터 타입 ("float16", "bfloat16", "float32")
        
        Returns:
            ValueModel 인스턴스
        """
        # dtype 변환
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # HuggingFace LlamaModel 로드
        config = LlamaConfig.from_pretrained(model_path)
        backbone = LlamaModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        
        # 디바이스 이동
        if device != "cpu":
            backbone = backbone.to(device)
        
        # Value Head 생성
        hidden_size = config.hidden_size
        value_head = create_value_head(hidden_size, value_head_type, dropout)
        value_head = value_head.to(device=device, dtype=torch_dtype)
        
        return cls(backbone, value_head, config)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "ValueModel":
        """Critic checkpoint에서 Value Model 로드
        
        Args:
            checkpoint_path: checkpoint 파일 경로
            device: 디바이스
        
        Returns:
            ValueModel 인스턴스 (학습된 weights 로드됨)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Config에서 모델 설정 추출
        config_dict = checkpoint.get("config", {})
        
        # 모델 경로 추출 (여러 경로 시도)
        models_config = config_dict.get("models", {})
        value_model_config = models_config.get("value_model", {})
        model_path = value_model_config.get("path")
        
        if model_path is None:
            raise ValueError(
                f"Checkpoint에 모델 경로가 없습니다. "
                f"config.models.value_model.path가 필요합니다."
            )
        
        # 학습 설정 추출
        training_config = config_dict.get("training", {})
        value_head_type = training_config.get("value_head_type", "mlp")
        dropout = training_config.get("dropout", 0.0)
        dtype = value_model_config.get("dtype", "bfloat16")
        
        # 모델 생성
        model = cls.from_pretrained(
            model_path=model_path,
            value_head_type=value_head_type,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # State dict 로드
        if "backbone_state_dict" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        if "value_head_state_dict" in checkpoint:
            model.value_head.load_state_dict(checkpoint["value_head_state_dict"])
        
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            input_ids: [batch, seq] 입력 토큰 ID
            attention_mask: [batch, seq] 어텐션 마스크 (optional)
        
        Returns:
            value_logits: [batch, seq, 1] 토큰별 value 예측
        """
        # Backbone forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # Value head
        value_logits = self.value_head(hidden_states)  # [batch, seq, 1]
        
        return value_logits
    
    def freeze_backbone(self) -> None:
        """Backbone frozen 설정 (value head만 학습)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Backbone 학습 가능 설정"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def eval_mode(self) -> None:
        """Eval only 모드 (전체 frozen, eval 상태)"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def get_trainable_parameters(self) -> list:
        """학습 가능한 파라미터 반환"""
        return [p for p in self.parameters() if p.requires_grad]
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension 반환"""
        return self.config.hidden_size
    
    @property
    def num_layers(self) -> int:
        """Transformer layer 수 반환"""
        return self.config.num_hidden_layers

