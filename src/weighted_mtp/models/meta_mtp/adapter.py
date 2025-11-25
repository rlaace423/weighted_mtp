"""Meta LLaMA MTP Adapter

Transformer를 감싸서 trunk/full forward를 제공하는 Adapter
"""

from typing import Optional

import torch
from torch import nn

from .transformer import Transformer, ModelArgs
from .value_head import ValueHeadType, create_value_head


class MetaLlamaMTPAdapter(nn.Module):
    """Meta LLaMA MTP Adapter

    Transformer를 감싸서 WMTP 학습에 필요한 기능 제공:
    - trunk_forward(): Value head 학습 전용 (Stage 1)
    - full_forward(): Weighted training 전용 (Stage 2)

    Args:
        transformer: Transformer 인스턴스
        model_args: ModelArgs (params.json)
        value_head: ValueHead 인스턴스 (선택적, Stage 1에서 추가)
    """

    def __init__(
        self,
        transformer: Transformer,
        model_args: ModelArgs,
        value_head: Optional[ValueHeadType] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.model_args = model_args
        self.value_head = value_head

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        dtype: Optional[str] = None,
        initialize_value_head: bool = True,
        value_head_type: str = "mlp",
    ) -> "MetaLlamaMTPAdapter":
        """Pretrained 모델에서 Adapter 로드

        Args:
            model_path: 모델 경로
                - 디렉토리: safetensors 형식 로드 (storage/models/meta-llama-mtp)
                - .pt 파일: checkpoint 형식 로드 (storage/checkpoints/.../checkpoint.pt)
            device: 디바이스 ("cuda", "mps", "cpu", "auto")
            dtype: 데이터 타입 ("float16", "bfloat16", None이면 safetensors 기본값 사용)
            initialize_value_head: Value head 초기화 여부 (safetensors 로드 시에만 적용)
                - True: Critic/Verifiable Stage용 (기본값)
                - False: Rho-1 Stage용 (Value head 불필요)
            value_head_type: Value head 타입 ("linear" 또는 "mlp")

        Returns:
            MetaLlamaMTPAdapter 인스턴스

        Raises:
            FileNotFoundError: params.json 또는 config.json 미발견
            KeyError: checkpoint에 adapter_state_dict 키 없음
        """
        import json
        from pathlib import Path

        from .checkpoints import load_meta_mtp_model

        model_path = Path(model_path)

        # Checkpoint 파일 (.pt) 감지 → 전체 adapter 로드
        if model_path.suffix == ".pt":
            return cls._from_checkpoint(
                checkpoint_path=model_path,
                device=device,
                dtype=dtype,
                value_head_type=value_head_type,
            )

        # Dtype 변환 (문자열 -> torch.dtype)
        dtype_obj = None
        if dtype is not None:
            dtype_obj = getattr(torch, dtype)

        # 1. Transformer 로드
        transformer = load_meta_mtp_model(
            model_dir=model_path,
            device=device,
            dtype=dtype_obj,
        )

        # 2. ModelArgs 파싱 (params.json 또는 config.json)
        params_path = model_path / "configs/params.json"
        config_path = model_path / "configs/config.json"

        if params_path.exists():
            with open(params_path) as f:
                params_dict = json.load(f)
        elif config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            # config.json 형식을 ModelArgs로 변환
            params_dict = {
                "dim": config_dict.get("hidden_size", config_dict.get("dim")),
                "n_layers": config_dict.get("num_hidden_layers", config_dict.get("n_layers")),
                "n_heads": config_dict.get("num_attention_heads", config_dict.get("n_heads")),
                "n_kv_heads": config_dict.get(
                    "num_key_value_heads", config_dict.get("n_kv_heads")
                ),
                "vocab_size": config_dict.get("vocab_size"),
                "n_future_tokens": config_dict.get("n_future_tokens", 1),
                "rope_theta": config_dict.get("rope_theta", 10000.0),
                "max_seq_len": config_dict.get("max_position_embeddings", 2048),
                "norm_eps": config_dict.get("rms_norm_eps", 1e-5),
            }
        else:
            raise FileNotFoundError(
                f"Neither params.json nor config.json found in {model_path}/configs/"
            )

        model_args = ModelArgs(**params_dict)

        # 3. Value Head 초기화 (선택적)
        value_head = None
        if initialize_value_head:
            value_head = create_value_head(
                hidden_size=model_args.dim,
                head_type=value_head_type,
            )

            # Device 이동 (Transformer와 동일 device)
            device_obj = transformer.tok_embeddings.weight.device
            value_head = value_head.to(device_obj)

            # Dtype 설정 (Transformer dtype과 일치)
            if dtype_obj is not None:
                value_head = value_head.to(dtype_obj)
            elif hasattr(transformer.tok_embeddings.weight, "dtype"):
                value_head = value_head.to(transformer.tok_embeddings.weight.dtype)

        # 4. Adapter 생성
        adapter = cls(
            transformer=transformer,
            model_args=model_args,
            value_head=value_head,
        )

        return adapter

    def attach_value_head(self, value_head: ValueHeadType):
        """Value head 추가 (Stage 1 시작 전)

        Args:
            value_head: ValueHead 인스턴스
        """
        self.value_head = value_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_value_logits: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """통일된 forward 인터페이스 (FSDP 호환)

        모든 파이프라인이 동일한 메서드를 통해 호출
        FSDP hook이 정상 동작하여 parameter unshard 보장

        Args:
            input_ids: [batch, seq] 입력 토큰
            attention_mask: [batch, seq] attention mask (현재 미사용, 향후 확장)
            return_value_logits: True면 value_logits 반환 (Critic/Verifiable용)
            return_hidden_states: True면 hidden_states 반환 (Verifiable용)

        Returns:
            return_value_logits=False:
                logits: [batch, seq, n_future_tokens, vocab]

            return_value_logits=True (Critic Stage 1: Value head만):
                {
                    "value_logits": [batch, seq, 1],
                    "hidden_states": [batch, seq, hidden_size],
                }

            return_value_logits=True + return_hidden_states=True (Verifiable Stage 2):
                {
                    "logits": [batch, seq, n_future_tokens, vocab],
                    "value_logits": [batch, seq, 1],
                    "hidden_states": [batch, seq, hidden_size],
                }

        Raises:
            ValueError: return_value_logits=True인데 Value head가 없음
        """
        if return_value_logits and self.value_head is None:
            raise ValueError(
                "Value head not initialized. "
                "Set initialize_value_head=True in from_pretrained() or call attach_value_head()."
            )

        # Critic Stage 1: Value head만 학습 (MTP heads 계산 생략)
        if return_value_logits and not return_hidden_states:
            _, hidden_states = self.transformer(
                input_ids,
                start_pos=0,
                return_all_heads=False,
                return_hidden_states=True,
            )
            value_logits = self.value_head(hidden_states)
            return {
                "value_logits": value_logits,
                "hidden_states": hidden_states,
            }

        # Verifiable Stage 2: MTP + Value 동시 학습
        if return_value_logits and return_hidden_states:
            logits, hidden_states = self.transformer(
                input_ids,
                start_pos=0,
                return_all_heads=True,
                return_hidden_states=True,
            )
            value_logits = self.value_head(hidden_states)
            return {
                "logits": logits,
                "value_logits": value_logits,
                "hidden_states": hidden_states,
            }

        # Baseline/Rho-1: MTP만 (Value head 없음)
        logits = self.transformer(
            input_ids,
            start_pos=0,
            return_all_heads=True,
        )
        return logits

    @classmethod
    def _from_checkpoint(
        cls,
        checkpoint_path,
        device: str = "auto",
        dtype: Optional[str] = None,
        value_head_type: str = "mlp",
    ) -> "MetaLlamaMTPAdapter":
        """Checkpoint 파일에서 전체 adapter 로드

        Args:
            checkpoint_path: .pt checkpoint 파일 경로
            device: 디바이스
            dtype: 데이터 타입

        Returns:
            MetaLlamaMTPAdapter (전체 state_dict 로드됨)

        Raises:
            KeyError: checkpoint에 adapter_state_dict 키 없음
        """
        import torch
        from pathlib import Path

        from .checkpoints import _get_device

        checkpoint_path = Path(checkpoint_path)
        device_obj = _get_device(device)

        # 1. Checkpoint 로드
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)

        if "adapter_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint에 'adapter_state_dict' 키가 없습니다: {checkpoint_path}")

        state_dict = checkpoint["adapter_state_dict"]

        # 2. State dict에서 모델 구조 추론
        n_layers = max(
            int(k.split(".")[2]) for k in state_dict.keys()
            if k.startswith("transformer.layers.")
        ) + 1

        dim = state_dict["transformer.tok_embeddings.weight"].shape[1]
        vocab_size = state_dict["transformer.tok_embeddings.weight"].shape[0]

        # n_heads 추론 (wq weight shape에서)
        wq_shape = state_dict["transformer.layers.0.attention.wq.weight"].shape
        head_dim = dim // 32  # 기본 head_dim 추정
        n_heads = wq_shape[0] // head_dim

        # n_future_tokens 추론 (extra_heads 개수에서)
        extra_heads_indices = set()
        for k in state_dict.keys():
            if "transformer.extra_heads." in k:
                idx = int(k.split("extra_heads.")[1].split(".")[0])
                extra_heads_indices.add(idx)
        n_future_tokens = len(extra_heads_indices) + 1  # extra_heads 개수 + 1 (최소 1)

        # 3. ModelArgs 생성
        model_args = ModelArgs(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            n_future_tokens=n_future_tokens,
        )

        # 4. Transformer 생성
        transformer = Transformer(model_args)

        # 5. ValueHead 생성 (checkpoint에 있는 경우, 타입 자동 감지)
        value_head = None
        value_head_keys = [k for k in state_dict.keys() if k.startswith("value_head.")]
        if value_head_keys:
            # state_dict 키 구조로 value_head_type 자동 감지
            if any("value_head.mlp." in k for k in value_head_keys):
                detected_type = "mlp"
            elif any("value_head.linear." in k for k in value_head_keys):
                # linear와 sigmoid는 구조가 동일, sigmoid는 forward에서만 차이
                # checkpoint만으로 구분 불가하므로 linear로 기본 설정
                detected_type = "linear"
            else:
                detected_type = value_head_type  # fallback

            value_head = create_value_head(
                hidden_size=dim,
                head_type=detected_type,
            )

        # 6. Adapter 생성 및 state_dict 로드
        adapter = cls(
            transformer=transformer,
            model_args=model_args,
            value_head=value_head,
        )

        adapter.load_state_dict(state_dict)

        # 7. Device 및 dtype 설정
        adapter = adapter.to(device_obj)
        if dtype is not None:
            dtype_obj = getattr(torch, dtype)
            adapter = adapter.to(dtype_obj)

        return adapter
