"""Meta LLaMA MTP Adapter

Transformer를 감싸서 MTP 학습에 필요한 기능 제공
LoRA (Low-Rank Adaptation) 적용 지원

Value function은 별도 ValueModel에서 처리 (독립 모델)
"""

from typing import Optional

import torch
from torch import nn

from .transformer import Transformer, ModelArgs


# LoRA 기본 설정
DEFAULT_LORA_CONFIG = {
    "rank": 8,
    "alpha": 16.0,
    "dropout": 0.0,
    "target_modules": ["wq", "wk", "wv", "wo"],
}


class MetaLlamaMTPAdapter(nn.Module):
    """Meta LLaMA MTP Adapter

    순수 MTP 모델. Value head 없음.
    Value function은 별도 ValueModel에서 처리.

    Args:
        transformer: Transformer 인스턴스
        model_args: ModelArgs (params.json)
        lora_enabled: LoRA 적용 여부
    """

    def __init__(
        self,
        transformer: Transformer,
        model_args: ModelArgs,
        lora_enabled: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.model_args = model_args
        self.lora_enabled = lora_enabled

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        dtype: Optional[str] = None,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        params_override: Optional[dict] = None,
    ) -> "MetaLlamaMTPAdapter":
        """Pretrained 모델에서 Adapter 로드

        Args:
            model_path: 모델 경로
                - 디렉토리: safetensors 형식 로드 (storage/models/meta-llama-mtp)
                - .pt 파일: checkpoint 형식 로드 (storage/checkpoints/.../checkpoint.pt)
            device: 디바이스 ("cuda", "mps", "cpu", "auto")
            dtype: 데이터 타입 ("float16", "bfloat16", None이면 safetensors 기본값 사용)
            use_lora: LoRA 적용 여부 (trunk layers에만 적용)
            lora_config: LoRA 설정 (rank, alpha, dropout, target_modules)
                - None이면 DEFAULT_LORA_CONFIG 사용
            params_override: ModelArgs 오버라이드 설정 (max_seq_len 등)
                - n_future_tokens 변경은 지원하지 않음 (MTP/NTP 구조 불일치)
                - NTP가 필요하면 순수 LLaMA2 checkpoint 사용 권장

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
                use_lora=use_lora,
                lora_config=lora_config,
                params_override=params_override,
            )

        # Dtype 변환 (문자열 -> torch.dtype)
        dtype_obj = None
        if dtype is not None:
            dtype_obj = getattr(torch, dtype)

        # 1. Transformer 로드 (params_override 전달)
        transformer = load_meta_mtp_model(
            model_dir=model_path,
            device=device,
            dtype=dtype_obj,
            params_override=params_override,
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

        # 파라미터 오버라이드 적용 (Adapter용 ModelArgs도 업데이트)
        if params_override:
            valid_keys = ModelArgs.__dataclass_fields__.keys()
            filtered_override = {k: v for k, v in params_override.items() if k in valid_keys}
            if filtered_override:
                params_dict.update(filtered_override)

        model_args = ModelArgs(**params_dict)

        # 3. Adapter 생성
        adapter = cls(
            transformer=transformer,
            model_args=model_args,
            lora_enabled=use_lora,
        )

        # 4. LoRA 적용 (FSDP wrapping 전에 적용해야 함)
        if use_lora:
            adapter.apply_lora(lora_config)

        return adapter

    def apply_lora(self, lora_config: Optional[dict] = None) -> None:
        """Trunk layers에 LoRA 적용

        FSDP wrapping 전에 호출해야 함
        Trunk layers의 attention 및 feed_forward에 적용 (extra_heads 제외)

        Args:
            lora_config: LoRA 설정 (rank, alpha, dropout, target_modules)
                - None이면 DEFAULT_LORA_CONFIG 사용
                - target_modules: attention (wq, wk, wv, wo), ffn (w1, w2, w3)
        """
        from weighted_mtp.models.lora import apply_lora_to_linear

        # Config 병합 (사용자 설정 + 기본값)
        config = {**DEFAULT_LORA_CONFIG}
        if lora_config:
            config.update(lora_config)

        rank = config["rank"]
        alpha = config["alpha"]
        lora_dropout = config["dropout"]
        target_modules = config["target_modules"]

        # Attention 모듈: wq, wk, wv, wo
        attention_targets = [t for t in target_modules if t in ["wq", "wk", "wv", "wo"]]
        # Feed-forward 모듈: w1, w2, w3
        ffn_targets = [t for t in target_modules if t in ["w1", "w2", "w3"]]

        # Trunk layers에 LoRA 적용
        for layer in self.transformer.layers:
            # Attention에 LoRA 적용
            if attention_targets:
                apply_lora_to_linear(
                    module=layer.attention,
                    target_names=attention_targets,
                    rank=rank,
                    alpha=alpha,
                    dropout=lora_dropout,
                )
            # Feed-forward에 LoRA 적용
            if ffn_targets:
                apply_lora_to_linear(
                    module=layer.feed_forward,
                    target_names=ffn_targets,
                    rank=rank,
                    alpha=alpha,
                    dropout=lora_dropout,
                )

        # 원본 trunk 파라미터 frozen (LoRA 파라미터만 학습)
        for layer in self.transformer.layers:
            for name, param in layer.named_parameters():
                # LoRA 파라미터는 이미 requires_grad=True
                if "lora_A" not in name and "lora_B" not in name:
                    param.requires_grad = False

        # Embedding, output, norm도 frozen
        for param in self.transformer.tok_embeddings.parameters():
            param.requires_grad = False
        for param in self.transformer.output.parameters():
            param.requires_grad = False
        for param in self.transformer.norm.parameters():
            param.requires_grad = False

        # extra_heads는 full fine-tuning 유지 (requires_grad=True)
        # MTP heads는 효율적 학습을 위해 frozen하지 않음

        self.lora_enabled = True

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """학습 대상 파라미터만 반환

        LoRA 모드: LoRA 파라미터 + extra_heads 파라미터
        일반 모드: 모든 파라미터 (requires_grad=True인 것만)

        Returns:
            학습 가능한 파라미터 리스트
        """
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def merge_lora(self) -> None:
        """LoRA 가중치를 원본에 병합 (inference 최적화)

        병합 후에는 추가 연산 없이 원본 Linear처럼 동작
        학습 재개 시에는 다시 LoRA를 적용해야 함
        """
        if not self.lora_enabled:
            return

        from weighted_mtp.models.lora import LoRALinear

        # Trunk layers의 LoRA 병합
        for layer in self.transformer.layers:
            # Attention 모듈
            for name in ["wq", "wk", "wv", "wo"]:
                if hasattr(layer.attention, name):
                    module = getattr(layer.attention, name)
                    if isinstance(module, LoRALinear):
                        module.merge_weights()
            # Feed-forward 모듈
            for name in ["w1", "w2", "w3"]:
                if hasattr(layer.feed_forward, name):
                    module = getattr(layer.feed_forward, name)
                    if isinstance(module, LoRALinear):
                        module.merge_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass

        Args:
            input_ids: [batch, seq] 입력 토큰
            return_hidden_states: True면 hidden_states도 함께 반환 (Value Head용)

        Returns:
            return_hidden_states=True:
                {"logits": tensor, "hidden_states": tensor}
            기본:
                logits: [batch, seq, n_future_tokens, vocab]
        """
        if return_hidden_states:
            logits, hidden_states = self.transformer(
                input_ids,
                start_pos=0,
                return_all_heads=True,
                return_hidden_states=True,
            )
            return {
                "logits": logits,
                "hidden_states": hidden_states,
            }

        # 기본: MTP logits만 반환
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
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
    ) -> "MetaLlamaMTPAdapter":
        """Checkpoint 파일에서 전체 adapter 로드

        Args:
            checkpoint_path: .pt checkpoint 파일 경로
            device: 디바이스
            dtype: 데이터 타입
            use_lora: LoRA 적용 여부
            lora_config: LoRA 설정

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

        # 2. LoRA 가중치 존재 여부 확인
        has_lora_weights = any("lora_A" in k or "lora_B" in k for k in state_dict.keys())

        # 3. State dict에서 모델 구조 추론
        # trunk layers 개수
        n_trunk = max(
            int(k.split(".")[2]) for k in state_dict.keys()
            if k.startswith("transformer.layers.") and "lora_" not in k
        ) + 1

        # extra_heads 개수 (n_future_tokens - 1)
        extra_heads_indices = set()
        for k in state_dict.keys():
            if "transformer.extra_heads." in k:
                idx = int(k.split("extra_heads.")[1].split(".")[0])
                extra_heads_indices.add(idx)
        n_extra = len(extra_heads_indices)
        n_future_tokens = n_extra + 1  # extra_heads 개수 + 1 (최소 1)

        # n_layers = trunk + extra (Transformer 구조에서 역산)
        n_layers = n_trunk + n_extra

        dim = state_dict["transformer.tok_embeddings.weight"].shape[1]
        vocab_size = state_dict["transformer.tok_embeddings.weight"].shape[0]

        # n_heads 추론 (wq weight 또는 wq.linear.weight에서)
        wq_key = "transformer.layers.0.attention.wq.weight"
        wq_linear_key = "transformer.layers.0.attention.wq.linear.weight"
        if wq_key in state_dict:
            wq_shape = state_dict[wq_key].shape
        elif wq_linear_key in state_dict:
            wq_shape = state_dict[wq_linear_key].shape
        else:
            raise KeyError(f"wq weight를 찾을 수 없습니다: {checkpoint_path}")

        head_dim = dim // 32  # 기본 head_dim 추정
        n_heads = wq_shape[0] // head_dim

        # 4. ModelArgs 생성
        model_args = ModelArgs(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            n_future_tokens=n_future_tokens,
        )

        # 5. Transformer 생성
        transformer = Transformer(model_args)

        # 6. LoRA 적용 결정
        # checkpoint에 LoRA 가중치가 있거나 use_lora=True면 LoRA 적용
        apply_lora = has_lora_weights or use_lora

        # 7. Adapter 생성
        adapter = cls(
            transformer=transformer,
            model_args=model_args,
            lora_enabled=apply_lora,
        )

        # 8. LoRA 적용 (state_dict 로드 전에 구조 생성 필요)
        if apply_lora:
            adapter.apply_lora(lora_config)

        # 9. state_dict 로드
        # value_head 관련 키 필터링 (기존 checkpoint 호환)
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if not k.startswith("value_head.")
        }

        # checkpoint에 LoRA 가중치가 없고 use_lora=True인 경우:
        # base weights만 로드하고 LoRA는 새로 초기화된 상태 유지
        if use_lora and not has_lora_weights:
            # LoRA 적용된 모듈의 키 변환: wq.weight → wq.linear.weight
            # LoRALinear 내부 구조: linear (nn.Linear) + lora_A + lora_B
            config = {**DEFAULT_LORA_CONFIG}
            if lora_config:
                config.update(lora_config)
            target_modules = config["target_modules"]
            
            # Attention 모듈: wq, wk, wv, wo
            attention_targets = [t for t in target_modules if t in ["wq", "wk", "wv", "wo"]]
            # Feed-forward 모듈: w1, w2, w3
            ffn_targets = [t for t in target_modules if t in ["w1", "w2", "w3"]]
            
            converted_state_dict = {}
            for key, value in filtered_state_dict.items():
                new_key = key
                
                # transformer.layers.X만 변환 (extra_heads는 LoRA 미적용이므로 변환 안함)
                if not key.startswith("transformer.layers."):
                    converted_state_dict[key] = value
                    continue
                
                # Attention 모듈 변환 (layers만)
                for target in attention_targets:
                    old_pattern = f".attention.{target}.weight"
                    new_pattern = f".attention.{target}.linear.weight"
                    if old_pattern in key:
                        new_key = key.replace(old_pattern, new_pattern)
                        break
                    old_bias = f".attention.{target}.bias"
                    new_bias = f".attention.{target}.linear.bias"
                    if old_bias in key:
                        new_key = key.replace(old_bias, new_bias)
                        break
                
                # Feed-forward 모듈 변환 (layers만)
                for target in ffn_targets:
                    old_pattern = f".feed_forward.{target}.weight"
                    new_pattern = f".feed_forward.{target}.linear.weight"
                    if old_pattern in key:
                        new_key = key.replace(old_pattern, new_pattern)
                        break
                    old_bias = f".feed_forward.{target}.bias"
                    new_bias = f".feed_forward.{target}.linear.bias"
                    if old_bias in key:
                        new_key = key.replace(old_bias, new_bias)
                        break
                
                converted_state_dict[new_key] = value
            
            # 변환된 state_dict 로드
            missing_keys, unexpected_keys = adapter.load_state_dict(converted_state_dict, strict=False)
            
            # LoRA 관련 missing keys는 정상 (새로 초기화됨)
            lora_missing = [k for k in missing_keys if "lora_A" in k or "lora_B" in k]
            other_missing = [k for k in missing_keys if "lora_A" not in k and "lora_B" not in k]
            
            if other_missing:
                raise KeyError(f"Checkpoint에서 필수 키 누락: {other_missing[:5]}...")
            
            print(f"[LoRA] Base checkpoint 로드 완료. LoRA weights 새로 초기화됨 ({len(lora_missing)} keys)")
        else:
            # 일반 로드 (LoRA 포함 checkpoint 또는 LoRA 미사용)
            adapter.load_state_dict(filtered_state_dict, strict=False)

        # 10. Device 및 dtype 설정
        adapter = adapter.to(device_obj)
        if dtype is not None:
            dtype_obj = getattr(torch, dtype)
            adapter = adapter.to(dtype_obj)

        return adapter
