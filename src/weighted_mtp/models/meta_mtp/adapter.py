"""Meta LLaMA MTP Adapter

Transformer를 감싸서 trunk/full forward를 제공하는 Adapter
"""

from typing import Optional

import torch
from torch import nn

from .transformer import Transformer, ModelArgs
from .value_head import ValueHead


class MetaLlamaMTPAdapter(nn.Module):
    """Meta LLaMA MTP Adapter

    Transformer를 감싸서 WMTP 학습에 필요한 기능 제공:
    - trunk_forward(): Value head 학습 전용 (Stage 1)
    - full_forward(): Weighted training 전용 (Stage 2)

    Args:
        transformer: Transformer 인스턴스
        model_args: ModelArgs (params.json)
        value_head: ValueHead (선택적, Stage 1에서 추가)
    """

    def __init__(
        self,
        transformer: Transformer,
        model_args: ModelArgs,
        value_head: Optional[ValueHead] = None,
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
    ) -> "MetaLlamaMTPAdapter":
        """Pretrained 모델에서 Adapter 로드

        Args:
            model_path: 모델 디렉터리 경로 (storage/models_v2/meta-llama-mtp 또는 micro-mtp)
            device: 디바이스 ("cuda", "mps", "cpu", "auto")
            dtype: 데이터 타입 ("float16", "bfloat16", None이면 safetensors 기본값 사용)
            initialize_value_head: Value head 초기화 여부
                - True: Critic/Verifiable Stage용 (기본값)
                - False: Rho-1 Stage용 (Value head 불필요)

        Returns:
            MetaLlamaMTPAdapter 인스턴스

        Raises:
            FileNotFoundError: params.json 또는 config.json 미발견
        """
        import json
        from pathlib import Path

        from .checkpoints import load_meta_mtp_model

        model_path = Path(model_path)

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
            value_head = ValueHead(hidden_size=model_args.dim)

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

    def attach_value_head(self, value_head: ValueHead):
        """Value head 추가 (Stage 1 시작 전)

        Args:
            value_head: ValueHead 인스턴스
        """
        self.value_head = value_head

    def trunk_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Stage 1: Value head 학습 전용 forward

        MTP output heads를 사용하지 않고 Value head만 학습
        학습 속도 향상 (output heads gradient 없음)

        Args:
            input_ids: [batch, seq] 입력 토큰
            attention_mask: [batch, seq] attention mask (현재 미사용, 향후 확장)

        Returns:
            {
                "hidden_states": [batch, seq, hidden_size],
                "value_logits": [batch, seq, 1],
            }

        Raises:
            ValueError: Value head가 초기화되지 않음
        """
        if self.value_head is None:
            raise ValueError("Value head not initialized. Call attach_value_head() first.")

        # Transformer forward (return_all_heads=False → 1개 head만)
        # output: [batch, seq, 1, vocab]
        output = self.transformer(input_ids, start_pos=0, return_all_heads=False)

        # hidden_states 추출: norm 적용 전 trunk의 마지막 hidden state
        # Transformer에서 h_trunk를 직접 반환하지 않으므로,
        # 대신 norm 전 마지막 layer의 출력을 재계산
        # 효율성을 위해 forward를 두 번 호출하지 않고, 내부 구조 활용

        # 간단한 방법: Transformer의 trunk layers를 직접 실행
        _bsz, seqlen = input_ids.shape
        h = self.transformer.tok_embeddings(input_ids)

        # freqs_cis를 입력 tokens와 동일한 device로 이동 (MPS/CUDA 호환성)
        freqs_cis = self.transformer.freqs_cis[0:seqlen].to(input_ids.device)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)

        # Trunk forward (마지막 layer 제외)
        for layer in self.transformer.layers[:-1]:
            h = layer(h, 0, freqs_cis, mask)

        # 마지막 trunk layer
        h_trunk = self.transformer.layers[-1](h, 0, freqs_cis, mask)

        # Normalization 적용 (Value head 입력 전 필수)
        hidden_states = self.transformer.norm(h_trunk.unsqueeze(-2)).squeeze(-2)

        # Value head
        value_logits = self.value_head(hidden_states)

        return {
            "hidden_states": hidden_states,
            "value_logits": value_logits,
        }

    def full_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Stage 2: Weighted training 전용 forward

        MTP output heads + Value head 모두 사용
        전체 gradient 계산 (MTP + Value)

        Args:
            input_ids: [batch, seq] 입력 토큰
            attention_mask: [batch, seq] attention mask (현재 미사용, 향후 확장)

        Returns:
            {
                "logits": [batch, seq, n_future_tokens, vocab],
                "value_logits": [batch, seq, 1],
                "hidden_states": [batch, seq, hidden_size],
            }

        Raises:
            ValueError: Value head가 초기화되지 않음
        """
        if self.value_head is None:
            raise ValueError("Value head not initialized. Call attach_value_head() first.")

        # Transformer forward (모든 MTP heads 사용)
        # output: [batch, seq, n_future_tokens, vocab]
        logits = self.transformer(input_ids, start_pos=0, return_all_heads=True)

        # hidden_states 추출 (trunk_forward와 동일 방식)
        _bsz, seqlen = input_ids.shape
        h = self.transformer.tok_embeddings(input_ids)

        # freqs_cis를 입력 tokens와 동일한 device로 이동 (MPS/CUDA 호환성)
        freqs_cis = self.transformer.freqs_cis[0:seqlen].to(input_ids.device)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)

        # Trunk forward
        for layer in self.transformer.layers[:-1]:
            h = layer(h, 0, freqs_cis, mask)
        h_trunk = self.transformer.layers[-1](h, 0, freqs_cis, mask)

        # Normalization
        hidden_states = self.transformer.norm(h_trunk.unsqueeze(-2)).squeeze(-2)

        # Value head
        value_logits = self.value_head(hidden_states)

        return {
            "logits": logits,
            "value_logits": value_logits,
            "hidden_states": hidden_states,
        }
