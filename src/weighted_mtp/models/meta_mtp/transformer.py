"""순수 PyTorch Meta LLaMA MTP Transformer 구현

Meta 레퍼런스 구조를 참고하되 다음을 개선:
- fairscale 의존성 제거 (순수 PyTorch)
- FSDP 호환성 확보
- device-agnostic 설계
- 학습 가능 (gradient 계산 가능)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    """Meta LLaMA MTP 모델 설정

    Meta params.json과 호환되는 구조
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    n_future_tokens: int = 1
    rope_theta: float = 10000.0
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """RoPE 주파수 사전 계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """주파수 텐서를 broadcasting을 위해 reshape"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE 적용"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA를 위한 KV heads 반복"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention with GQA support"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV cache (동적 생성으로 변경, device-agnostic)
        self.cache_k = None
        self.cache_v = None
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV cache 동적 생성 (device-agnostic)
        if self.cache_k is None or self.cache_k.device != x.device:
            self.cache_k = torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim),
                dtype=x.dtype,
                device=x.device
            )
            self.cache_v = torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim),
                dtype=x.dtype,
                device=x.device
            )

        # Training 시 computational graph 공유 방지를 위해 detach
        # xk, xv를 그대로 저장하면 이전 iteration의 graph가 cache에 남아
        # 다음 backward 시 "backward through the graph a second time" 에러 발생
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network"""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Meta LLaMA MTP Transformer (순수 PyTorch 구현)

    Meta 구조를 참고하되 다음을 개선:
    - fairscale 제거 (순수 PyTorch nn.Linear/Embedding)
    - @torch.inference_mode() 제거 (학습 가능)
    - device-agnostic 설계
    - FSDP 호환성
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_future_tokens = params.n_future_tokens

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Trunk layers (n_layers - n_future_tokens + 1개)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers - self.n_future_tokens + 1):
            self.layers.append(TransformerBlock(layer_id, params))

        # Extra heads for MTP (n_future_tokens - 1개)
        self.extra_heads = nn.ModuleList()
        for layer_id in range(self.n_layers - self.n_future_tokens + 1, self.n_layers):
            self.extra_heads.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # RoPE freqs_cis 사전 계산
        # register_buffer 대신 일반 속성으로 저장 (safetensors 호환성)
        # complex64 타입은 safetensors 미지원, state_dict에 포함 불필요
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            theta=params.rope_theta
        )

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        return_all_heads: bool = False,
        return_hidden_states: bool = False,
    ):
        """Forward pass

        Args:
            tokens: [batch, seq] 입력 토큰
            start_pos: KV cache 시작 위치 (학습 시 0)
            return_all_heads: True면 모든 MTP heads 반환
            return_hidden_states: True면 hidden_states도 함께 반환 (Value Head용)

        Returns:
            if return_hidden_states:
                (logits, hidden_states) tuple
                - logits: [batch, seq, n_future_tokens, vocab] or [batch, seq, vocab]
                - hidden_states: [batch, seq, hidden_size] (trunk 마지막 layer, norm 적용 후)
            else:
                logits: [batch, seq, n_future_tokens, vocab] or [batch, seq, vocab]
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # freqs_cis를 입력 tokens와 동일한 device로 이동
        # 일반 속성이므로 명시적으로 device 이동 필요
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(tokens.device)

        # Causal mask 생성
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        # Trunk forward (마지막 layer 제외)
        for layer in self.layers[:-1]:
            h = layer(h, start_pos, freqs_cis, mask)
        h_trunk = h

        # Prediction heads
        latents = []
        n_heads_to_use = self.n_future_tokens if return_all_heads else 1
        prediction_heads = [self.layers[-1]] + list(self.extra_heads)
        for layer in prediction_heads[:n_heads_to_use]:
            h = layer(h_trunk, start_pos, freqs_cis, mask)
            latents.append(h)

        # Stack and normalize
        h = torch.stack(latents, dim=-2)  # [batch, seq, n_heads_to_use, dim]
        h = self.norm(h)
        output = self.output(h)  # [batch, seq, n_heads_to_use, vocab]

        if return_hidden_states:
            # Value Head용 hidden_states 반환
            # h_trunk의 마지막 layer 출력에 norm 적용
            hidden_states = self.norm(h_trunk.unsqueeze(-2)).squeeze(-2)
            return output, hidden_states

        return output
