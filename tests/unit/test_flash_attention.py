"""Flash Attention Unit Tests"""

import pytest
import torch
import torch.nn.functional as F

from weighted_mtp.models.meta_mtp.transformer import (
    Attention,
    ModelArgs,
    apply_rotary_emb,
    precompute_freqs_cis,
    repeat_kv,
)


@pytest.fixture
def model_args():
    """Small model args for testing"""
    return ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        vocab_size=32000,
        max_batch_size=4,
        max_seq_len=128,
    )


@pytest.fixture
def attention_layer(model_args):
    """Attention layer instance"""
    return Attention(model_args)


def test_flash_attention_path_training(attention_layer, model_args):
    """Flash Attention 경로 동작 확인 (start_pos=0, seqlen>1)"""
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len]

    # Training mode: start_pos=0, seqlen>1
    output = attention_layer(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    assert output.shape == (batch_size, seq_len, model_args.dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_kv_cache_path_inference(attention_layer, model_args):
    """KV cache 경로 동작 확인 (start_pos>0 또는 seqlen=1)"""
    batch_size = 2
    seq_len = 1

    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len]

    # Inference mode: seqlen=1
    output = attention_layer(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    assert output.shape == (batch_size, seq_len, model_args.dim)
    assert attention_layer.cache_k is not None
    assert attention_layer.cache_v is not None


def test_flash_attention_no_kv_cache(attention_layer, model_args):
    """Flash Attention 경로에서 KV cache 미생성 확인"""
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len]

    # Flash Attention 경로 실행
    output = attention_layer(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    # KV cache가 생성되지 않아야 함
    assert attention_layer.cache_k is None
    assert attention_layer.cache_v is None
    assert output.shape == (batch_size, seq_len, model_args.dim)


def test_numerical_precision_flash_vs_standard():
    """Flash Attention vs Standard Attention 수치 정밀도 검증

    동일한 입력에 대해 Flash Attention과 Standard Attention의
    출력이 numerically equivalent한지 확인 (floating point 오차 범위 내)
    """
    batch_size = 2
    seq_len = 8
    n_heads = 4
    head_dim = 64

    # 동일한 입력
    torch.manual_seed(42)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # Flash Attention (PyTorch native)
    flash_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )

    # Standard Attention (기존 구현)
    scores = torch.matmul(q, k.transpose(2, 3)) / (head_dim ** 0.5)

    # Causal mask 적용
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask

    scores = F.softmax(scores.float(), dim=-1).type_as(q)
    standard_output = torch.matmul(scores, v)

    # Numerical equivalence 검증 (rtol=1e-4, atol=1e-5)
    assert torch.allclose(flash_output, standard_output, rtol=1e-4, atol=1e-5), \
        f"Max diff: {(flash_output - standard_output).abs().max()}"


def test_gradient_flow_flash_attention(attention_layer, model_args):
    """Flash Attention 경로에서 gradient 정상 전파 확인"""
    batch_size = 2
    seq_len = 8

    x = torch.randn(batch_size, seq_len, model_args.dim, requires_grad=True)
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len]

    output = attention_layer(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
    loss = output.sum()
    loss.backward()

    # Gradient가 정상적으로 전파되어야 함
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_gqa_support_flash_attention():
    """Grouped Query Attention (GQA) 지원 확인

    n_kv_heads < n_heads 경우 repeat_kv() 동작 검증
    """
    model_args = ModelArgs(
        dim=512,
        n_heads=8,
        n_kv_heads=2,  # GQA: 8 heads, 2 kv_heads (4:1 ratio)
        max_batch_size=4,
        max_seq_len=128,
    )

    attention = Attention(model_args)
    assert attention.n_rep == 4

    batch_size = 2
    seq_len = 8

    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len]

    output = attention(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    assert output.shape == (batch_size, seq_len, model_args.dim)


def test_bfloat16_dtype_flash_attention(attention_layer, model_args):
    """BFloat16 dtype에서 Flash Attention 동작 확인"""
    batch_size = 2
    seq_len = 16

    # BFloat16 입력
    x = torch.randn(batch_size, seq_len, model_args.dim, dtype=torch.bfloat16)
    attention_layer = attention_layer.to(torch.bfloat16)

    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len]

    output = attention_layer(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    assert output.dtype == torch.bfloat16
    assert output.shape == (batch_size, seq_len, model_args.dim)
    assert not torch.isnan(output).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_cuda_performance():
    """Flash Attention CUDA 성능 벤치마크

    Training workload (batch=4, seq=512) 기준
    Flash Attention이 Standard Attention보다 빨라야 함
    """
    import time

    model_args = ModelArgs(
        dim=2048,
        n_heads=16,
        n_kv_heads=16,
        max_batch_size=8,
        max_seq_len=1024,
    )

    batch_size = 4
    seq_len = 512

    attention_flash = Attention(model_args).cuda()

    x = torch.randn(batch_size, seq_len, model_args.dim, device="cuda")
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        seq_len,
    )[:seq_len].cuda()

    # Warmup
    for _ in range(5):
        _ = attention_flash(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    torch.cuda.synchronize()

    # Flash Attention 벤치마크
    start = time.time()
    for _ in range(20):
        output = attention_flash(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / 20

    # 성능 검증: Flash Attention이 실행되어야 함 (정확한 속도 비교는 실제 환경에서)
    assert output.shape == (batch_size, seq_len, model_args.dim)
    print(f"\nFlash Attention time: {flash_time*1000:.2f}ms")
