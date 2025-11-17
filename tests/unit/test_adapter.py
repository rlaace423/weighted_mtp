"""Meta Adapter Unit Tests"""

import pytest
import torch
from pathlib import Path

from weighted_mtp.models.meta_mtp import (
    MetaLlamaMTPAdapter,
    ValueHead,
    load_meta_mtp_model,
    ModelArgs,
    Transformer,
)


@pytest.fixture
def micro_model_dir():
    """Micro 모델 디렉터리"""
    return Path("storage/models/micro-mtp")


@pytest.fixture
def micro_transformer():
    """Micro Transformer 인스턴스 (직접 생성)"""
    model_args = ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        vocab_size=32000,
        n_future_tokens=4,
        rope_theta=10000.0,
        max_seq_len=2048,
    )
    return Transformer(model_args)


@pytest.fixture
def micro_adapter(micro_transformer):
    """Micro Adapter (Value head 포함)"""
    model_args = ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        vocab_size=32000,
        n_future_tokens=4,
    )
    value_head = ValueHead(hidden_size=512)
    adapter = MetaLlamaMTPAdapter(micro_transformer, model_args, value_head)
    return adapter


def test_model_args_creation():
    """ModelArgs 생성 테스트"""
    args = ModelArgs(dim=512, n_layers=4, n_heads=8)
    assert args.dim == 512
    assert args.n_layers == 4
    assert args.n_heads == 8
    assert args.n_future_tokens == 1  # default


def test_transformer_creation(micro_transformer):
    """Transformer 생성 테스트"""
    assert micro_transformer.params.dim == 512
    assert micro_transformer.params.n_layers == 4
    # layers: n_layers - n_future_tokens + 1 = 4 - 4 + 1 = 1
    assert len(micro_transformer.layers) == 1
    # extra_heads: n_future_tokens - 1 = 4 - 1 = 3
    assert len(micro_transformer.extra_heads) == 3


def test_value_head_forward():
    """ValueHead forward 테스트"""
    value_head = ValueHead(hidden_size=512)
    hidden_states = torch.randn(2, 10, 512)  # [batch=2, seq=10, hidden=512]

    value_logits = value_head(hidden_states)

    assert value_logits.shape == (2, 10, 1)


def test_value_head_checkpoint_save_load(tmp_path):
    """ValueHead checkpoint 저장/로드 테스트"""
    value_head = ValueHead(hidden_size=512)

    # 저장
    ckpt_path = tmp_path / "value_head.pt"
    value_head.save_checkpoint(ckpt_path)

    # 로드
    loaded = ValueHead.load_checkpoint(ckpt_path, device="cpu")

    # 검증
    assert torch.allclose(value_head.linear.weight, loaded.linear.weight)
    assert loaded.hidden_size == 512


def test_trunk_forward_shape(micro_adapter):
    """trunk_forward() 출력 shape 테스트"""
    input_ids = torch.randint(0, 32000, (2, 10))  # [batch=2, seq=10]

    outputs = micro_adapter.trunk_forward(input_ids)

    assert "hidden_states" in outputs
    assert "value_logits" in outputs
    assert outputs["hidden_states"].shape == (2, 10, 512)
    assert outputs["value_logits"].shape == (2, 10, 1)


def test_full_forward_shape(micro_adapter):
    """full_forward() 출력 shape 테스트"""
    input_ids = torch.randint(0, 32000, (2, 10))

    outputs = micro_adapter.full_forward(input_ids)

    assert "logits" in outputs
    assert "value_logits" in outputs
    assert "hidden_states" in outputs
    assert outputs["logits"].shape == (2, 10, 4, 32000)  # [batch, seq, n_future_tokens, vocab]
    assert outputs["value_logits"].shape == (2, 10, 1)
    assert outputs["hidden_states"].shape == (2, 10, 512)


def test_trunk_forward_without_value_head():
    """Value head 없이 trunk_forward() 호출 시 에러"""
    model_args = ModelArgs(dim=512, n_layers=4, n_heads=8, vocab_size=32000)
    transformer = Transformer(model_args)
    adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head=None)

    input_ids = torch.randint(0, 32000, (2, 10))

    with pytest.raises(ValueError, match="Value head not initialized"):
        adapter.trunk_forward(input_ids)


def test_attach_value_head():
    """Value head 추가 테스트"""
    model_args = ModelArgs(dim=512, n_layers=4, n_heads=8, vocab_size=32000)
    transformer = Transformer(model_args)
    adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head=None)

    # Value head 추가
    value_head = ValueHead(hidden_size=512)
    adapter.attach_value_head(value_head)

    # 이제 trunk_forward() 가능
    input_ids = torch.randint(0, 32000, (2, 10))
    outputs = adapter.trunk_forward(input_ids)

    assert outputs["value_logits"].shape == (2, 10, 1)


def test_load_micro_model(micro_model_dir):
    """Micro 모델 로딩 테스트"""
    transformer = load_meta_mtp_model(micro_model_dir, device="cpu")

    assert transformer.params.dim == 512
    assert transformer.params.n_layers == 4
    assert transformer.params.n_future_tokens == 4

    # 구조 검증
    assert len(transformer.layers) == 1  # trunk layers
    assert len(transformer.extra_heads) == 3  # extra heads


def test_device_auto_selection():
    """device='auto' 자동 선택 테스트"""
    from weighted_mtp.models.meta_mtp.checkpoints import _get_device

    device = _get_device("auto")

    # cuda > mps > cpu 우선순위 확인
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_device_explicit_selection():
    """device 명시적 선택 테스트"""
    from weighted_mtp.models.meta_mtp.checkpoints import _get_device

    assert _get_device("cpu").type == "cpu"
