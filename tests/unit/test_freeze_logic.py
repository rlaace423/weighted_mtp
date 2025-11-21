"""Freeze Logic Unit Tests

trunk 블록 선택적 unfreeze 기능 테스트
"""

import pytest
import torch

from weighted_mtp.models.meta_mtp import (
    MetaLlamaMTPAdapter,
    ValueHead,
    ModelArgs,
    Transformer,
)


@pytest.fixture
def adapter_8_layers():
    """8개 레이어를 가진 Adapter (테스트용)

    n_future_tokens=1로 설정하여 모든 레이어가 trunk에 포함되도록 함
    """
    model_args = ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=1000,
        n_future_tokens=1,  # 모든 레이어가 trunk로
        rope_theta=10000.0,
        max_seq_len=512,
    )
    transformer = Transformer(model_args)
    value_head = ValueHead(hidden_size=256)
    adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head)
    return adapter


def apply_freeze_logic(adapter, num_unfrozen: int):
    """run_critic.py의 freeze 로직을 재현

    Args:
        adapter: MetaLlamaMTPAdapter
        num_unfrozen: unfreeze할 마지막 블록 수
    """
    n_layers = len(adapter.transformer.layers)

    if num_unfrozen > 0:
        # 전체 frozen
        for param in adapter.transformer.parameters():
            param.requires_grad = False

        # 마지막 N개 블록 unfreeze
        for layer in adapter.transformer.layers[-num_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True

        # final norm도 unfreeze
        for param in adapter.transformer.norm.parameters():
            param.requires_grad = True
    else:
        # 기존 동작: value head만 학습
        for param in adapter.transformer.parameters():
            param.requires_grad = False

    # Value head는 항상 학습
    for param in adapter.value_head.parameters():
        param.requires_grad = True


def count_trainable_params(module) -> int:
    """모듈의 trainable 파라미터 수 계산"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_num_unfrozen_layers_zero(adapter_8_layers):
    """num_unfrozen_layers=0일 때 value head만 trainable"""
    adapter = adapter_8_layers
    apply_freeze_logic(adapter, num_unfrozen=0)

    # Transformer 전체가 frozen
    transformer_trainable = count_trainable_params(adapter.transformer)
    assert transformer_trainable == 0, f"Transformer should be frozen, got {transformer_trainable}"

    # Value head만 trainable
    value_head_trainable = count_trainable_params(adapter.value_head)
    assert value_head_trainable > 0, "Value head should be trainable"

    # 전체 trainable = value head만
    total_trainable = count_trainable_params(adapter)
    assert total_trainable == value_head_trainable


def test_num_unfrozen_layers_four(adapter_8_layers):
    """num_unfrozen_layers=4일 때 마지막 4개 블록 + norm + value_head만 trainable"""
    adapter = adapter_8_layers
    apply_freeze_logic(adapter, num_unfrozen=4)

    n_layers = len(adapter.transformer.layers)
    assert n_layers == 8, f"Expected 8 layers, got {n_layers}"

    # 앞쪽 4개 블록은 frozen
    for i in range(4):
        layer_trainable = count_trainable_params(adapter.transformer.layers[i])
        assert layer_trainable == 0, f"Layer {i} should be frozen, got {layer_trainable}"

    # 뒤쪽 4개 블록은 trainable
    for i in range(4, 8):
        layer_trainable = count_trainable_params(adapter.transformer.layers[i])
        assert layer_trainable > 0, f"Layer {i} should be trainable"

    # Norm은 trainable
    norm_trainable = count_trainable_params(adapter.transformer.norm)
    assert norm_trainable > 0, "Norm should be trainable"

    # Value head는 trainable
    value_head_trainable = count_trainable_params(adapter.value_head)
    assert value_head_trainable > 0, "Value head should be trainable"

    # Embedding은 frozen
    embedding_trainable = sum(
        p.numel() for p in adapter.transformer.tok_embeddings.parameters()
        if p.requires_grad
    )
    assert embedding_trainable == 0, "Embeddings should be frozen"


def test_num_unfrozen_layers_exceeds_total(adapter_8_layers):
    """num_unfrozen_layers > n_layers일 때 전체 블록 학습"""
    adapter = adapter_8_layers
    apply_freeze_logic(adapter, num_unfrozen=100)  # 8개 레이어보다 큼

    # 모든 블록이 trainable
    for i, layer in enumerate(adapter.transformer.layers):
        layer_trainable = count_trainable_params(layer)
        assert layer_trainable > 0, f"Layer {i} should be trainable"

    # Norm도 trainable
    norm_trainable = count_trainable_params(adapter.transformer.norm)
    assert norm_trainable > 0, "Norm should be trainable"


def test_num_unfrozen_layers_one(adapter_8_layers):
    """num_unfrozen_layers=1일 때 마지막 1개 블록만 trainable"""
    adapter = adapter_8_layers
    apply_freeze_logic(adapter, num_unfrozen=1)

    # 앞쪽 7개 블록은 frozen
    for i in range(7):
        layer_trainable = count_trainable_params(adapter.transformer.layers[i])
        assert layer_trainable == 0, f"Layer {i} should be frozen"

    # 마지막 블록만 trainable
    last_layer_trainable = count_trainable_params(adapter.transformer.layers[-1])
    assert last_layer_trainable > 0, "Last layer should be trainable"


def test_trainable_params_breakdown(adapter_8_layers):
    """trainable params breakdown 계산 검증"""
    adapter = adapter_8_layers
    num_unfrozen = 4
    apply_freeze_logic(adapter, num_unfrozen=num_unfrozen)

    # breakdown 계산 (run_critic.py 로직 재현)
    trainable_breakdown = {
        "value_head": sum(
            p.numel() for p in adapter.value_head.parameters()
            if p.requires_grad
        ),
        "trunk_blocks": sum(
            p.numel() for layer in adapter.transformer.layers[-num_unfrozen:]
            for p in layer.parameters() if p.requires_grad
        ),
        "norm": sum(
            p.numel() for p in adapter.transformer.norm.parameters()
            if p.requires_grad
        ),
    }

    # 각 컴포넌트가 0보다 큼
    assert trainable_breakdown["value_head"] > 0
    assert trainable_breakdown["trunk_blocks"] > 0
    assert trainable_breakdown["norm"] > 0

    # 합계 검증
    total_trainable = count_trainable_params(adapter)
    breakdown_sum = sum(trainable_breakdown.values())
    assert breakdown_sum == total_trainable, (
        f"Breakdown sum ({breakdown_sum}) != total trainable ({total_trainable})"
    )


def test_backward_pass_with_unfrozen_blocks(adapter_8_layers):
    """unfreeze된 블록들의 backward pass 정상 동작 확인"""
    adapter = adapter_8_layers
    apply_freeze_logic(adapter, num_unfrozen=4)

    # Forward pass
    input_ids = torch.randint(0, 1000, (2, 16))  # [batch=2, seq=16]
    outputs = adapter(input_ids, return_value_logits=True)
    value_logits = outputs["value_logits"]

    # Backward pass
    loss = value_logits.mean()
    loss.backward()

    # 적어도 일부 unfreeze된 레이어에 gradient가 있어야 함
    unfrozen_layers_with_grad = 0
    for i in range(4, 8):
        layer = adapter.transformer.layers[i]
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.parameters() if p.requires_grad
        )
        if has_grad:
            unfrozen_layers_with_grad += 1

    assert unfrozen_layers_with_grad > 0, "At least one unfrozen layer should have gradients"

    # 앞쪽 4개 블록은 gradient가 없어야 함
    for i in range(4):
        layer = adapter.transformer.layers[i]
        has_grad = any(
            p.grad is not None
            for p in layer.parameters()
        )
        assert not has_grad, f"Layer {i} should not have gradients"

    # Value head에 gradient가 있어야 함
    value_head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in adapter.value_head.parameters() if p.requires_grad
    )
    assert value_head_has_grad, "Value head should have gradients"


def test_value_head_always_trainable(adapter_8_layers):
    """어떤 설정에서도 value head는 항상 trainable"""
    adapter = adapter_8_layers

    for num_unfrozen in [0, 1, 4, 8, 100]:
        apply_freeze_logic(adapter, num_unfrozen=num_unfrozen)
        value_head_trainable = count_trainable_params(adapter.value_head)
        assert value_head_trainable > 0, (
            f"Value head should be trainable with num_unfrozen={num_unfrozen}"
        )
