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


def test_forward_value_logits_shape(micro_adapter):
    """forward(return_value_logits=True) 출력 shape 테스트"""
    input_ids = torch.randint(0, 32000, (2, 10))  # [batch=2, seq=10]

    outputs = micro_adapter(input_ids, return_value_logits=True)

    assert "hidden_states" in outputs
    assert "value_logits" in outputs
    assert outputs["hidden_states"].shape == (2, 10, 512)
    assert outputs["value_logits"].shape == (2, 10, 1)


def test_forward_full_shape(micro_adapter):
    """forward(return_value_logits=True, return_hidden_states=True) 출력 shape 테스트"""
    input_ids = torch.randint(0, 32000, (2, 10))

    outputs = micro_adapter(input_ids, return_value_logits=True, return_hidden_states=True)

    assert "logits" in outputs
    assert "value_logits" in outputs
    assert "hidden_states" in outputs
    assert outputs["logits"].shape == (2, 10, 4, 32000)  # [batch, seq, n_future_tokens, vocab]
    assert outputs["value_logits"].shape == (2, 10, 1)
    assert outputs["hidden_states"].shape == (2, 10, 512)


def test_forward_without_value_head():
    """Value head 없이 return_value_logits=True 호출 시 에러"""
    model_args = ModelArgs(dim=512, n_layers=4, n_heads=8, vocab_size=32000)
    transformer = Transformer(model_args)
    adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head=None)

    input_ids = torch.randint(0, 32000, (2, 10))

    with pytest.raises(ValueError, match="Value head not initialized"):
        adapter(input_ids, return_value_logits=True)


def test_attach_value_head():
    """Value head 추가 테스트"""
    model_args = ModelArgs(dim=512, n_layers=4, n_heads=8, vocab_size=32000)
    transformer = Transformer(model_args)
    adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head=None)

    # Value head 추가
    value_head = ValueHead(hidden_size=512)
    adapter.attach_value_head(value_head)

    # 이제 forward(return_value_logits=True) 가능
    input_ids = torch.randint(0, 32000, (2, 10))
    outputs = adapter(input_ids, return_value_logits=True)

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


class TestLoRAIntegration:
    """LoRA 통합 테스트"""

    @pytest.fixture
    def micro_adapter_with_lora(self):
        """LoRA가 적용된 Micro Adapter"""
        model_args = ModelArgs(
            dim=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=32000,
            n_future_tokens=4,
        )
        transformer = Transformer(model_args)
        value_head = ValueHead(hidden_size=512)
        adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head)

        # LoRA 적용
        adapter.apply_lora({"rank": 8, "alpha": 16.0})
        return adapter

    def test_lora_enabled_flag(self, micro_adapter_with_lora):
        """LoRA 적용 후 lora_enabled 플래그 검증"""
        assert micro_adapter_with_lora.lora_enabled is True

    def test_lora_layers_applied(self, micro_adapter_with_lora):
        """LoRA 레이어가 적용되었는지 검증"""
        from weighted_mtp.models.lora import LoRALinear

        adapter = micro_adapter_with_lora
        # trunk layers에 LoRA 적용됨
        for layer in adapter.transformer.layers:
            assert isinstance(layer.attention.wq, LoRALinear)
            assert isinstance(layer.attention.wk, LoRALinear)
            assert isinstance(layer.attention.wv, LoRALinear)
            assert isinstance(layer.attention.wo, LoRALinear)

    def test_lora_trainable_params_reduced(self, micro_adapter_with_lora):
        """LoRA 적용 후 학습 파라미터 수 감소 검증"""
        adapter = micro_adapter_with_lora

        # 전체 파라미터
        total_params = sum(p.numel() for p in adapter.parameters())
        # 학습 가능 파라미터 (LoRA + extra_heads + value_head)
        trainable_params = sum(p.numel() for p in adapter.get_trainable_parameters())

        # trunk layers에서 학습 파라미터 (LoRA만)
        trunk_trainable = 0
        for layer in adapter.transformer.layers:
            for p in layer.parameters():
                if p.requires_grad:
                    trunk_trainable += p.numel()

        # trunk의 전체 파라미터 (LoRA 포함)
        trunk_total = sum(p.numel() for layer in adapter.transformer.layers for p in layer.parameters())

        # trunk에서 LoRA 효과: 학습 파라미터가 전체의 10% 미만
        assert trunk_trainable < trunk_total * 0.1

        # 전체 모델에서도 학습 파라미터 감소 (extra_heads, value_head 제외하면)
        assert trainable_params < total_params  # 최소 감소 확인

    def test_lora_forward_shape_unchanged(self, micro_adapter_with_lora):
        """LoRA 적용 후 forward 출력 shape 동일"""
        adapter = micro_adapter_with_lora
        input_ids = torch.randint(0, 32000, (2, 10))

        # Value logits 모드
        outputs = adapter(input_ids, return_value_logits=True)
        assert outputs["value_logits"].shape == (2, 10, 1)

        # Full 모드
        outputs = adapter(input_ids, return_value_logits=True, return_hidden_states=True)
        assert outputs["logits"].shape == (2, 10, 4, 32000)
        assert outputs["value_logits"].shape == (2, 10, 1)

    def test_lora_gradient_flow(self, micro_adapter_with_lora):
        """LoRA 파라미터에만 gradient 흐름"""
        adapter = micro_adapter_with_lora
        input_ids = torch.randint(0, 32000, (2, 10))

        outputs = adapter(input_ids, return_value_logits=True, return_hidden_states=True)
        loss = outputs["logits"].sum() + outputs["value_logits"].sum()
        loss.backward()

        # LoRA 파라미터에 gradient 있음
        for layer in adapter.transformer.layers:
            assert layer.attention.wq.lora_A.grad is not None
            assert layer.attention.wq.lora_B.grad is not None

        # 원본 가중치에는 gradient 없음 (frozen)
        for layer in adapter.transformer.layers:
            assert layer.attention.wq.linear.weight.grad is None

    def test_merge_lora(self, micro_adapter_with_lora):
        """LoRA 병합 테스트"""
        adapter = micro_adapter_with_lora
        input_ids = torch.randint(0, 32000, (2, 10))

        # 병합 전 출력
        output_before = adapter(input_ids).clone()

        # LoRA 병합
        adapter.merge_lora()

        # 병합 후 출력
        output_after = adapter(input_ids)

        # 출력이 동일해야 함
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_get_trainable_parameters_without_lora(self, micro_adapter):
        """LoRA 없이 get_trainable_parameters 호출"""
        trainable = micro_adapter.get_trainable_parameters()

        # 모든 파라미터가 학습 가능
        total_params = sum(p.numel() for p in micro_adapter.parameters() if p.requires_grad)
        trainable_numel = sum(p.numel() for p in trainable)

        assert trainable_numel == total_params

    def test_lora_with_value_head(self, micro_adapter_with_lora):
        """LoRA + Value head 동시 학습"""
        adapter = micro_adapter_with_lora
        trainable = adapter.get_trainable_parameters()

        # Value head 파라미터도 포함됨
        value_head_param_count = sum(
            p.numel() for p in adapter.value_head.parameters() if p.requires_grad
        )
        assert value_head_param_count > 0

    def test_lora_config_custom(self):
        """커스텀 LoRA config 테스트"""
        model_args = ModelArgs(
            dim=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=32000,
            n_future_tokens=4,
        )
        transformer = Transformer(model_args)
        adapter = MetaLlamaMTPAdapter(transformer, model_args, value_head=None)

        # 커스텀 설정으로 LoRA 적용
        custom_config = {
            "rank": 16,
            "alpha": 32.0,
            "dropout": 0.1,
            "target_modules": ["wq", "wv"],  # wk, wo 제외
        }
        adapter.apply_lora(custom_config)

        from weighted_mtp.models.lora import LoRALinear
        from torch import nn

        # wq, wv만 LoRA 적용
        for layer in adapter.transformer.layers:
            assert isinstance(layer.attention.wq, LoRALinear)
            assert isinstance(layer.attention.wv, LoRALinear)
            assert isinstance(layer.attention.wk, nn.Linear)  # 원본 유지
            assert isinstance(layer.attention.wo, nn.Linear)  # 원본 유지

            # rank 검증
            assert layer.attention.wq.rank == 16
            assert layer.attention.wq.alpha == 32.0
