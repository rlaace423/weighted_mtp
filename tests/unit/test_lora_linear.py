"""LoRA Linear Unit Tests"""

import pytest
import torch
from torch import nn

from weighted_mtp.models.lora import (
    LoRALinear,
    apply_lora_to_linear,
    get_lora_parameters,
    merge_lora_weights,
)


class TestLoRALinear:
    """LoRALinear 클래스 테스트"""

    def test_init_shape(self):
        """LoRALinear 초기화 후 shape 검증"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8)

        assert lora.linear.weight.shape == (256, 512)
        assert lora.lora_A.shape == (8, 512)
        assert lora.lora_B.shape == (256, 8)

    def test_init_requires_grad(self):
        """원본 가중치는 frozen, LoRA는 학습 가능"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8)

        assert lora.linear.weight.requires_grad is False
        assert lora.lora_A.requires_grad is True
        assert lora.lora_B.requires_grad is True

    def test_init_with_bias(self):
        """bias=True인 경우 bias도 frozen"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8, bias=True)

        assert lora.linear.bias is not None
        assert lora.linear.bias.requires_grad is False

    def test_forward_shape(self):
        """forward 출력 shape 검증"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8)
        x = torch.randn(2, 10, 512)  # [batch, seq, in_features]

        output = lora(x)

        assert output.shape == (2, 10, 256)

    def test_forward_initial_output_matches_linear(self):
        """LoRA 초기화 직후 출력은 원본 Linear와 동일 (B가 0이므로)"""
        linear = nn.Linear(512, 256, bias=False)
        lora = LoRALinear.from_linear(linear, rank=8)

        x = torch.randn(2, 10, 512)

        linear_output = linear(x)
        lora_output = lora(x)

        assert torch.allclose(linear_output, lora_output, atol=1e-6)

    def test_forward_lora_contribution(self):
        """LoRA 학습 후 출력이 달라지는지 검증"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8, alpha=16.0)
        x = torch.randn(2, 10, 512)

        # 초기 출력
        initial_output = lora(x).clone()

        # LoRA 가중치 변경 (학습 시뮬레이션)
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)

        # 변경 후 출력
        modified_output = lora(x)

        # 출력이 달라져야 함
        assert not torch.allclose(initial_output, modified_output)

    def test_from_linear(self):
        """from_linear 클래스 메서드 테스트"""
        original = nn.Linear(512, 256, bias=False)
        original.weight.data.fill_(0.5)

        lora = LoRALinear.from_linear(original, rank=8, alpha=16.0)

        # 원본 가중치가 복사됨
        assert torch.allclose(lora.linear.weight, original.weight)
        # 원본은 frozen
        assert lora.linear.weight.requires_grad is False

    def test_from_linear_with_bias(self):
        """from_linear: bias 있는 경우"""
        original = nn.Linear(512, 256, bias=True)
        original.weight.data.fill_(0.5)
        original.bias.data.fill_(0.1)

        lora = LoRALinear.from_linear(original, rank=8)

        assert lora.linear.bias is not None
        assert torch.allclose(lora.linear.bias, original.bias)
        assert lora.linear.bias.requires_grad is False

    def test_from_linear_device_dtype(self):
        """from_linear: device/dtype 일치"""
        original = nn.Linear(512, 256, bias=False)
        original = original.to(dtype=torch.float16)

        lora = LoRALinear.from_linear(original, rank=8)

        assert lora.linear.weight.dtype == torch.float16
        assert lora.lora_A.dtype == torch.float16
        assert lora.lora_B.dtype == torch.float16

    def test_merge_weights(self):
        """merge_weights 후 출력 동일 검증"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8, alpha=16.0)

        # LoRA 가중치 설정
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)

        x = torch.randn(2, 10, 512)
        output_before_merge = lora(x).clone()

        # 가중치 병합
        lora.merge_weights()

        # 병합 후 LoRA 가중치는 0
        assert torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

        # 출력은 동일해야 함
        output_after_merge = lora(x)
        assert torch.allclose(output_before_merge, output_after_merge, atol=1e-5)

    def test_scaling_factor(self):
        """alpha/rank scaling 검증"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8, alpha=32.0)

        assert lora.scaling == 32.0 / 8  # alpha / rank = 4.0


class TestApplyLoRAToLinear:
    """apply_lora_to_linear 함수 테스트"""

    def test_apply_to_attention(self):
        """Attention 모듈에 LoRA 적용"""
        # 간단한 Attention mock
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(512, 512, bias=False)
                self.wk = nn.Linear(512, 512, bias=False)
                self.wv = nn.Linear(512, 512, bias=False)
                self.wo = nn.Linear(512, 512, bias=False)

        attention = MockAttention()

        # LoRA 적용
        apply_lora_to_linear(attention, ["wq", "wk", "wv", "wo"], rank=8)

        # 모든 타겟이 LoRALinear로 교체됨
        assert isinstance(attention.wq, LoRALinear)
        assert isinstance(attention.wk, LoRALinear)
        assert isinstance(attention.wv, LoRALinear)
        assert isinstance(attention.wo, LoRALinear)

    def test_apply_partial_targets(self):
        """일부 타겟만 적용"""
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(512, 512, bias=False)
                self.wk = nn.Linear(512, 512, bias=False)
                self.wv = nn.Linear(512, 512, bias=False)
                self.wo = nn.Linear(512, 512, bias=False)

        attention = MockAttention()

        # wq, wv만 적용
        apply_lora_to_linear(attention, ["wq", "wv"], rank=8)

        assert isinstance(attention.wq, LoRALinear)
        assert isinstance(attention.wk, nn.Linear)  # 그대로
        assert isinstance(attention.wv, LoRALinear)
        assert isinstance(attention.wo, nn.Linear)  # 그대로

    def test_apply_nonexistent_target(self):
        """존재하지 않는 타겟은 무시"""
        class MockModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(512, 512, bias=False)

        module = MockModule()

        # 존재하지 않는 타겟 포함 - 에러 없이 무시
        apply_lora_to_linear(module, ["wq", "nonexistent"], rank=8)

        assert isinstance(module.wq, LoRALinear)


class TestGetLoRAParameters:
    """get_lora_parameters 함수 테스트"""

    def test_get_lora_params(self):
        """LoRA 파라미터만 추출"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 256)
                self.lora = LoRALinear(512, 256, rank=8)

        model = MockModel()
        lora_params = get_lora_parameters(model)

        # lora_A, lora_B만 포함 (2개)
        assert len(lora_params) == 2

        param_names = {p.shape for p in lora_params}
        assert (8, 512) in param_names  # lora_A
        assert (256, 8) in param_names  # lora_B

    def test_get_lora_params_nested(self):
        """중첩 모듈에서 LoRA 파라미터 추출"""
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        "attention": nn.ModuleDict({
                            "wq": LoRALinear(512, 512, rank=8),
                            "wk": LoRALinear(512, 512, rank=8),
                        })
                    })
                    for _ in range(2)
                ])

        model = MockTransformer()
        lora_params = get_lora_parameters(model)

        # 2 layers * 2 modules * 2 params (A, B) = 8
        assert len(lora_params) == 8


class TestMergeLoRAWeights:
    """merge_lora_weights 함수 테스트"""

    def test_merge_all_lora(self):
        """모델 내 모든 LoRALinear 병합"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora1 = LoRALinear(512, 256, rank=8)
                self.lora2 = LoRALinear(256, 128, rank=8)

        model = MockModel()

        # LoRA 가중치 설정
        with torch.no_grad():
            model.lora1.lora_A.fill_(0.1)
            model.lora1.lora_B.fill_(0.1)
            model.lora2.lora_A.fill_(0.1)
            model.lora2.lora_B.fill_(0.1)

        # 전체 병합
        merge_lora_weights(model)

        # 모든 LoRA가 병합됨 (A, B가 0)
        assert torch.allclose(model.lora1.lora_A, torch.zeros_like(model.lora1.lora_A))
        assert torch.allclose(model.lora1.lora_B, torch.zeros_like(model.lora1.lora_B))
        assert torch.allclose(model.lora2.lora_A, torch.zeros_like(model.lora2.lora_A))
        assert torch.allclose(model.lora2.lora_B, torch.zeros_like(model.lora2.lora_B))


class TestLoRAGradient:
    """LoRA Gradient 흐름 테스트"""

    def test_gradient_only_lora(self):
        """Gradient가 LoRA 파라미터에만 흐르는지 검증"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8)
        x = torch.randn(2, 10, 512, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # LoRA 파라미터에는 gradient가 있음
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

        # 원본 가중치에는 gradient가 없음 (requires_grad=False)
        assert lora.linear.weight.grad is None


class TestFSDPCompatibility:
    """FSDP 호환성 기본 테스트"""

    def test_state_dict_keys(self):
        """state_dict 키 구조 검증 (FSDP wrapping 가정)"""
        lora = LoRALinear(in_features=512, out_features=256, rank=8, bias=True)
        state_dict = lora.state_dict()

        expected_keys = {
            "linear.weight",
            "linear.bias",
            "lora_A",
            "lora_B",
        }
        assert set(state_dict.keys()) == expected_keys

    def test_load_state_dict(self):
        """state_dict 저장/로드"""
        lora1 = LoRALinear(in_features=512, out_features=256, rank=8)

        # 가중치 변경
        with torch.no_grad():
            lora1.lora_A.fill_(0.5)
            lora1.lora_B.fill_(0.3)

        state_dict = lora1.state_dict()

        # 새 인스턴스에 로드
        lora2 = LoRALinear(in_features=512, out_features=256, rank=8)
        lora2.load_state_dict(state_dict)

        assert torch.allclose(lora2.lora_A, lora1.lora_A)
        assert torch.allclose(lora2.lora_B, lora1.lora_B)

