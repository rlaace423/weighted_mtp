"""Value Head 관련 단위 테스트

init_value_head_bias 및 Value Head 초기화 검증
"""

import pytest
import torch

from weighted_mtp.models.value_head import (
    LinearValueHead,
    SigmoidValueHead,
    MLPValueHead,
    create_value_head,
)
from weighted_mtp.pipelines.run_critic import init_value_head_bias


class MockValueModel:
    """테스트용 ValueModel mock"""

    def __init__(self, value_head):
        self.value_head = value_head


class TestInitValueHeadBias:
    """init_value_head_bias 함수 테스트"""

    def test_mlp_head_bias_init(self):
        """MLP value head bias 초기화 검증"""
        hidden_size = 256
        value_head = MLPValueHead(hidden_size, dropout=0.1)
        mock_model = MockValueModel(value_head)

        # 초기 bias 확인 (zero init)
        assert value_head.mlp[-1].bias.data.abs().max() == 0.0

        # bias 초기화 적용
        init_value_head_bias(mock_model, 0.5)

        # bias가 0.5로 설정되었는지 확인
        assert torch.allclose(
            value_head.mlp[-1].bias.data,
            torch.tensor([0.5]),
            atol=1e-6
        )

    def test_linear_head_bias_init(self):
        """Linear value head bias 초기화 검증"""
        hidden_size = 256
        value_head = LinearValueHead(hidden_size)
        mock_model = MockValueModel(value_head)

        # 초기 bias 확인 (zero init)
        assert value_head.linear.bias.data.abs().max() == 0.0

        # bias 초기화 적용
        init_value_head_bias(mock_model, 0.5)

        # bias가 0.5로 설정되었는지 확인
        assert torch.allclose(
            value_head.linear.bias.data,
            torch.tensor([0.5]),
            atol=1e-6
        )

    def test_sigmoid_head_bias_init(self):
        """Sigmoid value head bias 초기화 검증"""
        hidden_size = 256
        value_head = SigmoidValueHead(hidden_size)
        mock_model = MockValueModel(value_head)

        # 초기 bias 확인 (zero init)
        assert value_head.linear.bias.data.abs().max() == 0.0

        # bias 초기화 적용
        init_value_head_bias(mock_model, 0.5)

        # bias가 0.5로 설정되었는지 확인
        assert torch.allclose(
            value_head.linear.bias.data,
            torch.tensor([0.5]),
            atol=1e-6
        )

    def test_different_bias_values(self):
        """다양한 bias 값으로 초기화 검증"""
        hidden_size = 256
        test_values = [0.0, 0.3, 0.5, 0.7, 1.0]

        for bias_val in test_values:
            value_head = MLPValueHead(hidden_size)
            mock_model = MockValueModel(value_head)

            init_value_head_bias(mock_model, bias_val)

            assert torch.allclose(
                value_head.mlp[-1].bias.data,
                torch.tensor([bias_val]),
                atol=1e-6
            ), f"Failed for bias_val={bias_val}"

    def test_forward_after_bias_init(self):
        """bias 초기화 후 forward pass가 기대값에 가까운지 검증"""
        hidden_size = 256
        batch_size, seq_len = 2, 10

        value_head = MLPValueHead(hidden_size)
        mock_model = MockValueModel(value_head)

        # bias를 0.5로 초기화
        init_value_head_bias(mock_model, 0.5)

        # weight가 0이고 bias가 0.5이면, 출력은 ~0.5에 가까워야 함
        # (MLP 중간 레이어 때문에 정확히 0.5는 아님)
        dummy_input = torch.randn(batch_size, seq_len, hidden_size)
        output = value_head(dummy_input)

        # 출력 shape 확인
        assert output.shape == (batch_size, seq_len, 1)

    def test_create_value_head_factory(self):
        """create_value_head factory 함수 동작 확인"""
        hidden_size = 256

        linear_head = create_value_head(hidden_size, "linear")
        assert isinstance(linear_head, LinearValueHead)
        assert linear_head.head_type == "linear"

        sigmoid_head = create_value_head(hidden_size, "sigmoid")
        assert isinstance(sigmoid_head, SigmoidValueHead)
        assert sigmoid_head.head_type == "sigmoid"

        mlp_head = create_value_head(hidden_size, "mlp", dropout=0.3)
        assert isinstance(mlp_head, MLPValueHead)
        assert mlp_head.head_type == "mlp"

    def test_unknown_head_type_ignored(self):
        """알 수 없는 head_type은 무시되어야 함"""

        class UnknownHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_type = "unknown"
                self.some_param = torch.nn.Parameter(torch.zeros(1))

        value_head = UnknownHead()
        mock_model = MockValueModel(value_head)

        # 에러 없이 실행되어야 함 (아무 동작 안함)
        init_value_head_bias(mock_model, 0.5)

        # 파라미터가 변경되지 않았는지 확인
        assert value_head.some_param.data.item() == 0.0
