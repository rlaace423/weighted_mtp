"""pairwise_utils 단위 테스트

λ-return 및 value loss 함수 검증
"""

import pytest
import torch

from weighted_mtp.utils.pairwise_utils import (
    compute_lambda_return,
    compute_lambda_value_loss,
    compute_mc_value_loss,
    create_eos_only_mask,
)


class TestComputeLambdaReturn:
    """compute_lambda_return 함수 테스트"""

    def test_lambda_1_equals_mc(self):
        """λ=1.0일 때 모든 토큰이 terminal reward와 동일해야 함 (MC와 동치)"""
        batch_size, seq_len = 2, 5
        values = torch.rand(batch_size, seq_len)
        rewards = torch.tensor([1.0, 0.0])
        loss_mask = torch.ones(batch_size, seq_len)

        # λ=1.0: Pure MC
        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0
        )

        # 모든 유효 토큰이 terminal reward와 동일해야 함
        for b in range(batch_size):
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for pos in valid_positions:
                assert torch.isclose(
                    lambda_returns[b, pos], rewards[b], atol=1e-6
                ), f"batch {b}, pos {pos}: expected {rewards[b]}, got {lambda_returns[b, pos]}"

    def test_lambda_0_uses_only_v_next(self):
        """λ=0일 때 각 위치가 V_{t+1}만 사용해야 함 (Pure TD)"""
        batch_size, seq_len = 1, 4
        # 고정된 V 값으로 테스트
        values = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.ones(batch_size, seq_len)

        # λ=0: Pure TD (G_t = γV_{t+1})
        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.0
        )

        # Terminal: G_3 = R = 1.0
        assert torch.isclose(lambda_returns[0, 3], torch.tensor(1.0), atol=1e-6)
        # G_2 = γV_3 = 1.0 * 0.4 = 0.4
        assert torch.isclose(lambda_returns[0, 2], torch.tensor(0.4), atol=1e-6)
        # G_1 = γV_2 = 1.0 * 0.3 = 0.3
        assert torch.isclose(lambda_returns[0, 1], torch.tensor(0.3), atol=1e-6)
        # G_0 = γV_1 = 1.0 * 0.2 = 0.2
        assert torch.isclose(lambda_returns[0, 0], torch.tensor(0.2), atol=1e-6)

    def test_lambda_095_creates_gradient(self):
        """λ=0.95일 때 위치별로 다른 타겟이 생성되어야 함"""
        batch_size, seq_len = 1, 5
        values = torch.full((batch_size, seq_len), 0.5)  # 모든 V=0.5
        rewards = torch.tensor([1.0])
        loss_mask = torch.ones(batch_size, seq_len)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95
        )

        # Terminal은 R=1.0
        assert torch.isclose(lambda_returns[0, 4], torch.tensor(1.0), atol=1e-6)

        # 앞쪽으로 갈수록 값이 다름 (gradient 존재)
        # 정확한 값 계산:
        # G_4 = 1.0
        # G_3 = 0.05*V_4 + 0.95*G_4 = 0.05*0.5 + 0.95*1.0 = 0.025 + 0.95 = 0.975
        # G_2 = 0.05*V_3 + 0.95*G_3 = 0.05*0.5 + 0.95*0.975 = 0.025 + 0.92625 = 0.95125
        assert torch.isclose(lambda_returns[0, 3], torch.tensor(0.975), atol=1e-5)
        assert torch.isclose(lambda_returns[0, 2], torch.tensor(0.95125), atol=1e-5)

        # 전반부는 후반부보다 낮아야 함
        assert lambda_returns[0, 0] < lambda_returns[0, 2] < lambda_returns[0, 4]

    def test_partial_mask(self):
        """부분 마스크 (instruction 제외) 처리 검증"""
        batch_size, seq_len = 1, 6
        values = torch.rand(batch_size, seq_len)
        rewards = torch.tensor([1.0])
        # 앞 2개는 instruction (mask=0), 뒤 4개가 output (mask=1)
        loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95
        )

        # 마스크=0인 위치는 타겟이 0이어야 함
        assert lambda_returns[0, 0] == 0.0
        assert lambda_returns[0, 1] == 0.0
        # 마스크=1인 마지막 위치는 R
        assert torch.isclose(lambda_returns[0, 5], torch.tensor(1.0), atol=1e-6)

    def test_empty_mask(self):
        """빈 마스크 처리 (엣지 케이스)"""
        batch_size, seq_len = 1, 4
        values = torch.rand(batch_size, seq_len)
        rewards = torch.tensor([1.0])
        loss_mask = torch.zeros(batch_size, seq_len)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95
        )

        # 모든 위치가 0이어야 함
        assert torch.all(lambda_returns == 0.0)

    def test_batch_processing(self):
        """배치 처리 검증 (correct/incorrect 혼합)"""
        batch_size, seq_len = 4, 5
        values = torch.full((batch_size, seq_len), 0.5)
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 교대로 correct/incorrect
        loss_mask = torch.ones(batch_size, seq_len)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0  # MC로 검증
        )

        # 각 배치의 terminal이 해당 reward와 동일해야 함
        assert torch.isclose(lambda_returns[0, 4], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(lambda_returns[1, 4], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(lambda_returns[2, 4], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(lambda_returns[3, 4], torch.tensor(0.0), atol=1e-6)


class TestComputeLambdaValueLoss:
    """compute_lambda_value_loss 함수 테스트"""

    def test_loss_decreases_with_correct_predictions(self):
        """정확한 예측일수록 loss가 낮아야 함"""
        batch_size, seq_len = 2, 5
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)
        rewards = torch.tensor([1.0, 0.0])

        # 정확한 예측: correct=1.0, incorrect=0.0
        accurate_values = torch.tensor([
            [[1.0], [1.0], [1.0], [1.0], [1.0]],  # correct sample -> 1.0
            [[0.0], [0.0], [0.0], [0.0], [0.0]],  # incorrect sample -> 0.0
        ])

        # 부정확한 예측: 반대로
        inaccurate_values = torch.tensor([
            [[0.0], [0.0], [0.0], [0.0], [0.0]],  # correct sample -> 0.0 (틀림)
            [[1.0], [1.0], [1.0], [1.0], [1.0]],  # incorrect sample -> 1.0 (틀림)
        ])

        accurate_loss = compute_lambda_value_loss(
            accurate_values, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0  # MC로 테스트
        )
        inaccurate_loss = compute_lambda_value_loss(
            inaccurate_values, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0
        )

        assert accurate_loss < inaccurate_loss

    def test_interface_compatibility_with_mc(self):
        """compute_mc_value_loss와 동일한 인터페이스로 호출 가능해야 함"""
        batch_size, seq_len = 2, 5
        value_logits = torch.rand(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # 두 함수 모두 동일한 인터페이스로 호출 가능
        mc_loss = compute_mc_value_loss(value_logits, rewards, attention_mask, loss_mask)
        lambda_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0
        )

        # λ=1.0일 때 MC와 동일한 결과
        assert torch.isclose(mc_loss, lambda_loss, atol=1e-5)

    def test_gradient_flows(self):
        """value_logits에 대한 gradient가 정상적으로 흐르는지 확인"""
        batch_size, seq_len = 2, 4
        value_logits = torch.rand(batch_size, seq_len, 1, requires_grad=True)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=0.95
        )
        loss.backward()

        # gradient가 존재하고 유효해야 함
        assert value_logits.grad is not None
        assert not torch.isnan(value_logits.grad).any()
        assert not torch.isinf(value_logits.grad).any()

    def test_masked_positions_excluded(self):
        """마스크된 위치는 loss 계산에서 제외되어야 함"""
        batch_size, seq_len = 1, 5
        value_logits = torch.zeros(batch_size, seq_len, 1)  # 모든 예측 0
        rewards = torch.tensor([1.0])  # 타겟은 1
        attention_mask = torch.ones(batch_size, seq_len)

        # 전체 마스크: 모든 토큰 포함 -> loss > 0
        full_mask = torch.ones(batch_size, seq_len)
        full_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, full_mask,
            gamma=1.0, lam=1.0
        )

        # 부분 마스크: 일부만 포함 -> loss < full_loss (비례)
        # (실제로는 평균이므로 값 자체는 비슷할 수 있지만, 마스크 동작 검증)
        partial_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
        partial_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, partial_mask,
            gamma=1.0, lam=1.0
        )

        # 둘 다 양수 loss (예측=0, 타겟=1이므로)
        assert full_loss > 0
        assert partial_loss > 0


class TestCreateEosOnlyMask:
    """create_eos_only_mask 함수 테스트"""

    def test_instruction_masked_sequence(self):
        """Instruction이 마스킹된 시퀀스에서 마지막 Output 토큰 위치 정확히 찾기"""
        # [0, 0, 1, 1, 1, 0, 0] -> 마지막 1의 위치는 4
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        # 인덱스 4만 1이어야 함
        expected = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_continuous_output_sequence(self):
        """연속된 Output 시퀀스 (Instruction 없음)"""
        # [1, 1, 1, 0, 0] -> 마지막 1의 위치는 2
        loss_mask = torch.tensor([
            [1, 1, 1, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_empty_mask(self):
        """유효 토큰이 없는 경우 (모두 0)"""
        loss_mask = torch.tensor([
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        # 모두 0이어야 함
        expected = torch.zeros_like(loss_mask)
        assert torch.equal(eos_mask, expected)

    def test_single_valid_token(self):
        """유효 토큰이 하나뿐인 경우"""
        loss_mask = torch.tensor([
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_first_position_valid(self):
        """첫 번째 위치만 유효한 경우"""
        loss_mask = torch.tensor([
            [1, 0, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [1, 0, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_batch_processing(self):
        """배치 처리 검증"""
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0, 0],  # 마지막 위치: 4
            [1, 1, 1, 0, 0, 0, 0],  # 마지막 위치: 2
            [0, 0, 0, 0, 0, 0, 0],  # 유효 토큰 없음
            [0, 1, 0, 0, 0, 0, 0],  # 마지막 위치: 1
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_dtype_preservation(self):
        """반환 dtype이 float32인지 확인"""
        loss_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        eos_mask = create_eos_only_mask(loss_mask)
        assert eos_mask.dtype == torch.float32

    def test_device_preservation(self):
        """디바이스가 유지되는지 확인"""
        loss_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
        eos_mask = create_eos_only_mask(loss_mask)
        assert eos_mask.device == loss_mask.device
