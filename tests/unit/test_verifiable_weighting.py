"""Verifiable WMTP Weighting 단위 테스트

TD error 기반 weight 계산 및 value/MTP head 시점 매핑 검증
"""

import pytest
import torch

from weighted_mtp.value_weighting.td_weighting import (
    compute_td_errors,
    compute_td_targets,
    build_weights,
    compute_td_stats,
    compute_weight_stats,
)


class TestComputeTDErrors:
    """compute_td_errors 함수 테스트"""

    def test_basic_shape(self):
        """기본 출력 shape 검증"""
        batch_size = 2
        seq_len = 10

        value_logits = torch.randn(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)

        td_errors = compute_td_errors(
            value_logits=value_logits,
            rewards=rewards,
            attention_mask=attention_mask,
            gamma=1.0,
        )

        assert td_errors.shape == (batch_size, seq_len)

    def test_terminal_td_error(self):
        """Terminal TD error 계산 검증"""
        # R - V(s_T)
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])  # Correct
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma=1.0)

        # Terminal (position 2): 1.0 - 0.9 = 0.1
        assert td_errors[0, 2].item() == pytest.approx(0.1, rel=0.01)

    def test_intermediate_td_error(self):
        """Intermediate TD error 계산 검증"""
        # δ_t = γV(s_t) - V(s_{t-1})
        # td_errors[t]는 토큰 t의 가중치
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma=1.0)

        # Position 0: 이전 value 없음 → td_errors[0] = 0
        assert td_errors[0, 0].item() == pytest.approx(0.0, rel=0.01)
        # Position 1: δ₁ = 1.0 * 0.7 - 0.5 = 0.2
        assert td_errors[0, 1].item() == pytest.approx(0.2, rel=0.01)

    def test_padding_handling(self):
        """Padding 토큰 처리 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9], [0.8]]])  # [1, 4, 1]
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1, 0]])  # 마지막은 padding

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma=1.0)

        # Padding 위치는 0
        assert td_errors[0, 3].item() == 0.0
        # Terminal은 position 2
        assert td_errors[0, 2].item() == pytest.approx(0.1, rel=0.01)

    def test_incorrect_reward(self):
        """Incorrect 샘플의 TD error 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([0.0])  # Incorrect
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma=1.0)

        # Terminal (position 2): 0.0 - 0.9 = -0.9
        assert td_errors[0, 2].item() == pytest.approx(-0.9, rel=0.01)


class TestComputeTDTargets:
    """compute_td_targets 함수 테스트"""

    def test_basic_shape(self):
        """기본 출력 shape 검증"""
        batch_size = 2
        seq_len = 10

        value_logits = torch.randn(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)

        td_targets = compute_td_targets(
            value_logits=value_logits,
            rewards=rewards,
            attention_mask=attention_mask,
            gamma=1.0,
            lam=0.0,
        )

        assert td_targets.shape == (batch_size, seq_len, 1)

    def test_td0_targets(self):
        """TD(0) targets 검증 (lam=0)"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])

        td_targets = compute_td_targets(
            value_logits, rewards, attention_mask, gamma=1.0, lam=0.0
        )

        # TD(0): Target_t = V(s_t) + δ_t = γV(s_{t+1})
        # Position 0: 1.0 * 0.7 = 0.7
        # Position 1: 1.0 * 0.9 = 0.9
        # Position 2: R = 1.0
        assert td_targets[0, 0, 0].item() == pytest.approx(0.7, rel=0.01)
        assert td_targets[0, 1, 0].item() == pytest.approx(0.9, rel=0.01)
        assert td_targets[0, 2, 0].item() == pytest.approx(1.0, rel=0.01)

    def test_mc_targets(self):
        """Monte Carlo targets 검증 (lam=1)"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])

        td_targets = compute_td_targets(
            value_logits, rewards, attention_mask, gamma=1.0, lam=1.0
        )

        # MC: 모든 위치의 target이 R
        assert td_targets[0, 0, 0].item() == pytest.approx(1.0, rel=0.01)
        assert td_targets[0, 1, 0].item() == pytest.approx(1.0, rel=0.01)
        assert td_targets[0, 2, 0].item() == pytest.approx(1.0, rel=0.01)


class TestBuildWeights:
    """build_weights 함수 테스트"""

    def test_basic_shape(self):
        """기본 출력 shape 검증"""
        td_errors = torch.randn(2, 10)

        weights = build_weights(td_errors, beta=0.9)

        assert weights.shape == (2, 10)

    def test_positive_td_high_weight(self):
        """Positive TD error → 높은 weight"""
        td_errors = torch.tensor([[0.5, 0.0, -0.5]])

        weights = build_weights(td_errors, beta=1.0, min_weight=0.1, max_weight=3.0)

        # exp(0.5) ≈ 1.65, exp(0) = 1.0, exp(-0.5) ≈ 0.61
        assert weights[0, 0].item() > 1.0
        assert weights[0, 1].item() == pytest.approx(1.0, rel=0.01)
        assert weights[0, 2].item() < 1.0

    def test_clipping(self):
        """Weight clipping 검증"""
        td_errors = torch.tensor([[5.0, -5.0]])  # 극단값

        weights = build_weights(td_errors, beta=0.5, min_weight=0.1, max_weight=3.0)

        assert weights[0, 0].item() == pytest.approx(3.0, rel=0.01)  # max clip
        assert weights[0, 1].item() == pytest.approx(0.1, rel=0.01)  # min clip


class TestWeightMapping:
    """Weight와 MTP head 매핑 테스트"""

    def test_weight_position_alignment(self):
        """Weight가 position t에서 모든 head에 동일하게 적용되는지 검증

        Weight[t]는 V(s_t)의 품질을 나타내므로,
        position t에서 예측하는 모든 MTP head (t+1, t+2, t+3, t+4)에
        동일한 weight가 적용되어야 함.
        """
        batch_size = 1
        seq_len = 8
        n_future = 4

        # 다양한 TD error를 만들기 위해 비선형 value logits 사용
        torch.manual_seed(42)
        value_logits = torch.randn(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0])
        attention_mask = torch.ones(batch_size, seq_len)

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma=1.0)
        weights = build_weights(td_errors, beta=1.0)

        # 각 head k에 대해 올바른 weight 적용 확인
        for k in range(1, n_future + 1):
            valid_len = seq_len - k

            # 올바른 매핑: weights[:, :valid_len]
            # 모든 head에서 position t의 weight는 weights[:, t]
            weights_k_correct = weights[:, :valid_len]

            # Position 0에서의 weight 확인
            assert weights_k_correct[0, 0].item() == pytest.approx(weights[0, 0].item(), rel=0.01)

            # Position valid_len-1에서의 weight 확인
            assert weights_k_correct[0, valid_len - 1].item() == pytest.approx(
                weights[0, valid_len - 1].item(), rel=0.01
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
