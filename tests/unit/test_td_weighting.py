"""TD Weighting Unit Tests

td_weighting.py의 모든 함수 테스트:
- compute_td_errors()
- build_weights()
- compute_td_stats()
- compute_weight_stats()
"""

import pytest
import torch

from weighted_mtp.value_weighting.td_weighting import (
    compute_td_errors,
    build_weights,
    compute_td_stats,
    compute_weight_stats,
)


class TestComputeTDErrors:
    """compute_td_errors() 함수 테스트"""

    def test_basic_computation(self):
        """기본 TD error 계산 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])  # Correct
        attention_mask = torch.tensor([[1, 1, 1]])  # All valid
        gamma = 1.0

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma)

        # Intermediate (0→1): 1.0 * 0.7 - 0.5 = 0.2
        # Intermediate (1→2): 1.0 * 0.9 - 0.7 = 0.2
        # Terminal (2): 1.0 - 0.9 = 0.1
        expected = torch.tensor([[0.2, 0.2, 0.1]])

        assert td_errors.shape == (1, 3)
        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_incorrect_sample(self):
        """Incorrect 샘플 (reward=0) TD error 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([0.0])  # Incorrect
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Terminal (2): 0.0 - 0.9 = -0.9 (negative!)
        expected = torch.tensor([[0.2, 0.2, -0.9]])

        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_with_padding(self):
        """Padding이 있는 경우 masking 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.0], [0.0]]])  # [1, 4, 1]
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 0, 0]])  # 마지막 2개는 padding

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Terminal index = 1 (마지막 유효 토큰)
        # Intermediate (0→1): 0.7 - 0.5 = 0.2
        # Terminal (1): 1.0 - 0.7 = 0.3
        # Padding (2, 3): 0.0 (masked)
        expected = torch.tensor([[0.2, 0.3, 0.0, 0.0]])

        assert td_errors.shape == (1, 4)
        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_batch_computation(self):
        """배치 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9]],  # Sample 1: correct
            [[0.3], [0.6], [0.8]],  # Sample 2: incorrect
        ])  # [2, 3, 1]
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Sample 1 (correct)
        expected_s1 = torch.tensor([0.2, 0.2, 0.1])
        # Sample 2 (incorrect): Terminal: 0.0 - 0.8 = -0.8
        expected_s2 = torch.tensor([0.3, 0.2, -0.8])

        assert td_errors.shape == (2, 3)
        torch.testing.assert_close(td_errors[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(td_errors[1], expected_s2, rtol=1e-4, atol=1e-4)

    def test_different_gamma(self):
        """Gamma 값 변경 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])
        gamma = 0.99

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma)

        # Intermediate (0→1): 0.99 * 0.7 - 0.5 = 0.193
        # Intermediate (1→2): 0.99 * 0.9 - 0.7 = 0.191
        # Terminal (2): 1.0 - 0.9 = 0.1
        expected = torch.tensor([[0.193, 0.191, 0.1]])

        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_single_token_sequence(self):
        """단일 토큰 시퀀스 처리 검증"""
        value_logits = torch.tensor([[[0.8]]])  # [1, 1, 1]
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Terminal만 존재: 1.0 - 0.8 = 0.2
        expected = torch.tensor([[0.2]])

        assert td_errors.shape == (1, 1)
        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_gradient_computation(self):
        """Gradient 계산 가능 여부 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]], requires_grad=True)
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        loss = td_errors.pow(2).mean()
        loss.backward()

        assert value_logits.grad is not None
        assert value_logits.grad.shape == value_logits.shape

    def test_batch_with_different_lengths(self):
        """배치 내 다른 길이 시퀀스 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9], [0.0]],  # Length 3
            [[0.3], [0.6], [0.0], [0.0]],  # Length 2
        ])  # [2, 4, 1]
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Sample 1: terminal index = 2
        expected_s1 = torch.tensor([0.2, 0.2, 0.1, 0.0])
        # Sample 2: terminal index = 1
        expected_s2 = torch.tensor([0.3, -0.6, 0.0, 0.0])

        assert td_errors.shape == (2, 4)
        torch.testing.assert_close(td_errors[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(td_errors[1], expected_s2, rtol=1e-4, atol=1e-4)


class TestBuildWeights:
    """build_weights() 함수 테스트"""

    def test_basic_weighting(self):
        """기본 exponential weighting 검증"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]])
        beta = 0.9

        weights = build_weights(td_errors, beta=beta)

        expected = torch.exp(td_errors / beta)
        expected = torch.clamp(expected, min=0.1, max=3.0)

        assert weights.shape == (1, 3)
        torch.testing.assert_close(weights, expected, rtol=1e-4, atol=1e-4)

    def test_positive_td_error(self):
        """Positive TD error → weight > 1 검증"""
        td_errors = torch.tensor([[0.5, 1.0, 0.2]])

        weights = build_weights(td_errors, beta=0.9)

        assert (weights > 1.0).all()

    def test_negative_td_error(self):
        """Negative TD error → weight < 1 검증"""
        td_errors = torch.tensor([[-0.5, -1.0, -0.2]])

        weights = build_weights(td_errors, beta=0.9)

        assert (weights < 1.0).all()

    def test_zero_td_error(self):
        """Zero TD error → weight = 1.0 검증"""
        td_errors = torch.tensor([[0.0, 0.0, 0.0]])

        weights = build_weights(td_errors, beta=0.9)

        expected = torch.ones(1, 3)
        torch.testing.assert_close(weights, expected, rtol=1e-4, atol=1e-4)

    def test_clipping(self):
        """Weight clipping 검증 (min/max)"""
        # Extreme positive TD error
        td_errors_positive = torch.tensor([[10.0]])
        weights_positive = build_weights(td_errors_positive, beta=0.9, min_weight=0.1, max_weight=3.0)
        assert weights_positive.item() <= 3.0

        # Extreme negative TD error
        td_errors_negative = torch.tensor([[-10.0]])
        weights_negative = build_weights(td_errors_negative, beta=0.9, min_weight=0.1, max_weight=3.0)
        assert weights_negative.item() >= 0.1

    def test_beta_sensitivity(self):
        """Beta 값 변화 검증"""
        td_errors = torch.tensor([[1.0]])

        # Small beta → steeper (더 집중적)
        weights_small_beta = build_weights(td_errors, beta=0.5)
        # Large beta → flatter (더 완만함)
        weights_large_beta = build_weights(td_errors, beta=2.0)

        assert weights_small_beta > weights_large_beta

    def test_batch_weighting(self):
        """배치 처리 검증"""
        td_errors = torch.tensor([
            [0.2, -0.5, 0.1],
            [0.3, -0.3, 0.0],
        ])

        weights = build_weights(td_errors, beta=0.9)

        assert weights.shape == (2, 3)
        assert (weights >= 0.1).all()
        assert (weights <= 3.0).all()


class TestComputeTDStats:
    """compute_td_stats() 함수 테스트"""

    def test_basic_statistics(self):
        """기본 통계 계산 검증"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]])

        stats = compute_td_stats(td_errors)

        assert "td_mean" in stats
        assert "td_std" in stats
        assert "td_min" in stats
        assert "td_max" in stats

    def test_mean_calculation(self):
        """평균 계산 검증"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1], [0.3, -0.3, 0.0]])

        stats = compute_td_stats(td_errors)

        expected_mean = (0.2 - 0.5 + 0.1 + 0.3 - 0.3 + 0.0) / 6
        assert abs(stats["td_mean"] - expected_mean) < 1e-4

    def test_min_max(self):
        """최소/최대값 검증"""
        td_errors = torch.tensor([[0.5, -0.9, 0.3], [0.1, -0.2, 0.8]])

        stats = compute_td_stats(td_errors)

        assert abs(stats["td_min"] - (-0.9)) < 1e-4
        assert abs(stats["td_max"] - 0.8) < 1e-4

    def test_std_calculation(self):
        """표준편차 계산 검증"""
        td_errors = torch.ones(2, 3) * 0.5

        stats = compute_td_stats(td_errors)

        assert stats["td_std"] < 1e-6

    def test_mixed_td_errors(self):
        """Mixed (positive + negative) TD errors 검증"""
        td_errors = torch.tensor([[0.5, -0.5, 0.2, -0.2]])

        stats = compute_td_stats(td_errors)

        assert abs(stats["td_mean"]) < 0.1
        assert abs(stats["td_min"] - (-0.5)) < 1e-4
        assert abs(stats["td_max"] - 0.5) < 1e-4


class TestComputeWeightStats:
    """compute_weight_stats() 함수 테스트"""

    def test_basic_statistics(self):
        """기본 통계 계산 검증"""
        weights = torch.tensor([[1.0, 2.0, 3.0]])

        stats = compute_weight_stats(weights)

        assert "weight_mean" in stats
        assert "weight_std" in stats
        assert "weight_min" in stats
        assert "weight_max" in stats
        assert "weight_entropy" in stats

    def test_mean_calculation(self):
        """평균 계산 검증"""
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        stats = compute_weight_stats(weights)

        expected_mean = 3.5
        assert abs(stats["weight_mean"] - expected_mean) < 1e-4

    def test_min_max(self):
        """최소/최대값 검증"""
        weights = torch.tensor([[0.5, 2.0, 1.0], [0.1, 5.0, 1.5]])

        stats = compute_weight_stats(weights)

        assert abs(stats["weight_min"] - 0.1) < 1e-4
        assert abs(stats["weight_max"] - 5.0) < 1e-4

    def test_uniform_distribution_entropy(self):
        """균등 분포의 엔트로피 검증 (normalized entropy ≈ 1.0)"""
        weights = torch.ones(2, 10)

        stats = compute_weight_stats(weights)

        assert stats["weight_entropy"] > 0.99

    def test_skewed_distribution_entropy(self):
        """편향된 분포의 엔트로피 검증 (normalized entropy < 1.0)"""
        weights = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1]])

        stats = compute_weight_stats(weights)

        assert stats["weight_entropy"] < 0.8

    def test_batch_computation(self):
        """배치 처리 검증"""
        weights = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        stats = compute_weight_stats(weights)

        assert stats["weight_mean"] == 3.5
        assert stats["weight_std"] > 0

    def test_nan_inf_safety(self):
        """NaN/Inf 처리 검증"""
        weights = torch.tensor([[1e-10, 1e-9, 1e-8]])

        stats = compute_weight_stats(weights)

        for value in stats.values():
            assert not torch.isnan(torch.tensor(value))
            assert not torch.isinf(torch.tensor(value))


class TestTDWeightingIntegration:
    """통합 테스트: TD error → weights → stats"""

    def test_full_pipeline(self):
        """전체 파이프라인 검증"""
        # TD error 계산
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Weight 계산
        weights = build_weights(td_errors, beta=0.9)

        # Stats 계산
        td_stats = compute_td_stats(td_errors)
        weight_stats = compute_weight_stats(weights)

        assert td_stats["td_mean"] is not None
        assert weight_stats["weight_mean"] is not None
        assert weights.shape == td_errors.shape

    def test_positive_td_leads_to_high_weight(self):
        """Positive TD error → 높은 weight 검증"""
        td_errors_positive = torch.tensor([[0.5, 0.3, 0.2]])
        weights_positive = build_weights(td_errors_positive, beta=0.9)

        td_errors_negative = torch.tensor([[-0.5, -0.3, -0.2]])
        weights_negative = build_weights(td_errors_negative, beta=0.9)

        stats_positive = compute_weight_stats(weights_positive)
        stats_negative = compute_weight_stats(weights_negative)

        assert stats_positive["weight_mean"] > stats_negative["weight_mean"]

    def test_incorrect_sample_downweighting(self):
        """Incorrect 샘플 자동 down-weighting 검증"""
        # Correct sample
        value_logits_correct = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards_correct = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])
        td_errors_correct = compute_td_errors(value_logits_correct, rewards_correct, attention_mask)
        weights_correct = build_weights(td_errors_correct, beta=0.9)

        # Incorrect sample
        value_logits_incorrect = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards_incorrect = torch.tensor([0.0])
        td_errors_incorrect = compute_td_errors(value_logits_incorrect, rewards_incorrect, attention_mask)
        weights_incorrect = build_weights(td_errors_incorrect, beta=0.9)

        # Incorrect 샘플의 terminal weight가 낮아야 함
        assert weights_incorrect[0, -1] < weights_correct[0, -1]
