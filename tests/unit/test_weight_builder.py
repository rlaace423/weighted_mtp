"""Weight Builder Unit Tests"""

import pytest
import torch

from weighted_mtp.value_weighting.weight_builder import build_weights


class TestBuildWeights:
    """build_weights() 함수 테스트"""

    def test_basic_weighting(self):
        """기본 exponential weighting 검증"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]])
        beta = 0.9

        weights = build_weights(td_errors, beta=beta)

        # 예상 출력
        # exp(0.2 / 0.9) = exp(0.222) ≈ 1.249
        # exp(-0.5 / 0.9) = exp(-0.556) ≈ 0.573
        # exp(0.1 / 0.9) = exp(0.111) ≈ 1.117
        expected = torch.exp(td_errors / beta)

        assert weights.shape == (1, 3)
        torch.testing.assert_close(weights, expected, rtol=1e-4, atol=1e-4)

    def test_positive_td_error(self):
        """Positive TD error → weight > 1 검증"""
        td_errors = torch.tensor([[0.5, 1.0, 0.2]])

        weights = build_weights(td_errors, beta=0.9)

        # 모든 weight가 1보다 커야 함
        assert (weights > 1.0).all()

    def test_negative_td_error(self):
        """Negative TD error → weight < 1 검증"""
        td_errors = torch.tensor([[-0.5, -1.0, -0.2]])

        weights = build_weights(td_errors, beta=0.9)

        # 모든 weight가 1보다 작아야 함
        assert (weights < 1.0).all()

    def test_zero_td_error(self):
        """Zero TD error → weight = 1.0 검증"""
        td_errors = torch.tensor([[0.0, 0.0, 0.0]])

        weights = build_weights(td_errors, beta=0.9)

        # exp(0 / beta) = exp(0) = 1.0
        expected = torch.ones_like(td_errors)

        torch.testing.assert_close(weights, expected, rtol=1e-4, atol=1e-4)

    def test_clipping_min(self):
        """최소 가중치 clipping 검증"""
        # 매우 큰 negative TD error
        td_errors = torch.tensor([[-10.0, -5.0, -2.0]])
        min_weight = 0.1

        weights = build_weights(td_errors, beta=0.9, min_weight=min_weight)

        # 모든 weight가 min_weight 이상이어야 함
        assert (weights >= min_weight).all()
        # 적어도 하나는 min_weight에 clipping되어야 함
        assert (weights == min_weight).any()

    def test_clipping_max(self):
        """최대 가중치 clipping 검증"""
        # 매우 큰 positive TD error
        td_errors = torch.tensor([[10.0, 5.0, 2.0]])
        max_weight = 5.0

        weights = build_weights(td_errors, beta=0.9, max_weight=max_weight)

        # 모든 weight가 max_weight 이하여야 함
        assert (weights <= max_weight).all()
        # 적어도 하나는 max_weight에 clipping되어야 함
        assert (weights == max_weight).any()

    def test_beta_sensitivity(self):
        """Beta 값 변화에 따른 민감도 검증"""
        td_errors = torch.tensor([[0.5, -0.5]])

        # Beta가 낮을수록 가중치 차이가 커짐
        weights_low_beta = build_weights(td_errors, beta=0.1)
        weights_high_beta = build_weights(td_errors, beta=10.0)

        # Low beta: 더 극단적인 가중치
        assert weights_low_beta[0, 0] > weights_high_beta[0, 0]  # Positive TD
        assert weights_low_beta[0, 1] < weights_high_beta[0, 1]  # Negative TD

    def test_batch_computation(self):
        """배치 처리 검증"""
        td_errors = torch.tensor([
            [0.2, 0.3, -0.5],  # Sample 1
            [-0.3, 0.1, 0.4],  # Sample 2
        ])

        weights = build_weights(td_errors, beta=0.9)

        assert weights.shape == (2, 3)

        # 각 샘플별 독립적 계산 검증
        expected_s1 = torch.exp(td_errors[0] / 0.9)
        expected_s2 = torch.exp(td_errors[1] / 0.9)

        torch.testing.assert_close(weights[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(weights[1], expected_s2, rtol=1e-4, atol=1e-4)

    def test_gradient_computation(self):
        """Gradient 계산 가능 여부 검증"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]], requires_grad=True)

        weights = build_weights(td_errors, beta=0.9)

        # Weight로 loss 계산
        loss = weights.sum()
        loss.backward()

        # Gradient가 계산되었는지 확인
        assert td_errors.grad is not None
        assert td_errors.grad.shape == td_errors.shape

    def test_default_parameters(self):
        """기본 파라미터 검증"""
        td_errors = torch.tensor([[0.5, -0.5]])

        weights = build_weights(td_errors)

        # 기본값: beta=0.9, min_weight=0.1, max_weight=5.0
        expected = torch.exp(td_errors / 0.9)
        expected = torch.clamp(expected, min=0.1, max=5.0)

        torch.testing.assert_close(weights, expected, rtol=1e-4, atol=1e-4)

    def test_weight_distribution(self):
        """Weight 분포 검증 (평균 ≈ 1.0 부근)"""
        # 균형잡힌 TD errors
        td_errors = torch.tensor([
            [0.2, -0.2, 0.1, -0.1, 0.0],
            [0.3, -0.3, 0.2, -0.2, 0.0],
        ])

        weights = build_weights(td_errors, beta=0.9)

        # 평균이 1.0 부근이어야 함 (균형잡힌 가중화)
        mean_weight = weights.mean()
        assert 0.8 < mean_weight < 1.2

    def test_incorrect_sample_downweight(self):
        """Incorrect 샘플 자동 down-weighting 검증"""
        # Incorrect 샘플: 대부분 negative TD error
        td_errors_incorrect = torch.tensor([[-0.9, -0.5, -0.3]])
        # Correct 샘플: 대부분 positive TD error
        td_errors_correct = torch.tensor([[0.1, 0.2, 0.3]])

        weights_incorrect = build_weights(td_errors_incorrect, beta=0.9)
        weights_correct = build_weights(td_errors_correct, beta=0.9)

        # Incorrect 샘플의 평균 weight가 correct보다 낮아야 함
        assert weights_incorrect.mean() < weights_correct.mean()
        # Incorrect 샘플의 대부분 weight가 1.0보다 작아야 함
        assert (weights_incorrect < 1.0).sum() >= 2
