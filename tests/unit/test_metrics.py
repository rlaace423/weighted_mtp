"""Metrics Unit Tests"""

import pytest
import torch

from weighted_mtp.value_weighting.metrics import compute_weight_stats, compute_td_stats


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

        # 평균: (1+2+3+4+5+6) / 6 = 3.5
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
        # 모든 weight가 동일 → 최대 엔트로피
        weights = torch.ones(2, 10)

        stats = compute_weight_stats(weights)

        # Normalized entropy는 1.0에 가까워야 함
        assert stats["weight_entropy"] > 0.99

    def test_skewed_distribution_entropy(self):
        """편향된 분포의 엔트로피 검증 (normalized entropy < 1.0)"""
        # 하나만 큰 값, 나머지는 작은 값 → 낮은 엔트로피
        weights = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1]])

        stats = compute_weight_stats(weights)

        # Normalized entropy는 1.0보다 작아야 함
        assert stats["weight_entropy"] < 0.8

    def test_batch_computation(self):
        """배치 처리 검증"""
        weights = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        stats = compute_weight_stats(weights)

        # 전체 flatten된 통계
        assert stats["weight_mean"] == 3.5
        assert stats["weight_std"] > 0  # 분산이 있어야 함

    def test_zero_division_safety(self):
        """Zero division 방지 검증"""
        # 모든 weight가 0인 경우
        weights = torch.zeros(2, 3)

        stats = compute_weight_stats(weights)

        # NaN이 아니어야 함
        assert not torch.isnan(torch.tensor(stats["weight_mean"]))
        assert not torch.isnan(torch.tensor(stats["weight_entropy"]))

    def test_nan_inf_safety(self):
        """NaN/Inf 처리 검증"""
        # 매우 작은 값들
        weights = torch.tensor([[1e-10, 1e-9, 1e-8]])

        stats = compute_weight_stats(weights)

        # NaN/Inf가 아니어야 함
        for value in stats.values():
            assert not torch.isnan(torch.tensor(value))
            assert not torch.isinf(torch.tensor(value))


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

        # 평균: (0.2-0.5+0.1+0.3-0.3+0.0) / 6 = -0.2 / 6 ≈ -0.033
        expected_mean = -0.2 / 6
        assert abs(stats["td_mean"] - expected_mean) < 1e-4

    def test_min_max(self):
        """최소/최대값 검증"""
        td_errors = torch.tensor([[0.5, -0.9, 0.3], [0.1, -0.2, 0.8]])

        stats = compute_td_stats(td_errors)

        assert abs(stats["td_min"] - (-0.9)) < 1e-4
        assert abs(stats["td_max"] - 0.8) < 1e-4

    def test_std_calculation(self):
        """표준편차 계산 검증"""
        # 모든 값이 동일 → std = 0
        td_errors = torch.ones(2, 3) * 0.5

        stats = compute_td_stats(td_errors)

        assert stats["td_std"] < 1e-6

    def test_batch_computation(self):
        """배치 처리 검증"""
        td_errors = torch.tensor([
            [0.2, 0.3, -0.5],
            [-0.3, 0.1, 0.4],
        ])

        stats = compute_td_stats(td_errors)

        # 전체 flatten된 통계
        expected_mean = (0.2 + 0.3 - 0.5 - 0.3 + 0.1 + 0.4) / 6
        assert abs(stats["td_mean"] - expected_mean) < 1e-4

    def test_positive_td_errors(self):
        """Positive TD errors 검증"""
        td_errors = torch.tensor([[0.1, 0.2, 0.3]])

        stats = compute_td_stats(td_errors)

        assert stats["td_mean"] > 0
        assert stats["td_min"] > 0

    def test_negative_td_errors(self):
        """Negative TD errors 검증"""
        td_errors = torch.tensor([[-0.1, -0.2, -0.3]])

        stats = compute_td_stats(td_errors)

        assert stats["td_mean"] < 0
        assert stats["td_max"] < 0

    def test_mixed_td_errors(self):
        """Mixed (positive + negative) TD errors 검증"""
        td_errors = torch.tensor([[0.5, -0.5, 0.2, -0.2]])

        stats = compute_td_stats(td_errors)

        # 평균은 0에 가까워야 함
        assert abs(stats["td_mean"]) < 0.1
        # Min/Max는 대칭적이어야 함
        assert abs(stats["td_min"] - (-0.5)) < 1e-4
        assert abs(stats["td_max"] - 0.5) < 1e-4


class TestMetricsIntegration:
    """Metrics 통합 테스트"""

    def test_weight_and_td_stats_together(self):
        """Weight와 TD stats 함께 계산"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]])
        weights = torch.exp(td_errors / 0.9)

        td_stats = compute_td_stats(td_errors)
        weight_stats = compute_weight_stats(weights)

        # 두 통계 모두 유효해야 함
        assert td_stats["td_mean"] is not None
        assert weight_stats["weight_mean"] is not None

    def test_positive_td_leads_to_high_weight(self):
        """Positive TD error → 높은 weight 검증"""
        # Positive TD errors
        td_errors_positive = torch.tensor([[0.5, 0.3, 0.2]])
        weights_positive = torch.exp(td_errors_positive / 0.9)

        # Negative TD errors
        td_errors_negative = torch.tensor([[-0.5, -0.3, -0.2]])
        weights_negative = torch.exp(td_errors_negative / 0.9)

        stats_positive = compute_weight_stats(weights_positive)
        stats_negative = compute_weight_stats(weights_negative)

        # Positive TD의 평균 weight가 더 높아야 함
        assert stats_positive["weight_mean"] > stats_negative["weight_mean"]
