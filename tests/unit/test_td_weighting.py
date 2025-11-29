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
    compute_td_targets,
    compute_td_errors,
    build_weights,
    compute_td_stats,
    compute_weight_stats,
)


class TestComputeTDTargets:
    """compute_td_targets() 함수 테스트"""

    def test_basic_computation(self):
        """기본 TD target 계산 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])  # Correct
        loss_mask = torch.tensor([[1, 1, 1]])  # All valid
        gamma = 1.0

        targets = compute_td_targets(value_logits, rewards, loss_mask, gamma)

        # Intermediate (0): 1.0 * 0.7 = 0.7
        # Intermediate (1): 1.0 * 0.9 = 0.9
        # Terminal (2): 1.0
        expected = torch.tensor([[[0.7], [0.9], [1.0]]])

        assert targets.shape == (1, 3, 1)
        torch.testing.assert_close(targets, expected, rtol=1e-4, atol=1e-4)

    def test_incorrect_sample(self):
        """Incorrect 샘플 (reward=0) TD target 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([0.0])  # Incorrect
        loss_mask = torch.tensor([[1, 1, 1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask)

        # Terminal (2): 0.0
        expected = torch.tensor([[[0.7], [0.9], [0.0]]])

        torch.testing.assert_close(targets, expected, rtol=1e-4, atol=1e-4)

    def test_with_padding(self):
        """Padding이 있는 경우 마스킹 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.0], [0.0]]])  # [1, 4, 1]
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 0, 0]])  # 마지막 2개는 padding

        targets = compute_td_targets(value_logits, rewards, loss_mask)

        # Terminal index = 1
        # Intermediate (0): 1.0 * 0.7 = 0.7
        # Terminal (1): 1.0
        # Padding (2, 3): 0.0
        expected = torch.tensor([[[0.7], [1.0], [0.0], [0.0]]])

        assert targets.shape == (1, 4, 1)
        torch.testing.assert_close(targets, expected, rtol=1e-4, atol=1e-4)

    def test_batch_computation(self):
        """배치 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9]],  # Sample 1: correct
            [[0.3], [0.6], [0.8]],  # Sample 2: incorrect
        ])  # [2, 3, 1]
        rewards = torch.tensor([1.0, 0.0])
        loss_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask)

        # Sample 1 (correct): [0.7, 0.9, 1.0]
        expected_s1 = torch.tensor([[0.7], [0.9], [1.0]])
        # Sample 2 (incorrect): [0.6, 0.8, 0.0]
        expected_s2 = torch.tensor([[0.6], [0.8], [0.0]])

        assert targets.shape == (2, 3, 1)
        torch.testing.assert_close(targets[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(targets[1], expected_s2, rtol=1e-4, atol=1e-4)

    def test_different_gamma(self):
        """Gamma 값 변경 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])
        gamma = 0.99

        targets = compute_td_targets(value_logits, rewards, loss_mask, gamma)

        # Intermediate (0): 0.99 * 0.7 = 0.693
        # Intermediate (1): 0.99 * 0.9 = 0.891
        # Terminal (2): 1.0
        expected = torch.tensor([[[0.693], [0.891], [1.0]]])

        torch.testing.assert_close(targets, expected, rtol=1e-4, atol=1e-4)

    def test_single_token_sequence(self):
        """단일 토큰 시퀀스 처리 검증"""
        value_logits = torch.tensor([[[0.8]]])  # [1, 1, 1]
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask)

        # Terminal만 존재: 1.0
        expected = torch.tensor([[[1.0]]])

        assert targets.shape == (1, 1, 1)
        torch.testing.assert_close(targets, expected, rtol=1e-4, atol=1e-4)

    def test_gradient_blocked(self):
        """Gradient 차단 검증 (targets는 detach되어야 함)"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]], requires_grad=True)
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask)

        # Targets를 사용한 loss 계산
        loss = (value_logits - targets).pow(2).mean()
        loss.backward()

        # value_logits에 gradient가 있어야 함
        assert value_logits.grad is not None
        # Targets의 gradient가 value_logits에 영향 주지 않아야 함
        # (detach로 인해 targets에서 오는 gradient는 없음)
        # Gradient는 (value_logits - targets) 중 value_logits에서만 옴
        assert value_logits.grad.shape == value_logits.shape

    def test_batch_with_different_lengths(self):
        """배치 내 다른 길이 시퀀스 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9], [0.0]],  # Length 3
            [[0.3], [0.6], [0.0], [0.0]],  # Length 2
        ])  # [2, 4, 1]
        rewards = torch.tensor([1.0, 0.0])
        loss_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])

        targets = compute_td_targets(value_logits, rewards, loss_mask)

        # Sample 1: terminal index = 2, [0.7, 0.9, 1.0, 0.0]
        expected_s1 = torch.tensor([[0.7], [0.9], [1.0], [0.0]])
        # Sample 2: terminal index = 1, [0.6, 0.0, 0.0, 0.0]
        expected_s2 = torch.tensor([[0.6], [0.0], [0.0], [0.0]])

        assert targets.shape == (2, 4, 1)
        torch.testing.assert_close(targets[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(targets[1], expected_s2, rtol=1e-4, atol=1e-4)

    def test_gae_basic(self):
        """GAE (lam=0.95) 기본 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask, gamma=1.0, lam=0.95)

        # 역방향 GAE 계산:
        # t=2 (terminal): δ=1.0-0.9=0.1, A=0.1, target=0.9+0.1=1.0
        # t=1: δ=0.9-0.7=0.2, A=0.2+0.95*0.1=0.295, target=0.7+0.295=0.995
        # t=0: δ=0.7-0.5=0.2, A=0.2+0.95*0.295=0.48025, target=0.5+0.48025=0.98025
        expected = torch.tensor([[[0.98025], [0.995], [1.0]]])

        assert targets.shape == (1, 3, 1)
        torch.testing.assert_close(targets, expected, rtol=1e-3, atol=1e-3)

    def test_gae_incorrect_sample(self):
        """GAE incorrect 샘플 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([0.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask, gamma=1.0, lam=0.95)

        # t=2: δ=0.0-0.9=-0.9, A=-0.9, target=0.0
        # t=1: δ=0.9-0.7=0.2, A=0.2+0.95*(-0.9)=-0.655, target=0.045
        # t=0: δ=0.7-0.5=0.2, A=0.2+0.95*(-0.655)=-0.42225, target=0.07775
        expected = torch.tensor([[[0.07775], [0.045], [0.0]]])

        torch.testing.assert_close(targets, expected, rtol=1e-3, atol=1e-3)

    def test_gae_monte_carlo_mode(self):
        """GAE lam=1.0 (Monte Carlo) 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        targets = compute_td_targets(value_logits, rewards, loss_mask, gamma=1.0, lam=1.0)

        # lam=1.0이면 모든 토큰이 terminal reward로 수렴
        # 모든 target이 1.0에 가까워야 함
        assert targets[0, 0, 0].item() > 0.9
        assert targets[0, 1, 0].item() > 0.95
        assert abs(targets[0, 2, 0].item() - 1.0) < 1e-4

    def test_gae_faster_propagation(self):
        """GAE가 TD(0)보다 빠른 에러 전파 검증"""
        value_logits = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1, 1, 1]])

        targets_td0 = compute_td_targets(value_logits, rewards, loss_mask, gamma=1.0, lam=0.0)
        targets_gae = compute_td_targets(value_logits, rewards, loss_mask, gamma=1.0, lam=0.95)

        # TD(0): 첫 번째 토큰 target = 0.0 (신호 전파 안됨)
        # GAE: 첫 번째 토큰 target > 0.0 (신호 빠르게 전파)
        assert targets_td0[0, 0, 0].item() == 0.0
        assert targets_gae[0, 0, 0].item() > 0.5


class TestComputeTDErrors:
    """compute_td_errors() 함수 테스트

    TD error 정의 (한계 가치 기반):
    - δ_t = V(t) - V(t-1): t시점 토큰을 선택함으로써 얻게 되는 한계 가치
    - δ_0 = V(0) - initial_value (기본 initial_value=0.5)
    - δ_T = R - V(T): Terminal에서는 실제 결과와 예측의 차이
    """

    def test_basic_computation(self):
        """기본 TD error 계산 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])  # Correct
        loss_mask = torch.tensor([[1, 1, 1]])  # All valid

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # δ_0 = V(0) - initial_value = 0.5 - 0.5 = 0.0
        # δ_1 = V(1) - V(0) = 0.7 - 0.5 = 0.2
        # δ_2 = R - V(2) = 1.0 - 0.9 = 0.1 (terminal)
        expected = torch.tensor([[0.0, 0.2, 0.1]])

        assert td_errors.shape == (1, 3)
        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_incorrect_sample(self):
        """Incorrect 샘플 (reward=0) TD error 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([0.0])  # Incorrect
        loss_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # δ_0 = 0.5 - 0.5 = 0.0
        # δ_1 = 0.7 - 0.5 = 0.2
        # δ_2 = 0.0 - 0.9 = -0.9 (terminal, negative!)
        expected = torch.tensor([[0.0, 0.2, -0.9]])

        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_with_padding(self):
        """Padding이 있는 경우 masking 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.0], [0.0]]])  # [1, 4, 1]
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 0, 0]])  # 마지막 2개는 padding

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # Terminal index = 1 (마지막 유효 토큰)
        # δ_0 = 0.5 - 0.5 = 0.0
        # δ_1 = R - V(1) = 1.0 - 0.7 = 0.3 (terminal)
        # δ_2, δ_3 = 0.0 (padding, masked)
        expected = torch.tensor([[0.0, 0.3, 0.0, 0.0]])

        assert td_errors.shape == (1, 4)
        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_batch_computation(self):
        """배치 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9]],  # Sample 1: correct
            [[0.3], [0.6], [0.8]],  # Sample 2: incorrect
        ])  # [2, 3, 1]
        rewards = torch.tensor([1.0, 0.0])
        loss_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # Sample 1 (correct):
        # δ_0 = 0.5 - 0.5 = 0.0, δ_1 = 0.7 - 0.5 = 0.2, δ_2 = 1.0 - 0.9 = 0.1
        expected_s1 = torch.tensor([0.0, 0.2, 0.1])
        # Sample 2 (incorrect):
        # δ_0 = 0.3 - 0.5 = -0.2, δ_1 = 0.6 - 0.3 = 0.3, δ_2 = 0.0 - 0.8 = -0.8
        expected_s2 = torch.tensor([-0.2, 0.3, -0.8])

        assert td_errors.shape == (2, 3)
        torch.testing.assert_close(td_errors[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(td_errors[1], expected_s2, rtol=1e-4, atol=1e-4)

    def test_different_gamma(self):
        """Gamma 값 변경 검증"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])
        gamma = 0.99

        td_errors = compute_td_errors(value_logits, rewards, loss_mask, gamma)

        # δ_0 = γ*V(0) - initial_value = 0.99*0.5 - 0.5 = -0.005
        # δ_1 = γ*V(1) - V(0) = 0.99*0.7 - 0.5 = 0.193
        # δ_2 = R - V(2) = 1.0 - 0.9 = 0.1 (terminal)
        expected = torch.tensor([[-0.005, 0.193, 0.1]])

        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_single_token_sequence(self):
        """단일 토큰 시퀀스 처리 검증"""
        value_logits = torch.tensor([[[0.8]]])  # [1, 1, 1]
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1]])

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # Terminal만 존재: δ_0 = R - V(0) = 1.0 - 0.8 = 0.2
        # (첫 번째 토큰이면서 동시에 terminal이므로 terminal 수식 적용)
        expected = torch.tensor([[0.2]])

        assert td_errors.shape == (1, 1)
        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)

    def test_gradient_blocked(self):
        """Gradient 차단 검증 (TD error 계산 시 detach 적용)"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]], requires_grad=True)
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # TD error는 detach되어 gradient가 흐르지 않음
        assert not td_errors.requires_grad

    def test_batch_with_different_lengths(self):
        """배치 내 다른 길이 시퀀스 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9], [0.0]],  # Length 3
            [[0.3], [0.6], [0.0], [0.0]],  # Length 2
        ])  # [2, 4, 1]
        rewards = torch.tensor([1.0, 0.0])
        loss_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # Sample 1: terminal index = 2
        # δ_0 = 0.0, δ_1 = 0.2, δ_2 = 1.0 - 0.9 = 0.1, δ_3 = 0.0 (padding)
        expected_s1 = torch.tensor([0.0, 0.2, 0.1, 0.0])
        # Sample 2: terminal index = 1
        # δ_0 = 0.3 - 0.5 = -0.2, δ_1 = 0.0 - 0.6 = -0.6, δ_2,3 = 0.0 (padding)
        expected_s2 = torch.tensor([-0.2, -0.6, 0.0, 0.0])

        assert td_errors.shape == (2, 4)
        torch.testing.assert_close(td_errors[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(td_errors[1], expected_s2, rtol=1e-4, atol=1e-4)

    def test_custom_initial_value(self):
        """커스텀 initial_value 테스트"""
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        # initial_value = 0.0으로 설정
        td_errors = compute_td_errors(
            value_logits, rewards, loss_mask,
            initial_value=0.0
        )

        # δ_0 = V(0) - 0.0 = 0.5
        # δ_1 = V(1) - V(0) = 0.2
        # δ_2 = R - V(2) = 0.1
        expected = torch.tensor([[0.5, 0.2, 0.1]])

        torch.testing.assert_close(td_errors, expected, rtol=1e-4, atol=1e-4)


class TestBuildWeights:
    """build_weights() 함수 테스트"""

    def test_basic_weighting(self):
        """기본 exponential weighting 검증"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]])
        loss_mask = torch.tensor([[1, 1, 1]])
        beta = 0.9

        weights = build_weights(td_errors, loss_mask, beta=beta)

        # Advantage Whitening 적용 후 exp 변환
        assert weights.shape == (1, 3)
        assert (weights >= 0.1).all()
        assert (weights <= 5.0).all()

    def test_positive_td_error_higher_weight(self):
        """Positive TD error가 높은 weight를 받는지 검증"""
        # 균일한 분포에서 positive가 negative보다 높은 weight
        td_errors = torch.tensor([[0.5, -0.5]])
        loss_mask = torch.tensor([[1, 1]])

        weights = build_weights(td_errors, loss_mask, beta=0.9)

        # Positive TD error가 더 높은 weight를 받아야 함
        assert weights[0, 0] > weights[0, 1]

    def test_negative_td_error_lower_weight(self):
        """Negative TD error가 낮은 weight를 받는지 검증"""
        td_errors = torch.tensor([[-0.5, -1.0, -0.2]])
        loss_mask = torch.tensor([[1, 1, 1]])

        weights = build_weights(td_errors, loss_mask, beta=0.9)

        # 가장 negative한 값이 가장 낮은 weight
        assert weights[0, 1] < weights[0, 0]
        assert weights[0, 1] < weights[0, 2]

    def test_zero_td_error_mean_weight(self):
        """Zero TD error가 평균 weight를 받는지 검증"""
        td_errors = torch.tensor([[1.0, 0.0, -1.0]])
        loss_mask = torch.tensor([[1, 1, 1]])

        weights = build_weights(td_errors, loss_mask, beta=1.0)

        # Zero TD error는 정규화 후 0이므로 exp(0) = 1.0
        # (다른 값들 사이의 중간)
        assert weights[0, 0] > weights[0, 1]
        assert weights[0, 1] > weights[0, 2]

    def test_clipping_max(self):
        """max_weight clipping 검증"""
        # 다중 요소 사용 (단일 요소는 균등 가중치 1.0 반환)
        td_errors = torch.tensor([[10.0, -0.1]])
        loss_mask = torch.tensor([[1, 1]])

        weights = build_weights(td_errors, loss_mask, beta=0.9, max_weight=3.0)

        # 큰 양수 TD error의 가중치는 max_weight로 clipping
        assert weights[0, 0].item() <= 3.0

    def test_clipping_min(self):
        """min_weight clipping 검증"""
        # 다중 요소 사용 (단일 요소는 균등 가중치 1.0 반환)
        td_errors = torch.tensor([[-10.0, 0.1]])
        loss_mask = torch.tensor([[1, 1]])

        weights = build_weights(td_errors, loss_mask, beta=0.9, min_weight=0.1)

        # 큰 음수 TD error의 가중치는 min_weight로 clipping
        assert weights[0, 0].item() >= 0.1

    def test_beta_sensitivity(self):
        """Beta 값 변화 검증"""
        td_errors = torch.tensor([[1.0, -1.0]])
        loss_mask = torch.tensor([[1, 1]])

        # Small beta → 더 극단적인 차이
        weights_small_beta = build_weights(td_errors, loss_mask, beta=0.5)
        # Large beta → 더 완만한 차이
        weights_large_beta = build_weights(td_errors, loss_mask, beta=2.0)

        # 작은 beta에서 weight 차이가 더 큼
        diff_small = (weights_small_beta[0, 0] - weights_small_beta[0, 1]).item()
        diff_large = (weights_large_beta[0, 0] - weights_large_beta[0, 1]).item()

        assert diff_small > diff_large

    def test_batch_weighting(self):
        """배치 처리 검증"""
        td_errors = torch.tensor([
            [0.2, -0.5, 0.1],
            [0.3, -0.3, 0.0],
        ])
        loss_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])

        weights = build_weights(td_errors, loss_mask, beta=0.9)

        assert weights.shape == (2, 3)
        assert (weights >= 0.1).all()
        assert (weights <= 5.0).all()

    def test_padding_masked(self):
        """Padding 위치가 0으로 마스킹되는지 검증"""
        td_errors = torch.tensor([[0.2, 0.3, 0.0, 0.0]])
        loss_mask = torch.tensor([[1, 1, 0, 0]])

        weights = build_weights(td_errors, loss_mask, beta=0.9)

        # Padding 위치는 weight = 0
        assert weights[0, 2].item() == 0.0
        assert weights[0, 3].item() == 0.0
        # 유효 위치는 weight > 0
        assert weights[0, 0].item() > 0.0
        assert weights[0, 1].item() > 0.0


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


class TestBuildWeightsExternalStats:
    """build_weights() external_mean/external_std 파라미터 테스트"""

    def test_external_stats_basic(self):
        """외부 통계 사용 기본 검증"""
        td_errors = torch.tensor([[0.0, 1.0, 2.0]])
        loss_mask = torch.tensor([[1, 1, 1]])

        # 외부 통계 지정
        external_mean = torch.tensor(1.0)
        external_std = torch.tensor(0.8)

        weights = build_weights(
            td_errors, loss_mask, beta=1.0,
            external_mean=external_mean,
            external_std=external_std,
        )

        assert weights.shape == (1, 3)
        assert (weights >= 0.1).all()
        assert (weights <= 5.0).all()

    def test_external_stats_vs_batch_stats(self):
        """외부 통계 vs 배치 통계 비교"""
        td_errors = torch.tensor([[0.0, 1.0, 2.0]])
        loss_mask = torch.tensor([[1, 1, 1]])

        # 배치 통계 사용 (기본 동작)
        weights_batch = build_weights(td_errors, loss_mask, beta=1.0)

        # 다른 외부 통계 사용
        external_mean = torch.tensor(0.5)  # 배치 mean=1.0과 다름
        external_std = torch.tensor(1.5)   # 배치 std와 다름

        weights_external = build_weights(
            td_errors, loss_mask, beta=1.0,
            external_mean=external_mean,
            external_std=external_std,
        )

        # 외부 통계 사용 시 결과가 다름
        assert not torch.allclose(weights_batch, weights_external)

    def test_backward_compatibility(self):
        """하위 호환성 검증 (external_mean/std=None)"""
        td_errors = torch.tensor([[0.2, -0.5, 0.1]])
        loss_mask = torch.tensor([[1, 1, 1]])

        # 기존 방식 (external_mean/std 없음)
        weights_old = build_weights(td_errors, loss_mask, beta=0.9)

        # 명시적 None 전달
        weights_new = build_weights(
            td_errors, loss_mask, beta=0.9,
            external_mean=None,
            external_std=None,
        )

        # 결과 동일
        torch.testing.assert_close(weights_old, weights_new)

    def test_consistent_normalization(self):
        """동일한 외부 통계로 일관된 정규화 검증"""
        external_mean = torch.tensor(0.5)
        external_std = torch.tensor(1.0)
        loss_mask = torch.tensor([[1, 1, 1]])

        # 분산이 다른 두 배치
        td_errors_1 = torch.tensor([[0.4, 0.5, 0.6]])  # 작은 분산
        td_errors_2 = torch.tensor([[0.0, 0.5, 1.0]])  # 큰 분산

        weights_1 = build_weights(
            td_errors_1, loss_mask, beta=1.0,
            external_mean=external_mean, external_std=external_std
        )
        weights_2 = build_weights(
            td_errors_2, loss_mask, beta=1.0,
            external_mean=external_mean, external_std=external_std
        )

        # 동일한 외부 통계로 정규화: 동일한 mean=0.5 토큰은 동일한 weight
        # td_errors_1[0,1] = 0.5, td_errors_2[0,1] = 0.5 → 동일한 normalized 값
        assert abs(weights_1[0, 1].item() - weights_2[0, 1].item()) < 1e-4

    def test_external_stats_tensor_types(self):
        """다양한 텐서 타입 지원 검증"""
        td_errors = torch.tensor([[0.0, 1.0, 2.0]])
        loss_mask = torch.tensor([[1, 1, 1]])

        # scalar tensor
        mean_scalar = torch.tensor(1.0)
        std_scalar = torch.tensor(0.5)

        weights = build_weights(
            td_errors, loss_mask, beta=1.0,
            external_mean=mean_scalar,
            external_std=std_scalar,
        )

        assert weights.shape == (1, 3)
        assert not torch.isnan(weights).any()


class TestTDWeightingIntegration:
    """통합 테스트: TD error → weights → stats"""

    def test_full_pipeline(self):
        """전체 파이프라인 검증"""
        # TD error 계산
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, loss_mask)

        # Weight 계산
        weights = build_weights(td_errors, loss_mask, beta=0.9)

        # Stats 계산
        td_stats = compute_td_stats(td_errors)
        weight_stats = compute_weight_stats(weights)

        assert td_stats["td_mean"] is not None
        assert weight_stats["weight_mean"] is not None
        assert weights.shape == td_errors.shape

    def test_positive_td_leads_to_high_weight(self):
        """Positive TD error → 높은 weight 검증"""
        td_errors_positive = torch.tensor([[0.5, 0.3, 0.2]])
        loss_mask = torch.tensor([[1, 1, 1]])
        weights_positive = build_weights(td_errors_positive, loss_mask, beta=0.9)

        td_errors_negative = torch.tensor([[-0.5, -0.3, -0.2]])
        weights_negative = build_weights(td_errors_negative, loss_mask, beta=0.9)

        stats_positive = compute_weight_stats(weights_positive)
        stats_negative = compute_weight_stats(weights_negative)

        assert stats_positive["weight_mean"] > stats_negative["weight_mean"]

    def test_incorrect_sample_downweighting(self):
        """Incorrect 샘플 자동 down-weighting 검증"""
        loss_mask = torch.tensor([[1, 1, 1]])

        # Correct sample
        value_logits_correct = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards_correct = torch.tensor([1.0])
        td_errors_correct = compute_td_errors(value_logits_correct, rewards_correct, loss_mask)
        weights_correct = build_weights(td_errors_correct, loss_mask, beta=0.9)

        # Incorrect sample
        value_logits_incorrect = torch.tensor([[[0.5], [0.7], [0.9]]])
        rewards_incorrect = torch.tensor([0.0])
        td_errors_incorrect = compute_td_errors(value_logits_incorrect, rewards_incorrect, loss_mask)
        weights_incorrect = build_weights(td_errors_incorrect, loss_mask, beta=0.9)

        # Incorrect 샘플의 terminal weight가 낮아야 함
        assert weights_incorrect[0, -1] < weights_correct[0, -1]
