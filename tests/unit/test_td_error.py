"""TD Error 계산 Unit Tests"""

import pytest
import torch

from weighted_mtp.value_weighting.td_error import compute_td_errors


class TestComputeTDErrors:
    """compute_td_errors() 함수 테스트"""

    def test_basic_computation(self):
        """기본 TD error 계산 검증"""
        # 단일 시퀀스, 모든 토큰 유효
        value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
        rewards = torch.tensor([1.0])  # Correct
        attention_mask = torch.tensor([[1, 1, 1]])  # All valid
        gamma = 1.0

        td_errors = compute_td_errors(value_logits, rewards, attention_mask, gamma)

        # 예상 출력
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

        # Intermediate (0→1): 1.0 * 0.7 - 0.5 = 0.2
        # Intermediate (1→2): 1.0 * 0.9 - 0.7 = 0.2
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
        # Intermediate (0→1): 1.0 * 0.7 - 0.5 = 0.2
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
        # Sample 2 (incorrect)
        # Intermediate: (0.6-0.3, 0.8-0.6) = (0.3, 0.2)
        # Terminal: 0.0 - 0.8 = -0.8
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
        # Terminal (2): 1.0 - 0.9 = 0.1 (gamma는 terminal에 영향 없음)
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

        # TD error로 loss 계산
        loss = td_errors.pow(2).mean()
        loss.backward()

        # Gradient가 계산되었는지 확인
        assert value_logits.grad is not None
        assert value_logits.grad.shape == value_logits.shape

    def test_td_error_bounded(self):
        """Binary reward 환경에서 TD error bounded 검증"""
        # Extreme values
        value_logits = torch.tensor([[[0.0], [0.5], [1.0]]])
        rewards = torch.tensor([1.0])
        attention_mask = torch.tensor([[1, 1, 1]])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # TD error는 [-1, 1] 범위 내에 있어야 함
        # (Binary reward [0, 1], value unbounded이지만 실제로는 bounded)
        assert td_errors.abs().max() <= 2.0  # 보수적 상한

    def test_batch_with_different_lengths(self):
        """배치 내 다른 길이 시퀀스 처리 검증"""
        value_logits = torch.tensor([
            [[0.5], [0.7], [0.9], [0.0]],  # Length 3 (마지막 padding)
            [[0.3], [0.6], [0.0], [0.0]],  # Length 2 (마지막 2개 padding)
        ])  # [2, 4, 1]
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])

        td_errors = compute_td_errors(value_logits, rewards, attention_mask)

        # Sample 1: terminal index = 2
        # Intermediate: (0.7-0.5, 0.9-0.7) = (0.2, 0.2)
        # Terminal: 1.0 - 0.9 = 0.1
        expected_s1 = torch.tensor([0.2, 0.2, 0.1, 0.0])

        # Sample 2: terminal index = 1
        # Intermediate: (0.6-0.3) = 0.3
        # Terminal: 0.0 - 0.6 = -0.6
        expected_s2 = torch.tensor([0.3, -0.6, 0.0, 0.0])

        assert td_errors.shape == (2, 4)
        torch.testing.assert_close(td_errors[0], expected_s1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(td_errors[1], expected_s2, rtol=1e-4, atol=1e-4)
