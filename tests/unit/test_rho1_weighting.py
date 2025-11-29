"""Rho-1 Weighting 단위 테스트

compute_mtp_selective_weights 함수의 masking, padding, 토큰 매핑 검증
"""

import pytest
import torch

from weighted_mtp.value_weighting.rho1_weighting import (
    compute_mtp_selective_weights,
    compute_rho1_stats,
)


class TestComputeMTPSelectiveWeights:
    """compute_mtp_selective_weights 함수 테스트"""

    def test_basic_shape(self):
        """기본 출력 shape 검증"""
        batch_size = 2
        seq_len = 10
        n_future = 4
        vocab_size = 100

        policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = (labels != -100)  # 모두 True (유효한 토큰)

        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            loss_mask=loss_mask,
            k_percent=0.6,
        )

        assert weights.shape == (batch_size, seq_len, n_future)
        assert "selection_ratio" in stats
        assert "head_0_count" in stats  # compute_mtp_selective_weights는 count 반환

    def test_head_0_selection(self):
        """Head 0 (t+1)도 Rho-1 selection 적용"""
        batch_size = 2
        seq_len = 10
        n_future = 4
        vocab_size = 100

        policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = (labels != -100)  # 모두 True

        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            loss_mask=loss_mask,
            k_percent=0.6,
        )

        # Head 0도 k_percent에 따라 선택됨 (약 60%)
        # valid_len = seq_len - 1 = 9
        valid_len = seq_len - 1
        head_0_selected = weights[:, :valid_len, 0].sum().item()
        total_valid = loss_mask[:, 1:1+valid_len].sum().item()  # head 0은 labels[:, 1:1+valid_len] 참조
        selection_ratio = head_0_selected / total_valid if total_valid > 0 else 0
        assert 0.4 < selection_ratio < 0.8, f"Head 0 selection ratio {selection_ratio:.2f} not in expected range"
        assert "head_0_count" in stats

    def test_padding_handling(self):
        """Padding 토큰 (labels=-100) 제외 검증"""
        batch_size = 2
        seq_len = 10
        n_future = 4
        vocab_size = 100

        policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 마지막 3개 토큰은 padding (labels=-100)
        labels[:, -3:] = -100
        loss_mask = (labels != -100)

        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            loss_mask=loss_mask,
            k_percent=0.6,
        )

        # Padding 위치는 weight=0
        assert (weights[:, -3:, 0] == 0).all()

    def test_ignore_index_handling(self):
        """labels=-100 토큰 제외 검증"""
        batch_size = 2
        seq_len = 10
        n_future = 4
        vocab_size = 100

        policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 처음 3개 토큰은 instruction (labels=-100)
        labels[:, :3] = -100
        loss_mask = (labels != -100)

        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            loss_mask=loss_mask,
            k_percent=0.6,
        )

        # labels=-100인 위치는 head 1,2,3에서 weight=0이어야 함
        # head k의 경우 labels[:, k:k+valid_len]을 사용하므로 확인
        # head 1 (k=2): labels[:, 2:valid_len+2], 처음 3개가 -100이면
        # labels[:, 2]는 -100, labels[:, 3:]부터 유효

        # 실제로 head 1에서 labels=-100인 토큰이 0인지 확인
        # valid_len = seq_len - 2 = 8
        # labels_k = labels[:, 2:10] = labels의 인덱스 2~9
        # labels[:, 2] = -100이므로 weights[:, 0, 1] = 0이어야 함
        assert (weights[:, 0, 1] == 0).all(), "labels=-100 위치의 head 1 weight가 0이 아님"

    def test_top_k_selection(self):
        """Top-k selection이 올바르게 동작하는지 검증"""
        batch_size = 1
        seq_len = 10
        n_future = 4
        vocab_size = 100

        torch.manual_seed(42)  # 재현성
        policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = (labels != -100)  # 모두 True

        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            loss_mask=loss_mask,
            k_percent=0.5,
        )

        # selection_ratio가 k_percent 근처인지 확인
        # k_percent=0.5이므로 약 50%가 선택되어야 함
        selection_ratio = stats["selection_ratio"]
        assert 0.3 < selection_ratio < 0.7, f"Selection ratio {selection_ratio} not in expected range"

        # 각 head의 count가 기록되어야 함
        assert "head_0_count" in stats
        assert "head_1_count" in stats

    def test_token_position_mapping(self):
        """MTP head와 label의 토큰 위치 매핑 검증

        Head k (k=1,2,3,4)는 position t에서 t+k를 예측
        - policy_logits[:, t, k-1, :] -> predicts token at position t+k
        - labels[:, t+k] -> ground truth
        """
        batch_size = 1
        seq_len = 8
        n_future = 4
        vocab_size = 10

        policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = (labels != -100)  # 모두 True

        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            loss_mask=loss_mask,
            k_percent=0.6,
        )

        # Head 1 (k=2): valid_len = seq_len - 2 = 6
        # weights[:, 0:6, 1]이 유효
        assert weights.shape[1] == seq_len
        # 마지막 2 위치는 head 1에서 0이어야 함 (valid_len 초과)
        # 실제로는 초기화가 0이므로 확인

        # Head 3 (k=4): valid_len = seq_len - 4 = 4
        # weights[:, 0:4, 3]이 유효
        # 위치 4~7은 0이어야 함
        assert (weights[:, 4:, 3] == 0).all()


class TestComputeRho1Stats:
    """compute_rho1_stats 함수 테스트"""

    def test_basic_stats(self):
        """기본 통계 계산 검증"""
        batch_size = 2
        seq_len = 10
        n_future = 4

        # 모든 weight가 1인 경우
        weights = torch.ones(batch_size, seq_len, n_future)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        stats = compute_rho1_stats(weights, loss_mask)

        assert stats["selection_ratio"] == pytest.approx(1.0, rel=0.01)
        assert stats["avg_heads_per_position"] == pytest.approx(n_future, rel=0.01)
        assert stats["head_0_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_partial_selection(self):
        """부분 선택 통계 검증"""
        batch_size = 2
        seq_len = 10
        n_future = 4

        # Head 0만 선택된 경우
        weights = torch.zeros(batch_size, seq_len, n_future)
        weights[:, :, 0] = 1.0
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        stats = compute_rho1_stats(weights, loss_mask)

        assert stats["selection_ratio"] == pytest.approx(0.25, rel=0.01)  # 1/4
        assert stats["avg_heads_per_position"] == pytest.approx(1.0, rel=0.01)
        assert stats["head_0_ratio"] == pytest.approx(1.0, rel=0.01)
        assert stats["head_1_ratio"] == pytest.approx(0.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
