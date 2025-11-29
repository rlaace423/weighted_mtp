"""TD Stats EMA Unit Tests

td_stats_ema.py의 TDStatsEMA 클래스 테스트:
- EMA 초기화 및 업데이트
- Warmup 동작
- state_dict / load_state_dict
- 분산 학습 동기화 (단일 프로세스 모드)
"""

import pytest
import torch

from weighted_mtp.value_weighting.td_stats_ema import TDStatsEMA


class TestTDStatsEMAInitialization:
    """TDStatsEMA 초기화 테스트"""

    def test_default_initialization(self):
        """기본 초기화 검증"""
        device = torch.device("cpu")
        ema = TDStatsEMA(device)

        assert ema.device == device
        assert ema.momentum == 0.1
        assert ema.warmup_steps == 10
        assert ema.step_count == 0

        # 초기 EMA 값
        assert ema.ema_mean.item() == 0.0
        assert ema.ema_std.item() == 1.0

    def test_custom_parameters(self):
        """커스텀 파라미터 검증"""
        ema = TDStatsEMA(
            device=torch.device("cpu"),
            momentum=0.2,
            warmup_steps=5,
        )

        assert ema.momentum == 0.2
        assert ema.warmup_steps == 5


class TestGetStats:
    """get_stats() 함수 테스트"""

    def test_initial_stats(self):
        """초기 통계 반환 검증"""
        ema = TDStatsEMA(torch.device("cpu"))

        mean, std = ema.get_stats()

        assert mean.item() == 0.0
        assert std.item() == 1.0

    def test_stats_after_update(self):
        """업데이트 후 통계 반환 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)

        td_errors = torch.tensor([[1.0, 2.0, 3.0]])
        loss_mask = torch.tensor([[1, 1, 1]])
        ema.update(td_errors, loss_mask, distributed=False)

        mean, std = ema.get_stats()

        # warmup_steps=1 이후 업데이트됨
        assert mean.item() != 0.0


class TestUpdate:
    """update() 함수 테스트"""

    def test_single_update(self):
        """단일 업데이트 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)

        td_errors = torch.tensor([[0.0, 1.0, 2.0]])  # mean=1.0, std=sqrt(2/3)
        loss_mask = torch.tensor([[1, 1, 1]])

        ema.update(td_errors, loss_mask, distributed=False)

        assert ema.step_count == 1
        # warmup 첫 스텝: 현재 batch 통계로 초기화
        assert abs(ema.ema_mean.item() - 1.0) < 1e-4

    def test_warmup_progression(self):
        """Warmup 기간 점진적 초기화 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=4)

        # Step 1: warmup_weight = (1-1)/4 = 0 → 100% batch
        td_errors_1 = torch.tensor([[1.0, 1.0, 1.0]])  # mean=1.0
        loss_mask = torch.tensor([[1, 1, 1]])
        ema.update(td_errors_1, loss_mask, distributed=False)

        # ema_mean = 0.0 * 0.0 + 1.0 * 1.0 = 1.0
        assert abs(ema.ema_mean.item() - 1.0) < 1e-4

        # Step 2: warmup_weight = (2-1)/4 = 0.25
        td_errors_2 = torch.tensor([[2.0, 2.0, 2.0]])  # mean=2.0
        ema.update(td_errors_2, loss_mask, distributed=False)

        # ema_mean = 0.25 * 1.0 + 0.75 * 2.0 = 1.75
        assert abs(ema.ema_mean.item() - 1.75) < 1e-4

    def test_ema_update_after_warmup(self):
        """Warmup 이후 일반 EMA 업데이트 검증"""
        ema = TDStatsEMA(
            torch.device("cpu"),
            momentum=0.1,
            warmup_steps=2,
        )
        loss_mask = torch.tensor([[1, 1, 1]])

        # Warmup 완료
        # step 1: warmup_weight=0, ema=1.0
        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)
        # step 2: warmup_weight=0.5, ema=0.5*1.0 + 0.5*1.0 = 1.0
        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)

        # 3번째 업데이트 (일반 EMA)
        ema.update(torch.tensor([[2.0, 2.0, 2.0]]), loss_mask, distributed=False)

        # momentum=0.1: ema = 0.9 * old + 0.1 * new
        # old = 1.0, new = 2.0 → ema = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        assert ema.step_count == 3
        assert abs(ema.ema_mean.item() - 1.1) < 1e-4

    def test_skip_update_on_empty_mask(self):
        """유효 토큰 없을 때 업데이트 스킵 검증"""
        ema = TDStatsEMA(torch.device("cpu"))

        td_errors = torch.tensor([[1.0, 2.0, 3.0]])
        loss_mask = torch.tensor([[0, 0, 0]])  # 모든 토큰 masking

        ema.update(td_errors, loss_mask, distributed=False)

        # 업데이트 스킵
        assert ema.step_count == 0
        assert ema.ema_mean.item() == 0.0

    def test_mask_applied_correctly(self):
        """마스크 적용 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)

        # 유효 토큰: [1.0, 2.0], mean=1.5
        td_errors = torch.tensor([[1.0, 2.0, 999.0]])
        loss_mask = torch.tensor([[1, 1, 0]])

        ema.update(td_errors, loss_mask, distributed=False)

        # 마스킹된 999.0은 제외됨
        assert abs(ema.ema_mean.item() - 1.5) < 1e-4


class TestStateDict:
    """state_dict() / load_state_dict() 테스트"""

    def test_state_dict_contents(self):
        """state_dict 내용 검증"""
        ema = TDStatsEMA(torch.device("cpu"), momentum=0.2, warmup_steps=5)

        # 몇 번 업데이트
        loss_mask = torch.tensor([[1, 1, 1]])
        ema.update(torch.tensor([[1.0, 2.0, 3.0]]), loss_mask, distributed=False)
        ema.update(torch.tensor([[2.0, 3.0, 4.0]]), loss_mask, distributed=False)

        state = ema.state_dict()

        assert "ema_mean" in state
        assert "ema_std" in state
        assert "step_count" in state
        assert "momentum" in state
        assert "warmup_steps" in state
        assert state["step_count"] == 2

    def test_load_state_dict(self):
        """state_dict 로드 검증"""
        ema1 = TDStatsEMA(torch.device("cpu"))

        # 업데이트
        loss_mask = torch.tensor([[1, 1, 1]])
        ema1.update(torch.tensor([[1.0, 2.0, 3.0]]), loss_mask, distributed=False)
        ema1.update(torch.tensor([[2.0, 3.0, 4.0]]), loss_mask, distributed=False)

        state = ema1.state_dict()

        # 새 인스턴스에 로드
        ema2 = TDStatsEMA(torch.device("cpu"))
        ema2.load_state_dict(state)

        assert ema2.step_count == ema1.step_count
        assert abs(ema2.ema_mean.item() - ema1.ema_mean.item()) < 1e-6
        assert abs(ema2.ema_std.item() - ema1.ema_std.item()) < 1e-6

    def test_checkpoint_resume(self):
        """Checkpoint 저장/복원 시나리오 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=2)
        loss_mask = torch.tensor([[1, 1, 1]])

        # Warmup 완료 직전 저장
        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)
        checkpoint = ema.state_dict()

        # 복원
        ema_restored = TDStatsEMA(torch.device("cpu"), warmup_steps=2)
        ema_restored.load_state_dict(checkpoint)

        # 추가 업데이트
        ema_restored.update(torch.tensor([[2.0, 2.0, 2.0]]), loss_mask, distributed=False)

        # 원래 EMA에도 동일 업데이트
        ema.update(torch.tensor([[2.0, 2.0, 2.0]]), loss_mask, distributed=False)

        # 결과 동일해야 함
        assert ema_restored.step_count == ema.step_count
        assert abs(ema_restored.ema_mean.item() - ema.ema_mean.item()) < 1e-6


class TestGetDebugStats:
    """get_debug_stats() 테스트"""

    def test_debug_stats_contents(self):
        """디버그 통계 내용 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=5)
        loss_mask = torch.tensor([[1, 1, 1]])

        # Warmup 기간
        ema.update(torch.tensor([[1.0, 2.0, 3.0]]), loss_mask, distributed=False)

        stats = ema.get_debug_stats()

        assert "ema_mean" in stats
        assert "ema_std" in stats
        assert "step_count" in stats
        assert "is_warmup" in stats
        assert stats["is_warmup"] is True

    def test_warmup_status(self):
        """Warmup 상태 추적 검증"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=2)
        loss_mask = torch.tensor([[1, 1, 1]])

        assert ema.get_debug_stats()["is_warmup"] is True

        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)
        assert ema.get_debug_stats()["is_warmup"] is True

        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)
        assert ema.get_debug_stats()["is_warmup"] is True  # step_count=2 <= warmup_steps=2

        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)
        assert ema.get_debug_stats()["is_warmup"] is False  # step_count=3 > warmup_steps=2


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_token(self):
        """단일 토큰 처리"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)

        td_errors = torch.tensor([[0.5]])
        loss_mask = torch.tensor([[1]])

        ema.update(td_errors, loss_mask, distributed=False)

        # 단일 토큰: mean=0.5, std=1.0 (기본값)
        assert abs(ema.ema_mean.item() - 0.5) < 1e-4
        assert ema.ema_std.item() == 1.0  # numel=1이면 std=1.0 반환

    def test_all_same_values(self):
        """모든 값이 동일한 경우"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)

        td_errors = torch.tensor([[0.5, 0.5, 0.5]])
        loss_mask = torch.tensor([[1, 1, 1]])

        ema.update(td_errors, loss_mask, distributed=False)

        # mean=0.5, std≈0
        assert abs(ema.ema_mean.item() - 0.5) < 1e-4
        assert ema.ema_std.item() < 1e-4

    def test_large_batch(self):
        """대용량 배치 처리"""
        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)

        td_errors = torch.randn(32, 2048)
        loss_mask = torch.ones(32, 2048, dtype=torch.long)

        ema.update(td_errors, loss_mask, distributed=False)

        # NaN/Inf 없음
        assert not torch.isnan(ema.ema_mean)
        assert not torch.isinf(ema.ema_mean)
        assert not torch.isnan(ema.ema_std)
        assert not torch.isinf(ema.ema_std)

    def test_momentum_zero(self):
        """momentum=0 (EMA 고정)"""
        ema = TDStatsEMA(torch.device("cpu"), momentum=0.0, warmup_steps=1)
        loss_mask = torch.tensor([[1, 1, 1]])

        # Warmup 완료
        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)
        initial_mean = ema.ema_mean.item()

        # 추가 업데이트 (momentum=0이면 변화 없음)
        ema.update(torch.tensor([[5.0, 5.0, 5.0]]), loss_mask, distributed=False)

        assert abs(ema.ema_mean.item() - initial_mean) < 1e-6

    def test_momentum_one(self):
        """momentum=1 (완전 교체)"""
        ema = TDStatsEMA(torch.device("cpu"), momentum=1.0, warmup_steps=1)
        loss_mask = torch.tensor([[1, 1, 1]])

        # Warmup 완료
        ema.update(torch.tensor([[1.0, 1.0, 1.0]]), loss_mask, distributed=False)

        # 추가 업데이트 (momentum=1이면 완전 교체)
        ema.update(torch.tensor([[5.0, 5.0, 5.0]]), loss_mask, distributed=False)

        assert abs(ema.ema_mean.item() - 5.0) < 1e-4


class TestIntegrationWithBuildWeights:
    """build_weights와의 통합 테스트"""

    def test_ema_stats_with_build_weights(self):
        """EMA 통계를 build_weights에 전달하는 시나리오"""
        from weighted_mtp.value_weighting.td_weighting import build_weights

        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=1)
        loss_mask = torch.tensor([[1, 1, 1]])

        # EMA 초기화
        td_errors_warmup = torch.tensor([[0.0, 0.5, 1.0]])
        ema.update(td_errors_warmup, loss_mask, distributed=False)

        # 새로운 batch의 TD errors
        td_errors_new = torch.tensor([[0.2, -0.3, 0.1]])

        # EMA 통계로 weights 계산
        ema_mean, ema_std = ema.get_stats()
        weights = build_weights(
            td_errors_new,
            loss_mask,
            beta=1.0,
            external_mean=ema_mean,
            external_std=ema_std,
        )

        # weights 유효성 검증
        assert weights.shape == td_errors_new.shape
        assert (weights >= 0.1).all()
        assert (weights <= 5.0).all()
        assert not torch.isnan(weights).any()

    def test_consistent_normalization_across_batches(self):
        """다중 배치에서 일관된 정규화 검증"""
        from weighted_mtp.value_weighting.td_weighting import build_weights

        ema = TDStatsEMA(torch.device("cpu"), warmup_steps=2)
        loss_mask = torch.tensor([[1, 1, 1]])

        # Warmup 배치들
        ema.update(torch.tensor([[0.1, 0.2, 0.3]]), loss_mask, distributed=False)
        ema.update(torch.tensor([[0.4, 0.5, 0.6]]), loss_mask, distributed=False)

        # 동일한 EMA 통계로 여러 배치 처리
        ema_mean, ema_std = ema.get_stats()

        td_errors_1 = torch.tensor([[0.2, 0.3, 0.4]])
        td_errors_2 = torch.tensor([[0.5, 0.6, 0.7]])

        weights_1 = build_weights(
            td_errors_1, loss_mask, beta=1.0,
            external_mean=ema_mean, external_std=ema_std
        )
        weights_2 = build_weights(
            td_errors_2, loss_mask, beta=1.0,
            external_mean=ema_mean, external_std=ema_std
        )

        # 두 배치 모두 동일한 정규화 기준 사용
        # td_errors_2가 더 크므로 weights_2가 더 커야 함
        assert weights_2.mean() > weights_1.mean()
