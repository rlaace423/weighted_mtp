"""가중치 계산 모듈

두 가지 Weighting 방식:
- td_weighting: TD Error 기반 (Verifiable WMTP용)
- rho1_weighting: Reference model 기반 (Rho-1 WMTP용)

Effective Batch 정규화:
- td_stats_ema: EMA 기반 TD Error 통계 추적 (BatchNorm과 유사한 running statistics)
"""

from .td_weighting import (
    build_weights as build_td_weights,
    compute_td_errors,
    compute_td_targets,
    compute_td_stats,
    compute_weight_stats,
)
from .td_stats_ema import TDStatsEMA
from .rho1_weighting import (
    compute_mtp_selective_weights,
    compute_rho1_stats,
)

__all__ = [
    # TD Error 기반 (Verifiable)
    "compute_td_errors",
    "compute_td_targets",
    "build_td_weights",
    "compute_td_stats",
    "compute_weight_stats",
    # TD Error EMA (Effective Batch 정규화)
    "TDStatsEMA",
    # Rho-1 기반
    "compute_mtp_selective_weights",
    "compute_rho1_stats",
]
