"""TD error 기반 가중치 계산

Modules:
- td_error: 표준 TD error 계산 (Intermediate + Terminal)
- weight_builder: Exponential weighting (IQL/AWR 방식)
- metrics: TD error/weight 모니터링
"""

from .td_error import compute_td_errors
from .weight_builder import build_weights
from .metrics import compute_weight_stats, compute_td_stats

__all__ = [
    "compute_td_errors",
    "build_weights",
    "compute_weight_stats",
    "compute_td_stats",
]
