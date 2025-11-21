"""Critic 분석 평가 지표

토큰 단위 value 분석 결과에 대한 정량적 평가 지표 계산
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_detection_metrics(results: list[dict]) -> dict:
    """샘플 그룹에 대한 탐지 통계 계산

    Args:
        results: analyze_sample_full() 결과 리스트

    Returns:
        {
            "n_samples": int,
            "drop_rate": float,
            "early_drop_rate": float,
            "mean_value": float,
            "std_value": float,
            "mean_max_drop": float,
            "mean_num_drops": float,
        }
    """
    metrics = {
        "n_samples": len(results),
        "drop_rate": 0.0,
        "early_drop_rate": 0.0,
        "mean_value": 0.0,
        "std_value": 0.0,
        "mean_max_drop": 0.0,
        "mean_num_drops": 0.0,
    }

    if len(results) == 0:
        return metrics

    # Drop rate: 급락이 있는 샘플 비율
    has_drop = [r.get("num_drops", 0) > 0 for r in results]
    metrics["drop_rate"] = float(np.mean(has_drop))

    # Early drop rate: 시퀀스 앞부분(30%)에서 급락이 있는 비율
    early_drops = []
    for r in results:
        seq_len = len(r["values"])
        early_threshold = seq_len * 0.3
        drop_indices = r.get("drop_indices", [])
        early_drop = any(idx < early_threshold for idx in drop_indices)
        early_drops.append(early_drop)
    metrics["early_drop_rate"] = float(np.mean(early_drops))

    # Value 통계
    all_mean_values = [r.get("mean_value", 0) for r in results]
    metrics["mean_value"] = float(np.mean(all_mean_values))
    metrics["std_value"] = float(np.std(all_mean_values))

    # Drop 통계
    max_drops = [r["max_drop"]["value"] for r in results if "max_drop" in r]
    if max_drops:
        metrics["mean_max_drop"] = float(np.mean(max_drops))

    num_drops = [r.get("num_drops", 0) for r in results]
    metrics["mean_num_drops"] = float(np.mean(num_drops))

    return metrics


def compare_groups(
    correct_results: list[dict],
    incorrect_results: list[dict],
) -> dict:
    """정답 vs 오답 그룹 비교

    Args:
        correct_results: 정답 샘플 결과 리스트
        incorrect_results: 오답 샘플 결과 리스트

    Returns:
        {
            "correct": dict,
            "incorrect": dict,
            "false_positive_rate": float,
            "discrimination": float,
            "value_gap": float,
        }
    """
    metrics = {
        "correct": compute_detection_metrics(correct_results),
        "incorrect": compute_detection_metrics(incorrect_results),
    }

    # False positive rate: 정답 코드에서 급락이 있는 비율
    metrics["false_positive_rate"] = metrics["correct"]["drop_rate"]

    # Discrimination: 오답과 정답의 drop rate 차이
    metrics["discrimination"] = (
        metrics["incorrect"]["drop_rate"] - metrics["correct"]["drop_rate"]
    )

    # Value gap: 정답과 오답의 mean value 차이
    metrics["value_gap"] = (
        metrics["correct"]["mean_value"] - metrics["incorrect"]["mean_value"]
    )

    return metrics


def compute_position_metrics(results: list[dict]) -> dict:
    """위치별 value 통계

    Args:
        results: analyze_sample_full() 결과 리스트

    Returns:
        {
            "start_mean": float,
            "middle_mean": float,
            "end_mean": float,
            "trajectory_slope": float,
        }
    """
    if not results:
        return {
            "start_mean": 0.0,
            "middle_mean": 0.0,
            "end_mean": 0.0,
            "trajectory_slope": 0.0,
        }

    start_values = []
    middle_values = []
    end_values = []
    slopes = []

    for r in results:
        values = r["values"]
        seq_len = len(values)
        if seq_len < 3:
            continue

        # 구간 분할
        start_idx = seq_len // 5
        end_idx = seq_len - seq_len // 5

        start_values.extend(values[:start_idx])
        middle_values.extend(values[start_idx:end_idx])
        end_values.extend(values[end_idx:])

        # 기울기 계산 (선형 회귀 대신 단순 차이)
        slope = (np.mean(values[end_idx:]) - np.mean(values[:start_idx])) / seq_len
        slopes.append(slope)

    return {
        "start_mean": float(np.mean(start_values)) if start_values else 0.0,
        "middle_mean": float(np.mean(middle_values)) if middle_values else 0.0,
        "end_mean": float(np.mean(end_values)) if end_values else 0.0,
        "trajectory_slope": float(np.mean(slopes)) if slopes else 0.0,
    }


def generate_report(results: list[dict]) -> dict:
    """전체 분석 리포트 생성

    Args:
        results: analyze_sample_full() 결과 리스트

    Returns:
        전체 평가 지표를 포함한 리포트
    """
    # 정답/오답 분리
    correct_results = [r for r in results if r.get("is_correct", False)]
    incorrect_results = [r for r in results if not r.get("is_correct", False)]

    report = {
        "summary": {
            "total_samples": len(results),
            "correct_samples": len(correct_results),
            "incorrect_samples": len(incorrect_results),
        },
        "group_comparison": compare_groups(correct_results, incorrect_results),
        "position_metrics": {
            "correct": compute_position_metrics(correct_results),
            "incorrect": compute_position_metrics(incorrect_results),
        },
    }

    # 평가 요약
    if incorrect_results and correct_results:
        report["evaluation"] = {
            "drop_discrimination": report["group_comparison"]["discrimination"],
            "value_gap": report["group_comparison"]["value_gap"],
            "false_positive_rate": report["group_comparison"]["false_positive_rate"],
            "is_effective": (
                report["group_comparison"]["discrimination"] > 0.3
                and report["group_comparison"]["value_gap"] > 0.1
            ),
        }

    return report
