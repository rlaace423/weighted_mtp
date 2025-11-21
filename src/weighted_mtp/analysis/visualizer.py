"""토큰 단위 Value 시각화

Value trajectory와 gradient를 시각화하여 에러 인지 패턴 분석
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_single_sample(
    result: dict,
    save_path: Path,
    figsize: tuple[int, int] = (15, 8),
) -> None:
    """단일 샘플의 value 및 gradient 시각화

    Args:
        result: analyze_sample_full() 결과
        save_path: 저장 경로
        figsize: 그림 크기
    """
    tokens = result["tokens"]
    values = result["values"]
    gradient = result.get("gradient", [])
    drop_indices = result.get("drop_indices", [])
    is_correct = result.get("is_correct", False)

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 1. Value plot
    ax1 = axes[0]
    positions = np.arange(len(values))
    ax1.plot(positions, values, marker="o", markersize=2, linewidth=1, label="Value")

    # 급락 지점 표시
    if drop_indices:
        drop_values = [values[i] for i in drop_indices if i < len(values)]
        ax1.scatter(
            drop_indices[:len(drop_values)],
            drop_values,
            color="red",
            s=50,
            label="Drop points",
            zorder=5,
        )

    ax1.set_ylabel("Value")
    ax1.set_title(f"Token-level Values (is_correct={is_correct})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gradient plot
    ax2 = axes[1]
    if len(gradient) > 0:
        ax2.plot(
            gradient,
            marker="o",
            markersize=2,
            linewidth=1,
            color="orange",
            label="dV/dt",
        )
        ax2.axhline(y=-0.1, color="r", linestyle="--", alpha=0.7, label="Drop threshold")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    ax2.set_ylabel("Value Gradient")
    ax2.set_xlabel("Token Position")
    ax2.set_title("Value Changes (Gradient)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Plot 저장: {save_path}")


def plot_multiple_samples(
    results: list[dict],
    save_path: Path,
    max_samples: int = 10,
    figsize: tuple[int, int] = (15, 8),
) -> None:
    """여러 샘플의 value trajectory 비교

    Args:
        results: analyze_sample_full() 결과 리스트
        save_path: 저장 경로
        max_samples: 최대 표시할 샘플 수
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)

    for i, result in enumerate(results[:max_samples]):
        values = result["values"]
        is_correct = result.get("is_correct", False)
        label = f"Sample {i} ({'C' if is_correct else 'I'})"
        plt.plot(values, alpha=0.6, label=label)

    plt.ylabel("Value")
    plt.xlabel("Token Position")
    plt.title("Multiple Sample Comparison")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Comparison plot 저장: {save_path}")


def plot_correct_vs_incorrect(
    correct_results: list[dict],
    incorrect_results: list[dict],
    save_path: Path,
    max_samples: int = 5,
    figsize: tuple[int, int] = (18, 6),
) -> None:
    """정답 vs 오답 코드의 value 패턴 비교

    Args:
        correct_results: 정답 샘플 결과 리스트
        incorrect_results: 오답 샘플 결과 리스트
        save_path: 저장 경로
        max_samples: 각 그룹별 최대 표시할 샘플 수
        figsize: 그림 크기
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 정답 코드
    for i, result in enumerate(correct_results[:max_samples]):
        ax1.plot(result["values"], alpha=0.6, label=f"Correct {i}")
    ax1.set_title("Correct Codes")
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Token Position")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 오답 코드
    for i, result in enumerate(incorrect_results[:max_samples]):
        ax2.plot(result["values"], alpha=0.6, label=f"Incorrect {i}")
    ax2.set_title("Incorrect Codes")
    ax2.set_xlabel("Token Position")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Correct vs Incorrect plot 저장: {save_path}")


def plot_value_distribution(
    results: list[dict],
    save_path: Path,
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """Value 분포 히스토그램

    Args:
        results: analyze_sample_full() 결과 리스트
        save_path: 저장 경로
        figsize: 그림 크기
    """
    # 정답/오답 분리
    correct_values = []
    incorrect_values = []

    for result in results:
        if result.get("is_correct", False):
            correct_values.extend(result["values"])
        else:
            incorrect_values.extend(result["values"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 전체 분포
    if correct_values:
        ax1.hist(correct_values, bins=50, alpha=0.7, label="Correct", color="green")
    if incorrect_values:
        ax1.hist(incorrect_values, bins=50, alpha=0.7, label="Incorrect", color="red")
    ax1.set_title("Value Distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mean value 분포
    correct_means = [r["mean_value"] for r in results if r.get("is_correct", False)]
    incorrect_means = [r["mean_value"] for r in results if not r.get("is_correct", False)]

    if correct_means:
        ax2.hist(correct_means, bins=30, alpha=0.7, label="Correct", color="green")
    if incorrect_means:
        ax2.hist(incorrect_means, bins=30, alpha=0.7, label="Incorrect", color="red")
    ax2.set_title("Mean Value per Sample")
    ax2.set_xlabel("Mean Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Distribution plot 저장: {save_path}")


def plot_drop_analysis(
    results: list[dict],
    save_path: Path,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """급락 지점 분석 시각화

    Args:
        results: analyze_sample_full() 결과 리스트
        save_path: 저장 경로
        figsize: 그림 크기
    """
    # 오답 샘플만 분석
    incorrect_results = [r for r in results if not r.get("is_correct", False)]

    if not incorrect_results:
        logger.warning("오답 샘플이 없어 급락 분석을 건너뜁니다.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # 1. 급락 횟수 분포
    num_drops = [r.get("num_drops", 0) for r in incorrect_results]
    ax1.hist(num_drops, bins=range(max(num_drops) + 2), alpha=0.7, color="red", edgecolor="black")
    ax1.set_title("Number of Drops per Sample")
    ax1.set_xlabel("Number of Drops")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # 2. 최대 급락 크기 분포
    max_drops = [r["max_drop"]["value"] for r in incorrect_results if "max_drop" in r]
    ax2.hist(max_drops, bins=30, alpha=0.7, color="orange", edgecolor="black")
    ax2.set_title("Max Drop Value Distribution")
    ax2.set_xlabel("Max Drop Value")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # 3. 최대 급락 위치 분포 (정규화)
    drop_positions = []
    for r in incorrect_results:
        if "max_drop" in r and len(r["values"]) > 0:
            pos = r["max_drop"]["position"]
            normalized_pos = pos / len(r["values"])
            drop_positions.append(normalized_pos)

    ax3.hist(drop_positions, bins=20, alpha=0.7, color="purple", edgecolor="black")
    ax3.set_title("Max Drop Position (Normalized)")
    ax3.set_xlabel("Position (0=start, 1=end)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Drop analysis plot 저장: {save_path}")
