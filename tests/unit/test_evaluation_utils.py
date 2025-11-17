"""Evaluation Utils Unit Tests"""

import pytest

from weighted_mtp.utils.evaluation_utils import (
    compute_pass_at_k,
    evaluate_pass_at_k,
    execute_code_with_tests,
)


def test_execute_code_correct():
    """정답 코드 실행 테스트"""
    code = """
def add(a, b):
    return a + b
"""
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
"""
    passed = execute_code_with_tests(code, test, "add", timeout=5)
    assert passed is True


def test_execute_code_incorrect():
    """오답 코드 실행 테스트"""
    code = """
def add(a, b):
    return a - b  # 잘못된 구현
"""
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
    passed = execute_code_with_tests(code, test, "add", timeout=5)
    assert passed is False


def test_execute_code_syntax_error():
    """Syntax error 코드 테스트"""
    code = """
def add(a, b)
    return a + b  # Missing colon
"""
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
    passed = execute_code_with_tests(code, test, "add", timeout=5)
    assert passed is False


def test_execute_code_runtime_error():
    """Runtime error 코드 테스트"""
    code = """
def add(a, b):
    return a / 0  # Division by zero
"""
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
    passed = execute_code_with_tests(code, test, "add", timeout=5)
    assert passed is False


def test_execute_code_timeout():
    """Timeout 테스트"""
    code = """
def add(a, b):
    while True:  # Infinite loop
        pass
    return a + b
"""
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
    passed = execute_code_with_tests(code, test, "add", timeout=1)
    assert passed is False


def test_compute_pass_at_k_basic():
    """Pass@K 기본 계산 테스트"""
    # n=10, c=3 (30% 정답)
    pass_at_1 = compute_pass_at_k(n=10, c=3, k=1)
    assert abs(pass_at_1 - 0.3) < 0.01  # ~30%

    pass_at_5 = compute_pass_at_k(n=10, c=3, k=5)
    assert abs(pass_at_5 - 0.9167) < 0.01  # ~91.67%

    pass_at_10 = compute_pass_at_k(n=10, c=3, k=10)
    assert pass_at_10 == 1.0  # 100% (k=n이므로)


def test_compute_pass_at_k_edge_cases():
    """Pass@K 경계 케이스 테스트"""
    # 모든 샘플이 정답
    assert compute_pass_at_k(n=10, c=10, k=1) == 1.0
    assert compute_pass_at_k(n=10, c=10, k=5) == 1.0

    # 모든 샘플이 오답
    assert compute_pass_at_k(n=10, c=0, k=1) == 0.0
    assert compute_pass_at_k(n=10, c=0, k=5) == 0.0

    # k > n-c (정답 보장)
    assert compute_pass_at_k(n=10, c=8, k=5) == 1.0


def test_evaluate_pass_at_k():
    """Pass@K 메트릭 계산 테스트"""
    # n=10, c=3 (30% 정답)
    results = [True, True, True, False, False, False, False, False, False, False]
    metrics = evaluate_pass_at_k(results, k_values=[1, 5, 10])

    assert "pass@1" in metrics
    assert "pass@5" in metrics
    assert "pass@10" in metrics

    assert abs(metrics["pass@1"] - 0.3) < 0.01
    assert abs(metrics["pass@5"] - 0.9167) < 0.01
    assert metrics["pass@10"] == 1.0


def test_evaluate_pass_at_k_skip_large_k():
    """Pass@K에서 k > n 케이스 skip 테스트"""
    results = [True, False, False]  # n=3
    metrics = evaluate_pass_at_k(results, k_values=[1, 5, 10])

    # k=1만 계산되어야 함 (k=5, 10은 skip)
    assert "pass@1" in metrics
    assert "pass@5" not in metrics
    assert "pass@10" not in metrics


def test_evaluate_pass_at_k_all_correct():
    """모든 샘플이 정답인 경우"""
    results = [True, True, True, True, True]
    metrics = evaluate_pass_at_k(results, k_values=[1, 5])

    assert metrics["pass@1"] == 1.0
    assert metrics["pass@5"] == 1.0


def test_evaluate_pass_at_k_all_incorrect():
    """모든 샘플이 오답인 경우"""
    results = [False, False, False, False, False]
    metrics = evaluate_pass_at_k(results, k_values=[1, 5])

    assert metrics["pass@1"] == 0.0
    assert metrics["pass@5"] == 0.0
