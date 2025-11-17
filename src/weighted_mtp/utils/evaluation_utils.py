"""Code Evaluation 유틸리티

생성된 코드의 execution-based 평가 및 Pass@K 메트릭 계산
"""

import subprocess
import tempfile
from pathlib import Path

from scipy.special import comb


def execute_code_with_tests(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 5,
) -> bool:
    """생성된 코드 실행 및 테스트

    Args:
        code: 생성된 함수 코드
        test_code: 채점용 test case 코드
        entry_point: 함수 이름
        timeout: 실행 제한 시간 (초)

    Returns:
        정답 여부 (True=pass, False=fail)

    Examples:
        >>> code = '''
        ... def add(a, b):
        ...     return a + b
        ... '''
        >>> test = '''
        ... def check(candidate):
        ...     assert candidate(1, 2) == 3
        ... '''
        >>> execute_code_with_tests(code, test, "add", timeout=5)
        True
    """
    # 전체 코드 조합
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        # subprocess로 안전하게 실행
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        # 정상 종료 여부 확인
        passed = (result.returncode == 0)

    except subprocess.TimeoutExpired:
        # Timeout 발생 시 오답 처리
        passed = False
    except Exception:
        # 기타 예외 발생 시 오답 처리
        passed = False
    finally:
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)

    return passed


def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """Pass@K 메트릭 계산 (unbiased estimator)

    Args:
        n: 총 생성 개수
        c: 정답 개수
        k: 평가할 개수

    Returns:
        Pass@K 확률 [0.0, 1.0]

    Formula:
        Pass@K = 1 - C(n-c, k) / C(n, k)

    Interpretation:
        n개 중 k개를 무작위로 선택했을 때, 최소 1개가 정답일 확률

    References:
        Chen et al. (2021) "Evaluating Large Language Models Trained on Code"

    Examples:
        >>> compute_pass_at_k(n=10, c=3, k=1)  # 30%
        0.3
        >>> compute_pass_at_k(n=10, c=3, k=5)  # ~83%
        0.8333...
    """
    # k가 n보다 크면 모든 샘플 선택 → 정답 보장
    if n - c < k:
        return 1.0

    # Pass@K 공식: 1 - (오답만 k개 선택할 확률)
    return 1.0 - float(comb(n - c, k, exact=True) / comb(n, k, exact=True))


def evaluate_pass_at_k(
    results: list[bool],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Pass@K 메트릭 계산

    Args:
        results: 샘플별 정답 여부 리스트 [True, False, True, ...]
        k_values: 계산할 K 값들

    Returns:
        {"pass@1": 0.2, "pass@5": 0.65, "pass@10": 0.85}

    Examples:
        >>> results = [True, True, False, False, False]
        >>> evaluate_pass_at_k(results, k_values=[1, 5])
        {'pass@1': 0.4, 'pass@5': 1.0}
    """
    n = len(results)
    c = sum(results)

    metrics = {}
    for k in k_values:
        # k가 n보다 크면 skip
        if k > n:
            continue
        metrics[f"pass@{k}"] = compute_pass_at_k(n, c, k)

    return metrics
