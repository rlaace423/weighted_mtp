"""Code Evaluation 유틸리티

생성된 코드의 execution-based 평가 및 Pass@K 메트릭 계산
GSM8K 등 Math 벤치마크의 exact match 평가 지원
"""

import re
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


def execute_mbpp_tests(
    code: str,
    test_list: list[str],
    test_setup_code: str = "",
    timeout: int = 5,
) -> bool:
    """MBPP 형식의 테스트 실행

    Args:
        code: 생성된 함수 코드
        test_list: assert 문 리스트 (예: ["assert func(x) == y", ...])
        test_setup_code: 테스트 실행 전 설정 코드
        timeout: 실행 제한 시간 (초)

    Returns:
        정답 여부 (True=pass, False=fail)
    """
    # 테스트 코드 조합
    test_code = "\n".join(test_list)
    full_code = f"{test_setup_code}\n{code}\n\n{test_code}\n"

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        passed = (result.returncode == 0)

    except subprocess.TimeoutExpired:
        passed = False
    except Exception:
        passed = False
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return passed


def extract_gsm8k_answer(text: str) -> str | None:
    """GSM8K 형식의 텍스트에서 최종 답 추출

    Args:
        text: 생성된 텍스트 (풀이 과정 포함)

    Returns:
        추출된 숫자 문자열 또는 None
    """
    # 1순위: #### 뒤의 숫자
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')

    # 2순위: "answer is" 뒤의 숫자
    match = re.search(r'answer\s+is\s*:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # 3순위: 마지막 숫자
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def evaluate_gsm8k_answer(
    generated_text: str,
    ground_truth: str,
) -> bool:
    """GSM8K 답변 평가 (exact match)

    Args:
        generated_text: 모델이 생성한 텍스트
        ground_truth: 정답 숫자 문자열

    Returns:
        정답 여부 (True=correct, False=incorrect)
    """
    extracted = extract_gsm8k_answer(generated_text)
    if extracted is None:
        return False

    # 숫자 비교 (정수/실수 모두 처리)
    try:
        extracted_num = float(extracted)
        ground_truth_num = float(ground_truth)
        # 부동소수점 비교 허용 오차
        return abs(extracted_num - ground_truth_num) < 1e-6
    except ValueError:
        # 숫자 변환 실패 시 문자열 비교
        return extracted.strip() == ground_truth.strip()


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
