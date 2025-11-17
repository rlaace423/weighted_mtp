# Phase 7: SFT 평가 파이프라인 구축

## 1. Phase 7 개요

Phase 7은 **학습된 모델의 Code Generation 성능 평가 파이프라인 구축**을 담당한다. Meta MTP 논문의 SFT 평가 방식을 참고하여, HumanEval/MBPP 벤치마크에서 Pass@K 메트릭을 계산하고, MLflow로 결과를 추적한다.

### 1.1 Phase 7의 범위

```
Phase 6 (학습 파이프라인)  →  [Phase 7 (평가 파이프라인)]  →  Production 분석
 4개 pipeline 완성              Checkpoint 기반 평가          실험 결과 비교
 분산 학습 최적화              Pass@K 메트릭 계산
```

**구현 대상**:
- **Checkpoint 관리 체계**: 학습 중 저장된 checkpoint 구조 표준화
- **Inference 파이프라인**: MTP 기반 autoregressive generation
- **Pass@K 평가**: HumanEval/MBPP execution-based evaluation
- **MLflow 통합**: 평가 결과 자동 기록 및 비교

**평가 대상 모델**:
- Baseline MTP (Stage 0)
- Critic-pretrained MTP (Stage 1)
- Verifiable WMTP (Stage 2)
- Rho-1 Weighted MTP (Stage 3)

### 1.2 Phase 7 완료 후 달성될 상태

| 항목 | 구현 목표 |
|------|-----------|
| **Checkpoint 구조** | storage/checkpoints/{pipeline}/checkpoint_{best\|final\|epoch_N}.pt |
| **Checkpoint Metadata** | epoch, loss, hyperparams 포함 |
| **Generation 유틸리티** | MTP autoregressive generation (temperature, top-p) |
| **HumanEval 평가** | 164개 샘플 Pass@K 계산 (K=1,5,10) |
| **MBPP 평가** | 500개 샘플 Pass@K 계산 |
| **Execution Sandbox** | 안전한 code execution 환경 |
| **MLflow 기록** | 평가 메트릭, 생성 샘플, 비교 차트 |
| **CLI 통합** | `python -m weighted_mtp.cli.evaluate --checkpoint ... --dataset ...` |

---

## 2. Checkpoint 관리 전략 (사전 정의)

### 2.1 Checkpoint 저장 구조

Phase 6에서 이미 구축된 checkpoint 저장 방식을 활용한다.

**디렉터리 구조**:
```
storage/checkpoints/
├── baseline/
│   ├── checkpoint_best.pt       # Validation loss 기준 최고 성능
│   ├── checkpoint_final.pt      # 학습 종료 시점
│   └── checkpoint_epoch_10.pt   # 중간 checkpoint (옵션)
├── critic/
│   ├── checkpoint_best.pt
│   └── checkpoint_final.pt
├── verifiable/
│   ├── checkpoint_best.pt
│   └── checkpoint_final.pt
└── rho1/
    ├── checkpoint_best.pt
    └── checkpoint_final.pt
```

**Checkpoint 파일 포맷** (이미 구현됨):
```python
{
    'model_state_dict': unwrap_model(model).state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': int,
    'train_metrics': {
        'train_loss': float,
        'train_weighted_ce_loss': float,  # Verifiable/Rho1
        'train_value_loss': float,        # Verifiable
        ...
    },
    'val_metrics': {
        'val_loss': float,
        ...
    },
    'config': {
        'model': {...},
        'training': {...},
        'dataset': {...},
    }
}
```

### 2.2 Checkpoint 선택 전략

**기본 전략**:
- **checkpoint_best.pt**: Validation loss 기준 최고 성능 (기본 평가 대상)
- **checkpoint_final.pt**: 학습 종료 시점 (과적합 확인용)

**평가 시나리오**:
1. **단일 checkpoint 평가**: `--checkpoint storage/checkpoints/baseline/checkpoint_best.pt`
2. **Pipeline별 비교**: `--pipeline baseline critic verifiable rho1` (각각의 checkpoint_best.pt 자동 로드)
3. **Epoch별 성능 추적**: `--checkpoint-pattern checkpoint_epoch_*.pt` (학습 곡선 분석)

### 2.3 Checkpoint 로딩 함수

`src/utils/checkpoint_utils.py`에 이미 구현된 함수를 활용:

```python
def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Checkpoint 로드 (학습 재개 또는 평가용)

    Returns:
        checkpoint dict (epoch, metrics, config 포함)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
```

**평가용 로딩 헬퍼 추가 예정**:
```python
def load_checkpoint_for_evaluation(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MetaLlamaMTPAdapter, dict]:
    """평가용 checkpoint 로딩

    Returns:
        (model, checkpoint_metadata)
    """
    # Model 초기화 (checkpoint의 config 사용)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config['model']['path'],
        device=device,
        dtype=config['model']['dtype'],
        initialize_value_head=False,  # 평가 시 불필요
    )

    # State dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Evaluation mode

    return model, checkpoint
```

---

## 3. 평가 데이터셋 구조 및 로딩

### 3.1 현재 데이터셋 현황

**HumanEval** (`storage/datasets_v2/humaneval/processed/test.jsonl`):
```json
{
  "instruction": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
  "input": "",
  "output": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
  "task_id": "HumanEval/0",
  "metadata": {
    "source": "humaneval",
    "test": "\n\nMETADATA = {...}\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    ...",
    "entry_point": "has_close_elements",
    "has_tests": true
  }
}
```

**필드 설명**:
- `instruction`: 함수 시그니처 + docstring (프롬프트로 사용)
- `input`: 빈 문자열 (HumanEval은 input 없음)
- `output`: 정답 코드
- `task_id`: 고유 ID
- `metadata.test`: 채점용 test case 코드
- `metadata.entry_point`: 함수 이름

**데이터셋 규모**:
- HumanEval: test 164개
- MBPP: test 500개
- CodeContests: test 14,851개 (너무 많아 샘플링 필요)

### 3.2 평가 데이터 로딩

`src/data/datasets.py`에 평가용 로더 추가:

```python
def load_evaluation_dataset(
    dataset_name: str,
    split: str = "test",
) -> Dataset:
    """평가용 데이터셋 로드 (메타데이터 불필요, 전체 로드)

    Args:
        dataset_name: "humaneval", "mbpp", "codecontests"
        split: "test" (기본)

    Returns:
        HuggingFace Dataset (instruction, output, task_id, metadata 포함)
    """
    dataset_path = f"storage/datasets_v2/{dataset_name}/processed/{split}.jsonl"

    # JSONL 전체 로드 (평가는 샘플링 불필요)
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # CodeContests는 너무 많아 1000개 샘플링
    if dataset_name == "codecontests" and len(samples) > 1000:
        samples = random.sample(samples, 1000)

    return Dataset.from_list(samples)
```

---

## 4. Step-by-Step 구현 계획

### Step 1: Generation 유틸리티 구현

**목표**: MTP 모델로 autoregressive generation 수행

**구현 위치**: `src/utils/generation_utils.py` (신규)

**핵심 기능**:

```python
def generate_with_mtp(
    model: MetaLlamaMTPAdapter,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """MTP 모델로 autoregressive generation

    MTP는 n_future_tokens개를 예측하지만, generation 시에는
    head 0 (다음 토큰)만 사용하여 표준 autoregressive 방식으로 생성.

    Args:
        model: MetaLlamaMTPAdapter (eval mode)
        tokenizer: Tokenizer
        prompt: 생성 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        temperature: Sampling temperature (0=greedy, >0=sampling)
        top_p: Nucleus sampling threshold
        num_return_sequences: 생성할 시퀀스 개수 (Pass@K용)
        device: 디바이스

    Returns:
        생성된 텍스트 리스트 (길이 num_return_sequences)
    """
    model.eval()
    generated_texts = []

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(num_return_sequences):
        current_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass (MTP trunk)
            with torch.no_grad():
                outputs = model.trunk_forward(current_ids)
                logits = outputs["logits"]  # [batch, seq, n_future, vocab]

            # MTP head 0만 사용 (다음 토큰 예측)
            next_token_logits = logits[:, -1, 0, :]  # [batch, vocab]

            # Temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                next_token_logits[sorted_indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode
        generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts
```

**검증 기준**:
- Greedy decoding (temperature=0) 동작 확인
- Sampling (temperature=0.2~0.8) 다양성 확인
- MTP head 0만 사용하여 표준 LLM과 동일 결과
- EOS token 자동 정지
- Batch generation 정상 동작

**Unit Test** (`tests/unit/test_generation_utils.py`):
```python
def test_generate_with_mtp_greedy(micro_model, tokenizer):
    """Greedy decoding 테스트"""
    prompt = "def hello():\n"
    outputs = generate_with_mtp(
        model=micro_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.0,
        num_return_sequences=1,
    )
    assert len(outputs) == 1
    assert len(outputs[0]) > len(prompt)

def test_generate_with_mtp_sampling(micro_model, tokenizer):
    """Sampling 다양성 테스트"""
    prompt = "def hello():\n"
    outputs = generate_with_mtp(
        model=micro_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        num_return_sequences=5,
    )
    assert len(outputs) == 5
    # 다양성 확인 (5개가 모두 다를 확률 높음)
    assert len(set(outputs)) >= 3
```

---

### Step 2: Execution-based Evaluation

**목표**: 생성된 코드를 실행하여 정답 여부 판단

**구현 위치**: `src/utils/evaluation_utils.py` (신규)

**핵심 기능**:

```python
import subprocess
import tempfile
from pathlib import Path

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
        passed = False
    except Exception:
        passed = False
    finally:
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)

    return passed
```

**Pass@K 계산**:

```python
import math
from scipy.special import comb

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
        Pass@K 확률

    Formula:
        Pass@K = 1 - C(n-c, k) / C(n, k)

    Interpretation:
        n개 중 k개를 선택했을 때, 최소 1개가 정답일 확률
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

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
    """
    n = len(results)
    c = sum(results)

    metrics = {}
    for k in k_values:
        if k > n:
            continue
        metrics[f"pass@{k}"] = compute_pass_at_k(n, c, k)

    return metrics
```

**검증 기준**:
- Code execution 정상 동작
- Timeout 처리
- 예외 처리 (syntax error, runtime error)
- Pass@K 계산 정확성 (수식 검증)

**Unit Test** (`tests/unit/test_evaluation_utils.py`):
```python
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

def test_pass_at_k_calculation():
    """Pass@K 계산 검증"""
    # n=10, c=3 (30% 정답)
    results = [True, True, True, False, False, False, False, False, False, False]
    metrics = evaluate_pass_at_k(results, k_values=[1, 5, 10])

    assert 0.25 < metrics["pass@1"] < 0.35  # ~30%
    assert 0.75 < metrics["pass@5"] < 0.85  # ~80%
    assert metrics["pass@10"] == 1.0        # 100% (n=10이므로)
```

---

### Step 3: 평가 파이프라인 구현

**목표**: Checkpoint를 로드하여 벤치마크 평가 수행

**구현 위치**: `src/pipelines/run_evaluation.py` (신규)

**핵심 흐름**:

```python
def run_evaluation(
    checkpoint_path: str,
    dataset_name: str = "humaneval",
    num_samples_per_task: int = 20,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    device: str = "auto",
    mlflow_enabled: bool = True,
):
    """평가 파이프라인 실행

    Args:
        checkpoint_path: Checkpoint 경로
        dataset_name: "humaneval", "mbpp", "codecontests"
        num_samples_per_task: 각 문제당 생성 개수 (Pass@K 계산용)
        temperature: Sampling temperature
        max_new_tokens: 최대 생성 토큰 수
        device: 디바이스
        mlflow_enabled: MLflow 로깅 여부
    """
    # 1. Setup
    logger = setup_logging("EVALUATION")
    device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Samples per task: {num_samples_per_task}")

    # 2. Load checkpoint
    logger.info("Loading checkpoint...")
    model, checkpoint_metadata = load_checkpoint_for_evaluation(
        checkpoint_path=Path(checkpoint_path),
        device=device,
    )
    logger.info(f"Checkpoint epoch: {checkpoint_metadata['epoch']}")
    logger.info(f"Validation loss: {checkpoint_metadata['val_metrics']['val_loss']:.4f}")

    # 3. Load tokenizer
    tokenizer = load_tokenizer(checkpoint_metadata['config']['model']['path'])

    # 4. Load evaluation dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_evaluation_dataset(dataset_name, split="test")
    logger.info(f"Dataset size: {len(dataset)}")

    # 5. MLflow setup
    if mlflow_enabled:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.start_run(run_name=f"eval_{dataset_name}_{Path(checkpoint_path).stem}")
        mlflow.log_params({
            "checkpoint": checkpoint_path,
            "dataset": dataset_name,
            "num_samples_per_task": num_samples_per_task,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        })

    # 6. Evaluation loop
    logger.info("Starting evaluation...")
    all_results = []

    for sample in tqdm(dataset, desc="Evaluating"):
        task_id = sample["task_id"]
        prompt = sample["instruction"]
        reference_output = sample["output"]
        test_code = sample["metadata"]["test"]
        entry_point = sample["metadata"]["entry_point"]

        # Generate N samples
        generated_codes = generate_with_mtp(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_samples_per_task,
            device=device,
        )

        # Execute and check
        task_results = []
        for code in generated_codes:
            passed = execute_code_with_tests(
                code=code,
                test_code=test_code,
                entry_point=entry_point,
                timeout=5,
            )
            task_results.append(passed)

        # Record
        all_results.append({
            "task_id": task_id,
            "results": task_results,
            "num_correct": sum(task_results),
            "num_total": len(task_results),
        })

    # 7. Compute Pass@K
    logger.info("Computing Pass@K metrics...")

    # Flatten all results
    flat_results = []
    for task in all_results:
        flat_results.extend(task["results"])

    pass_at_k_metrics = evaluate_pass_at_k(
        flat_results,
        k_values=[1, 5, 10, 20] if num_samples_per_task >= 20 else [1, 5, 10],
    )

    # Per-task Pass@K
    per_task_pass_at_k = []
    for task in all_results:
        task_metrics = evaluate_pass_at_k(
            task["results"],
            k_values=[1, 5, 10],
        )
        per_task_pass_at_k.append({
            "task_id": task["task_id"],
            **task_metrics,
        })

    # 8. Log results
    logger.info("=== Evaluation Results ===")
    for k, v in pass_at_k_metrics.items():
        logger.info(f"{k}: {v:.2%}")

    if mlflow_enabled:
        mlflow.log_metrics(pass_at_k_metrics)

        # Save per-task results
        results_df = pd.DataFrame(per_task_pass_at_k)
        results_csv = f"results_{dataset_name}.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)

        mlflow.end_run()

    # 9. Return results
    return {
        "pass_at_k": pass_at_k_metrics,
        "per_task": per_task_pass_at_k,
        "checkpoint_metadata": checkpoint_metadata,
    }
```

**검증 기준**:
- Checkpoint 로딩 정상 동작
- Generation loop 안정성
- Execution 오류 처리
- Pass@K 계산 정확성
- MLflow 로깅 확인

**Integration Test** (`tests/integration/test_pipeline_evaluation.py`):
```python
def test_evaluation_pipeline_micro_mtp():
    """Micro 모델로 평가 파이프라인 E2E 테스트"""
    # Micro 모델 checkpoint 생성 (dummy)
    checkpoint_path = create_dummy_checkpoint()

    # 평가 실행 (샘플 5개, 샘플당 2개 생성)
    results = run_evaluation(
        checkpoint_path=checkpoint_path,
        dataset_name="humaneval",
        num_samples_per_task=2,
        temperature=0.2,
        mlflow_enabled=False,
    )

    # 결과 검증
    assert "pass_at_k" in results
    assert "pass@1" in results["pass_at_k"]
    assert len(results["per_task"]) > 0
```

---

### Step 4: CLI 통합

**목표**: CLI로 평가 파이프라인 실행

**구현 위치**: `src/weighted_mtp/cli/evaluate.py` (신규)

**CLI 구조**:

```python
import argparse
from weighted_mtp.pipelines.run_evaluation import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MTP model")

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., storage/checkpoints/baseline/checkpoint_best.pt)",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "codecontests"],
        help="Evaluation dataset",
    )

    # Generation
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples per task (for Pass@K)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0=greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens",
    )

    # Environment
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, mps, auto)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    args = parser.parse_args()

    # Run evaluation
    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        num_samples_per_task=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        device=args.device,
        mlflow_enabled=not args.no_mlflow,
    )

    # Print summary
    print("\n=== Evaluation Summary ===")
    for k, v in results["pass_at_k"].items():
        print(f"{k}: {v:.2%}")

if __name__ == "__main__":
    main()
```

**실행 예시**:

```bash
# Baseline 모델 평가
python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/baseline/checkpoint_best.pt \
    --dataset humaneval \
    --num-samples 20 \
    --temperature 0.2

# Verifiable 모델 평가 (MBPP)
python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/verifiable/checkpoint_best.pt \
    --dataset mbpp \
    --num-samples 10 \
    --temperature 0.8

# 4개 모델 비교 (스크립트로)
for pipeline in baseline critic verifiable rho1; do
    python -m weighted_mtp.cli.evaluate \
        --checkpoint storage/checkpoints/$pipeline/checkpoint_best.pt \
        --dataset humaneval
done
```

**`__main__.py` 통합**:

```python
# src/weighted_mtp/__main__.py
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        from weighted_mtp.cli.evaluate import main
        sys.argv = sys.argv[1:]  # Remove 'evaluate'
        main()
    else:
        from weighted_mtp.cli.train import main
        main()
```

**실행**:
```bash
# Train
python -m weighted_mtp --config configs/baseline/baseline.yaml

# Evaluate
python -m weighted_mtp evaluate --checkpoint storage/checkpoints/baseline/checkpoint_best.pt
```

---

### Step 5: MLflow 결과 분석 및 비교

**목표**: 4개 모델의 평가 결과를 MLflow에서 비교

**MLflow 기록 항목**:

**Params**:
- `checkpoint`: Checkpoint 경로
- `dataset`: 데이터셋 이름
- `num_samples_per_task`: 샘플링 개수
- `temperature`: Temperature
- `max_new_tokens`: 최대 토큰 수
- `checkpoint_epoch`: Checkpoint epoch
- `checkpoint_val_loss`: Checkpoint validation loss

**Metrics**:
- `pass@1`: Pass@1 확률
- `pass@5`: Pass@5 확률
- `pass@10`: Pass@10 확률
- `pass@20`: Pass@20 확률 (샘플 충분 시)

**Artifacts**:
- `results_{dataset}.csv`: Task별 Pass@K 결과
- `samples_{dataset}.jsonl`: 생성 샘플 예시 (처음 10개)

**비교 스크립트** (`scripts/compare_evaluation_results.py`):

```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

def compare_evaluation_results(
    experiment_name: str,
    dataset: str = "humaneval",
):
    """MLflow에서 평가 결과 비교

    Args:
        experiment_name: MLflow experiment 이름
        dataset: 데이터셋 이름
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Get all evaluation runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.dataset = '{dataset}'",
    )

    # Extract metrics
    results = []
    for _, run in runs.iterrows():
        results.append({
            "run_name": run["tags.mlflow.runName"],
            "checkpoint": run["params.checkpoint"],
            "pass@1": run["metrics.pass@1"],
            "pass@5": run["metrics.pass@5"],
            "pass@10": run["metrics.pass@10"],
        })

    results_df = pd.DataFrame(results)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(results_df))
    width = 0.25

    ax.bar([i - width for i in x], results_df["pass@1"], width, label="Pass@1")
    ax.bar(x, results_df["pass@5"], width, label="Pass@5")
    ax.bar([i + width for i in x], results_df["pass@10"], width, label="Pass@10")

    ax.set_xlabel("Model")
    ax.set_ylabel("Pass Rate")
    ax.set_title(f"Evaluation Results Comparison ({dataset})")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["run_name"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"comparison_{dataset}.png")
    print(f"Comparison plot saved: comparison_{dataset}.png")

    return results_df
```

**실행**:
```bash
python scripts/compare_evaluation_results.py
```

---

## 5. 개발원칙 준수 체크

### 원칙 1: 앞/뒤 흐름 확인

**확인 사항**:
- ✅ `src/models/meta_mtp/adapter.py`: `from_pretrained()` 이미 구현됨
- ✅ `src/utils/checkpoint_utils.py`: `load_checkpoint()` 이미 구현됨
- ✅ `vendor/meta_llama/generation.py`: 참고만 (fairscale 의존성)
- ✅ `src/data/datasets.py`: 학습용 로더 이미 구현됨

**Phase 7에서 추가할 것**:
- `src/utils/generation_utils.py`: Pure PyTorch generation (신규)
- `src/utils/evaluation_utils.py`: Execution-based evaluation (신규)
- `src/pipelines/run_evaluation.py`: 평가 파이프라인 (신규)
- `src/weighted_mtp/cli/evaluate.py`: CLI 통합 (신규)

### 원칙 2: 기존 구조 존중, 중복 제거

**기존 구조 활용**:
- ✅ `checkpoint_utils.py` 재사용 (로딩 로직)
- ✅ `datasets.py` 패턴 따름 (평가용 로더 추가)
- ✅ `pipelines/` 구조 유지 (run_evaluation.py 추가)
- ✅ `cli/` 구조 유지 (evaluate.py 추가)

**중복 제거**:
- ❌ vendor/meta_llama/generation.py 사용 안 함 (fairscale)
- ✅ Pure PyTorch로 새로 구현 (generation_utils.py)

### 원칙 3-4: 잘못된 구조 삭제, 깨끗한 코드

**삭제 대상**:
- 없음 (평가 파이프라인은 신규 기능)

**코드 품질**:
- ✅ 한글 주석 (이모지 없음)
- ✅ 타입 힌트 완벽 적용
- ✅ Docstring 작성 (Args, Returns, Examples)
- ✅ 에러 처리 (timeout, syntax error, runtime error)

### 원칙 5: 구현 후 계획 비교

**각 Step 완료 후**:
- Step 1: Generation 유틸리티 동작 확인 → Unit test 통과
- Step 2: Evaluation 유틸리티 정확성 확인 → Pass@K 계산 검증
- Step 3: 평가 파이프라인 E2E 동작 → Integration test 통과
- Step 4: CLI 실행 확인 → 4개 모델 평가 성공
- Step 5: MLflow 비교 → 차트 생성 및 결과 분석

### 원칙 6: 의존성 도구 활용

**사용 도구**:
- ✅ PyTorch (generation, inference)
- ✅ HuggingFace Datasets (평가 데이터 로딩)
- ✅ subprocess (code execution)
- ✅ scipy (Pass@K 계산)
- ✅ pandas (결과 분석)
- ✅ matplotlib (비교 차트)
- ✅ MLflow (결과 추적)

**추가 의존성** (`pyproject.toml`):
```toml
[project]
dependencies = [
    ...
    "scipy>=1.10.0",        # Pass@K 계산
    "pandas>=2.0.0",        # 결과 분석
    "matplotlib>=3.7.0",    # 차트 생성
]
```

---

## 6. 최종 완료 기준

### 6.1 산출물 체크리스트

| 항목 | 파일 경로 | 검증 기준 |
|------|-----------|-----------|
| **Generation 유틸리티** | `src/utils/generation_utils.py` | Unit test 통과, MTP head 0 사용 확인 |
| **Evaluation 유틸리티** | `src/utils/evaluation_utils.py` | Pass@K 계산 정확성, code execution 안정성 |
| **평가 파이프라인** | `src/pipelines/run_evaluation.py` | Integration test 통과, MLflow 로깅 확인 |
| **CLI 통합** | `src/weighted_mtp/cli/evaluate.py` | 4개 모델 평가 성공 |
| **비교 스크립트** | `scripts/compare_evaluation_results.py` | 차트 생성, CSV 출력 |
| **Unit Tests** | `tests/unit/test_generation_utils.py` | 3개 테스트 통과 |
| **Unit Tests** | `tests/unit/test_evaluation_utils.py` | 3개 테스트 통과 |
| **Integration Test** | `tests/integration/test_pipeline_evaluation.py` | E2E 테스트 통과 |

### 6.2 기능 검증

**Generation**:
- ✅ Greedy decoding (temperature=0) 동작
- ✅ Sampling (temperature>0) 다양성 확인
- ✅ Top-p nucleus sampling 동작
- ✅ EOS token 자동 정지
- ✅ Batch generation 정상

**Evaluation**:
- ✅ Code execution 정상 (정답/오답 구분)
- ✅ Timeout 처리
- ✅ 예외 처리 (syntax error, runtime error)
- ✅ Pass@K 계산 정확성

**Pipeline**:
- ✅ Checkpoint 로딩
- ✅ HumanEval 164개 평가 완료
- ✅ MBPP 500개 평가 완료
- ✅ MLflow 메트릭 기록
- ✅ Artifact 저장 (CSV, 샘플)

**CLI**:
- ✅ `python -m weighted_mtp evaluate --checkpoint ...` 실행
- ✅ 4개 모델 비교 스크립트 실행
- ✅ MLflow UI에서 결과 확인

### 6.3 성능 목표

**평가 시간**:
- HumanEval 164개 (샘플 20개/문제): ~30분 (A100 기준)
- MBPP 500개 (샘플 20개/문제): ~90분 (A100 기준)

**메모리 사용**:
- Micro 모델: ~1GB
- 6.7B 모델: ~15GB (float16)

**Pass@K 목표** (참고):
- Baseline: Pass@1 ~10-15%, Pass@10 ~25-35%
- Verifiable: Pass@1 ~12-18%, Pass@10 ~30-40% (개선 목표)
- Rho-1: Pass@1 ~13-20%, Pass@10 ~32-42% (최고 목표)

---

## 7. Phase 7 완료 후 상태

### 7.1 디렉터리 구조 (추가)

```
src/weighted_mtp/
├── cli/
│   ├── train.py
│   └── evaluate.py          # [신규] 평가 CLI
├── pipelines/
│   ├── run_baseline.py
│   ├── run_critic.py
│   ├── run_verifiable.py
│   ├── run_rho1.py
│   └── run_evaluation.py    # [신규] 평가 파이프라인
└── utils/
    ├── checkpoint_utils.py
    ├── generation_utils.py  # [신규] Generation 유틸리티
    └── evaluation_utils.py  # [신규] Evaluation 유틸리티

scripts/
└── compare_evaluation_results.py  # [신규] 결과 비교

tests/
├── unit/
│   ├── test_generation_utils.py   # [신규]
│   └── test_evaluation_utils.py   # [신규]
└── integration/
    └── test_pipeline_evaluation.py  # [신규]
```

### 7.2 실행 워크플로우

**학습** (Phase 6):
```bash
# 1. Baseline 학습
python -m weighted_mtp --config configs/baseline/baseline.yaml

# 2. Critic 학습
python -m weighted_mtp --config configs/critic/critic.yaml

# 3. Verifiable 학습
python -m weighted_mtp --config configs/verifiable/verifiable.yaml

# 4. Rho-1 학습
python -m weighted_mtp --config configs/rho1/rho1.yaml
```

**평가** (Phase 7):
```bash
# 1. 각 모델 평가 (HumanEval)
for pipeline in baseline critic verifiable rho1; do
    python -m weighted_mtp evaluate \
        --checkpoint storage/checkpoints/$pipeline/checkpoint_best.pt \
        --dataset humaneval \
        --num-samples 20
done

# 2. MBPP 평가
for pipeline in baseline critic verifiable rho1; do
    python -m weighted_mtp evaluate \
        --checkpoint storage/checkpoints/$pipeline/checkpoint_best.pt \
        --dataset mbpp \
        --num-samples 10
done

# 3. 결과 비교
python scripts/compare_evaluation_results.py
```

**MLflow 확인**:
```bash
# MLflow UI에서 확인
# http://13.50.240.176
# Experiment: weighted-mtp
# Runs: eval_humaneval_*, eval_mbpp_*
```

### 7.3 최종 달성 목표

Phase 7 완료 시:
- ✅ 4개 모델 (Baseline, Critic, Verifiable, Rho-1) 평가 완료
- ✅ HumanEval Pass@K 메트릭 계산 완료
- ✅ MBPP Pass@K 메트릭 계산 완료
- ✅ MLflow에 평가 결과 기록 완료
- ✅ 모델 간 성능 비교 차트 생성 완료
- ✅ 논문 작성용 실험 결과 확보 완료

---

## 8. 참조

**관련 문서**:
- `docs/00_codebase_structure.md`: 전체 코드베이스 구조
- `docs/02_implementation_plan.md`: Phase 7 개요
- `docs/08_phase6_detailed_plan.md`: 학습 파이프라인 (선행)

**외부 참조**:
- HumanEval: https://github.com/openai/human-eval
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- Pass@K 논문: Chen et al. (2021) "Evaluating Large Language Models Trained on Code"

**의존성**:
```bash
# 추가 의존성 설치
uv pip install scipy pandas matplotlib
```

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2025-11-17
**작성자**: Claude Code (Weighted MTP Team)
**Phase 상태**: 계획 완료, 구현 대기 중
