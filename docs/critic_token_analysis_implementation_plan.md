# Critic Token Analysis Implementation Plan

## 1. Overview

### 1.1 Purpose

Critic head 학습 이후, 학습된 value head가 토큰 단위로 코드의 정답 가능성을 평가하는 능력을 검증합니다.
특히 오답 코드에서 value가 급락하는 지점을 분석하여, critic head가 "에러 인지 시점"을 얼마나 잘 포착하는지 확인합니다.

### 1.2 Theoretical Background

**AlphaGo Value Network와의 유사성:**

AlphaGo는 Monte Carlo 학습으로 각 보드 상태의 승률을 예측합니다:
- 서로 다른 보드 상태 → 서로 다른 CNN features
- 동일한 target (최종 승패 ±1) → 하지만 각 상태마다 다른 value 학습
- 결과: 각 수의 승률 변화를 통해 "악수"를 탐지 가능

우리 Critic Head도 동일한 메커니즘:
- 서로 다른 partial code (position 0~t) → 서로 다른 transformer hidden states
- 동일한 target (is_correct 0/1) → 하지만 각 position마다 다른 value 학습
- 결과: 각 토큰 위치에서 "현재까지의 정보로 판단한 전체 코드의 정답 확률" 평가 가능

**핵심 통찰:**
Causal attention으로 인해 각 position은 서로 다른 partial context를 보므로,
동일한 target이어도 자연스럽게 position별로 다른 value를 학습합니다.

### 1.3 Validation Approach

오답 코드에서 value trajectory 분석:
- 초반 position: 정보 부족 → 불확실 (value ≈ 0.5)
- 중반 position: 정보 증가 → 점진적 판단
- 에러 인지 시점: value 급락 → 틀렸음을 인지
- 후반 position: 확신 (value ≈ 0.0 or 1.0)

바둑에서 "악수"를 찾는 것과 동일한 방식으로, value 급락 지점을 탐지합니다.

---

## 2. Architecture Design

### 2.1 File Structure

```
weighted_mtp/
├── src/weighted_mtp/
│   ├── analysis/                      # 새 모듈
│   │   ├── __init__.py
│   │   ├── token_value_analyzer.py    # 핵심 분석 로직
│   │   ├── visualizer.py              # 시각화
│   │   └── metrics.py                 # 평가 지표
│   ├── pipelines/
│   │   └── run_critic.py              # 재사용: load_adapter()
│   ├── data/
│   │   ├── datasets.py                # 재사용: load_dataset()
│   │   └── dataloader.py
│   └── models/meta_mtp/
│       └── adapter.py                 # 재사용: trunk_forward()
├── scripts/
│   └── evaluate_critic_token_analysis.py  # Main entry point
├── configs/
│   └── analysis/
│       └── token_analysis.yaml        # Analysis config (optional)
└── results/
    └── token_analysis/                # Output directory
        ├── values.json                # Phase 1 output
        ├── analysis.json              # Phase 2 output
        ├── metrics.json               # Phase 4 output
        └── plots/                     # Phase 3 output
            ├── sample_0.png
            └── ...
```

### 2.2 Module Design

**src/weighted_mtp/analysis/token_value_analyzer.py:**
```python
class TokenValueAnalyzer:
    def __init__(self, adapter, tokenizer, device):
        """Initialize analyzer with trained critic model"""

    def analyze_sample(self, code_text: str, is_correct: bool) -> dict:
        """Extract token-level values for a single code sample

        Returns:
            {
                "code": str,
                "tokens": List[str],
                "values": List[float],
                "is_correct": bool,
            }
        """

    def compute_value_changes(self, values: List[float]) -> dict:
        """Compute value gradient and detect drop points

        Returns:
            {
                "gradient": List[float],
                "drop_indices": List[int],
                "max_drop": {"position": int, "value": float},
            }
        """
```

**src/weighted_mtp/analysis/visualizer.py:**
```python
def plot_single_sample(tokens, values, gradient, drop_indices, save_path):
    """Generate line plot for single sample"""

def plot_multiple_samples(samples_data, save_path):
    """Compare multiple samples in one plot"""

def plot_heatmap(samples_data, save_path):
    """Heatmap of values across samples"""

def plot_correct_vs_incorrect(correct_data, incorrect_data, save_path):
    """Compare correct vs incorrect code patterns"""
```

**src/weighted_mtp/analysis/metrics.py:**
```python
def compute_detection_metrics(samples_data: List[dict]) -> dict:
    """Compute detection statistics

    Returns:
        {
            "drop_rate": float,
            "early_drop_rate": float,
            "mean_value": float,
            "std_value": float,
            "mean_max_drop": float,
        }
    """
```

### 2.3 Reusing Existing Code

**From run_critic.py:**
- `load_adapter(config, device)`: Checkpoint 로딩
- Tokenizer 초기화 로직

**From datasets.py:**
- `load_dataset()`: 데이터 로딩
- Filtering by `is_correct` field

**From adapter.py:**
- `trunk_forward(input_ids, attention_mask)`: Inference

---

## 3. Phase-by-Phase Implementation Plan

### Phase 1: Basic Inference (필수)

**목표:** 토큰별 value 추출 및 저장

**구현 내용:**

1. `token_value_analyzer.py` 핵심 로직:
```python
def analyze_sample(self, code_text, is_correct):
    # 1. Tokenize
    tokens = self.tokenizer.encode(code_text, add_special_tokens=True)
    input_ids = torch.tensor([tokens]).to(self.device)
    attention_mask = torch.ones_like(input_ids)

    # 2. Inference
    with torch.no_grad():
        outputs = self.adapter.trunk_forward(input_ids, attention_mask)
        value_logits = outputs["value_logits"]  # [1, seq_len, 1]

    # 3. Extract values
    values = value_logits[0, :, 0].cpu().numpy()

    # 4. Decode tokens
    token_texts = [self.tokenizer.decode([t]) for t in tokens]

    return {
        "code": code_text,
        "tokens": token_texts,
        "values": values.tolist(),
        "is_correct": is_correct,
    }
```

2. `scripts/evaluate_critic_token_analysis.py` Main loop:
```python
# Load model
adapter = load_adapter(config, device)
tokenizer = load_tokenizer(config)
analyzer = TokenValueAnalyzer(adapter, tokenizer, device)

# Load data (incorrect only)
dataset = load_dataset(
    "codecontests",
    split="validation",
    n_samples=args.n_samples,
    balance_correct=False,
    correct_ratio=0.0,  # Only incorrect
)

# Analyze
results = []
for sample in dataset:
    result = analyzer.analyze_sample(sample["solution"], sample["is_correct"])
    results.append(result)

# Save
with open(output_dir / "values.json", "w") as f:
    json.dump(results, f, indent=2)
```

**검증:**
- 1-2개 샘플로 수동 확인
- Value 범위 체크 (합리적인 범위인지)
- JSON 파일 생성 확인

**예상 소요 시간:** 3-4시간

---

### Phase 2: Value Change Analysis (필수)

**목표:** Value gradient 계산 및 급락 지점 탐지

**구현 내용:**

1. `token_value_analyzer.py` 확장:
```python
def compute_value_changes(self, values):
    values_arr = np.array(values)

    # Value gradient (dV/dt)
    gradient = np.diff(values_arr)

    # Drop detection (threshold-based)
    threshold = -0.1
    drop_indices = np.where(gradient < threshold)[0]

    # Maximum drop
    max_drop_idx = np.argmin(gradient)
    max_drop_value = float(gradient[max_drop_idx])

    # Statistics
    stats = {
        "gradient": gradient.tolist(),
        "drop_indices": drop_indices.tolist(),
        "max_drop": {
            "position": int(max_drop_idx),
            "value": max_drop_value,
        },
        "mean_value": float(values_arr.mean()),
        "std_value": float(values_arr.std()),
        "num_drops": len(drop_indices),
    }

    return stats
```

2. Main script 확장:
```python
# Load Phase 1 results
with open(output_dir / "values.json") as f:
    results = json.load(f)

# Analyze value changes
for result in results:
    changes = analyzer.compute_value_changes(result["values"])
    result.update(changes)

# Save
with open(output_dir / "analysis.json", "w") as f:
    json.dump(results, f, indent=2)
```

**검증:**
- 단위 테스트: edge cases (모든 값 동일, 단조 증가/감소)
- Gradient 계산 정확성 확인
- Drop detection threshold 적절성 확인

**예상 소요 시간:** 2-3시간

---

### Phase 3: Visualization (중요)

**목표:** 토큰별 value 및 gradient 시각화

**구현 내용:**

1. `visualizer.py`:
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_single_sample(result, save_path):
    """Single sample: value + gradient plots"""
    tokens = result["tokens"]
    values = result["values"]
    gradient = result["gradient"]
    drop_indices = result["drop_indices"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # Value plot
    positions = np.arange(len(values))
    ax1.plot(positions, values, marker='o', markersize=2, label='Value')
    ax1.scatter(drop_indices, [values[i] for i in drop_indices],
                color='red', s=50, label='Drop points', zorder=5)
    ax1.set_ylabel('Value')
    ax1.set_title(f'Token-level Values (is_correct={result["is_correct"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gradient plot
    ax2.plot(gradient, marker='o', markersize=2, color='orange', label='dV/dt')
    ax2.axhline(y=-0.1, color='r', linestyle='--', label='Drop threshold')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Value Gradient')
    ax2.set_xlabel('Token Position')
    ax2.set_title('Value Changes (Gradient)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_multiple_samples(results, save_path, max_samples=10):
    """Compare multiple samples"""
    plt.figure(figsize=(15, 8))

    for i, result in enumerate(results[:max_samples]):
        values = result["values"]
        plt.plot(values, alpha=0.6, label=f'Sample {i}')

    plt.ylabel('Value')
    plt.xlabel('Token Position')
    plt.title('Multiple Sample Comparison (Incorrect Codes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_correct_vs_incorrect(correct_results, incorrect_results, save_path):
    """Compare correct vs incorrect code patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Correct codes
    for i, result in enumerate(correct_results[:5]):
        ax1.plot(result["values"], alpha=0.6, label=f'Correct {i}')
    ax1.set_title('Correct Codes')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Token Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Incorrect codes
    for i, result in enumerate(incorrect_results[:5]):
        ax2.plot(result["values"], alpha=0.6, label=f'Incorrect {i}')
    ax2.set_title('Incorrect Codes')
    ax2.set_xlabel('Token Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

2. Main script에서 호출:
```python
if args.plot:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Individual plots
    for i, result in enumerate(results[:args.n_plot_samples]):
        plot_single_sample(result, plot_dir / f"sample_{i}.png")

    # Comparison plot
    plot_multiple_samples(results, plot_dir / "comparison.png")
```

**검증:**
- Plot 파일 생성 확인
- 수동 검토: 그래프가 의미 있는 패턴을 보이는지
- Value 범위 및 trajectory 합리성

**예상 소요 시간:** 2-3시간

---

### Phase 4: Evaluation Metrics (선택)

**목표:** 정량적 평가 지표 계산

**구현 내용:**

1. `metrics.py`:
```python
def compute_detection_metrics(results):
    """Compute aggregate metrics for a group of samples"""
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

    # Drop rate: percentage of samples with significant drops
    has_drop = [len(r["drop_indices"]) > 0 for r in results]
    metrics["drop_rate"] = sum(has_drop) / len(results)

    # Early drop rate: drops in first 30% of sequence
    early_drops = []
    for r in results:
        seq_len = len(r["values"])
        early_threshold = seq_len * 0.3
        early_drop = any(idx < early_threshold for idx in r["drop_indices"])
        early_drops.append(early_drop)
    metrics["early_drop_rate"] = sum(early_drops) / len(results)

    # Value statistics
    all_values = np.concatenate([r["values"] for r in results])
    metrics["mean_value"] = float(all_values.mean())
    metrics["std_value"] = float(all_values.std())

    # Drop statistics
    max_drops = [r["max_drop"]["value"] for r in results]
    metrics["mean_max_drop"] = float(np.mean(max_drops))

    num_drops = [r["num_drops"] for r in results]
    metrics["mean_num_drops"] = float(np.mean(num_drops))

    return metrics

def compare_groups(correct_results, incorrect_results):
    """Compare correct vs incorrect groups"""
    metrics = {
        "correct": compute_detection_metrics(correct_results),
        "incorrect": compute_detection_metrics(incorrect_results),
    }

    # False positive rate: correct codes showing drops
    metrics["false_positive_rate"] = metrics["correct"]["drop_rate"]

    # Discrimination: difference in drop rates
    metrics["discrimination"] = (
        metrics["incorrect"]["drop_rate"] - metrics["correct"]["drop_rate"]
    )

    return metrics
```

2. Main script 확장:
```python
# Load both correct and incorrect samples
incorrect_results = [r for r in results if not r["is_correct"]]
correct_results = [r for r in results if r["is_correct"]]

# Compute metrics
metrics = compare_groups(correct_results, incorrect_results)

# Save
with open(output_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Print summary
print("\n=== Evaluation Metrics ===")
print(f"Incorrect samples: {metrics['incorrect']['n_samples']}")
print(f"  Drop rate: {metrics['incorrect']['drop_rate']:.2%}")
print(f"  Early drop rate: {metrics['incorrect']['early_drop_rate']:.2%}")
print(f"  Mean value: {metrics['incorrect']['mean_value']:.3f}")
print(f"\nCorrect samples: {metrics['correct']['n_samples']}")
print(f"  Drop rate: {metrics['correct']['drop_rate']:.2%}")
print(f"  Mean value: {metrics['correct']['mean_value']:.3f}")
print(f"\nFalse positive rate: {metrics['false_positive_rate']:.2%}")
print(f"Discrimination: {metrics['discrimination']:.2%}")
```

**검증:**
- Sanity check: 오답 코드가 정답 코드보다 높은 drop_rate를 보이는지
- 통계 계산 정확성
- 메트릭의 해석 가능성

**예상 소요 시간:** 2-3시간

---

## 4. Implementation Guide

### 4.1 Development Order

**Step 1: Phase 1 구현 (Day 1)**
1. `src/weighted_mtp/analysis/__init__.py` 생성
2. `src/weighted_mtp/analysis/token_value_analyzer.py` 작성
3. `scripts/evaluate_critic_token_analysis.py` 기본 구조 작성
4. 단일 샘플로 테스트
5. JSON 출력 검증

**Step 2: Phase 2 구현 (Day 1-2)**
1. `token_value_analyzer.py`에 `compute_value_changes()` 추가
2. 단위 테스트 작성
3. Main script에 통합
4. Analysis JSON 출력 검증

**Step 3: Phase 3 구현 (Day 2)**
1. `src/weighted_mtp/analysis/visualizer.py` 작성
2. Plot 함수들 구현
3. Main script에 통합
4. 생성된 plot 수동 검토

**Step 4: Phase 4 구현 (Day 2-3, 선택)**
1. `src/weighted_mtp/analysis/metrics.py` 작성
2. Main script에 통합
3. Metrics 검증

### 4.2 CLI Interface

```bash
# Phase 1: Basic inference (incorrect only)
PYTHONPATH=src python scripts/evaluate_critic_token_analysis.py \
  --checkpoint storage/checkpoints/critic/best.pt \
  --dataset codecontests \
  --split validation \
  --n_samples 100 \
  --incorrect_only \
  --output_dir results/token_analysis

# Phase 2+3: Analysis + Visualization
PYTHONPATH=src python scripts/evaluate_critic_token_analysis.py \
  --checkpoint storage/checkpoints/critic/best.pt \
  --n_samples 100 \
  --incorrect_only \
  --plot \
  --n_plot_samples 20 \
  --output_dir results/token_analysis

# Phase 4: Comparison with correct samples
PYTHONPATH=src python scripts/evaluate_critic_token_analysis.py \
  --checkpoint storage/checkpoints/critic/best.pt \
  --n_samples 200 \
  --output_dir results/token_analysis_comparison
```

### 4.3 Expected Results

**Phase 1 Output (`values.json`):**
```json
[
  {
    "code": "def sum(a, b):\n    return a - b",
    "tokens": ["def", " sum", "(", "a", ",", " b", "):", ...],
    "values": [0.5, 0.52, 0.51, ..., 0.05],
    "is_correct": false
  }
]
```

**Phase 2 Output (`analysis.json`):**
```json
[
  {
    "code": "...",
    "tokens": [...],
    "values": [...],
    "is_correct": false,
    "gradient": [0.02, -0.01, ..., -0.25],
    "drop_indices": [15, 23],
    "max_drop": {"position": 23, "value": -0.25},
    "mean_value": 0.35,
    "std_value": 0.18,
    "num_drops": 2
  }
]
```

**Phase 3 Output:**
- `plots/sample_0.png`: Value + gradient plot
- `plots/comparison.png`: Multiple samples overlay
- `plots/correct_vs_incorrect.png`: Group comparison

**Phase 4 Output (`metrics.json`):**
```json
{
  "incorrect": {
    "n_samples": 100,
    "drop_rate": 0.75,
    "early_drop_rate": 0.45,
    "mean_value": 0.35,
    "std_value": 0.25,
    "mean_max_drop": -0.18,
    "mean_num_drops": 2.3
  },
  "correct": {
    "n_samples": 100,
    "drop_rate": 0.15,
    "mean_value": 0.82,
    ...
  },
  "false_positive_rate": 0.15,
  "discrimination": 0.60
}
```

---

## 5. Development Principles Compliance

### [원칙 1] 앞/뒤 흐름 분석

**기존 코드 파악:**
- ✅ `run_critic.py`: `load_adapter()`, tokenizer 초기화
- ✅ `datasets.py`: `load_dataset()`, is_correct filtering
- ✅ `adapter.py`: `trunk_forward()` 구현

**재사용 계획:**
- `load_adapter()`: 그대로 사용
- `load_dataset()`: `correct_ratio=0.0` 파라미터로 오답만 필터링
- `trunk_forward()`: Inference에 직접 사용

### [원칙 2] 기존 구조 존중 & 중복 제거

**구조 존중:**
- 새 모듈 `src/weighted_mtp/analysis/` 생성 (기존 pipelines/와 분리)
- 기존 함수 재사용 (중복 제거)
- 독립적인 `scripts/` 추가

**중복 제거:**
- Checkpoint 로딩: `load_adapter()` 재사용
- 데이터 로딩: `load_dataset()` 재사용
- Inference: `trunk_forward()` 재사용

### [원칙 3] 잘못된 구조 확인

**현재 검토:**
- 기존 코드에서 잘못된 구조 발견 안됨
- 새 기능 추가이므로 기존 삭제 불필요

### [원칙 4] 깨끗한 구현

**하위 호환성:**
- 새 기능이므로 하위 호환성 고려 불필요
- 기존 코드 수정 없음

**주석 작성:**
- 한글 사용
- 이모지 사용 금지
- 핵심 설명만 (phase, version 등 불필요한 주석 제외)

**명확한 네이밍:**
- `TokenValueAnalyzer`: 역할 명확
- `compute_value_changes()`: 동작 명확
- `drop_indices`, `gradient`: 의미 명확

### [원칙 5] 계획 대비 검토

**각 Phase 완료 후:**
1. 출력 파일 생성 확인
2. 예상 결과와 비교
3. 객관적으로 기술 (과장 금지)
4. 발견된 문제점 보고

---

## 6. Risk Mitigation

### 6.1 Potential Issues

| 문제 | 영향 | 해결 방안 |
|------|------|-----------|
| Checkpoint 로딩 실패 | 치명적 | 명확한 에러 메시지, path 검증 |
| 메모리 부족 (많은 샘플) | 높음 | Batch 처리, 점진적 저장 |
| Tokenizer 불일치 | 중간 | Checkpoint에서 tokenizer config 로드 |
| Value 범위 이상 | 낮음 | Sigmoid 적용 또는 clipping |
| 시각화 한글 깨짐 | 낮음 | 영어 라벨 사용 |
| 오답 데이터 부족 | 중간 | Train set 활용 또는 샘플 수 조정 |

### 6.2 Validation Strategy

**Phase 1:**
- 단위 테스트: `analyze_sample()` 함수
- 통합 테스트: 1-2개 샘플로 전체 실행
- 출력 검증: JSON 생성 확인, value 범위 체크

**Phase 2:**
- 단위 테스트: `compute_value_changes()` edge cases
- 정확성 검증: Gradient 계산 수동 확인
- Threshold 적절성 검토

**Phase 3:**
- Plot 생성 확인
- 수동 검토: 그래프 패턴 분석
- 여러 샘플로 재현성 확인

**Phase 4:**
- Sanity check: 오답 > 정답 drop rate
- 통계 계산 검증
- 메트릭 해석 가능성 확인

---

## 7. Timeline and Priorities

### 7.1 Estimated Timeline

| Phase | 내용 | 소요 시간 | 우선순위 |
|-------|------|-----------|----------|
| Phase 1 | Basic Inference | 3-4시간 | 필수 (높음) |
| Phase 2 | Value Change Analysis | 2-3시간 | 필수 (중간) |
| Phase 3 | Visualization | 2-3시간 | 중요 (중간) |
| Phase 4 | Evaluation Metrics | 2-3시간 | 선택 (낮음) |
| **Total** | | **9-13시간** | |

### 7.2 Minimum Viable Implementation

Phase 1 + Phase 2만으로도 검증 가능:
- 토큰별 value 추출 (Phase 1)
- Value 급락 지점 탐지 (Phase 2)
- JSON 출력으로 수동 분석 가능

Phase 3는 시각적 확인용, Phase 4는 정량 평가용으로 선택 사항.

### 7.3 Development Schedule

**Day 1 (6-7시간):**
- Morning: Phase 1 구현 및 검증
- Afternoon: Phase 2 구현 및 검증

**Day 2 (3-4시간):**
- Morning: Phase 3 구현 및 검증
- Afternoon: (선택) Phase 4 구현

**Day 3 (2-3시간, 선택):**
- Phase 4 완성 및 전체 검증
- 문서화 및 정리

---

## 8. Dependencies

### 8.1 Existing Dependencies
- torch
- transformers
- omegaconf
- datasets
- numpy

### 8.2 New Dependencies
- matplotlib (필수, Phase 3)
- seaborn (선택, 더 나은 시각화)

### 8.3 Installation
```bash
# Matplotlib (required for Phase 3)
uv add matplotlib

# Seaborn (optional)
uv add seaborn
```

---

## 9. Summary

이 계획서는 Critic head의 토큰 단위 value 분석 기능을 4개 Phase로 나누어 구현합니다.

**핵심 아이디어:**
- AlphaGo와 동일한 메커니즘으로 각 토큰 위치에서 다른 value 학습
- 오답 코드에서 value 급락 지점을 통해 "에러 인지 시점" 탐지
- 바둑의 "악수" 찾기와 동일한 접근

**구현 전략:**
- 기존 코드 최대한 재사용 (중복 제거)
- 독립적인 analysis 모듈 생성
- Phase별 점진적 구현 및 검증

**예상 결과:**
- 오답 코드의 value trajectory에서 급락 패턴 관찰
- Critic head가 실제로 "틀린 지점"을 인지하는지 검증
- 정답 코드와의 비교를 통한 정량 평가

이 검증을 통해 Critic head의 실제 능력을 확인하고, 향후 PPO 학습에서의 활용 가능성을 평가할 수 있습니다.
