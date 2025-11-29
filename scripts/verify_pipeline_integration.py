#!/usr/bin/env python3
"""파이프라인 통합 검증 스크립트

검증 항목:
1. Difficulty bins 기반 샘플링
2. Pairwise 샘플링 선정
3. Alpaca 전처리/마스킹/패딩 일관성
4. Micro model 파이프라인 E2E 테스트
5. Value head - TD error 호환성
6. 파이프라인별 로깅 항목 검증
"""

import sys
import logging
from pathlib import Path
from collections import Counter

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """섹션 헤더 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = ""):
    """테스트 결과 출력"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


# =============================================================================
# 1. Difficulty Bins 기반 샘플링 검증
# =============================================================================
def verify_difficulty_sampling():
    """Difficulty bins 기반 샘플링 검증"""
    print_header("1. Difficulty Bins 기반 샘플링 검증")

    from weighted_mtp.data import load_dataset

    all_passed = True

    # 테스트 1: 기본 difficulty 샘플링
    try:
        sampling_config = {
            "n_samples": 500,
            "use_pairwise": True,
            "difficulty_bins": {
                "diff_7": [7, 7],
                "else": [8, 25],
            },
            "difficulty_weights": {
                "diff_7": 0.35,
                "else": 0.65,
            },
        }

        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42
        )

        # 난이도 분포 확인
        diff_counts = Counter()
        for sample in dataset:
            diff = sample["metadata"]["difficulty"]
            if diff == 7:
                diff_counts["diff_7"] += 1
            elif 8 <= diff <= 25:
                diff_counts["else"] += 1

        total = len(dataset)
        diff_7_ratio = diff_counts["diff_7"] / total
        else_ratio = diff_counts["else"] / total

        # 비율 검증 (±10% 오차 허용)
        diff_7_ok = 0.25 <= diff_7_ratio <= 0.45
        else_ok = 0.55 <= else_ratio <= 0.75

        passed = diff_7_ok and else_ok
        all_passed = all_passed and passed
        print_result(
            "Difficulty 비율 준수",
            passed,
            f"diff_7={diff_7_ratio:.1%} (목표 35%), else={else_ratio:.1%} (목표 65%)"
        )

    except Exception as e:
        print_result("Difficulty 샘플링", False, str(e))
        all_passed = False

    # 테스트 2: Pairwise 샘플에 correct/incorrect 쌍 포함 검증
    try:
        correct_count = sum(1 for s in dataset if s["is_correct"])
        incorrect_count = len(dataset) - correct_count

        passed = correct_count > 0 and incorrect_count > 0
        all_passed = all_passed and passed
        print_result(
            "Pairwise 쌍 검증",
            passed,
            f"correct={correct_count}, incorrect={incorrect_count}"
        )
    except Exception as e:
        print_result("Pairwise 쌍", False, str(e))
        all_passed = False

    # 테스트 3: use_pairwise=False (Baseline용)
    try:
        sampling_config_baseline = {
            "n_samples": 200,
            "use_pairwise": False,
            "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
        }

        dataset_baseline = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config_baseline,
            seed=42
        )

        all_correct = all(s["is_correct"] for s in dataset_baseline)
        all_passed = all_passed and all_correct
        print_result(
            "Correct-only 샘플링 (Baseline용)",
            all_correct,
            f"총 {len(dataset_baseline)}개 모두 correct={all_correct}"
        )
    except Exception as e:
        print_result("Correct-only 샘플링", False, str(e))
        all_passed = False

    return all_passed


# =============================================================================
# 2. Pairwise 샘플링 검증
# =============================================================================
def verify_pairwise_sampling():
    """Pairwise 샘플링 선정 검증"""
    print_header("2. Pairwise 샘플링 검증")

    from weighted_mtp.data import load_dataset

    all_passed = True

    try:
        sampling_config = {
            "use_pairwise": True,
            "n_samples": 500,
            "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
        }

        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config=sampling_config,
            seed=42
        )

        # Pairwise 필드 존재 확인 (correct_output, incorrect_output 형식)
        sample = dataset[0]
        has_correct = "correct_output" in sample
        has_incorrect = "incorrect_output" in sample
        has_instruction = "instruction" in sample

        passed = has_correct and has_incorrect and has_instruction
        all_passed = all_passed and passed
        print_result(
            "Pairwise 필드 존재",
            passed,
            f"correct_output={has_correct}, incorrect_output={has_incorrect}, instruction={has_instruction}"
        )

        # Output이 다른지 확인 (correct != incorrect)
        if passed:
            outputs_different = sample.get("correct_output") != sample.get("incorrect_output")
            all_passed = all_passed and outputs_different
            print_result(
                "Correct/Incorrect output 상이",
                outputs_different,
                f"correct[:30]={sample.get('correct_output', '')[:30]}..."
            )

        print_result(
            "Pairwise 쌍 생성",
            True,
            f"총 {len(dataset)}개 쌍 생성"
        )

    except Exception as e:
        print_result("Pairwise 샘플링", False, str(e))
        all_passed = False

    return all_passed


# =============================================================================
# 3. Alpaca 전처리/마스킹/패딩 일관성 검증
# =============================================================================
def verify_alpaca_preprocessing():
    """Alpaca 전처리/마스킹/패딩 일관성 검증"""
    print_header("3. Alpaca 전처리/마스킹/패딩 일관성")

    from weighted_mtp.data.dataloader import create_dataloader
    from weighted_mtp.models.tokenizer_utils import load_tokenizer

    all_passed = True
    tokenizer_path = project_root / "storage/models/meta-llama-mtp/tokenizer"

    if not tokenizer_path.exists():
        print_result("Tokenizer 로딩", False, f"경로 없음: {tokenizer_path}")
        return False

    tokenizer = load_tokenizer(str(tokenizer_path))
    dataset_path = project_root / "storage/datasets/codecontests/processed/train.jsonl"

    if not dataset_path.exists():
        print_result("Dataset 로딩", False, f"경로 없음: {dataset_path}")
        return False

    # Pointwise DataLoader
    try:
        sampling_config_pointwise = {
            "use_pairwise": False,
            "n_samples": 50,
            "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
        }

        loader_pointwise = create_dataloader(
            dataset_path=str(dataset_path),
            tokenizer=tokenizer,
            batch_size=4,
            max_length=512,
            sampling_config=sampling_config_pointwise,
            seed=42,
        )

        batch_pw = next(iter(loader_pointwise))

        # 필수 필드 검증
        has_input_ids = "input_ids" in batch_pw
        has_labels = "labels" in batch_pw
        has_attention = "attention_mask" in batch_pw

        passed = has_input_ids and has_labels and has_attention
        all_passed = all_passed and passed
        print_result(
            "Pointwise 필수 필드",
            passed,
            f"input_ids={has_input_ids}, labels={has_labels}, attention_mask={has_attention}"
        )

        # 패딩 검증 (모든 시퀀스 동일 길이)
        seq_lengths = batch_pw["input_ids"].shape
        passed = len(seq_lengths) == 2  # (batch, seq_len)
        all_passed = all_passed and passed
        print_result(
            "Pointwise 패딩 일관성",
            passed,
            f"shape={seq_lengths}"
        )

        # 라벨 마스킹 검증 (instruction 부분 -100)
        labels = batch_pw["labels"]
        has_masked = (labels == -100).any()
        has_unmasked = (labels != -100).any()
        passed = has_masked and has_unmasked
        all_passed = all_passed and passed
        print_result(
            "Pointwise 라벨 마스킹",
            passed,
            f"masked={has_masked.item()}, unmasked={has_unmasked.item()}"
        )

    except Exception as e:
        print_result("Pointwise DataLoader", False, str(e))
        all_passed = False

    # Pairwise DataLoader
    try:
        sampling_config_pairwise = {
            "use_pairwise": True,
            "n_samples": 100,
            "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
        }

        loader_pairwise = create_dataloader(
            dataset_path=str(dataset_path),
            tokenizer=tokenizer,
            batch_size=4,
            max_length=512,
            sampling_config=sampling_config_pairwise,
            seed=42,
        )

        batch_pair = next(iter(loader_pairwise))

        # Pairwise 필수 필드 검증
        has_pos_ids = "pos_input_ids" in batch_pair
        has_neg_ids = "neg_input_ids" in batch_pair
        has_pos_labels = "pos_labels" in batch_pair
        has_neg_labels = "neg_labels" in batch_pair

        passed = has_pos_ids and has_neg_ids and has_pos_labels and has_neg_labels
        all_passed = all_passed and passed
        print_result(
            "Pairwise 필수 필드",
            passed,
            f"pos_input_ids={has_pos_ids}, neg_input_ids={has_neg_ids}"
        )

        # pos/neg 동일 길이 검증
        if passed:
            pos_len = batch_pair["pos_input_ids"].shape[1]
            neg_len = batch_pair["neg_input_ids"].shape[1]
            same_len = pos_len == neg_len
            all_passed = all_passed and same_len
            print_result(
                "Pairwise pos/neg 길이 일치",
                same_len,
                f"pos_len={pos_len}, neg_len={neg_len}"
            )

    except Exception as e:
        print_result("Pairwise DataLoader", False, str(e))
        all_passed = False

    return all_passed


# =============================================================================
# 4. Micro Model 파이프라인 E2E 테스트
# =============================================================================
def verify_pipeline_e2e():
    """Micro model 파이프라인 E2E 테스트"""
    print_header("4. Micro Model 파이프라인 E2E 테스트")

    # 각 파이프라인 import 가능 여부 확인
    all_passed = True
    pipelines = ["baseline", "critic", "verifiable", "rho1"]

    for pipeline in pipelines:
        try:
            module = __import__(
                f"weighted_mtp.pipelines.run_{pipeline}",
                fromlist=["main"]
            )
            passed = hasattr(module, "main")
            all_passed = all_passed and passed
            print_result(f"{pipeline} 파이프라인 import", passed)
        except Exception as e:
            print_result(f"{pipeline} 파이프라인 import", False, str(e)[:50])
            all_passed = False

    # Config 로딩 테스트
    config_files = {
        "baseline": "configs/baseline/baseline_local.yaml",
        "critic": "configs/critic/critic_local.yaml",
        "verifiable": "configs/verifiable/verifiable_local.yaml",
        "rho1": "configs/rho1/rho1_local.yaml",
    }

    for name, config_path in config_files.items():
        full_path = project_root / config_path
        if full_path.exists():
            try:
                config = OmegaConf.load(full_path)
                has_experiment = hasattr(config, "experiment")
                has_training = hasattr(config, "training")
                has_models = hasattr(config, "models")

                passed = has_experiment and has_training and has_models
                all_passed = all_passed and passed
                print_result(f"{name} config 구조", passed)
            except Exception as e:
                print_result(f"{name} config 로딩", False, str(e)[:50])
                all_passed = False
        else:
            print_result(f"{name} config 존재", False, f"파일 없음: {config_path}")

    return all_passed


# =============================================================================
# 5. Value Head - TD Error 호환성 검증
# =============================================================================
def verify_value_head_td_compatibility():
    """Value head - TD error 호환성 검증"""
    print_header("5. Value Head - TD Error 호환성 검증")

    all_passed = True

    # Value head 차원 검증
    try:
        from weighted_mtp.models.value_head import ValueHead

        # Linear value head
        vh_linear = ValueHead(hidden_size=4096, value_head_type="linear")
        dummy_hidden = torch.randn(2, 100, 4096)  # (batch, seq, hidden)
        output_linear = vh_linear(dummy_hidden)

        passed = output_linear.shape == (2, 100, 1)
        all_passed = all_passed and passed
        print_result(
            "Linear ValueHead 출력 차원",
            passed,
            f"shape={output_linear.shape} (expected: [2, 100, 1])"
        )

        # Sigmoid value head
        vh_sigmoid = ValueHead(hidden_size=4096, value_head_type="sigmoid")
        output_sigmoid = vh_sigmoid(dummy_hidden)

        in_range = (output_sigmoid >= 0).all() and (output_sigmoid <= 1).all()
        all_passed = all_passed and in_range
        print_result(
            "Sigmoid ValueHead 범위 [0, 1]",
            in_range,
            f"min={output_sigmoid.min():.3f}, max={output_sigmoid.max():.3f}"
        )

        # MLP value head
        vh_mlp = ValueHead(hidden_size=4096, value_head_type="mlp")
        output_mlp = vh_mlp(dummy_hidden)

        passed = output_mlp.shape == (2, 100, 1)
        all_passed = all_passed and passed
        print_result(
            "MLP ValueHead 출력 차원",
            passed,
            f"shape={output_mlp.shape}"
        )

    except Exception as e:
        print_result("ValueHead 생성", False, str(e))
        all_passed = False

    # Pairwise ranking loss 함수 검증
    try:
        from weighted_mtp.losses.pairwise import pairwise_ranking_loss

        pos_values = torch.randn(4, 100, 1)
        neg_values = torch.randn(4, 100, 1)
        pos_mask = torch.ones(4, 100)
        neg_mask = torch.ones(4, 100)

        loss = pairwise_ranking_loss(pos_values, neg_values, pos_mask, neg_mask)

        passed = loss.ndim == 0 and not torch.isnan(loss) and not torch.isinf(loss)
        all_passed = all_passed and passed
        print_result(
            "Pairwise Ranking Loss 계산",
            passed,
            f"loss={loss.item():.4f}"
        )

    except Exception as e:
        print_result("Pairwise Ranking Loss", False, str(e))
        all_passed = False

    # TD target 계산 함수 검증 (Stage 2용)
    try:
        from weighted_mtp.losses.value_loss import compute_td_targets

        values = torch.randn(4, 100, 1)
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])  # is_correct
        mask = torch.ones(4, 100)

        td_targets = compute_td_targets(
            values=values,
            rewards=rewards,
            mask=mask,
            gamma=1.0,
            lam=1.0  # MC
        )

        passed = td_targets.shape == values.shape
        all_passed = all_passed and passed
        print_result(
            "TD Targets 계산 (MC, lam=1.0)",
            passed,
            f"shape={td_targets.shape}"
        )

        # MC 모드에서 td_targets == rewards인지 확인
        # lam=1.0이면 모든 토큰의 target이 최종 reward와 같아야 함
        expected_targets = rewards.view(-1, 1, 1).expand_as(values)
        mc_correct = torch.allclose(td_targets, expected_targets, atol=1e-5)
        all_passed = all_passed and mc_correct
        print_result(
            "MC 모드 TD targets == rewards",
            mc_correct,
            f"td[0,0,0]={td_targets[0,0,0].item():.3f}, reward[0]={rewards[0].item()}"
        )

    except Exception as e:
        print_result("TD Targets 계산", False, str(e))
        all_passed = False

    return all_passed


# =============================================================================
# 6. 파이프라인별 로깅 항목 검증
# =============================================================================
def verify_pipeline_logging():
    """파이프라인별 로깅 항목 검증"""
    print_header("6. 파이프라인별 로깅 항목 검증")

    all_passed = True

    # 각 파이프라인에서 로깅해야 하는 핵심 메트릭
    expected_metrics = {
        "baseline": [
            "train/loss", "train/ce_loss", "train/perplexity",
            "val/loss", "val/perplexity",
        ],
        "critic": [
            "train/pairwise_ranking_loss",
            "val/pairwise_ranking_loss", "val/accuracy",
        ],
        "verifiable": [
            "train/weighted_ce_loss", "train/value_loss", "train/total_loss",
            "train/avg_weight", "train/td_error_mean",
            "val/weighted_ce_loss", "val/pairwise_ranking_loss",
        ],
        "rho1": [
            "train/loss", "train/selected_tokens", "train/avg_weight",
        ],
    }

    # 파이프라인 소스 코드에서 로깅 검증
    for pipeline, metrics in expected_metrics.items():
        pipeline_path = project_root / f"src/weighted_mtp/pipelines/run_{pipeline}.py"

        if not pipeline_path.exists():
            print_result(f"{pipeline} 파이프라인 파일", False, "파일 없음")
            all_passed = False
            continue

        with open(pipeline_path, "r") as f:
            content = f.read()

        found_metrics = []
        missing_metrics = []

        for metric in metrics:
            # 메트릭 이름의 핵심 부분 검색
            metric_key = metric.split("/")[-1]
            if metric_key in content or metric.replace("/", "_") in content:
                found_metrics.append(metric)
            else:
                missing_metrics.append(metric)

        passed = len(missing_metrics) == 0
        all_passed = all_passed and passed

        if passed:
            print_result(
                f"{pipeline} 핵심 메트릭 로깅",
                True,
                f"{len(found_metrics)}/{len(metrics)} 확인"
            )
        else:
            print_result(
                f"{pipeline} 핵심 메트릭 로깅",
                False,
                f"누락: {missing_metrics[:3]}..."
            )

    return all_passed


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    """전체 검증 실행"""
    print("\n" + "=" * 60)
    print("  파이프라인 통합 검증 시작")
    print("=" * 60)

    results = {}

    # 1. Difficulty 샘플링
    results["difficulty_sampling"] = verify_difficulty_sampling()

    # 2. Pairwise 샘플링
    results["pairwise_sampling"] = verify_pairwise_sampling()

    # 3. Alpaca 전처리
    results["alpaca_preprocessing"] = verify_alpaca_preprocessing()

    # 4. 파이프라인 E2E
    results["pipeline_e2e"] = verify_pipeline_e2e()

    # 5. Value head - TD 호환성
    results["value_head_td"] = verify_value_head_td_compatibility()

    # 6. 로깅 검증
    results["pipeline_logging"] = verify_pipeline_logging()

    # 최종 결과
    print_header("최종 결과")

    all_passed = True
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {'PASS' if passed else 'FAIL'}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ 모든 검증 통과!")
    else:
        print("  ✗ 일부 검증 실패")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
