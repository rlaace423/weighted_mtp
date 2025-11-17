#!/usr/bin/env python3
"""
Phase1 데이터셋 설정 통합 스크립트

HuggingFace datasets 라이브러리를 사용하여:
1. 데이터셋 다운로드 (Parquet 형식)
2. Alpaca 형식으로 변환하여 processed/ JSONL 생성
3. 메타데이터 추출 (is_correct, difficulty 정보만 별도 저장, 99% 메모리 절감)
4. Small 버전 생성 (datasets_local_small/)
5. Stats 생성 (stats/)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sentencepiece as spm
from datasets import load_dataset


# ============================================================================
# 데이터셋 설정 정의
# ============================================================================

DATASET_CONFIGS = {
    "codecontests": {
        "repo": "deepmind/code_contests",
        "base_dir": "storage/datasets/codecontests",
        "splits": ["train", "valid", "test"],
        "small_sizes": {"train": 100, "valid": 32, "test": 32},
    },
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "config": "full",  # "full" or "sanitized"
        "base_dir": "storage/datasets/mbpp",
        "splits": ["train", "validation", "test"],
        "small_sizes": {"train": 100, "validation": 32, "test": 32},
    },
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "base_dir": "storage/datasets/humaneval",
        "splits": ["test"],  # HumanEval only has test split
        "small_sizes": {"test": 32},
    },
}


# ============================================================================
# DatasetSetup 클래스
# ============================================================================

class DatasetSetup:
    def __init__(self, dataset_name: str, max_tokens: int = 2048):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.base_dir = Path(self.config["base_dir"])
        self.max_tokens = max_tokens

        # Tokenizer 로딩
        tokenizer_path = Path("storage/models/meta-llama-mtp/tokenizer/tokenizer.model")
        if tokenizer_path.exists():
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_path))
            print(f"✓ Loaded tokenizer: {tokenizer_path.name} (vocab_size={self.tokenizer.vocab_size()})")
        else:
            self.tokenizer = None
            print(f"⚠️  Tokenizer not found at {tokenizer_path}, skipping token-based filtering")
        print()

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer.encode(text))

    def _filter_by_token_length(self, samples: List[Dict]) -> List[Dict]:
        """토큰 길이 기준으로 샘플 필터링"""
        if self.tokenizer is None:
            return samples

        filtered_samples = []

        for sample in samples:
            inst_tokens = self._count_tokens(sample["instruction"])
            input_tokens = self._count_tokens(sample["input"])
            output_tokens = self._count_tokens(sample["output"])
            total_tokens = inst_tokens + input_tokens + output_tokens

            if total_tokens <= self.max_tokens:
                # 토큰 정보를 샘플에 저장 (stats에서 재사용)
                sample["_token_counts"] = {
                    "instruction": inst_tokens,
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                }
                filtered_samples.append(sample)

        filtered_count = len(samples) - len(filtered_samples)
        if filtered_count > 0:
            print(f"  ⚠️  Filtered {filtered_count} samples (>{self.max_tokens} tokens)")
            print(f"  ✓ Remaining: {len(filtered_samples)} samples")

        return filtered_samples

    def download_and_process(self) -> None:
        """HuggingFace에서 데이터셋을 다운로드하고 Alpaca 형식으로 변환"""
        print("=" * 70)
        print(f"Downloading and processing {self.dataset_name}")
        print("=" * 70)
        print()

        # 디렉터리 생성
        processed_dir = self.base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # HuggingFace에서 데이터셋 로드
        repo = self.config["repo"]
        config_name = self.config.get("config", None)

        print(f"Loading from HuggingFace: {repo}")
        if config_name:
            print(f"  Config: {config_name}")
            dataset = load_dataset(repo, config_name)
        else:
            dataset = load_dataset(repo)

        print(f"✓ Loaded dataset")
        print(f"  Splits: {list(dataset.keys())}")
        print()

        # 각 split을 Alpaca 형식으로 변환
        for split in self.config["splits"]:
            if split not in dataset:
                print(f"⚠️  Split '{split}' not found, skipping")
                continue

            print(f"Processing {split} split...")
            split_data = dataset[split]

            # 변환 함수 호출
            if self.dataset_name == "codecontests":
                alpaca_samples = self._convert_codecontests(split_data)
            elif self.dataset_name == "mbpp":
                alpaca_samples = self._convert_mbpp(split_data)
            elif self.dataset_name == "humaneval":
                alpaca_samples = self._convert_humaneval(split_data)
            else:
                raise NotImplementedError(f"No converter for {self.dataset_name}")

            print(f"  Before filtering: {len(alpaca_samples)} samples")

            # 토큰 길이 필터링
            alpaca_samples = self._filter_by_token_length(alpaca_samples)

            # JSONL 저장
            output_file = processed_dir / f"{split}.jsonl"
            with open(output_file, "w") as f:
                for sample in alpaca_samples:
                    # _token_counts는 내부 캐시용이므로 저장하지 않음
                    sample_to_save = {k: v for k, v in sample.items() if k != "_token_counts"}
                    f.write(json.dumps(sample_to_save) + "\n")

            print(f"✓ Saved {split}.jsonl ({len(alpaca_samples)} samples)")

        # Schema 생성
        self._generate_schema(processed_dir)

        print()
        print("✅ Processing completed!")
        print()

    def _convert_codecontests(self, split_data) -> List[Dict]:
        """CodeContests를 Alpaca 형식으로 변환 (correct + incorrect 모두 포함)"""
        alpaca_samples = []

        for idx, example in enumerate(split_data):
            # CodeContests 필드: name, description, public_tests, solutions, incorrect_solutions
            instruction = example.get("description", "")
            task_name = example.get("name", f"task_{idx}")

            # public_tests를 입력 예시로 사용
            public_tests = example.get("public_tests", {})
            input_list = public_tests.get("input", [])
            output_list = public_tests.get("output", [])

            input_examples = []
            for inp, out in zip(input_list[:2], output_list[:2]):  # 최대 2개만
                if inp and out:
                    input_examples.append(f"Input: {inp}\nOutput: {out}")

            input_text = "\n\n".join(input_examples) if input_examples else ""

            # 1. Correct solutions 처리
            solutions = example.get("solutions", {})
            language_list = solutions.get("language", [])
            solution_list = solutions.get("solution", [])

            for sol_idx, (lang, sol) in enumerate(zip(language_list, solution_list)):
                if lang in [1, 3]:  # PYTHON or PYTHON3
                    alpaca_sample = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": sol,
                        "task_id": f"{task_name}_correct_{sol_idx}",
                        "is_correct": True,
                        "metadata": {
                            "source": "code_contests",
                            "difficulty": example.get("difficulty", -1),
                            "has_tests": len(input_list) > 0,
                        }
                    }
                    alpaca_samples.append(alpaca_sample)

            # 2. Incorrect solutions 처리
            incorrect_solutions = example.get("incorrect_solutions", {})
            incorrect_language_list = incorrect_solutions.get("language", [])
            incorrect_solution_list = incorrect_solutions.get("solution", [])

            for sol_idx, (lang, sol) in enumerate(zip(incorrect_language_list, incorrect_solution_list)):
                if lang in [1, 3]:  # PYTHON or PYTHON3
                    alpaca_sample = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": sol,
                        "task_id": f"{task_name}_incorrect_{sol_idx}",
                        "is_correct": False,
                        "metadata": {
                            "source": "code_contests",
                            "difficulty": example.get("difficulty", -1),
                            "has_tests": len(input_list) > 0,
                        }
                    }
                    alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _convert_mbpp(self, split_data) -> List[Dict]:
        """MBPP를 Alpaca 형식으로 변환"""
        alpaca_samples = []

        for example in split_data:
            # MBPP 필드: task_id, text, code, test_list, test_setup_code, challenge_test_list
            test_list = example.get("test_list", [])
            if isinstance(test_list, np.ndarray):
                test_list = test_list.tolist()

            alpaca_sample = {
                "instruction": example.get("text", ""),
                "input": "",
                "output": example.get("code", ""),
                "task_id": str(example.get("task_id", "")),
                "metadata": {
                    "source": "mbpp",
                    "test_list": test_list,
                    "test_setup_code": example.get("test_setup_code", ""),
                    "has_tests": len(test_list) > 0,
                }
            }
            alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _convert_humaneval(self, split_data) -> List[Dict]:
        """HumanEval을 Alpaca 형식으로 변환"""
        alpaca_samples = []

        for example in split_data:
            # HumanEval 필드: task_id, prompt, canonical_solution, test, entry_point
            alpaca_sample = {
                "instruction": example.get("prompt", ""),
                "input": "",
                "output": example.get("canonical_solution", ""),
                "task_id": example.get("task_id", ""),
                "metadata": {
                    "source": "humaneval",
                    "test": example.get("test", ""),
                    "entry_point": example.get("entry_point", ""),
                    "has_tests": bool(example.get("test", "")),
                }
            }
            alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _generate_schema(self, processed_dir: Path) -> None:
        """스키마 생성"""
        # CodeContests는 is_correct 필드 포함
        required_fields = ["instruction", "input", "output", "task_id"]
        if self.dataset_name == "codecontests":
            required_fields.append("is_correct")

        schema = {
            "dataset": self.dataset_name,
            "format": "alpaca",
            "required_fields": required_fields,
            "optional_fields": ["metadata"],
            "description": f"{self.dataset_name} dataset in Alpaca format",
            "source": self.config["repo"],
            "notes": "CodeContests includes is_correct field for correct/incorrect solution classification" if self.dataset_name == "codecontests" else None
        }

        schema_file = processed_dir / "schema.json"
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"✓ Generated schema.json")

    def create_small(self) -> None:
        """Small 버전 생성"""
        print("=" * 70)
        print(f"Creating small version of {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"
        small_dir = Path(f"storage/datasets_local_small/{self.dataset_name}_small")
        small_dir.mkdir(parents=True, exist_ok=True)

        small_sizes = self.config["small_sizes"]

        for split in self.config["splits"]:
            input_file = processed_dir / f"{split}.jsonl"
            output_file = small_dir / f"{split}_small.jsonl"

            if not input_file.exists():
                print(f"⚠️  {input_file} not found, skipping {split}")
                continue

            size = small_sizes.get(split, 32)
            count = 0

            with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
                for line in f_in:
                    if count >= size:
                        break
                    f_out.write(line)
                    count += 1

            print(f"✓ Created {split}_small.jsonl ({count} samples)")

        print()
        print("✅ Small version created!")
        print()

    def extract_metadata(self) -> None:
        """메타데이터 추출 (is_correct, difficulty 정보만 별도 저장)"""
        print("=" * 70)
        print(f"Extracting metadata for {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"

        for split in self.config["splits"]:
            file = processed_dir / f"{split}.jsonl"

            if not file.exists():
                print(f"⚠️  {file} not found, skipping {split}")
                continue

            print(f"Processing {split}...")

            metadata_list = []
            stats = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "difficulty_dist": {},
                "has_is_correct": False,
                "has_difficulty": False,
            }

            with open(file, "r") as f:
                for idx, line in enumerate(f):
                    if idx % 100000 == 0 and idx > 0:
                        print(f"  진행: {idx:,} 샘플")

                    item = json.loads(line.strip())

                    # 메타데이터 추출
                    meta = {}

                    # is_correct 필드
                    if "is_correct" in item:
                        meta["is_correct"] = item["is_correct"]
                        stats["has_is_correct"] = True
                        if item["is_correct"]:
                            stats["correct"] += 1
                        else:
                            stats["incorrect"] += 1

                    # difficulty 필드
                    if "metadata" in item and "difficulty" in item["metadata"]:
                        difficulty = item["metadata"]["difficulty"]
                        meta["difficulty"] = difficulty
                        stats["has_difficulty"] = True

                        diff_str = str(difficulty)
                        stats["difficulty_dist"][diff_str] = stats["difficulty_dist"].get(diff_str, 0) + 1

                    metadata_list.append(meta)
                    stats["total"] += 1

            # 메타데이터 저장
            output_data = {
                "metadata": metadata_list,
                "stats": stats,
                "source_file": str(file),
            }

            output_file = processed_dir / f"{split}_metadata.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"✓ {split}_metadata.json ({stats['total']:,} samples)")
            if stats["has_is_correct"]:
                print(f"  correct={stats['correct']:,}, incorrect={stats['incorrect']:,}")

        print()
        print("✅ Metadata extraction completed!")
        print()

    def generate_stats(self) -> None:
        """통계 생성 (토큰 길이 포함)"""
        print("=" * 70)
        print(f"Generating stats for {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"
        stats_dir = self.base_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "dataset": self.dataset_name,
            "max_tokens": self.max_tokens,
            "tokenizer": "meta-llama-mtp" if self.tokenizer else None,
            "splits": {},
        }

        for split in self.config["splits"]:
            file = processed_dir / f"{split}.jsonl"

            if not file.exists():
                print(f"⚠️  {file} not found, skipping {split}")
                continue

            samples = []
            with open(file, "r") as f:
                for line in f:
                    samples.append(json.loads(line))

            # 문자 길이 통계
            inst_lengths = [len(s["instruction"]) for s in samples]
            out_lengths = [len(s["output"]) for s in samples]

            # is_correct 분류 (CodeContests만)
            correct_count = sum(1 for s in samples if s.get("is_correct", True))
            incorrect_count = sum(1 for s in samples if not s.get("is_correct", True))

            split_stats = {
                "count": len(samples),
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "correct_ratio": round(correct_count / len(samples), 3) if samples else 0.0,
                "avg_instruction_length": int(np.mean(inst_lengths)) if inst_lengths else 0,
                "avg_output_length": int(np.mean(out_lengths)) if out_lengths else 0,
                "max_instruction_length": int(np.max(inst_lengths)) if inst_lengths else 0,
                "max_output_length": int(np.max(out_lengths)) if out_lengths else 0,
            }

            # 토큰 길이 통계 (tokenizer 사용 가능한 경우)
            if self.tokenizer:
                # process 단계에서 저장한 토큰 정보 재사용 (중복 계산 방지)
                if samples and "_token_counts" in samples[0]:
                    # 이미 계산된 토큰 정보 사용
                    inst_tokens = [s["_token_counts"]["instruction"] for s in samples]
                    out_tokens = [s["_token_counts"]["output"] for s in samples]
                    total_tokens = [s["_token_counts"]["total"] for s in samples]
                else:
                    # 토큰 정보가 없는 경우만 계산
                    inst_tokens = [self._count_tokens(s["instruction"]) for s in samples]
                    out_tokens = [self._count_tokens(s["output"]) for s in samples]
                    total_tokens = [
                        self._count_tokens(s["instruction"]) +
                        self._count_tokens(s["input"]) +
                        self._count_tokens(s["output"])
                        for s in samples
                    ]

                split_stats.update({
                    "avg_instruction_tokens": int(np.mean(inst_tokens)) if inst_tokens else 0,
                    "avg_output_tokens": int(np.mean(out_tokens)) if out_tokens else 0,
                    "avg_total_tokens": int(np.mean(total_tokens)) if total_tokens else 0,
                    "max_instruction_tokens": int(np.max(inst_tokens)) if inst_tokens else 0,
                    "max_output_tokens": int(np.max(out_tokens)) if out_tokens else 0,
                    "max_total_tokens": int(np.max(total_tokens)) if total_tokens else 0,
                })

            stats["splits"][split] = split_stats
            print(f"✓ {split}: {len(samples)} samples")

        # 저장
        timestamp = datetime.now().strftime("%Y-%m-%d")
        stats_file = stats_dir / f"{timestamp}_summary.json"

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print()
        print(f"✓ Stats saved to {stats_file}")
        print()

        print("✅ Stats generated!")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase1 데이터셋 설정 스크립트 (HuggingFace datasets 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 데이터셋 설정 (다운로드 + 변환 + 메타데이터 + small + stats)
  uv run python scripts/setup_datasets.py --datasets all --steps all

  # MBPP만 처리
  uv run python scripts/setup_datasets.py --datasets mbpp --steps all

  # 다운로드+전처리만
  uv run python scripts/setup_datasets.py --datasets all --steps process

  # 메타데이터만 추출
  uv run python scripts/setup_datasets.py --datasets codecontests --steps metadata

  # 단계별 실행
  uv run python scripts/setup_datasets.py --datasets codecontests --steps process,metadata,small,stats
        """,
    )

    parser.add_argument(
        "--datasets",
        required=True,
        help="Datasets to setup (comma-separated or 'all'): codecontests,mbpp,humaneval,all",
    )

    parser.add_argument(
        "--steps",
        default="all",
        help="Steps to run (comma-separated): process,metadata,small,stats,all",
    )

    args = parser.parse_args()

    # Datasets 파싱
    if args.datasets == "all":
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Steps 파싱
    if args.steps == "all":
        steps = ["process", "metadata", "small", "stats"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    # Max tokens 파라미터 추가 (필요 시 argparse에서 받을 수 있음)
    max_tokens = 2048

    # 각 데이터셋 처리
    for dataset_name in datasets:
        try:
            setup = DatasetSetup(dataset_name, max_tokens=max_tokens)

            if "process" in steps:
                setup.download_and_process()

            if "metadata" in steps:
                setup.extract_metadata()

            if "small" in steps:
                setup.create_small()

            if "stats" in steps:
                setup.generate_stats()

        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print()
    print("=" * 70)
    print("✅ All datasets processed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
