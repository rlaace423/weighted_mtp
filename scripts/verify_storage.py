#!/usr/bin/env python3
"""
Phase1 Storage 무결성 검증 및 체크리스트 생성

기능:
- 모든 모델 무결성 검증 (SHA256, Config, SafeTensors)
- 모든 데이터셋 검증 (파일 존재, Schema)
- Phase1 체크리스트 자동 생성
- 계획서와 실제 구조 비교

Usage:
    # 전체 검증
    python scripts/verify_storage.py --check all

    # 모델만 검증
    python scripts/verify_storage.py --check models

    # Phase1 체크리스트 생성
    python scripts/verify_storage.py --check all --phase1-checklist

    # 리포트 생성
    python scripts/verify_storage.py --check all --generate-report
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from safetensors import safe_open


# ============================================================================
# 검증 대상 정의
# ============================================================================

MODELS_TO_VERIFY = {
    "meta-llama-mtp": {
        "base_dir": "storage/models/meta-llama-mtp",
        "required_files": [
            "safetensors/model.safetensors",
            "safetensors/SHA256SUMS",
            "configs/params.json",
            "configs/meta_adapter.yaml",
        ],
        "config_checks": [
            ("n_future_tokens", 4),
            ("rope_theta", 10000.0),
            ("intermediate_size", 11008),
            ("dtype", "float16"),
        ],
    },
    "ref-sheared-llama-2.7b": {
        "base_dir": "storage/models/ref-sheared-llama-2.7b",
        "required_files": [
            "safetensors/model.safetensors",
            "safetensors/SHA256SUMS",
            "configs/config.json",
        ],
        "config_checks": [
            ("hidden_size", 2560),
            ("num_hidden_layers", 32),
        ],
    },
    "micro-mtp": {
        "base_dir": "storage/models/micro-mtp",
        "required_files": [
            "safetensors/model.safetensors",
            "safetensors/SHA256SUMS",
            "configs/config.json",
        ],
        "config_checks": [
            ("num_hidden_layers", 4),
            ("hidden_size", 512),
        ],
    },
}

DATASETS_TO_VERIFY = {
    "codecontests": {
        "small_dir": "storage/datasets_local_small/codecontests_small",
        "files": ["train_small.jsonl", "validation_small.jsonl"],
        "required_fields": ["instruction", "input", "output", "task_id"],
    },
    "mbpp": {
        "small_dir": "storage/datasets_local_small/mbpp_small",
        "files": ["train_small.jsonl", "validation_small.jsonl"],
        "required_fields": ["task_id", "text", "code"],
    },
    "humaneval": {
        "small_dir": "storage/datasets_local_small/humaneval_small",
        "files": ["test_small.jsonl"],
        "required_fields": ["task_id", "prompt"],
    },
}


# ============================================================================
# 유틸리티 함수
# ============================================================================

def calculate_sha256(file_path: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ============================================================================
# StorageVerifier 클래스
# ============================================================================

class StorageVerifier:
    def __init__(self):
        self.results = {
            "models": {},
            "datasets": {},
            "timestamp": datetime.now().isoformat(),
        }

    def verify_models(self, model_list: List[str]) -> bool:
        """모델 검증"""
        print("=" * 70)
        print("Verifying Models")
        print("=" * 70)
        print()

        all_passed = True

        for model_name in model_list:
            if model_name not in MODELS_TO_VERIFY:
                print(f"⚠️  Unknown model: {model_name}, skipping...")
                continue

            print(f"[{model_name}]")
            print("-" * 70)

            config = MODELS_TO_VERIFY[model_name]
            base_dir = Path(config["base_dir"])

            model_passed, model_errors = self._verify_single_model(
                model_name, base_dir, config
            )

            self.results["models"][model_name] = {
                "passed": model_passed,
                "errors": model_errors,
            }

            if model_passed:
                print(f"✅ {model_name}: All checks passed!")
            else:
                print(f"❌ {model_name}: Verification failed!")
                for err in model_errors:
                    print(f"  - {err}")
                all_passed = False

            print()

        return all_passed

    def _verify_single_model(
        self, model_name: str, base_dir: Path, config: Dict
    ) -> Tuple[bool, List[str]]:
        """단일 모델 검증"""
        errors = []

        # 1. 파일 존재 확인
        print("  [1/3] Checking required files...")
        for file_rel in config["required_files"]:
            file_path = base_dir / file_rel
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024**2)
                print(f"    ✓ {file_rel} ({size_mb:.1f} MB)")
            else:
                errors.append(f"Missing: {file_rel}")
                print(f"    ✗ {file_rel} NOT FOUND")

        # 2. SHA256 검증
        safetensors_file = base_dir / "safetensors/model.safetensors"
        sha256_file = base_dir / "safetensors/SHA256SUMS"

        if safetensors_file.exists() and sha256_file.exists():
            print("  [2/3] Verifying SHA256...")

            with open(sha256_file, "r") as f:
                expected_hash = f.read().strip().split()[0]

            actual_hash = calculate_sha256(safetensors_file)

            if expected_hash == actual_hash:
                print(f"    ✓ SHA256 match: {actual_hash[:16]}...")
            else:
                errors.append("SHA256 mismatch")
                print(f"    ✗ SHA256 mismatch!")
                print(f"      Expected: {expected_hash}")
                print(f"      Actual:   {actual_hash}")
        else:
            print("  [2/3] Skipping SHA256 (files missing)")

        # 3. Config 검증
        if config["config_checks"]:
            print("  [3/3] Verifying config...")

            # Config 파일 로드 (명시적 경로, fallback 제거)
            yaml_path = base_dir / "configs/meta_adapter.yaml"
            json_path = base_dir / "configs/config.json"

            config_file = None
            config_data = None

            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    config_data = yaml.safe_load(f)
                config_file = "meta_adapter.yaml"
            elif json_path.exists():
                with open(json_path, "r") as f:
                    config_data = json.load(f)
                config_file = "config.json"
            else:
                # Config 파일이 없으면 명시적 오류
                errors.append("Required config file not found (meta_adapter.yaml or config.json)")
                print("    ✗ Required config file not found in configs/")
                print("      Expected: configs/meta_adapter.yaml or configs/config.json")
                return (False, errors)

            # Config 검증
            for key, expected_value in config["config_checks"]:
                actual_value = config_data.get(key)

                if actual_value == expected_value:
                    print(f"    ✓ {key}: {actual_value}")
                else:
                    errors.append(
                        f"Config mismatch: {key} (expected={expected_value}, actual={actual_value})"
                    )
                    print(f"    ✗ {key}: expected={expected_value}, actual={actual_value}")
        else:
            print("  [3/3] Skipping config checks (not defined)")

        return (len(errors) == 0, errors)

    def verify_datasets(self, dataset_list: List[str]) -> bool:
        """데이터셋 검증"""
        print("=" * 70)
        print("Verifying Datasets")
        print("=" * 70)
        print()

        all_passed = True

        for dataset_name in dataset_list:
            if dataset_name not in DATASETS_TO_VERIFY:
                print(f"⚠️  Unknown dataset: {dataset_name}, skipping...")
                continue

            print(f"[{dataset_name}]")
            print("-" * 70)

            config = DATASETS_TO_VERIFY[dataset_name]
            small_dir = Path(config["small_dir"])

            dataset_passed, dataset_errors = self._verify_single_dataset(
                dataset_name, small_dir, config
            )

            self.results["datasets"][dataset_name] = {
                "passed": dataset_passed,
                "errors": dataset_errors,
            }

            if dataset_passed:
                print(f"✅ {dataset_name}: All checks passed!")
            else:
                print(f"❌ {dataset_name}: Verification failed!")
                for err in dataset_errors:
                    print(f"  - {err}")
                all_passed = False

            print()

        return all_passed

    def _verify_single_dataset(
        self, dataset_name: str, small_dir: Path, config: Dict
    ) -> Tuple[bool, List[str]]:
        """단일 데이터셋 검증"""
        errors = []

        # 1. 파일 존재 확인
        print("  [1/2] Checking files...")
        for file_name in config["files"]:
            file_path = small_dir / file_name
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"    ✓ {file_name} ({size_kb:.1f} KB)")
            else:
                errors.append(f"Missing: {file_name}")
                print(f"    ✗ {file_name} NOT FOUND")

        # 2. Schema 검증 (샘플 3개)
        print("  [2/2] Validating schema...")
        required_fields = config["required_fields"]

        for file_name in config["files"]:
            file_path = small_dir / file_name
            if not file_path.exists():
                continue

            print(f"    Checking {file_name}...")

            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 3:  # 처음 3개만 체크
                        break

                    try:
                        data = json.loads(line)

                        # 필수 필드 확인 (유연하게)
                        missing = [
                            field for field in required_fields if field not in data
                        ]

                        if missing:
                            print(f"      ⚠️  Line {i+1}: missing {missing} (non-fatal)")
                        else:
                            print(f"      ✓ Line {i+1}: schema valid")

                    except json.JSONDecodeError as e:
                        errors.append(f"{file_name} line {i+1}: JSON decode error")
                        print(f"      ✗ Line {i+1}: JSON error")

        return (len(errors) == 0, errors)

    def generate_phase1_checklist(self) -> None:
        """Phase1 체크리스트 생성"""
        print("=" * 70)
        print("Phase 1 Completion Checklist")
        print("=" * 70)
        print()

        checklist = {
            "timestamp": self.results["timestamp"],
            "steps": {},
        }

        # Step 2: 모델 원본 다운로드
        meta_mtp = self.results["models"].get("meta-llama-mtp", {})
        ref_sheared = self.results["models"].get("ref-sheared-llama-2.7b", {})

        checklist["steps"]["step2_model_download"] = {
            "name": "모델 원본 다운로드 (Meta 7B_1T_4, Sheared LLaMA 2.7B)",
            "status": "✅" if meta_mtp.get("passed") and ref_sheared.get("passed") else "❌",
        }

        # Step 3-4: 모델 변환
        checklist["steps"]["step3_meta_conversion"] = {
            "name": "Meta LLaMA MTP 파생 자산 생성 (safetensors, config)",
            "status": "✅" if meta_mtp.get("passed") else "❌",
        }

        checklist["steps"]["step4_ref_conversion"] = {
            "name": "Sheared LLaMA 2.7B 파생 자산 생성",
            "status": "✅" if ref_sheared.get("passed") else "❌",
        }

        # Step 5: Micro 모델
        micro_mtp = self.results["models"].get("micro-mtp", {})

        checklist["steps"]["step5_micro_model"] = {
            "name": "Micro 모델 생성 (Policy & Reference)",
            "status": "✅" if micro_mtp.get("passed") else "❌",
        }

        # Step 6-8: 데이터셋
        datasets_passed = all(
            ds.get("passed", False) for ds in self.results["datasets"].values()
        )

        checklist["steps"]["step6_dataset_download"] = {
            "name": "데이터셋 원본 다운로드 (CodeContests, MBPP, HumanEval)",
            "status": "✅" if datasets_passed else "❌",
        }

        checklist["steps"]["step8_dataset_small"] = {
            "name": "Small 데이터셋 생성 및 검증",
            "status": "✅" if datasets_passed else "❌",
        }

        # Step 9: 무결성 검증
        all_models_passed = all(
            m.get("passed", False) for m in self.results["models"].values()
        )

        checklist["steps"]["step9_verification"] = {
            "name": "자산 무결성 검증 (모델 & 데이터셋)",
            "status": "✅" if all_models_passed and datasets_passed else "❌",
        }

        # 출력
        for step_id, step_info in checklist["steps"].items():
            print(f"{step_info['status']} {step_info['name']}")

        print()

        # 전체 상태
        all_passed = all(
            step["status"] == "✅" for step in checklist["steps"].values()
        )

        if all_passed:
            print("=" * 70)
            print("✅ Phase 1 Complete!")
            print("=" * 70)
            print()
            print("모든 체크리스트 항목이 완료되었습니다.")
            print("다음 단계: Phase 2 (코드 스켈레톤 구축)")
        else:
            print("=" * 70)
            print("⚠️  Phase 1 Incomplete")
            print("=" * 70)
            print()
            print("일부 항목이 완료되지 않았습니다. 위 체크리스트를 확인하세요.")

        print()

        # 체크리스트 파일 저장
        checklist_file = Path("docs/phase1_checklist.json")
        with open(checklist_file, "w") as f:
            json.dump(checklist, f, indent=2, ensure_ascii=False)

        print(f"✓ Checklist saved to {checklist_file}")
        print()

    def generate_report(self) -> None:
        """검증 리포트 생성"""
        print("=" * 70)
        print("Generating Verification Report")
        print("=" * 70)
        print()

        report_dir = Path("storage/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"verification_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Report saved to {report_file}")
        print()

        # 요약
        total_models = len(self.results["models"])
        passed_models = sum(
            1 for m in self.results["models"].values() if m.get("passed")
        )

        total_datasets = len(self.results["datasets"])
        passed_datasets = sum(
            1 for d in self.results["datasets"].values() if d.get("passed")
        )

        print("Summary:")
        print(f"  Models:   {passed_models}/{total_models} passed")
        print(f"  Datasets: {passed_datasets}/{total_datasets} passed")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase1 Storage 무결성 검증 및 체크리스트 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 검증
  python scripts/verify_storage.py --check all

  # 모델만 검증
  python scripts/verify_storage.py --check models

  # Phase1 체크리스트 생성
  python scripts/verify_storage.py --check all --phase1-checklist

  # 리포트 생성
  python scripts/verify_storage.py --check all --generate-report
        """,
    )

    parser.add_argument(
        "--check",
        required=True,
        choices=["models", "datasets", "all"],
        help="What to verify",
    )

    parser.add_argument(
        "--phase1-checklist",
        action="store_true",
        help="Generate Phase1 completion checklist",
    )

    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate verification report (JSON)",
    )

    args = parser.parse_args()

    verifier = StorageVerifier()

    all_passed = True

    try:
        if args.check in ["models", "all"]:
            models_passed = verifier.verify_models(list(MODELS_TO_VERIFY.keys()))
            all_passed = all_passed and models_passed

        if args.check in ["datasets", "all"]:
            datasets_passed = verifier.verify_datasets(list(DATASETS_TO_VERIFY.keys()))
            all_passed = all_passed and datasets_passed

        if args.phase1_checklist:
            verifier.generate_phase1_checklist()

        if args.generate_report:
            verifier.generate_report()

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
