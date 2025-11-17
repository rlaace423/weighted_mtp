"""JSONL 파일에서 메타데이터만 추출하여 인덱스 파일 생성

전체 JSONL을 한 번 스캔하여 각 샘플의 is_correct, difficulty 정보만 추출합니다.
이를 통해 런타임에 전체 데이터를 로드하지 않고 필요한 샘플만 선택할 수 있습니다.

Usage:
    python scripts/extract_metadata.py
    python scripts/extract_metadata.py --dataset codecontests
    python scripts/extract_metadata.py --dataset codecontests --split train
"""

import json
import logging
from pathlib import Path
from typing import Optional
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_metadata_from_jsonl(
    jsonl_path: Path,
    output_path: Path,
) -> dict:
    """JSONL 파일에서 메타데이터 추출

    Args:
        jsonl_path: 입력 JSONL 파일 경로
        output_path: 출력 메타데이터 JSON 파일 경로

    Returns:
        추출된 메타데이터 (샘플 통계)
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 파일이 존재하지 않습니다: {jsonl_path}")

    logger.info(f"메타데이터 추출 시작: {jsonl_path}")

    metadata_list = []
    stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "difficulty_dist": {},
        "has_is_correct": False,
        "has_difficulty": False,
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0 and idx > 0:
                logger.info(f"  진행 중: {idx:,} 샘플")

            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"라인 {idx} 파싱 오류: {e}")
                continue

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

                # 난이도 분포 집계
                diff_str = str(difficulty)
                stats["difficulty_dist"][diff_str] = stats["difficulty_dist"].get(diff_str, 0) + 1

            metadata_list.append(meta)
            stats["total"] += 1

    logger.info(f"메타데이터 추출 완료: {stats['total']:,} 샘플")

    # 통계 출력
    if stats["has_is_correct"]:
        logger.info(
            f"  is_correct 분포: "
            f"correct={stats['correct']:,}, "
            f"incorrect={stats['incorrect']:,}"
        )

    if stats["has_difficulty"]:
        sorted_diff = sorted(stats["difficulty_dist"].items(), key=lambda x: int(x[0]))
        diff_str = ", ".join([f"{k}:{v:,}" for k, v in sorted_diff[:5]])  # 상위 5개만
        logger.info(f"  difficulty 분포 (상위 5개): {diff_str}")

    # 메타데이터 저장
    output_data = {
        "metadata": metadata_list,
        "stats": stats,
        "source_file": str(jsonl_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"메타데이터 저장 완료: {output_path}")
    logger.info(f"  파일 크기: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return stats


def extract_dataset_metadata(
    dataset_name: str,
    split: Optional[str] = None,
    use_small: bool = False,
):
    """데이터셋의 메타데이터 추출

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval)
        split: 특정 스플릿만 추출 (None이면 전체)
        use_small: 작은 테스트용 데이터셋 사용 여부
    """
    if use_small:
        base_dir = Path("storage/datasets_local_small")
        dataset_dir = base_dir / f"{dataset_name}_small"
        suffix = "_small"
    else:
        base_dir = Path("storage/datasets")
        dataset_dir = base_dir / dataset_name / "processed"
        suffix = ""

    if not dataset_dir.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리가 존재하지 않습니다: {dataset_dir}")

    # 처리할 스플릿 결정
    if split:
        splits_to_process = [split]
    else:
        splits_to_process = ["train", "validation", "test"]

    # 스플릿별로 처리
    for split_name in splits_to_process:
        # JSONL 파일 찾기
        if use_small:
            candidates = [
                dataset_dir / f"{split_name}_small.jsonl",
                dataset_dir / f"valid_small.jsonl" if split_name == "validation" else None,
            ]
        else:
            candidates = [
                dataset_dir / f"{split_name}.jsonl",
                dataset_dir / "valid.jsonl" if split_name == "validation" else None,
            ]

        jsonl_path = None
        for candidate in candidates:
            if candidate and candidate.exists():
                jsonl_path = candidate
                break

        if not jsonl_path:
            logger.warning(f"스플릿 '{split_name}'의 JSONL 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 출력 경로
        output_path = dataset_dir / f"{split_name}{suffix}_metadata.json"

        # 메타데이터 추출
        try:
            extract_metadata_from_jsonl(jsonl_path, output_path)
        except Exception as e:
            logger.error(f"메타데이터 추출 실패 ({split_name}): {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="JSONL 파일에서 메타데이터 추출"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="codecontests",
        choices=["codecontests", "mbpp", "humaneval"],
        help="데이터셋 이름 (기본: codecontests)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="특정 스플릿만 추출 (기본: 전체)"
    )
    parser.add_argument(
        "--use-small",
        action="store_true",
        help="작은 테스트용 데이터셋 사용"
    )

    args = parser.parse_args()

    logger.info(f"===== 메타데이터 추출 시작: {args.dataset} =====")

    extract_dataset_metadata(
        dataset_name=args.dataset,
        split=args.split,
        use_small=args.use_small,
    )

    logger.info("===== 메타데이터 추출 완료 =====")


if __name__ == "__main__":
    main()
