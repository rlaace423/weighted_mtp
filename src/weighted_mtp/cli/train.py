"""학습 파이프라인 CLI 진입점"""

import argparse
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def load_config(config_path: Path, recipe_path: Path | None = None) -> dict:
    """설정 파일 로딩"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if recipe_path:
        with open(recipe_path) as f:
            recipe = yaml.safe_load(f)
        config.update(recipe)

    return config


def main():
    parser = argparse.ArgumentParser(description="Weighted MTP 학습 파이프라인")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="기본 설정 파일 경로",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        help="실험 recipe 파일 (baseline, verifiable, rho1_weighted)",
    )
    parser.add_argument(
        "--preset",
        choices=["local-light"],
        help="사전 정의된 preset",
    )
    parser.add_argument(
        "--use-micro-model",
        action="store_true",
        help="Micro 모델 사용 (로컬 테스트)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="설정만 출력하고 종료",
    )
    parser.add_argument(
        "--run-name",
        help="MLflow 실험 run 이름",
    )

    args = parser.parse_args()

    # 설정 로딩
    config = load_config(args.config, args.recipe)

    if args.preset == "local-light":
        preset_path = Path("configs/local-light.yaml")
        with open(preset_path) as f:
            preset = yaml.safe_load(f)
        config.update(preset.get("override", {}))

    if args.use_micro_model:
        config["models"]["policy"]["name"] = "micro-mtp"
        config["models"]["policy"]["path"] = "storage/models_v2/micro-mtp"

    # Dry-run
    if args.dry_run:
        console.print("[bold green]Dry-run mode: 설정 확인[/bold green]")
        console.print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        return

    # TODO: Phase 6에서 실제 파이프라인 연결
    console.print("[yellow]Phase 2: 파이프라인 미구현 (스텁)[/yellow]")
    console.print(f"실험: {config.get('experiment', {}).get('name', 'N/A')}")
    console.print(f"모델: {config['models']['policy']['name']}")
    console.print(f"데이터셋: {config.get('dataset', {}).get('name', 'N/A')}")


if __name__ == "__main__":
    main()
