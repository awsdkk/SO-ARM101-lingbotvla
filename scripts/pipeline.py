from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.pipeline import build_stage_commands, load_pipeline_config, run_commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Print or run SO-ARM101 LingBot-VLA pipeline commands.")
    parser.add_argument("--config", default="configs/pipeline_so_arm101.yaml")
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "merge_v21",
            "convert_v30",
            "quality",
            "norm",
            "train",
            "eval",
            "serve",
            "local_serve",
            "local_stack",
            "deploy_client",
        ],
        required=True,
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of only printing them.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    commands = build_stage_commands(config, args.stage)
    if args.execute:
        run_commands(commands, cwd=config.official_repo)
    else:
        for command in commands:
            print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
