from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.config import load_config, validate_config
from lingbot_so_arm101.runner import run_client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_so_arm101.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"Config error: {error}", file=sys.stderr)
        return 1
    run_client(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

