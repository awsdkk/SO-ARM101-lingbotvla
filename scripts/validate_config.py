from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.config import load_config, validate_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_so_arm101.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        print("Config validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Config OK: {config.path}")
    print(f"Mode: {config.mode}")
    print(f"Server URL: {config.server_url}")
    print(f"Task: {config.task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

