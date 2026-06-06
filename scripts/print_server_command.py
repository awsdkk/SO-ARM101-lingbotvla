from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.config import load_config
from lingbot_so_arm101.server_command import build_server_command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_so_arm101.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(build_server_command(config.raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

