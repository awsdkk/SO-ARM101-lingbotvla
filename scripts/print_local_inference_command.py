from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.config import load_config
from lingbot_so_arm101.local_inference import format_local_server_command, get_local_inference_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the local 4090 LingBot-VLA inference server command.")
    parser.add_argument("--config", default="configs/local_so_arm101.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    local = get_local_inference_config(config)
    print(format_local_server_command(local))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

