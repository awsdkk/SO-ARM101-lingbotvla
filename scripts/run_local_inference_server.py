from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.config import load_config
from lingbot_so_arm101.local_inference import get_local_inference_config, start_local_server, wait_for_health


def main() -> int:
    parser = argparse.ArgumentParser(description="Start LingBot-VLA inference on the local 4090.")
    parser.add_argument("--config", default="configs/local_so_arm101.yaml")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for /healthz.")
    args = parser.parse_args()

    config = load_config(args.config)
    local = get_local_inference_config(config)
    process = start_local_server(local)
    print(f"Started local LingBot-VLA server on ws://{local.host}:{local.port} (pid={process.pid})")
    try:
        if not args.no_wait:
            wait_for_health(local)
            print(f"Health check OK: {local.health_url}")
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

