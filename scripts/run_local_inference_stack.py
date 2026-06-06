from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_so_arm101.config import load_config, validate_config
from lingbot_so_arm101.local_inference import get_local_inference_config, start_local_server, wait_for_health


def main() -> int:
    parser = argparse.ArgumentParser(description="Start local LingBot-VLA server, then run the SO-ARM101 client.")
    parser.add_argument("--config", default="configs/local_so_arm101.yaml")
    parser.add_argument("--dry-run-client", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"Config error: {error}", file=sys.stderr)
        return 1

    local = get_local_inference_config(config)
    server = start_local_server(local)
    print(f"Started local LingBot-VLA server on ws://{local.host}:{local.port} (pid={server.pid})")
    try:
        wait_for_health(local)
        print(f"Health check OK: {local.health_url}")
        client_cmd = [sys.executable, str(ROOT / "scripts" / "run_client.py"), "--config", str(config.path)]
        if args.dry_run_client:
            client_cmd.append("--dry-run")
        return subprocess.call(client_cmd, cwd=ROOT)
    finally:
        server.terminate()


if __name__ == "__main__":
    raise SystemExit(main())

