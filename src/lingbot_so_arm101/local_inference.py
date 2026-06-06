from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .config import RuntimeConfig


@dataclass(frozen=True)
class LocalInferenceConfig:
    official_repo: Path
    qwen25_path: str
    model_path: str
    norm_path: str
    cuda_visible_devices: str
    host: str
    port: int
    use_length: int
    num_denoising_step: int
    use_compile: bool
    tokenizers_parallelism: str
    health_timeout_seconds: int

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/healthz"


def get_local_inference_config(config: RuntimeConfig) -> LocalInferenceConfig:
    raw = config.raw.get("local_inference", {})
    root = config.path.parent.parent
    official_repo = Path(str(raw.get("official_repo", ".codex/vendor/lingbot-vla")))
    if not official_repo.is_absolute():
        official_repo = (root / official_repo).resolve()
    return LocalInferenceConfig(
        official_repo=official_repo,
        qwen25_path=str(raw["qwen25_path"]),
        model_path=str(raw["model_path"]),
        norm_path=str(raw.get("norm_path", "")),
        cuda_visible_devices=str(raw.get("cuda_visible_devices", "0")),
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8006)),
        use_length=int(raw.get("use_length", 25)),
        num_denoising_step=int(raw.get("num_denoising_step", 5)),
        use_compile=bool(raw.get("use_compile", True)),
        tokenizers_parallelism=str(raw.get("tokenizers_parallelism", "false")),
        health_timeout_seconds=int(raw.get("health_timeout_seconds", 180)),
    )


def build_local_server_args(local: LocalInferenceConfig) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "deploy.lingbot_vla_policy",
        "--model_path",
        local.model_path,
        "--use_length",
        str(local.use_length),
        "--num_denoising_step",
        str(local.num_denoising_step),
        "--port",
        str(local.port),
    ]
    if local.norm_path:
        args.extend(["--norm_path", local.norm_path])
    if local.use_compile:
        args.append("--use_compile")
    return args


def format_local_server_command(local: LocalInferenceConfig) -> str:
    env_lines = [
        f"$env:TOKENIZERS_PARALLELISM = {quote_ps(local.tokenizers_parallelism)}",
        f"$env:QWEN25_PATH = {quote_ps(local.qwen25_path)}",
        f"$env:CUDA_VISIBLE_DEVICES = {quote_ps(local.cuda_visible_devices)}",
    ]
    args = build_local_server_args(local)
    command = " ".join([f"& {quote_ps(args[0])}", *[quote_ps(part) for part in args[1:]]])
    return "\n".join(env_lines + [f"Set-Location -LiteralPath {quote_ps(str(local.official_repo))}", command])


def start_local_server(local: LocalInferenceConfig) -> subprocess.Popen:
    if not local.official_repo.exists():
        raise FileNotFoundError(f"official_repo does not exist: {local.official_repo}")
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = local.tokenizers_parallelism
    env["QWEN25_PATH"] = local.qwen25_path
    env["CUDA_VISIBLE_DEVICES"] = local.cuda_visible_devices
    return subprocess.Popen(build_local_server_args(local), cwd=local.official_repo, env=env)


def wait_for_health(local: LocalInferenceConfig) -> None:
    deadline = time.monotonic() + local.health_timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(local.health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Local inference server did not become healthy at {local.health_url}: {last_error}")


def quote_ps(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
