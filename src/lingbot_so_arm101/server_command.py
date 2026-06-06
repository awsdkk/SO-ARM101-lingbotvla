from __future__ import annotations

import shlex
from typing import Any


def build_server_command(raw_config: dict[str, Any]) -> str:
    launch = raw_config.get("server_launch", {})
    args = [
        "python",
        "-m",
        "deploy.lingbot_vla_policy",
        "--model_path",
        str(launch["model_path"]),
        "--use_length",
        str(launch.get("use_length", 25)),
        "--num_denoising_step",
        str(launch.get("num_denoising_step", 10)),
        "--port",
        str(launch.get("port", 8006)),
    ]
    if launch.get("norm_path"):
        args.extend(["--norm_path", str(launch["norm_path"])])
    if launch.get("use_compile", False):
        args.append("--use_compile")

    exports = [
        "export TOKENIZERS_PARALLELISM=false",
        f"export QWEN25_PATH={shlex.quote(str(launch['qwen25_path']))}",
    ]
    command = _format_shell_command(args)
    return "\n".join(exports + [command])


def _format_shell_command(args: list[str]) -> str:
    lines: list[str] = []
    current: list[str] = []
    for arg in args:
        if arg.startswith("--") and current:
            lines.append(" ".join(shlex.quote(part) for part in current))
            current = [arg]
        else:
            current.append(arg)
    if current:
        lines.append(" ".join(shlex.quote(part) for part in current))
    return " \\\n  ".join(lines)
