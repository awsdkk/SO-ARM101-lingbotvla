from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - friendly runtime error
    raise RuntimeError("Missing dependency: pyyaml. Install project dependencies before running.") from exc


@dataclass(frozen=True)
class RuntimeConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "single"))

    @property
    def fps(self) -> float:
        return float(self.raw.get("fps", 15))

    @property
    def task(self) -> str:
        return str(self.raw.get("task", ""))

    @property
    def server_url(self) -> str:
        server = self.raw["server"]
        host = str(server["host"]).rstrip("/")
        path = str(server.get("path", "")).strip("/")
        port = int(server["port"])
        url = f"{host}:{port}"
        return f"{url}/{path}" if path else url

    @property
    def timeout_seconds(self) -> float:
        return float(self.raw.get("server", {}).get("timeout_seconds", 30))

    @property
    def actions_per_chunk(self) -> int:
        return int(self.raw.get("lingbot", {}).get("actions_per_chunk", 25))

    @property
    def robot_config_name(self) -> str:
        default = "bi_so_arm101" if self.mode == "dual" else "so_arm101"
        return str(self.raw.get("lingbot", {}).get("robot_config_name", default))


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return RuntimeConfig(raw=raw, path=config_path)


def validate_config(config: RuntimeConfig) -> list[str]:
    errors: list[str] = []
    raw = config.raw

    if config.mode not in {"single", "dual"}:
        errors.append("mode must be 'single' or 'dual'")
    if config.fps <= 0:
        errors.append("fps must be greater than 0")
    if not config.task:
        errors.append("task must not be empty")

    server = raw.get("server", {})
    if not server.get("host"):
        errors.append("server.host is required")
    if not server.get("port"):
        errors.append("server.port is required")

    if config.mode == "single":
        robot = raw.get("robot", {})
        if not robot.get("port"):
            errors.append("robot.port is required for single mode")
    else:
        dual_robot = raw.get("dual_robot", {})
        for arm in ("left_arm", "right_arm"):
            if not dual_robot.get(arm, {}).get("port"):
                errors.append(f"dual_robot.{arm}.port is required for dual mode")

    enabled_cameras = [
        name for name, camera in raw.get("cameras", {}).items()
        if isinstance(camera, dict) and camera.get("enabled", False)
    ]
    if not enabled_cameras:
        errors.append("at least one camera must be enabled")

    return errors
