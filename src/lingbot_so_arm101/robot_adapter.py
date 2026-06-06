from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class RobotLike(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: np.ndarray) -> None: ...


@dataclass
class DryRunRobot:
    action_dim: int = 7

    def connect(self) -> None:
        print("[dry-run] robot connected")

    def disconnect(self) -> None:
        print("[dry-run] robot disconnected")

    def get_observation(self) -> dict[str, Any]:
        return {
            "observation.state": np.zeros(self.action_dim, dtype=np.float32),
            "observation.images.camera_top": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation.images.camera_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
            "task": "dry run",
        }

    def send_action(self, action: np.ndarray) -> None:
        print(f"[dry-run] action={np.array2string(action, precision=4)}")


def create_lerobot_adapter(raw_config: dict[str, Any]) -> RobotLike:
    """Create a LeRobot-backed adapter.

    LeRobot hardware APIs move between versions, so this function keeps the
    integration point narrow. Fill in the factory below to match the exact
    LeRobot package installed on the robot computer.
    """
    try:
        from lerobot.common.robot_devices.robots.factory import make_robot  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not importable. Install/use your LeRobot environment, "
            "or run with --dry-run to test the LingBot-VLA client path."
        ) from exc

    robot = make_robot(raw_config)
    for method in ("connect", "disconnect", "get_observation", "send_action"):
        if not hasattr(robot, method):
            raise RuntimeError(f"LeRobot object is missing required method: {method}")
    return robot

