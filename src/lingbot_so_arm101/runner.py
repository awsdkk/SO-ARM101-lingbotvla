from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import RuntimeConfig
from .robot_adapter import DryRunRobot, RobotLike, create_lerobot_adapter
from .rtc import ActionChunkScheduler, RTCConfig
from .websocket_policy import WebsocketClientPolicy


def build_rtc_config(raw: dict[str, Any]) -> RTCConfig:
    rtc = raw.get("rtc", {})
    return RTCConfig(
        enabled=bool(rtc.get("enabled", True)),
        lookahead_ratio=float(rtc.get("lookahead_ratio", 0.4)),
        blend_steps=int(rtc.get("blend_steps", 5)),
        action_ema_alpha=float(rtc.get("action_ema_alpha", 0.4)),
        safety_margin_steps=int(rtc.get("safety_margin_steps", 2)),
        latency_window=int(rtc.get("latency_window", 5)),
    )


def create_robot(config: RuntimeConfig, dry_run: bool) -> RobotLike:
    if dry_run:
        return DryRunRobot(action_dim=14 if config.mode == "dual" else 7)
    robot_config = config.raw["dual_robot"] if config.mode == "dual" else config.raw["robot"]
    return create_lerobot_adapter(robot_config)


def run_client(config: RuntimeConfig, dry_run: bool = False) -> None:
    robot = create_robot(config, dry_run=dry_run)
    protocol = config.raw.get("protocol", {})
    policy = WebsocketClientPolicy(
        config.server_url,
        timeout_seconds=config.timeout_seconds,
        wrap_request=bool(protocol.get("wrap_request", False)),
        wait_for_metadata=bool(protocol.get("wait_for_metadata", True)),
    )
    scheduler = ActionChunkScheduler(build_rtc_config(config.raw))
    period = 1.0 / config.fps
    safety = config.raw.get("safety", {})
    max_runtime = float(safety.get("max_runtime_seconds", 120))

    robot.connect()
    try:
        if not dry_run:
            policy.reset(config.robot_config_name)

        if safety.get("require_enter_to_start", True):
            input("Press Enter to start SO-ARM101 LingBot-VLA control loop...")

        started_at = time.monotonic()
        while time.monotonic() - started_at < max_runtime:
            loop_start = time.monotonic()
            observation = robot.get_observation()
            observation["task"] = config.task

            if scheduler.should_request_next(config.actions_per_chunk):
                request_start = scheduler.note_request_start()
                actions = _infer_or_fake(policy, observation, dry_run=dry_run)
                scheduler.note_request_done(request_start)
                scheduler.push_chunk(_flatten_action_response(actions))

            action = scheduler.pop_action()
            if action is None:
                if safety.get("stop_on_empty_action", True):
                    raise RuntimeError("No action available from policy.")
                continue

            robot.send_action(np.asarray(action, dtype=np.float32))
            sleep_for = period - (time.monotonic() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        robot.disconnect()


def _infer_or_fake(
    policy: WebsocketClientPolicy,
    observation: dict[str, Any],
    dry_run: bool,
) -> np.ndarray:
    if dry_run:
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        return np.repeat(state.reshape(1, -1), repeats=25, axis=0)
    return policy.infer(observation)


def _flatten_action_response(actions: Any) -> np.ndarray:
    if isinstance(actions, np.ndarray):
        return actions.astype(np.float32)
    if isinstance(actions, dict):
        action_items = [
            (key, value)
            for key, value in actions.items()
            if key.startswith("action") and value is not None
        ]
        if not action_items:
            raise RuntimeError(f"No action tensors in policy response keys: {sorted(actions)}")
        action_items.sort(key=lambda item: item[0])
        arrays = [np.asarray(value, dtype=np.float32) for _, value in action_items]
        arrays = [array.reshape(1, -1) if array.ndim == 1 else array for array in arrays]
        return np.concatenate(arrays, axis=-1)
    return np.asarray(actions, dtype=np.float32)
