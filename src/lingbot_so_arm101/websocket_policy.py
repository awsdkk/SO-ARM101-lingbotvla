from __future__ import annotations

import asyncio
from typing import Any

import websockets

from .serialization import packb, unpackb


class WebsocketClientPolicy:
    """Small client matching the LingBot-VLA websocket policy flow."""

    def __init__(
        self,
        url: str,
        timeout_seconds: float = 30,
        wrap_request: bool = False,
        wait_for_metadata: bool = True,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.wrap_request = wrap_request
        self.wait_for_metadata = wait_for_metadata

    async def infer_async(self, observation: dict[str, Any]) -> Any:
        request = {"type": "infer", "observation": observation} if self.wrap_request else observation
        return await self._send_request(request)

    async def reset_async(self, robot_config_name: str) -> Any:
        return await self._send_request({"reset": True, "robo_name": robot_config_name})

    async def _send_request(self, request: dict[str, Any]) -> Any:
        payload = packb(request)
        async with websockets.connect(self.url, compression=None, max_size=None) as websocket:
            if self.wait_for_metadata:
                metadata = await asyncio.wait_for(websocket.recv(), timeout=self.timeout_seconds)
                if not isinstance(metadata, str):
                    unpackb(metadata)
            await asyncio.wait_for(websocket.send(payload), timeout=self.timeout_seconds)
            response = await asyncio.wait_for(websocket.recv(), timeout=self.timeout_seconds)
        if isinstance(response, str):
            raise RuntimeError(f"Server returned text response: {response}")
        decoded = unpackb(response)
        if isinstance(decoded, dict) and "action" in decoded:
            return decoded["action"]
        if isinstance(decoded, dict) and "actions" in decoded:
            return decoded["actions"]
        return decoded

    def infer(self, observation: dict[str, Any]) -> Any:
        return asyncio.run(self.infer_async(observation))

    def reset(self, robot_config_name: str) -> Any:
        return asyncio.run(self.reset_async(robot_config_name))
