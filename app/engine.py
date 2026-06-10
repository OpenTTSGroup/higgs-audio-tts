from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import time
from typing import AsyncIterator, Optional

import httpx
import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class TTSEngine:
    """Thin async wrapper around a sglang-omni backend serving Higgs Audio."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._device = settings.resolved_device
        self._dtype_str = settings.higgs_dtype
        self._sample_rate = SAMPLE_RATE
        self._proc: Optional[subprocess.Popen] = None

        if settings.higgs_backend_url:
            self._base_url = settings.higgs_backend_url.rstrip("/")
            log.info("connecting to external sglang-omni backend at %s", self._base_url)
        else:
            self._base_url = f"http://127.0.0.1:{settings.higgs_internal_port}"
            self._proc = self._start_backend()

        self._wait_for_backend()
        log.info("sglang-omni backend ready at %s", self._base_url)

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_str(self) -> str:
        return self._dtype_str

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # ------------------------------------------------------------------
    # Backend lifecycle

    def _start_backend(self) -> subprocess.Popen:
        s = self._settings
        cmd = [
            "sgl-omni", "serve",
            "--model-path", s.higgs_model,
            "--port", str(s.higgs_internal_port),
        ]
        quant = s.effective_quantization
        if quant != "none":
            cmd.extend(["--quantization", quant])
        if s.higgs_tp_size > 1:
            cmd.extend(["--tp-size", str(s.higgs_tp_size)])
        env = None
        if self._device.startswith("cuda"):
            gpu_idx = self._device.split(":")[-1] if ":" in self._device else "0"
            import os
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_idx}

        log.info("launching sglang-omni: %s", " ".join(cmd))
        return subprocess.Popen(cmd, env=env)

    def _wait_for_backend(self, timeout: float = 1800.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"sglang-omni exited with code {self._proc.returncode}"
                )
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=3.0)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            except Exception:
                pass
            time.sleep(3.0)
        raise TimeoutError(
            f"sglang-omni backend did not become healthy within {timeout}s"
        )

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            log.info("stopping sglang-omni backend (pid %d)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)

    # ------------------------------------------------------------------
    # Helpers

    def _build_payload(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> dict:
        input_text = f"{instructions}{text}" if instructions else text

        payload: dict = {
            "input": input_text,
            "response_format": "pcm",
            "speed": speed,
        }

        if ref_audio:
            payload["references"] = [
                {"audio_path": ref_audio, "text": ref_text or ""}
            ]

        s = self._settings
        payload["temperature"] = temperature if temperature is not None else s.higgs_temperature
        payload["top_k"] = top_k if top_k is not None else s.higgs_top_k
        payload["max_new_tokens"] = max_new_tokens if max_new_tokens is not None else s.higgs_max_new_tokens

        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        if stream:
            payload["stream"] = True

        return payload

    @staticmethod
    def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
        if len(pcm_bytes) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Non-streaming synthesis

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float] = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **_: object,
    ) -> np.ndarray:
        payload = self._build_payload(
            text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instructions=instructions,
            speed=speed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/speech", json=payload
            )
            resp.raise_for_status()

        return self._pcm_to_float32(resp.content)

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **_: object,
    ) -> np.ndarray:
        payload = self._build_payload(
            text,
            instructions=instruct,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/speech", json=payload
            )
            resp.raise_for_status()

        return self._pcm_to_float32(resp.content)

    # ------------------------------------------------------------------
    # Streaming synthesis

    async def synthesize_realtime(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **_: object,
    ) -> AsyncIterator[np.ndarray]:
        payload = self._build_payload(
            text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instructions=instructions,
            speed=speed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            stream=True,
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{self._base_url}/v1/audio/speech", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("finish_reason") == "stop":
                        break
                    audio_data = event.get("audio") or {}
                    b64 = audio_data.get("data")
                    if b64:
                        pcm_bytes = base64.b64decode(b64)
                        if len(pcm_bytes) >= 2:
                            yield self._pcm_to_float32(pcm_bytes)
