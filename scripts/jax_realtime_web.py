# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve live MRT2 JAX audio to a browser over a WebSocket.

The server keeps the model's recurrent state in one inference thread, paces
48 kHz stereo float32 PCM at 25 frames/s, and accepts prompt changes without
resetting that state. Bind to localhost and use an SSH port forward when the
GPU is on a remote server.
"""

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path
import time

from magenta_rt.realtime import StereoRingBuffer
from magenta_rt.realtime_server import CHANNELS
from magenta_rt.realtime_server import encode_pcm_f32le
from magenta_rt.realtime_server import FRAME_DURATION_SECONDS
from magenta_rt.realtime_server import FRAME_RATE
from magenta_rt.realtime_server import FRAME_SAMPLES
from magenta_rt.realtime_server import JaxRealtimeProducer
from magenta_rt.realtime_server import PromptConditioning
from magenta_rt.realtime_server import SAMPLE_RATE
from magenta_rt.realtime_server import validate_prompt


LOGGER = logging.getLogger("jax_realtime_web")
WEB_ROOT = Path(__file__).resolve().parents[1] / "examples" / "linux_web"


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Stream stateful MRT2 JAX audio to a browser."
  )
  parser.add_argument("--model", default="mrt2_small")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--cfg-musiccoca", type=float, default=1.6)
  parser.add_argument("--cfg-notes", type=float, default=2.4)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument(
      "--buffer-frames",
      type=int,
      default=3,
      help="server ring-buffer capacity in 40 ms frames (default: 3)",
  )
  parser.add_argument(
      "--prime-frames",
      type=int,
      default=3,
      help="server frames generated before streaming starts (default: 3)",
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8000)
  return parser


def validate_args(args) -> None:
  if args.buffer_frames < 1:
    raise ValueError("--buffer-frames must be at least 1")
  if not 1 <= args.prime_frames <= args.buffer_frames:
    raise ValueError("--prime-frames must be within [1, --buffer-frames]")
  if not 1 <= args.port <= 65_535:
    raise ValueError("--port must be within [1, 65535]")


def create_app(*, mrt, model_name: str, buffer_frames: int, prime_frames: int):
  """Create a FastAPI app around an already compiled MRT2 system."""
  from fastapi import FastAPI  # pylint: disable=import-outside-toplevel
  from fastapi import WebSocket  # pylint: disable=import-outside-toplevel
  from fastapi import (  # pylint: disable=import-outside-toplevel
      WebSocketDisconnect,
  )
  from fastapi.responses import (  # pylint: disable=import-outside-toplevel
      FileResponse,
  )
  from fastapi.staticfiles import (  # pylint: disable=import-outside-toplevel
      StaticFiles,
  )

  if not WEB_ROOT.is_dir():
    raise RuntimeError(f"web client directory not found: {WEB_ROOT}")

  app = FastAPI(title="MRT2 Linux Real-time", version="0.1.0")
  app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
  stream_lock = asyncio.Lock()

  @app.get("/")
  async def index():
    return FileResponse(WEB_ROOT / "index.html")

  @app.get("/api/health")
  async def health():
    return {
        "status": "ok",
        "model": model_name,
        "stream_active": stream_lock.locked(),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "frame_samples": FRAME_SAMPLES,
        "server_buffer_frames": buffer_frames,
    }

  @app.websocket("/ws/audio")
  async def audio_socket(websocket: WebSocket):
    await websocket.accept()
    if stream_lock.locked():
      await websocket.send_json(
          {
              "type": "error",
              "code": "stream_busy",
              "message": "another browser is already using the model",
          }
      )
      await websocket.close(code=1013, reason="model is busy")
      return

    async with stream_lock:
      await _run_stream(
          websocket=websocket,
          mrt=mrt,
          model_name=model_name,
          buffer_frames=buffer_frames,
          prime_frames=prime_frames,
          disconnect_type=WebSocketDisconnect,
      )

  return app


async def _run_stream(
    *,
    websocket,
    mrt,
    model_name: str,
    buffer_frames: int,
    prime_frames: int,
    disconnect_type,
) -> None:
  ring_buffer = None
  producer = None
  send_lock = asyncio.Lock()

  async def send_json(message: dict) -> None:
    async with send_lock:
      await websocket.send_json(message)

  try:
    await send_json(
        {
            "type": "hello",
            "protocol_version": 1,
            "model": model_name,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "frame_rate": FRAME_RATE,
            "frame_samples": FRAME_SAMPLES,
            "pcm_format": "float32le",
            "server_buffer_frames": buffer_frames,
            "server_prime_frames": prime_frames,
        }
    )
    start_message = await asyncio.wait_for(
        websocket.receive_json(), timeout=30.0
    )
    if start_message.get("type") != "start":
      raise ValueError("the first client message must have type 'start'")
    initial_prompt = validate_prompt(start_message.get("prompt"))

    await send_json(
        {
            "type": "status",
            "stage": "encoding_prompt",
            "message": "Encoding the initial prompt",
        }
    )
    initial_style = await asyncio.to_thread(
        mrt.embed_style, initial_prompt, use_mapper=True, seed=0
    )
    prompt = PromptConditioning(initial_prompt, initial_style)
    ring_buffer = StereoRingBuffer(buffer_frames * FRAME_SAMPLES)
    producer = JaxRealtimeProducer(
        mrt=mrt,
        prompt=prompt,
        ring_buffer=ring_buffer,
        logger=LOGGER,
    )
    producer.start()

    prime_samples = prime_frames * FRAME_SAMPLES
    primed = await asyncio.to_thread(
        ring_buffer.wait_for_available, prime_samples, 120.0
    )
    if not primed:
      if producer.error is not None:
        raise producer.error
      raise RuntimeError("timed out while priming the server audio buffer")

    await send_json(
        {
            "type": "stream_started",
            "prompt": initial_prompt,
            "prompt_revision": 0,
            "browser_buffer_frames": 3,
        }
    )
    LOGGER.info(
        "Browser stream started: prompt=%r, server buffer=%d frames",
        initial_prompt,
        buffer_frames,
    )

    sender = asyncio.create_task(
        _send_audio(
            websocket=websocket,
            send_lock=send_lock,
            send_json=send_json,
            ring_buffer=ring_buffer,
            producer=producer,
        ),
        name="mrt2-websocket-audio-sender",
    )
    receiver = asyncio.create_task(
        _receive_controls(
            websocket=websocket,
            send_json=send_json,
            mrt=mrt,
            prompt=prompt,
        ),
        name="mrt2-websocket-control-receiver",
    )
    done, pending = await asyncio.wait(
        {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
      task.cancel()
    for task in pending:
      with contextlib.suppress(asyncio.CancelledError):
        await task
    for task in done:
      task.result()
  except (disconnect_type, asyncio.CancelledError):
    pass
  except Exception as exc:  # Keep protocol errors visible to the browser.
    LOGGER.exception("WebSocket stream failed")
    with contextlib.suppress(Exception):
      await send_json(
          {"type": "error", "code": "stream_failed", "message": str(exc)}
      )
  finally:
    if producer is not None:
      producer.stop()
      await asyncio.to_thread(producer.join, 5.0)
      if producer.is_alive():
        LOGGER.error("JAX producer did not stop within five seconds")
    elif ring_buffer is not None:
      ring_buffer.close()
    with contextlib.suppress(Exception):
      await websocket.close()
    LOGGER.info("Browser stream stopped")


async def _send_audio(
    *, websocket, send_lock, send_json, ring_buffer, producer
) -> None:
  sequence = 0
  underrun_frames = 0
  missing_samples_total = 0
  loop = asyncio.get_running_loop()
  next_deadline = loop.time()

  while True:
    sleep_seconds = next_deadline - loop.time()
    if sleep_seconds > 0:
      await asyncio.sleep(sleep_seconds)

    if producer.error is not None:
      raise producer.error
    if ring_buffer.closed and ring_buffer.available == 0:
      raise RuntimeError("JAX producer stopped unexpectedly")

    samples, missing = ring_buffer.read(
        FRAME_SAMPLES, zero_pad=True, timeout=0.0
    )
    underrun_frames += int(missing > 0)
    missing_samples_total += missing
    async with send_lock:
      await websocket.send_bytes(encode_pcm_f32le(samples))

    sequence += 1
    next_deadline += FRAME_DURATION_SECONDS
    if sequence % FRAME_RATE == 0:
      stats = producer.stats()
      await send_json(
          {
              "type": "metrics",
              "sequence": sequence,
              "server_buffer_ms": (
                  1000.0 * ring_buffer.available / SAMPLE_RATE
              ),
              "server_underrun_frames": underrun_frames,
              "server_missing_samples": missing_samples_total,
              "generation_ms_latest": stats.latest_generation_ms,
              "generation_ms_mean_10s": stats.mean_generation_ms,
              "generation_ms_max": stats.max_generation_ms,
              "generation_deadline_misses": stats.deadline_misses,
              "prompt_revision": stats.prompt_revision,
          }
      )

    # Do not send a long catch-up burst after event-loop or network stalls.
    now = loop.time()
    if next_deadline < now - FRAME_DURATION_SECONDS:
      next_deadline = now


async def _receive_controls(*, websocket, send_json, mrt, prompt) -> None:
  while True:
    message = await websocket.receive_json()
    message_type = message.get("type")
    if message_type == "stop":
      return
    if message_type != "set_prompt":
      await send_json(
          {
              "type": "control_error",
              "message": f"unsupported control message: {message_type!r}",
          }
      )
      continue

    try:
      text = validate_prompt(message.get("prompt"))
    except ValueError as exc:
      await send_json(
          {
              "type": "control_error",
              "message": str(exc),
              "request_id": message.get("request_id"),
          }
      )
      continue
    request_id = message.get("request_id")
    await send_json(
        {
            "type": "prompt_encoding",
            "prompt": text,
            "request_id": request_id,
        }
    )
    started = time.perf_counter()
    style = await asyncio.to_thread(
        mrt.embed_style, text, use_mapper=True, seed=0
    )
    snapshot = prompt.update(text, style)
    await send_json(
        {
            "type": "prompt_applied",
            "prompt": snapshot.text,
            "prompt_revision": snapshot.revision,
            "encoding_ms": (time.perf_counter() - started) * 1000.0,
            "request_id": request_id,
        }
    )
    LOGGER.info(
        "Prompt revision %d applied without resetting state: %r",
        snapshot.revision,
        snapshot.text,
    )


def main() -> None:
  args = make_parser().parse_args()
  try:
    validate_args(args)
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from exc

  try:
    import uvicorn  # pylint: disable=import-outside-toplevel
    from magenta_rt import (  # pylint: disable=import-outside-toplevel
        MagentaRT2Jax,
    )
  except ImportError as exc:
    raise SystemExit(
        "FastAPI server dependencies are missing. Run: "
        "uv sync --extra jax --extra realtime"
    ) from exc

  logging.basicConfig(level=logging.INFO, force=True)
  LOGGER.info("Loading %s and compiling the streaming step", args.model)
  mrt = MagentaRT2Jax(
      size=args.model,
      checkpoint=args.checkpoint,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
  )
  app = create_app(
      mrt=mrt,
      model_name=args.model,
      buffer_frames=args.buffer_frames,
      prime_frames=args.prime_frames,
  )
  LOGGER.info(
      "Open http://%s:%d after creating an SSH tunnel",
      args.host,
      args.port,
  )
  uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
  main()
