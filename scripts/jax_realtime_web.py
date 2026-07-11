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

from magenta_rt.live_compat import APPLE_LIVE_MUSICCOCA_MASKED_TAIL_LEVELS
from magenta_rt.realtime import StereoRingBuffer
from magenta_rt.realtime_server import CHANNELS
from magenta_rt.realtime_server import encode_pcm_f32le
from magenta_rt.realtime_server import FRAME_DURATION_SECONDS
from magenta_rt.realtime_server import FRAME_RATE
from magenta_rt.realtime_server import FRAME_SAMPLES
from magenta_rt.realtime_server import JaxRealtimeProducer
from magenta_rt.realtime_server import MAX_PROMPTS
from magenta_rt.realtime_server import MidiConditioning
from magenta_rt.realtime_server import PromptDefinition
from magenta_rt.realtime_server import PromptMixer
from magenta_rt.realtime_server import prompt_definitions_to_json
from magenta_rt.realtime_server import SAMPLE_RATE
from magenta_rt.realtime_server import validate_prompt
from magenta_rt.realtime_server import validate_prompt_definitions


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
      "--apple-live-parity",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "match Apple live conditioning/output: mask the final 6 MusicCoCa "
          "RVQ levels and keep decoder audio as float32 (default: enabled)"
      ),
  )
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


def create_app(
    *,
    mrt,
    model_name: str,
    buffer_frames: int,
    prime_frames: int,
    apple_live_parity: bool = False,
):
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
        "apple_live_parity": apple_live_parity,
        "musiccoca_masked_tail_levels": (
            APPLE_LIVE_MUSICCOCA_MASKED_TAIL_LEVELS
            if apple_live_parity
            else 0
        ),
        "decoder_output": "float32" if apple_live_parity else "int16",
        "classifier_free_guidance_branches": 2 if apple_live_parity else 0,
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
          apple_live_parity=apple_live_parity,
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
    apple_live_parity: bool,
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
            "protocol_version": 3,
            "model": model_name,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "frame_rate": FRAME_RATE,
            "frame_samples": FRAME_SAMPLES,
            "pcm_format": "float32le",
            "server_buffer_frames": buffer_frames,
            "server_prime_frames": prime_frames,
            "max_prompts": MAX_PROMPTS,
            "apple_live_parity": apple_live_parity,
            "musiccoca_masked_tail_levels": (
                APPLE_LIVE_MUSICCOCA_MASKED_TAIL_LEVELS
                if apple_live_parity
                else 0
            ),
            "decoder_output": "float32" if apple_live_parity else "int16",
            "classifier_free_guidance_branches": (
                2 if apple_live_parity else 0
            ),
        }
    )
    start_message = await asyncio.wait_for(
        websocket.receive_json(), timeout=30.0
    )
    if start_message.get("type") != "start":
      raise ValueError("the first client message must have type 'start'")
    initial_definitions = _definitions_from_message(start_message)

    await send_json(
        {
            "type": "status",
            "stage": "encoding_prompt",
            "message": (
                f"Encoding {len(initial_definitions)} initial prompt(s)"
            ),
        }
    )
    embedding_cache = {}
    initial_embeddings = await _encode_definitions(
        mrt=mrt,
        definitions=initial_definitions,
        embedding_cache=embedding_cache,
    )
    prompt = PromptMixer(initial_definitions, initial_embeddings)
    midi = MidiConditioning()
    ring_buffer = StereoRingBuffer(buffer_frames * FRAME_SAMPLES)
    producer = JaxRealtimeProducer(
        mrt=mrt,
        prompt=prompt,
        ring_buffer=ring_buffer,
        midi=midi,
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
            "prompts": prompt_definitions_to_json(initial_definitions),
            "prompt_revision": 0,
            "browser_buffer_frames": 3,
            "browser_max_buffer_frames": 6,
            "midi": {
                "enabled": False,
                "auto_strum": True,
                "unmask_width": 4,
            },
        }
    )
    LOGGER.info(
        "Browser stream started: prompts=%s, server buffer=%d frames",
        [definition.text for definition in initial_definitions],
        buffer_frames,
    )

    sender = asyncio.create_task(
        _send_audio(
            websocket=websocket,
            send_lock=send_lock,
            send_json=send_json,
            ring_buffer=ring_buffer,
            producer=producer,
            midi=midi,
        ),
        name="mrt2-websocket-audio-sender",
    )
    receiver = asyncio.create_task(
        _receive_controls(
            websocket=websocket,
            send_json=send_json,
            mrt=mrt,
            prompt=prompt,
            embedding_cache=embedding_cache,
            midi=midi,
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
    *, websocket, send_lock, send_json, ring_buffer, producer, midi
) -> None:
  sequence = 0
  underrun_frames = 0
  missing_samples_total = 0
  recovery_waits = 0
  recovery_wait_ms_total = 0.0
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

    if ring_buffer.available < FRAME_SAMPLES:
      wait_started = time.perf_counter()
      recovery_waits += 1
      await asyncio.to_thread(
          ring_buffer.wait_for_available, FRAME_SAMPLES, 0.02
      )
      recovery_wait_ms_total += (
          time.perf_counter() - wait_started
      ) * 1000.0

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
      midi_stats = midi.snapshot()
      await send_json(
          {
              "type": "metrics",
              "sequence": sequence,
              "server_buffer_ms": (
                  1000.0 * ring_buffer.available / SAMPLE_RATE
              ),
              "server_underrun_frames": underrun_frames,
              "server_missing_samples": missing_samples_total,
              "server_recovery_waits": recovery_waits,
              "server_recovery_wait_ms": recovery_wait_ms_total,
              "generation_ms_latest": stats.latest_generation_ms,
              "generation_ms_mean_10s": stats.mean_generation_ms,
              "generation_ms_max": stats.max_generation_ms,
              "generation_deadline_misses": stats.deadline_misses,
              "prompt_revision": stats.prompt_revision,
              "midi_enabled": midi_stats.enabled,
              "midi_active_notes": list(midi_stats.active_notes),
              "midi_revision": midi_stats.revision,
          }
      )

    # Do not send a long catch-up burst after event-loop or network stalls.
    now = loop.time()
    if next_deadline < now - FRAME_DURATION_SECONDS:
      next_deadline = now


def _definitions_from_message(message) -> tuple[PromptDefinition, ...]:
  """Read protocol-v2 prompts, retaining protocol-v1 compatibility."""
  if "prompts" in message:
    return validate_prompt_definitions(message.get("prompts"))
  text = validate_prompt(message.get("prompt"))
  return (PromptDefinition(prompt_id=0, text=text, weight=1.0),)


async def _encode_definitions(
    *, mrt, definitions, embedding_cache, send_json=None, request_id=None
):
  """Encode only unseen prompt texts and return embeddings in slot order."""
  missing_texts = list(
      dict.fromkeys(
          definition.text
          for definition in definitions
          if definition.text not in embedding_cache
      )
  )
  for index, text in enumerate(missing_texts):
    if send_json is not None:
      await send_json(
          {
              "type": "prompt_encoding_progress",
              "prompt": text,
              "completed": index,
              "total": len(missing_texts),
              "request_id": request_id,
          }
      )
    embedding_cache[text] = await asyncio.to_thread(
        mrt.embed_style, text, use_mapper=True, seed=0
    )
  return [embedding_cache[definition.text] for definition in definitions]


async def _receive_controls(
    *, websocket, send_json, mrt, prompt, embedding_cache, midi
) -> None:
  while True:
    message = await websocket.receive_json()
    message_type = message.get("type")
    if message_type == "stop":
      return

    if message_type == "midi_config":
      try:
        midi_snapshot = midi.configure(
            enabled=message.get("enabled"),
            auto_strum=message.get("auto_strum"),
            unmask_width=message.get("unmask_width"),
        )
      except ValueError as exc:
        await send_json(
            {"type": "control_error", "message": str(exc)}
        )
        continue
      await send_json(
          {
              "type": "midi_config_applied",
              "enabled": midi_snapshot.enabled,
              "auto_strum": midi_snapshot.auto_strum,
              "unmask_width": midi_snapshot.unmask_width,
          }
      )
      continue

    if message_type == "midi_note":
      try:
        note_on = message.get("on")
        if not isinstance(note_on, bool):
          raise ValueError("MIDI note on must be a boolean")
        if note_on:
          midi.note_on(message.get("pitch"))
        else:
          midi.note_off(message.get("pitch"))
      except ValueError as exc:
        await send_json(
            {"type": "control_error", "message": str(exc)}
        )
      continue

    if message_type == "midi_all_notes_off":
      midi.all_notes_off()
      continue

    if message_type == "set_weights":
      try:
        prompt.update_weights(message.get("weights"))
      except ValueError as exc:
        await send_json(
            {
                "type": "control_error",
                "message": str(exc),
                "request_id": message.get("request_id"),
            }
        )
      continue

    if message_type not in ("set_prompts", "set_prompt"):
      await send_json(
          {
              "type": "control_error",
              "message": f"unsupported control message: {message_type!r}",
          }
      )
      continue

    try:
      definitions = _definitions_from_message(message)
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
            "prompts": prompt_definitions_to_json(definitions),
            "uncached_count": len(
                {
                    definition.text
                    for definition in definitions
                    if definition.text not in embedding_cache
                }
            ),
            "request_id": request_id,
        }
    )
    started = time.perf_counter()
    embeddings = await _encode_definitions(
        mrt=mrt,
        definitions=definitions,
        embedding_cache=embedding_cache,
        send_json=send_json,
        request_id=request_id,
    )
    snapshot = prompt.configure(definitions, embeddings)
    await send_json(
        {
            "type": "prompt_applied",
            "prompts": prompt_definitions_to_json(definitions),
            "prompt_revision": snapshot.revision,
            "encoding_ms": (time.perf_counter() - started) * 1000.0,
            "request_id": request_id,
        }
    )
    LOGGER.info(
        "Prompt bank revision %d applied without resetting state: %s",
        snapshot.revision,
        [definition.text for definition in definitions],
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
        'uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.34"'
    ) from exc

  logging.basicConfig(level=logging.INFO, force=True)
  LOGGER.info("Loading %s and compiling the streaming step", args.model)
  masked_tail_levels = (
      APPLE_LIVE_MUSICCOCA_MASKED_TAIL_LEVELS
      if args.apple_live_parity
      else 0
  )
  mrt = MagentaRT2Jax(
      size=args.model,
      checkpoint=args.checkpoint,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
      musiccoca_masked_tail_levels=masked_tail_levels,
      int16_outputs=not args.apple_live_parity,
      num_cfgs=2 if args.apple_live_parity else 0,
  )
  LOGGER.info(
      "Apple live parity: %s | MusicCoCa masked tail=%d/12 | "
      "CFG branches=%d (batch=%dx) | decoder=%s",
      "enabled" if args.apple_live_parity else "disabled",
      masked_tail_levels,
      2 if args.apple_live_parity else 0,
      3 if args.apple_live_parity else 1,
      "float32" if args.apple_live_parity else "legacy int16",
  )
  app = create_app(
      mrt=mrt,
      model_name=args.model,
      buffer_frames=args.buffer_frames,
      prime_frames=args.prime_frames,
      apple_live_parity=args.apple_live_parity,
  )
  LOGGER.info(
      "Open http://%s:%d after creating an SSH tunnel",
      args.host,
      args.port,
  )
  uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
  main()
