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
import secrets
import time

from magenta_rt.live_compat import APPLE_LIVE_MUSICCOCA_MASKED_TAIL_LEVELS
from magenta_rt.realtime import StereoRingBuffer
from magenta_rt.realtime_server import CHANNELS
from magenta_rt.realtime_server import encode_pcm_f32le
from magenta_rt.realtime_server import DrumConditioning
from magenta_rt.realtime_server import FRAME_DURATION_SECONDS
from magenta_rt.realtime_server import FRAME_RATE
from magenta_rt.realtime_server import FRAME_SAMPLES
from magenta_rt.realtime_server import JaxRealtimeProducer
from magenta_rt.realtime_server import MAX_PROMPTS
from magenta_rt.realtime_server import MidiConditioning
from magenta_rt.realtime_server import PerformanceConditioning
from magenta_rt.realtime_server import PromptDefinition
from magenta_rt.realtime_server import prompt_definitions_to_json
from magenta_rt.realtime_server import SAMPLE_RATE
from magenta_rt.realtime_server import SamplingConditioning
from magenta_rt.realtime_server import validate_prompt
from magenta_rt.realtime_server import validate_prompt_definitions
from magenta_rt.realtime_server import WavRecorder
from magenta_rt.realtime_server import apply_solo_gate
from magenta_rt.style_encoding import StyleEncoder
from magenta_rt.prompt_updates import PromptUpdates
from magenta_rt.visual_conditioning import VisualConditioning, normalize_ratio
from magenta_rt.visual_session import VisualSession
from magenta_rt.vlm import DEFAULT_MODEL, MODELS, MAX_IMAGE_BYTES


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
  parser.add_argument("--ollama-host", default="http://127.0.0.1:21434")
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
    ollama_host: str = "http://127.0.0.1:21434",
):
  """Create a FastAPI app around an already compiled MRT2 system."""
  from fastapi import FastAPI  # pylint: disable=import-outside-toplevel
  from fastapi import WebSocket  # pylint: disable=import-outside-toplevel
  from fastapi import Request, HTTPException
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

  encoder = StyleEncoder(mrt)
  sessions = {}
  inference_lock = asyncio.Lock()

  @contextlib.asynccontextmanager
  async def lifespan(app):
    try:
      yield
    finally:
      for session in tuple(sessions.values()):
        await session.close()
      encoder.close()

  app = FastAPI(title="MRT2 Linux Real-time", version="0.2.0", lifespan=lifespan)
  app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
  stream_lock = asyncio.Lock()

  @app.post("/api/visual/frame/{token}", status_code=202)
  async def visual_frame(token: str, generation: int, sequence: int, request: Request):
    session = sessions.get(token)
    if session is None or session.closed:
      raise HTTPException(409, "audio session is not ready")
    data = bytearray()
    async for chunk in request.stream():
      if len(data) + len(chunk) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image exceeds 8 MiB")
      data.extend(chunk)
    try:
      session.submit(bytes(data), generation, sequence)
    except ValueError as exc:
      raise HTTPException(409, str(exc)) from exc
    return {"accepted": True, "generation": generation, "sequence": sequence}

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
          encoder=encoder,
          visual_sessions=sessions,
          inference_lock=inference_lock,
          ollama_host=ollama_host,
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
    encoder=None,
    visual_sessions=None,
    inference_lock=None,
    ollama_host="http://127.0.0.1:21434",
) -> None:
  ring_buffer = None
  producer = None
  visual = None
  updates = None
  tasks = []
  token = secrets.token_urlsafe(24)
  own_encoder = encoder is None
  encoder = encoder or StyleEncoder(mrt)
  sampling = SamplingConditioning(
      temperature=getattr(mrt, "temperature", 1.1),
      top_k=getattr(mrt, "top_k", 50),
  )
  recorder = WavRecorder("outputs")
  performance = PerformanceConditioning(
      base_cfg_notes=getattr(mrt, "cfg_notes", 2.4)
  )
  send_lock = asyncio.Lock()

  async def send_json(message: dict) -> None:
    async with send_lock:
      await websocket.send_json(message)

  try:
    await send_json(
        {
            "type": "hello",
            "protocol_version": 6,
            "visual_session": token,
            "visual_models": {key: value[1] for key, value in MODELS.items()},
            "visual_default_model": DEFAULT_MODEL,
            "model": model_name,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "frame_rate": FRAME_RATE,
            "frame_samples": FRAME_SAMPLES,
            "pcm_format": "float32le",
            "server_buffer_frames": buffer_frames,
            "server_prime_frames": prime_frames,
            "max_prompts": MAX_PROMPTS,
            "sampling_ranges": {
                "temperature": {"min": 0.0, "max": 3.0},
                "top_k": {"min": 1, "max": 1024},
            },
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
    initial_visual = start_message.get("visual") or {}
    if not isinstance(initial_visual, dict):
      raise ValueError("visual settings must be an object")
    initial_ratio = normalize_ratio(initial_visual.get("ratio", 2))
    initial_definitions = _definitions_from_message(
        start_message, allow_zero=initial_ratio == 0
    )
    initial_sampling = start_message.get("sampling")
    if initial_sampling is not None:
      if not isinstance(initial_sampling, dict):
        raise ValueError("sampling must be an object")
      sampling.configure(
          temperature=initial_sampling.get("temperature"),
          top_k=initial_sampling.get("top_k"),
      )
    initial_recording = start_message.get("recording", False)
    if not isinstance(initial_recording, bool):
      raise ValueError("recording must be a boolean")
    if initial_recording:
      recorder.start()
    performance.configure(
        mode=start_message.get("performance_mode", "jam")
    )

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
    initial_embeddings = await encoder.encode([d.text for d in initial_definitions])
    prompt = VisualConditioning(initial_definitions, initial_embeddings, ratio=initial_ratio)
    updates = PromptUpdates(prompt, encoder, send_json)
    visual = VisualSession(prompt, encoder, send_json, host=ollama_host,
                           inference_lock=inference_lock or asyncio.Lock())
    visual.set_performance(performance.snapshot().mode == "solo")
    if visual_sessions is not None:
      visual_sessions[token] = visual
    if start_message.get("visual") is not None:
      await visual.configure(start_message["visual"])
      visual.restore(start_message.get("visual_tags"))
    midi = MidiConditioning(
        unmask_width=(127 if performance.snapshot().mode == "solo" else 0)
    )
    drums = DrumConditioning()
    ring_buffer = StereoRingBuffer(buffer_frames * FRAME_SAMPLES)
    producer = JaxRealtimeProducer(
        mrt=mrt,
        prompt=prompt,
        ring_buffer=ring_buffer,
        midi=midi,
        drums=drums,
        sampling=sampling,
        performance=performance,
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
            "prompts": prompt_definitions_to_json(
                initial_definitions, allow_zero=prompt.allow_zero_weights
            ),
            "prompt_revision": prompt.get().revision,
            "browser_buffer_frames": 3,
            "browser_max_buffer_frames": 6,
            "midi": {
                "enabled": False,
                "auto_strum": True,
                "unmask_width": midi.snapshot().unmask_width,
            },
            "drums": {"no_drums": False},
            "sampling": {
                "temperature": sampling.snapshot().temperature,
                "top_k": sampling.snapshot().top_k,
            },
            "recording": {
                "active": recorder.snapshot().active,
                "path": recorder.snapshot().path,
                "duration_seconds": recorder.snapshot().duration_seconds,
            },
            "performance_mode": performance.snapshot().mode,
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
            drums=drums,
            sampling=sampling,
            recorder=recorder,
            performance=performance,
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
            visual=visual,
            updates=updates,
            midi=midi,
            drums=drums,
            sampling=sampling,
            recorder=recorder,
            performance=performance,
        ),
        name="mrt2-websocket-control-receiver",
    )
    tasks = [sender, receiver]
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
    for task in tasks:
      if not task.done():
        task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)
    if visual_sessions is not None:
      visual_sessions.pop(token, None)
    if updates is not None:
      await updates.close()
    if visual is not None:
      await visual.close()
    if own_encoder:
      encoder.close()
    if producer is not None:
      producer.stop()
      await asyncio.to_thread(producer.join, 5.0)
      if producer.is_alive():
        LOGGER.error("JAX producer did not stop within five seconds")
    elif ring_buffer is not None:
      ring_buffer.close()
    recording_was_active = recorder.snapshot().active
    recording = recorder.stop()
    if recording_was_active and recording.path is not None:
      LOGGER.info(
          "Recording saved: %s (%.2f s)",
          recording.path,
          recording.duration_seconds,
      )
    with contextlib.suppress(Exception):
      await websocket.close()
    LOGGER.info("Browser stream stopped")


async def _send_audio(
    *, websocket, send_lock, send_json, ring_buffer, producer, midi, drums,
    sampling, recorder, performance
) -> None:
  sequence = 0
  underrun_frames = 0
  missing_samples_total = 0
  recovery_waits = 0
  recovery_wait_ms_total = 0.0
  loop = asyncio.get_running_loop()
  next_deadline = loop.time()
  solo_gate = 1.0

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
    performance_stats = performance.snapshot()
    midi_now = midi.snapshot()
    samples, solo_gate = apply_solo_gate(
        samples,
        gate=solo_gate,
        note_held=bool(midi_now.active_notes),
        solo=performance_stats.mode == "solo",
    )
    recorder.write(samples)
    underrun_frames += int(missing > 0)
    missing_samples_total += missing
    async with send_lock:
      await websocket.send_bytes(encode_pcm_f32le(samples))

    sequence += 1
    next_deadline += FRAME_DURATION_SECONDS
    if sequence % FRAME_RATE == 0:
      stats = producer.stats()
      midi_stats = midi.snapshot()
      drum_stats = drums.snapshot()
      sampling_stats = sampling.snapshot()
      recording_stats = recorder.snapshot()
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
              "no_drums": drum_stats.no_drums,
              "drum_revision": drum_stats.revision,
              "temperature": sampling_stats.temperature,
              "top_k": sampling_stats.top_k,
              "sampling_revision": sampling_stats.revision,
              "recording_active": recording_stats.active,
              "recording_duration_seconds": recording_stats.duration_seconds,
              "performance_mode": performance_stats.mode,
              "solo_gate": solo_gate,
          }
      )

    # Do not send a long catch-up burst after event-loop or network stalls.
    now = loop.time()
    if next_deadline < now - FRAME_DURATION_SECONDS:
      next_deadline = now


def _definitions_from_message(message, *, allow_zero=False) -> tuple[PromptDefinition, ...]:
  """Read protocol-v2 prompts, retaining protocol-v1 compatibility."""
  if "prompts" in message:
    return validate_prompt_definitions(message.get("prompts"), allow_zero=allow_zero)
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
    *, websocket, send_json, mrt, prompt, embedding_cache, midi, drums,
    sampling, recorder, performance, visual=None, updates=None
) -> None:
  while True:
    message = await websocket.receive_json()
    message_type = message.get("type")
    if message_type == "visual_config" and visual is not None:
      try:
        await visual.configure(message)
      except (ValueError, TypeError) as exc:
        await send_json({"type": "visual_error", "generation": message.get("generation"),
                         "message": str(exc)})
      continue
    if message_type == "stop":
      recording = recorder.stop()
      if recording.path is not None:
        await send_json(
            {
                "type": "recording_saved",
                "path": recording.path,
                "duration_seconds": recording.duration_seconds,
            }
        )
        LOGGER.info(
            "Recording saved: %s (%.2f s)",
            recording.path,
            recording.duration_seconds,
        )
      return

    if message_type == "recording_config":
      enabled = message.get("enabled")
      if not isinstance(enabled, bool):
        await send_json(
            {"type": "control_error", "message": "recording enabled must be a boolean"}
        )
        continue
      recording = recorder.start() if enabled else recorder.stop()
      await send_json(
          {
              "type": "recording_config_applied",
              "active": recording.active,
              "path": recording.path,
              "duration_seconds": recording.duration_seconds,
              "recording_revision": recording.revision,
          }
      )
      if not recording.active and recording.path is not None:
        LOGGER.info(
            "Recording saved: %s (%.2f s)",
            recording.path,
            recording.duration_seconds,
        )
      continue

    if message_type == "performance_config":
      try:
        performance_snapshot = performance.configure(
            mode=message.get("mode")
        )
      except ValueError as exc:
        await send_json(
            {"type": "control_error", "message": str(exc)}
        )
        continue
      midi_snapshot = midi.snapshot()
      if visual is not None:
        visual.set_performance(performance_snapshot.mode == "solo")
      midi.configure(
          enabled=midi_snapshot.enabled,
          auto_strum=midi_snapshot.auto_strum,
          unmask_width=(
              127 if performance_snapshot.mode == "solo" else 0
          ),
      )
      await send_json(
          {
              "type": "performance_config_applied",
              "mode": performance_snapshot.mode,
              "performance_revision": performance_snapshot.revision,
          }
      )
      LOGGER.info(
          "Performance mode switched to %s without resetting streaming state",
          performance_snapshot.mode.upper(),
      )
      continue

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

    if message_type == "drum_config":
      try:
        drum_snapshot = drums.configure(
            no_drums=message.get("no_drums")
        )
      except ValueError as exc:
        await send_json(
            {"type": "control_error", "message": str(exc)}
        )
        continue
      await send_json(
          {
              "type": "drum_config_applied",
              "no_drums": drum_snapshot.no_drums,
              "drum_revision": drum_snapshot.revision,
          }
      )
      LOGGER.info(
          "No Drums %s without resetting streaming state",
          "enabled" if drum_snapshot.no_drums else "disabled",
      )
      continue

    if message_type == "sampling_config":
      try:
        sampling_snapshot = sampling.configure(
            temperature=message.get("temperature"),
            top_k=message.get("top_k"),
        )
      except ValueError as exc:
        await send_json(
            {"type": "control_error", "message": str(exc)}
        )
        continue
      await send_json(
          {
              "type": "sampling_config_applied",
              "temperature": sampling_snapshot.temperature,
              "top_k": sampling_snapshot.top_k,
              "sampling_revision": sampling_snapshot.revision,
          }
      )
      LOGGER.debug(
          "Sampling updated without resetting state: temperature=%.2f, top_k=%d",
          sampling_snapshot.temperature,
          sampling_snapshot.top_k,
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
        if updates is not None:
          updates.weight_version += 1
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
      definitions = _definitions_from_message(
          message, allow_zero=getattr(prompt, "allow_zero_weights", False)
      )
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
    if updates is not None:
      updates.submit(definitions, request_id)
      continue
    await send_json(
        {
            "type": "prompt_encoding",
            "prompts": prompt_definitions_to_json(
                definitions, allow_zero=getattr(prompt, "allow_zero_weights", False)
            ),
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
            "prompts": prompt_definitions_to_json(
                definitions, allow_zero=getattr(prompt, "allow_zero_weights", False)
            ),
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
      ollama_host=args.ollama_host,
  )
  LOGGER.info(
      "Open http://%s:%d after creating an SSH tunnel",
      args.host,
      args.port,
  )
  uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
  main()
