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

"""Core primitives for serving stateful JAX audio to a web client."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import numpy as np

from magenta_rt.realtime import BufferClosedError
from magenta_rt.realtime import StereoRingBuffer


SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_RATE = 25
FRAME_SAMPLES = SAMPLE_RATE // FRAME_RATE
FRAME_DURATION_SECONDS = 1.0 / FRAME_RATE
FRAME_DURATION_MS = 1000.0 * FRAME_DURATION_SECONDS


@dataclass(frozen=True)
class PromptSnapshot:
  """An immutable prompt/style pair safe to retain for one model step."""

  text: str
  style: Any
  revision: int


class PromptConditioning:
  """Atomically replaceable prompt conditioning for an inference thread."""

  def __init__(self, text: str, style: Any):
    text = validate_prompt(text)
    self._snapshot = PromptSnapshot(text=text, style=style, revision=0)
    self._lock = threading.Lock()

  def get(self) -> PromptSnapshot:
    with self._lock:
      return self._snapshot

  def update(self, text: str, style: Any) -> PromptSnapshot:
    text = validate_prompt(text)
    with self._lock:
      self._snapshot = PromptSnapshot(
          text=text,
          style=style,
          revision=self._snapshot.revision + 1,
      )
      return self._snapshot


@dataclass(frozen=True)
class ProducerStats:
  """A consistent snapshot of inference-thread timing counters."""

  frames_generated: int
  deadline_misses: int
  latest_generation_ms: float
  mean_generation_ms: float
  max_generation_ms: float
  prompt_revision: int


class JaxRealtimeProducer(threading.Thread):
  """Continuously generate MRT2 frames while preserving model state."""

  def __init__(
      self,
      *,
      mrt,
      prompt: PromptConditioning,
      ring_buffer: StereoRingBuffer,
      logger: logging.Logger | None = None,
  ):
    super().__init__(name="mrt2-jax-web-producer", daemon=True)
    self._mrt = mrt
    self._prompt = prompt
    self._ring_buffer = ring_buffer
    self._logger = logger or logging.getLogger(__name__)
    self._stop_event = threading.Event()
    self._stats_lock = threading.Lock()
    self._recent_times_ms: deque[float] = deque(maxlen=250)
    self._frames_generated = 0
    self._deadline_misses = 0
    self._latest_generation_ms = 0.0
    self._max_generation_ms = 0.0
    self._prompt_revision = 0
    self.error: BaseException | None = None

  def stop(self) -> None:
    """Stop inference and wake a producer blocked on a full buffer."""
    self._stop_event.set()
    self._ring_buffer.close()

  def stats(self) -> ProducerStats:
    with self._stats_lock:
      mean_ms = (
          sum(self._recent_times_ms) / len(self._recent_times_ms)
          if self._recent_times_ms
          else 0.0
      )
      return ProducerStats(
          frames_generated=self._frames_generated,
          deadline_misses=self._deadline_misses,
          latest_generation_ms=self._latest_generation_ms,
          mean_generation_ms=mean_ms,
          max_generation_ms=self._max_generation_ms,
          prompt_revision=self._prompt_revision,
      )

  def run(self) -> None:
    state = None
    try:
      while not self._stop_event.is_set():
        prompt = self._prompt.get()
        step_start = time.perf_counter()
        waveform, state = self._mrt.generate(
            style=prompt.style,
            frames=1,
            state=state,
        )
        generation_ms = (time.perf_counter() - step_start) * 1000.0
        samples = validate_audio_frame(waveform)

        if not self._ring_buffer.write(samples, timeout=0.1):
          continue

        with self._stats_lock:
          self._frames_generated += 1
          self._deadline_misses += int(generation_ms > FRAME_DURATION_MS)
          self._latest_generation_ms = generation_ms
          self._max_generation_ms = max(
              self._max_generation_ms, generation_ms
          )
          self._prompt_revision = prompt.revision
          self._recent_times_ms.append(generation_ms)
    except BufferClosedError:
      # Expected when the WebSocket disconnects while the producer is blocked.
      pass
    except BaseException as exc:  # Propagate worker errors to the event loop.
      self.error = exc
      self._logger.exception("JAX real-time producer failed")
    finally:
      self._ring_buffer.close()


def validate_prompt(value: object, *, max_length: int = 500) -> str:
  """Validate a prompt received from an untrusted WebSocket client."""
  if not isinstance(value, str):
    raise ValueError("prompt must be a string")
  prompt = value.strip()
  if not prompt:
    raise ValueError("prompt must not be empty")
  if len(prompt) > max_length:
    raise ValueError(f"prompt must be at most {max_length} characters")
  return prompt


def validate_audio_frame(waveform) -> np.ndarray:
  """Return one host float32 MRT2 frame after validating its format."""
  if waveform.sample_rate != SAMPLE_RATE:
    raise RuntimeError(
        f"expected {SAMPLE_RATE} Hz, got {waveform.sample_rate} Hz"
    )
  if waveform.num_channels != CHANNELS:
    raise RuntimeError(
        f"expected {CHANNELS} channels, got {waveform.num_channels}"
    )
  if waveform.num_samples != FRAME_SAMPLES:
    raise RuntimeError(
        f"expected {FRAME_SAMPLES} samples, got {waveform.num_samples}"
    )
  samples = np.asarray(waveform.samples, dtype=np.float32)
  if samples.shape != (FRAME_SAMPLES, CHANNELS):
    raise RuntimeError(
        "unexpected MRT2 frame shape: "
        f"{samples.shape}; expected {(FRAME_SAMPLES, CHANNELS)}"
    )
  return samples


def encode_pcm_f32le(samples: np.ndarray) -> bytes:
  """Encode interleaved stereo samples as little-endian float32 PCM."""
  samples = np.asarray(samples)
  if samples.shape != (FRAME_SAMPLES, CHANNELS):
    raise ValueError(
        f"expected {(FRAME_SAMPLES, CHANNELS)}, got {samples.shape}"
    )
  return np.ascontiguousarray(samples, dtype="<f4").tobytes()
