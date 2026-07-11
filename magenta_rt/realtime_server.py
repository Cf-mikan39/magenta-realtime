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
import math
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
MAX_PROMPTS = 6
MIDI_PITCHES = 128

_MIDI_IDLE = 0
_MIDI_ONSET = 1
_MIDI_SUSTAIN = 2
_MIDI_ONSET_RELEASED = 3


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
class PromptDefinition:
  """A client-visible prompt slot before its text is embedded."""

  prompt_id: int
  text: str
  weight: float


class PromptMixer:
  """Cache prompt embeddings and atomically publish their weighted blend."""

  def __init__(self, definitions, embeddings):
    definitions = tuple(definitions)
    embeddings = tuple(_validate_embeddings(definitions, embeddings))
    style = blend_style_embeddings(
        embeddings, [definition.weight for definition in definitions]
    )
    self._definitions = definitions
    self._embeddings = embeddings
    self._snapshot = PromptSnapshot(
        text=_describe_mix(definitions),
        style=style,
        revision=0,
    )
    self._lock = threading.Lock()

  def get(self) -> PromptSnapshot:
    with self._lock:
      return self._snapshot

  def definitions(self) -> tuple[PromptDefinition, ...]:
    with self._lock:
      return self._definitions

  def configure(self, definitions, embeddings) -> PromptSnapshot:
    """Atomically replace prompt slots after their embeddings are ready."""
    definitions = tuple(definitions)
    embeddings = tuple(_validate_embeddings(definitions, embeddings))
    style = blend_style_embeddings(
        embeddings, [definition.weight for definition in definitions]
    )
    with self._lock:
      self._definitions = definitions
      self._embeddings = embeddings
      self._snapshot = PromptSnapshot(
          text=_describe_mix(definitions),
          style=style,
          revision=self._snapshot.revision + 1,
      )
      return self._snapshot

  def update_weights(self, values: object) -> PromptSnapshot:
    """Apply raw 0–1 weights without re-running the text encoder."""
    with self._lock:
      weights_by_id = validate_weight_update(values, self._definitions)
      definitions = tuple(
          PromptDefinition(
              prompt_id=definition.prompt_id,
              text=definition.text,
              weight=weights_by_id[definition.prompt_id],
          )
          for definition in self._definitions
      )
      style = blend_style_embeddings(
          self._embeddings,
          [definition.weight for definition in definitions],
      )
      self._definitions = definitions
      self._snapshot = PromptSnapshot(
          text=_describe_mix(definitions),
          style=style,
          revision=self._snapshot.revision + 1,
      )
      return self._snapshot


@dataclass(frozen=True)
class MidiSnapshot:
  """Current browser-MIDI configuration and held notes."""

  enabled: bool
  auto_strum: bool
  unmask_width: int
  active_notes: tuple[int, ...]
  revision: int


class MidiConditioning:
  """Latch browser MIDI events and emit one 128-token JAX frame."""

  def __init__(
      self,
      *,
      enabled: bool = False,
      auto_strum: bool = True,
      unmask_width: int = 4,
  ):
    self._states = [_MIDI_IDLE] * MIDI_PITCHES
    self._enabled = bool(enabled)
    self._auto_strum = bool(auto_strum)
    self._unmask_width = _validate_unmask_width(unmask_width)
    self._revision = 0
    self._lock = threading.Lock()

  def configure(
      self,
      *,
      enabled: object,
      auto_strum: object,
      unmask_width: object,
  ) -> MidiSnapshot:
    if not isinstance(enabled, bool):
      raise ValueError("MIDI enabled must be a boolean")
    if not isinstance(auto_strum, bool):
      raise ValueError("MIDI auto_strum must be a boolean")
    width = _validate_unmask_width(unmask_width)
    with self._lock:
      self._enabled = enabled
      self._auto_strum = auto_strum
      self._unmask_width = width
      if not enabled:
        self._states = [_MIDI_IDLE] * MIDI_PITCHES
      self._revision += 1
      return self._snapshot_locked()

  def note_on(self, pitch: object) -> None:
    pitch = _validate_midi_pitch(pitch)
    with self._lock:
      if self._enabled:
        self._states[pitch] = _MIDI_ONSET
        self._revision += 1

  def note_off(self, pitch: object) -> None:
    pitch = _validate_midi_pitch(pitch)
    with self._lock:
      if not self._enabled:
        return
      state = self._states[pitch]
      if state == _MIDI_ONSET:
        self._states[pitch] = _MIDI_ONSET_RELEASED
      elif state == _MIDI_SUSTAIN:
        self._states[pitch] = _MIDI_IDLE
      self._revision += 1

  def all_notes_off(self) -> None:
    with self._lock:
      self._states = [_MIDI_IDLE] * MIDI_PITCHES
      self._revision += 1

  def snapshot(self) -> MidiSnapshot:
    with self._lock:
      return self._snapshot_locked()

  def frame_tokens(self) -> list[int] | None:
    """Advance onset latches exactly once and return model note tokens."""
    with self._lock:
      if not self._enabled:
        return None

      observed = tuple(self._states)
      for pitch, state in enumerate(observed):
        if state == _MIDI_ONSET:
          self._states[pitch] = _MIDI_SUSTAIN
        elif state == _MIDI_ONSET_RELEASED:
          self._states[pitch] = _MIDI_IDLE

      active = [
          pitch for pitch, state in enumerate(observed) if state != _MIDI_IDLE
      ]
      if self._unmask_width >= MIDI_PITCHES - 1:
        tokens = [0] * MIDI_PITCHES
      else:
        tokens = [-1] * MIDI_PITCHES
        for pitch in active:
          start = max(0, pitch - self._unmask_width)
          end = min(MIDI_PITCHES, pitch + self._unmask_width + 1)
          tokens[start:end] = [0] * (end - start)

      for pitch in active:
        state = observed[pitch]
        if self._auto_strum:
          tokens[pitch] = 3
        else:
          tokens[pitch] = (
              2
              if state in (_MIDI_ONSET, _MIDI_ONSET_RELEASED)
              else 1
          )
      return tokens

  def _snapshot_locked(self) -> MidiSnapshot:
    return MidiSnapshot(
        enabled=self._enabled,
        auto_strum=self._auto_strum,
        unmask_width=self._unmask_width,
        active_notes=tuple(
            pitch
            for pitch, state in enumerate(self._states)
            if state != _MIDI_IDLE
        ),
        revision=self._revision,
    )

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
      prompt: PromptConditioning | PromptMixer,
      ring_buffer: StereoRingBuffer,
      midi: MidiConditioning | None = None,
      logger: logging.Logger | None = None,
  ):
    super().__init__(name="mrt2-jax-web-producer", daemon=True)
    self._mrt = mrt
    self._prompt = prompt
    self._ring_buffer = ring_buffer
    self._midi = midi
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
        generate_kwargs = {
            "style": prompt.style,
            "frames": 1,
            "state": state,
        }
        if self._midi is not None:
          notes = self._midi.frame_tokens()
          if notes is not None:
            generate_kwargs["notes"] = notes
        waveform, state = self._mrt.generate(**generate_kwargs)
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


def validate_prompt_definitions(value: object) -> tuple[PromptDefinition, ...]:
  """Validate one to six prompt slots from a WebSocket message."""
  if not isinstance(value, list):
    raise ValueError("prompts must be an array")
  if not 1 <= len(value) <= MAX_PROMPTS:
    raise ValueError(f"prompts must contain between 1 and {MAX_PROMPTS} items")

  definitions = []
  seen_ids = set()
  for index, item in enumerate(value):
    if not isinstance(item, dict):
      raise ValueError(f"prompts[{index}] must be an object")
    prompt_id = item.get("id")
    if isinstance(prompt_id, bool) or not isinstance(prompt_id, int):
      raise ValueError(f"prompts[{index}].id must be an integer")
    if prompt_id < 0:
      raise ValueError(f"prompts[{index}].id must be non-negative")
    if prompt_id in seen_ids:
      raise ValueError(f"duplicate prompt id: {prompt_id}")
    seen_ids.add(prompt_id)

    text = validate_prompt(item.get("text"))
    weight = _validate_weight(item.get("weight"), f"prompts[{index}].weight")
    definitions.append(PromptDefinition(prompt_id, text, weight))

  if sum(definition.weight for definition in definitions) <= 0:
    raise ValueError("at least one prompt weight must be greater than zero")
  return tuple(definitions)


def validate_weight_update(
    value: object, definitions
) -> dict[int, float]:
  """Validate a complete weight update for the active prompt IDs."""
  if not isinstance(value, list):
    raise ValueError("weights must be an array")
  expected_ids = {definition.prompt_id for definition in definitions}
  weights = {}
  for index, item in enumerate(value):
    if not isinstance(item, dict):
      raise ValueError(f"weights[{index}] must be an object")
    prompt_id = item.get("id")
    if isinstance(prompt_id, bool) or not isinstance(prompt_id, int):
      raise ValueError(f"weights[{index}].id must be an integer")
    if prompt_id in weights:
      raise ValueError(f"duplicate weight id: {prompt_id}")
    weights[prompt_id] = _validate_weight(
        item.get("weight"), f"weights[{index}].weight"
    )
  if set(weights) != expected_ids:
    raise ValueError("weight update IDs must match the active prompt IDs")
  if sum(weights.values()) <= 0:
    raise ValueError("at least one prompt weight must be greater than zero")
  return weights


def normalize_prompt_weights(values) -> np.ndarray:
  """Normalize non-negative prompt weights to sum to one."""
  weights = np.asarray(values, dtype=np.float32)
  if weights.ndim != 1 or weights.size == 0:
    raise ValueError("weights must be a non-empty one-dimensional array")
  if not np.all(np.isfinite(weights)) or np.any(weights < 0):
    raise ValueError("weights must be finite and non-negative")
  total = float(np.sum(weights))
  if total <= 0:
    raise ValueError("at least one prompt weight must be greater than zero")
  return weights / total


def blend_style_embeddings(embeddings, weights) -> np.ndarray:
  """Blend MusicCoCa embeddings before MRT2 performs RVQ quantization."""
  embedding_array = np.stack(
      [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]
  )
  normalized = normalize_prompt_weights(weights)
  if embedding_array.ndim != 2:
    raise ValueError("each style embedding must be one-dimensional")
  if embedding_array.shape[0] != normalized.size:
    raise ValueError("embedding and weight counts must match")
  return np.sum(embedding_array * normalized[:, np.newaxis], axis=0)


def prompt_definitions_to_json(definitions) -> list[dict]:
  """Return normalized client metadata for a prompt bank."""
  normalized = normalize_prompt_weights(
      [definition.weight for definition in definitions]
  )
  return [
      {
          "id": definition.prompt_id,
          "text": definition.text,
          "weight": definition.weight,
          "normalized_weight": float(normalized[index]),
      }
      for index, definition in enumerate(definitions)
  ]


def _validate_weight(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a number")
  weight = float(value)
  if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
    raise ValueError(f"{name} must be within [0, 1]")
  return weight


def _validate_midi_pitch(value: object) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError("MIDI pitch must be an integer")
  if not 0 <= value < MIDI_PITCHES:
    raise ValueError(f"MIDI pitch must be within [0, {MIDI_PITCHES - 1}]")
  return value


def _validate_unmask_width(value: object) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError("MIDI unmask_width must be an integer")
  if not 0 <= value < MIDI_PITCHES:
    raise ValueError(
        f"MIDI unmask_width must be within [0, {MIDI_PITCHES - 1}]"
    )
  return value


def _validate_embeddings(definitions, embeddings) -> list[np.ndarray]:
  if len(definitions) != len(embeddings):
    raise ValueError("prompt and embedding counts must match")
  arrays = [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]
  if not arrays or any(array.ndim != 1 for array in arrays):
    raise ValueError("each style embedding must be one-dimensional")
  if len({array.shape for array in arrays}) != 1:
    raise ValueError("all style embeddings must have the same shape")
  return arrays


def _describe_mix(definitions) -> str:
  normalized = normalize_prompt_weights(
      [definition.weight for definition in definitions]
  )
  return ", ".join(
      f"{definition.text}:{normalized[index]:.3f}"
      for index, definition in enumerate(definitions)
  )


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
