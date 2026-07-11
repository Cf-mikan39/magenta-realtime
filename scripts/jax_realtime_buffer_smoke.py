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

"""Test a JAX inference thread against a real-time audio consumer clock.

Unlike ``jax_streaming_smoke.py``, this program decouples generation from
playback with a bounded stereo ring buffer. The producer generates 40 ms MRT2
frames as buffer space becomes available. The consumer removes smaller blocks
at wall-clock deadlines, zero-padding and recording any underrun.

No physical audio device is required. The consumed stream is written to a WAV,
which makes this suitable for remote SSH GPU servers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time

from magenta_rt.realtime import BufferClosedError
from magenta_rt.realtime import StereoRingBuffer
from scripts.jax_streaming_smoke import build_schedule
from scripts.jax_streaming_smoke import FRAME_DURATION_MS
from scripts.jax_streaming_smoke import FRAME_RATE
from scripts.jax_streaming_smoke import parse_prompt_change
from scripts.jax_streaming_smoke import percentile


SAMPLE_RATE = 48_000
FRAME_SAMPLES = SAMPLE_RATE // FRAME_RATE


@dataclass(frozen=True)
class ProducerMetric:
  frame_index: int
  audio_time_seconds: float
  prompt_index: int
  prompt: str
  generation_ms: float
  deadline_miss: bool
  buffer_samples_after_write: int


@dataclass(frozen=True)
class ConsumerMetric:
  block_index: int
  audio_time_seconds: float
  wake_lateness_ms: float
  buffer_samples_before_read: int
  buffer_samples_after_read: int
  missing_samples: int


class JaxFrameProducer(threading.Thread):
  """Generate stateful MRT2 frames and write them to a ring buffer."""

  def __init__(
      self,
      *,
      mrt,
      embeddings,
      schedule,
      total_frames: int,
      ring_buffer: StereoRingBuffer,
      logger: logging.Logger,
  ):
    super().__init__(name="mrt2-jax-producer", daemon=True)
    self._mrt = mrt
    self._embeddings = embeddings
    self._schedule = schedule
    self._total_frames = total_frames
    self._ring_buffer = ring_buffer
    self._logger = logger
    self.metrics: list[ProducerMetric] = []
    self.error: BaseException | None = None

  def run(self) -> None:
    state = None
    event_index = 0
    try:
      for frame_index in range(self._total_frames):
        if (
            event_index + 1 < len(self._schedule)
            and frame_index >= self._schedule[event_index + 1].start_frame
        ):
          event_index += 1
          self._logger.info(
              "Generated frame %d (%.2f s): prompt -> %r; preserving state",
              frame_index,
              frame_index / FRAME_RATE,
              self._schedule[event_index].text,
          )

        step_start = time.perf_counter()
        waveform, state = self._mrt.generate(
            style=self._embeddings[event_index],
            frames=1,
            state=state,
        )
        generation_ms = (time.perf_counter() - step_start) * 1000.0

        if waveform.sample_rate != SAMPLE_RATE:
          raise RuntimeError(
              f"expected {SAMPLE_RATE} Hz, got {waveform.sample_rate} Hz"
          )
        if waveform.num_channels != 2 or waveform.num_samples != FRAME_SAMPLES:
          raise RuntimeError(
              "unexpected MRT2 frame shape: "
              f"{waveform.samples.shape}; expected ({FRAME_SAMPLES}, 2)"
          )

        self._ring_buffer.write(waveform.samples)
        self.metrics.append(
            ProducerMetric(
                frame_index=frame_index,
                audio_time_seconds=frame_index / FRAME_RATE,
                prompt_index=event_index,
                prompt=self._schedule[event_index].text,
                generation_ms=generation_ms,
                deadline_miss=generation_ms > FRAME_DURATION_MS,
                buffer_samples_after_write=self._ring_buffer.available,
            )
        )
    except BufferClosedError:
      # Normal only when the controller stops early after another error.
      pass
    except BaseException as exc:  # Propagate worker errors to the main thread.
      self.error = exc
      self._logger.exception("JAX producer failed")
    finally:
      self._ring_buffer.close()


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Run MRT2 generation and a wall-clock audio consumer through a "
          "bounded stereo ring buffer."
      )
  )
  parser.add_argument("--model", default="mrt2_small")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--prompt", default="disco funk")
  parser.add_argument(
      "--prompt-change",
      action="append",
      type=parse_prompt_change,
      default=[],
      metavar="SECONDS:PROMPT",
  )
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--cfg-musiccoca", type=float, default=1.6)
  parser.add_argument("--cfg-notes", type=float, default=2.4)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument(
      "--buffer-frames",
      type=int,
      default=6,
      help="ring-buffer capacity in 40 ms MRT2 frames",
  )
  parser.add_argument(
      "--prime-frames",
      type=int,
      default=3,
      help="frames to generate before the consumer clock starts",
  )
  parser.add_argument(
      "--block-samples",
      type=int,
      default=480,
      help="consumer block size; 480 samples = 10 ms at 48 kHz",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("outputs/jax_realtime_buffer_smoke.wav"),
  )
  parser.add_argument(
      "--producer-metrics",
      type=Path,
      default=Path("outputs/jax_realtime_buffer_producer.csv"),
  )
  parser.add_argument(
      "--consumer-metrics",
      type=Path,
      default=Path("outputs/jax_realtime_buffer_consumer.csv"),
  )
  return parser


def validate_args(args) -> None:
  if args.duration <= 0:
    raise ValueError("--duration must be greater than zero")
  if args.buffer_frames < 1:
    raise ValueError("--buffer-frames must be at least 1")
  if not 1 <= args.prime_frames <= args.buffer_frames:
    raise ValueError("--prime-frames must be within [1, --buffer-frames]")
  if args.block_samples <= 0:
    raise ValueError("--block-samples must be greater than zero")
  total_samples = round(args.duration * FRAME_RATE) * FRAME_SAMPLES
  if total_samples % args.block_samples:
    raise ValueError(
        "the generated sample count must be divisible by --block-samples"
    )


def write_producer_metrics(path: Path, metrics: list[ProducerMetric]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "frame_index",
            "audio_time_seconds",
            "prompt_index",
            "prompt",
            "generation_ms",
            "deadline_miss",
            "buffer_samples_after_write",
            "buffer_ms_after_write",
        ]
    )
    for metric in metrics:
      writer.writerow(
          [
              metric.frame_index,
              f"{metric.audio_time_seconds:.6f}",
              metric.prompt_index,
              metric.prompt,
              f"{metric.generation_ms:.6f}",
              int(metric.deadline_miss),
              metric.buffer_samples_after_write,
              f"{1000.0 * metric.buffer_samples_after_write / SAMPLE_RATE:.3f}",
          ]
      )


def write_consumer_metrics(path: Path, metrics: list[ConsumerMetric]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "block_index",
            "audio_time_seconds",
            "wake_lateness_ms",
            "buffer_samples_before_read",
            "buffer_samples_after_read",
            "buffer_ms_before_read",
            "missing_samples",
        ]
    )
    for metric in metrics:
      writer.writerow(
          [
              metric.block_index,
              f"{metric.audio_time_seconds:.6f}",
              f"{metric.wake_lateness_ms:.6f}",
              metric.buffer_samples_before_read,
              metric.buffer_samples_after_read,
              f"{1000.0 * metric.buffer_samples_before_read / SAMPLE_RATE:.3f}",
              metric.missing_samples,
          ]
      )


def main() -> None:
  args = make_parser().parse_args()
  try:
    validate_args(args)
    schedule = build_schedule(args.prompt, args.prompt_change, args.duration)
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from exc

  import numpy as np  # pylint: disable=import-outside-toplevel
  import soundfile as sf  # pylint: disable=import-outside-toplevel
  from magenta_rt import MagentaRT2Jax  # pylint: disable=import-outside-toplevel

  logging.basicConfig(level=logging.INFO, force=True)
  logger = logging.getLogger("jax_realtime_buffer_smoke")
  total_frames = round(args.duration * FRAME_RATE)
  total_samples = total_frames * FRAME_SAMPLES
  total_blocks = total_samples // args.block_samples
  capacity_samples = args.buffer_frames * FRAME_SAMPLES
  prime_samples = args.prime_frames * FRAME_SAMPLES

  logger.info("Loading %s and compiling the streaming step", args.model)
  mrt = MagentaRT2Jax(
      size=args.model,
      checkpoint=args.checkpoint,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
  )
  logger.info("Pre-encoding %d prompt(s) with MusicCoCa", len(schedule))
  embeddings = [
      mrt.embed_style(event.text, use_mapper=True, seed=0)
      for event in schedule
  ]

  ring_buffer = StereoRingBuffer(capacity_samples)
  producer = JaxFrameProducer(
      mrt=mrt,
      embeddings=embeddings,
      schedule=schedule,
      total_frames=total_frames,
      ring_buffer=ring_buffer,
      logger=logger,
  )
  producer.start()

  logger.info(
      "Priming %d frame(s) = %.0f ms; capacity %d frame(s) = %.0f ms",
      args.prime_frames,
      1000.0 * prime_samples / SAMPLE_RATE,
      args.buffer_frames,
      1000.0 * capacity_samples / SAMPLE_RATE,
  )
  if not ring_buffer.wait_for_available(prime_samples, timeout=120.0):
    ring_buffer.close()
    producer.join(timeout=5.0)
    if producer.error is not None:
      raise producer.error
    raise RuntimeError("timed out while priming the audio ring buffer")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  consumer_metrics: list[ConsumerMetric] = []
  consumed_audio = np.empty((total_samples, 2), dtype=np.float32)
  underrun_blocks = 0
  missing_samples_total = 0
  block_seconds = args.block_samples / SAMPLE_RATE

  logger.info(
      "Starting real-time consumer: %d blocks × %.1f ms = %.2f s",
      total_blocks,
      block_seconds * 1000.0,
      total_samples / SAMPLE_RATE,
  )
  playback_start = time.perf_counter()
  for block_index in range(total_blocks):
    target_time = playback_start + block_index * block_seconds
    sleep_seconds = target_time - time.perf_counter()
    if sleep_seconds > 0:
      time.sleep(sleep_seconds)
    woke_at = time.perf_counter()

    available_before = ring_buffer.available
    audio_block, missing = ring_buffer.read(
        args.block_samples,
        zero_pad=True,
        timeout=0.0,
    )
    available_after = ring_buffer.available
    start_sample = block_index * args.block_samples
    consumed_audio[start_sample : start_sample + args.block_samples] = audio_block

    underrun_blocks += int(missing > 0)
    missing_samples_total += missing
    consumer_metrics.append(
        ConsumerMetric(
            block_index=block_index,
            audio_time_seconds=block_index * block_seconds,
            wake_lateness_ms=max(0.0, (woke_at - target_time) * 1000.0),
            buffer_samples_before_read=available_before,
            buffer_samples_after_read=available_after,
            missing_samples=missing,
        )
    )

  # Account for the device playing the final block after it is submitted.
  final_deadline = playback_start + total_blocks * block_seconds
  final_sleep = final_deadline - time.perf_counter()
  if final_sleep > 0:
    time.sleep(final_sleep)

  producer.join(timeout=30.0)
  if producer.is_alive():
    ring_buffer.close()
    raise RuntimeError("producer did not stop after the consumer finished")
  if producer.error is not None:
    raise producer.error

  sf.write(args.output, consumed_audio, SAMPLE_RATE, subtype="PCM_16")
  write_producer_metrics(args.producer_metrics, producer.metrics)
  write_consumer_metrics(args.consumer_metrics, consumer_metrics)

  frame_times = [metric.generation_ms for metric in producer.metrics]
  deadline_misses = sum(metric.deadline_miss for metric in producer.metrics)
  minimum_buffer = min(
      metric.buffer_samples_before_read for metric in consumer_metrics
  )
  logger.info("Finished real-time buffer simulation")
  logger.info("Audio: %s", args.output)
  logger.info("Producer metrics: %s", args.producer_metrics)
  logger.info("Consumer metrics: %s", args.consumer_metrics)
  logger.info(
      "Frame time ms: mean %.2f | p95 %.2f | p99 %.2f | max %.2f",
      sum(frame_times) / len(frame_times),
      percentile(frame_times, 95),
      percentile(frame_times, 99),
      max(frame_times),
  )
  logger.info(
      "40 ms generation deadline misses: %d/%d (%.2f%%)",
      deadline_misses,
      len(frame_times),
      100.0 * deadline_misses / len(frame_times),
  )
  logger.info(
      "Audio underruns: %d/%d blocks | missing %d samples (%.3f ms)",
      underrun_blocks,
      total_blocks,
      missing_samples_total,
      1000.0 * missing_samples_total / SAMPLE_RATE,
  )
  logger.info(
      "Minimum buffer before read: %d samples (%.1f ms)",
      minimum_buffer,
      1000.0 * minimum_buffer / SAMPLE_RATE,
  )


if __name__ == "__main__":
  main()
