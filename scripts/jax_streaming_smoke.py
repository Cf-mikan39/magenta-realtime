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

"""Stateful, frame-by-frame JAX streaming smoke test for Magenta RealTime 2.

This script is the first Linux/NVIDIA real-time milestone.  It intentionally
does not play audio yet: it verifies that the JAX model can generate one 40 ms
frame at a time while carrying model state across every call.  Prompt changes
are applied without resetting that state.

Example:

    python scripts/jax_streaming_smoke.py \
      --model mrt2_small \
      --duration 60 \
      --prompt "disco funk" \
      --prompt-change "20:ambient synth pads" \
      --prompt-change "40:fast drum and bass"

The script writes a gapless 48 kHz stereo WAV and a per-frame timing CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import statistics
import time

FRAME_RATE = 25
FRAME_DURATION_MS = 1000.0 / FRAME_RATE


@dataclass(frozen=True)
class PromptEvent:
  """A prompt that becomes active at ``start_seconds``."""

  start_seconds: float
  text: str

  @property
  def start_frame(self) -> int:
    # Apply the event on the first frame whose timestamp is not earlier than
    # the requested time.
    return math.ceil(self.start_seconds * FRAME_RATE)


def parse_prompt_change(value: str) -> PromptEvent:
  """Parse a ``SECONDS:PROMPT`` command-line value."""
  seconds_text, separator, prompt = value.partition(":")
  if not separator:
    raise argparse.ArgumentTypeError(
        "prompt changes must use the form SECONDS:PROMPT"
    )
  try:
    seconds = float(seconds_text)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"invalid prompt-change time: {seconds_text!r}"
    ) from exc
  prompt = prompt.strip()
  if seconds < 0:
    raise argparse.ArgumentTypeError("prompt-change time must be non-negative")
  if not prompt:
    raise argparse.ArgumentTypeError("prompt text must not be empty")
  return PromptEvent(seconds, prompt)


def build_schedule(
    initial_prompt: str, changes: list[PromptEvent], duration: float
) -> list[PromptEvent]:
  """Validate and normalize prompt events for a finite generation run."""
  initial_prompt = initial_prompt.strip()
  if not initial_prompt:
    raise ValueError("the initial prompt must not be empty")
  if duration <= 0:
    raise ValueError("duration must be greater than zero")

  schedule = [PromptEvent(0.0, initial_prompt), *changes]
  schedule.sort(key=lambda event: event.start_seconds)

  if any(event.start_seconds >= duration for event in schedule[1:]):
    raise ValueError("every prompt change must occur before --duration")
  for previous, current in zip(schedule, schedule[1:]):
    if previous.start_frame == current.start_frame:
      raise ValueError(
          "only one prompt event may be scheduled for each 40 ms frame"
      )
  return schedule


def percentile(values: list[float], q: float) -> float:
  """Return a percentile as a Python float."""
  if not values:
    raise ValueError("cannot calculate a percentile of an empty sequence")
  if not 0 <= q <= 100:
    raise ValueError("q must be in the range [0, 100]")
  ordered = sorted(values)
  position = (len(ordered) - 1) * q / 100.0
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  fraction = position - lower
  return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Generate MRT2 audio one 40 ms frame at a time while preserving "
          "JAX streaming state."
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
      help="change prompt without resetting state; may be repeated",
  )
  parser.add_argument("--temperature", type=float, default=1.1)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--cfg-musiccoca", type=float, default=1.6)
  parser.add_argument("--cfg-notes", type=float, default=2.4)
  parser.add_argument("--cfg-drums", type=float, default=4.0)
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("outputs/jax_streaming_smoke.wav"),
  )
  parser.add_argument(
      "--metrics",
      type=Path,
      default=Path("outputs/jax_streaming_smoke_frames.csv"),
  )
  parser.add_argument(
      "--log-every",
      type=int,
      default=25,
      help="print progress every N frames; 0 disables periodic logs",
  )
  return parser


def main() -> None:
  args = make_parser().parse_args()
  try:
    schedule = build_schedule(args.prompt, args.prompt_change, args.duration)
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from exc

  # Importing JAX initializes the CUDA runtime, so keep it after argument
  # validation.  Keep the audio writer here for the same reason.  This makes
  # `--help` fast and usable outside the project's fully installed environment.
  import soundfile as sf  # pylint: disable=import-outside-toplevel
  from magenta_rt import MagentaRT2Jax  # pylint: disable=import-outside-toplevel

  logging.basicConfig(level=logging.INFO, force=True)
  logger = logging.getLogger("jax_streaming_smoke")

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

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.metrics.parent.mkdir(parents=True, exist_ok=True)

  total_frames = round(args.duration * FRAME_RATE)
  state = None
  event_index = 0
  frame_times_ms: list[float] = []
  deadline_misses = 0
  sample_rate: int | None = None
  expected_samples_per_frame: int | None = None

  logger.info(
      "Generating %d frames (%.2f s); the same state will be carried across "
      "all frames",
      total_frames,
      total_frames / FRAME_RATE,
  )

  run_start = time.perf_counter()
  with sf.SoundFile(
      args.output,
      mode="w",
      samplerate=48_000,
      channels=2,
      subtype="PCM_16",
  ) as wav_file, args.metrics.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "frame_index",
            "audio_time_seconds",
            "prompt_index",
            "prompt",
            "generation_ms",
            "deadline_miss",
            "samples",
        ]
    )

    for frame_index in range(total_frames):
      if (
          event_index + 1 < len(schedule)
          and frame_index >= schedule[event_index + 1].start_frame
      ):
        event_index += 1
        logger.info(
            "Frame %d (%.2f s): prompt -> %r; preserving streaming state",
            frame_index,
            frame_index / FRAME_RATE,
            schedule[event_index].text,
        )

      step_start = time.perf_counter()
      waveform, state = mrt.generate(
          style=embeddings[event_index],
          frames=1,
          state=state,
      )
      generation_ms = (time.perf_counter() - step_start) * 1000.0
      frame_times_ms.append(generation_ms)

      if sample_rate is None:
        sample_rate = waveform.sample_rate
        expected_samples_per_frame = sample_rate // FRAME_RATE
        if sample_rate != 48_000:
          raise RuntimeError(f"expected 48 kHz output, got {sample_rate} Hz")
      if waveform.num_channels != 2:
        raise RuntimeError(
            f"expected stereo output, got {waveform.num_channels} channels"
        )
      if waveform.num_samples != expected_samples_per_frame:
        raise RuntimeError(
            "unexpected frame length: "
            f"expected {expected_samples_per_frame}, got {waveform.num_samples}"
        )

      missed = generation_ms > FRAME_DURATION_MS
      deadline_misses += int(missed)
      wav_file.write(waveform.samples)
      writer.writerow(
          [
              frame_index,
              f"{frame_index / FRAME_RATE:.6f}",
              event_index,
              schedule[event_index].text,
              f"{generation_ms:.6f}",
              int(missed),
              waveform.num_samples,
          ]
      )

      if args.log_every > 0 and (
          (frame_index + 1) % args.log_every == 0
          or frame_index + 1 == total_frames
      ):
        elapsed = time.perf_counter() - run_start
        logger.info(
            "%d/%d frames | latest %.1f ms | average %.1f ms | %.1f fps",
            frame_index + 1,
            total_frames,
            generation_ms,
            1000.0 * elapsed / (frame_index + 1),
            (frame_index + 1) / elapsed,
        )

  elapsed = time.perf_counter() - run_start
  audio_seconds = total_frames / FRAME_RATE
  logger.info("Finished without resetting streaming state")
  logger.info("Audio: %s", args.output)
  logger.info("Metrics: %s", args.metrics)
  logger.info(
      "Runtime %.2f s for %.2f s audio | RTF %.3f | %.1f frames/s",
      elapsed,
      audio_seconds,
      elapsed / audio_seconds,
      total_frames / elapsed,
  )
  logger.info(
      "Frame time ms: mean %.2f | p50 %.2f | p95 %.2f | p99 %.2f | max %.2f",
      statistics.fmean(frame_times_ms),
      percentile(frame_times_ms, 50),
      percentile(frame_times_ms, 95),
      percentile(frame_times_ms, 99),
      max(frame_times_ms),
  )
  logger.info(
      "40 ms deadline misses: %d/%d (%.2f%%)",
      deadline_misses,
      total_frames,
      100.0 * deadline_misses / total_frames,
  )


if __name__ == "__main__":
  main()
