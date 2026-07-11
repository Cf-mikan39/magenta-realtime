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

"""Small real-time primitives shared by Linux/JAX streaming applications."""

from __future__ import annotations

import threading
import time

import numpy as np


class BufferClosedError(RuntimeError):
  """Raised when writing to a closed ring buffer."""


class StereoRingBuffer:
  """Bounded, thread-safe SPSC ring buffer for float32 stereo samples.

  The intended topology is one JAX inference producer and one audio consumer.
  Writes may block while the buffer is full. Reads are non-blocking by default
  so an audio callback can return on time and zero-fill an underrun.
  """

  def __init__(self, capacity_samples: int):
    if capacity_samples <= 0:
      raise ValueError("capacity_samples must be greater than zero")
    self._data = np.zeros((capacity_samples, 2), dtype=np.float32)
    self._capacity = capacity_samples
    self._read_index = 0
    self._write_index = 0
    self._size = 0
    self._closed = False
    self._condition = threading.Condition()

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def available(self) -> int:
    with self._condition:
      return self._size

  @property
  def free_space(self) -> int:
    with self._condition:
      return self._capacity - self._size

  @property
  def closed(self) -> bool:
    with self._condition:
      return self._closed

  def close(self) -> None:
    """Wake blocked readers/writers and reject subsequent writes."""
    with self._condition:
      self._closed = True
      self._condition.notify_all()

  def wait_for_available(
      self, count: int, timeout: float | None = None
  ) -> bool:
    """Wait until at least ``count`` samples are available.

    Returns false on timeout, or when the buffer closes before reaching the
    requested level.
    """
    if not 0 <= count <= self._capacity:
      raise ValueError("count must fit within the ring buffer")
    deadline = None if timeout is None else time.monotonic() + timeout
    with self._condition:
      while self._size < count and not self._closed:
        remaining = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        if remaining == 0.0:
          return False
        self._condition.wait(remaining)
      return self._size >= count

  def write(self, samples: np.ndarray, timeout: float | None = None) -> bool:
    """Write an ``[N, 2]`` float array, waiting for enough free space.

    Returns false when the timeout expires. A closed buffer raises
    ``BufferClosedError``.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 2:
      raise ValueError(f"expected stereo samples shaped [N, 2], got {samples.shape}")
    count = samples.shape[0]
    if count > self._capacity:
      raise ValueError("a single write cannot exceed ring-buffer capacity")
    if count == 0:
      return True

    deadline = None if timeout is None else time.monotonic() + timeout
    with self._condition:
      while self._capacity - self._size < count and not self._closed:
        remaining = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        if remaining == 0.0:
          return False
        self._condition.wait(remaining)
      if self._closed:
        raise BufferClosedError("cannot write to a closed ring buffer")

      first = min(count, self._capacity - self._write_index)
      self._data[self._write_index : self._write_index + first] = samples[:first]
      second = count - first
      if second:
        self._data[:second] = samples[first:]
      self._write_index = (self._write_index + count) % self._capacity
      self._size += count
      self._condition.notify_all()
      return True

  def read(
      self,
      count: int,
      *,
      zero_pad: bool = True,
      timeout: float = 0.0,
  ) -> tuple[np.ndarray, int]:
    """Read up to ``count`` samples and return ``(audio, missing_count)``.

    When ``zero_pad`` is true, the returned array always contains ``count``
    samples. ``timeout=0`` is non-blocking and is suitable for audio callbacks.
    A positive timeout can be useful for offline or network consumers.
    """
    if count < 0:
      raise ValueError("count must be non-negative")
    if count == 0:
      return np.zeros((0, 2), dtype=np.float32), 0

    deadline = time.monotonic() + timeout
    with self._condition:
      while self._size < count and not self._closed and timeout > 0:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining == 0.0:
          break
        self._condition.wait(remaining)

      actual = min(count, self._size)
      output_count = count if zero_pad else actual
      output = np.zeros((output_count, 2), dtype=np.float32)

      first = min(actual, self._capacity - self._read_index)
      output[:first] = self._data[self._read_index : self._read_index + first]
      second = actual - first
      if second:
        output[first : first + second] = self._data[:second]
      self._read_index = (self._read_index + actual) % self._capacity
      self._size -= actual
      self._condition.notify_all()

    return output, count - actual
