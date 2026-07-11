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

"""Tests for Linux/JAX real-time primitives."""

import threading
import time
import unittest

import numpy as np

from magenta_rt.realtime import BufferClosedError
from magenta_rt.realtime import StereoRingBuffer


def stereo(values) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  return np.column_stack([values, -values])


class StereoRingBufferTest(unittest.TestCase):

  def test_write_read_and_wraparound(self):
    buffer = StereoRingBuffer(6)
    self.assertTrue(buffer.write(stereo([1, 2, 3, 4])))
    first, missing = buffer.read(3)

    np.testing.assert_array_equal(first, stereo([1, 2, 3]))
    self.assertEqual(missing, 0)

    self.assertTrue(buffer.write(stereo([5, 6, 7, 8])))
    second, missing = buffer.read(5)

    np.testing.assert_array_equal(second, stereo([4, 5, 6, 7, 8]))
    self.assertEqual(missing, 0)
    self.assertEqual(buffer.available, 0)

  def test_underrun_is_zero_padded(self):
    buffer = StereoRingBuffer(8)
    buffer.write(stereo([1, 2]))

    output, missing = buffer.read(5)

    np.testing.assert_array_equal(output[:2], stereo([1, 2]))
    np.testing.assert_array_equal(output[2:], np.zeros((3, 2)))
    self.assertEqual(missing, 3)

  def test_writer_waits_for_consumer(self):
    buffer = StereoRingBuffer(4)
    buffer.write(stereo([1, 2, 3, 4]))
    completed = threading.Event()

    def writer():
      buffer.write(stereo([5, 6]), timeout=1.0)
      completed.set()

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.01)
    self.assertFalse(completed.is_set())

    buffer.read(2)
    thread.join(timeout=1.0)

    self.assertTrue(completed.is_set())
    output, missing = buffer.read(4)
    np.testing.assert_array_equal(output, stereo([3, 4, 5, 6]))
    self.assertEqual(missing, 0)

  def test_close_wakes_and_rejects_writer(self):
    buffer = StereoRingBuffer(2)
    buffer.close()

    with self.assertRaises(BufferClosedError):
      buffer.write(stereo([1]))

  def test_wait_for_available(self):
    buffer = StereoRingBuffer(4)
    buffer.write(stereo([1, 2]))

    self.assertTrue(buffer.wait_for_available(2, timeout=0.01))
    self.assertFalse(buffer.wait_for_available(3, timeout=0.01))


if __name__ == "__main__":
  unittest.main()
