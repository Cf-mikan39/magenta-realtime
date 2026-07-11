# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Apple-live compatibility helpers in the JAX system."""

import unittest

from magenta_rt.live_compat import mask_musiccoca_tail


class MaskMusiccocaTailTest(unittest.TestCase):

  def test_masks_only_the_fine_tail(self):
    tokens = list(range(12))
    self.assertEqual(
        mask_musiccoca_tail(tokens, 6),
        [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1],
    )
    self.assertEqual(tokens, list(range(12)))

  def test_zero_preserves_all_tokens(self):
    self.assertEqual(mask_musiccoca_tail([1, 2, 3], 0), [1, 2, 3])

  def test_rejects_invalid_tail_sizes(self):
    for value in (-1, 4):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          mask_musiccoca_tail([1, 2, 3], value)
    with self.assertRaises(TypeError):
      mask_musiccoca_tail([1, 2, 3], True)


if __name__ == "__main__":
  unittest.main()
