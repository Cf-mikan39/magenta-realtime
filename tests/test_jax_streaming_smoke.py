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

"""Tests for the GPU-independent parts of jax_streaming_smoke.py."""

import argparse
import unittest

from scripts.jax_streaming_smoke import build_schedule
from scripts.jax_streaming_smoke import parse_prompt_change


class JaxStreamingSmokeTest(unittest.TestCase):

  def test_parse_prompt_change_preserves_colons_in_prompt(self):
    event = parse_prompt_change("12.5:ambient: slow and spacious")

    self.assertEqual(event.start_seconds, 12.5)
    self.assertEqual(event.start_frame, 313)
    self.assertEqual(event.text, "ambient: slow and spacious")

  def test_parse_prompt_change_rejects_invalid_values(self):
    for value in ["missing separator", "bad:prompt", "-1:prompt", "1:   "]:
      with self.subTest(value=value):
        with self.assertRaises(argparse.ArgumentTypeError):
          parse_prompt_change(value)

  def test_build_schedule_sorts_changes_and_starts_at_zero(self):
    schedule = build_schedule(
        "disco funk",
        [
            parse_prompt_change("40:drum and bass"),
            parse_prompt_change("20:ambient"),
        ],
        duration=60,
    )

    self.assertEqual(
        [event.start_seconds for event in schedule], [0, 20, 40]
    )
    self.assertEqual(
        [event.text for event in schedule],
        ["disco funk", "ambient", "drum and bass"],
    )

  def test_build_schedule_rejects_changes_outside_duration(self):
    with self.assertRaisesRegex(ValueError, "before --duration"):
      build_schedule(
          "disco funk", [parse_prompt_change("60:ambient")], duration=60
      )

  def test_build_schedule_rejects_two_events_in_same_frame(self):
    with self.assertRaisesRegex(ValueError, "one prompt event"):
      build_schedule(
          "disco funk",
          [
              parse_prompt_change("1.001:ambient"),
              parse_prompt_change("1.010:drum and bass"),
          ],
          duration=2,
      )


if __name__ == "__main__":
  unittest.main()
