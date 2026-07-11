# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light protocol tests for the JAX WebSocket server."""

import unittest

from magenta_rt.realtime_server import DrumConditioning
from magenta_rt.realtime_server import MidiConditioning
from magenta_rt.realtime_server import SamplingConditioning
from scripts.jax_realtime_web import _receive_controls


class _FakeWebSocket:

  def __init__(self, messages):
    self._messages = iter(messages)

  async def receive_json(self):
    return next(self._messages)


class JaxRealtimeWebProtocolTest(unittest.IsolatedAsyncioTestCase):

  async def test_applies_no_drums_without_touching_prompt_state(self):
    sent = []

    async def send_json(message):
      sent.append(message)

    drums = DrumConditioning()
    await _receive_controls(
        websocket=_FakeWebSocket([
            {"type": "drum_config", "no_drums": True},
            {"type": "stop"},
        ]),
        send_json=send_json,
        mrt=None,
        prompt=None,
        embedding_cache={},
        midi=MidiConditioning(),
        drums=drums,
        sampling=SamplingConditioning(),
    )
    self.assertTrue(drums.snapshot().no_drums)
    self.assertEqual(sent[0]["type"], "drum_config_applied")
    self.assertTrue(sent[0]["no_drums"])

  async def test_applies_sampling_without_touching_prompt_state(self):
    sent = []

    async def send_json(message):
      sent.append(message)

    sampling = SamplingConditioning()
    await _receive_controls(
        websocket=_FakeWebSocket([
            {"type": "sampling_config", "temperature": 1.75, "top_k": 96},
            {"type": "stop"},
        ]),
        send_json=send_json,
        mrt=None,
        prompt=None,
        embedding_cache={},
        midi=MidiConditioning(),
        drums=DrumConditioning(),
        sampling=sampling,
    )
    self.assertEqual(sampling.snapshot().temperature, 1.75)
    self.assertEqual(sampling.snapshot().top_k, 96)
    self.assertEqual(sent[0]["type"], "sampling_config_applied")


if __name__ == "__main__":
  unittest.main()
