# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light protocol tests for the JAX WebSocket server."""

import unittest

from magenta_rt.realtime_server import DrumConditioning
from magenta_rt.realtime_server import MidiConditioning
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
    )
    self.assertTrue(drums.snapshot().no_drums)
    self.assertEqual(sent[0]["type"], "drum_config_applied")
    self.assertTrue(sent[0]["no_drums"])


if __name__ == "__main__":
  unittest.main()
