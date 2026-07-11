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

"""Tests for the dependency-light Linux/JAX WebSocket core."""

import time
import unittest

import numpy as np

from magenta_rt.realtime import StereoRingBuffer
from magenta_rt.realtime_server import blend_style_embeddings
from magenta_rt.realtime_server import encode_pcm_f32le
from magenta_rt.realtime_server import FRAME_SAMPLES
from magenta_rt.realtime_server import JaxRealtimeProducer
from magenta_rt.realtime_server import MidiConditioning
from magenta_rt.realtime_server import PromptConditioning
from magenta_rt.realtime_server import PromptDefinition
from magenta_rt.realtime_server import PromptMixer
from magenta_rt.realtime_server import SAMPLE_RATE
from magenta_rt.realtime_server import normalize_prompt_weights
from magenta_rt.realtime_server import validate_prompt_definitions
from magenta_rt.realtime_server import validate_prompt


class _FakeWaveform:

  def __init__(self, value: float):
    self.sample_rate = SAMPLE_RATE
    self.num_channels = 2
    self.num_samples = FRAME_SAMPLES
    self.samples = np.full((FRAME_SAMPLES, 2), value, dtype=np.float32)


class _FakeMrt:

  def __init__(self):
    self.states = []
    self.notes = []

  def generate(self, *, style, frames, state, notes=None):
    self.states.append(state)
    self.notes.append(notes)
    next_state = 1 if state is None else state + 1
    return _FakeWaveform(float(style)), next_state


class PromptConditioningTest(unittest.TestCase):

  def test_update_is_revisioned(self):
    prompt = PromptConditioning("disco", 1)
    self.assertEqual(prompt.get().revision, 0)
    updated = prompt.update("ambient", 2)
    self.assertEqual(updated.text, "ambient")
    self.assertEqual(updated.style, 2)
    self.assertEqual(updated.revision, 1)

  def test_validate_prompt(self):
    self.assertEqual(validate_prompt("  disco funk  "), "disco funk")
    with self.assertRaises(ValueError):
      validate_prompt("  ")
    with self.assertRaises(ValueError):
      validate_prompt(123)


class PromptMixerTest(unittest.TestCase):

  def test_normalizes_and_blends_embeddings(self):
    normalized = normalize_prompt_weights([1.0, 3.0])
    np.testing.assert_allclose(normalized, [0.25, 0.75])
    blended = blend_style_embeddings(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
        [1.0, 3.0],
    )
    np.testing.assert_allclose(blended, [0.25, 0.75])

  def test_validates_prompt_bank(self):
    definitions = validate_prompt_definitions(
        [
            {"id": 10, "text": "disco", "weight": 1.0},
            {"id": 20, "text": "ambient", "weight": 0.5},
        ]
    )
    self.assertEqual(
        [definition.prompt_id for definition in definitions], [10, 20]
    )
    with self.assertRaises(ValueError):
      validate_prompt_definitions(
          [
              {"id": 1, "text": "one", "weight": 0.0},
              {"id": 1, "text": "two", "weight": 0.0},
          ]
      )

  def test_weight_updates_do_not_replace_cached_embeddings(self):
    definitions = (
        PromptDefinition(0, "left", 1.0),
        PromptDefinition(1, "right", 0.0),
    )
    mixer = PromptMixer(
        definitions,
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
    )
    updated = mixer.update_weights(
        [{"id": 0, "weight": 0.25}, {"id": 1, "weight": 0.75}]
    )
    self.assertEqual(updated.revision, 1)
    np.testing.assert_allclose(updated.style, [0.25, 0.75])

  def test_configure_replaces_bank_atomically(self):
    mixer = PromptMixer(
        (PromptDefinition(0, "old", 1.0),),
        [np.array([1.0, 0.0], dtype=np.float32)],
    )
    updated = mixer.configure(
        (PromptDefinition(2, "new", 1.0),),
        [np.array([0.0, 1.0], dtype=np.float32)],
    )
    self.assertEqual(updated.revision, 1)
    self.assertEqual(mixer.definitions()[0].text, "new")
    np.testing.assert_array_equal(updated.style, [0.0, 1.0])


class MidiConditioningTest(unittest.TestCase):

  def test_disabled_midi_is_unconditioned(self):
    midi = MidiConditioning()
    self.assertIsNone(midi.frame_tokens())

  def test_onset_then_sustain_in_explicit_onset_mode(self):
    midi = MidiConditioning(enabled=True, auto_strum=False, unmask_width=4)
    midi.note_on(60)
    onset = midi.frame_tokens()
    sustain = midi.frame_tokens()
    self.assertEqual(onset[60], 2)
    self.assertEqual(sustain[60], 1)
    self.assertTrue(
        all(
            onset[pitch] == 0
            for pitch in range(56, 65)
            if pitch != 60
        )
    )
    self.assertEqual(onset[55], -1)
    self.assertEqual(onset[65], -1)

  def test_short_note_is_latched_for_one_frame(self):
    midi = MidiConditioning(enabled=True, auto_strum=False)
    midi.note_on(64)
    midi.note_off(64)
    self.assertEqual(midi.frame_tokens()[64], 2)
    self.assertEqual(midi.frame_tokens()[64], -1)

  def test_auto_strum_and_solo_modes(self):
    midi = MidiConditioning(
        enabled=True, auto_strum=True, unmask_width=127
    )
    midi.note_on(67)
    tokens = midi.frame_tokens()
    self.assertEqual(tokens[67], 3)
    self.assertEqual(tokens.count(0), 127)

  def test_producer_passes_midi_tokens_to_model(self):
    mrt = _FakeMrt()
    midi = MidiConditioning(enabled=True, auto_strum=False)
    midi.note_on(72)
    ring = StereoRingBuffer(FRAME_SAMPLES)
    producer = JaxRealtimeProducer(
        mrt=mrt,
        prompt=PromptConditioning("one", 0.5),
        ring_buffer=ring,
        midi=midi,
    )
    producer.start()
    self.assertTrue(ring.wait_for_available(FRAME_SAMPLES, timeout=1.0))
    producer.stop()
    producer.join(timeout=1.0)
    self.assertEqual(mrt.notes[0][72], 2)


class PcmEncodingTest(unittest.TestCase):

  def test_encode_pcm_f32le_is_interleaved(self):
    samples = np.zeros((FRAME_SAMPLES, 2), dtype=np.float32)
    samples[0] = [0.25, -0.5]
    encoded = encode_pcm_f32le(samples)
    decoded = np.frombuffer(encoded, dtype="<f4")
    self.assertEqual(len(encoded), FRAME_SAMPLES * 2 * 4)
    np.testing.assert_array_equal(decoded[:4], [0.25, -0.5, 0.0, 0.0])

  def test_encode_rejects_wrong_shape(self):
    with self.assertRaises(ValueError):
      encode_pcm_f32le(np.zeros((100, 2), dtype=np.float32))


class JaxRealtimeProducerTest(unittest.TestCase):

  def test_producer_carries_state_between_frames(self):
    mrt = _FakeMrt()
    prompt = PromptConditioning("one", 0.25)
    ring = StereoRingBuffer(2 * FRAME_SAMPLES)
    producer = JaxRealtimeProducer(
        mrt=mrt,
        prompt=prompt,
        ring_buffer=ring,
    )
    producer.start()
    self.assertTrue(ring.wait_for_available(2 * FRAME_SAMPLES, timeout=1.0))
    producer.stop()
    producer.join(timeout=1.0)

    self.assertFalse(producer.is_alive())
    self.assertIsNone(producer.error)
    self.assertEqual(mrt.states[:2], [None, 1])
    self.assertEqual(producer.stats().frames_generated, 2)
    audio, missing = ring.read(2 * FRAME_SAMPLES, zero_pad=False)
    self.assertEqual(missing, 0)
    np.testing.assert_array_equal(audio, 0.25)

  def test_stop_wakes_full_buffer_writer(self):
    mrt = _FakeMrt()
    ring = StereoRingBuffer(FRAME_SAMPLES)
    producer = JaxRealtimeProducer(
        mrt=mrt,
        prompt=PromptConditioning("one", 0.5),
        ring_buffer=ring,
    )
    producer.start()
    self.assertTrue(ring.wait_for_available(FRAME_SAMPLES, timeout=1.0))
    time.sleep(0.01)
    producer.stop()
    producer.join(timeout=1.0)
    self.assertFalse(producer.is_alive())


if __name__ == "__main__":
  unittest.main()
