"""Visual conditioning and asynchronous work, without a GPU or external API."""

import asyncio
from io import BytesIO
import threading
import unittest
from unittest.mock import patch

import numpy as np

from magenta_rt.realtime_server import (
    PromptDefinition, normalize_prompt_weights, prompt_definitions_to_json,
    validate_prompt_definitions, validate_weight_update,
)
from magenta_rt.visual_conditioning import VisualConditioning
from magenta_rt.visual_session import VisualSession
from magenta_rt.style_encoding import StyleEncoder
from magenta_rt.prompt_updates import PromptUpdates
from magenta_rt.vlm import DEFAULT_MODEL, MAX_TAGS, INSTRUCTION, parse_music_tags, prepare_image


def conditioning(count=2):
  definitions = tuple(PromptDefinition(i, f"text {i}", 0.75 if i == 0 else 0.25)
                      for i in range(count))
  return VisualConditioning(definitions, [np.array([1, 0], dtype=np.float32)
                                        for _ in definitions])


class MixingTest(unittest.TestCase):
  def test_all_zero_normal_weights_are_allowed_only_for_vlm_only(self):
    raw = [{"id": 0, "text": "a", "weight": 0}, {"id": 1, "text": "b", "weight": 0}]
    with self.assertRaises(ValueError):
      validate_prompt_definitions(raw)
    definitions = validate_prompt_definitions(raw, allow_zero=True)
    mixer = VisualConditioning(definitions, [np.array([1., 0.]), np.array([0., 1.])],
                               ratio="0:1")
    self.assertIsNone(mixer.get().style)
    self.assertEqual([d.weight for d in mixer.definitions()], [0, 0])
    metadata = prompt_definitions_to_json(mixer.definitions(), allow_zero=True)
    self.assertEqual([p["normalized_weight"] for p in metadata], [0, 0])
    np.testing.assert_array_equal(normalize_prompt_weights([0, 0], allow_zero=True), [0, 0])
    for invalid in ([0, -1], [0, float("nan")], []):
      with self.subTest(invalid=invalid), self.assertRaises(ValueError):
        normalize_prompt_weights(invalid, allow_zero=True)
    mixer.set_visual([[.25, .75]])
    weights = [{"id": 0, "weight": 0}, {"id": 1, "weight": 0}]
    with self.assertRaises(ValueError):
      validate_weight_update(weights, definitions)
    mixer.update_weights(weights)
    mixer.configure(definitions, [np.array([9., 9.]), np.array([8., 8.])])
    np.testing.assert_array_equal(mixer.get().style, [.25, .75])
    with self.assertRaises(ValueError):
      mixer.set_ratio("2:1")
    self.assertTrue(mixer.allow_zero_weights)
    np.testing.assert_array_equal(mixer.get().style, [.25, .75])
    mixer.update_weights([{"id": 0, "weight": 1}, {"id": 1, "weight": 0}])
    mixer.set_ratio("2:1")
    with self.assertRaises(ValueError):
      mixer.update_weights(weights)

  def test_hierarchical_weights_and_32_normal_prompts(self):
    mixer = VisualConditioning(
        [PromptDefinition(0, "a", .75), PromptDefinition(1, "b", .25)],
        [np.array([1., 0.]), np.array([0., 1.])],
    )
    np.testing.assert_array_equal(mixer.get().style, [.75, .25])
    mixer.set_visual([[1., 0.], [0., 1.]])
    np.testing.assert_allclose(mixer.get().style, [2/3, 1/3])
    mixer.set_ratio(3)
    np.testing.assert_allclose(mixer.get().style, [.6875, .3125])
    mixer.set_visual(None)
    np.testing.assert_array_equal(mixer.get().style, [.75, .25])
    full = conditioning(32)
    full.set_visual([[0., 1.]] * 5)
    self.assertEqual(len(full.definitions()), 32)
    full.update_weights([{"id": i, "weight": 1} for i in range(32)])
    np.testing.assert_allclose(full.get().style, [2/3, 1/3])

  def test_tags_and_default(self):
    self.assertEqual(DEFAULT_MODEL, "gemma")
    self.assertEqual(parse_music_tags(" Calm , soft piano,calm, "), ("calm", "soft piano"))
    self.assertEqual(MAX_TAGS, 5)
    self.assertIn("never more than 5", INSTRUCTION)
    self.assertEqual(len(parse_music_tags("a,b,c,d,e")), 5)
    for bad in ("", "explanation\ncalm, piano", ", ", "a,b,c,d,e,f", "a,a,b,c,d,e"):
      with self.subTest(bad=bad), self.assertRaises(ValueError):
        parse_music_tags(bad)

  def test_ratios_on_both_sides_preserve_equal_tag_weights(self):
    mixer = conditioning()
    mixer.set_visual([[0., 1.]] * 5)
    for normal in range(1, 5):
      for visual in range(1, 5):
        with self.subTest(normal=normal, visual=visual):
          mixer.set_ratio(f"{normal}:{visual}")
          np.testing.assert_allclose(mixer.get().style,
                                     [normal/(normal+visual), visual/(normal+visual)])
    for bad in (True, -1, .1, float("nan"), float("inf"), "1:5", "0:2", "1:0", "2", "x"):
      with self.subTest(bad=bad), self.assertRaises(ValueError):
        mixer.set_ratio(bad)

  def test_vlm_only_never_falls_back_to_normal(self):
    mixer = conditioning()
    for ratio in ("0:1", 0):
      mixer.set_ratio(ratio)
      self.assertIsNone(mixer.get().style)
      mixer.set_visual([[0., 1.], [.25, .75]])
      np.testing.assert_allclose(mixer.get().style, [.125, .875])
      mixer.configure([PromptDefinition(0, "different normal", 1)], [np.array([9., 9.])])
      np.testing.assert_allclose(mixer.get().style, [.125, .875])
      mixer.set_visual(None)
      self.assertIsNone(mixer.get().style)
    mixer.set_ratio("2:1")
    np.testing.assert_allclose(mixer.get().style, [9., 9.])

  def test_extra_tags_are_opt_in_without_truncation(self):
    text = "calm,piano,ambient,slow,soft,warm,strings"
    with self.assertRaises(ValueError):
      parse_music_tags(text)
    self.assertEqual(parse_music_tags(text, allow_extra_tags=True), tuple(text.split(",")))
    self.assertEqual(len(parse_music_tags(",".join(f"tag{i}" for i in range(40)),
                                          allow_extra_tags=True)), 40)
    for invalid in ("", "a\nb", "a" * 129, "x" * 1025):
      with self.subTest(text=invalid), self.assertRaises(ValueError):
        parse_music_tags(invalid, allow_extra_tags=True)
    with self.assertRaises(ValueError):
      parse_music_tags(text, allow_extra_tags="true")

  def test_image_preprocessing(self):
    from PIL import Image
    output = BytesIO()
    Image.new("RGBA", (1800, 1200), (1, 2, 3, 4)).save(output, "PNG")
    result = prepare_image(output.getvalue())
    with Image.open(BytesIO(result)) as image:
      self.assertEqual(image.format, "JPEG")
      self.assertEqual(image.size, (1024, 683))
    with self.assertRaises(Exception):
      prepare_image(b"not an image")


class AsyncVisualTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.messages = []
    self.calls = []
    self.started = asyncio.Event()
    self.release = asyncio.Event()
    self.mixer = conditioning()
    test = self

    class Encoder:
      async def encode(self, texts):
        test.calls.append(tuple(texts))
        return [np.array([0., 1.]) for _ in texts]

    class Backend:
      def __init__(self, model, **kwargs):
        self.model = model

      async def infer(self, jpeg):
        test.started.set()
        if jpeg in (b"first", b"many-blocked"):
          await test.release.wait()
        if jpeg == b"bad":
          raise RuntimeError("backend unavailable")
        if jpeg.startswith(b"many"):
          return "calm,piano,ambient,slow,soft,warm"
        return "calm, soft piano"

      async def close(self):
        pass

    self.encoder = Encoder()
    async def send(message):
      self.messages.append(message)
    self.session = VisualSession(self.mixer, self.encoder, send, host="unused",
                                 inference_lock=asyncio.Lock(), backend_factory=Backend)
    self.image_patch = patch("magenta_rt.visual_session.prepare_image", side_effect=lambda b: b)
    self.image_patch.start()
    await self.config(1, mode="camera")

  async def asyncTearDown(self):
    self.release.set()
    await self.session.close()
    self.image_patch.stop()

  async def config(self, generation, **values):
    await self.session.configure({"enabled": True, "mode": "camera", "model": "gemma",
                                  "ratio": 2, "generation": generation, **values})

  async def until(self, predicate):
    async with asyncio.timeout(2):
      while not predicate():
        await asyncio.sleep(.001)

  async def test_latest_pending_frame_and_error_retains_style(self):
    self.session.submit(b"first", 1, 1)
    await self.started.wait()
    self.session.submit(b"discarded", 1, 2)
    self.session.submit(b"latest", 1, 3)
    self.assertEqual(self.session.pending[2], 3)
    self.release.set()
    await self.until(lambda: len([m for m in self.messages if m["type"] == "visual_applied"]) == 2)
    self.assertEqual([m["sequence"] for m in self.messages if m["type"] == "visual_applied"], [1, 3])
    revision = self.mixer.get().revision
    self.session.submit(b"bad", 1, 4)
    await self.until(lambda: any(m["type"] == "visual_error" for m in self.messages))
    self.assertEqual(self.mixer.get().revision, revision)

  async def test_old_generation_result_never_applies(self):
    self.session.submit(b"first", 1, 1)
    await self.started.wait()
    await self.config(2, model="qwen", mode="screen")
    self.release.set()
    self.session.submit(b"second", 2, 1)
    await self.until(lambda: any(m["type"] == "visual_applied" for m in self.messages))
    self.assertEqual([m["generation"] for m in self.messages if m["type"] == "visual_applied"], [2])

  async def test_still_is_once_ratio_and_solo_do_not_reinfer(self):
    await self.config(2, mode="still")
    self.session.submit(b"image", 2, 1)
    await self.until(lambda: bool(self.session.tags))
    with self.assertRaises(ValueError):
      self.session.submit(b"image", 2, 2)
    calls = len(self.calls)
    await self.config(2, mode="still", ratio=3)
    self.assertEqual(len(self.calls), calls)
    self.session.set_performance(True)
    await self.until(lambda: ("SOLO calm", "SOLO soft piano") in self.calls)

  async def test_vlm_heavy_ratio_is_reported_and_applied(self):
    await self.config(2, ratio="1:4")
    self.session.submit(b"image", 2, 1)
    await self.until(lambda: bool(self.session.tags))
    np.testing.assert_allclose(self.mixer.get().style, [.2, .8])
    self.assertEqual(self.session.state()["effective_vlm_share"], .8)

  async def test_extra_tags_setting_and_vlm_only(self):
    await self.config(2, ratio="0:1")
    self.assertIsNone(self.mixer.get().style)
    self.assertFalse(self.session.state()["allow_extra_tags"])
    self.assertEqual(self.session.state()["effective_normal_share"], 0)
    self.session.submit(b"many", 2, 1)
    await self.until(lambda: any(m["type"] == "visual_error" for m in self.messages))
    self.assertFalse(self.calls)
    await self.config(2, ratio="0:1", allow_extra_tags=True)
    self.session.submit(b"many", 2, 2)
    await self.until(lambda: len(self.session.tags) == 6)
    self.assertEqual(len(self.calls[-1]), 6)
    np.testing.assert_array_equal(self.mixer.get().style, [0., 1.])
    self.assertEqual(self.session.state()["effective_vlm_share"], 1)
    await self.config(2, ratio="0:1", allow_extra_tags=False)
    self.assertEqual(self.session.tags, ())
    self.assertTrue(self.messages[-1]["tags_cleared"])
    self.assertIsNone(self.mixer.get().style)

  async def test_extra_tag_policy_changes_during_inference(self):
    await self.config(2, allow_extra_tags=True)
    self.session.submit(b"many-blocked", 2, 1)
    await self.started.wait()
    await self.config(2, allow_extra_tags=False)
    self.release.set()
    await self.until(lambda: any(m["type"] == "visual_error" for m in self.messages))
    self.assertFalse(self.session.tags)
    self.assertFalse(self.calls)

  async def test_extra_tag_policy_changes_during_encoding(self):
    entered = asyncio.Event()
    release = asyncio.Event()
    class Encoder:
      async def encode(self, texts):
        entered.set()
        await release.wait()
        return [np.array([0., 1.]) for _ in texts]
    self.session.encoder = Encoder()
    await self.config(2, ratio="0:1", allow_extra_tags=True)
    self.session.submit(b"many", 2, 1)
    await asyncio.wait_for(entered.wait(), 2)
    await self.config(2, ratio="0:1", allow_extra_tags=False)
    release.set()
    await self.until(lambda: any(m["type"] == "visual_error" for m in self.messages))
    self.assertIsNone(self.mixer.get().style)

  async def test_extra_static_tags_restore_and_reject_without_reinference(self):
    await self.config(2, mode="still", ratio="0:1", allow_extra_tags=True)
    tags = ["calm", "piano", "ambient", "slow", "soft", "warm"]
    self.session.restore(tags)
    await self.until(lambda: len(self.session.tags) == 6)
    self.assertFalse(self.started.is_set())
    await self.config(3, mode="still", ratio="0:1", allow_extra_tags=False)
    self.session.restore(tags)
    await self.until(lambda: any(m["type"] == "visual_error" for m in self.messages))
    self.assertIsNone(self.mixer.get().style)
    self.assertFalse(self.started.is_set())

  async def test_extra_tag_setting_requires_boolean(self):
    for value in ("false", "true", 1, None):
      with self.subTest(value=value), self.assertRaises(ValueError):
        await self.config(2, allow_extra_tags=value)
    self.assertFalse(self.session.allow_extra_tags)

  async def test_restore_does_not_call_backend(self):
    await self.config(2, mode="still")
    self.session.restore(["ambient", "piano"])
    await self.until(lambda: bool(self.session.tags))
    self.assertFalse(self.started.is_set())

  async def test_disable_during_inference(self):
    self.session.submit(b"first", 1, 1)
    await self.started.wait()
    await self.config(2, enabled=False)
    self.release.set()
    await asyncio.sleep(.03)
    self.assertFalse(any(m["type"] == "visual_applied" for m in self.messages))
    np.testing.assert_array_equal(self.mixer.get().style, [1, 0])


class StyleWorkerTest(unittest.IsolatedAsyncioTestCase):
  async def test_newer_normal_edit_wins_and_live_weights_are_preserved(self):
    mixer = conditioning()
    gate = asyncio.Event()
    messages = []
    class Encoder:
      async def encode(self, texts):
        await gate.wait()
        return [np.eye(2, dtype=np.float32)[i] for i, _ in enumerate(texts)]
    async def send(message):
      messages.append(message)
    updates = PromptUpdates(mixer, Encoder(), send)
    try:
      updates.submit([PromptDefinition(0, "obsolete", 1)], 1)
      await asyncio.sleep(0)
      updates.submit([PromptDefinition(0, "latest a", .75),
                      PromptDefinition(1, "latest b", .25)], 2)
      mixer.update_weights([{"id": 0, "weight": .2}, {"id": 1, "weight": .8}])
      updates.weight_version += 1
      gate.set()
      await updates.task
      np.testing.assert_allclose(mixer.get().style, [.2, .8])
      self.assertEqual(mixer.definitions()[0].text, "latest a")
      self.assertEqual([m["request_id"] for m in messages if m["type"] == "prompt_applied"], [2])
    finally:
      await updates.close()

  async def test_cancelled_encoding_stays_serial_and_cache_is_bounded(self):
    active = 0
    peak = 0
    gate = threading.Event()
    entered = threading.Event()
    class Mrt:
      def embed_style(self, text, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        if text == "blocked":
          entered.set()
          gate.wait(2)
        active -= 1
        return np.array([1., 0.])
    encoder = StyleEncoder(Mrt(), capacity=2)
    try:
      first = asyncio.create_task(encoder.encode(["blocked"]))
      await asyncio.to_thread(entered.wait, 1)
      first.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await first
      second = asyncio.create_task(encoder.encode(["a", "b", "c"]))
      gate.set()
      await second
      self.assertEqual(peak, 1)
      self.assertEqual(len(encoder.cache), 2)
    finally:
      gate.set()
      encoder.close()

  async def test_normal_update_does_not_block_controls_or_replace_visual(self):
    mixer = conditioning()
    mixer.set_visual([[0., 1.]])
    gate = asyncio.Event()
    messages = []
    class Encoder:
      async def encode(self, texts):
        await gate.wait()
        return [np.array([1., 0.]) for _ in texts]
    async def send(message):
      messages.append(message)
    updates = PromptUpdates(mixer, Encoder(), send)
    try:
      updates.submit([PromptDefinition(0, "new", 1)], 5)
      await asyncio.sleep(0)
      self.assertEqual(messages[0]["type"], "prompt_encoding")
      self.assertEqual(mixer.definitions()[0].text, "text 0")
      gate.set()
      await updates.task
      np.testing.assert_allclose(mixer.get().style, [2/3, 1/3])
    finally:
      await updates.close()


class BackendTest(unittest.IsolatedAsyncioTestCase):
  async def test_ollama_request_matches_benchmark_without_ollama_sdk(self):
    import httpx
    import json
    from magenta_rt.vlm import VlmBackend
    for selection, model in (("gemma", "gemma4:e4b"), ("qwen", "qwen3.5:4b")):
      received = []
      async def respond(request):
        self.assertEqual(request.url.path, "/api/chat")
        received.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "calm, piano"},
                                         "done_reason": "stop"})
      backend = VlmBackend(selection)
      await backend.client.aclose()
      backend.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond))
      try:
        self.assertEqual(await backend.infer(b"jpeg"), "calm, piano")
        payload = received[0]
        self.assertEqual(payload["model"], model)
        self.assertIs(payload["think"], False)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["messages"][0]["images"], ["anBlZw=="])
        self.assertEqual(payload["options"]["num_predict"], 256)
      finally:
        await backend.close()

  async def test_gemini_async_request(self):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from google import genai
    from magenta_rt.vlm import VlmBackend
    response = SimpleNamespace(text="calm, piano", candidates=[])
    generate = AsyncMock(return_value=response)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate),
                             aclose=AsyncMock())
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-not-a-key"}):
      with patch.object(genai, "Client", return_value=SimpleNamespace(aio=client)):
        backend = VlmBackend("gemini")
        self.assertEqual(await backend.infer(b"jpeg"), "calm, piano")
        self.assertEqual(generate.call_args.kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(generate.call_args.kwargs["config"].temperature, 0)
        await backend.close()
        client.aclose.assert_awaited_once()
