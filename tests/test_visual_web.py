"""Exercise the actual HTTP/WebSocket integration with fake generation."""

from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from magenta_rt.audio import Waveform
from scripts.jax_realtime_web import create_app


class FakeMrt:
  def __init__(self):
    self.states = []
    self.styles = []

  def embed_style(self, text, **kwargs):
    embedding = np.zeros(768, dtype=np.float32)
    embedding[0 if text == "disco" else 1] = 1
    return embedding

  def generate(self, *, style, state, **kwargs):
    self.states.append(state)
    self.styles.append(None if style is None else style.copy())
    return Waveform(np.zeros((1920, 2), dtype=np.float32), sample_rate=48000), (
        1 if state is None else state + 1
    )


def receive_type(ws, kind):
  # Audio continues between control replies; do not mistake PCM for JSON.
  import json
  for _ in range(150):
    message = ws.receive()
    if "text" in message:
      value = json.loads(message["text"])
      if value["type"] == "error" or (value["type"] == "control_error" and kind != "control_error"):
        raise AssertionError(value)
      if value["type"] == kind:
        return value
  raise AssertionError(f"no {kind} response")


class VisualWebTest(unittest.TestCase):
  def test_start_and_update_all_zero_normal_weights_in_vlm_only(self):
    mrt = FakeMrt()
    app = create_app(mrt=mrt, model_name="fake", buffer_frames=3, prime_frames=3)
    image = BytesIO()
    Image.new("RGB", (16, 16)).save(image, "JPEG")
    prompts = [{"id": 0, "text": "disco", "weight": 0},
               {"id": 1, "text": "ambient", "weight": 0}]
    visual = {"enabled": True, "mode": "still", "model": "gemma",
              "ratio": "0:1", "generation": 1}
    async def infer(backend, jpeg):
      return "calm, piano"
    with patch("magenta_rt.vlm.VlmBackend.infer", infer), TestClient(app) as client:
      with client.websocket_connect("/ws/audio") as ws:
        token = receive_type(ws, "hello")["visual_session"]
        ws.send_json({"type": "start", "prompts": prompts, "visual": visual})
        started = receive_type(ws, "stream_started")
        self.assertEqual([p["normalized_weight"] for p in started["prompts"]], [0, 0])
        self.assertTrue(all(style is None for style in mrt.styles))
        self.assertEqual(client.post(f"/api/visual/frame/{token}?generation=1&sequence=1",
                                     content=image.getvalue()).status_code, 202)
        self.assertEqual(receive_type(ws, "visual_applied")["effective_vlm_share"], 1)
        ws.send_json({"type": "set_weights", "weights": [
            {"id": 0, "weight": 0}, {"id": 1, "weight": 0}]})
        prompts[1]["text"] = "changed"
        ws.send_json({"type": "set_prompts", "prompts": prompts, "request_id": 8})
        applied = receive_type(ws, "prompt_applied")
        self.assertEqual([p["normalized_weight"] for p in applied["prompts"]], [0, 0])
        receive_type(ws, "metrics")
        self.assertEqual(mrt.styles[-1][1], 1)
        self.assertEqual(mrt.styles[-1][0], 0)
        ws.send_json({"type": "visual_config", **visual, "ratio": "2:1"})
        self.assertIn("greater than zero", receive_type(ws, "visual_error")["message"])
        ws.send_json({"type": "set_weights", "weights": [
            {"id": 0, "weight": 1}, {"id": 1, "weight": 0}]})
        ws.send_json({"type": "visual_config", **visual, "ratio": "2:1"})
        self.assertAlmostEqual(receive_type(ws, "visual_state")["effective_vlm_share"], 1/3)
        receive_type(ws, "metrics")
        self.assertAlmostEqual(float(mrt.styles[-1][0]), 2/3, places=6)
        ws.send_json({"type": "stop"})
    self.assertTrue(all(state is not None for state in mrt.states[1:]))

  def test_normal_start_still_rejects_all_zero_weights(self):
    app = create_app(mrt=FakeMrt(), model_name="fake", buffer_frames=3, prime_frames=3)
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
      receive_type(ws, "hello")
      ws.send_json({"type": "start", "prompts": [{"id": 0, "text": "disco", "weight": 0}],
                    "visual": {"ratio": "2:1"}})
      import json
      error = json.loads(ws.receive()["text"])
      self.assertEqual(error["type"], "error")
      self.assertIn("greater than zero", error["message"])

  def test_vlm_only_and_extra_tag_policy_over_websocket(self):
    mrt = FakeMrt()
    app = create_app(mrt=mrt, model_name="fake", buffer_frames=3, prime_frames=3)
    image = BytesIO()
    Image.new("RGB", (16, 16)).save(image, "JPEG")
    tags = ["calm", "piano", "ambient", "slow", "soft", "warm"]
    async def infer(backend, jpeg):
      return ",".join(tags)
    config = {"enabled": True, "mode": "camera", "model": "gemma",
              "ratio": "0:1", "generation": 1, "allow_extra_tags": False}
    with patch("magenta_rt.vlm.VlmBackend.infer", infer), TestClient(app) as client:
      with client.websocket_connect("/ws/audio") as ws:
        token = receive_type(ws, "hello")["visual_session"]
        ws.send_json({"type": "start", "prompt": "disco", "visual": config})
        receive_type(ws, "stream_started")
        self.assertTrue(mrt.styles)
        self.assertTrue(all(style is None for style in mrt.styles))
        url = f"/api/visual/frame/{token}?generation=1&sequence="
        self.assertEqual(client.post(url + "1", content=image.getvalue()).status_code, 202)
        receive_type(ws, "visual_error")
        ws.send_json({"type": "visual_config", **config, "allow_extra_tags": True})
        state = receive_type(ws, "visual_state")
        self.assertTrue(state["allow_extra_tags"])
        self.assertEqual(client.post(url + "2", content=image.getvalue()).status_code, 202)
        applied = receive_type(ws, "visual_applied")
        self.assertEqual(applied["tags"], tags)
        self.assertEqual(applied["effective_vlm_share"], 1)
        self.assertEqual(applied["effective_normal_share"], 0)
        receive_type(ws, "metrics")
        self.assertTrue(any(style is not None and style[1] == 1 for style in mrt.styles))
        ws.send_json({"type": "visual_config", **config})
        self.assertTrue(receive_type(ws, "visual_state")["tags_cleared"])
        receive_type(ws, "metrics")
        self.assertIsNone(mrt.styles[-1])
        ws.send_json({"type": "stop"})
    self.assertTrue(all(state is not None for state in mrt.states[1:]))

  def test_upload_updates_style_without_resetting_generation(self):
    mrt = FakeMrt()
    app = create_app(mrt=mrt, model_name="fake", buffer_frames=3, prime_frames=3)
    image = BytesIO()
    Image.new("RGB", (16, 16)).save(image, "JPEG")
    async def infer(backend, jpeg):
      return "calm, soft piano"
    with patch("magenta_rt.vlm.VlmBackend.infer", infer):
      with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as ws:
          hello = receive_type(ws, "hello")
          token = hello["visual_session"]
          self.assertEqual(hello["visual_default_model"], "gemma")
          ws.send_json({"type": "start", "prompt": "disco", "visual": {
              "enabled": True, "mode": "still", "model": "gemma",
              "ratio": 2, "generation": 1,
          }})
          receive_type(ws, "stream_started")
          response = client.post(f"/api/visual/frame/{token}?generation=1&sequence=1",
                                 content=image.getvalue())
          self.assertEqual(response.status_code, 202)
          applied = receive_type(ws, "visual_applied")
          self.assertEqual(applied["tags"], ["calm", "soft piano"])
          self.assertAlmostEqual(applied["effective_vlm_share"], 1/3)
          # Let the bounded audio buffer drain so the producer actually consumes
          # the newly published style before another normal-text update.
          receive_type(ws, "metrics")
          ws.send_json({"type": "sampling_config", "temperature": 1.2, "top_k": 60})
          receive_type(ws, "sampling_config_applied")
          ws.send_json({"type": "set_prompts", "prompt": "ambient", "request_id": 7})
          receive_type(ws, "prompt_applied")
          duplicate = client.post(f"/api/visual/frame/{token}?generation=1&sequence=2",
                                  content=image.getvalue())
          self.assertEqual(duplicate.status_code, 409)
          ws.send_json({"type": "stop"})
        self.assertEqual(client.post(f"/api/visual/frame/{token}?generation=1&sequence=3",
                                     content=image.getvalue()).status_code, 409)
    self.assertIsNone(mrt.states[0])
    self.assertTrue(all(s is not None for s in mrt.states[1:]))
    self.assertTrue(any(np.isclose(style[1], 1/3) for style in mrt.styles))

  def test_missing_session_and_health_remain_available(self):
    app = create_app(mrt=FakeMrt(), model_name="fake", buffer_frames=3, prime_frames=3)
    with TestClient(app) as client:
      self.assertEqual(client.get("/api/health").status_code, 200)
      self.assertEqual(client.get("/").status_code, 200)
      self.assertEqual(client.get("/static/visual-input.js").status_code, 200)
      self.assertEqual(client.post("/api/visual/frame/unknown?generation=1&sequence=1",
                                  content=b"image").status_code, 409)
