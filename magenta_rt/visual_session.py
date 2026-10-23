"""Session-scoped latest-frame worker, independent of audio/control tasks."""

import asyncio
import contextlib
import time

from magenta_rt.vlm import DEFAULT_MODEL, MODELS, MAX_IMAGE_BYTES
from magenta_rt.vlm import VlmBackend, parse_music_tags, prepare_image
from magenta_rt.visual_conditioning import normalize_ratio


class VisualSession:
  def __init__(self, conditioning, encoder, send_json, *, host, inference_lock,
               backend_factory=VlmBackend, timeout=60):
    self.conditioning = conditioning
    self.encoder = encoder
    self.send_json = send_json
    self.host = host
    self.inference_lock = inference_lock
    self.backend_factory = backend_factory
    self.timeout = timeout
    self.enabled = False
    self.mode = "still"
    self.model = DEFAULT_MODEL
    self._ratio = 2
    self.generation = -1
    self.sequence = 0
    self.tags = ()
    self.tag_text = ""
    self.allow_extra_tags = False
    self.solo = False
    self.style_version = 0
    self.closed = False
    self.pending = None
    self.backends = {}
    self.event = asyncio.Event()
    self.task = asyncio.create_task(self._run(), name="mrt-visual-worker")

  def state(self):
    visual_share = 1 / (self.ratio + 1) if self.tags and self.enabled else 0
    return {
        "enabled": self.enabled, "mode": self.mode, "model": self.model,
        "generation": self.generation, "ratio": self.ratio,
        "tags": list(self.tags),
        "allow_extra_tags": self.allow_extra_tags,
        "effective_vlm_share": visual_share,
        "effective_normal_share": 1 - visual_share if self.ratio > 0 else 0,
    }

  @property
  def ratio(self):
    return self._ratio

  async def configure(self, message):
    if not isinstance(message, dict):
      raise ValueError("visual settings must be an object")
    enabled = message.get("enabled", False)
    mode = message.get("mode", "still")
    model = message.get("model", DEFAULT_MODEL)
    generation = message.get("generation")
    ratio = normalize_ratio(message.get("ratio", 2))
    allow_extra_tags = message.get("allow_extra_tags", self.allow_extra_tags)
    if not isinstance(allow_extra_tags, bool):
      raise ValueError("allow_extra_tags must be a boolean")
    if not isinstance(enabled, bool) or mode not in ("still", "screen", "camera"):
      raise ValueError("invalid visual mode")
    if not isinstance(model, str) or model not in MODELS:
      raise ValueError("unknown VLM model")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
      raise ValueError("invalid visual generation")
    if generation < self.generation:
      return
    changed = (enabled, mode, model) != (self.enabled, self.mode, self.model)
    if changed and generation == self.generation:
      raise ValueError("mode/model changes require a new visual generation")
    self.conditioning.set_ratio(ratio)
    self._ratio = ratio
    tags_cleared = False
    if generation != self.generation:
      self.pending = None
      tags_cleared = bool(self.tags)
      self.tags = ()
      self.tag_text = ""
      self.sequence = 0
      self.conditioning.set_visual(None)
    elif not allow_extra_tags and self.tags:
      try:
        parse_music_tags(self.tag_text)
      except ValueError:
        # Changing policy also excludes a currently applied oversized result.
        tags_cleared = True
        self.tags = ()
        self.tag_text = ""
        self.conditioning.set_visual(None)
    self.enabled, self.mode, self.model = enabled, mode, model
    self.allow_extra_tags = allow_extra_tags
    self.generation = generation
    await self.send_json({"type": "visual_state", **self.state(),
                          "tags_cleared": tags_cleared})

  def submit(self, data, generation, sequence):
    if self.closed or not self.enabled or generation != self.generation:
      raise ValueError("visual input is disabled or belongs to an old session")
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
      raise ValueError("image must be between 1 byte and 8 MiB")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= self.sequence:
      raise ValueError("image sequence must increase")
    if self.mode == "still" and self.sequence:
      raise ValueError("a still image is inferred once; select a new image to run again")
    self.sequence = sequence
    self.pending = ("frame", generation, sequence, data, self.model)
    self.event.set()

  def restore(self, tags):
    """Reuse browser-held static tags on audio restart without another VLM call."""
    if not self.enabled or self.mode != "still" or not tags:
      return
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
      raise ValueError("invalid restored tags")
    self.sequence = 1
    # Validate in the worker so a rejected cached result cannot stop audio.
    self.pending = ("restore", self.generation, 0, tuple(tags), self.model)
    self.event.set()

  def set_performance(self, solo):
    if self.solo != solo:
      self.solo = solo
      self.style_version += 1
      if self.tags and self.pending is None:
        self.pending = ("refresh", self.generation, self.sequence, None, self.model)
        self.event.set()

  async def _apply(self, text, generation, sequence, started):
    while not self.closed and generation == self.generation and self.enabled:
      tags = parse_music_tags(text, allow_extra_tags=self.allow_extra_tags)
      version = self.style_version
      texts = [("SOLO " if self.solo else "") + t for t in tags]
      embeddings = await self.encoder.encode(texts)
      if version != self.style_version:
        continue
      if self.closed or generation != self.generation or not self.enabled:
        return
      # A policy change may have arrived while MusicCoCa was encoding.
      parse_music_tags(text, allow_extra_tags=self.allow_extra_tags)
      self.conditioning.set_visual(embeddings)
      self.tags = tags
      self.tag_text = text
      await self.send_json({
          "type": "visual_applied", **self.state(), "sequence": sequence,
          "elapsed_ms": (time.perf_counter() - started) * 1000,
          "prompt_revision": self.conditioning.get().revision,
      })
      return

  async def _run(self):
    while True:
      await self.event.wait()
      self.event.clear()
      job, self.pending = self.pending, None
      if job is None:
        continue
      kind, generation, sequence, data, model = job
      if generation != self.generation or not self.enabled:
        continue
      started = time.perf_counter()
      try:
        if kind == "frame":
          await self.send_json({"type": "visual_processing", "generation": generation,
                                "sequence": sequence})
          jpeg = await asyncio.to_thread(prepare_image, data)
          async with self.inference_lock:
            if generation != self.generation or self.closed:
              continue
            if model not in self.backends:
              self.backends[model] = self.backend_factory(
                  model, host=self.host, timeout=self.timeout
              )
            text = await asyncio.wait_for(self.backends[model].infer(jpeg), self.timeout)
        else:
          text = self.tag_text if kind == "refresh" else ",".join(data)
        if text:
          await self._apply(text, generation, sequence, started)
      except Exception as exc:
        if not self.closed and generation == self.generation:
          await self.send_json({"type": "visual_error", **self.state(),
                                "sequence": sequence, "message": str(exc)})

  async def close(self):
    self.closed = True
    self.pending = None
    self.task.cancel()
    await asyncio.gather(self.task, return_exceptions=True)
    for backend in self.backends.values():
      with contextlib.suppress(Exception):
        await backend.close()
