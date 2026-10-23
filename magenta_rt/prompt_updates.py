"""Prepare normal prompt changes without stalling MIDI and stop messages."""

import asyncio
import time

from magenta_rt.realtime_server import prompt_definitions_to_json


class PromptUpdates:
  def __init__(self, prompt, encoder, send_json):
    self.prompt = prompt
    self.encoder = encoder
    self.send_json = send_json
    self.task = None
    self.generation = 0
    self.closed = False
    self.weight_version = 0

  def submit(self, definitions, request_id):
    self.generation += 1
    generation = self.generation
    if self.task:
      self.task.cancel()
    self.task = asyncio.create_task(
        self._apply(definitions, request_id, generation, self.weight_version),
        name="mrt-prompt-update"
    )

  async def _apply(self, definitions, request_id, generation, weight_version):
    started = time.perf_counter()
    try:
      await self.send_json({"type": "prompt_encoding", "request_id": request_id,
                            "uncached_count": len(definitions)})
      embeddings = await self.encoder.encode([d.text for d in definitions])
      if self.closed or generation != self.generation:
        return
      # A weight update may have arrived during encoding. Keep its latest values
      # when the old and new banks have the same IDs.
      current = {d.prompt_id: d.weight for d in self.prompt.definitions()}
      if (weight_version != self.weight_version
          and set(current) == {d.prompt_id for d in definitions}):
        from dataclasses import replace
        definitions = tuple(replace(d, weight=current[d.prompt_id]) for d in definitions)
      snapshot = self.prompt.configure(definitions, embeddings)
      await self.send_json({
          "type": "prompt_applied", "request_id": request_id,
          "prompts": prompt_definitions_to_json(
              definitions, allow_zero=getattr(self.prompt, "allow_zero_weights", False)
          ),
          "prompt_revision": snapshot.revision,
          "encoding_ms": (time.perf_counter() - started) * 1000,
      })
    except Exception as exc:
      if not self.closed:
        await self.send_json({"type": "control_error", "request_id": request_id,
                              "message": str(exc)})

  async def close(self):
    self.closed = True
    if self.task:
      self.task.cancel()
      await asyncio.gather(self.task, return_exceptions=True)
