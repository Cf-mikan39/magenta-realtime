"""Two independent prompt groups; only their final embedding reaches MRT2."""

import math
import threading

import numpy as np

from magenta_rt.realtime_server import PromptMixer, PromptSnapshot


def normalize_ratio(value):
  """Accept a:b (each side 1–4), 0:1, and legacy numeric ratios."""
  if isinstance(value, str):
    parts = value.split(":")
    if value == "0:1":
      return 0.0
    if len(parts) != 2 or any(p not in ("1", "2", "3", "4") for p in parts):
      raise ValueError("normal:VLM ratio must be 0:1 or use integers from 1 to 4")
    value = int(parts[0]) / int(parts[1])
  if (isinstance(value, bool) or not isinstance(value, (int, float))
      or not math.isfinite(value)
      or (value != 0 and not any(math.isclose(value, a / b, rel_tol=1e-9, abs_tol=0)
                                for a in range(1, 5) for b in range(1, 5)))):
    raise ValueError("normal:VLM ratio must be 0:1 or use integers from 1 to 4")
  return float(value)


class VisualConditioning:
  """Keep the existing normal-prompt interface and add a read-only VLM group."""

  def __init__(self, definitions, embeddings, *, ratio=2):
    self._ratio = normalize_ratio(ratio)
    self.normal = PromptMixer(definitions, embeddings, allow_zero=self.allow_zero_weights)
    self._lock = threading.RLock()
    self._visual = None
    self._snapshot = self.normal.get()
    if self.allow_zero_weights:
      self._snapshot = PromptSnapshot(text=self._snapshot.text, style=None, revision=0)

  @property
  def allow_zero_weights(self):
    return self._ratio == 0

  def get(self):
    with self._lock:
      return self._snapshot

  def definitions(self):
    return self.normal.definitions()

  def _publish(self):
    normal = self.normal.get()
    style = normal.style
    if self._visual is not None:
      style = (self._ratio * style + self._visual) / (self._ratio + 1)
    elif self._ratio == 0:
      # generate(style=None) uses masked style tokens; never leak normal text
      # into the VLM-only condition while waiting for an accepted image result.
      style = None
    self._snapshot = PromptSnapshot(
        text=normal.text, style=style, revision=self._snapshot.revision + 1
    )
    return self._snapshot

  def configure(self, definitions, embeddings):
    with self._lock:
      self.normal.configure(definitions, embeddings)
      return self._publish()

  def update_weights(self, values):
    with self._lock:
      self.normal.update_weights(values)
      return self._publish()

  def set_ratio(self, ratio):
    ratio = normalize_ratio(ratio)
    with self._lock:
      self.normal.set_allow_zero(ratio == 0)
      self._ratio = ratio
      return self._publish()

  def set_visual(self, embeddings):
    with self._lock:
      if embeddings is None:
        self._visual = None
      else:
        values = np.asarray(embeddings, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1:] != self.normal.get().style.shape
            or not len(values) or not np.all(np.isfinite(values))):
          raise ValueError("invalid visual style embeddings")
        self._visual = np.mean(values, axis=0)
      return self._publish()
