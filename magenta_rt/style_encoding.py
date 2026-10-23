"""Serialize shared LiteRT text/mapper calls, including across reconnects."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class StyleEncoder:
  def __init__(self, mrt, capacity=256):
    self.mrt = mrt
    self.capacity = capacity
    self.cache = OrderedDict()
    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mrt-style")

  def _encode(self, texts):
    results = []
    for text in texts:
      if text not in self.cache:
        embedding = np.asarray(
            self.mrt.embed_style(text, use_mapper=True, seed=0), dtype=np.float32
        ).copy()
        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
          raise ValueError("invalid MusicCoCa embedding")
        self.cache[text] = embedding
      self.cache.move_to_end(text)
      results.append(self.cache[text])
      while len(self.cache) > self.capacity:
        self.cache.popitem(last=False)
    return results

  async def encode(self, texts):
    return await asyncio.get_running_loop().run_in_executor(
        self.executor, self._encode, tuple(texts)
    )

  def close(self):
    self.executor.shutdown(wait=False, cancel_futures=True)
