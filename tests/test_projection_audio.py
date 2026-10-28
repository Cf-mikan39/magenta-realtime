"""Teacher cache tests in MRT2 environment, without JAX or learned weights."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import soundfile as sf

from projection.cache import Cache, cache_audio


class AudioCacheTest(unittest.TestCase):
  def test_shared_audio_is_encoded_once_raw_windows_and_mean_preserved(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      for i in range(3):
        (root / f"image{i}").write_bytes(str(i).encode())
        sf.write(root / f"audio{i}.wav", np.full(16000 * 12, .01 * i, dtype=np.float32), 16000)
      csv_path = root / "pairs.csv"
      csv_path.write_text("image_id,image_path,audio_id,audio_path,split,pooling,window_index\n"
                          "i0,image0,a0,audio0.wav,train,mean,\n"
                          "i0,image0,a0,audio0.wav,train,window,1\n"
                          "i1,image1,a1,audio1.wav,val,mean,\n"
                          "i2,image2,a2,audio2.wav,test,mean,\n")

      class Teacher:
        config = SimpleNamespace(sample_rate=16000, clip_length=10., embedding_dim=768)
        def __init__(self):
          self.calls = []
        def embed(self, waveform, **kwargs):
          self.calls.append(kwargs)
          return np.stack([np.full(768, 2., dtype=np.float32), np.full(768, 4., dtype=np.float32)])

      teacher = Teacher()
      cache_audio(csv_path, root / "cache", model=teacher)
      cache = Cache(root / "cache", "audio")
      self.assertEqual(len(teacher.calls), 3)
      self.assertTrue(all(c["pool_across_time"] is False for c in teacher.calls))
      np.testing.assert_array_equal(cache.audio(cache.pairs[0].pair_id), np.full(768, 3.))
      np.testing.assert_array_equal(cache.audio(cache.pairs[1].pair_id), np.full(768, 4.))
      self.assertEqual(cache.manifest["contract"]["normalization"], "raw")


if __name__ == "__main__":
  unittest.main()
