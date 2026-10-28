"""Explicitly artificial caches for CPU wiring tests, not CLIP/ImageBind data."""

import numpy as np

from .cache import finish_cache
from .common import fingerprint, new_directory, save_array, sha256, write_json
from .dataset import Pair, assign_splits


def synthetic_cache(output_dir, samples=30, input_dim=8, seed=42):
  if samples < 6 or input_dim < 1:
    raise ValueError("Synthetic test needs samples>=6, input_dim>=1")
  root = new_directory(output_dir)
  image_root, audio_root = root / "images", root / "audio"
  for directory in (image_root, audio_root):
    (directory / "features").mkdir(parents=True)
  rng = np.random.RandomState(seed)
  x = rng.normal(size=(samples, input_dim)).astype(np.float32)
  matrix = (rng.normal(size=(input_dim, 768)) / np.sqrt(input_dim)).astype(np.float32)
  y = (x @ matrix + .2).astype(np.float32)
  pairs = [Pair(f"p{i:04d}", f"img{i:04d}", str(root / f"not_a_real_image_{i}"),
                f"audio{i:04d}", str(root / f"not_real_audio_{i}")) for i in range(samples)]
  pairs = assign_splits(pairs, seed, ratios=(.6, .2, .2))
  image_contract = {"name": "synthetic", "model": "artificial_linear_features_v1",
                    "dimension": input_dim, "preprocessing": "none; artificial data",
                    "feature_normalization": "none", "weights": []}
  audio_contract = {"model": "synthetic_linear_targets_not_musiccoca", "assets": [],
                    "normalization": "raw", "output_dimension": 768, "seed": seed}
  images, audios, image_refs, audio_refs = {}, {}, {}, {}
  for i, pair in enumerate(pairs):
    key = fingerprint(pair.pair_id)
    file = f"features/{key}.npy"
    save_array(image_root / file, x[i])
    images[key] = {"file": file, "shape": [input_dim], "dtype": "float32",
                   "file_sha256": sha256(image_root / file), "source_sha256": key}
    image_refs[pair.pair_id] = {"key": key}
    mean, windows = f"features/{key}.mean.npy", f"features/{key}.windows.npy"
    save_array(audio_root / mean, y[i])
    save_array(audio_root / windows, np.stack([y[i], y[i]]))
    audios[key] = {"mean": mean, "windows": windows, "windows_shape": [2, 768],
                   "dtype": "float32", "sha256": {"mean": sha256(audio_root / mean),
                   "windows": sha256(audio_root / windows)}, "source_sha256": key}
    audio_refs[pair.pair_id] = {"key": key, "pooling": "mean", "window_index": None}
  finish_cache(image_root, "image", pairs, image_contract, images, image_refs, synthetic=True)
  finish_cache(audio_root, "audio", pairs, audio_contract, audios, audio_refs, synthetic=True)
  test = next(i for i, pair in enumerate(pairs) if pair.split == "test")
  save_array(root / "example_feature.npy", x[test])
  write_json(root / "example_encoder.json", image_contract)
  write_json(root / "SYNTHETIC_ONLY.json", {"warning": "Artificial features/targets; no image model, music or dataset training",
                                          "samples": samples, "seed": seed})
  return root
