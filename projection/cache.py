"""Immutable, self-describing image and raw Audio caches."""

from collections import OrderedDict
from pathlib import Path
import time

import numpy as np

from .common import (SCHEMA_VERSION, fingerprint, local_array, new_directory,
                     read_json, save_array, sha256, write_json)
from .dataset import (Pair, assign_splits, pairs_fingerprint, pairs_payload,
                      read_pairs, write_pairs)


def prepare_pairs(csv_path, seed=42):
  pairs = read_pairs(csv_path, require_files=True)
  # Check byte-identical renamed copies as well as IDs, paths and source IDs.
  hashes = {p: sha256(Path(p)) for p in
            dict.fromkeys(v for row in pairs for v in (row.image_path, row.audio_path))}
  return assign_splits(pairs, seed=seed, content_keys=hashes), hashes


def finish_cache(root, kind, pairs, contract, entries, refs, synthetic=False):
  manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "kind": kind,
              "synthetic": synthetic, "pairs": pairs_payload(pairs),
              "pairs_fingerprint": pairs_fingerprint(pairs), "contract": contract,
              "contract_fingerprint": fingerprint(contract), "entries": entries, "refs": refs}
  manifest["fingerprint"] = fingerprint(manifest)
  write_pairs(root / "pairs.csv", pairs)
  write_json(root / "manifest.json", manifest)
  return manifest


def cache_images(csv_path, output_dir, encoder, seed=42, batch_size=16):
  if batch_size < 1:
    raise ValueError("batch_size must be positive")
  pairs, hashes = prepare_pairs(csv_path, seed)
  root = new_directory(output_dir)
  (root / "features").mkdir()
  by_content = {}
  refs = {}
  for pair in pairs:
    key = hashes[pair.image_path]
    refs[pair.pair_id] = {"key": key}
    by_content.setdefault(key, {"path": pair.image_path, "aliases": []})["aliases"].append({
        "image_id": pair.image_id, "path": pair.image_path, "pair_id": pair.pair_id})
  entries = {}
  items = list(by_content.items())
  for offset in range(0, len(items), batch_size):
    batch = items[offset:offset + batch_size]
    start = time.perf_counter()
    values = np.asarray(encoder.encode([data["path"] for _, data in batch]))
    seconds = time.perf_counter() - start
    if (values.shape != (len(batch), encoder.metadata["dimension"]) or
        values.dtype != np.float32 or not np.all(np.isfinite(values))):
      raise ValueError("Encoder returned invalid float32 image features")
    for (key, data), vector in zip(batch, values):
      file = f"features/{key}.npy"
      save_array(root / file, vector)
      entries[key] = {"file": file, "shape": list(vector.shape), "dtype": "float32",
                      "file_sha256": sha256(root / file), "source_sha256": key,
                      "aliases": data["aliases"], "batch_seconds": seconds}
  return finish_cache(root, "image", pairs, encoder.metadata, entries, refs)


def cache_audio(csv_path, output_dir, resource_dir=None, seed=42, hop_seconds=10.,
                pad_end=True, pooling="mean", window_index=0, model=None):
  from scripts.musiccoca_audio_baseline import asset_metadata
  from magenta_rt import audio, musiccoca, paths
  import importlib.metadata
  import math

  if (not math.isfinite(hop_seconds) or hop_seconds <= 0 or
      round(hop_seconds * 16000) < 1 or pooling not in ("mean", "window") or window_index < 0):
    raise ValueError("Invalid audio hop/pooling/window index")
  pairs, hashes = prepare_pairs(csv_path, seed)
  resources = Path(resource_dir or paths.musiccoca_dir()).expanduser().resolve()
  assets = asset_metadata(resources, False) if model is None else []
  model = model if model is not None else musiccoca.MusicCoCa(resource_dir=resources)
  contract = {"model": "MusicCoCa", "assets": [{"name": Path(a["path"]).name,
                "sha256": a["sha256"]} for a in assets], "config": vars(model.config),
              "implementation_sha256": sha256(Path(musiccoca.__file__)),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("magenta-rt", "ai-edge-litert", "numpy", "resampy")},
              "normalization": "raw", "hop_seconds": hop_seconds, "pad_end": pad_end,
              "mono_strategy": "average", "default_pooling": pooling,
              "default_window_index": window_index}
  root = new_directory(output_dir)
  (root / "features").mkdir()
  entries, refs = {}, {}
  for pair in pairs:
    key = fingerprint([hashes[pair.audio_path], pair.start_seconds, pair.duration_seconds])
    if key not in entries:
      started = time.perf_counter()
      waveform = audio.Waveform.from_file(pair.audio_path)
      total_seconds = waveform.seconds
      start = round(pair.start_seconds * waveform.sample_rate)
      end = len(waveform) if pair.duration_seconds is None else min(
          len(waveform), start + round(pair.duration_seconds * waveform.sample_rate))
      if start >= end:
        raise ValueError(f"Empty audio interval: {pair.pair_id}")
      selected = waveform[start:end]
      windows = np.asarray(model.embed(selected, pool_across_time=False,
          hop_length=hop_seconds, pad_end=pad_end, mono_strategy="average"))
      if (windows.ndim != 2 or windows.shape[0] == 0 or windows.shape[1] != 768 or
          windows.dtype != np.float32 or not np.all(np.isfinite(windows))):
        raise ValueError(f"Invalid or empty raw MusicCoCa windows: {pair.pair_id}")
      mean = np.mean(windows, axis=0)
      files = {"windows": f"features/{key}.windows.npy", "mean": f"features/{key}.mean.npy"}
      for kind, value in (("windows", windows), ("mean", mean)):
        save_array(root / files[kind], value)
      entries[key] = {**files, "windows_shape": list(windows.shape), "dtype": "float32",
          "mean_shape": [768],
          "sha256": {k: sha256(root / v) for k, v in files.items()},
          "source_sha256": hashes[pair.audio_path], "aliases": [],
          "original_seconds": total_seconds, "used_start_seconds": start / waveform.sample_rate,
          "used_end_seconds": end / waveform.sample_rate,
          "window_intervals": [{
              "index": i,
              "start_seconds": start / waveform.sample_rate + i * round(hop_seconds * model.config.sample_rate) / model.config.sample_rate,
              "end_seconds": min(end / waveform.sample_rate,
                  start / waveform.sample_rate + i * round(hop_seconds * model.config.sample_rate) / model.config.sample_rate + model.config.clip_length)}
              for i in range(len(windows))],
          "embedding_seconds": time.perf_counter() - started,
          "pooling_definition": "mean of raw 10-second windows; padded tail has equal weight"}
    entry = entries[key]
    selected_pooling = pair.pooling or pooling
    selected_window = pair.window_index if pair.window_index is not None else window_index
    if selected_pooling == "window" and selected_window >= entry["windows_shape"][0]:
      raise ValueError(f"window_index outside cached windows: {pair.pair_id}")
    refs[pair.pair_id] = {"key": key, "pooling": selected_pooling,
                          "window_index": selected_window if selected_pooling == "window" else None}
    entry["aliases"].append({"audio_id": pair.audio_id, "path": pair.audio_path,
                             "pair_id": pair.pair_id})
  return finish_cache(root, "audio", pairs, contract, entries, refs)


class Cache:
  def __init__(self, directory, expected_kind=None):
    self.root = Path(directory).resolve()
    self.manifest = read_json(self.root / "manifest.json")
    m = self.manifest
    if (m.get("schema_version") != SCHEMA_VERSION or m.get("status") != "complete" or
        expected_kind and m["kind"] != expected_kind):
      raise ValueError("Incompatible or incomplete cache")
    if m["fingerprint"] != fingerprint({k: v for k, v in m.items() if k != "fingerprint"}):
      raise ValueError("Cache manifest fingerprint mismatch")
    if m["contract_fingerprint"] != fingerprint(m["contract"]):
      raise ValueError("Cache model/preprocessing mismatch")
    self.pairs = [Pair(**p) for p in m["pairs"]]
    if pairs_fingerprint(self.pairs) != m["pairs_fingerprint"]:
      raise ValueError("Cache dataset fingerprint mismatch")
    # Also validate labels independently; explicit cross-split leakage is never
    # accepted merely because a manifest was serialized with that split.
    assign_splits(self.pairs)
    if set(m["refs"]) != {p.pair_id for p in self.pairs}:
      raise ValueError("Missing cache pair references")
    self._arrays = OrderedDict()

  def array(self, relative, checksum, shape):
    if relative not in self._arrays:
      value = local_array(self.root, relative, shape)
      if sha256(self.root / relative) != checksum:
        raise ValueError("Cache array checksum mismatch")
      self._arrays[relative] = value
      while len(self._arrays) > 256:
        self._arrays.popitem(last=False)
    self._arrays.move_to_end(relative)
    return self._arrays[relative]

  def image(self, pair_id):
    entry = self.manifest["entries"][self.manifest["refs"][pair_id]["key"]]
    return self.array(entry["file"], entry["file_sha256"], entry["shape"])

  def audio(self, pair_id):
    ref = self.manifest["refs"][pair_id]
    entry = self.manifest["entries"][ref["key"]]
    if ref["pooling"] == "mean":
      return self.array(entry["mean"], entry["sha256"]["mean"], (768,))
    return self.array(entry["windows"], entry["sha256"]["windows"],
                      entry["windows_shape"])[ref["window_index"]]

  def target_key(self, pair_id):
    return fingerprint(self.manifest["refs"][pair_id])


def paired_caches(image_dir, audio_dir):
  image, audio = Cache(image_dir, "image"), Cache(audio_dir, "audio")
  if image.manifest["pairs_fingerprint"] != audio.manifest["pairs_fingerprint"]:
    raise ValueError("Image/Audio caches have different pairs or splits; do not mix them")
  # Content-key leakage check also works for caches copied to another machine.
  seen = {}
  for pair in image.pairs:
    for kind, cache in (("image", image), ("audio", audio)):
      entry = cache.manifest["entries"][cache.manifest["refs"][pair.pair_id]["key"]]
      key = kind, entry["source_sha256"]
      if key in seen and seen[key] != pair.split:
        raise ValueError("Byte-identical image/audio appears in multiple splits")
      seen[key] = pair.split
  return image, audio
