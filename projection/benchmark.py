"""Explicit real-image encoder probe; no model or dataset downloads."""

from pathlib import Path
import time

import numpy as np

from .common import checked_device, new_directory, save_array, sha256, write_csv, write_json
from .encoders import create_encoder


def benchmark_encoder(name, weights, files, output_dir, device="cpu", repeats=3, max_images=5):
  if repeats < 2 or max_images < 1:
    raise ValueError("repeats must be >=2 and max_images must be positive")
  files = list(dict.fromkeys(Path(p).resolve() for p in files))[:max_images]
  if not files or any(not p.is_file() for p in files):
    raise ValueError("Provide existing images for the real-model probe")
  checked_device(device)
  import torch
  torch.set_num_threads(4)
  cuda = device.startswith("cuda")
  if cuda:
    # PyTorch 2.1 memory-stat APIs require an initialized CUDA context.
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
  started = time.perf_counter()
  encoder = create_encoder(name, weights, device)
  if cuda:
    torch.cuda.synchronize(device)
  load_seconds = time.perf_counter() - started
  rows, vectors = [], []
  for path in files:
    outputs, durations = [], []
    for repeat in range(repeats):
      started = time.perf_counter()
      outputs.append(encoder.encode([str(path)])[0])
      if cuda:
        torch.cuda.synchronize(device)
      durations.append(time.perf_counter() - started)
    vector = outputs[0]
    vectors.append(vector)
    rows.append({"image": str(path), "image_sha256": sha256(path), "shape": list(vector.shape),
                 "dtype": str(vector.dtype), "norm": float(np.linalg.norm(vector)),
                 "bitwise_repeatable": all(np.array_equal(vector, other) for other in outputs[1:]),
                 "max_repeat_absolute_error": max(float(np.max(np.abs(vector - v))) for v in outputs[1:]),
                 "inference_seconds": durations, "warm_mean_seconds": float(np.mean(durations[1:]))})
  values = np.stack(vectors).astype(np.float32)
  cosine = values @ values.T
  gpu = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
         "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
         "allocated_after_bytes": int(torch.cuda.memory_allocated(device))} if cuda else None
  root = new_directory(output_dir)
  save_array(root / "features.npy", values)
  save_array(root / "cosine.npy", cosine)
  write_csv(root / "images.csv", rows)
  write_json(root / "benchmark.json", {"encoder": encoder.metadata, "device": device,
      "model_load_seconds": load_seconds, "repeats": repeats, "images": rows, "gpu": gpu,
      "gpu_memory_note": "PyTorch allocator statistics; excludes CUDA driver/context overhead",
      "cosine_matrix": cosine.tolist(), "quality_evaluation": False})
  return root
