"""Small shared helpers; safe in either the MRT2 or training environment."""

import json
from pathlib import Path

import numpy as np

from scripts.musiccoca_audio_baseline import (
    norm, save_array, sha256, token_difference, unit,
    write_csv, write_json,
)

SCHEMA_VERSION = 1


def fingerprint(value):
  import hashlib
  return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False).encode()).hexdigest()


def new_directory(path):
  path = Path(path).expanduser().resolve()
  path.mkdir(parents=True, exist_ok=False)
  return path


def read_json(path):
  with Path(path).open(encoding="utf-8") as file:
    return json.load(file)


def local_array(root, relative, expected_shape=None):
  root = Path(root).resolve()
  path = (root / relative).resolve()
  if root not in path.parents:
    raise ValueError("Cache array must be inside its cache directory")
  array = np.load(path, allow_pickle=False)
  if (array.dtype != np.float32 or not np.all(np.isfinite(array)) or
      (expected_shape is not None and array.shape != tuple(expected_shape))):
    raise ValueError(f"Invalid float32 cache array: {path}")
  return array


def checked_device(device):
  if device != "cpu" and not (device == "cuda" or device.startswith("cuda:")):
    raise ValueError("device must be cpu, cuda or cuda:N")
  if device.startswith("cuda"):
    import os
    import subprocess
    try:
      query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=10)
      active = [int(row.strip()) for row in query.stdout.splitlines() if row.strip()]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
      raise RuntimeError("Cannot confirm idle GPU; CUDA operation not started") from exc
    if any(pid != os.getpid() for pid in active):
      raise RuntimeError("Other GPU processes are active; CUDA operation not started")
  return device


def unit_rows(array):
  return np.stack([unit(row) for row in array]).astype(np.float32)
