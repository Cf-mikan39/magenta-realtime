"""Offline Audio/Mapper/RVQ baseline; never connects to the live server.

Run from the repository with ``python -m scripts.musiccoca_audio_baseline``.
See docs/audio_embedding_baseline.md for the experiment and output schema.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np


LOG = logging.getLogger(__name__)
DEFAULT_PROMPTS = ("lofi, soft piano", "energetic rock", "calm ambient")
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}


def parse_ratio(value):
  try:
    a, b = (float(x) for x in value.split(":"))
    if not all(math.isfinite(x) and x >= 0 for x in (a, b)) or a + b <= 0:
      raise ValueError
    if not math.isfinite(a + b):
      raise ValueError
    return a, b
  except ValueError as exc:
    raise argparse.ArgumentTypeError("ratio must be finite nonnegative Text:Audio, e.g. 2:1") from exc


def make_parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("audio_paths", nargs="*", type=Path)
  p.add_argument("--audio-dir", action="append", type=Path, default=[],
                 help="Recursively find common audio extensions; repeatable.")
  p.add_argument("--file-list", action="append", type=Path, default=[],
                 help="UTF-8 paths, one per line; relative to the list file.")
  p.add_argument("--output-dir", required=True, type=Path,
                 help="New directory only; existing directories are never reused.")
  p.add_argument("--max-files", type=int, default=10, help="Default 10; 0 means all.")
  p.add_argument("--start-seconds", type=float, default=0)
  p.add_argument("--duration-seconds", type=float, default=None,
                 help="Analyze this duration from the start offset; default whole remainder.")
  p.add_argument("--hop-seconds", type=float, default=10)
  p.add_argument("--pad-end", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--prompt", action="append", default=None,
                 help="Repeatable; replaces the three default mapped Text prompts.")
  p.add_argument("--no-text", action="store_true")
  p.add_argument("--ratio", action="append", type=parse_ratio, default=None,
                 help="Text:Audio; repeatable, defaults to 1:1 and 2:1.")
  p.add_argument("--final-normalization", choices=("none", "l2", "both"), default="none")
  p.add_argument("--mix-windows", action="store_true",
                 help="Also mix each 10-second window; default only pooled embeddings.")
  p.add_argument("--mapper-seed", type=int, default=0)
  p.add_argument("--resource-dir", type=Path, default=None)
  p.add_argument("--generate", action="store_true",
                 help="Explicitly load MRT2 and generate WAVs; requires an idle NVIDIA GPU.")
  p.add_argument("--generate-condition", action="append", default=None,
                 help="Glob of condition_id; repeatable. Default: all pooled conditions.")
  p.add_argument("--generate-seconds", type=float, default=5)
  p.add_argument("--generation-model", choices=("mrt2_small", "mrt2_base"), default="mrt2_small")
  p.add_argument("--checkpoint", type=Path, default=None)
  p.add_argument("--masked-tail-levels", type=int, choices=range(13), default=6)
  p.add_argument("--temperature", type=float, default=1.1)
  p.add_argument("--top-k", type=int, default=50)
  p.add_argument("--cfg-musiccoca", type=float, default=1.6)
  p.add_argument("--cfg-notes", type=float, default=2.4)
  p.add_argument("--cfg-drums", type=float, default=4)
  return p


def validate_args(args):
  for name in ("start_seconds", "hop_seconds", "generate_seconds", "temperature",
               "cfg_musiccoca", "cfg_notes", "cfg_drums"):
    if not math.isfinite(getattr(args, name)):
      raise ValueError(f"{name} must be finite")
  if args.start_seconds < 0 or args.max_files < 0:
    raise ValueError("start-seconds and max-files must be nonnegative")
  if args.hop_seconds <= 0 or round(args.hop_seconds * 16000) < 1:
    raise ValueError("hop-seconds must be at least one 16 kHz sample")
  if args.duration_seconds is not None and (
      not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0):
    raise ValueError("duration-seconds must be finite and positive")
  if args.generate_seconds <= 0 or not 0 <= args.temperature <= 3:
    raise ValueError("generate-seconds must be positive; temperature must be 0..3")
  if not 1 <= args.top_k <= 1024:
    raise ValueError("top-k must be 1..1024")
  if not all(-1 <= x <= 7 for x in (args.cfg_musiccoca, args.cfg_notes, args.cfg_drums)):
    raise ValueError("CFG values must be -1..7")
  if not 0 <= args.mapper_seed <= 2**32 - 1:
    raise ValueError("mapper-seed must be 0..2**32-1")
  if args.prompt and any(not s.strip() for s in args.prompt):
    raise ValueError("prompts must not be empty")
  if args.no_text and args.prompt:
    raise ValueError("--no-text and --prompt cannot be combined")
  if args.generate_condition and not args.generate:
    raise ValueError("--generate-condition requires --generate")


def collect_inputs(args):
  candidates = list(args.audio_paths)
  for directory in args.audio_dir:
    if not directory.is_dir():
      raise ValueError(f"audio directory does not exist: {directory}")
    candidates.extend(sorted(p for p in directory.rglob("*")
                             if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES))
  for listing in args.file_list:
    for line in listing.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      path = Path(line).expanduser()
      candidates.append(path if path.is_absolute() else listing.parent / path)
  result = list(dict.fromkeys(p.expanduser().resolve() for p in candidates))
  if not result:
    raise ValueError("No audio files specified. Supply paths, --audio-dir or --file-list.")
  for path in result:
    if not path.is_file():
      raise ValueError(f"audio file does not exist: {path}")
  return result[:args.max_files] if args.max_files else result


def norm(vector):
  return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def unit(vector):
  vector = np.asarray(vector, dtype=np.float32)
  length = norm(vector)
  if not np.all(np.isfinite(vector)) or not math.isfinite(length) or length <= 1e-12:
    raise ValueError("Cannot L2-normalize a nonfinite or zero embedding")
  return (vector / length).astype(np.float32)


def cosine(a, b):
  denominator = norm(a) * norm(b)
  if denominator <= 1e-24:
    return None  # Undefined, not an artificial zero similarity.
  return float(np.clip(np.dot(np.asarray(a, dtype=np.float64), b) / denominator, -1, 1))


def token_difference(a, b):
  a, b = np.asarray(a), np.asarray(b)
  if a.shape != (12,) or b.shape != (12,):
    raise ValueError("Expected twelve MusicCoCa RVQ levels")
  return {
      "match_rate_12": float(np.mean(a == b)),
      "match_rate_first_6": float(np.mean(a[:6] == b[:6])),
      "changed_positions_1based": (np.flatnonzero(a != b) + 1).tolist(),
  }


def write_json(path, value):
  with path.open("x", encoding="utf-8") as f:
    json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")


def write_csv(path, rows):
  if not rows:
    return
  keys = list(dict.fromkeys(key for row in rows for key in row))
  with path.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                       if isinstance(v, (list, dict)) else v for k, v in row.items()})


def save_array(path, array):
  with path.open("xb") as f:
    np.save(f, array, allow_pickle=False)


def sha256(path):
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def asset_metadata(resource_dir, with_text):
  names = ["audio_preprocessor.tflite", "music_encoder.tflite",
           "pretrained_vector_quantizer.tflite"]
  if with_text:
    names += ["text_encoder.tflite", "mapper.tflite", "spm.model"]
  result = []
  for name in names:
    path = resource_dir / name
    if not path.is_file():
      raise ValueError(f"Required local model missing (no downloads attempted): {path}")
    result.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
  return result


class Results:
  def __init__(self, root, model, masked_tail_levels):
    self.root, self.model = root, model
    self.masked_tail_levels = masked_tail_levels
    self.conditions = []
    self.comparisons = []
    self.sources = []
    self.errors = []
    self.generations = []
    self.pooled = []
    self.pooled_ids = []
    self.stat_vectors = defaultdict(list)

  def add(self, vector, metadata, array_file=None, array_row=None):
    vector = np.asarray(vector)
    if (vector.shape != (self.model.config.embedding_dim,) or
        vector.dtype != np.float32 or not np.all(np.isfinite(vector))):
      raise ValueError(f"Invalid embedding: shape={vector.shape}, dtype={vector.dtype}")
    started = time.perf_counter()
    tokens = np.asarray(self.model.tokenize(vector))
    if (tokens.shape != (12,) or not np.issubdtype(tokens.dtype, np.integer) or
        np.any(tokens < 0) or np.any(tokens >= self.model.config.rvq_codebook_size)):
      raise ValueError("Invalid RVQ tokens")
    quantization_seconds = time.perf_counter() - started
    if array_file is None:
      array_file = f"embeddings/{metadata['condition_id']}.npy"
      save_array(self.root / array_file, vector)
    masked = tokens.tolist()
    if self.masked_tail_levels:
      masked[-self.masked_tail_levels:] = [-1] * self.masked_tail_levels
    row = {
        **metadata, "shape": list(vector.shape), "dtype": str(vector.dtype),
        "norm": norm(vector), "coordinate_mean": float(np.mean(vector)),
        "coordinate_std": float(np.std(vector)), "tokens": tokens.tolist(),
        "generation_tokens_before_offset": masked,
        "quantization_seconds": quantization_seconds,
        "embedding_file": array_file, "embedding_row": array_row,
    }
    self.conditions.append(row)
    return row

  def compare(self, a, b, kind):
    diff = token_difference(a["tokens"], b["tokens"])
    active = 12 - self.masked_tail_levels
    diff["match_rate_active_levels"] = (
        float(np.mean(np.array(a["tokens"][:active]) == b["tokens"][:active]))
        if active else None)
    self.comparisons.append({
        "comparison": kind, "condition_a": a["condition_id"],
        "condition_b": b["condition_id"], **diff,
    })


def analyze_audio(path, index, args, results, texts):
  from magenta_rt import audio

  started = time.perf_counter()
  original = audio.Waveform.from_file(str(path))
  read_seconds = time.perf_counter() - started
  start = round(args.start_seconds * original.sample_rate)
  end = len(original) if args.duration_seconds is None else min(
      len(original), start + round(args.duration_seconds * original.sample_rate))
  if start >= end:
    raise ValueError("Requested audio interval is empty or outside the file")
  waveform = original[start:end]
  sr = results.model.config.sample_rate
  clip_seconds = results.model.config.clip_length
  resampled_length = int(len(waveform) * sr / original.sample_rate)
  if resampled_length < 1 or (
      not args.pad_end and resampled_length < results.model.config.clip_length_samples):
    raise ValueError("Selected interval produces no MusicCoCa windows")
  started = time.perf_counter()
  windows = np.asarray(results.model.embed(
      waveform, pool_across_time=False, hop_length=args.hop_seconds,
      pad_end=args.pad_end, mono_strategy="average"))
  embedding_seconds = time.perf_counter() - started
  if (windows.ndim != 2 or windows.shape[0] == 0 or
      windows.shape[1] != results.model.config.embedding_dim or
      windows.dtype != np.float32 or not np.all(np.isfinite(windows))):
    raise ValueError("MusicCoCa returned no valid window embeddings")
  # This is exactly embed_batch_audio(pool_across_time=True)'s mean; no second
  # encoder pass or independently normalized-window mean is substituted.
  pooled = np.mean(windows, axis=0)
  windows_unit = np.stack([unit(v) for v in windows])
  pooled_unit = unit(pooled)
  slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem)[:40] or "audio"
  sample_id = f"a{index:03d}_{slug}"
  arrays = {"windows_raw": windows, "windows_l2": windows_unit,
            "pooled_raw": pooled, "pooled_l2": pooled_unit}
  for name, vector in arrays.items():
    save_array(results.root / "embeddings" / f"{sample_id}.{name}.npy", vector)
  source = {
      "sample_id": sample_id, "file": str(path), "sha256": sha256(path),
      "analysis_status": "partial",
      "original_seconds": original.seconds,
      "original_sample_rate": original.sample_rate, "original_channels": original.num_channels,
      "used_start_seconds": start / original.sample_rate,
      "used_end_seconds": end / original.sample_rate,
      "used_seconds": waveform.seconds, "num_windows": len(windows),
      "embedding_shape": list(windows.shape), "embedding_dtype": str(windows.dtype),
      "embedding_seconds": embedding_seconds, "read_seconds": read_seconds,
      "embedding_timing_includes_lazy_load": not results.sources,
      "embedding_method": "MusicCoCa.embed(pool_across_time=False); np.mean(axis=0)",
      "pooling": "arithmetic mean of raw windows, including padded tail if enabled",
  }
  results.sources.append(source)
  results.pooled.append(pooled)
  results.pooled_ids.append(sample_id)
  for kind, values in arrays.items():
    results.stat_vectors[kind].extend(values if values.ndim == 2 else [values])
  for scope, vectors, array_kind in (
      ("pooled", [pooled], "pooled"), ("window", windows, "windows")):
    for wi, raw in enumerate(vectors):
      window_start = start / original.sample_rate
      if scope == "window":
        window_start += wi * round(args.hop_seconds * sr) / sr
      window_end = min(end / original.sample_rate, window_start + clip_seconds)
      if scope == "pooled":
        window_end = end / original.sample_rate
      common = {
          "sample_id": sample_id, "file": str(path), "scope": scope,
          "window_index": wi if scope == "window" else None,
          "start_seconds": window_start, "end_seconds": window_end,
          "padded_seconds": max(0., clip_seconds - (window_end - window_start))
                            if scope == "window" else None,
          "embedding_method": "musiccoca_audio",
          "cosine_to_raw_pooled": cosine(raw, pooled),
      }
      base_id = f"{sample_id}.{scope}" + (f"{wi:04d}" if scope == "window" else "")
      variants = {}
      for normalization, vector in (("raw", raw), ("l2", unit(raw))):
        metadata = {**common, "condition_id": f"{base_id}.{normalization}",
                    "audio_normalization": normalization, "final_normalization": "none"}
        variants[normalization] = results.add(
            vector, metadata, f"embeddings/{sample_id}.{array_kind}_{normalization}.npy",
            wi if scope == "window" else None)
      results.compare(variants["raw"], variants["l2"], "audio_raw_vs_l2")
      if scope == "pooled" or args.mix_windows:
        analyze_mixes(raw, common, base_id, args, results, texts)
  source["analysis_status"] = "complete"
  LOG.info("%s: %d windows, pooled norm %.4f, encode %.3fs",
           path.name, len(windows), norm(pooled), embedding_seconds)


def analyze_mixes(raw, common, base_id, args, results, texts):
  from magenta_rt.realtime_server import blend_style_embeddings

  ratios = args.ratio or [(1., 1.), (2., 1.)]
  final_options = ("none", "l2") if args.final_normalization == "both" else (args.final_normalization,)
  # This metric describes the Audio input, not the mixed output.
  common = dict(common)
  common["audio_cosine_to_raw_pooled"] = common.pop("cosine_to_raw_pooled")
  for text_id, prompt, text_vector in texts:
    rows = {}
    for audio_kind, audio_vector in (("raw", raw), ("l2", unit(raw))):
      for ri, ratio in enumerate(ratios):
        text_share = ratio[0] / sum(ratio)
        audio_share = ratio[1] / sum(ratio)
        mixed = blend_style_embeddings([text_vector, audio_vector], [text_share, audio_share])
        for final in final_options:
          vector = unit(mixed) if final == "l2" else mixed
          condition_id = f"{base_id}.mix_{text_id}.r{ri:02d}.audio_{audio_kind}.final_{final}"
          row = results.add(vector, {
              **common, "condition_id": condition_id, "embedding_method": "text_audio_mix",
              "text_id": text_id, "text": prompt, "mapper_seed": args.mapper_seed,
              "ratio_text_audio": f"{ratio[0]:g}:{ratio[1]:g}",
              "text_share": text_share, "audio_share": audio_share,
              "audio_normalization": audio_kind, "final_normalization": final,
              "text_input_norm": norm(text_vector), "audio_input_norm": norm(audio_vector),
              "weighted_text_norm": text_share * norm(text_vector),
              "weighted_audio_norm": audio_share * norm(audio_vector),
              "mixed_norm_before_final_normalization": norm(mixed),
              "cosine_to_text_input": cosine(vector, text_vector),
              "cosine_to_audio_input": cosine(vector, audio_vector),
          })
          rows[audio_kind, ri, final] = row
          if ri:
            results.compare(rows[audio_kind, 0, final], row, "ratio_vs_first_ratio")
          if audio_kind == "l2":
            results.compare(rows["raw", ri, final], row, "mixed_audio_raw_vs_l2")
          if final == "l2" and "none" in final_options:
            results.compare(rows[audio_kind, ri, "none"], row, "final_none_vs_l2")


def summaries(results):
  dimensions = []
  for group, vectors in results.stat_vectors.items():
    matrix = np.asarray(vectors, dtype=np.float64)
    means, stds = matrix.mean(axis=0), matrix.std(axis=0)
    for d, (mean, std) in enumerate(zip(means, stds)):
      dimensions.append({"group": group, "dimension": d, "count": len(matrix),
                         "mean": float(mean), "std_ddof_0": float(std)})
  pairwise = []
  if results.pooled:
    matrix = np.array([[cosine(a, b) for b in results.pooled] for a in results.pooled])
    save_array(results.root / "pooled_cosine.npy", matrix)
    for i, a in enumerate(results.pooled_ids):
      for j in range(i + 1, len(results.pooled_ids)):
        pairwise.append({"sample_a": a, "sample_b": results.pooled_ids[j],
                         "cosine": float(matrix[i, j])})
  groups = defaultdict(list)
  for row in results.conditions:
    key = tuple(str(row.get(k, "")) for k in (
        "scope", "embedding_method", "audio_normalization", "final_normalization",
        "text_id", "ratio_text_audio"))
    groups[key].append(row["tokens"])
  diversity = []
  for group, token_rows in groups.items():
    for level in range(12):
      counts = Counter(t[level] for t in token_rows)
      probabilities = np.array(list(counts.values()), dtype=np.float64) / len(token_rows)
      diversity.append({
          **dict(zip(("scope", "embedding_method", "audio_normalization", "final_normalization",
                      "text_id", "ratio_text_audio"), group)),
          "level_1based": level + 1, "count": len(token_rows), "unique_tokens": len(counts),
          "unique_sequences_12": len({tuple(t) for t in token_rows}),
          "unique_sequences_first_6": len({tuple(t[:6]) for t in token_rows}),
          "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
          "histogram": dict(sorted(counts.items())),
      })
  return dimensions, pairwise, diversity


def require_idle_gpu():
  try:
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10)
  except (OSError, subprocess.SubprocessError) as exc:
    raise RuntimeError("Cannot confirm an idle NVIDIA GPU; generation was not started") from exc
  if output.stdout.strip():
    raise RuntimeError("GPU processes are active; generation was not started. "
                       "Embedding/RVQ results are retained. Run generation during an idle period.")
  return output.stdout


def generate_audio(args, results):
  # Import JAX only after explicit opt-in and the idle-GPU check. Never use the
  # WebSocket/server instance or mutate its state. No checkpoint downloading.
  patterns = args.generate_condition or ["*.pooled.*"]
  rows = [r for r in results.conditions
          if any(fnmatch.fnmatchcase(r["condition_id"], p) for p in patterns)]
  if not rows:
    raise ValueError("No generation conditions match --generate-condition")
  require_idle_gpu()
  from magenta_rt import paths
  checkpoint = (args.checkpoint.expanduser().resolve() if args.checkpoint else
                paths.magenta_home() / "checkpoints" / f"{args.generation_model}.safetensors")
  if not checkpoint.is_file():
    raise ValueError(f"Local checkpoint missing (no downloads attempted): {checkpoint}")
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  from magenta_rt import MagentaRT2Jax
  import soundfile as sf

  started = time.perf_counter()
  mrt = MagentaRT2Jax(
      size=args.generation_model, checkpoint=str(checkpoint), style_model=results.model,
      temperature=args.temperature, top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca, cfg_notes=args.cfg_notes, cfg_drums=args.cfg_drums,
      musiccoca_masked_tail_levels=args.masked_tail_levels, int16_outputs=False, num_cfgs=2)
  initialization_seconds = time.perf_counter() - started
  results.root.joinpath("generated").mkdir()
  frames = max(1, math.ceil(args.generate_seconds * 25))
  for row in rows:
    vector = np.load(results.root / row["embedding_file"], allow_pickle=False)
    if row["embedding_row"] is not None:
      vector = vector[row["embedding_row"]]
    # Public generate accepts the continuous vector and deterministically repeats
    # this exact RVQ operation. Assert correspondence to the recorded tokens.
    if mrt.tokenize_style(vector).tolist() != row["tokens"]:
      raise RuntimeError("Generation tokens differ from recorded RVQ tokens")
    started = time.perf_counter()
    waveform, _ = mrt.generate(style=vector, frames=frames, state=None,
                               notes=[-1] * 128, drums=[-1])
    generation_seconds = time.perf_counter() - started
    file = f"generated/{row['condition_id']}.wav"
    with (results.root / file).open("xb") as f:
      sf.write(f, waveform.samples, waveform.sample_rate, format="WAV", subtype="FLOAT")
    results.generations.append({
        "condition_id": row["condition_id"], "file": file,
        "generation_seconds": generation_seconds, "output_seconds": waveform.seconds,
        "initialization_seconds": initialization_seconds, "frames": frames,
        "generation_seed": 0, "parameter_rng_seed": 42,
        "seed_source": "existing MagentaRT2System._compile fixed RNGs; state=None per condition",
        "midi": "all 128 masked (-1)", "drums": "masked (-1)",
        "masked_tail_levels": args.masked_tail_levels, "num_cfgs": 2,
        "temperature": args.temperature, "top_k": args.top_k,
        "cfg_musiccoca": args.cfg_musiccoca, "cfg_notes": args.cfg_notes,
        "cfg_drums": args.cfg_drums, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint) if not results.generations else
                              results.generations[0]["checkpoint_sha256"],
        "tokens": row["tokens"],
        "generation_tokens_before_offset": row["generation_tokens_before_offset"],
    })


def run_experiment(args, model=None):
  validate_args(args)
  inputs = collect_inputs(args)
  output = args.output_dir.expanduser().resolve()
  if output.exists():
    raise ValueError(f"Output directory already exists; refusing to overwrite: {output}")
  from magenta_rt import musiccoca, paths
  resource_dir = (args.resource_dir or paths.musiccoca_dir()).expanduser().resolve()
  assets = asset_metadata(resource_dir, not args.no_text) if model is None else []
  model = model if model is not None else musiccoca.MusicCoCa(resource_dir=resource_dir)
  if model.config.rvq_depth != 12:
    raise ValueError("This MRT2 baseline expects twelve MusicCoCa RVQ levels")
  output.mkdir(parents=True, exist_ok=False)
  (output / "embeddings").mkdir()
  results = Results(output, model, args.masked_tail_levels)
  settings = {k: str(v) if isinstance(v, Path) else
              [str(x) if isinstance(x, Path) else x for x in v] if isinstance(v, list) else v
              for k, v in vars(args).items()}
  manifest = {
      "schema_version": 1, "status": "running",
      "started_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
      "inputs": [str(p) for p in inputs], "resource_dir": str(resource_dir), "assets": assets,
      "model_class": type(model).__name__, "musiccoca_config": vars(model.config),
      "python": sys.version, "platform": platform.platform(),
      "versions": {n: importlib.metadata.version(n) for n in ("numpy", "magenta-rt", "ai-edge-litert")},
      "script_sha256": sha256(Path(__file__)),
      "position_convention": "RVQ positions are 1-based; dimensions/window indices are 0-based",
      "timing_note": "First audio/Text calls include lazy model loading; no separate warmup.",
      "generation": "requested" if args.generate else "not_requested; JAX not loaded",
  }
  write_json(output / "settings.json", manifest)
  started = time.perf_counter()
  fatal_error = None
  try:
    texts = []
    for i, prompt in enumerate([] if args.no_text else (args.prompt or DEFAULT_PROMPTS), 1):
      tick = time.perf_counter()
      vector = np.asarray(model.embed(prompt, use_mapper=True, seed=args.mapper_seed))
      seconds = time.perf_counter() - tick
      text_id = f"t{i:03d}"
      results.add(vector, {"condition_id": text_id, "scope": "text",
                          "embedding_method": "text_mapper_l2", "text_id": text_id,
                          "text": prompt, "mapper_seed": args.mapper_seed,
                          "embedding_seconds": seconds, "embedding_timing_includes_lazy_load": i == 1})
      results.stat_vectors["text_mapper_l2"].append(vector)
      texts.append((text_id, prompt, vector))
    for i, path in enumerate(inputs, 1):
      try:
        analyze_audio(path, i, args, results, texts)
      except Exception as exc:
        LOG.exception("Audio analysis failed: %s", path)
        results.errors.append({"file": str(path), "stage": "audio_analysis", "error": str(exc)})
    if not results.sources:
      raise ValueError("No audio file was successfully analyzed; see errors.json")
    if args.generate:
      try:
        generate_audio(args, results)
      except Exception as exc:
        results.errors.append({"stage": "generation", "error": str(exc)})
        LOG.exception("Generation failed; retaining Embedding/RVQ results")
  except BaseException as exc:
    fatal_error = exc
    results.errors.append({"stage": "experiment", "error": str(exc)})
  finally:
    dimensions, pairwise, diversity = summaries(results)
    for name, rows in (("sources", results.sources), ("conditions", results.conditions),
                       ("token_comparisons", results.comparisons), ("dimension_stats", dimensions),
                       ("pooled_cosine", pairwise), ("token_diversity", diversity),
                       ("generations", results.generations)):
      write_csv(output / f"{name}.csv", rows)
    write_json(output / "conditions.json", results.conditions)
    write_json(output / "errors.json", results.errors)
    manifest.update(status="failed" if fatal_error or not results.sources else
                    "partial" if results.errors else "complete",
                    encoded_audio_files=len(results.sources),
                    successful_audio_files=sum(s["analysis_status"] == "complete"
                                               for s in results.sources),
                    errors=len(results.errors),
                    conditions=len(results.conditions), generated_wavs=len(results.generations),
                    experiment_seconds=time.perf_counter() - started,
                    generation="completed" if args.generate and not any(
                        r["stage"] == "generation" for r in results.errors) and results.generations
                    else "failed_or_not_started" if args.generate else "not_requested; JAX not loaded")
    write_json(output / "manifest.json", manifest)
  if fatal_error:
    raise fatal_error
  LOG.info("Results: %s (%s)", output, manifest["status"])
  return results


def main(argv=None):
  parser = make_parser()
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  try:
    results = run_experiment(args)
  except (OSError, ValueError, RuntimeError) as exc:
    LOG.error("%s", exc)
    return 1
  return 1 if results.errors else 0


if __name__ == "__main__":
  raise SystemExit(main())
