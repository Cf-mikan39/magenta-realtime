"""RVQ postprocessing in the MRT2 environment; no PyTorch/JAX import."""

from collections import Counter
from pathlib import Path
import time

import numpy as np

from .common import (new_directory, read_json, save_array, sha256,
                     token_difference, write_csv, write_json)


def quantize_results(input_dir, output_dir, resource_dir=None, model=None):
  source = Path(input_dir).resolve()
  evaluation = (source / "evaluation.json").is_file()
  metadata = read_json(source / ("evaluation.json" if evaluation else "inference.json"))
  if model is None:
    from magenta_rt import musiccoca, paths
    resources = Path(resource_dir or paths.musiccoca_dir()).expanduser().resolve()
    quantizer = resources / "pretrained_vector_quantizer.tflite"
    if not quantizer.is_file():
      raise ValueError("Local MusicCoCa RVQ is missing; no downloads attempted")
    checksum = sha256(quantizer)
    expected = [a["sha256"] for a in metadata["audio_contract"].get("assets", [])
                if a["name"] == quantizer.name]
    if expected and expected != [checksum]:
      raise ValueError("RVQ weights differ from the Audio teacher cache")
    model = musiccoca.MusicCoCa(resource_dir=resources)
  else:
    checksum = "injected_test_quantizer"
  arrays = {}
  for name, file in (("predicted", "predictions.npy" if evaluation else "prediction.npy"),
                     ("teacher", "teachers.npy"), ("baseline", "baseline.npy")):
    if not evaluation and name != "predicted":
      continue
    expected_hash = (metadata["array_sha256"][file] if evaluation else metadata["prediction_sha256"])
    if sha256(source / file) != expected_hash:
      raise ValueError("Inference/evaluation arrays changed after export")
    values = np.load(source / file, allow_pickle=False)
    if values.ndim == 1:
      values = values[None, :]
    if (values.ndim != 2 or values.shape[1] != 768 or values.dtype != np.float32 or
        not np.all(np.isfinite(values))):
      raise ValueError("RVQ requires finite float32 Audio embeddings of dimension 768")
    arrays[name] = values
  started = time.perf_counter()
  tokens = {name: np.asarray(model.tokenize(values)) for name, values in arrays.items()}
  elapsed = time.perf_counter() - started
  if any(t.shape != (len(arrays[name]), 12) or not np.issubdtype(t.dtype, np.integer) or
         np.any(t < 0) or np.any(t >= 1024) for name, t in tokens.items()):
    raise ValueError("Invalid RVQ token shape/dtype/range")
  root = new_directory(output_dir)
  comparisons, metrics = [], {}
  for name in ("predicted", "baseline"):
    if "teacher" in tokens:
      matches = tokens[name] == tokens["teacher"]
      metrics[name] = {"match_rate_12": float(matches.mean()),
                       "match_rate_first_6": float(matches[:, :6].mean()),
                       "per_level_match_rate": matches.mean(0).tolist()}
      for i, (a, b) in enumerate(zip(tokens[name], tokens["teacher"])):
        comparisons.append({"row": i, "prediction": name, **token_difference(a, b)})
  distributions = []
  for name, values in tokens.items():
    save_array(root / f"{name}_tokens.npy", values.astype(np.int32))
    for level in range(12):
      histogram = Counter(int(v) for v in values[:, level])
      probabilities = np.array(list(histogram.values()), dtype=np.float64) / len(values)
      distributions.append({"group": name, "level_1based": level + 1,
                            "unique_tokens": len(histogram), "histogram": dict(sorted(histogram.items())),
                            "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
                            "unique_sequences_12": len({tuple(row) for row in values}),
                            "unique_sequences_first_6": len({tuple(row[:6]) for row in values})})
  write_csv(root / "comparisons.csv", comparisons)
  write_csv(root / "token_distributions.csv", distributions)
  write_json(root / "tokens.json", {k: v.tolist() for k, v in tokens.items()})
  write_json(root / "metrics.json", metrics)
  write_json(root / "rvq.json", {"input_dir": str(source), "source_metadata": metadata,
      "quantizer_sha256": checksum, "seconds": elapsed,
      "normalization": "unchanged from exported prediction/teacher; same effective policy",
      "warning": "Token IDs have no numeric distance; token agreement does not measure music quality"})
  return metrics
