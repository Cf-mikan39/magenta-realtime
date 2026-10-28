"""Embedding metrics, train-mean baseline and observed-pair retrieval."""

from pathlib import Path

import numpy as np

from .cache import paired_caches
from .common import (new_directory, save_array, sha256, unit_rows, write_csv, write_json)
from .encoders import encoder_identity
from .inference import ProjectionPredictor


def embedding_metrics(predicted, teachers):
  predicted, teachers = np.asarray(predicted, dtype=np.float64), np.asarray(teachers, dtype=np.float64)
  pn, tn = np.linalg.norm(predicted, axis=1), np.linalg.norm(teachers, axis=1)
  denominator = pn * tn
  valid = denominator > 1e-20
  cosines = np.zeros(len(predicted))
  cosines[valid] = np.clip(np.sum(predicted[valid] * teachers[valid], axis=1) /
                          denominator[valid], -1, 1)
  rows = [{"cosine": float(cosines[i]) if valid[i] else None,
           "mse": float(np.mean((predicted[i] - teachers[i])**2)),
           "predicted_norm": float(pn[i]), "teacher_norm": float(tn[i]),
           "absolute_norm_error": float(abs(pn[i] - tn[i]))} for i in range(len(predicted))]
  summary = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
             if any(r[key] is not None for r in rows) else None for key in rows[0]}
  summary.update(count=len(rows), undefined_cosines=int(np.sum(~valid)),
                 output_mean_variance=float(np.var(predicted, axis=0).mean()))
  return rows, summary


def retrieval(predicted, teachers, image_keys, audio_keys):
  """Every observed Audio target for an image is positive; unique-image average.

  Unpaired gallery members are not asserted to be semantic negatives.
  """
  gallery, gallery_keys, gallery_seen, queries, positives = [], [], set(), {}, {}
  for index, (image_key, audio_key) in enumerate(zip(image_keys, audio_keys)):
    if audio_key not in gallery_seen:
      gallery_seen.add(audio_key)
      gallery_keys.append(audio_key)
      gallery.append(teachers[index])
    queries.setdefault(image_key, predicted[index])
    positives.setdefault(image_key, set()).add(audio_key)
  gallery = unit_rows(np.asarray(gallery, dtype=np.float32))
  hits = {1: 0, 5: 0}
  undefined = 0
  for image_key, vector in queries.items():
    if np.linalg.norm(vector) <= 1e-12:
      undefined += 1
      continue
    query = unit_rows(np.asarray(vector, dtype=np.float32)[None, :])[0]
    ranking = np.argsort(-(gallery @ query), kind="stable")
    for k in hits:
      hits[k] += any(gallery_keys[i] in positives[image_key] for i in ranking[:k])
  valid = len(queries) - undefined
  return {"recall_at_1": hits[1] / valid if valid else None,
          "recall_at_5": hits[5] / valid if valid else None,
          "undefined_zero_queries": undefined,
          "unique_images": len(queries), "gallery_size": len(gallery),
          "positive_definition": "all observed pairs sharing image content key in this split",
          "warning": "Unpaired candidates may also be musically valid; R@5 is trivial for gallery <=5"}


def evaluate(checkpoint, image_dir, audio_dir, output_dir, split="test",
             output_normalization="as_trained", device="cpu", batch_size=64):
  from .training import CachedPairs
  from torch.utils.data import DataLoader

  if batch_size < 1:
    raise ValueError("batch_size must be positive")
  predictor = ProjectionPredictor(checkpoint, device)
  cp = predictor.checkpoint
  image, audio = paired_caches(image_dir, audio_dir)
  if encoder_identity(image.manifest["contract"]) != encoder_identity(cp["image_contract"]):
    raise ValueError("Evaluation Image Encoder/weights/preprocessing differs from checkpoint")
  if audio.manifest["contract"] != cp["audio_contract"]:
    raise ValueError("Evaluation Audio model/preprocessing/pooling differs from checkpoint")
  data = CachedPairs(image, audio, split, cp["config"]["target_normalization"])
  predictions, teachers = [], []
  for x, y in DataLoader(data, batch_size=batch_size, shuffle=False):
    predictions.extend(predictor.predict_features(x.numpy(), image.manifest["contract"],
                                                 output_normalization))
    teachers.extend(y.numpy())
  predictions, teachers = np.stack(predictions), np.stack(teachers)
  mean = cp["train_mean"].numpy()
  baseline = np.repeat(mean[None, :], len(data), axis=0)
  # Teacher and prediction receive the SAME effective output normalization for
  # evaluation/RVQ. Raw source teachers are separately retained for auditing.
  l2 = output_normalization == "l2" or cp["config"]["training_output_normalization"] == "l2"
  if l2:
    teachers, baseline = unit_rows(teachers), unit_rows(baseline)
  raw_teachers = np.stack([audio.audio(p.pair_id) for p in data.pairs])
  rows, scores = embedding_metrics(predictions, teachers)
  baseline_rows, baseline_scores = embedding_metrics(baseline, teachers)
  image_keys = [image.manifest["refs"][p.pair_id]["key"] for p in data.pairs]
  audio_keys = [audio.target_key(p.pair_id) for p in data.pairs]
  for i, (pair, row, base) in enumerate(zip(data.pairs, rows, baseline_rows)):
    row.update(pair_id=pair.pair_id, image_id=pair.image_id, audio_id=pair.audio_id,
               image_key=image_keys[i], audio_key=audio_keys[i], split=pair.split,
               raw_teacher_norm=float(np.linalg.norm(raw_teachers[i])),
               **{"mean_baseline_" + k: v for k, v in base.items()})
  metrics = {"projection": scores, "train_mean_baseline": baseline_scores,
             "retrieval": retrieval(predictions, teachers, image_keys, audio_keys),
             "baseline_retrieval": retrieval(baseline, teachers, image_keys, audio_keys)}
  root = new_directory(output_dir)
  dimension_rows = []
  for name, values in (("predicted", predictions), ("teacher", teachers), ("baseline", baseline)):
    for dimension, (mean, variance) in enumerate(zip(values.mean(0), values.var(0))):
      dimension_rows.append({"group": name, "dimension": dimension, "mean": float(mean),
                             "variance_ddof_0": float(variance)})
  arrays = {"predictions.npy": predictions, "teachers.npy": teachers,
            "raw_teachers.npy": raw_teachers, "baseline.npy": baseline}
  for name, values in arrays.items():
    save_array(root / name, np.asarray(values, dtype=np.float32))
  write_csv(root / "samples.csv", rows)
  write_csv(root / "dimension_stats.csv", dimension_rows)
  write_json(root / "metrics.json", metrics)
  write_json(root / "evaluation.json", {
      "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(Path(checkpoint)),
      "image_contract": cp["image_contract"], "audio_contract": cp["audio_contract"],
      "target_normalization": cp["config"]["target_normalization"],
      "training_output_normalization": cp["config"]["training_output_normalization"],
      "output_normalization": output_normalization, "effective_output_l2": l2,
      "teacher_policy": "training target transform, then same effective output L2 as predictions",
      "cache_ids": {"image": image.manifest["fingerprint"], "audio": audio.manifest["fingerprint"]},
      "split": split, "synthetic": cp["synthetic"],
      "array_sha256": {name: sha256(root / name) for name in arrays}})
  return metrics
