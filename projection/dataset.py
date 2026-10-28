"""CSV pair manifests with connected-component split protection."""

import csv
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random

from .common import fingerprint, write_csv

SPLITS = ("train", "val", "test")
FIELDS = ("pair_id", "image_id", "image_path", "audio_id", "audio_path", "split",
          "start_seconds", "duration_seconds", "group_id", "image_source_id",
          "audio_source_id", "pooling", "window_index")


@dataclass(frozen=True)
class Pair:
  pair_id: str
  image_id: str
  image_path: str
  audio_id: str
  audio_path: str
  split: str = ""
  start_seconds: float = 0.
  duration_seconds: object = None
  group_id: str = ""
  image_source_id: str = ""
  audio_source_id: str = ""
  pooling: str = ""
  window_index: object = None


def read_pairs(csv_path, require_files=False):
  csv_path = Path(csv_path).expanduser().resolve()
  result, ids = [], set()
  with csv_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    required = {"image_id", "image_path", "audio_id", "audio_path"}
    if not required.issubset(reader.fieldnames or []):
      raise ValueError(f"CSV requires {sorted(required)}")
    for index, row in enumerate(reader, 1):
      row = {k: (v or "").strip() for k, v in row.items() if k is not None}
      if any(not row[k] for k in required):
        raise ValueError(f"Empty required CSV value at row {index}")
      for key in ("image_path", "audio_path"):
        path = Path(row[key]).expanduser()
        path = path if path.is_absolute() else csv_path.parent / path
        row[key] = str(path.resolve())
        if require_files and not path.is_file():
          raise ValueError(f"Missing file: {path}")
      pair_id = row.get("pair_id") or f"pair{index:08d}"
      if pair_id in ids:
        raise ValueError(f"Duplicate pair_id: {pair_id}")
      ids.add(pair_id)
      start = float(row.get("start_seconds") or 0)
      duration = float(row["duration_seconds"]) if row.get("duration_seconds") else None
      if (not math.isfinite(start) or start < 0 or duration is not None and
          (not math.isfinite(duration) or duration <= 0)):
        raise ValueError(f"Invalid audio interval at row {index}")
      split = {"validation": "val"}.get(row.get("split", ""), row.get("split", ""))
      if split and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
      pooling = row.get("pooling", "")
      window = int(row["window_index"]) if row.get("window_index") else None
      if pooling not in ("", "mean", "window") or window is not None and window < 0:
        raise ValueError(f"Invalid pooling/window_index at row {index}")
      result.append(Pair(
          pair_id, row["image_id"], row["image_path"], row["audio_id"], row["audio_path"],
          split, start, duration, row.get("group_id") or row.get("source_id", ""),
          row.get("image_source_id", ""), row.get("audio_source_id", ""), pooling, window))
  if not result:
    raise ValueError("Pair CSV is empty")
  for modality in ("image", "audio"):
    paths = {}
    for pair in result:
      identifier, path = getattr(pair, modality + "_id"), getattr(pair, modality + "_path")
      if identifier in paths and paths[identifier] != path:
        raise ValueError(f"{modality}_id {identifier} points to different files")
      paths[identifier] = path
  return result


def assign_splits(pairs, seed=42, ratios=(.8, .1, .1), content_keys=None):
  """Unite shared IDs/files/sources (including transitive links), then split.

  Explicit labels are preserved; a conflicting connected component is rejected.
  Optional content_keys catches byte-identical copies with different IDs/paths.
  """
  if len(ratios) != 3 or any(not math.isfinite(v) or v <= 0 for v in ratios):
    raise ValueError("Three positive finite split ratios are required")
  parent = list(range(len(pairs)))

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i

  seen = {}
  for i, pair in enumerate(pairs):
    keys = [("image_id", pair.image_id), ("image_path", pair.image_path),
            ("audio_id", pair.audio_id), ("audio_path", pair.audio_path)]
    keys += [("source", value) for value in
             (pair.group_id, pair.image_source_id, pair.audio_source_id) if value]
    if content_keys:
      keys += [("image_content", content_keys[pair.image_path]),
               ("audio_content", content_keys[pair.audio_path])]
    for key in keys:
      if key in seen:
        parent[find(i)] = find(seen[key])
      else:
        seen[key] = i
  groups = {}
  for i in range(len(pairs)):
    groups.setdefault(find(i), []).append(i)
  counts, assigned, unlabelled = {s: 0 for s in SPLITS}, {}, []
  for key, members in groups.items():
    labels = {pairs[i].split for i in members if pairs[i].split}
    if len(labels) > 1:
      raise ValueError("Data leakage: shared image/audio/source crosses splits: " +
                       ", ".join(pairs[i].pair_id for i in members))
    if labels:
      label = labels.pop()
      assigned[key] = label
      counts[label] += 1
    else:
      unlabelled.append(key)
  random.Random(seed).shuffle(unlabelled)
  targets = dict(zip(SPLITS, [len(groups) * r / sum(ratios) for r in ratios]))
  for position, key in enumerate(unlabelled):
    # Reserve a component for each missing split when possible. Allocation is
    # by independent component, not row; large groups can skew row proportions.
    missing = [s for s in SPLITS if counts[s] == 0]
    remaining = len(unlabelled) - position
    label = missing[0] if missing and remaining <= len(missing) else max(
        SPLITS, key=lambda s: targets[s] - counts[s])
    assigned[key] = label
    counts[label] += 1
  return [replace(pair, split=assigned[find(i)]) for i, pair in enumerate(pairs)]


def pairs_payload(pairs):
  return [asdict(pair) for pair in pairs]


def pairs_fingerprint(pairs):
  return fingerprint(pairs_payload(pairs))


def write_pairs(path, pairs):
  write_csv(path, pairs_payload(pairs))
