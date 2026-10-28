"""Optional metrics-only W&B monitoring; local training is always authoritative.

No wandb import/network when disabled. Online operations require explicit config.
Offline resume creates a linked new segment because SDK offline resume is ignored.
"""

from copy import deepcopy
from importlib import import_module
import json
import logging
import math
import os
from pathlib import Path
import uuid

from .common import fingerprint, read_json

LOG = logging.getLogger(__name__)


def public_contract(contract):
  """Allowlist only model identities; never paths, aliases, IDs or media."""
  result = {k: contract[k] for k in ("name", "model", "dimension", "implementation",
      "feature_normalization", "normalization", "config", "versions", "adapter_sha256",
      "implementation_sha256", "source_sha256", "hop_seconds", "pad_end", "mono_strategy",
      "default_pooling", "default_window_index") if k in contract}
  result["weight_sha256"] = [entry["sha256"] for entry in contract.get("weights", [])]
  result["asset_sha256"] = [entry["sha256"] for entry in contract.get("assets", [])]
  return result


def training_run_config(config, image, audio, model_config, device):
  counts = {split: sum(p.split == split for p in image.pairs) for split in ("train", "val", "test")}
  data = {"image_encoder": image.manifest["contract"].get("name"),
      "target_normalization": config["target_normalization"],
      "training_output_normalization": config["training_output_normalization"],
      "projection": model_config, "loss": config["loss"], "training": config["training"],
      "device_type": device.split(":")[0],
      "dataset": {"pairs_fingerprint": image.manifest["pairs_fingerprint"], "counts": counts,
                  "synthetic": image.manifest["synthetic"] or audio.manifest["synthetic"]},
      "cache_ids": {"image": image.manifest["fingerprint"], "audio": audio.manifest["fingerprint"]},
      "image_model": public_contract(image.manifest["contract"]),
      "audio_model": public_contract(audio.manifest["contract"])}
  identity = deepcopy(data)
  identity["training"].pop("epochs")
  data["training_signature"] = fingerprint(identity)
  return data


class TrackingUnavailable(RuntimeError):
  pass


class WandbTracker:
  def __init__(self, root, config, run_config, previous=None, start_epoch=1, job_type="train"):
    self.root = Path(root)
    self.config, self.run_config = deepcopy(config), deepcopy(run_config)
    self.run = None
    self.module = None
    self.state = {"enabled": config["enabled"], "mode": config["mode"],
                  "status": "disabled" if not config["enabled"] else "initializing",
                  "run_id": None, "project": config["project"], "entity": config["entity"],
                  "last_logged_epoch": 0, "errors": [], "parent_run_id": None,
                  "training_signature": run_config["training_signature"]}
    if config["enabled"]:
      try:
        self._start(previous, start_epoch, job_type)
      except Exception as exc:
        self._failure("initialization", exc)
    self._save()

  def _save(self):
    temporary = self.root / "wandb_status.json.tmp"
    with temporary.open("w", encoding="utf-8") as file:
      json.dump(self.state, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(self.root / "wandb_status.json")

  def _failure(self, stage, exc):
    # SDK exception strings may contain URLs or credentials. Persist only our
    # controlled guidance and the exception type; detailed SDK logs stay local.
    message = str(exc) if isinstance(exc, TrackingUnavailable) else (
        "W&B operation failed; local CSV/JSON/checkpoints are retained. See local SDK logs.")
    self.state["status"] = "failed"
    self.state["errors"].append({"stage": stage, "type": type(exc).__name__, "message": message})
    LOG.error("W&B %s: %s (%s)", stage, message, type(exc).__name__)

  def _start(self, previous, start_epoch, job_type):
    try:
      self.module = import_module("wandb")
    except ImportError as exc:
      raise TrackingUnavailable("W&B is not installed. Install wandb in the Projection environment, "
                                "or set wandb.enabled=false.") from exc
    if getattr(self.module, "run", None) is not None:
      raise TrackingUnavailable("Another W&B run is active in this process; finish it before starting monitoring.")
    settings = self.module.Settings(
        init_timeout=20, disable_git=True, disable_code=True, save_code=False,
        console="off", x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True)
    mode = self.config["mode"]
    if mode == "online":
      apikey = import_module("wandb.sdk.lib.apikey")
      if not os.environ.get("WANDB_API_KEY") and not apikey.api_key(settings=settings):
        raise TrackingUnavailable("W&B login required: run wandb login or set WANDB_API_KEY. "
                                  "Use mode=offline for network-free logging.")
    previous = previous or {}
    compatible = (previous.get("run_id") and previous.get("project") == self.config["project"] and
                  previous.get("training_signature") == self.state["training_signature"] and
                  previous.get("mode") == mode and self.config["resume"] == "auto" and
                  (self.config["entity"] is None or previous.get("entity") == self.config["entity"]))
    resume = bool(compatible and mode == "online")
    run_id = previous["run_id"] if resume else uuid.uuid4().hex[:12]
    parent = previous.get("run_id") if start_epoch > 1 else None
    group = self.config["group"] or previous.get("group") or (
        "dataset-" + self.run_config.get("dataset", {}).get("pairs_fingerprint", self.state["training_signature"])[:12])
    if parent and mode == "offline":
      self.state["resume_note"] = "offline SDK cannot resume: new linked run segment, remaining epochs only"
      LOG.info("W&B offline resume: new run segment linked to %s", parent)
    self.state.update(run_id=run_id, group=group, parent_run_id=parent)
    data = {**self.run_config, "parent_run_id": parent, "start_epoch": start_epoch}
    tags = list(dict.fromkeys(self.config["tags"] + [str(data.get("image_encoder", "evaluation")),
                          str(data.get("target_normalization", "evaluation"))]))
    name = self.config["name"] or "{}_{}_{}".format(
        data.get("image_encoder", "projection"), data.get("target_normalization", job_type), run_id)
    kwargs = {"project": self.config["project"], "entity": self.config["entity"] or
              (previous.get("entity") if resume else None), "name": name, "group": group,
              "tags": tags, "mode": mode, "id": run_id, "dir": str(self.root),
              "config": data, "settings": settings, "job_type": job_type,
              "save_code": False, "allow_val_change": True}
    if mode == "online":
      kwargs["resume"] = "must" if resume else "never"
    self.run = self.module.init(**kwargs)
    if self.run is None:
      raise TrackingUnavailable("W&B did not create a run; local training will continue.")
    signature = self.run.config.get("training_signature")
    if signature != self.state["training_signature"]:
      raise TrackingUnavailable("W&B run identity differs; refusing to mix experiments.")
    if resume and int(getattr(self.run, "step", 0)) > start_epoch:
      self.run.finish()
      self.state["resume_note"] = "remote run is ahead of checkpoint: forked to avoid duplicate epochs"
      LOG.warning("W&B run ahead of checkpoint; creating a separate linked run")
      kwargs["id"], kwargs["resume"] = uuid.uuid4().hex[:12], "never"
      kwargs["config"] = {**data, "parent_run_id": run_id}
      self.run = self.module.init(**kwargs)
      self.state.update(run_id=kwargs["id"], parent_run_id=run_id)
    self.state.update(entity=getattr(self.run, "entity", self.state["entity"]), status="active",
                      last_logged_epoch=start_epoch - 1)
    self.run.define_metric("epoch")
    for prefix in ("train/*", "val/*", "best/*", "learning_rate", "evaluation/*"):
      self.run.define_metric(prefix, step_metric="epoch")

  def log(self, metrics, epoch):
    if not self.config["enabled"]:
      return
    # The journal records attempts separately from acknowledged SDK logging.
    with (self.root / "wandb_metrics.jsonl").open("a", encoding="utf-8") as file:
      file.write(json.dumps({"epoch": epoch, "metrics": metrics}, allow_nan=False) + "\n")
    if self.state["status"] != "active":
      return
    try:
      if epoch <= self.state["last_logged_epoch"]:
        raise TrackingUnavailable("Duplicate/non-increasing W&B epoch refused.")
      self.run.log({"epoch": epoch, **metrics}, step=epoch)
      self.state["last_logged_epoch"] = epoch
    except Exception as exc:
      self._failure("logging", exc)
    self._save()

  def epoch(self, row, best_value, best_epoch):
    metrics = {"train/" + key[6:]: value for key, value in row.items() if key.startswith("train_")}
    metrics.update({"val/" + key[4:]: value for key, value in row.items() if key.startswith("val_")})
    metrics.update({"learning_rate": row["learning_rate"], "best/val_loss": best_value,
                    "best/epoch": best_epoch})
    self.log(metrics, row["epoch"])

  def snapshot(self):
    return deepcopy(self.state)

  def model_artifact(self, checkpoint):
    if not self.config["enabled"] or not self.config["log_model"] or self.state["status"] != "active":
      return
    try:
      import torch
      # Export only model tensors and public contracts, never dataset paths,
      # local config model paths, source aliases, media, RNG or optimizer state.
      file = self.root / "wandb_projection_weights.pt"
      payload = {"state_dict": checkpoint["state_dict"], "model_config": checkpoint["model_config"],
                 "image_model": public_contract(checkpoint["image_contract"]),
                 "audio_model": public_contract(checkpoint["audio_contract"]),
                 "target_normalization": checkpoint["config"]["target_normalization"],
                 "training_output_normalization": checkpoint["config"]["training_output_normalization"],
                 "loss": checkpoint["config"]["loss"]}
      with file.open("xb") as handle:
        torch.save(payload, handle)
      artifact = self.module.Artifact("projection-" + self.state["run_id"], type="model")
      artifact.add_file(str(file), name="projection.pt")
      self.run.log_artifact(artifact)
    except Exception as exc:
      self._failure("artifact", exc)
    self._save()

  def finish(self, failed=False):
    if self.run is not None:
      try:
        self.run.finish(exit_code=1 if failed or self.state["status"] == "failed" else 0)
      except Exception as exc:
        self._failure("finish", exc)
    if self.state["status"] == "active":
      self.state["status"] = "offline_saved" if self.config["mode"] == "offline" else "finished"
    self._save()


def numeric_metrics(value, prefix="evaluation"):
  result = {}
  for key, item in value.items():
    path = prefix + "/" + key
    if isinstance(item, dict):
      result.update(numeric_metrics(item, path))
    elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
      result[path] = item
    elif isinstance(item, list) and all(isinstance(v, (int, float)) for v in item):
      result.update({path + "/" + str(i + 1): v for i, v in enumerate(item) if math.isfinite(v)})
  return result


def log_results(input_dir, output_dir, config, training_dir=None):
  from .common import new_directory
  root = new_directory(output_dir)
  previous = read_json(Path(training_dir) / "wandb_status.json") if training_dir else None
  # Evaluation uses a separate linked run rather than altering finished training.
  metrics = read_json(Path(input_dir) / "metrics.json")
  run_config = {"training_signature": fingerprint(metrics), "purpose": "explicit_metrics_export",
                "parent_training_run_id": previous.get("run_id") if previous else None}
  if training_dir:
    trained = read_json(Path(training_dir) / "config.json")
    provenance = read_json(Path(training_dir) / "provenance.json")
    run_config.update({key: trained[key] for key in (
        "target_normalization", "training_output_normalization", "projection", "loss", "training")})
    run_config.update(image_encoder=provenance["image_contract"].get("name"),
                      image_model=public_contract(provenance["image_contract"]),
                      audio_model=public_contract(provenance["audio_contract"]),
                      cache_ids=provenance["cache_ids"])
  tracker = WandbTracker(root, config["wandb"], run_config, previous=previous,
                        start_epoch=1, job_type="evaluation")
  tracker.log(numeric_metrics(metrics), 1)
  tracker.finish()
  return tracker.snapshot()
