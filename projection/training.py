"""Mini-batch training on immutable caches, with exact epoch-boundary resume."""

from copy import deepcopy
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import paired_caches
from .common import (SCHEMA_VERSION, checked_device, new_directory, unit,
                     write_csv, write_json)
from .config import resolve_config
from .model import ProjectionMLP, loss_terms, normalized_output
from .tracking import WandbTracker, training_run_config

LOG = logging.getLogger(__name__)


class CachedPairs(Dataset):
  def __init__(self, image, audio, split, target_normalization):
    self.image, self.audio = image, audio
    self.pairs = [p for p in image.pairs if p.split == split]
    self.target_normalization = target_normalization
    if not self.pairs:
      raise ValueError(f"No {split} pairs; supply independent train/val/test groups")

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, index):
    pair = self.pairs[index]
    image = self.image.image(pair.pair_id)
    target = self.target(index)
    if image.shape != (self.image.manifest["contract"]["dimension"],):
      raise ValueError("Cached image dimension does not match encoder contract")
    return torch.from_numpy(image.copy()), target

  def target(self, index):
    pair = self.pairs[index]
    target = self.audio.audio(pair.pair_id)
    if self.target_normalization == "l2":
      target = unit(target)
    if target.shape != (768,):
      raise ValueError("Cached teacher dimension must be 768")
    return torch.from_numpy(target.copy())


def load_checkpoint(path):
  # Checkpoints consist of tensors and primitive containers only. No unpickled
  # modules or NumPy objects are needed for model or optimizer restoration.
  checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
  if checkpoint.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("Unsupported Projection checkpoint version")
  if checkpoint.get("output_space") != "musiccoca_audio":
    raise ValueError("Checkpoint is not a MusicCoCa Audio Projection")
  return checkpoint


def save_checkpoint(path, checkpoint):
  with Path(path).open("xb") as file:
    torch.save(checkpoint, file)


def rng_state(device):
  state = np.random.get_state()
  return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(),
          int(state[2]), int(state[3]), float(state[4])],
          "torch": torch.get_rng_state(),
          "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}


def restore_rng(state, device):
  random.setstate(state["python"])
  n = state["numpy"]
  np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
  torch.set_rng_state(state["torch"])
  if device.startswith("cuda"):
    torch.cuda.set_rng_state_all(state["cuda"])


def epoch_metrics(model, loader, config, device, optimizer=None):
  model.train(optimizer is not None)
  totals = np.zeros(4, dtype=np.float64)
  count = 0
  with torch.set_grad_enabled(optimizer is not None):
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      predicted = normalized_output(model(x), config["training_output_normalization"])
      terms = loss_terms(predicted, y, config["loss"])
      if not all(torch.isfinite(t).all().item() for t in terms):
        raise ValueError("Nonfinite training/validation loss")
      if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        terms[0].mean().backward()
        optimizer.step()
      totals += np.array([t.detach().sum().item() for t in terms])
      count += len(x)
  metrics = dict(zip(("loss", "mse", "cosine_loss", "norm_squared_error"),
                     (totals / count).tolist()))
  metrics["norm_rmse"] = float(np.sqrt(metrics["norm_squared_error"]))
  metrics["cosine_similarity"] = 1 - metrics["cosine_loss"]
  return metrics


def monitor_call(function, device, *args, **kwargs):
  """Monitoring must not consume training RNG streams."""
  state = rng_state(device)
  try:
    return function(*args, **kwargs)
  finally:
    restore_rng(state, device)


def train(image_dir, audio_dir, output_dir, config_file=None, overrides=None,
          resume=None, device="cpu"):
  checked_device(device)
  previous = load_checkpoint(resume) if resume else None
  if previous is not None and previous.get("kind") != "resume":
    raise ValueError("Resume requires last.pt or epoch_*.pt; best.pt is inference-only")
  config = resolve_config(config_file, overrides, previous["config"] if previous else None)
  if previous:
    old = resolve_config(resume_config=previous["config"])
    new = deepcopy(config)
    old["training"].pop("epochs")
    new["training"].pop("epochs")
    for key in ("wandb", "models"):
      old.pop(key)
      new.pop(key)
    if old != new:
      raise ValueError("Resume training settings changed; only epochs and monitoring/model paths can change")
    if previous["device_type"] != device.split(":")[0]:
      raise ValueError("Exact resume requires the same CPU/CUDA device type")
  image, audio = paired_caches(image_dir, audio_dir)
  if config["image_encoder"] is not None and config["image_encoder"] != image.manifest["contract"]["name"]:
    raise ValueError("Configured Image Encoder differs from image cache")
  cache_ids = {"image": image.manifest["fingerprint"], "audio": audio.manifest["fingerprint"]}
  if previous and previous["cache_ids"] != cache_ids:
    raise ValueError("Resume caches differ from checkpoint")
  train_data = CachedPairs(image, audio, "train", config["target_normalization"])
  val_data = CachedPairs(image, audio, "val", config["target_normalization"])
  settings = config["training"]
  seed = settings["seed"]
  os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
  torch.set_num_threads(settings["num_threads"])
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  model_config = {"input_dim": image.manifest["contract"]["dimension"], **config["projection"]}
  model = ProjectionMLP(**model_config).to(device)
  optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                     "sgd": torch.optim.SGD}[settings["optimizer"]]
  optimizer = optimizer_class(model.parameters(), lr=settings["learning_rate"],
                              weight_decay=settings["weight_decay"])
  mean = torch.zeros(768, dtype=torch.float64)
  for i in range(len(train_data)):
    mean += train_data.target(i).double()
  mean = (mean / len(train_data)).float()
  train_mean = normalized_output(mean, config["training_output_normalization"])
  if not torch.isfinite(train_mean).all() or train_mean.norm().item() < 1e-12:
    raise ValueError("Training teacher mean is zero/nonfinite; mean baseline is undefined")
  history, best_value, best_epoch, best_state, bad_epochs, start_epoch = [], float("inf"), 0, None, 0, 1
  if previous:
    model.load_state_dict(previous["state_dict"], strict=True)
    optimizer.load_state_dict(previous["optimizer"])
    # Optimizer moments follow the destination parameters even on CUDA.
    for state in optimizer.state.values():
      for key, value in state.items():
        if isinstance(value, torch.Tensor):
          state[key] = value.to(device)
    history = previous["history"]
    best_value, best_epoch = previous["best_val_loss"], previous["best_epoch"]
    best_state, bad_epochs = previous["best_state_dict"], previous["bad_epochs"]
    start_epoch = previous["epoch"] + 1
    restore_rng(previous["rng"], device)
  if start_epoch > settings["epochs"]:
    raise ValueError("Total --epochs must exceed the resumed epoch")
  root = new_directory(output_dir)
  (root / "checkpoints").mkdir()
  write_json(root / "config.json", config)
  write_json(root / "provenance.json", {
      "schema_version": SCHEMA_VERSION, "cache_ids": cache_ids,
      "image_contract": image.manifest["contract"], "audio_contract": audio.manifest["contract"],
      "pairs_fingerprint": image.manifest["pairs_fingerprint"],
      "synthetic": image.manifest["synthetic"] or audio.manifest["synthetic"],
      "resume_from": str(Path(resume).resolve()) if resume else None,
      "torch_version": str(torch.__version__), "numpy_version": np.__version__, "device": device,
  })
  validation_loader = DataLoader(val_data, batch_size=settings["batch_size"], shuffle=False,
                               generator=torch.Generator().manual_seed(seed))
  initial = epoch_metrics(model, validation_loader, config, device)
  write_json(root / "initial_validation.json", initial)
  tracker = monitor_call(WandbTracker, device, root, config["wandb"],
      training_run_config(config, image, audio, model_config, device),
      previous=previous.get("wandb") if previous else None, start_epoch=start_epoch)
  checkpoint = None
  local_complete = False
  try:
    for epoch in range(start_epoch, settings["epochs"] + 1):
      started = time.perf_counter()
      generator = torch.Generator().manual_seed(seed + epoch)
      loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True, generator=generator)
      training = epoch_metrics(model, loader, config, device, optimizer)
      validation = epoch_metrics(model, validation_loader, config, device)
      improved = validation["loss"] < best_value
      if improved:
        best_value, best_epoch, bad_epochs = validation["loss"], epoch, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      else:
        bad_epochs += 1
      row = {"epoch": epoch, **{"train_" + k: v for k, v in training.items()},
             **{"val_" + k: v for k, v in validation.items()},
             "learning_rate": optimizer.param_groups[0]["lr"],
             "seconds": time.perf_counter() - started, "is_best": improved}
      history.append(row)
      # Local append-only logs precede all network/monitoring operations.
      import csv
      with (root / "history.csv").open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if epoch == start_epoch:
          writer.writeheader()
        writer.writerow(row)
      monitor_call(tracker.epoch, device, row, best_value, best_epoch)
      checkpoint = {
          "schema_version": SCHEMA_VERSION, "kind": "resume", "output_space": "musiccoca_audio",
          "config": config, "model_config": model_config,
          "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
          "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history,
          "best_state_dict": best_state, "best_val_loss": best_value, "best_epoch": best_epoch,
          "bad_epochs": bad_epochs, "rng": rng_state(device), "device_type": device.split(":")[0],
          "cache_ids": cache_ids, "image_contract": image.manifest["contract"],
          "audio_contract": audio.manifest["contract"], "train_mean": train_mean.cpu(),
          "synthetic": image.manifest["synthetic"] or audio.manifest["synthetic"],
          "torch_version": str(torch.__version__), "wandb": tracker.snapshot(),
      }
      save_checkpoint(root / "checkpoints" / f"epoch_{epoch:04d}.pt", checkpoint)
      LOG.info("epoch %d train %.6f val %.6f", epoch, training["loss"], validation["loss"])
      if settings["patience"] and bad_epochs >= settings["patience"]:
        break
    save_checkpoint(root / "last.pt", checkpoint)
    best = {k: v for k, v in checkpoint.items() if k not in (
        "optimizer", "rng", "history", "best_state_dict", "bad_epochs")}
    best.update(kind="inference", state_dict=best_state, epoch=best_epoch)
    save_checkpoint(root / "best.pt", best)
    write_json(root / "history.json", history)
    local_complete = True
    monitor_call(tracker.model_artifact, device, best)
  finally:
    monitor_call(tracker.finish, device, failed=not local_complete)
  write_json(root / "summary.json", {"best_epoch": best_epoch, "best_val_loss": best_value,
      "last_epoch": checkpoint["epoch"], "status": "complete",
      "wandb_status": tracker.state["status"], "wandb_errors": len(tracker.state["errors"])})
  return root
