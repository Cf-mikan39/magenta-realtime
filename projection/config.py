"""JSON config and explicit CLI override resolution."""

from copy import deepcopy
import math

from .common import read_json

DEFAULT = {
    "image_encoder": None,
    "models": {"clip": None, "imagebind": None, "musiccoca": None},
    "wandb": {"enabled": False, "project": "mrt2-image-audio-projection",
              "entity": None, "name": None, "group": None, "tags": [],
              "mode": "online", "log_model": False, "resume": "auto"},
    "target_normalization": "raw", "training_output_normalization": None,
    "loss": {"mse_weight": None, "cosine_weight": None, "norm_weight": 0.},
    "projection": {"hidden_dim": 1024, "activation": "gelu", "dropout": .1,
                   "layer_norm": False, "output_dim": 768},
    "training": {"learning_rate": .0001, "batch_size": 64, "epochs": 50,
                 "seed": 42, "optimizer": "adamw", "weight_decay": .0001,
                 "patience": 0, "num_threads": 1},
}


def merge(base, other):
  for key, value in other.items():
    if key not in base:
      raise ValueError(f"Unknown config key: {key}")
    if isinstance(base[key], dict):
      if not isinstance(value, dict):
        raise ValueError(f"Config {key} must be an object")
      merge(base[key], value)
    else:
      base[key] = value
  return base


def resolve_config(path=None, overrides=None, resume_config=None):
  # Schema-1 checkpoints created before tracking/models were added remain usable.
  config = deepcopy(DEFAULT)
  if resume_config is not None:
    merge(config, resume_config)
  if path:
    merge(config, read_json(path))
  if overrides:
    merge(config, overrides)
  l2 = config["target_normalization"] == "l2"
  if config["image_encoder"] not in (None, "clip", "imagebind", "synthetic"):
    raise ValueError("Invalid image_encoder")
  for value in config["models"].values():
    if value is not None and (not isinstance(value, str) or not value.strip()):
      raise ValueError("Model paths must be nonempty strings or null")
  tracking = config["wandb"]
  if (not isinstance(tracking["enabled"], bool) or not isinstance(tracking["log_model"], bool) or
      tracking["mode"] not in ("online", "offline") or tracking["resume"] not in ("auto", "new") or
      not isinstance(tracking["project"], str) or not tracking["project"].strip() or
      any(value is not None and (not isinstance(value, str) or not value.strip())
          for value in (tracking["entity"], tracking["name"], tracking["group"])) or
      not isinstance(tracking["tags"], list) or
      any(not isinstance(tag, str) or not tag.strip() for tag in tracking["tags"])):
    raise ValueError("Invalid wandb configuration")
  if config["target_normalization"] not in ("raw", "l2"):
    raise ValueError("target_normalization must be raw or l2")
  if config["training_output_normalization"] is None:
    config["training_output_normalization"] = "l2" if l2 else "none"
  if config["training_output_normalization"] not in ("none", "l2"):
    raise ValueError("Invalid training_output_normalization")
  for name, default in (("mse_weight", 0. if l2 else 1.), ("cosine_weight", 1. if l2 else .1)):
    if config["loss"][name] is None:
      config["loss"][name] = default
  if (any(not math.isfinite(v) or v < 0 for v in config["loss"].values()) or
      sum(config["loss"].values()) == 0):
    raise ValueError("Loss weights must be finite, nonnegative and not all zero")
  train, model = config["training"], config["projection"]
  if (any(not isinstance(train[k], int) or train[k] < 1
          for k in ("batch_size", "epochs", "num_threads")) or
      not isinstance(train["seed"], int) or not 0 <= train["seed"] < 2**32 or
      not isinstance(train["patience"], int) or train["patience"] < 0 or
      train["optimizer"] not in ("adam", "adamw", "sgd") or
      not math.isfinite(train["learning_rate"]) or train["learning_rate"] <= 0 or
      not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0):
    raise ValueError("Invalid training configuration")
  if (not isinstance(model["hidden_dim"], int) or model["hidden_dim"] < 1 or
      model["output_dim"] != 768 or model["activation"] not in ("relu", "gelu", "silu") or
      not math.isfinite(model["dropout"]) or not 0 <= model["dropout"] < 1 or
      not isinstance(model["layer_norm"], bool)):
    raise ValueError("Invalid projection configuration")
  return config


def model_path(config, name, config_file=None, override=None):
  """Config paths are relative to their JSON file; CLI paths to the cwd."""
  from pathlib import Path
  value = override if override is not None else config["models"][name]
  if value is None:
    raise ValueError(f"Specify --weights/--resource-dir or models.{name} in --config")
  path = Path(value).expanduser()
  if not path.is_absolute() and override is None and config_file:
    path = Path(config_file).resolve().parent / path
  return path.resolve()
