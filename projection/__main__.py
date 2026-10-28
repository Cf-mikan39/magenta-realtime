"""Command line with lazy imports separating Audio, image and training stages."""

import argparse
import logging
from pathlib import Path


def parser():
  p = argparse.ArgumentParser(description="Offline image → MusicCoCa Audio Projection (no downloads)")
  sub = p.add_subparsers(dest="command", required=True)
  split = sub.add_parser("split", help="Resolve leakage-safe train/val/test splits")
  images = sub.add_parser("cache-images", help="Extract frozen local CLIP/ImageBind features")
  audio = sub.add_parser("cache-audio", help="Cache raw MusicCoCa teachers in MRT2 environment")
  train = sub.add_parser("train", help="Train Projection from caches only")
  evaluate = sub.add_parser("evaluate", help="Embedding/retrieval/mean baseline evaluation")
  infer = sub.add_parser("infer", help="One image or an explicitly identified feature vector")
  rvq = sub.add_parser("rvq", help="Quantize exported inference/evaluation in MRT2 environment")
  synthetic = sub.add_parser("synthetic-cache", help="Artificial CPU wiring data; not real model features")
  benchmark = sub.add_parser("benchmark-encoder", help="Real local image model reproducibility/timing/memory")
  log_results = sub.add_parser("log-results", help="Explicit metrics-only W&B export, no media")
  for child in (split, images, audio, train, evaluate, infer, rvq, synthetic, benchmark, log_results):
    child.add_argument("--output-dir", required=True, type=Path)
  for child in (split, images, audio):
    child.add_argument("--csv", required=True, type=Path)
    child.add_argument("--seed", type=int, default=42, help="Only for assigning missing splits")
  for child in (images, train, evaluate, infer, benchmark):
    child.add_argument("--device", default="cpu", help="Explicit cuda:N requires no other GPU processes")
  images.add_argument("--image-encoder", choices=("clip", "imagebind"))
  images.add_argument("--weights", type=Path)
  images.add_argument("--batch-size", type=int, default=16)
  for child in (audio, rvq):
    child.add_argument("--resource-dir", type=Path)
  audio.add_argument("--hop-seconds", type=float, default=10.)
  audio.add_argument("--pad-end", action=argparse.BooleanOptionalAction, default=True)
  audio.add_argument("--pooling", choices=("mean", "window"), default="mean")
  audio.add_argument("--window-index", type=int, default=0)
  for child in (train, evaluate):
    child.add_argument("--image-cache", required=True, type=Path)
    child.add_argument("--audio-cache", required=True, type=Path)
  train.add_argument("--config", type=Path)
  train.add_argument("--resume", type=Path)
  train.add_argument("--image-encoder", choices=("clip", "imagebind", "synthetic"))
  train.add_argument("--target-normalization", choices=("raw", "l2"))
  train.add_argument("--training-output-normalization", choices=("none", "l2"))
  for name in ("mse", "cosine", "norm"):
    train.add_argument(f"--{name}-weight", type=float)
  train.add_argument("--hidden-dim", type=int)
  train.add_argument("--activation", choices=("relu", "gelu", "silu"))
  train.add_argument("--dropout", type=float)
  train.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=None)
  for name in ("epochs", "batch-size", "seed", "patience", "num-threads"):
    train.add_argument("--" + name, type=int)
  for name in ("learning-rate", "weight-decay"):
    train.add_argument("--" + name, type=float)
  train.add_argument("--optimizer", choices=("adam", "adamw", "sgd"))
  for child in (evaluate, infer):
    child.add_argument("--checkpoint", required=True, type=Path)
    child.add_argument("--output-normalization", choices=("as_trained", "l2"), default="as_trained")
  evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
  evaluate.add_argument("--batch-size", type=int, default=64)
  source = infer.add_mutually_exclusive_group(required=True)
  source.add_argument("--image", type=Path)
  source.add_argument("--feature-file", type=Path)
  infer.add_argument("--feature-metadata", type=Path)
  infer.add_argument("--weights", type=Path, help="Relocate identical encoder weights")
  infer.add_argument("--image-encoder", choices=("clip", "imagebind", "synthetic"))
  for child in (images, audio, infer, benchmark, log_results):
    child.add_argument("--config", type=Path, help="JSON config; model paths are relative to this file")
  for child in (train, log_results):
    child.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    child.add_argument("--wandb-mode", choices=("online", "offline"))
    for name in ("project", "entity", "name", "group"):
      child.add_argument("--wandb-" + name)
    child.add_argument("--wandb-tag", action="append")
    child.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=None)
    child.add_argument("--wandb-resume", choices=("auto", "new"))
  log_results.add_argument("--input-dir", required=True, type=Path)
  log_results.add_argument("--training-dir", type=Path)
  benchmark.add_argument("images", nargs="*", type=Path)
  benchmark.add_argument("--image-dir", type=Path)
  benchmark.add_argument("--image-encoder", choices=("clip", "imagebind"))
  benchmark.add_argument("--weights", type=Path)
  benchmark.add_argument("--repeats", type=int, default=3)
  benchmark.add_argument("--max-images", type=int, default=5)
  rvq.add_argument("--input-dir", required=True, type=Path)
  synthetic.add_argument("--samples", type=int, default=30)
  synthetic.add_argument("--input-dim", type=int, default=8)
  synthetic.add_argument("--seed", type=int, default=42)
  return p


def wandb_overrides(args):
  mapping = {"enabled": "wandb", "mode": "wandb_mode", "project": "wandb_project",
             "entity": "wandb_entity", "name": "wandb_name", "group": "wandb_group",
             "tags": "wandb_tag", "log_model": "wandb_log_model", "resume": "wandb_resume"}
  return {key: getattr(args, arg) for key, arg in mapping.items() if getattr(args, arg) is not None}


def main(argv=None):
  args = parser().parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  try:
    if args.output_dir.exists():
      raise ValueError(f"Output directory already exists; refusing to overwrite: {args.output_dir}")
    if args.command == "split":
      from .cache import prepare_pairs
      from .common import new_directory, write_json
      from .dataset import write_pairs, pairs_fingerprint
      pairs, _ = prepare_pairs(args.csv, args.seed)
      root = new_directory(args.output_dir)
      write_pairs(root / "pairs.csv", pairs)
      write_json(root / "split.json", {"seed": args.seed, "pairs_fingerprint": pairs_fingerprint(pairs),
                                      "counts": {s: sum(p.split == s for p in pairs)
                                                 for s in ("train", "val", "test")}})
    elif args.command == "cache-images":
      from .cache import cache_images
      from .encoders import create_encoder
      from .config import resolve_config, model_path
      config = resolve_config(args.config)
      name = args.image_encoder or config["image_encoder"]
      if name not in ("clip", "imagebind"):
        raise ValueError("Specify --image-encoder or image_encoder in --config")
      encoder = create_encoder(name, model_path(config, name, args.config, args.weights), args.device)
      cache_images(args.csv, args.output_dir, encoder, args.seed, args.batch_size)
    elif args.command == "cache-audio":
      from .cache import cache_audio
      from .config import resolve_config, model_path
      config = resolve_config(args.config)
      resources = model_path(config, "musiccoca", args.config, args.resource_dir) if (
          args.resource_dir is not None or config["models"]["musiccoca"] is not None) else None
      cache_audio(args.csv, args.output_dir, resources, args.seed,
                  args.hop_seconds, args.pad_end, args.pooling, args.window_index)
    elif args.command == "train":
      from .training import train
      overrides = {}
      for group, keys in ((None, ("image_encoder", "target_normalization", "training_output_normalization")),
                          ("loss", ("mse_weight", "cosine_weight", "norm_weight")),
                          ("projection", ("hidden_dim", "activation", "dropout", "layer_norm")),
                          ("training", ("epochs", "batch_size", "seed", "patience", "num_threads",
                                        "learning_rate", "weight_decay", "optimizer"))):
        values = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
        if group and values:
          overrides[group] = values
        elif group is None:
          overrides.update(values)
      tracking = wandb_overrides(args)
      if tracking:
        overrides["wandb"] = tracking
      root = train(args.image_cache, args.audio_cache, args.output_dir, args.config,
            overrides, args.resume, args.device)
      from .common import read_json
      if read_json(root / "summary.json")["wandb_status"] == "failed":
        logging.error("Training completed locally; W&B failed. See %s", root / "wandb_status.json")
        return 2
    elif args.command == "evaluate":
      from .evaluation import evaluate
      evaluate(args.checkpoint, args.image_cache, args.audio_cache, args.output_dir,
               args.split, args.output_normalization, args.device, args.batch_size)
    elif args.command == "infer":
      from .inference import infer
      from .config import resolve_config, model_path
      config = resolve_config(args.config)
      weights = args.weights
      name = args.image_encoder or config["image_encoder"]
      if args.image is not None and args.config and weights is None:
        if name is None:
          from .training import load_checkpoint
          name = load_checkpoint(args.checkpoint)["image_contract"]["name"]
        if config["models"].get(name) is not None:
          weights = model_path(config, name, args.config)
      infer(args.checkpoint, args.output_dir, args.image, args.feature_file, args.feature_metadata,
            args.output_normalization, args.device, weights, name)
    elif args.command == "rvq":
      from .rvq import quantize_results
      quantize_results(args.input_dir, args.output_dir, args.resource_dir)
    elif args.command == "synthetic-cache":
      from .synthetic import synthetic_cache
      synthetic_cache(args.output_dir, args.samples, args.input_dim, args.seed)
    elif args.command == "benchmark-encoder":
      from .benchmark import benchmark_encoder
      from .config import resolve_config, model_path
      config = resolve_config(args.config)
      name = args.image_encoder or config["image_encoder"]
      if name not in ("clip", "imagebind"):
        raise ValueError("Specify --image-encoder or image_encoder in --config")
      files = args.images
      if args.image_dir:
        files += sorted(p for p in args.image_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
      benchmark_encoder(name, model_path(config, name, args.config, args.weights), files,
                        args.output_dir, args.device, args.repeats, args.max_images)
    elif args.command == "log-results":
      from .config import resolve_config
      from .tracking import log_results as export_results
      tracking = {"enabled": True, "mode": "offline", **wandb_overrides(args)}
      config = resolve_config(args.config, {"wandb": tracking})
      state = export_results(args.input_dir, args.output_dir, config, args.training_dir)
      if state["status"] == "failed":
        return 2
  except (ValueError, OSError, RuntimeError, ImportError) as exc:
    logging.error("%s", exc)
    return 1
  logging.info("Saved: %s", args.output_dir.resolve())
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
