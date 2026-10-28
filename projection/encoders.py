"""Frozen, local-weights-only CLIP and official ImageBind adapters."""

from contextlib import ExitStack
import importlib.metadata
from pathlib import Path

import numpy as np

from .common import checked_device, fingerprint, sha256, unit_rows


def package_versions(names):
  result = {}
  for name in names:
    try:
      result[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      result[name] = "not_installed"
  return result


def create_encoder(name, weights, device="cpu"):
  """No hub identifiers/downloads; weights must already exist locally."""
  if name not in ("clip", "imagebind"):
    raise ValueError("image_encoder must be clip or imagebind")
  weights = Path(weights).expanduser().resolve()
  if not weights.exists():
    raise ValueError(f"Local {name} weights missing; nothing downloaded: {weights}")
  device = checked_device(device)
  if name == "clip":
    return ClipEncoder(weights, device)
  return ImageBindEncoder(weights, device)


class ClipEncoder:
  def __init__(self, directory, device):
    if not directory.is_dir():
      raise ValueError("CLIP weights must be a local Hugging Face snapshot directory")
    try:
      import torch
      from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    except ImportError as exc:
      raise RuntimeError("CLIP needs PyTorch, Pillow and transformers in the separate environment") from exc
    self.torch, self.device = torch, device
    self.model, loading = CLIPVisionModelWithProjection.from_pretrained(
        str(directory), local_files_only=True, output_loading_info=True)
    parameters = dict(self.model.named_parameters())
    if any(k in parameters for k in loading.get("missing_keys", [])) or loading.get("mismatched_keys"):
      raise ValueError("CLIP snapshot is missing pretrained vision/projection parameters")
    self.processor = CLIPImageProcessor.from_pretrained(str(directory), local_files_only=True)
    c = self.model.config
    if (c.image_size, c.patch_size, c.hidden_size, c.projection_dim) != (224, 32, 768, 512):
      raise ValueError("Expected openai/clip-vit-base-patch32 vision configuration")
    self.model.eval().requires_grad_(False).to(device)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and
                   (p.suffix in (".json", ".bin", ".safetensors")))
    self.metadata = {
        "name": "clip", "model": "openai/clip-vit-base-patch32", "dimension": 512,
        "weights": [{"file": str(p.relative_to(directory)), "sha256": sha256(p)} for p in files],
        "weights_location": str(directory),
        "preprocessing": self.processor.to_dict(), "feature_normalization": "l2",
        "implementation": "transformers.CLIPVisionModelWithProjection.image_embeds",
        "adapter_sha256": sha256(Path(__file__)),
        "versions": package_versions(("torch", "transformers", "Pillow")),
    }

  def encode(self, paths):
    from PIL import Image
    with ExitStack() as stack:
      images = [stack.enter_context(Image.open(path)).convert("RGB") for path in paths]
      pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
      with self.torch.inference_mode():
        vectors = self.model(pixel_values=pixels).image_embeds.cpu().float().numpy()
    return unit_rows(vectors)


class ImageBindEncoder:
  def __init__(self, weights, device):
    if not weights.is_file():
      raise ValueError("ImageBind weights must be a local imagebind_huge.pth")
    try:
      import torch
      from imagebind import data
      from imagebind.models import imagebind_model
    except ImportError as exc:
      raise RuntimeError("Install the official ImageBind package in the separate environment") from exc
    self.torch, self.device, self.data = torch, device, data
    self.modality = imagebind_model.ModalityType.VISION
    # pretrained=True would download. Never use it here.
    self.model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    self.model.load_state_dict(state, strict=True)
    self.model.eval().requires_grad_(False).to(device)
    self.metadata = {
        "name": "imagebind", "model": "facebookresearch/ImageBind:imagebind_huge",
        "dimension": 1024, "weights": [{"file": "imagebind_huge.pth", "sha256": sha256(weights)}],
        "weights_location": str(weights), "feature_normalization": "l2",
        "preprocessing": {"implementation": "imagebind.data.load_and_transform_vision_data",
                          "resize_short_edge": 224, "center_crop": 224,
                          "interpolation": "bicubic", "rgb": True,
                          "mean": [.48145466, .4578275, .40821073],
                          "std": [.26862954, .26130258, .27577711]},
        "source_sha256": {"model": sha256(Path(imagebind_model.__file__)),
                          "data": sha256(Path(data.__file__))},
        "adapter_sha256": sha256(Path(__file__)),
        "versions": package_versions(("torch", "torchvision", "Pillow", "timm", "imagebind")),
    }

  def encode(self, paths):
    inputs = self.data.load_and_transform_vision_data([str(p) for p in paths], self.device)
    with self.torch.inference_mode():
      vectors = self.model({self.modality: inputs})[self.modality].cpu().float().numpy()
    return unit_rows(vectors)


def encoder_identity(metadata):
  # Moving an identical local snapshot must not invalidate a checkpoint.
  return fingerprint({k: v for k, v in metadata.items() if k != "weights_location"})
