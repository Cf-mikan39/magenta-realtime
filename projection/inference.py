"""Web-independent prediction API returning float32[768] before RVQ."""

from pathlib import Path
import time

import numpy as np
import torch

from .common import checked_device, new_directory, save_array, sha256, unit_rows, write_json
from .encoders import create_encoder, encoder_identity
from .model import ProjectionMLP, normalized_output
from .training import load_checkpoint


class ProjectionPredictor:
  def __init__(self, checkpoint_path, device="cpu"):
    checked_device(device)
    self.checkpoint_path = Path(checkpoint_path).resolve()
    self.checkpoint = load_checkpoint(checkpoint_path)
    self.device = device
    self.model = ProjectionMLP(**self.checkpoint["model_config"])
    self.model.load_state_dict(self.checkpoint["state_dict"], strict=True)
    self.model.eval().requires_grad_(False).to(device)

  def predict_features(self, features, encoder_metadata, output_normalization="as_trained"):
    if encoder_identity(encoder_metadata) != encoder_identity(self.checkpoint["image_contract"]):
      raise ValueError("Image Encoder/weights/preprocessing differs from checkpoint")
    features = np.asarray(features)
    if features.ndim == 1:
      features = features[None, :]
    if (features.ndim != 2 or features.shape[1] != self.checkpoint["model_config"]["input_dim"] or
        features.dtype != np.float32 or not np.all(np.isfinite(features))):
      raise ValueError("Invalid float32 image features or input dimension")
    if output_normalization not in ("as_trained", "l2"):
      raise ValueError("output_normalization must be as_trained or l2 (not raw)")
    with torch.inference_mode():
      tensor = torch.from_numpy(features.copy()).to(self.device)
      trained = normalized_output(self.model(tensor),
                                  self.checkpoint["config"]["training_output_normalization"])
      result = trained.cpu().float().numpy()
    if not np.all(np.isfinite(result)):
      raise ValueError("Projection produced nonfinite values")
    if (self.checkpoint["config"]["training_output_normalization"] == "l2" and
        not np.allclose(np.linalg.norm(result, axis=-1), 1., rtol=1e-5, atol=1e-5)):
      raise ValueError("Trained L2 output collapsed to zero/near-zero")
    return unit_rows(result) if output_normalization == "l2" else result


class ImageStylePredictor(ProjectionPredictor):
  def __init__(self, checkpoint_path, device="cpu", weights=None, image_encoder=None,
               encoder=None):
    super().__init__(checkpoint_path, device)
    expected = self.checkpoint["image_contract"]
    if image_encoder is not None and image_encoder != expected["name"]:
      raise ValueError("Requested Image Encoder differs from checkpoint")
    self.encoder = encoder or create_encoder(expected["name"],
        weights or expected["weights_location"], device)
    if encoder_identity(self.encoder.metadata) != encoder_identity(expected):
      raise ValueError("Image Encoder/weights/preprocessing differs from checkpoint")

  def predict_image(self, path, output_normalization="as_trained"):
    features = self.encoder.encode([str(path)])
    return self.predict_features(features, self.encoder.metadata, output_normalization)[0]


def infer(checkpoint, output_dir, image=None, feature_file=None, feature_metadata=None,
          output_normalization="as_trained", device="cpu", weights=None, image_encoder=None):
  started = time.perf_counter()
  if image is not None:
    predictor = ImageStylePredictor(checkpoint, device, weights, image_encoder)
    embedding = predictor.predict_image(image, output_normalization)
    source = {"image": str(Path(image).resolve()), "image_sha256": sha256(Path(image))}
  else:
    from .common import read_json
    if feature_file is None or feature_metadata is None:
      raise ValueError("Feature inference requires --feature-file and --feature-metadata")
    predictor = ProjectionPredictor(checkpoint, device)
    if image_encoder is not None and image_encoder != predictor.checkpoint["image_contract"]["name"]:
      raise ValueError("Requested Image Encoder differs from checkpoint")
    embedding = predictor.predict_features(np.load(feature_file, allow_pickle=False),
        read_json(feature_metadata), output_normalization)
    if len(embedding) != 1:
      raise ValueError("Offline inference expects one image feature; use evaluate for batches")
    embedding = embedding[0]
    source = {"feature_file": str(Path(feature_file).resolve()),
              "feature_sha256": sha256(Path(feature_file))}
  seconds = time.perf_counter() - started
  root = new_directory(output_dir)
  save_array(root / "prediction.npy", embedding)
  write_json(root / "inference.json", {
      **source, "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(Path(checkpoint)),
      "image_contract": predictor.checkpoint["image_contract"],
      "audio_contract": predictor.checkpoint["audio_contract"],
      "target_normalization": predictor.checkpoint["config"]["target_normalization"],
      "training_output_normalization": predictor.checkpoint["config"]["training_output_normalization"],
      "output_normalization": output_normalization, "output_space": "musiccoca_audio",
      "norm": float(np.linalg.norm(embedding)), "shape": list(embedding.shape),
      "dtype": str(embedding.dtype), "prediction_sha256": sha256(root / "prediction.npy"),
      "seconds_including_model_load": seconds, "device": device,
      "synthetic": predictor.checkpoint["synthetic"], "rvq": "not_yet_run; use projection rvq"})
  return embedding
