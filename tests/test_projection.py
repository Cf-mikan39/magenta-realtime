"""CPU tests use artificial features; never download or load image weights."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from projection.cache import Cache, cache_images, paired_caches
from projection.common import fingerprint, read_json, sha256
from projection.config import resolve_config
from projection.dataset import Pair, assign_splits, read_pairs


class DatasetTest(unittest.TestCase):
  def test_transitive_shared_image_audio_groups_never_cross_splits(self):
    pairs = [Pair("a", "i1", "/i1", "a1", "/a1", "train"),
             Pair("b", "i1", "/i1", "a2", "/a2"),
             Pair("c", "i2", "/i2", "a2", "/a2"),
             Pair("d", "i3", "/i3", "a3", "/a3", "val"),
             Pair("e", "i4", "/i4", "a4", "/a4", "test")]
    split = assign_splits(pairs)
    self.assertEqual([p.split for p in split], ["train", "train", "train", "val", "test"])
    from dataclasses import replace
    with self.assertRaisesRegex(ValueError, "leakage"):
      assign_splits([*pairs[:2], replace(pairs[2], split="test"), *pairs[3:]])

  def test_source_groups_and_byte_identical_copies_are_protected(self):
    a = Pair("a", "i1", "/i1", "a1", "/a1", "train", audio_source_id="song")
    b = Pair("b", "i2", "/i2", "a2", "/a2", "test", audio_source_id="song")
    with self.assertRaisesRegex(ValueError, "leakage"):
      assign_splits([a, b])
    from dataclasses import replace
    b = replace(b, audio_source_id="")
    a = replace(a, audio_source_id="")
    with self.assertRaisesRegex(ValueError, "leakage"):
      assign_splits([a, b], content_keys={"/i1": "same", "/i2": "same", "/a1": "a", "/a2": "b"})

  def test_auto_split_is_reproducible_and_reserves_three_splits(self):
    pairs = [Pair(str(i), str(i), "/i" + str(i), str(i), "/a" + str(i)) for i in range(20)]
    a, b = assign_splits(pairs, 7), assign_splits(pairs, 7)
    self.assertEqual(a, b)
    self.assertEqual({p.split for p in a}, {"train", "val", "test"})
    self.assertNotEqual([p.split for p in a], [p.split for p in assign_splits(pairs, 9)])

  def test_csv_relative_paths_optional_intervals_and_duplicate_ids(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      csv_path = root / "pairs.csv"
      csv_path.write_text("image_id,image_path,audio_id,audio_path,split,start_seconds,duration_seconds,pooling,window_index\n"
                          "i,image.png,a,song.wav,train,10,20,window,1\n"
                          "i,image.png,a,song.wav,train,0,10,mean,\n")
      pairs = read_pairs(csv_path)
      self.assertEqual(pairs[0].image_path, str(root / "image.png"))
      self.assertEqual((pairs[0].start_seconds, pairs[0].duration_seconds, pairs[0].window_index), (10, 20, 1))
      self.assertEqual(len(pairs), 2)

  def test_raw_l2_and_loss_choices_are_independent(self):
    raw = resolve_config()
    l2 = resolve_config(overrides={"target_normalization": "l2"})
    self.assertEqual(raw["training_output_normalization"], "none")
    self.assertEqual(l2["training_output_normalization"], "l2")
    self.assertEqual((l2["loss"]["mse_weight"], l2["loss"]["cosine_weight"]), (0, 1))
    explicit = resolve_config(overrides={"target_normalization": "l2",
        "training_output_normalization": "none", "loss": {"mse_weight": 1., "cosine_weight": .1}})
    self.assertEqual(explicit["loss"], raw["loss"])

  def test_image_cache_deduplicates_content_and_records_aliases(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      for i in range(3):
        (root / f"im{i}").write_bytes(str(i).encode())
        (root / f"au{i}").write_bytes(("audio" + str(i)).encode())
      csv_path = root / "pairs.csv"
      csv_path.write_text("image_id,image_path,audio_id,audio_path,split\n"
                          "i0,im0,a0,au0,train\ni0,im0,a0,au0,train\n"
                          "i1,im1,a1,au1,val\ni2,im2,a2,au2,test\n")
      calls = []

      class Encoder:
        metadata = {"name": "test_encoder", "dimension": 4}
        def encode(self, paths):
          calls.extend(paths)
          return np.ones((len(paths), 4), dtype=np.float32)

      cache_images(csv_path, root / "cache", Encoder())
      cache = Cache(root / "cache", "image")
      self.assertEqual(len(calls), 3)
      ref = cache.manifest["refs"][cache.pairs[0].pair_id]["key"]
      self.assertEqual(len(cache.manifest["entries"][ref]["aliases"]), 2)
      np.testing.assert_array_equal(cache.image(cache.pairs[0].pair_id), [1] * 4)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Run PyTorch tests in the separate Projection environment")
class TrainingTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from projection.synthetic import synthetic_cache
    from projection.training import train
    cls.temp = tempfile.TemporaryDirectory()
    cls.root = Path(cls.temp.name)
    synthetic_cache(cls.root / "cache", samples=30, input_dim=8)
    cls.image, cls.audio = cls.root / "cache/images", cls.root / "cache/audio"
    cls.overrides = {"projection": {"hidden_dim": 24, "dropout": .2},
                     "training": {"epochs": 8, "batch_size": 6, "learning_rate": .01}}
    train(cls.image, cls.audio, cls.root / "raw", overrides=cls.overrides)

  @classmethod
  def tearDownClass(cls):
    cls.temp.cleanup()

  def test_cpu_training_reduces_known_synthetic_training_loss(self):
    history = read_json(self.root / "raw/history.json")
    self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
    self.assertTrue((self.root / "raw/best.pt").is_file())
    self.assertTrue((self.root / "raw/last.pt").is_file())
    self.assertTrue(read_json(self.root / "raw/provenance.json")["synthetic"])

  def test_resume_matches_uninterrupted_training_and_preserves_source(self):
    import torch
    from projection.training import train, load_checkpoint
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      first = {**self.overrides, "training": {**self.overrides["training"], "epochs": 3}}
      train(self.image, self.audio, root / "first", overrides=first)
      original = sha256(root / "first/last.pt")
      train(self.image, self.audio, root / "resume", resume=root / "first/last.pt",
            overrides={"training": {"epochs": 8}})
      full = load_checkpoint(self.root / "raw/last.pt")
      resumed = load_checkpoint(root / "resume/last.pt")
      for key in full["state_dict"]:
        self.assertTrue(torch.equal(full["state_dict"][key], resumed["state_dict"][key]), key)
      self.assertEqual([r["val_loss"] for r in full["history"]],
                       [r["val_loss"] for r in resumed["history"]])
      self.assertEqual(sha256(root / "first/last.pt"), original)

  def test_l2_target_output_and_explicit_inference_policy(self):
    from projection.training import train, load_checkpoint
    from projection.inference import ProjectionPredictor
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      train(self.image, self.audio, root / "l2", overrides={
          "target_normalization": "l2", "projection": {"hidden_dim": 8, "dropout": 0.},
          "training": {"epochs": 2}})
      cp = load_checkpoint(root / "l2/best.pt")
      predictor = ProjectionPredictor(root / "l2/best.pt")
      features = np.load(self.root / "cache/example_feature.npy", allow_pickle=False)
      result = predictor.predict_features(features, cp["image_contract"])
      np.testing.assert_allclose(np.linalg.norm(result, axis=-1), [1], atol=1e-6)
      self.assertEqual(cp["config"]["loss"]["mse_weight"], 0)
      with self.assertRaisesRegex(ValueError, "as_trained"):
        predictor.predict_features(features, cp["image_contract"], "raw")

  def test_prediction_reload_metadata_and_wrong_encoder_rejection(self):
    from projection.inference import ProjectionPredictor, ImageStylePredictor, infer
    cp = self.root / "raw/best.pt"
    features = np.load(self.root / "cache/example_feature.npy", allow_pickle=False)
    metadata = read_json(self.root / "cache/example_encoder.json")
    predictor = ProjectionPredictor(cp)
    a = predictor.predict_features(features, metadata)
    b = ProjectionPredictor(cp).predict_features(features, metadata)
    np.testing.assert_array_equal(a, b)
    with self.assertRaisesRegex(ValueError, "differs"):
      predictor.predict_features(features, {**metadata, "preprocessing": "different"})
    with self.assertRaisesRegex(ValueError, "differs"):
      ImageStylePredictor(cp, image_encoder="imagebind")
    with tempfile.TemporaryDirectory() as temp:
      output = Path(temp) / "infer"
      result = infer(cp, output, feature_file=self.root / "cache/example_feature.npy",
                     feature_metadata=self.root / "cache/example_encoder.json")
      np.testing.assert_array_equal(result, a[0])
      self.assertEqual(read_json(output / "inference.json")["output_space"], "musiccoca_audio")

  def test_evaluation_baseline_retrieval_and_same_l2_rvq_inputs(self):
    from projection.evaluation import evaluate, retrieval
    from projection.rvq import quantize_results
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      metrics = evaluate(self.root / "raw/best.pt", self.image, self.audio,
                         root / "eval", output_normalization="l2")
      self.assertEqual(metrics["projection"]["count"], 6)
      self.assertEqual(metrics["train_mean_baseline"]["output_mean_variance"], 0)
      for file in ("predictions.npy", "teachers.npy", "baseline.npy"):
        array = np.load(root / "eval" / file, allow_pickle=False)
        np.testing.assert_allclose(np.linalg.norm(array, axis=-1), np.ones(6), atol=1e-6)

      class Quantizer:
        def tokenize(self, values):
          return ((values[:, :1] * 10).astype(np.int32) + np.arange(12)[None, :]) % 1024

      rvq = quantize_results(root / "eval", root / "rvq", model=Quantizer())
      self.assertEqual(len(rvq["predicted"]["per_level_match_rate"]), 12)
      self.assertNotIn("numeric_token_distance", rvq["predicted"])
    scores = retrieval(np.array([[1., 0.], [1., 0.]], dtype=np.float32),
                       np.array([[0., 1.], [1., 0.]], dtype=np.float32),
                       ["same_image", "same_image"], ["audio_a", "audio_b"])
    self.assertEqual(scores["recall_at_1"], 1)
    self.assertEqual(scores["unique_images"], 1)

  def test_incompatible_cache_and_existing_run_are_rejected(self):
    from projection.synthetic import synthetic_cache
    from projection.training import train
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      synthetic_cache(root / "other", samples=30, seed=99)
      with self.assertRaisesRegex(ValueError, "different pairs"):
        paired_caches(self.image, root / "other/audio")
    marker = sha256(self.root / "raw/best.pt")
    with self.assertRaises(FileExistsError):
      train(self.image, self.audio, self.root / "raw", overrides=self.overrides)
    self.assertEqual(sha256(self.root / "raw/best.pt"), marker)

  def test_cache_tampering_is_detected(self):
    from projection.synthetic import synthetic_cache
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      synthetic_cache(root / "cache", samples=6)
      cache = Cache(root / "cache/images", "image")
      entry = next(iter(cache.manifest["entries"].values()))
      with (cache.root / entry["file"]).open("ab") as file:
        file.write(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum"):
        cache.image(cache.pairs[0].pair_id)


@unittest.skipUnless(TORCH_AVAILABLE, "Run adapter tests in separate PyTorch environment")
class EncoderAdapterTest(unittest.TestCase):
  def test_clip_local_only_frozen_and_normalized(self):
    import torch
    from PIL import Image
    from projection.encoders import create_encoder
    calls = []

    class Vision(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(512))
        self.config = SimpleNamespace(image_size=224, patch_size=32, hidden_size=768, projection_dim=512)
      @classmethod
      def from_pretrained(cls, path, **kwargs):
        calls.append(kwargs)
        return cls(), {"missing_keys": [], "mismatched_keys": []}
      def forward(self, pixel_values):
        return SimpleNamespace(image_embeds=self.weight[None, :].repeat(len(pixel_values), 1))

    class Processor:
      @classmethod
      def from_pretrained(cls, path, **kwargs):
        calls.append(kwargs)
        return cls()
      def to_dict(self):
        return {"test_fixture": True}
      def __call__(self, images, **kwargs):
        return {"pixel_values": torch.ones((len(images), 3, 2, 2))}

    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      (root / "config.json").write_text("{}")
      Image.new("RGB", (2, 2)).save(root / "image.png")
      module = SimpleNamespace(CLIPImageProcessor=Processor, CLIPVisionModelWithProjection=Vision)
      with mock.patch.dict("sys.modules", {"transformers": module}):
        encoder = create_encoder("clip", root)
        result = encoder.encode([root / "image.png"])
      self.assertEqual(result.shape, (1, 512))
      np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1], atol=1e-6)
      self.assertFalse(encoder.model.training)
      self.assertFalse(next(encoder.model.parameters()).requires_grad)
      self.assertTrue(all(c["local_files_only"] for c in calls))
      self.assertTrue(calls[0]["output_loading_info"])

  def test_imagebind_never_requests_pretrained_download(self):
    import torch
    from projection.encoders import create_encoder
    calls = []

    class Vision(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1024))
      def forward(self, inputs):
        return {"vision": self.weight[None, :].repeat(len(inputs["vision"]), 1)}

    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      model_file, data_file = root / "model.py", root / "data.py"
      model_file.write_text("# fixture")
      data_file.write_text("# fixture")
      weights = root / "weights.pth"
      torch.save(Vision().state_dict(), weights)
      def make(pretrained):
        calls.append(pretrained)
        return Vision()
      model = SimpleNamespace(__file__=str(model_file), imagebind_huge=make,
                              ModalityType=SimpleNamespace(VISION="vision"))
      data = SimpleNamespace(__file__=str(data_file),
                             load_and_transform_vision_data=lambda paths, device: torch.ones((len(paths), 3)))
      modules = {"imagebind": SimpleNamespace(data=data),
                 "imagebind.models": SimpleNamespace(imagebind_model=model)}
      with mock.patch.dict("sys.modules", modules):
        encoder = create_encoder("imagebind", weights)
        result = encoder.encode(["test_fixture_not_an_image"])
      self.assertEqual(calls, [False])
      self.assertEqual(result.shape, (1, 1024))
      self.assertFalse(next(encoder.model.parameters()).requires_grad)


if __name__ == "__main__":
  unittest.main()
