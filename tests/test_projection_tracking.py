"""W&B protocol tests without an account, credentials or network."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from projection.config import resolve_config, model_path
from projection.tracking import WandbTracker, numeric_metrics


class FakeRun:
  def __init__(self, module, kwargs, step=0):
    self.module = module
    self.config = deepcopy(kwargs["config"])
    self.id, self.entity = kwargs["id"], "mock_entity"
    self.step = step
    self.events, self.definitions, self.artifacts = [], [], []
  def define_metric(self, *args, **kwargs):
    self.definitions.append((args, kwargs))
  def log(self, values, step):
    if self.module.fail_log:
      raise ConnectionError("mock network failure")
    self.events.append((deepcopy(values), step))
    self.step = step + 1
    # Deliberately consume random streams: integration must isolate them.
    import random
    import numpy as np
    random.random()
    np.random.random()
  def log_artifact(self, artifact):
    self.artifacts.append(artifact)
  def finish(self, **kwargs):
    self.module.run = None


class FakeWandb:
  def __init__(self):
    self.run = None
    self.calls, self.runs = [], []
    self.fail_log = False
    self.remote_step = None
  def Settings(self, **kwargs):
    return kwargs
  def init(self, **kwargs):
    self.calls.append(deepcopy(kwargs))
    previous = next((r for r in reversed(self.runs) if r.id == kwargs["id"]), None)
    step = previous.step if previous else 0
    if self.remote_step is not None and kwargs.get("resume") == "must":
      step = self.remote_step
    self.run = FakeRun(self, kwargs, step)
    self.runs.append(self.run)
    return self.run
  class Artifact:
    def __init__(self, name, type):
      self.files = []
    def add_file(self, path, name):
      self.files.append((path, name))
  def importer(self, name):
    if name == "wandb":
      return self
    if name == "wandb.sdk.lib.apikey":
      return SimpleNamespace(api_key=lambda **kwargs: True)
    raise ImportError(name)


class TrackerTest(unittest.TestCase):
  def test_disabled_does_not_import_sdk_or_require_login(self):
    with tempfile.TemporaryDirectory() as temp, \
         mock.patch("projection.tracking.import_module", side_effect=AssertionError("must not import")):
      tracker = WandbTracker(temp, resolve_config()["wandb"], {"training_signature": "test"})
      tracker.finish()
      self.assertEqual(tracker.snapshot()["status"], "disabled")

  def test_epoch_step_definitions_and_privacy_settings(self):
    sdk = FakeWandb()
    config = resolve_config(overrides={"wandb": {"enabled": True, "mode": "offline"}})
    with tempfile.TemporaryDirectory() as temp, \
         mock.patch("projection.tracking.import_module", side_effect=sdk.importer):
      tracker = WandbTracker(temp, config["wandb"], {"training_signature": "test",
          "image_encoder": "clip", "target_normalization": "raw"})
      for epoch in (1, 2):
        tracker.epoch({"epoch": epoch, "learning_rate": .01, "train_loss": .5,
                       "val_loss": .7, "train_norm_rmse": .2}, .7, 1)
      tracker.finish()
      self.assertEqual([step for values, step in sdk.runs[0].events], [1, 2])
      self.assertTrue(all(values["epoch"] == step for values, step in sdk.runs[0].events))
      self.assertTrue(sdk.calls[0]["settings"]["disable_git"])
      self.assertTrue(sdk.calls[0]["settings"]["x_disable_meta"])
      self.assertFalse(sdk.calls[0]["save_code"])
      self.assertEqual(tracker.snapshot()["status"], "offline_saved")
      self.assertTrue(any(k[1].get("step_metric") == "epoch" for k in sdk.runs[0].definitions))

  def test_offline_resume_links_new_segment_without_duplicate_epochs(self):
    sdk = FakeWandb()
    config = resolve_config(overrides={"wandb": {"enabled": True, "mode": "offline"}})["wandb"]
    with tempfile.TemporaryDirectory() as temp, \
         mock.patch("projection.tracking.import_module", side_effect=sdk.importer):
      a, b = Path(temp) / "a", Path(temp) / "b"
      a.mkdir(); b.mkdir()
      first = WandbTracker(a, config, {"training_signature": "test"})
      first.log({"train/loss": 1}, 1)
      prior = first.snapshot()
      first.finish()
      resumed = WandbTracker(b, config, {"training_signature": "test"}, prior, start_epoch=2)
      resumed.log({"train/loss": .5}, 2)
      resumed.finish()
      self.assertNotEqual(prior["run_id"], resumed.state["run_id"])
      self.assertEqual(resumed.state["parent_run_id"], prior["run_id"])
      self.assertEqual(resumed.state["group"], prior["group"])
      self.assertNotIn("resume", sdk.calls[1])

  def test_online_resume_uses_same_run_and_forks_if_remote_is_ahead(self):
    sdk = FakeWandb()
    config = resolve_config(overrides={"wandb": {"enabled": True, "mode": "online"}})["wandb"]
    with tempfile.TemporaryDirectory() as temp, \
         mock.patch("projection.tracking.import_module", side_effect=sdk.importer):
      directories = [Path(temp) / str(i) for i in range(3)]
      for d in directories: d.mkdir()
      first = WandbTracker(directories[0], config, {"training_signature": "test"})
      first.log({"train/loss": 1}, 1)
      prior = first.snapshot(); first.finish()
      second = WandbTracker(directories[1], config, {"training_signature": "test"}, prior, 2)
      second.log({"train/loss": .5}, 2)
      self.assertEqual(second.state["run_id"], prior["run_id"])
      self.assertEqual(sdk.calls[-1]["resume"], "must")
      second.finish()
      sdk.remote_step = 10
      third = WandbTracker(directories[2], config, {"training_signature": "test"}, prior, 2)
      self.assertNotEqual(third.state["run_id"], prior["run_id"])
      self.assertIn("ahead", third.state["resume_note"])
      third.finish()

  def test_missing_installation_and_login_have_explicit_local_status(self):
    config = resolve_config(overrides={"wandb": {"enabled": True}})["wandb"]
    with tempfile.TemporaryDirectory() as temp, \
         mock.patch("projection.tracking.import_module", side_effect=ImportError), \
         self.assertLogs("projection.tracking", level="ERROR"):
      tracker = WandbTracker(temp, config, {"training_signature": "test"})
      self.assertEqual(tracker.state["status"], "failed")
      self.assertIn("not installed", tracker.state["errors"][0]["message"])
    sdk = FakeWandb()
    def importer(name):
      return sdk if name == "wandb" else SimpleNamespace(api_key=lambda **kwargs: None)
    with tempfile.TemporaryDirectory() as temp, \
         mock.patch.dict("os.environ", {"WANDB_API_KEY": ""}), \
         mock.patch("projection.tracking.import_module", side_effect=importer), \
         self.assertLogs("projection.tracking", level="ERROR"):
      tracker = WandbTracker(temp, config, {"training_signature": "test"})
      self.assertIn("login required", tracker.state["errors"][0]["message"])

  def test_model_paths_relative_to_json_and_metrics_export_is_numeric_only(self):
    config = resolve_config(overrides={"models": {"imagebind": "../weights/model.pth"}})
    self.assertEqual(model_path(config, "imagebind", "/tmp/configs/train.json"),
                     Path("/tmp/weights/model.pth"))
    metrics = numeric_metrics({"cosine": .5, "nested": {"recall": 1},
                               "image_path": "/private/image.jpg", "warning": "text"})
    self.assertEqual(metrics, {"evaluation/cosine": .5, "evaluation/nested/recall": 1})
    with self.assertRaises(ValueError):
      resolve_config(overrides={"wandb": {"api_key": "forbidden_config_field"}})


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires separate Torch environment")
class TrackingTrainingTest(unittest.TestCase):
  def setUp(self):
    from projection.synthetic import synthetic_cache
    self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    synthetic_cache(self.root / "cache", samples=9, input_dim=4)
    self.image, self.audio = self.root / "cache/images", self.root / "cache/audio"
    self.base = {"projection": {"hidden_dim": 8, "dropout": .2},
                 "training": {"epochs": 2, "batch_size": 3, "learning_rate": .005}}

  def test_raw_l2_configs_logging_do_not_change_training_or_leak_paths(self):
    import torch
    from projection.training import train, load_checkpoint
    sdk = FakeWandb()
    train(self.image, self.audio, self.root / "disabled", overrides=self.base)
    with mock.patch("projection.tracking.import_module", side_effect=sdk.importer):
      train(self.image, self.audio, self.root / "enabled", overrides={**self.base,
          "models": {"imagebind": "/private/model/location"},
          "wandb": {"enabled": True, "mode": "offline"}})
      train(self.image, self.audio, self.root / "l2", overrides={**self.base,
          "target_normalization": "l2", "wandb": {"enabled": True, "mode": "offline"}})
    a, b = (load_checkpoint(self.root / path / "last.pt") for path in ("disabled", "enabled"))
    for key in a["state_dict"]:
      self.assertTrue(torch.equal(a["state_dict"][key], b["state_dict"][key]))
    self.assertEqual([r["val_loss"] for r in a["history"]], [r["val_loss"] for r in b["history"]])
    self.assertNotIn("/private", json.dumps(sdk.calls[0]["config"]))
    self.assertNotIn(str(self.root), json.dumps(sdk.calls[0]["config"]))
    self.assertEqual([c["config"]["target_normalization"] for c in sdk.calls], ["raw", "l2"])
    self.assertEqual(sdk.runs[0].events[0][0]["val/loss"], a["history"][0]["val_loss"])
    self.assertEqual(sdk.runs[0].artifacts, [])

  def test_logging_failure_keeps_local_checkpoints_and_error_status(self):
    from projection.training import train
    sdk = FakeWandb(); sdk.fail_log = True
    with mock.patch("projection.tracking.import_module", side_effect=sdk.importer), \
         self.assertLogs("projection.tracking", level="ERROR"):
      train(self.image, self.audio, self.root / "failed", overrides={**self.base,
          "wandb": {"enabled": True, "mode": "offline"}})
    self.assertTrue((self.root / "failed/best.pt").exists())
    self.assertTrue((self.root / "failed/last.pt").exists())
    summary = json.loads((self.root / "failed/summary.json").read_text())
    self.assertEqual(summary["status"], "complete")
    self.assertEqual(summary["wandb_status"], "failed")
    self.assertEqual(len((self.root / "failed/wandb_metrics.jsonl").read_text().splitlines()), 2)

  def test_optional_artifact_contains_no_source_paths_or_media(self):
    import torch
    from projection.training import train
    sdk = FakeWandb()
    with mock.patch("projection.tracking.import_module", side_effect=sdk.importer):
      train(self.image, self.audio, self.root / "artifacts", overrides={**self.base,
          "wandb": {"enabled": True, "mode": "offline", "log_model": True}})
    artifact = sdk.runs[0].artifacts[0]
    self.assertEqual(len(artifact.files), 1)
    payload = torch.load(artifact.files[0][0], weights_only=True)
    self.assertNotIn("optimizer", payload)
    self.assertNotIn("models", payload)
    self.assertNotIn("weights_location", payload["image_model"])


if __name__ == "__main__":
  unittest.main()
