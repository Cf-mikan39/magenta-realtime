"""Baseline protocol/serialization tests, without model loading or GPU use."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from scripts import musiccoca_audio_baseline as baseline


class SmallStyleModel:
  """Known unequal-norm window vectors, not a scientific MusicCoCa substitute."""

  def __init__(self):
    self.config = SimpleNamespace(embedding_dim=3, sample_rate=16000,
                                  clip_length=10., clip_length_samples=160000,
                                  rvq_depth=12, rvq_codebook_size=1024)
    self.calls = []

  def embed(self, value, **kwargs):
    self.calls.append(kwargs)
    if isinstance(value, str):
      return np.array([0., 0., 1.], dtype=np.float32)
    return np.array([[3., 0., 0.], [0., 4., 0.]], dtype=np.float32)

  def tokenize(self, vector):
    scalar = int(np.dot(vector, [11, 7, 3]) * 100)
    return ((scalar + np.arange(12)) % 1024).astype(np.int32)


class AudioBaselineTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.audio = self.root / "a song.wav"
    sf.write(self.audio, np.zeros((16000 * 12, 2), dtype=np.float32), 16000)
    self.output = self.root / "results"

  def args(self, *extra):
    return baseline.make_parser().parse_args([
        str(self.audio), "--output-dir", str(self.output), *extra])

  def test_raw_teacher_pooling_norm_comparisons_and_mix_are_preserved(self):
    model = SmallStyleModel()
    results = baseline.run_experiment(
        self.args("--prompt", "piano", "--final-normalization", "both"), model=model)
    self.assertEqual(results.errors, [])
    rows = {r["condition_id"]: r for r in results.conditions}
    audio_id = results.sources[0]["sample_id"]
    pooled = rows[f"{audio_id}.pooled.raw"]
    teacher = np.load(self.output / pooled["embedding_file"], allow_pickle=False)
    # Mean(raw windows), not mean(unit windows) or unit(mean(raw windows)).
    np.testing.assert_array_equal(teacher, [1.5, 2., 0.])
    self.assertEqual(teacher.dtype, np.float32)
    self.assertAlmostEqual(pooled["norm"], 2.5)
    self.assertEqual(pooled["generation_tokens_before_offset"][6:], [-1] * 6)
    for kind, expected in (("raw", [.75, 1., .5]), ("l2", [.3, .4, .5])):
      row = rows[f"{audio_id}.pooled.mix_t001.r00.audio_{kind}.final_none"]
      vector = np.load(self.output / row["embedding_file"], allow_pickle=False)
      np.testing.assert_allclose(vector, expected)
      self.assertEqual(row["tokens"], model.tokenize(vector).tolist())
    final = rows[f"{audio_id}.pooled.mix_t001.r00.audio_raw.final_l2"]
    self.assertAlmostEqual(final["norm"], 1, places=6)
    self.assertTrue(any(r["comparison"] == "ratio_vs_first_ratio" for r in results.comparisons))
    self.assertEqual(model.calls[0], {"use_mapper": True, "seed": 0})
    self.assertFalse(model.calls[1]["pool_across_time"])
    self.assertEqual(model.calls[1]["mono_strategy"], "average")
    manifest = json.loads((self.output / "manifest.json").read_text())
    self.assertEqual(manifest["status"], "complete")
    self.assertEqual(manifest["generated_wavs"], 0)
    with (self.output / "dimension_stats.csv").open() as f:
      stats = list(csv.DictReader(f))
    dim0 = next(r for r in stats if r["group"] == "windows_raw" and r["dimension"] == "0")
    self.assertEqual(float(dim0["mean"]), 1.5)
    self.assertEqual(float(dim0["std_ddof_0"]), 1.5)
    windows = [r for r in results.conditions if r["scope"] == "window" and
               r["audio_normalization"] == "raw"]
    self.assertEqual([r["padded_seconds"] for r in windows], [0., 8.])

  def test_pairwise_cosine_is_keyed_to_source_order_and_saved(self):
    other = self.root / "b.wav"
    sf.write(other, np.zeros(16000 * 12, dtype=np.float32), 16000)
    args = self.args("--no-text")
    args.audio_paths.append(other)
    results = baseline.run_experiment(args, model=SmallStyleModel())
    matrix = np.load(self.output / "pooled_cosine.npy", allow_pickle=False)
    np.testing.assert_allclose(matrix, np.ones((2, 2)))
    with (self.output / "pooled_cosine.csv").open() as f:
      row = next(csv.DictReader(f))
    self.assertEqual(row["sample_a"], results.sources[0]["sample_id"])
    self.assertEqual(row["sample_b"], results.sources[1]["sample_id"])

  def test_token_positions_and_active_prefix_are_not_conflated(self):
    a = np.arange(12)
    b = a.copy()
    b[6:] += 20
    diff = baseline.token_difference(a, b)
    self.assertEqual(diff["match_rate_12"], .5)
    self.assertEqual(diff["match_rate_first_6"], 1.)
    self.assertEqual(diff["changed_positions_1based"], [7, 8, 9, 10, 11, 12])
    results = baseline.Results(self.root, SmallStyleModel(), 12)
    results.compare({"condition_id": "a", "tokens": a.tolist()},
                    {"condition_id": "b", "tokens": b.tolist()}, "test")
    self.assertIsNone(results.comparisons[0]["match_rate_active_levels"])

  def test_zero_norm_and_nonfinite_normalization_are_rejected(self):
    for vector in ([0, 0], [np.nan, 1], [np.inf, 1]):
      with self.assertRaises(ValueError):
        baseline.unit(vector)
    self.assertIsNone(baseline.cosine([0, 0], [1, 0]))

  def test_file_list_relative_paths_deduplication_and_limit(self):
    listing = self.root / "files.txt"
    listing.write_text("# comment\n\na song.wav\n", encoding="utf-8")
    args = self.args("--file-list", str(listing), "--audio-dir", str(self.root))
    self.assertEqual(baseline.collect_inputs(args), [self.audio.resolve()])

  def test_no_input_and_missing_files_do_not_create_results(self):
    for paths in ([], [self.root / "missing.wav"]):
      args = self.args()
      args.audio_paths = paths
      with self.assertRaises(ValueError):
        baseline.run_experiment(args, model=SmallStyleModel())
      self.assertFalse(self.output.exists())

  def test_existing_results_are_never_reused_even_when_empty(self):
    self.output.mkdir()
    marker = self.output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
      baseline.run_experiment(self.args(), model=SmallStyleModel())
    self.assertEqual(marker.read_text(), "keep")

  def test_invalid_ratios_and_timing_are_rejected(self):
    for value in ("0:0", "-1:1", "nan:1", "inf:2", "1", "1:2:3"):
      with self.assertRaises(argparse.ArgumentTypeError):
        baseline.parse_ratio(value)
    for flag, value in (("--hop-seconds", "0"), ("--hop-seconds", "nan"),
                        ("--duration-seconds", "0"), ("--start-seconds", "-1")):
      with self.assertRaises(ValueError):
        baseline.validate_args(self.args(flag, value))

  def test_failed_audio_is_recorded_and_does_not_claim_success(self):
    self.audio.write_bytes(b"invalid audio file")
    with self.assertLogs(baseline.LOG, level="ERROR"):
      with self.assertRaisesRegex(ValueError, "No audio file"):
        baseline.run_experiment(self.args("--no-text"), model=SmallStyleModel())
    manifest = json.loads((self.output / "manifest.json").read_text())
    self.assertEqual(manifest["status"], "failed")
    self.assertEqual(manifest["successful_audio_files"], 0)
    self.assertTrue(json.loads((self.output / "errors.json").read_text()))

  def test_generation_failure_retains_successful_embedding_results(self):
    with mock.patch.object(baseline, "generate_audio", side_effect=RuntimeError("GPU busy")), \
         self.assertLogs(baseline.LOG, level="ERROR"):
      results = baseline.run_experiment(self.args("--no-text", "--generate"),
                                        model=SmallStyleModel())
    manifest = json.loads((self.output / "manifest.json").read_text())
    self.assertEqual(manifest["status"], "partial")
    self.assertEqual(manifest["successful_audio_files"], 1)
    self.assertEqual(manifest["generation"], "failed_or_not_started")
    self.assertTrue(results.conditions)
    self.assertTrue((self.output / "conditions.csv").is_file())
    self.assertEqual(results.errors[0]["stage"], "generation")

  def test_partial_file_failure_is_not_counted_as_fully_successful(self):
    with mock.patch.object(baseline, "analyze_mixes", side_effect=ValueError("invalid mix")), \
         self.assertLogs(baseline.LOG, level="ERROR"):
      results = baseline.run_experiment(self.args("--prompt", "piano"), model=SmallStyleModel())
    manifest = json.loads((self.output / "manifest.json").read_text())
    self.assertEqual(manifest["status"], "partial")
    self.assertEqual(manifest["encoded_audio_files"], 1)
    self.assertEqual(manifest["successful_audio_files"], 0)
    self.assertEqual(results.sources[0]["analysis_status"], "partial")

  @mock.patch.object(baseline.subprocess, "run")
  def test_generation_refuses_busy_or_unverifiable_gpu(self, run):
    run.return_value = SimpleNamespace(stdout="123, server, 12000 MiB\n")
    with self.assertRaisesRegex(RuntimeError, "GPU processes are active"):
      baseline.require_idle_gpu()
    run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 10)
    with self.assertRaisesRegex(RuntimeError, "Cannot confirm"):
      baseline.require_idle_gpu()

  def test_optional_generation_resets_state_and_uses_recorded_embeddings(self):
    import magenta_rt
    from magenta_rt.audio import Waveform
    calls = []
    model = SmallStyleModel()
    checkpoint = self.root / "existing.safetensors"
    checkpoint.write_bytes(b"unit-test placeholder, never loaded")

    class FakeMrt:
      def __init__(self, **kwargs):
        self.kwargs = kwargs

      def tokenize_style(self, vector):
        return model.tokenize(vector)

      def generate(self, **kwargs):
        calls.append(kwargs)
        return Waveform(np.zeros((kwargs["frames"] * 1920, 2), dtype=np.float32), 48000), None

    with mock.patch.dict(magenta_rt.__dict__, {"MagentaRT2Jax": FakeMrt}), \
         mock.patch.dict(os.environ), \
         mock.patch.object(baseline, "require_idle_gpu"):
      results = baseline.run_experiment(self.args(
          "--no-text", "--generate", "--checkpoint", str(checkpoint),
          "--generate-condition", "*.pooled.*", "--generate-seconds", "0.08"), model=model)
    self.assertEqual(results.errors, [])
    self.assertEqual(len(calls), 2)
    self.assertTrue(all(c["state"] is None for c in calls))
    self.assertTrue(all(c["notes"] == [-1] * 128 for c in calls))
    self.assertTrue(all(c["frames"] == 2 for c in calls))
    for row in results.generations:
      info = sf.info(self.output / row["file"])
      self.assertEqual((info.samplerate, info.channels, info.frames), (48000, 2, 3840))


if __name__ == "__main__":
  unittest.main()
