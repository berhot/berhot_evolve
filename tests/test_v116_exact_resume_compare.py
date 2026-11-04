"""v1.16: verify honest checkpoint continuation and historical comparison.

Fixture content is self-authored, never downloaded. Full CPU-PyTorch
training path and immutable checkpoint hashes are exercised.
"""
import math
import tempfile
import unittest
from pathlib import Path

from evolve.brain import load_checkpoint, sha256_file
from evolve.char_compare import compare_char_checkpoints
from evolve.char_lab import train_char_lab, predict_char_lab
from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


def _fixture(folder):
    store = Store(folder)
    article = " ".join(
        f"Energy transfer describes physical processes in systems. "
        f"The numerical observation {i} examines scientific data "
        f"from light, motion, heat, and electrical sources."
        for i in range(200)
    )
    archive_page(
        store, "https://learn.example.org/articles/energy",
        "<html><head><title>Energy</title></head><body><p>" + article +
        "</p></body></html>", article, title="Energy")
    cfg = load_config()
    cfg["storage_root"] = folder
    cfg["train"]["learning_rate"] = 0.002
    cfg["train"]["max_seconds"] = 90
    return store, cfg


def _opts(steps):
    return dict(steps=steps, seed=31415, device="cpu",
                include_unknown=True, diagnostic=True,
                context=64, embedding_dim=32, layers=1,
                heads=2, vocab_limit=128)


@unittest.skipIf(torch is None, "PyTorch not installed")
class HonestResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cpu_split_resume_matches_uninterrupted_weight_training(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as split:
            a, a_cfg = _fixture(full)
            b, b_cfg = _fixture(split)
            whole = train_char_lab(a, a_cfg, **_opts(30))
            half = train_char_lab(b, b_cfg, **_opts(15))
            self.assertEqual(whole["status"], "trained")
            self.assertEqual(half["status"], "trained")
            self.assertTrue(half["accepted_by_validation"])
            self.assertEqual(half["continuation_mode"], "new_model")
            self.assertTrue(half["optimizer_state_saved"])
            self.assertTrue(half["batch_sampler_rng_state_saved"])
            _, half_meta = load_checkpoint(half["checkpoint"])
            self.assertIn("optimizer_state", half_meta)
            self.assertIn("sampler_rng_state", half_meta)
            self.assertIn("torch_cpu_rng_state", half_meta)

            continued = train_char_lab(b, b_cfg, **_opts(15), resume=True)
            self.assertEqual(continued["status"], "trained")
            self.assertTrue(continued["exact_cpu_state_restored"])
            self.assertEqual(continued["continuation_mode"], "stateful_optimizer_rng")
            self.assertEqual(continued["total_training_steps"], 30)
            self.assertEqual(continued["resumed_from_sha256"], half["sha256"])
            whole_model, _ = load_checkpoint(whole["checkpoint"])
            resumed_model, _ = load_checkpoint(continued["checkpoint"])
            for key, tensor in whole_model.state_dict().items():
                torch.testing.assert_close(
                    tensor, resumed_model.state_dict()[key],
                    rtol=0, atol=0)
            self.assertAlmostEqual(
                whole["validation_char_loss_after"],
                continued["validation_char_loss_after"], places=5)
            self.assertAlmostEqual(
                whole["heldout_test_char_loss_after"],
                continued["heldout_test_char_loss_after"], places=5)

            comparison = compare_char_checkpoints(
                b, half["sha256"], prompt="Energy is", new_chars=20)
            self.assertEqual(comparison["status"], "compared")
            self.assertEqual(comparison["reference_training_steps"], 15)
            self.assertEqual(comparison["latest_training_steps"], 30)
            self.assertEqual(comparison["reference_checkpoint_sha256"], half["sha256"])
            self.assertEqual(comparison["latest_checkpoint_sha256"], continued["sha256"])
            self.assertFalse(comparison["test_used_for_checkpoint_selection"])
            self.assertEqual(comparison["generation"]["seed"], 42)
            self.assertTrue(comparison["no_weights_modified"])
            self.assertEqual(
                len(comparison["generation"]["reference_continuation"]), 20)
            self.assertEqual(
                len(comparison["generation"]["latest_continuation"]), 20)
            self.assertEqual(
                b.get("char_lab_latest_candidate")["sha256"],
                continued["sha256"])
            with self.assertRaisesRegex(ValueError, "64 lowercase"):
                compare_char_checkpoints(b, "abc")
            with self.assertRaisesRegex(ValueError, "not found"):
                compare_char_checkpoints(b, "0" * 64)
            a.close()
            b.close()

    def test_legacy_resume_is_labeled_weights_only_not_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            store, cfg = _fixture(folder)
            first = train_char_lab(store, cfg, **_opts(15))
            self.assertTrue(first["accepted_by_validation"])
            data = torch.load(first["checkpoint"], map_location="cpu",
                              weights_only=True)
            for field in (
                "optimizer_state", "sampler_rng_state",
                "torch_cpu_rng_state", "optimizer_learning_rate",
                "train_batch_size", "continuation_mode",
            ):
                data.pop(field, None)
            old = Path(folder) / "models" / "char_lab" / "legacy-v115.pt"
            torch.save(data, old)
            changed = dict(store.get("char_lab_latest"))
            changed["checkpoint"] = str(old)
            changed["sha256"] = sha256_file(old)
            store.put("char_lab_latest", changed)
            after = train_char_lab(store, cfg, **_opts(8), resume=True)
            self.assertEqual(after["status"], "trained")
            self.assertEqual(after["continuation_mode"], "legacy_weights_only")
            self.assertFalse(after["exact_cpu_state_restored"])
            self.assertEqual(after["resumed_from_sha256"], changed["sha256"])
            self.assertEqual(after["total_training_steps"], 23)
            store.close()

    def test_stateful_resume_rejects_optimizer_hyperparam_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            store, cfg = _fixture(folder)
            first = train_char_lab(store, cfg, **_opts(15))
            self.assertTrue(first["accepted_by_validation"])
            cfg["train"]["learning_rate"] = 0.01
            skip = train_char_lab(store, cfg, **_opts(8), resume=True)
            self.assertEqual(skip["status"], "skipped")
            self.assertEqual(skip["reason"],
                             "resume_optimizer_hyperparameters_mismatch")
            self.assertEqual(store.get("char_lab_latest")["sha256"],
                             first["sha256"])
            store.close()


if __name__ == "__main__":
    unittest.main()
