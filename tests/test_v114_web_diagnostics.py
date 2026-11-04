"""v1.14: quantify whether web LM beats an independent byte-frequency baseline.

All texts here are original synthetic fixtures. Real PyTorch checkpoint tests
perform actual optimizer updates. No public network requests and no changes
to the user's local saved checkpoints.
"""
import math
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_diagnostics import (
    _generation_stats, diagnose_web_lab, unigram_baseline,
)
from evolve.web_lab import train_web_lab, predict_web_lab

try:
    import torch
except ImportError:
    torch = None


def fixture(folder):
    store = Store(folder)
    raw = " ".join(
        f"Energy transfer is studied with observation, careful scientific "
        f"experiments and technical measurements. Trial {i} describes a "
        f"physical process using material systems and analytical evidence."
        for i in range(180)
    )
    archive_page(store, "https://original.example.org/research/energy",
                 f"<html><head><title>Energy research</title></head><body>{raw}</body></html>",
                 raw, "Energy research")
    cfg = load_config()
    cfg["storage_root"] = folder
    cfg["train"]["max_seconds"] = 90
    cfg["train"]["learning_rate"] = 0.001
    return store, cfg


class UnigramDiagnosticUnitTests(unittest.TestCase):
    def test_unigram_fits_training_bytes_only_and_uses_correct_smoothing(self):
        train = b"a" * 100
        validation = b"a" * 70
        expected = -math.log((100 + 1) / (100 + 256))
        loss = unigram_baseline(train, validation, block=16,
                                eval_batches=2, batch_size=4)
        self.assertAlmostEqual(loss, expected, places=12)
        # Out-of-distribution heldout bytes are expensive; not trained on.
        unknown = unigram_baseline(train, b"z" * 70, block=16,
                                   eval_batches=2, batch_size=4)
        self.assertAlmostEqual(unknown, math.log(100 + 256), places=12)
        self.assertGreater(unknown, loss)

    def test_repetition_stats_analyze_continuation_not_prompt(self):
        result = _generation_stats("Energy is" + "     eee", "Energy is")
        self.assertEqual(result["generated_bytes"], 8)
        self.assertEqual(result["unique_bytes"], 2)
        self.assertAlmostEqual(result["whitespace_fraction"], 5/8, places=4)
        self.assertEqual(result["longest_identical_byte_run"], 5)

    def test_diagnose_skips_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(diagnose_web_lab(store)["status"], "skipped")
            store.close()


@unittest.skipIf(torch is None, "PyTorch required for neural checkpoint test")
class WebNeuralDiagnosticIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_saved_model_against_unigram_and_deterministic_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            store, settings = fixture(folder)
            trained = train_web_lab(
                store, settings, steps=8, seed=55, device="cpu",
                include_unknown=True, diagnostic=True)
            self.assertEqual(trained["status"], "trained")
            self.assertEqual(trained["actual_steps"], 8)
            reference_sha = trained["sha256"]
            report = diagnose_web_lab(store, prompt="Energy is", seed=123,
                                      temperature=0.8, max_new_bytes=16)
            self.assertEqual(report["status"], "diagnosed")
            self.assertEqual(report["checkpoint_sha256"], reference_sha)
            self.assertEqual(report["dataset_sha256"], trained["dataset_sha256"])
            self.assertTrue(report["diagnostic_source_leakage"])
            self.assertFalse(report["generation"]["model_weights_changed"])
            self.assertEqual(report["train_only_fitted_baseline"],
                             "Laplace-smoothed 256-byte unigram")
            for split in ("train", "val", "test"):
                row = report["evaluation"][split]
                self.assertGreater(row["neural_perplexity"], 0)
                self.assertGreater(row["unigram_perplexity"], 0)
                self.assertTrue(math.isfinite(row["neural_minus_unigram_nats"]))
                self.assertAlmostEqual(
                    row["neural_byte_ce_nats"],
                    row["train_unigram_byte_ce_nats"] + row["neural_minus_unigram_nats"],
                    places=3)
            again = diagnose_web_lab(store, prompt="Energy is", seed=123,
                                     temperature=0.8, max_new_bytes=16)
            self.assertEqual(report["generation"], again["generation"])
            sampled = predict_web_lab(
                store, "Energy is", decoding="sample",
                temperature=0.8, seed=123, max_new_bytes=16)
            self.assertEqual(sampled["decoding"], "sample")
            self.assertEqual(sampled["neural_generation"],
                             report["generation"]["sampled"]["text"])
            greedy = predict_web_lab(
                store, "Energy is", decoding="greedy", max_new_bytes=16)
            self.assertEqual(greedy["decoding"], "greedy")
            self.assertEqual(greedy["neural_generation"],
                             report["generation"]["greedy"]["text"])
            self.assertEqual(store.get("web_lab_latest_candidate")["sha256"],
                             reference_sha)
            self.assertIsNone(store.latest_promoted())
            for invalid in (0.001, 4.0):
                with self.assertRaises(ValueError):
                    predict_web_lab(store, "Energy", decoding="sample",
                                    temperature=invalid)
                with self.assertRaises(ValueError):
                    diagnose_web_lab(store, temperature=invalid)
            store.close()


if __name__ == "__main__":
    unittest.main()
