"""v1.8: PyTorch-only gradient training of frozen v1.5 output heads."""
import tempfile
import unittest
from unittest.mock import patch

from evolve.gradient_lab import (
    OPTIMIZERS, _load_gradient, run_gradient_lab,
    run_gradient_study, predict_gradient_lab,
)
from evolve.basis_lab import GROUPS
from evolve.multiop_lab import OPS
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class GradientContracts(unittest.TestCase):
    def test_no_source_and_strict_prediction_domain(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            skipped = run_gradient_lab(store, seed=31415, steps=30, count=3)
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(
                predict_gradient_lab(store, 1, 2, "+")["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_gradient_lab(store, 1000, 2, "+")
            with self.assertRaises(ValueError):
                predict_gradient_lab(store, 2, 2, "/")
            with self.assertRaises(ValueError):
                predict_gradient_lab(store, 2, 2, "+", optimizer="unknown")
            with self.assertRaises(ValueError):
                run_gradient_study(store, seeds=(42, 42))
            with self.assertRaises(ValueError):
                run_gradient_study(store, seeds=(42,))
            with self.assertRaises(ValueError):
                run_gradient_study(store, optimizers=("adam", "adam"))
            with self.assertRaises(ValueError):
                run_gradient_lab(store, optimizer="adam", steps=0)
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class RealGradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def cfg(self, folder):
        cfg = load_config()
        cfg["storage_root"] = folder
        cfg["train"]["device"] = "cpu"
        cfg["train"]["max_seconds"] = 90
        return cfg

    def test_actual_adam_updates_head_only_without_least_squares(self):
        from evolve.interaction_lab import run_interaction_lab, _load_checkpoint
        from evolve.calibration_lab import run_calibration_lab, _load_calibrated
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            original = run_interaction_lab(store, self.cfg(folder),
                                           seed=31415, steps=140,
                                           rank=8, count=3)
            old_record = store.get("interaction_lab_seed_31415")
            old_sha = sha256_file(original["checkpoint"])
            old_model, _ = _load_checkpoint(store, seed=31415)

            ls_ref = run_calibration_lab(store, seed=31415, count=3)
            ls_record = store.get("calibration_lab_seed_31415")
            ls_sha = sha256_file(ls_ref["checkpoint"])
            with patch.object(
                torch.linalg, "lstsq",
                side_effect=AssertionError("Gradient optimizer cannot call least squares")
            ):
                result = run_gradient_lab(
                    store, seed=31415, optimizer="adam", steps=280,
                    count=4, max_seconds=90)

            self.assertEqual(result["status"], "trained")
            self.assertEqual(result["optimizer"], "adam")
            self.assertIsInstance(result["accepted"], bool)
            self.assertTrue(result["frozen_layers_unchanged"])
            self.assertEqual(result["train_groups"], 35)
            self.assertEqual(result["validation_groups"], 10)
            self.assertEqual(result["test_groups"], 10)
            self.assertEqual(result["saved_v17_reference"]["status"],
                             "same_seed_saved_v17_reference")
            self.assertEqual(result["saved_v17_reference"]["checkpoint_sha256"], ls_sha)
            self.assertEqual(set(result["test"]), set(OPS))
            self.assertEqual(set(result["ood"]), set(GROUPS))
            self.assertEqual(store.get("interaction_lab_seed_31415"), old_record)
            self.assertEqual(store.get("calibration_lab_seed_31415"), ls_record)
            self.assertEqual(sha256_file(original["checkpoint"]), old_sha)
            self.assertEqual(sha256_file(ls_ref["checkpoint"]), ls_sha)
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())

            fitted, record = _load_gradient(store, 31415, "adam")
            for key, old in old_model.state_dict().items():
                if key.startswith(("left.", "right.")):
                    self.assertTrue(torch.equal(fitted.state_dict()[key],
                                                old.double()))
            self.assertEqual(record["source_sha256"], old_sha)
            for op in OPS:
                predicted = predict_gradient_lab(
                    store, 27, 28, op, seed=31415, optimizer="adam")
                self.assertEqual(predicted["status"], "predicted")
                self.assertEqual(predicted["checkpoint"], result["sha256"])
            store.close()
            reopened = Store(folder)
            a = predict_gradient_lab(reopened, 999, 998, "*", seed=31415)
            b = predict_gradient_lab(reopened, 999, 998, "*", seed=31415)
            self.assertEqual(a["model_answer"], b["model_answer"])
            reopened.close()

    def test_sgd_trains_from_saved_head_and_reports_failures(self):
        from evolve.interaction_lab import run_interaction_lab
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            run_interaction_lab(store, self.cfg(folder), steps=120,
                                rank=4, seed=42, count=3)
            result = run_gradient_lab(store, seed=42,
                                      optimizer="sgd", steps=150, count=5)
            self.assertEqual(result["status"], "trained")
            self.assertEqual(result["optimizer"], "sgd")
            self.assertTrue(1 <= result["best_step"] <= result["actual_steps"])
            self.assertTrue(result["best_validation_normalized_mse"] >= 0)
            self.assertTrue(result["final_training_normalized_mse"] >= 0)
            self.assertEqual(result["saved_v17_reference"]["status"],
                             "missing_saved_v17_reference")
            self.assertIsNone(result["coefficient_distance_from_v17"])
            self.assertEqual(set(result["ood"]), set(GROUPS))
            for group in GROUPS:
                self.assertIsNone(result["ood"][group]["least_squares_v17"])
                for op in OPS:
                    exact = result["ood"][group]["gradient"][op]["accuracy"]
                    self.assertTrue(0 <= exact <= 1)
            self.assertIsNone(store.latest_promoted())
            store.close()

    def test_two_optimizers_two_seeds_report_all_comparisons(self):
        from evolve.interaction_lab import run_interaction_lab
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = self.cfg(folder)
            for seed in (31415, 42):
                run_interaction_lab(store, cfg, seed=seed,
                                    rank=4, steps=80, count=3)
            study = run_gradient_study(
                store, seeds=(31415, 42), optimizers=("adam", "sgd"),
                steps=100, count=3, max_seconds=30)
            self.assertEqual(study["status"], "evaluated")
            self.assertEqual(len(study["runs"]), 4)
            self.assertEqual(study["optimizers"], ["adam", "sgd"])
            self.assertEqual(set(study["ood_summary"]), set(GROUPS))
            self.assertEqual([(r["seed"], r["optimizer"]) for r in study["runs"]],
                [(31415,"adam"), (31415,"sgd"), (42,"adam"), (42,"sgd")])
            for item in study["runs"]:
                self.assertIsInstance(item["accepted"], bool)
                self.assertEqual(set(item["hardest_ood_accuracy"]), set(OPS))
            for group in GROUPS:
                for operation in OPS:
                    for opt in OPTIMIZERS:
                        stat = study["ood_summary"][group][operation][opt]
                        self.assertEqual(stat["models"], 2)
                        self.assertTrue(0 <= stat["min_accuracy"] <=
                                        stat["mean_accuracy"] <=
                                        stat["max_accuracy"] <= 1)
            store.close()

    def test_corrupted_gradient_checkpoint_cannot_predict(self):
        from evolve.interaction_lab import run_interaction_lab
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            run_interaction_lab(store, self.cfg(folder),
                                steps=90, count=3, rank=4)
            result = run_gradient_lab(store, steps=90, count=3)
            with open(result["checkpoint"], "ab") as output:
                output.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_gradient_lab(store, 1, 2, "+")
            store.close()


if __name__ == "__main__":
    unittest.main()
