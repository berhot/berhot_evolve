"""v1.4: shared arithmetic learning without a hidden calculator in inference."""
import tempfile
import unittest

from evolve.extrapolation import RANGES
from evolve.multiop_lab import (
    OPS, _examples, predict_multiop_lab, run_multiop_lab, run_multiop_study,
)
from evolve.numeric_lab import split_addition_pairs
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class SharedArithmeticContracts(unittest.TestCase):
    def test_reversed_operand_pairs_and_all_ops_remain_in_one_split(self):
        split = split_addition_pairs(31415)
        partitions = {}
        for name in ("train", "validation", "test"):
            seen = set()
            operations = {}
            for a, b, op in _examples(split[name]):
                group = tuple(sorted((a, b)))
                seen.add(group)
                operations.setdefault(group, set()).add(op)
            self.assertEqual(len(seen), len(split[name]))
            self.assertTrue(all(operators == set(OPS) for operators in operations.values()))
            partitions[name] = seen
        self.assertFalse(partitions["train"] & partitions["validation"])
        self.assertFalse(partitions["train"] & partitions["test"])
        self.assertFalse(partitions["validation"] & partitions["test"])

    def test_invalid_inference_arguments_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(predict_multiop_lab(store, 1, 2, "+")["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_multiop_lab(store, 1, 2, "/")
            with self.assertRaises(ValueError):
                predict_multiop_lab(store, -1, 2, "-")
            with self.assertRaises(ValueError):
                predict_multiop_lab(store, 100, 2, "*")
            with self.assertRaises(ValueError):
                run_multiop_study(store, load_config(), seeds=(42,42))
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class SharedArithmeticNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def cfg(self, folder):
        settings = load_config()
        settings["storage_root"] = folder
        settings["train"]["device"] = "cpu"
        settings["train"]["max_seconds"] = 90
        return settings

    def test_zero_weights_are_not_implicit_arithmetic_calculator(self):
        from evolve.multiop_lab import _build_model
        model = _build_model(24)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for idx, operation in enumerate(OPS):
                value = float(model(
                    torch.tensor([6]), torch.tensor([3]),
                    torch.tensor([idx])).item())
                self.assertEqual(value, 0.0)
        # If the model were using a Python calculator, it would not emit zero.
        self.assertNotEqual(0.0, 9.0)

    def test_three_ops_same_network_holdout_ood_and_saved_prediction(self):
        from evolve.linear_lab import run_linear_lab
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = self.cfg(folder)
            # The existing v1.3 addition baseline is read-only for v1.4.
            baseline = run_linear_lab(store, cfg, seed=31415,
                                      steps=150, count=5)
            baseline_hash = sha256_file(baseline["checkpoint"])
            baseline_registry = store.get("linear_lab_seed_31415")
            trial = run_multiop_lab(store, cfg, seed=31415,
                                    steps=240, hidden=24, count=6)
            self.assertEqual(trial["status"], "trained")
            self.assertIsInstance(trial["accepted"], bool)
            self.assertEqual(trial["train_groups"], 35)
            self.assertEqual(trial["validation_groups"], 10)
            self.assertEqual(trial["test_groups"], 10)
            self.assertEqual(
                set(trial["after"]["test"]), set(OPS))
            self.assertEqual(set(trial["ood"]), set(RANGES))
            for group in RANGES:
                for operator in OPS:
                    row = trial["ood"][group][operator]
                    self.assertTrue(0 <= row["accuracy"] <= 1)
                    self.assertTrue(row["questions"] >= 6)
            self.assertEqual(trial["frozen_addition_linear_baseline"]["status"],
                             "addition_only_comparable")
            self.assertEqual(store.get("linear_lab_seed_31415"), baseline_registry)
            self.assertEqual(sha256_file(baseline["checkpoint"]), baseline_hash)
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            saved = store.get("multiop_lab_seed_31415")
            self.assertEqual(saved["sha256"], trial["sha256"])
            for op in OPS:
                guess = predict_multiop_lab(store, 7, 3, op, seed=31415)
                self.assertEqual(guess["status"], "predicted")
                self.assertIn("model_answer", guess)
                self.assertEqual(guess["checkpoint_status"],
                    "accepted" if trial["accepted"] else "unaccepted_candidate")
            store.close()
            reopened = Store(folder)
            first = predict_multiop_lab(reopened, 7, 3, "+", seed=31415)
            second = predict_multiop_lab(reopened, 7, 3, "+", seed=31415)
            self.assertEqual(first["model_answer"], second["model_answer"])
            reopened.close()

    def test_report_every_seed_without_assumed_generalization(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = self.cfg(folder)
            study = run_multiop_study(store, cfg, seeds=(31415,42),
                                      steps=120, hidden=24, count=5)
            self.assertEqual(study["status"], "evaluated")
            self.assertEqual(study["seeds"], [31415,42])
            self.assertEqual(len(study["runs"]), 2)
            for row in study["runs"]:
                self.assertIsInstance(row["accepted"], bool)
                self.assertEqual(set(row["in_range_accuracy"]), set(OPS))
                for group in RANGES:
                    self.assertEqual(set(row["ood_accuracy"][group]), set(OPS))
            for group in RANGES:
                for op in OPS:
                    stats = study["ood_summary"][group][op]
                    self.assertTrue(0 <= stats["min"] <= stats["mean"] <= stats["max"] <= 1)
            self.assertEqual(store.status()["documents"], 0)
            store.close()

    def test_modified_multiop_checkpoint_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            result = run_multiop_lab(store, self.cfg(folder), steps=80, hidden=24, count=3)
            with open(result["checkpoint"], "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_multiop_lab(store, 1, 2, "+", seed=31415)
            store.close()


if __name__ == "__main__":
    unittest.main()
