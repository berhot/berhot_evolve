"""V0.9 regression tests: no overfit promotion and a separate neural baseline."""
import tempfile
import unittest
from pathlib import Path

from evolve.answer_training import should_accept_math_checkpoint
from evolve.numeric_lab import split_addition_pairs, run_numeric_lab, predict_numeric_lab
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class PromotionSafetyTests(unittest.TestCase):
    def test_first_overfit_checkpoint_is_rejected(self):
        # Reproduces the measured v0.8 mixed result on macOS.
        self.assertFalse(should_accept_math_checkpoint(2.5784, 3.3191, 0.001))
        # A genuine held-out improvement is eligible, not guaranteed accurate.
        self.assertTrue(should_accept_math_checkpoint(2.3033, 1.9399, 0.001))
        self.assertFalse(should_accept_math_checkpoint(
            2.0, 1.8, 0.001, previous_checkpoint=1.7))
        self.assertFalse(should_accept_math_checkpoint(
            2.0, float("nan"), 0.001))
        self.assertFalse(should_accept_math_checkpoint(
            2.0, 1.0, 0.001, best_step=0))

    def test_composition_splits_have_no_forward_or_reverse_leakage(self):
        split = split_addition_pairs(seed=42)
        buckets = [
            {tuple(sorted(pair)) for pair in group}
            for group in (split["train"], split["validation"], split["test"])
        ]
        self.assertEqual([len(g) for g in buckets], [35, 10, 10])
        self.assertFalse(buckets[0] & buckets[1])
        self.assertFalse(buckets[0] & buckets[2])
        self.assertFalse(buckets[1] & buckets[2])
        self.assertEqual(len(set().union(*buckets)), 55)
        self.assertEqual(split, split_addition_pairs(seed=42))


@unittest.skipIf(torch is None, "PyTorch not installed")
class StructuredNeuralAdditionTests(unittest.TestCase):
    def test_learned_values_generalize_without_lookup_or_test_leakage(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as path:
            settings = load_config()
            settings["storage_root"] = path
            settings["train"]["device"] = "cpu"
            settings["train"]["max_seconds"] = 120
            store = Store(path)
            before = store.status()
            self.assertEqual(before["documents"], 0)
            self.assertEqual(before["research_questions_verified"], 0)
            result = run_numeric_lab(store, settings, steps=400, seed=42)
            self.assertEqual(result["status"], "trained")
            self.assertEqual(result["train_groups"], 35)
            self.assertEqual(result["validation_groups"], 10)
            self.assertEqual(result["test_groups"], 10)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["after"]["test"]["correct"], 10)
            self.assertLess(result["after"]["test"]["mse"],
                            result["before"]["test"]["mse"])
            self.assertEqual(result["after"]["test"]["accuracy"], 1.0)
            self.assertEqual(len(result["learned_digit_values"]), 10)
            self.assertTrue(Path(result["checkpoint"]).exists())
            # The independent numeric model must not imply that web/LM
            # neural skills or source-backed answers have progressed.
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            self.assertEqual(store.status()["research_questions_verified"], 0)
            p = predict_numeric_lab(store, 9, 8)
            self.assertEqual(p["status"], "predicted")
            self.assertEqual(p["model_answer"], 17)
            store.close()

            reopened = Store(path)
            again = predict_numeric_lab(reopened, 9, 8)
            self.assertEqual(again["model_answer"], 17)
            self.assertEqual(again["checkpoint"], p["checkpoint"])
            reopened.close()

    def test_numeric_specialist_refuses_out_of_scope_operands(self):
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            self.assertEqual(predict_numeric_lab(store, 2, 3)["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_numeric_lab(store, 11, 4)
            store.close()


if __name__ == "__main__":
    unittest.main()
