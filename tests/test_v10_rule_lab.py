"""v1.0 neural arithmetic baselines, with no hidden addition in pair MLP."""
import tempfile
import unittest

from evolve.numeric_lab import split_addition_pairs
from evolve.settings import load_config
from evolve.store import Store
from evolve.rule_lab import run_rule_lab, predict_rule_lab

try:
    import torch
except ImportError:
    torch = None


class RuleLabContracts(unittest.TestCase):
    def test_no_operand_group_leakage_and_multiple_seed_splits(self):
        for seed in (42, 31415, 1001):
            groups = split_addition_pairs(seed)
            seen = set()
            for name in ("train", "validation", "test"):
                current = {tuple(sorted(p)) for p in groups[name]}
                self.assertEqual(len(current), len(groups[name]))
                self.assertFalse(current & seen)
                seen |= current
            self.assertEqual(len(seen), 55)
        self.assertNotEqual(split_addition_pairs(42), split_addition_pairs(31415))

    def test_rule_predict_checks_operand_range_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            self.assertEqual(predict_rule_lab(store, 1, 2)["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_rule_lab(store, 10, 2)
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class RuleLabNeuralTraining(unittest.TestCase):
    def test_black_box_pair_network_and_fair_additive_comparison(self):
        from evolve.numeric_lab import run_numeric_lab
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            settings = load_config()
            settings["storage_root"] = tmp
            settings["train"]["device"] = "cpu"
            settings["train"]["max_seconds"] = 120
            # The baseline uses the SAME seed and partition to compare fairly.
            previous = run_numeric_lab(store, settings, steps=400, seed=31415)
            self.assertEqual(previous["test_groups"], 10)
            before = store.get("numeric_lab_latest")
            result = run_rule_lab(store, settings, steps=500, seed=31415)
            self.assertEqual(result["status"], "trained")
            self.assertEqual(result["train_groups"], 35)
            self.assertEqual(result["validation_groups"], 10)
            self.assertEqual(result["test_groups"], 10)
            self.assertEqual(result["explicit_additive_baseline"]["status"],
                             "comparable")
            self.assertEqual(result["explicit_additive_baseline"]["test"]["total"], 10)
            self.assertGreaterEqual(result["after"]["test"]["accuracy"], 0.0)
            self.assertLessEqual(result["after"]["test"]["accuracy"], 1.0)
            self.assertEqual(len(result["test_examples"]), 10)
            self.assertEqual(result["after"]["test"]["correct"],
                             sum(int(row["correct"]) for row in result["test_examples"]))
            self.assertTrue(result["accepted"])
            self.assertEqual(store.get("numeric_lab_latest"), before)
            self.assertEqual(store.status()["documents"], 0)
            self.assertEqual(store.status()["research_questions_verified"], 0)
            self.assertIsNone(store.latest_promoted())
            self.assertTrue(store.get("rule_lab_latest"))
            guess = predict_rule_lab(store, 9, 8)
            self.assertEqual(guess["status"], "predicted")
            self.assertTrue(0 <= guess["model_answer"] <= 18)
            self.assertTrue(0 <= guess["confidence"] <= 1)
            self.assertEqual(guess["model_answer"],
                             predict_rule_lab(store, 8, 9)["model_answer"])
            store.close()

            reopened = Store(tmp)
            replay = predict_rule_lab(reopened, 9, 8)
            self.assertEqual(replay["model_answer"], guess["model_answer"])
            self.assertEqual(replay["checkpoint"], guess["checkpoint"])
            reopened.close()

    def test_no_matching_additive_seed_is_made_up(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            settings = load_config()
            settings["storage_root"] = tmp
            settings["train"]["device"] = "cpu"
            settings["train"]["max_seconds"] = 120
            report = run_rule_lab(store, settings, steps=250, seed=1001)
            self.assertEqual(report["explicit_additive_baseline"]["status"],
                             "not_available_for_this_seed")
            store.close()


if __name__ == "__main__":
    unittest.main()
