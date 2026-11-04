"""v1.1: numerical magnitude vs categorical embedding on leakage-free splits."""
import tempfile
import unittest

from evolve.numeric_lab import split_addition_pairs
from evolve.settings import load_config
from evolve.store import Store
from evolve.magnitude_lab import run_magnitude_lab, run_magnitude_study, predict_magnitude_lab

try:
    import torch
except ImportError:
    torch = None


class MagnitudeContracts(unittest.TestCase):
    def test_disjoint_unordered_train_val_test_groups(self):
        for seed in (31415, 42, 1001):
            split = split_addition_pairs(seed)
            groups = [{tuple(sorted(p)) for p in split[k]}
                      for k in ("train", "validation", "test")]
            self.assertEqual([len(g) for g in groups], [35, 10, 10])
            self.assertEqual(len(set().union(*groups)), 55)
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])

    def test_out_of_domain_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(predict_magnitude_lab(store, 4, 8)["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_magnitude_lab(store, 10, 1)
            store.close()

    def test_study_requires_multiple_distinct_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            settings = load_config()
            with self.assertRaises(ValueError):
                run_magnitude_study(store, settings, seeds=(42, 42))
            with self.assertRaises(ValueError):
                run_magnitude_study(store, settings, seeds=(42,))
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class MagnitudeLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_magnitude_neural_fit_and_matching_baselines(self):
        from evolve.numeric_lab import run_numeric_lab
        from evolve.rule_lab import run_rule_lab
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = load_config()
            cfg["storage_root"] = folder
            cfg["train"]["device"] = "cpu"
            cfg["train"]["max_seconds"] = 120

            # Each model receives the same pairs and seed.
            old_additive = run_numeric_lab(store, cfg, steps=400, seed=31415)
            old_classifier = run_rule_lab(store, cfg, steps=120, seed=31415)
            saved_additive = store.get("numeric_lab_latest")
            saved_classifier = store.get("rule_lab_latest_candidate")

            output = run_magnitude_lab(store, cfg, steps=800, seed=31415)
            self.assertEqual(output["status"], "trained")
            self.assertEqual(output["experiment"], "magnitude_mlp_addition_v1")
            self.assertEqual(output["train_groups"], 35)
            self.assertEqual(output["validation_groups"], 10)
            self.assertEqual(output["test_groups"], 10)
            self.assertTrue(output["accepted"])
            self.assertLess(output["after"]["validation"]["mse"],
                            output["before"]["validation"]["mse"])
            self.assertEqual(output["after"]["test"]["correct"],
                             sum(int(item["correct"]) for item in output["test_examples"]))
            self.assertEqual(output["categorical_embedding_baseline"]["status"],
                             "comparable")
            self.assertEqual(output["explicit_additive_baseline"]["status"],
                             "comparable")
            self.assertEqual(output["explicit_additive_baseline"]["test"]["total"], 10)
            self.assertEqual(store.get("numeric_lab_latest"), saved_additive)
            self.assertEqual(store.get("rule_lab_latest_candidate"), saved_classifier)

            pred = predict_magnitude_lab(store, 9, 8)
            reverse = predict_magnitude_lab(store, 8, 9)
            self.assertEqual(pred["status"], "predicted")
            self.assertEqual(pred["model_answer"], reverse["model_answer"])
            self.assertEqual(pred["checkpoint_status"], "accepted")
            self.assertEqual(store.status()["research_questions_verified"], 0)
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            store.close()

            store = Store(folder)
            replay = predict_magnitude_lab(store, 9, 8)
            self.assertEqual(replay["model_answer"], pred["model_answer"])
            self.assertEqual(replay["checkpoint"], pred["checkpoint"])
            store.close()

    def test_study_reports_all_predeclared_seeds_without_cherry_picking(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = load_config()
            cfg["storage_root"] = folder
            cfg["train"]["device"] = "cpu"
            cfg["train"]["max_seconds"] = 120
            result = run_magnitude_study(store, cfg, seeds=(31415, 42, 1001),
                                         steps=300)
            self.assertEqual(result["status"], "evaluated")
            self.assertEqual(result["seeds"], [31415, 42, 1001])
            self.assertEqual([r["seed"] for r in result["results"]],
                             [31415, 42, 1001])
            self.assertEqual(len(result["results"]), 3)
            scores = [r["test_accuracy"] for r in result["results"]]
            self.assertAlmostEqual(result["test_mean"],
                                   round(sum(scores) / 3, 4))
            self.assertEqual(result["test_min"], min(scores))
            self.assertEqual(result["test_max"], max(scores))
            self.assertEqual(store.status()["documents"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
