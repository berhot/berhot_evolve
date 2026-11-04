"""v1.3: compare learned non-saturating numeric prior to frozen tanh MLP."""
import tempfile
import unittest

from evolve.extrapolation import RANGES, select_ood_pairs
from evolve.linear_lab import (
    _build_model, predict_linear_lab, run_linear_lab, run_linear_study,
)
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class LinearContracts(unittest.TestCase):
    def test_same_ood_probes_as_v12(self):
        for group in RANGES:
            pairs = select_ood_pairs(group, count=20, seed=20261104)
            self.assertEqual(len(pairs), 20)
            self.assertTrue(all(max(p) > 9 for p in pairs))

    def test_no_checkpoint_yet_and_outside_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(predict_linear_lab(store, 14, 17)["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_linear_lab(store, -1, 17)
            with self.assertRaises(ValueError):
                predict_linear_lab(store, 10001, 3)
            with self.assertRaises(ValueError):
                run_linear_study(store, load_config(), seeds=(42, 42))
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class LinearNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_untrained_zero_weights_do_not_return_true_sum(self):
        model = _build_model()
        with torch.no_grad():
            model.mapping.weight.zero_()
            model.mapping.bias.zero_()
            prediction = model(torch.tensor([14]), torch.tensor([17])) * 18
        self.assertEqual(float(prediction.item()), 0.0)
        self.assertNotEqual(int(prediction.round().item()), 31)

    def test_learned_coefficients_and_frozen_mlp_comparison(self):
        from evolve.magnitude_lab import run_magnitude_lab
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = load_config()
            cfg["storage_root"] = folder
            cfg["train"]["device"] = "cpu"
            cfg["train"]["max_seconds"] = 120
            baseline = run_magnitude_lab(store, cfg, seed=31415, steps=300)
            baseline_status = store.get("magnitude_lab_latest")
            baseline_hash = sha256_file(baseline["checkpoint"])
            trial = run_linear_lab(store, cfg, seed=31415, steps=500,
                                   count=12, probe_seed=20261104)
            self.assertEqual(trial["status"], "trained")
            self.assertTrue(trial["accepted"])
            self.assertEqual(trial["train_groups"], 35)
            self.assertEqual(trial["validation_groups"], 10)
            self.assertEqual(trial["test_groups"], 10)
            self.assertLess(trial["after"]["validation"]["mse"],
                            trial["before"]["validation"]["mse"])
            self.assertEqual(trial["frozen_tanh_mlp_baseline"]["status"],
                             "comparable_frozen_checkpoint")
            self.assertEqual(trial["frozen_tanh_mlp_baseline"]["checkpoint_sha256"],
                             baseline["sha256"])
            self.assertEqual(store.get("magnitude_lab_latest"), baseline_status)
            self.assertEqual(sha256_file(baseline["checkpoint"]), baseline_hash)
            for group in RANGES:
                data = trial["ood"][group]
                self.assertEqual(data["questions"], 12)
                self.assertTrue(0 <= data["accuracy"] <= 1)
                self.assertTrue(data["mean_absolute_error"] >= 0)
            # Model learning, not an oracle: coefficients are saved and decoded.
            self.assertEqual(len(trial["learned_coefficients"]), 2)
            answer = predict_linear_lab(store, 14, 17, seed=31415)
            self.assertEqual(answer["status"], "predicted")
            self.assertEqual(answer["model_status"], "accepted")
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            store.close()
            reopened = Store(folder)
            replay = predict_linear_lab(reopened, 14, 17, seed=31415)
            self.assertEqual(answer["model_answer"], replay["model_answer"])
            self.assertEqual(answer["checkpoint"], replay["checkpoint"])
            reopened.close()

    def test_study_reports_all_seed_failures_as_well_as_successes(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = load_config()
            cfg["storage_root"] = folder
            cfg["train"]["device"] = "cpu"
            cfg["train"]["max_seconds"] = 120
            study = run_linear_study(store, cfg, seeds=(31415, 1001),
                                     steps=300, count=10)
            self.assertEqual(study["seeds"], [31415, 1001])
            self.assertEqual(len(study["results"]), 2)
            for group in RANGES:
                values = [r["ood_accuracy"][group] for r in study["results"]]
                self.assertAlmostEqual(study["ood_summary"][group]["mean"],
                                       round(sum(values)/2, 4))
            self.assertEqual(store.status()["documents"], 0)
            store.close()

    def test_tampered_linear_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = load_config()
            cfg["storage_root"] = folder
            cfg["train"]["device"] = "cpu"
            cfg["train"]["max_seconds"] = 120
            trial = run_linear_lab(store, cfg, seed=31415, steps=120, count=5)
            with open(trial["checkpoint"], "ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_linear_lab(store, 4, 3, seed=31415)
            store.close()


if __name__ == "__main__":
    unittest.main()
