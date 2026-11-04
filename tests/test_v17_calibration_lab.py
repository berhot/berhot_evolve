"""v1.7 frozen v1.5 features: head-only calibration, no OOD-label leakage."""
import tempfile
import unittest

from evolve.calibration_lab import (
    _build_model, _fit_head, _load_calibrated,
    run_calibration_lab, run_calibration_study,
    predict_calibration_lab,
)
from evolve.basis_lab import GROUPS
from evolve.multiop_lab import OPS
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class CalibrationContracts(unittest.TestCase):
    def test_skips_without_saved_source_and_validates_operands(self):
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            self.assertEqual(run_calibration_lab(store)["status"], "skipped")
            self.assertEqual(predict_calibration_lab(store, 1, 2, "+")["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_calibration_lab(store, 1, 2, "/")
            with self.assertRaises(ValueError):
                predict_calibration_lab(store, -1, 2, "+")
            with self.assertRaises(ValueError):
                predict_calibration_lab(store, 1000, 999, "*")
            with self.assertRaises(ValueError):
                run_calibration_study(store, seeds=(42, 42))
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class FrozenHeadNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _cfg(self, path):
        cfg = load_config()
        cfg["storage_root"] = path
        cfg["train"]["device"] = "cpu"
        cfg["train"]["max_seconds"] = 120
        return cfg

    def test_rank_deficient_hidden_features_fail_instead_of_fabricating_answers(self):
        from evolve.numeric_lab import split_addition_pairs
        model = _build_model(4)
        with torch.no_grad():
            model.left.weight.zero_()
            model.left.bias.zero_()
            model.right.weight.zero_()
            model.right.bias.zero_()
            model.head.weight.zero_()
            model.head.bias.zero_()
        raw = model(torch.tensor([27]), torch.tensor([28]),
                    torch.tensor([OPS.index("*")]))
        self.assertEqual(float(raw.item()), 0.0)
        with self.assertRaisesRegex(ValueError, "rank"):
            _fit_head(model, split_addition_pairs(31415)["train"])

    def test_train_only_head_calibration_freezes_hidden_and_preserves_v15(self):
        from evolve.brain import sha256_file
        from evolve.interaction_lab import run_interaction_lab, _load_checkpoint
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            source = run_interaction_lab(store, self._cfg(path), seed=31415,
                                         steps=200, rank=8, count=4)
            old_sha = sha256_file(source["checkpoint"])
            old_record = store.get("interaction_lab_seed_31415")
            old_weights, _ = _load_checkpoint(store, seed=31415)
            report = run_calibration_lab(store, seed=31415, count=8)
            self.assertEqual(report["status"], "calibrated")
            self.assertEqual(report["train_group_count"], 35)
            self.assertEqual(report["validation_group_count"], 10)
            self.assertEqual(report["test_group_count"], 10)
            self.assertTrue(report["hidden_layers_unchanged"])
            self.assertEqual(report["source_checkpoint_sha256"], old_sha)
            self.assertEqual(store.get("interaction_lab_seed_31415"), old_record)
            self.assertEqual(sha256_file(source["checkpoint"]), old_sha)
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            for op in OPS:
                self.assertEqual(report["fit_diagnostics"][op]["feature_rank"], 4)
                self.assertLess(report["fit_diagnostics"][op]["max_train_normalized_residual"], 1e-8)
                self.assertEqual(report["test"][op]["accuracy"], 1.0)
                for group in GROUPS:
                    self.assertEqual(report["ood"][group][op]["accuracy"], 1.0)
            saved, record = _load_calibrated(store, seed=31415)
            for key, original in old_weights.state_dict().items():
                if key.startswith(("left.", "right.")):
                    self.assertTrue(torch.equal(
                        saved.state_dict()[key], original.double()))
            for op in OPS:
                prediction = predict_calibration_lab(
                    store, 999, 998, op, seed=31415)
                expected = {"+": 1997, "-": 1, "*": 997002}[op]
                self.assertEqual(prediction["model_answer"], expected)
                self.assertEqual(prediction["source_checkpoint_sha256"], old_sha)
            store.close()
            reopened = Store(path)
            self.assertEqual(
                predict_calibration_lab(reopened, 999, 998, "*", seed=31415)["model_answer"],
                997002)
            reopened.close()

    def test_all_seeds_reported_without_ood_model_selection(self):
        from evolve.interaction_lab import run_interaction_lab
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            settings = self._cfg(path)
            for seed in (31415, 42, 1001):
                run_interaction_lab(store, settings, seed=seed,
                                    steps=120, rank=8, count=3)
            study = run_calibration_study(store, seeds=(31415, 42, 1001), count=5)
            self.assertEqual(study["status"], "evaluated")
            self.assertEqual(study["seeds"], [31415, 42, 1001])
            self.assertEqual(len(study["runs"]), 3)
            for group in GROUPS:
                for op in OPS:
                    scores = study["ood_summary"][group][op]
                    self.assertEqual(scores["seeds_evaluated"], 3)
                    self.assertEqual(scores["calibrated_mean"], 1.0)
            self.assertEqual(store.status()["documents"], 0)
            store.close()

    def test_calibrated_checkpoint_integrity(self):
        from evolve.interaction_lab import run_interaction_lab
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            run_interaction_lab(store, self._cfg(path),
                                steps=100, seed=31415, rank=8, count=3)
            report = run_calibration_lab(store, seed=31415, count=4)
            with open(report["checkpoint"], "ab") as output:
                output.write(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_calibration_lab(store, 9, 8, "+", seed=31415)
            store.close()


if __name__ == "__main__":
    unittest.main()
