"""v1.6: audit exact bilinear coefficient structure and fit a strong supervised prior."""
import tempfile
import unittest

from evolve.basis_lab import (
    BASIS, GROUPS, OPS, _build_model, _effective_v15_coefficients,
    _fit_from_train_pairs, _ood_pairs, _score, _reference_value,
    audit_interaction_coefficients, run_basis_lab, run_basis_study,
    predict_basis_lab,
)
from evolve.numeric_lab import split_addition_pairs
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class BasisContracts(unittest.TestCase):
    def test_grouped_training_split_does_not_contain_evaluation_problems(self):
        for seed in (31415, 42, 1001):
            split = split_addition_pairs(seed)
            training = set(split["train"])
            self.assertFalse(training & set(split["validation"]))
            self.assertFalse(training & set(split["test"]))
            for group in GROUPS:
                sample = _ood_pairs(group, 12, seed=20261104)
                self.assertEqual(len(sample), 12)
                self.assertEqual(len(set(sample)), 12)
                self.assertTrue(all(max(a, b) >= 10 for a, b in sample))
                self.assertFalse(set(sample) & training)
        self.assertEqual(len(BASIS), 4)

    def test_evaluation_oracle_handles_large_ood_without_widening_practice_api(self):
        from evolve.practice import checked_answer
        self.assertEqual(_reference_value(999, 998, "+"), 1997)
        self.assertEqual(_reference_value(999, 998, "-"), 1)
        self.assertEqual(_reference_value(999, 998, "*"), 997002)
        self.assertEqual(_reference_value(100, 199, "-"), -99)
        with self.assertRaises(ValueError):
            _reference_value(1000, 998, "*")
        with self.assertRaises(ValueError):
            checked_answer(999, 998, "*")  # production restriction intact

    def test_seed_and_input_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(audit_interaction_coefficients(store)["status"], "skipped")
            self.assertEqual(predict_basis_lab(store, 2, 3, "+")["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_basis_lab(store, 2, 3, "/")
            with self.assertRaises(ValueError):
                predict_basis_lab(store, -1, 4, "+")
            with self.assertRaises(ValueError):
                run_basis_study(store, seeds=(42, 42))
            with self.assertRaises(ValueError):
                _ood_pairs("bad-group", 4)
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class SupervisedBasisNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_weight_neural_head_cannot_invent_math_answers(self):
        model = _build_model()
        with torch.no_grad():
            model.head.weight.zero_()
        predictions = _score(model, [(27, 28, "+"), (27, 28, "-"), (27, 28, "*")])
        self.assertEqual(predictions["+"]["correct"], 0)
        self.assertEqual(predictions["-"]["correct"], 0)
        self.assertEqual(predictions["*"]["correct"], 0)

    def test_fit_recovers_coefficients_and_extrapolates_without_ood_training(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            report = run_basis_lab(store, seed=31415, count=12)
            self.assertEqual(report["status"], "fitted")
            self.assertTrue(report["accepted"])
            self.assertEqual((report["train_groups"], report["validation_groups"],
                              report["test_groups"]), (35, 10, 10))
            for operator in OPS:
                self.assertEqual(report["in_range_test"][operator]["accuracy"], 1.0)
                for group in GROUPS:
                    result = report["ood"][group][operator]
                    self.assertEqual(result["accuracy"], 1.0)
                    self.assertEqual(result["correct"], result["questions"])
            self.assertEqual(report["frozen_v15_baseline"]["status"],
                             "no_matching_seed_v15_checkpoint")
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            store.close()
            reopened = Store(folder)
            for op in OPS:
                predicted = predict_basis_lab(reopened, 999, 998, op, seed=31415)
                answers = {"+": 1997, "-": 1, "*": 997002}
                self.assertEqual(predicted["model_answer"], answers[op])
            reopened.close()

    def test_effective_coefficients_match_actual_v15_predictions(self):
        from evolve.interaction_lab import _build_model as v15_model
        torch.manual_seed(4321)
        model = v15_model(8)
        model.eval()
        effective = _effective_v15_coefficients(model)
        for a, b in ((0, 9), (7, 3), (27, 28), (99, 87)):
            x, y = a / 9.0, b / 9.0
            features = [1.0, x, y, x * y]
            for idx, op in enumerate(OPS):
                with torch.no_grad():
                    actual = float(model(
                        torch.tensor([a]), torch.tensor([b]),
                        torch.tensor([idx])).item())
                expanded = sum(v * w for v, w in zip(features, effective[op]))
                self.assertAlmostEqual(actual, expanded, delta=max(1e-5,abs(actual)*1e-5))

    def test_audit_v15_read_only_and_report_error_amplification(self):
        from evolve.interaction_lab import run_interaction_lab
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            settings = load_config()
            settings["storage_root"] = folder
            settings["train"]["device"] = "cpu"
            settings["train"]["max_seconds"] = 120
            prior = run_interaction_lab(store, settings, seed=31415,
                                        steps=220, rank=4, count=3)
            hash_before = sha256_file(prior["checkpoint"])
            registry_before = store.get("interaction_lab_seed_31415")
            diagnosis = audit_interaction_coefficients(store, seed=31415,
                                                       a=27, b=28)
            self.assertEqual(diagnosis["status"], "audited")
            self.assertFalse(diagnosis["weights_updated"])
            self.assertEqual(diagnosis["checkpoint"], prior["sha256"])
            for op in OPS:
                coef = diagnosis["operations"][op]["coefficients"]
                err = diagnosis["operations"][op]["coefficient_errors"]
                self.assertEqual(set(coef), set(BASIS))
                self.assertEqual(set(err), set(BASIS))
            result = run_basis_lab(store, seed=31415, count=5)
            self.assertEqual(result["frozen_v15_baseline"]["status"],
                             "comparable_read_only")
            self.assertEqual(store.get("interaction_lab_seed_31415"), registry_before)
            self.assertEqual(sha256_file(prior["checkpoint"]), hash_before)
            self.assertIsNone(store.latest_promoted())
            store.close()

    def test_multiple_seeds_and_tamper_rejection(self):
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            report = run_basis_study(store, seeds=(31415, 42, 1001), count=5)
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(len(report["results"]), 3)
            for group in GROUPS:
                for op in OPS:
                    stats = report["ood_summary"][group][op]
                    self.assertEqual(stats["mean"], 1.0)
                    self.assertEqual(stats["min"], 1.0)
                    self.assertEqual(stats["max"], 1.0)
            newest = store.get("basis_lab_seed_31415")
            self.assertEqual(sha256_file(newest["path"]), newest["sha256"])
            with open(newest["path"], "ab") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_basis_lab(store, 4, 3, "+", seed=31415)
            self.assertEqual(store.status()["documents"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
