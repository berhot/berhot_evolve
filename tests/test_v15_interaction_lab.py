"""v1.5: honest learned-interaction evaluation across three supervised operations."""
import tempfile
import unittest

from evolve.interaction_lab import (
    OOD_GROUPS, OPS, _build_model, _ood_pairs,
    predict_interaction_lab, run_interaction_lab, run_interaction_study,
)
from evolve.multiop_lab import _examples
from evolve.numeric_lab import split_addition_pairs
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class InteractionContracts(unittest.TestCase):
    def test_fresh_ood_groups_are_unseen_and_reproducible(self):
        seen = {
            tuple(sorted(pair))
            for section in split_addition_pairs(31415).values()
            for pair in section
        }
        for group in OOD_GROUPS:
            x = _ood_pairs(group, count=12, seed=20261104)
            y = _ood_pairs(group, count=12, seed=20261104)
            self.assertEqual(x, y)
            self.assertEqual(len(x), 12)
            self.assertEqual(len(set(x)), 12)
            self.assertFalse(set(x) & seen)
            if group == "both_30_to_49":
                self.assertTrue(all(30 <= a <= b <= 49 for a, b in x))
            if group == "both_50_to_99":
                self.assertTrue(all(50 <= a <= b <= 99 for a, b in x))
        with self.assertRaises(ValueError):
            _ood_pairs("invalid", count=10, seed=1)

    def test_forward_reverse_pairs_and_operators_cannot_leak(self):
        groups = split_addition_pairs(31415)
        indices = {}
        for split_name, pairs in groups.items():
            mappings = {}
            for a, b, op in _examples(pairs):
                key = tuple(sorted((a, b)))
                mappings.setdefault(key, set()).add(op)
            self.assertTrue(all(ops == set(OPS) for ops in mappings.values()))
            indices[split_name] = set(mappings)
        self.assertFalse(indices["train"] & indices["validation"])
        self.assertFalse(indices["train"] & indices["test"])
        self.assertFalse(indices["validation"] & indices["test"])

    def test_inference_validation_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(predict_interaction_lab(store, 1, 2, "+")["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_interaction_lab(store, 1, 2, "/")
            with self.assertRaises(ValueError):
                predict_interaction_lab(store, -1, 2, "+")
            with self.assertRaises(ValueError):
                predict_interaction_lab(store, 100, 2, "*")
            with self.assertRaises(ValueError):
                run_interaction_study(store, load_config(), seeds=(42, 42))
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class InteractionNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def cfg(self, folder):
        cfg = load_config()
        cfg["storage_root"] = folder
        cfg["train"]["device"] = "cpu"
        cfg["train"]["max_seconds"] = 120
        return cfg

    def test_zero_weights_cannot_secretly_return_calculator_answers(self):
        model = _build_model(rank=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for op_idx in range(3):
                prediction = model(torch.tensor([7]), torch.tensor([8]),
                                   torch.tensor([op_idx])).item()
                self.assertEqual(prediction, 0.0)

    def test_hidden_bilinear_gradients_can_be_learned(self):
        torch.manual_seed(99)
        model = _build_model(rank=4)
        x = torch.tensor([2, 6])
        y = torch.tensor([5, 3])
        op = torch.tensor([2, 2])
        loss = model(x, y, op).square().sum()
        loss.backward()
        self.assertIsNotNone(model.left.weight.grad)
        self.assertIsNotNone(model.right.weight.grad)
        self.assertGreater(float(model.left.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.right.weight.grad.abs().sum()), 0.0)

    def test_same_seed_reference_checkpoint_integrity_and_model_reload(self):
        from evolve.multiop_lab import run_multiop_lab
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = self.cfg(folder)
            old = run_multiop_lab(store, cfg, steps=140, seed=31415,
                                  hidden=24, count=5)
            old_ref = store.get("multiop_lab_seed_31415")
            old_hash = sha256_file(old["checkpoint"])
            experiment = run_interaction_lab(store, cfg, steps=320,
                rank=6, seed=31415, count=5)
            self.assertEqual(experiment["status"], "trained")
            self.assertIsInstance(experiment["accepted"], bool)
            self.assertEqual(experiment["train_groups"], 35)
            self.assertEqual(experiment["validation_groups"], 10)
            self.assertEqual(experiment["test_groups"], 10)
            self.assertEqual(set(experiment["in_range_test"]), set(OPS))
            self.assertEqual(set(experiment["ood"]), set(OOD_GROUPS))
            self.assertEqual(experiment["frozen_v14_reference"]["status"], "comparable")
            self.assertEqual(store.get("multiop_lab_seed_31415"), old_ref)
            self.assertEqual(sha256_file(old["checkpoint"]), old_hash)
            self.assertEqual(store.status()["documents"], 0)
            self.assertIsNone(store.latest_promoted())
            for group in OOD_GROUPS:
                for op in OPS:
                    trial = experiment["ood"][group][op]
                    self.assertTrue(0 <= trial["accuracy"] <= 1)
                    self.assertTrue(trial["questions"] >= 5)
            for operator in OPS:
                guess = predict_interaction_lab(store, 27, 28, operator, seed=31415)
                self.assertEqual(guess["status"], "predicted")
                self.assertIn("model_answer", guess)
                self.assertEqual(guess["checkpoint_status"],
                    "accepted" if experiment["accepted"] else "unaccepted_candidate")
            store.close()
            reopened = Store(folder)
            for operator in OPS:
                a = predict_interaction_lab(reopened, 27, 28, operator, seed=31415)
                b = predict_interaction_lab(reopened, 27, 28, operator, seed=31415)
                self.assertEqual(a["model_answer"], b["model_answer"])
            reopened.close()

    def test_multi_seed_reports_every_operator_even_if_accuracy_is_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            report = run_interaction_study(
                store, self.cfg(folder), seeds=(31415, 42),
                steps=160, rank=4, count=3)
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["seeds"], [31415, 42])
            self.assertEqual(len(report["runs"]), 2)
            for row in report["runs"]:
                self.assertIsInstance(row["accepted"], bool)
                self.assertEqual(set(row["in_range_accuracy"]), set(OPS))
                self.assertEqual(set(row["ood_accuracy"]), set(OOD_GROUPS))
            for group in OOD_GROUPS:
                for op in OPS:
                    stat = report["ood_summary"][group][op]
                    self.assertTrue(0 <= stat["min"] <= stat["mean"] <= stat["max"] <= 1)
            self.assertEqual(store.status()["documents"], 0)
            store.close()

    def test_tampered_interaction_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            output = run_interaction_lab(store, self.cfg(folder),
                                         steps=80, rank=4, count=3)
            with open(output["checkpoint"], "ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_interaction_lab(store, 1, 2, "+", seed=31415)
            store.close()


if __name__ == "__main__":
    unittest.main()
