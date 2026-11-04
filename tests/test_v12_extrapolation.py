"""v1.2: frozen-checkpoint extrapolation; no OOD training or model promotion."""
import tempfile
import unittest

from evolve.extrapolation import (
    RANGES, select_ood_pairs, magnitude_extrapolate,
    magnitude_extrapolation_study,
)
from evolve.numeric_lab import split_addition_pairs
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class ExtrapolationContracts(unittest.TestCase):
    def test_all_probe_problems_are_outside_training_domain(self):
        train = {tuple(pair) for pair in split_addition_pairs(31415)["train"]}
        for name in RANGES:
            a = select_ood_pairs(name, count=20, seed=20261104)
            b = select_ood_pairs(name, count=20, seed=20261104)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 20)
            self.assertEqual(len(set(a)), 20)
            self.assertTrue(all(max(pair) >= 10 for pair in a))
            self.assertFalse(set(a) & train)
            if name == "both_20_to_29":
                self.assertTrue(all(min(pair) >= 20 for pair in a))
            if name == "both_10_to_19":
                self.assertTrue(all(10 <= min(pair) and max(pair) <= 19 for pair in a))

    def test_validate_bad_bounds_and_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            select_ood_pairs("both_10_to_19", count=51)
        with self.assertRaises(ValueError):
            select_ood_pairs("made-up-group", count=10)
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            with self.assertRaises(ValueError):
                magnitude_extrapolation_study(store, model_seeds=(42, 42))
            with self.assertRaises(ValueError):
                magnitude_extrapolation_study(store, model_seeds=(42,))
            missing = magnitude_extrapolate(store, model_seed=42)
            self.assertEqual(missing["status"], "skipped")
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class FrozenCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_frozen_audit_is_independent_of_true_answer_at_inference(self):
        from evolve.magnitude_lab import _build_model, FORMAT
        from evolve.brain import sha256_file
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            path = Path(folder) / "models" / "magnitude_lab" / "test-zero.pt"
            path.parent.mkdir(parents=True)
            model = _build_model(32)
            with torch.no_grad():
                for value in model.parameters():
                    value.zero_()
            torch.save({
                "format": FORMAT,
                "weights": model.state_dict(),
                "hidden": 32, "seed": 31415,
                "splits": split_addition_pairs(31415),
            }, path)
            digest = sha256_file(path)
            store.put("magnitude_lab_latest_candidate", {
                "path": str(path), "sha256": digest,
                "seed": 31415, "hidden": 32, "accepted": True,
            })
            report = magnitude_extrapolate(store, count=12)
            self.assertEqual(report["status"], "evaluated")
            self.assertFalse(report["weights_updated"])
            self.assertEqual(report["checkpoint_sha256"], digest)
            for group in RANGES:
                self.assertEqual(report["results"][group]["questions"], 12)
                self.assertEqual(report["results"][group]["correct"], 0)
                self.assertTrue(report["results"][group]["mean_absolute_error"] > 0)
            self.assertEqual(sha256_file(path), digest)
            self.assertIsNone(store.latest_promoted())
            store.close()

    def test_real_trained_weights_can_be_audited_for_multiple_existing_seeds(self):
        from evolve.magnitude_lab import run_magnitude_lab
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            settings = load_config()
            settings["storage_root"] = folder
            settings["train"]["device"] = "cpu"
            settings["train"]["max_seconds"] = 120
            first = run_magnitude_lab(store, settings, steps=250, seed=31415)
            second = run_magnitude_lab(store, settings, steps=250, seed=42)
            self.assertTrue(store.get("magnitude_lab_seed_31415"))
            self.assertTrue(store.get("magnitude_lab_seed_42"))
            before_documents = store.status()["documents"]

            study = magnitude_extrapolation_study(
                store, model_seeds=(31415, 42), count=10, probe_seed=999)
            self.assertEqual(study["status"], "evaluated")
            self.assertEqual(study["seeds_requested"], [31415, 42])
            self.assertEqual(len(study["models"]), 2)
            for group in RANGES:
                agg = study["aggregate"][group]
                self.assertEqual(agg["models_evaluated"], 2)
                self.assertEqual(agg["total_questions"], 20)
                self.assertTrue(0 <= agg["min_exact_accuracy"] <= 1)
                self.assertTrue(0 <= agg["max_exact_accuracy"] <= 1)
            self.assertEqual(sha256_file(first["checkpoint"]), first["sha256"])
            self.assertEqual(sha256_file(second["checkpoint"]), second["sha256"])
            self.assertEqual(store.status()["documents"], before_documents)
            self.assertIsNone(store.latest_promoted())
            store.close()

    def test_rejected_or_modified_registered_checkpoint_never_passes_hash(self):
        from evolve.magnitude_lab import run_magnitude_lab
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            settings = load_config()
            settings["storage_root"] = folder
            settings["train"]["device"] = "cpu"
            output = run_magnitude_lab(store, settings, steps=100, seed=1001)
            with open(output["checkpoint"], "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                magnitude_extrapolate(store, model_seed=1001)
            store.close()


if __name__ == "__main__":
    unittest.main()
