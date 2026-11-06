"""v1.13 full corpus -> frozen dataset -> real neural checkpoint smoke test.

Tests use locally authored synthetic text and NO network access. The actual
user's collected 4-page archive and checkpoints remain untouched.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

from evolve.brain import sha256_file
from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_lab import (
    _chunks, _remove_near_copies, web_inventory, freeze_dataset,
    _read_frozen, train_web_lab, evaluate_web_lab, predict_web_lab,
)

try:
    import torch
except ImportError:
    torch = None


def make_text(subject, salt, count=220):
    return " ".join(
        f"The {subject} field studies measurement, evidence, and scientific "
        f"processes using clear observations, distinct sample {salt}-{i}, "
        f"independent investigations, and careful educational research."
        for i in range(count)
    )


def make_store(folder, hosts=("alpha.example.org",)):
    store = Store(folder)
    for i, host in enumerate(hosts):
        subject = ["Energy", "Volcano", "River", "Science", "Physics"][i % 5]
        text = make_text(subject, i)
        archive_page(
            store, f"https://{host}/articles/{subject.lower()}",
            f"<html><head><title>{subject}</title></head><body><p>{text}</p></body></html>",
            text, title=subject, declared_license=None)
    # This is long but should be excluded as directory/navigation, not learnable text.
    archive_page(store, "https://alpha.example.org/wiki/Main_Page",
                 "<html><body><p>Navigation content</p></body></html>",
                 make_text("Directory", 99), title="Main Page")
    return store


def settings(folder):
    cfg = load_config()
    cfg["storage_root"] = folder
    cfg["train"]["learning_rate"] = 0.001
    cfg["train"]["max_seconds"] = 60
    return cfg


class CorpusCurationAndFreezeTests(unittest.TestCase):
    def test_indexed_near_copy_filter_preserves_jaccard_decisions(self):
        import copy
        import random
        from evolve.web_lab import _shingles
        rng = random.Random(42)
        texts = [" ".join(str(rng.randrange(20)) for _ in range(80)) for _ in range(35)]
        texts += [texts[0], texts[1] + " tail", "", "one two three", texts[2].replace("1", "x", 1)]
        for threshold in (0, .5, .86, 1):
            actual = [{"chunks": [{"text": text} for text in texts]}]
            expected = copy.deepcopy(actual)
            kept, dropped = [], 0
            filtered = []
            for chunk in expected[0]["chunks"]:
                shingles = _shingles(chunk["text"])
                near = bool(shingles) and any(
                    len(shingles & prior) / len(shingles | prior) >= threshold
                    for prior in kept)
                if near:
                    dropped += 1
                else:
                    kept.append(shingles)
                    filtered.append(chunk)
            expected[0]["chunks"] = filtered
            self.assertEqual(_remove_near_copies(actual, threshold), dropped)
            self.assertEqual(actual, expected)

    def test_no_uninformed_rights_training_and_navigation_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            store = make_store(folder)
            unacknowledged = web_inventory(store)
            self.assertEqual(unacknowledged["candidate_article_count"], 0)
            self.assertGreaterEqual(
                unacknowledged["excluded_by_reason"][
                    "unknown_rights_requires_explicit_research_opt_in"], 1)
            inventory = web_inventory(store, include_unknown=True)
            self.assertEqual(inventory["candidate_article_count"], 1)
            self.assertEqual(inventory["candidate_hosts"], 1)
            self.assertGreaterEqual(inventory["candidate_chunks"], 12)
            self.assertEqual(inventory["excluded_by_reason"]["navigation_or_dynamic_index"], 1)
            self.assertTrue(inventory["unknown_rights_opted_in"])
            strict = freeze_dataset(store, include_unknown=True)
            self.assertEqual(strict["status"], "skipped")
            self.assertEqual(strict["reason"], "strict_source_holdout_unavailable")
            store.close()

    def test_within_document_explicit_diagnostic_is_frozen_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            store = make_store(folder)
            first = freeze_dataset(store, include_unknown=True, diagnostic=True)
            self.assertEqual(first["status"], "frozen")
            self.assertTrue(first["diagnostic_source_leakage"])
            self.assertEqual(first["split_method"],
                             "within_document_diagnostic_not_independent")
            self.assertEqual(first["dataset_sha256"],
                             freeze_dataset(store, include_unknown=True,
                                            diagnostic=True)["dataset_sha256"])
            meta, streams = _read_frozen(store, first["dataset_sha256"])
            self.assertTrue(all(len(streams[p]) > 514 for p in streams))
            ids = {name: set(meta["partitions"][name]["chunk_sha256"])
                   for name in ("train", "val", "test")}
            self.assertFalse(ids["train"] & ids["val"])
            self.assertFalse(ids["val"] & ids["test"])
            self.assertFalse(ids["test"] & ids["train"])
            # No mutation of existing archive is needed for training prep.
            self.assertEqual(store.count("documents"), 0)
            self.assertEqual(store.status()["unique_cleaned_web_texts"], 2)
            # Tampering is detected before training/evaluation.
            test_file = Path(first["dataset_dir"]) / "test.bin"
            test_file.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                _read_frozen(store, first["dataset_sha256"])
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze_dataset(store, include_unknown=True, diagnostic=True)
            store.close()

    def test_true_source_host_holdout_uses_distinct_hosts(self):
        with tempfile.TemporaryDirectory() as folder:
            store = make_store(folder, (
                "alpha.example.org", "beta.example.net", "gamma.example.edu"))
            result = freeze_dataset(store, include_unknown=True)
            self.assertEqual(result["status"], "frozen")
            self.assertEqual(result["split_method"], "heldout_source_hosts")
            self.assertFalse(result["diagnostic_source_leakage"])
            host_sets = {
                split: set(p["source_hosts"]) for split, p in result["partitions"].items()
            }
            self.assertTrue(all(len(s) == 1 for s in host_sets.values()))
            self.assertFalse(host_sets["train"] & host_sets["test"])
            self.assertFalse(host_sets["val"] & host_sets["train"])
            self.assertFalse(host_sets["val"] & host_sets["test"])
            self.assertEqual(result["candidate_articles"], 3)
            store.close()

    def test_near_duplicate_chunk_elimination(self):
        same = ("A sequence of carefully measured scientific "
                "examples supports consistent observations and facts. " * 15)
        chunks = _chunks(same + " Next sequence with extra words.", target=640)
        left = {"chunks": chunks}
        right = {"chunks": [dict(chunk) for chunk in chunks]}
        removed = _remove_near_copies([left, right])
        self.assertGreaterEqual(removed, 1)
        self.assertEqual(len(right["chunks"]), 0)


@unittest.skipIf(torch is None, "PyTorch is required for neural experiment")
class RealWebNeuralTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_complete_one_shot_train_evaluate_predict_and_preserve_base_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            store = make_store(folder)
            cfg = settings(folder)
            denied = train_web_lab(store, cfg, steps=4)
            self.assertEqual(denied["status"], "skipped")
            strict = train_web_lab(
                store, cfg, steps=4, include_unknown=True, diagnostic=False)
            self.assertEqual(strict["reason"], "strict_source_holdout_unavailable")
            result = train_web_lab(
                store, cfg, steps=8, seed=777,
                include_unknown=True, diagnostic=True, device="cpu")
            self.assertEqual(result["status"], "trained")
            self.assertEqual(result["actual_steps"], 8)
            self.assertEqual(result["model_backend"], "PyTorch byte-level causal Transformer")
            self.assertTrue(result["diagnostic_source_leakage"])
            self.assertFalse(result["test_used_for_checkpoint_selection"])
            self.assertFalse(result["trained_on_verified_facts"])
            self.assertFalse(result["arithmetic_accuracy_tested"])
            self.assertTrue(math.isfinite(result["heldout_test_byte_loss_after"]))
            self.assertGreater(result["train_byte_loss_before"], 0)
            self.assertGreater(result["train_byte_loss_after"], 0)
            self.assertEqual(sha256_file(result["checkpoint"]), result["sha256"])
            self.assertIsNone(store.latest_promoted())
            self.assertIsNotNone(store.get("web_lab_latest_candidate"))
            outcome = evaluate_web_lab(store)
            self.assertEqual(outcome["status"], "evaluated")
            self.assertEqual(outcome["dataset_sha256"], result["dataset_sha256"])
            self.assertTrue(outcome["diagnostic_source_leakage"])
            self.assertAlmostEqual(outcome["loss"]["test"],
                                   result["heldout_test_byte_loss_after"], places=4)
            pred = predict_web_lab(store, "Energy is", max_new_bytes=12)
            self.assertEqual(pred["status"], "predicted")
            self.assertEqual(pred["checkpoint_sha256"], result["sha256"])
            self.assertEqual(pred["method"], "checkpoint_weights_only_no_source_lookup")
            self.assertTrue(pred["neural_generation"].startswith("Energy is"))
            self.assertEqual(store.status()["unique_cleaned_web_texts"], 2)
            self.assertEqual(store.count("documents"), 0)
            with self.assertRaises(ValueError):
                predict_web_lab(store, "Energy", max_new_bytes=301)
            checkpoint = Path(result["checkpoint"])
            checkpoint.write_bytes(b"malicious checkpoint content")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_web_lab(store, "Energy is", max_new_bytes=4)
            store.close()

    def test_real_source_host_holdout_neural_train(self):
        with tempfile.TemporaryDirectory() as folder:
            store = make_store(folder, (
                "alpha.example.org", "beta.example.net", "gamma.example.edu"))
            cfg = settings(folder)
            report = train_web_lab(
                store, cfg, steps=5, seed=17, include_unknown=True, diagnostic=False)
            self.assertEqual(report["status"], "trained")
            self.assertFalse(report["diagnostic_source_leakage"])
            self.assertEqual(report["split_method"], "heldout_source_hosts")
            self.assertTrue(math.isfinite(report["validation_byte_loss_after"]))
            self.assertTrue(math.isfinite(report["heldout_test_byte_loss_after"]))
            store.close()


if __name__ == "__main__":
    unittest.main()
