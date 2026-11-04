"""v1.17: offline, fixture-only BPE, source holdout and checkpoint invariants."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from evolve.byte_bpe import (fit_byte_bpe, encode_bpe, decode_bpe,
                             validate_bpe, token_bytes, digest)
from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v117 import plan, prepare, train, evaluate, smoke, quality_proxy

try:
    import torch
except ImportError:
    torch = None


def seed_authored(store):
    subjects = (
        ("physics", "magnetic thermal electrical kinetic mechanical solar"),
        ("biology", "orchids flowers chlorophyll roots fungi tissue"),
        ("geography", "plateaus dunes oceans rainfall erosion longitude"),
    )
    for i, (title, terms) in enumerate(subjects):
        tokens = terms.split()
        segments = []
        for n in range(160):
            rotated = tokens[n % 6:] + tokens[:n % 6]
            segments.append(
                f"{title} observation {n} measures " +
                " ".join(rotated) +
                f" using sample {n*n+7} and instrument {n+3}. " +
                f"Written evidence {n+17} supports cautious investigation "
                "of variations and alternative explanations.")
        article = " ".join(segments)
        archive_page(
            store, f"https://synthetic-{i}.example.org/articles/{title}",
            "<html><title>" + title + "</title><body>" + article + "</body></html>",
            article, title=title, declared_license="CC0-1.0",
            rights_url="fixture:self-authored")


class ByteBpeTests(unittest.TestCase):
    def test_train_only_byte_bpe_roundtrip_and_determinism(self):
        raw = (("Hello مرحبا 🙂 你好 world. " * 350).encode("utf-8"))
        tok = fit_byte_bpe(raw, vocab_size=288, sample_bytes=4000)
        self.assertEqual(tok, fit_byte_bpe(raw, vocab_size=288,
                                            sample_bytes=4000))
        validate_bpe(tok)
        sample = "مرحبا 🦊 new unseen glyphs مرحبا.".encode("utf-8")
        ids = encode_bpe(sample, tok)
        self.assertEqual(decode_bpe(ids, tok), sample)
        self.assertEqual(sum(len(token_bytes(tok)[i]) for i in ids),
                         len(sample))
        self.assertEqual(tok["train_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertTrue(tok["vocab_size"] > 256)
        self.assertNotIn("test_bytes", tok)
        self.assertEqual(len(digest(tok)), 64)

    def test_invalid_merge_chain_rejected(self):
        tok = fit_byte_bpe(b"abracadabra " * 400, vocab_size=270)
        tok["merges"][0] = [4000, 1]
        with self.assertRaisesRegex(ValueError, "forward"):
            validate_bpe(tok)

    def test_quality_is_explicitly_nonsemantic(self):
        stats = quality_proxy("lalalalalala is is is !!!")
        self.assertTrue(stats["human_coherence_required"])
        self.assertGreater(stats["repeated_3gram_fraction"], 0)


class StrictHostTests(unittest.TestCase):
    def test_one_host_does_not_claim_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                article = ("The documented energy experiment describes " +
                           "measurements of temperature, time and velocity. ") * 250
                archive_page(
                    store, "https://single.example.org/articles/energy",
                    "<html>" + article + "</html>", article,
                    title="Energy", declared_license="CC0-1.0")
                result = prepare(store)
                self.assertEqual(result["status"], "skipped")
                self.assertLess(result["plan"]["eligible_article_count"], 3)
            finally:
                store.close()

    def test_strict_host_split_has_distinct_hosts_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                seed_authored(store)
                result = prepare(store)
                self.assertEqual(result["status"], "prepared", result)
                self.assertFalse(result["diagnostic_source_leakage"])
                hosts = result["source_hosts"]
                self.assertEqual(set(hosts), {"train", "val", "test"})
                self.assertEqual(len(set(sum(hosts.values(), []))), 3)
                self.assertEqual(len(set(result["shard_sha256"].values())), 3)
                self.assertEqual(plan(store)["status"], "ready_for_strict_freeze")
            finally:
                store.close()


@unittest.skipIf(torch is None, "CPU PyTorch required")
class NeuralBenchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_smoke_is_synthetic_but_runs_both_real_gradient_models(self):
        cfg = load_config()
        report = smoke(cfg, steps=5)
        self.assertEqual(report["status"], "smoke_test_complete", report)
        self.assertTrue(report["synthetic_fixture_only"])
        self.assertTrue(report["strict_host_holdout"])
        self.assertFalse(report["real_world_generalization_demonstrated"])
        self.assertTrue(report["training"]["previous_checkpoints_unchanged"])
        self.assertEqual(report["evaluation"]["status"], "evaluated")
        models = report["evaluation"]["models"]
        self.assertEqual(set(models), {"char", "byte_bpe"})
        for kind in models:
            self.assertEqual(models[kind]["actual_training_steps"], 5)
            self.assertEqual(len(models[kind]["unseen_prompt_probes"]), 3)
            self.assertGreater(
                models[kind]["scores"]["val"]["utf8_target_bytes"], 500)

    def test_real_checkpoint_isolation_and_sha_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = folder
            cfg["train"]["max_seconds"] = 120
            try:
                seed_authored(store)
                legacy_path = Path(folder) / "models" / "char_lab" / "old.pt"
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path.write_bytes(b"immutable old checkpoint fixture")
                legacy_sha = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
                legacy_record = {"checkpoint": str(legacy_path), "sha256": legacy_sha}
                store.put("char_lab_latest", legacy_record)
                result = train(store, cfg, steps=5)
                self.assertEqual(result["status"], "trained", result)
                self.assertEqual(legacy_path.read_bytes(), b"immutable old checkpoint fixture")
                self.assertEqual(store.get("char_lab_latest"), legacy_record)
                self.assertEqual(len(store.get("v117_history")), 1)
                evaluated = evaluate(store)
                self.assertEqual(evaluated["status"], "evaluated")
                self.assertTrue(all(
                    info["scores"]["test"]["input_prefix_sha256"] ==
                    evaluated["models"]["char"]["scores"]["test"]["input_prefix_sha256"]
                    for info in evaluated["models"].values()))
                bpe = Path(result["candidates"]["byte_bpe"]["checkpoint"])
                with bpe.open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    evaluate(store)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
