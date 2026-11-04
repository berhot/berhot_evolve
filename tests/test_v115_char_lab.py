"""BERHOT v1.15: Unicode tokenizer/Transformer from real frozen data.

Offline synthetic training data only. Tests actual training, safe generation,
continuation, SHA verification and preservation of previous checkpoints.
"""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from evolve.brain import build_model, load_checkpoint, require_torch, sha256_file
from evolve.char_lab import (
    _load_char_model, _load_tokenizer, build_tokenizer,
    encode_chars, decode_chars, train_char_lab, predict_char_lab,
    evaluate_char_lab,
)
from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


def _setup(folder, hosts=("alpha.example.org",)):
    store = Store(folder)
    subjects = [
        ("Energy", "electric current heat kinetic motion solar panels"),
        ("Biology", "plants tissue ecology respiration cells forests"),
        ("Geography", "river mountains ocean altitude climate continents"),
    ]
    for i, host in enumerate(hosts):
        subject, words = subjects[i % 3]
        article = " ".join(
            f"{subject} scientific observation number {n} explores "
            f"{words} as examples of repeatable research methods, "
            "systematic evidence and educational explanation. "
            for n in range(210))
        archive_page(
            store, f"https://{host}/research/{subject.lower()}",
            f"<html><head><title>{subject}</title></head><body>{article}</body></html>",
            article, title=subject, declared_license=None)
    cfg = load_config()
    cfg["storage_root"] = folder
    cfg["train"]["learning_rate"] = 0.002
    cfg["train"]["max_seconds"] = 90
    return store, cfg


class CharacterTokenizerUnitTests(unittest.TestCase):
    def test_unicode_vocab_train_only_and_unk_is_explicit(self):
        tokenizer = build_tokenizer("Hello مرحبا عالم world\n".encode("utf-8"),
                                    vocab_limit=128)
        self.assertEqual(tokenizer["unk_id"], 0)
        ids = encode_chars("Hello مرحبا", tokenizer)
        self.assertTrue(all(id > 0 for id in ids))
        self.assertEqual(decode_chars(ids, tokenizer), "Hello مرحبا")
        self.assertEqual(encode_chars("🍏", tokenizer), [0])
        self.assertEqual(decode_chars([0], tokenizer), "\ufffd")
        self.assertNotIn("🍏", tokenizer["chars"])
        self.assertEqual(tokenizer["vocab_size"], len(tokenizer["chars"]) + 1)
        with self.assertRaises(ValueError):
            build_tokenizer(b"abc", vocab_limit=1)

    def test_legacy_byte_model_defaults_still_256_after_vocab_parameterization(self):
        if torch is None:
            self.skipTest("No PyTorch")
        default_cfg = dict(block_size=32, embedding_dim=16, layers=1,
                           heads=2, dropout=0.0)
        legacy = build_model(default_cfg)
        self.assertEqual(legacy.out.out_features, 256)
        char_model = build_model(dict(default_cfg, vocab_size=93))
        self.assertEqual(char_model.out.out_features, 93)
        self.assertEqual(char_model.token.num_embeddings, 93)


@unittest.skipIf(torch is None, "PyTorch required for actual neural steps")
class CharacterTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.options = dict(
            steps=14, seed=42, include_unknown=True,
            diagnostic=True, device="cpu", context=64,
            embedding_dim=32, layers=1, heads=2, vocab_limit=128)

    def test_all_stages_train_resume_re_evaluate_predict_with_original_registry_intact(self):
        with tempfile.TemporaryDirectory() as folder:
            store, config = _setup(folder)
            # A fake file is enough to check the old general model path
            # remains byte-for-byte intact: it is never deserialized.
            old_path = Path(folder) / "models" / "old_general.pt"
            old_path.parent.mkdir(exist_ok=True)
            old_path.write_bytes(b"original general model placeholder")
            old_sha = sha256_file(old_path)
            store.insert_model(old_path, old_sha, 32, 1.7, None, True)

            denied = train_char_lab(
                store, config, **{**self.options, "include_unknown": False})
            self.assertEqual(denied["status"], "skipped")
            strict = train_char_lab(
                store, config, **{**self.options, "diagnostic": False})
            self.assertEqual(strict["reason"], "strict_source_holdout_unavailable")
            first = train_char_lab(store, config, **self.options)
            self.assertEqual(first["status"], "trained")
            self.assertTrue(first["accepted_by_validation"])
            self.assertEqual(first["actual_steps"], 14)
            self.assertEqual(first["total_training_steps"], 14)
            self.assertIsNone(first["resumed_from_sha256"])
            self.assertTrue(first["diagnostic_source_leakage"])
            self.assertTrue(first["existing_general_checkpoint_unchanged"])
            self.assertEqual(sha256_file(old_path), old_sha)
            self.assertEqual(sha256_file(first["checkpoint"]), first["sha256"])
            self.assertEqual(first["tokenizer_owner"],
                             "BERHOT custom Python Unicode characters")
            self.assertGreater(first["vocab_size"], 2)
            self.assertTrue(math.isfinite(first["validation_char_loss_after"]))
            self.assertTrue(math.isfinite(first["heldout_test_char_loss_after"]))

            evaluation = evaluate_char_lab(store)
            self.assertEqual(evaluation["status"], "evaluated")
            self.assertEqual(evaluation["checkpoint_sha256"], first["sha256"])
            self.assertAlmostEqual(
                evaluation["character_cross_entropy_nats"]["test"],
                first["heldout_test_char_loss_after"], places=4)

            generated = predict_char_lab(
                store, "Energy is", new_chars=50, seed=17, top_k=20)
            self.assertEqual(generated["status"], "predicted")
            self.assertEqual(generated["checkpoint_sha256"], first["sha256"])
            self.assertTrue(generated["text"].startswith("Energy is"))
            self.assertEqual(len(generated["continuation"]), 50)
            self.assertNotIn("\ufffd", generated["continuation"])
            self.assertEqual(generated["continuation"].encode("utf-8").decode(
                "utf-8"), generated["continuation"])
            self.assertEqual(generated, predict_char_lab(
                store, "Energy is", new_chars=50, seed=17, top_k=20))
            self.assertEqual(predict_char_lab(
                store, "Energy is", new_chars=10, greedy=True)["decoding"],
                "greedy")

            bad_resume = train_char_lab(
                store, config, **{**self.options, "resume": True, "context": 128})
            self.assertEqual(
                bad_resume["reason"],
                "incompatible_previous_dataset_or_model_configuration")
            second = train_char_lab(
                store, config, **{**self.options, "resume": True, "steps": 6})
            self.assertEqual(second["status"], "trained")
            self.assertEqual(second["resumed_from_sha256"], first["sha256"])
            self.assertEqual(second["total_training_steps"], 20)
            self.assertEqual(first["tokenizer_sha256"],
                             second["tokenizer_sha256"])
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(sha256_file(old_path), old_sha)
            self.assertEqual(store.latest_promoted()["sha256"], old_sha)
            self.assertEqual(store.count("documents"), 0)
            self.assertEqual(store.status()["unique_cleaned_web_texts"], 1)

            # The accepted checkpoint remains valid after a separate candidate.
            assert _load_char_model(store, candidate=False) is not None
            tokenizer_file = Path(first["tokenizer_path"])
            tokenizer_file.write_text('{"tampered":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Tokenizer SHA256 mismatch"):
                evaluate_char_lab(store)
            store.close()

    def test_true_host_holdout_and_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            store, config = _setup(
                folder, ("alpha.example.org", "beta.example.net",
                         "gamma.example.edu"))
            opts = dict(self.options, diagnostic=False, steps=5)
            report = train_char_lab(store, config, **opts)
            self.assertEqual(report["status"], "trained")
            self.assertFalse(report["diagnostic_source_leakage"])
            self.assertEqual(report["split_method"], "heldout_source_hosts")
            self.assertFalse(report["test_used_for_selection"])
            self.assertTrue(math.isfinite(report["unigram_validation_char_loss"]))
            self.assertTrue(math.isfinite(report["heldout_test_char_loss_after"]))
            self.assertIn("unknown_character_rates", report)
            store.close()


if __name__ == "__main__":
    unittest.main()
