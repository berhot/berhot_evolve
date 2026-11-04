"""v1.9: actual licensed-corroborated text -> saved neural weights -> unseen QA."""
import tempfile
import unittest
from unittest.mock import patch

from evolve.concept_lab import (
    _example, _inventory, TRAIN_TEMPLATES, TEST_TEMPLATES,
    concept_inventory, evaluate_concept_lab, predict_concept_lab,
    train_concept_lab,
)
from evolve.research import research_questions
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None

SOURCE_A = "https://alpha.example.org/definitions"
SOURCE_B = "https://beta.other.edu/reference"
LICENSE = '<link rel="license" href="https://creativecommons.org/publicdomain/zero/1.0/">'
DEFINITIONS = {
    "alphabet": "a collection of letters used to represent words in language",
    "volcano": "an opening in the earth that releases hot rock and gas",
    "compass": "a navigation tool that shows directions using a magnetic needle",
    "thermometer": "a measuring instrument that indicates temperature in objects",
    "river": "a natural stream of flowing water moving across the land",
}


def page(definitions, license_markup=LICENSE):
    clauses = "".join(
        f"<p>An {concept} is {meaning}.</p>"
        for concept, meaning in definitions.items()
    )
    return f"<html><head>{license_markup}</head><body>{clauses}</body></html>"


def config(folder):
    cfg = load_config()
    cfg.update(storage_root=folder, daily_request_budget=100,
               crawl_delay_seconds=0)
    cfg["research"] = {
        "questions_per_cycle": 20,
        "max_pages_per_question": 3,
        "retry_cooldown_seconds": 0,
    }
    cfg["model"] = {
        "block_size": 128, "embedding_dim": 16,
        "heads": 2, "layers": 1, "dropout": 0.0,
    }
    cfg["train"].update(seed=42, device="cpu", steps=30,
                         batch_size=4, eval_batches=1,
                         learning_rate=0.003, max_seconds=90,
                         min_improvement=0.0001)
    return cfg


def populate(store, cfg, names, license_markup=LICENSE):
    store.save_observation(SOURCE_A, "Definitions one", "Independent source one",
                           names, "research_only")
    store.save_observation(SOURCE_B, "Definitions two", "Independent source two",
                           names, "research_only")
    for name in names:
        store.add_research_question(name, f"What is {name}?", SOURCE_A)
    pages = {
        SOURCE_A: page(DEFINITIONS, license_markup),
        SOURCE_B: page(DEFINITIONS, ""),
    }
    return research_questions(store, cfg, limit=len(names),
                              fetcher=lambda url: pages[url],
                              sleep_fn=lambda seconds: None)


class VerifiedTextContracts(unittest.TestCase):
    def test_masked_answer_does_not_train_question_tokens(self):
        row = _example("What is alphabet?", "Alphabet is a language tool.", 256)
        self.assertIsNotNone(row)
        x, labels = row
        context = "Question: What is alphabet?\nAnswer:"
        self.assertEqual(labels[:len(context)-1], [-100]*(len(context)-1))
        self.assertEqual(labels[len(context)-1], ord(" "))
        self.assertEqual(labels[-1], ord("\n"))
        self.assertIsNone(_example("What is alphabet?", "x"*500, 256))
        self.assertEqual(len(TRAIN_TEMPLATES), 3)
        self.assertEqual(len(TEST_TEMPLATES), 2)

    def test_unlicensed_and_research_only_evidence_not_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = config(folder)
            result = populate(store, cfg, list(DEFINITIONS)[:3], license_markup="")
            self.assertEqual(result["verified_answers"], 3)
            self.assertEqual(result["training_examples_created"], 0)
            self.assertEqual(concept_inventory(store)["eligible_concepts"], 0)
            self.assertEqual(train_concept_lab(store, cfg)["status"], "skipped")
            store.close()

    def test_only_training_split_licensed_verified_records_enter_neural_training(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = config(folder)
            outcome = populate(store, cfg, list(DEFINITIONS)[:4])
            self.assertEqual(outcome["verified_answers"], 4)
            self.assertEqual(outcome["training_examples_created"], 4)
            selected = _inventory(store)
            # Original storage split reserves first record for validation.
            self.assertEqual(len(selected), 3)
            self.assertNotIn("alphabet", [r["concept"] for r in selected])
            self.assertEqual([r["concept"] for r in selected],
                             ["volcano", "compass", "thermometer"])
            self.assertTrue(all("Question: What is" not in r["answer"]
                                for r in selected))
            store.close()

    def test_inference_validation_does_not_require_training_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertEqual(predict_concept_lab(store, "What is a compass?")["status"], "skipped")
            with self.assertRaises(ValueError):
                predict_concept_lab(store, "")
            with self.assertRaises(ValueError):
                predict_concept_lab(store, "What is a compass?", max_new_bytes=999)
            self.assertEqual(evaluate_concept_lab(store)["status"], "skipped")
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class RealTextNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_actual_verified_neural_training_holdout_and_checkpoint_persistence(self):
        from evolve.brain import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = config(folder)
            populate(store, cfg, list(DEFINITIONS)[:4])
            report = train_concept_lab(
                store, cfg, steps=40, max_concepts=4,
                seed=31415, max_new_bytes=16)
            self.assertEqual(report["status"], "trained")
            self.assertEqual(report["concept_count"], 3)
            self.assertEqual(report["train_examples"], 9)
            self.assertEqual(report["validation_examples"], 3)
            self.assertEqual(report["heldout_question_count"], 6)
            self.assertGreater(report["actual_steps"], 0)
            self.assertGreater(report["train_answer_loss_before"],
                               report["train_answer_loss_after"])
            self.assertIsInstance(report["accepted"], bool)
            self.assertTrue(0 <= report["after_heldout"]["exact_accuracy"] <= 1)
            self.assertEqual(len(report["after_heldout"]["examples"]), 6)
            train_set = set(report["training_prompt_templates"])
            self.assertTrue(all(q not in train_set for q in
                                report["heldout_prompt_templates"]))
            self.assertEqual(store.status()["research_only_pages"], 2)
            self.assertEqual(store.status()["verified_arithmetic_documents"], 0)
            self.assertIsNone(store.latest_promoted())
            self.assertEqual(sha256_file(report["checkpoint"]), report["sha256"])

            # Evaluate the saved candidate even if honest heldout accuracy is low.
            candidate = evaluate_concept_lab(store, candidate=True, max_new_bytes=16)
            self.assertEqual(candidate["status"], "evaluated")
            self.assertEqual(candidate["checkpoint"], report["sha256"])
            self.assertEqual(candidate["evaluation"]["questions"], 6)
            self.assertEqual(
                train_concept_lab(store, cfg, steps=40, seed=31415,
                                  max_concepts=4, max_new_bytes=16)["status"],
                "skipped" if report["accepted"] else "trained")

            # Inference uses the checkpoint only; no evidence lookup or memory.
            with patch.object(store, "documents",
                              side_effect=AssertionError("No docs at inference")), \
                 patch.object(store, "list_verified_answers",
                              side_effect=AssertionError("No answer lookup")), \
                 patch.object(store, "relevant_memories",
                              side_effect=AssertionError("No retrieval")):
                prediction = predict_concept_lab(
                    store, "Describe compass.", candidate=True,
                    max_new_bytes=16)
            self.assertEqual(prediction["status"], "predicted")
            self.assertIsInstance(prediction["raw_completion"], str)
            store.close()
            reopened = Store(folder)
            replay = predict_concept_lab(reopened, "Describe compass.",
                                         candidate=True, max_new_bytes=16)
            self.assertEqual(prediction["raw_completion"], replay["raw_completion"])
            self.assertEqual(prediction["checkpoint"], replay["checkpoint"])
            reopened.close()

    def test_later_verified_concepts_can_create_separate_new_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = config(folder)
            populate(store, cfg, list(DEFINITIONS)[:3])
            first = train_concept_lab(store, cfg, steps=24, seed=42,
                                      max_new_bytes=16)
            self.assertEqual(first["status"], "trained")
            populate(store, cfg, ["thermometer", "river"])
            second = train_concept_lab(store, cfg, steps=24, seed=42,
                                       max_new_bytes=16)
            self.assertEqual(second["status"], "trained")
            self.assertNotEqual(first["dataset_sha256"],
                                second["dataset_sha256"])
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertTrue(set(first["concepts"]).issubset(second["concepts"]))
            self.assertEqual(store.status()["verified_neural_documents"], 5)
            self.assertIsNone(store.latest_promoted())
            store.close()

    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = config(folder)
            populate(store, cfg, list(DEFINITIONS)[:3])
            result = train_concept_lab(store, cfg, steps=16, seed=42,
                                       max_new_bytes=16)
            with open(result["checkpoint"], "ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                predict_concept_lab(store, "Describe a compass.",
                                    candidate=True, max_new_bytes=16)
            store.close()


if __name__ == "__main__":
    unittest.main()
