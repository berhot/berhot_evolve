"""v0.7: live-style research diagnostics, self-checking examples and unseen model assessment."""
import tempfile
import unittest

from evolve.practice import checked_answer, practice_arithmetic
from evolve.research import research_questions
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import register_exploration, discover_web

try:
    import torch
except ImportError:
    torch = None

A = "https://alpha.example.org/alphabet"
B = "https://independent.example.net/topics/alphabet"


def config(folder):
    cfg = load_config()
    cfg.update(storage_root=folder, daily_request_budget=100, crawl_delay_seconds=0,
               minimum_training_documents=6, minimum_new_documents_for_training=6)
    cfg["research"] = {"max_pages_per_question": 4, "questions_per_cycle": 5,
                       "retry_cooldown_seconds": 3600}
    cfg["model"] = {"block_size": 16, "embedding_dim": 16, "layers": 1,
                     "heads": 2, "dropout": 0.0}
    cfg["train"].update(device="cpu", steps=10, batch_size=2, eval_batches=1,
                         learning_rate=0.002, max_seconds=120)
    return cfg


class ResearchTriageTests(unittest.TestCase):
    def test_only_supported_questions_consume_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.add_research_question("alphabet", "What is alphabet?", A)
            store.add_research_question("alphabet", "Which ideas relate to alphabet?", A)
            store.add_research_question("alphabet", "How can alphabet be checked?", A)
            results = research_questions(store, config(folder), limit=3,
                                         fetcher=lambda _: "<html><body>No definitions here</body></html>")
            self.assertEqual(results["attempted_questions"], 1)
            self.assertEqual(results["unsupported_questions"], 2)
            self.assertEqual(results["deferred_unsupported_questions"], 2)
            self.assertEqual(results["unresolved_questions"], 1)
            self.assertEqual(results["question_outcomes"][0]["outcome"], "no_definition_found")
            self.assertEqual(store.open_questions_count(), 3)
            store.close()

    def test_external_discovered_link_can_be_independent_evidence(self):
        first = "<html><body>An alphabet is a collection of letters used to represent words in language.</body></html>"
        second = "<html><body>An alphabet is a collection of letters used to represent words in language.</body></html>"
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.add_web_url(B)
            store.add_research_question("alphabet", "What is alphabet?", A)
            docs = {A: first, B: second}
            results = research_questions(store, config(folder), limit=1, fetcher=lambda u: docs[u])
            self.assertEqual(results["verified_answers"], 1)
            self.assertEqual(results["training_examples_created"], 0)
            self.assertEqual(results["answers"][0]["training_eligible"], False)
            store.close()

    def test_discovery_creates_supported_questions_only(self):
        html = """<html><head><title>Alphabet lessons</title></head><body>
        <h1>Letters of the alphabet</h1>
        <p>Alphabet letters form words. Alphabet letters are used for writing.
        Readers recognize letters and combine letters to form different words.</p>
        </body></html>"""
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            register_exploration(store, A)
            result = discover_web(store, config(folder), limit=1,
                                  fetcher=lambda _: html, sleep_fn=lambda _: None)
            self.assertGreater(result["questions_created"], 0)
            self.assertTrue(all(q["question"].startswith("What is ") for q in store.list_research_questions(100)))
            store.close()


class ArithmeticPracticeTests(unittest.TestCase):
    def test_arithmetic_is_independently_checked_and_deduplicated(self):
        self.assertEqual(checked_answer(24, 8, "+"), 32)
        self.assertEqual(checked_answer(24, 8, "-"), 16)
        self.assertEqual(checked_answer(24, 8, "*"), 192)
        with self.assertRaises(ValueError):
            checked_answer(2, 3, "/")
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.save_observation(A, "Unlicensed website", "Unlicensed content",
                                   ["alphabet"], "research_only")
            first = practice_arithmetic(store, limit=32)
            self.assertEqual(first["new_examples"], 32)
            self.assertEqual(len(store.documents(verified_only=True)), 32)
            self.assertTrue(all(row["topic"] == "verified:arithmetic" and
                                row["url"].startswith("internal://")
                                for row in store.documents(verified_only=True)))
            second = practice_arithmetic(store, limit=32)
            self.assertEqual(second["new_examples"], 32)
            self.assertEqual(len(store.documents(verified_only=True)), 64)
            self.assertEqual(store.status()["research_questions_verified"], 0)
            self.assertEqual(store.status()["verified_arithmetic_documents"], 64)
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class NeuralGeneralizationTests(unittest.TestCase):
    def test_real_two_round_training_then_unseen_exact_match(self):
        from pathlib import Path
        from evolve.evaluation import evaluate_arithmetic
        from evolve.training import train_candidate
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = config(folder)
            self.assertEqual(evaluate_arithmetic(store, cfg)["status"], "skipped")
            practice_arithmetic(store, limit=32)
            first = train_candidate(store, cfg, verified_only=True)
            self.assertEqual(first["status"], "trained")
            self.assertTrue(first["promoted"])
            self.assertTrue(Path(first["candidate_path"]).exists())
            self.assertLess(first["post_train_loss"], first["pre_train_loss"])
            evaluation = evaluate_arithmetic(store, cfg, questions=4)
            self.assertEqual(evaluation["status"], "evaluated")
            self.assertEqual(evaluation["questions"], 4)
            self.assertGreaterEqual(evaluation["accuracy"], 0)
            self.assertLessEqual(evaluation["accuracy"], 1)
            self.assertEqual(len(evaluation["examples"]), 4)
            self.assertIsNotNone(store.status()["last_arithmetic_evaluation"])
            practice_arithmetic(store, limit=32)
            second = train_candidate(store, cfg, verified_only=True)
            self.assertEqual(second["status"], "trained")
            self.assertEqual(second["training_documents"], 64)
            self.assertIsNotNone(second["baseline_val_loss"])
            self.assertEqual(train_candidate(store, cfg, verified_only=True)["status"], "skipped")
            store.close()


if __name__ == "__main__":
    unittest.main()
