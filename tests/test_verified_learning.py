"""End-to-end independent evidence, eligibility and continued neural learning."""
import tempfile
import unittest

from evolve.research import (
    _same_claim, extract_definition, research_questions,
)
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None

LICENSE = '<link rel="license" href="https://creativecommons.org/publicdomain/zero/1.0/">'
FIRST = "https://alpha.example.org/articles"
SECOND = "https://beta.other.edu/reference"
THIRD_SAME_PUBLISHER = "https://mirror.example.org/articles"

DEFINITIONS = {
    "alphabet": "a collection of letters used to represent words in language",
    "circle": "a round shape with all points equally distant from its center",
    "volcano": "an opening in the earth that releases hot rock and gas",
    "planet": "a large round celestial body orbiting a star in space",
    "vowel": "a speech sound formed with open airflow through the mouth",
    "triangle": "a geometric shape with three straight sides and angles",
    "gravity": "a natural force that attracts objects with mass toward each other",
    "ocean": "a large body of salty water covering part of the earth",
    "river": "a natural flowing stream of water moving through the land",
    "cloud": "a visible mass of tiny water droplets floating in the atmosphere",
    "mountain": "a large natural elevation of the earth above surrounding land",
    "forest": "a large area of land covered with many trees and plants",
}


def page(definitions, license_html=""):
    statements = [
        f"<p>An {term} is {definition}.</p>"
        for term, definition in definitions.items()
    ]
    return "<html><head><title>Definitions</title>" + license_html + \
        "</head><body>" + "".join(statements) + "</body></html>"


def setup_store(store, concepts, second_url=SECOND):
    store.save_observation(FIRST, "Primary definitions", "Distinct educational examples",
                           concepts, "research_only")
    store.save_observation(second_url, "Independent definitions", "Different editorial source",
                           concepts, "research_only")
    for term in concepts:
        store.add_research_question(term, f"What is {term}?", FIRST)


def settings_for(tmp):
    cfg = load_config()
    cfg.update(storage_root=tmp, daily_request_budget=200, crawl_delay_seconds=0,
               minimum_training_documents=6, minimum_new_documents_for_training=6)
    cfg["research"] = dict(questions_per_cycle=20, max_pages_per_question=3,
                           retry_cooldown_seconds=3600)
    cfg["model"] = dict(block_size=16, embedding_dim=16, layers=1, heads=2, dropout=0.0)
    cfg["train"].update(steps=12, batch_size=2, eval_batches=1,
                         device="cpu", max_seconds=90, learning_rate=0.002)
    return cfg


class EvidenceVerificationTests(unittest.TestCase):
    def test_extract_and_disagreement(self):
        self.assertIsNotNone(extract_definition(
            "An alphabet is a collection of letters used to represent words in language.",
            "alphabet"))
        self.assertFalse(_same_claim(
            "a set of 26 letters used for writing words in English",
            "a set of 24 letters used for writing words in English"))
        self.assertFalse(_same_claim(
            "a group of letters not used to represent words in language",
            "a group of letters used to represent words in language"))

    def test_two_host_reusable_evidence_can_create_verified_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            setup_store(store, ["alphabet"])
            pages = {FIRST: page({"alphabet": DEFINITIONS["alphabet"]}, LICENSE),
                     SECOND: page({"alphabet": DEFINITIONS["alphabet"]})}
            report = research_questions(store, settings_for(tmp), limit=1,
                                        fetcher=lambda url: pages[url])
            self.assertEqual(report["verified_answers"], 1)
            self.assertEqual(report["training_examples_created"], 1)
            self.assertEqual(report["source_requests"], 2)
            answer = store.list_verified_answers()[0]
            self.assertEqual(len(answer["evidence"]), 2)
            self.assertTrue(answer["training_eligible"])
            self.assertEqual(store.status()["verified_neural_documents"], 1)
            self.assertIn("Question: What is alphabet?", store.documents(verified_only=True)[0]["text"])
            self.assertEqual(research_questions(store, settings_for(tmp), limit=1,
                                                fetcher=lambda url: pages[url])["verified_answers"], 0)
            store.close()

    def test_unlicensed_agreement_never_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            setup_store(store, ["alphabet"])
            pages = {FIRST: page({"alphabet": DEFINITIONS["alphabet"]}),
                     SECOND: page({"alphabet": DEFINITIONS["alphabet"]})}
            report = research_questions(store, settings_for(tmp), limit=1,
                                        fetcher=lambda url: pages[url])
            self.assertEqual(report["verified_answers"], 1)
            self.assertEqual(report["training_examples_created"], 0)
            self.assertEqual(store.status()["documents"], 0)
            store.close()

    def test_conflicting_sources_or_one_publisher_cannot_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            setup_store(store, ["alphabet"])
            pages = {FIRST: page({"alphabet": DEFINITIONS["alphabet"]}, LICENSE),
                     SECOND: page({"alphabet": "a collection of numeric symbols for representing distant objects"})}
            report = research_questions(store, settings_for(tmp), limit=1,
                                        fetcher=lambda url: pages[url])
            self.assertEqual(report["verified_answers"], 0)
            self.assertEqual(report["unresolved_questions"], 1)
            self.assertEqual(store.status()["verified_neural_documents"], 0)
            store.close()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            setup_store(store, ["alphabet"], second_url=THIRD_SAME_PUBLISHER)
            pages = {FIRST: page({"alphabet": DEFINITIONS["alphabet"]}, LICENSE),
                     THIRD_SAME_PUBLISHER: page({"alphabet": DEFINITIONS["alphabet"]})}
            report = research_questions(store, settings_for(tmp), limit=1,
                                        fetcher=lambda url: pages[url])
            self.assertEqual(report["verified_answers"], 0)
            store.close()

    def test_unsupported_question_remains_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.add_research_question("alphabet", "Which concepts relate to alphabet?", FIRST)
            report = research_questions(store, settings_for(tmp), limit=1,
                                        fetcher=lambda _url: None)
            self.assertEqual(report["unsupported_questions"], 1)
            self.assertEqual(report["source_requests"], 0)
            self.assertEqual(store.open_questions_count(), 1)
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class VerifiedNeuralLearningTests(unittest.TestCase):
    def test_two_incremental_verified_training_rounds(self):
        from pathlib import Path
        from evolve.training import train_candidate
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            cfg = settings_for(tmp)
            first_concepts = list(DEFINITIONS)[:6]
            second_concepts = list(DEFINITIONS)[6:]
            setup_store(store, first_concepts)
            pages = {FIRST: page(DEFINITIONS, LICENSE), SECOND: page(DEFINITIONS)}
            fetch = lambda url: pages[url]

            round1 = research_questions(store, cfg, limit=6, fetcher=fetch)
            self.assertEqual(round1["verified_answers"], 6)
            self.assertEqual(round1["training_examples_created"], 6)
            first = train_candidate(store, cfg, verified_only=True)
            self.assertEqual(first["status"], "trained")
            self.assertEqual(first["dataset_scope"], "verified_only")
            self.assertEqual(first["training_documents"], 6)
            self.assertGreater(first["actual_steps"], 0)
            self.assertLess(first["post_train_loss"], first["pre_train_loss"])
            self.assertTrue(first["promoted"])
            self.assertTrue(Path(first["candidate_path"]).is_file())

            store.save_observation(FIRST, "More verified definitions", "Updated evidence",
                                   list(DEFINITIONS), "research_only")
            store.save_observation(SECOND, "More corroboration", "Independent publisher",
                                   list(DEFINITIONS), "research_only")
            for term in second_concepts:
                store.add_research_question(term, f"What is {term}?", FIRST)
            round2 = research_questions(store, cfg, limit=6, fetcher=fetch)
            self.assertEqual(round2["verified_answers"], 6)
            self.assertEqual(round2["training_examples_created"], 6)
            second = train_candidate(store, cfg, verified_only=True)
            self.assertEqual(second["status"], "trained")
            self.assertEqual(second["training_documents"], 12)
            self.assertGreater(second["actual_steps"], 0)
            self.assertIsNotNone(second["baseline_val_loss"])
            self.assertTrue(Path(second["candidate_path"]).is_file())
            again = train_candidate(store, cfg, verified_only=True)
            self.assertEqual(again["status"], "skipped")
            self.assertEqual(store.status()["verified_neural_documents"], 12)
            self.assertEqual(store.status()["research_questions_verified"], 12)
            store.close()


if __name__ == "__main__":
    unittest.main()
