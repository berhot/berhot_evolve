"""v1.25 manually judged ranking benchmark: real quotes, no auto labels."""
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.store import Store
from evolve.v120_grounded import _inventory
from evolve.v125_reviewed_benchmark import (
    candidate_pairs, review_pair, score_reviewed_pairs,
)

TEXTS = (
    ("https://a.example.org/facts/current",
     "A current is a stream of moving ocean water that travels along a coast.",
     "CC0-1.0"),
    ("https://b.example.org/facts/current",
     "The current is a continuous flow of seawater within the ocean basin.",
     "CC0-1.0"),
    ("https://c.example.org/facts/current",
     "A current is a movement of seawater across the ocean surface.",
     "CC0-1.0"),
    ("https://unknown.example.org/facts/current",
     "A current is a flowing ocean-water stream near coastal landforms.",
     None),
)


def prepare(store):
    for url, text, license_id in TEXTS:
        filler = " ".join(
            "Researchers recorded ocean measurements using independent "
            "scientific equipment in the coastal region at station " + str(i) + "."
            for i in range(65))
        raw = text + " " + filler
        result = archive_page(
            store, url, "<html><body>" + raw + "</body></html>",
            raw, title="Ocean studies",
            declared_license=license_id,
            rights_url="fixture:self-authored" if license_id else None)
        assert result["status"] == "archived", result


class HumanBenchmarkTests(unittest.TestCase):
    def test_candidates_and_manual_review_are_real_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                prepare(store)
                before = store.db.total_changes
                overview = candidate_pairs(store)
                self.assertEqual(store.db.total_changes, before)
                self.assertEqual(overview["topics_with_two_or_more_quotes"], 1)
                self.assertEqual(overview["total_eligible_quotes"], 3)
                self.assertEqual(overview["unlabeled_potential_same_topic_pairs"], 3)
                available = overview["shown_topics"][0]["candidates"]
                self.assertEqual(len(available), 3)
                good, bad = available[0], available[1]
                args = ("What is a current?", good["url"], good["sentence_sha256"],
                        bad["url"], bad["sentence_sha256"])
                with self.assertRaises(ValueError):
                    review_pair(store, *args, confirm=False)
                saved = review_pair(store, *args, confirm=True)
                self.assertEqual(saved["status"], "manual_judgment_saved")
                self.assertFalse(saved["fed_to_neural_training"])
                with self.assertRaises(ValueError):
                    review_pair(store, *args, confirm=True)
                with self.assertRaises(ValueError):
                    review_pair(store, "What is quantum entanglement?",
                                *args[1:], confirm=True)
                with self.assertRaises(ValueError):
                    review_pair(store, "What is a current?",
                                bad["url"], bad["sentence_sha256"],
                                good["url"], good["sentence_sha256"],
                                confirm=True)
                unknown = TEXTS[-1][0]
                with self.assertRaises(ValueError):
                    review_pair(store, "What is a current?",
                                unknown, "0" * 64,
                                bad["url"], bad["sentence_sha256"],
                                confirm=True)
                before = store.db.total_changes
                result = score_reviewed_pairs(store)
                self.assertEqual(store.db.total_changes, before)
                self.assertEqual(result["reviewed_pairs_evaluated"], 1)
                self.assertIsNone(result["neural_accuracy"])
                self.assertIsNotNone(result["lexical_accuracy"])
                self.assertEqual(result["manual_judgments_recorded"], 1)
                self.assertEqual(result["invalid_or_stale_pairs"], [])
                self.assertTrue(
                    result["limitations"]["development_judgments_not_blind_test"])
            finally:
                store.close()

    def test_disqualifying_source_after_review_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                prepare(store)
                facts = _inventory(store)
                self.assertEqual(len(facts), 3)
                good, bad = facts[:2]
                review_pair(
                    store, "Define current", good["url"],
                    good["sentence_sha256"], bad["url"],
                    bad["sentence_sha256"], confirm=True)
                with store.db:
                    store.db.execute(
                        "UPDATE web_corpus_sources SET rights_status='UNKNOWN' "
                        "WHERE url=?", (good["url"],))
                result = score_reviewed_pairs(store)
                self.assertEqual(result["reviewed_pairs_evaluated"], 0)
                self.assertEqual(len(result["invalid_or_stale_pairs"]), 1)
                self.assertIsNone(result["lexical_accuracy"])
                self.assertIsNone(result["neural_accuracy"])
            finally:
                store.close()

    def test_no_candidates_is_not_fabricated_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                pairs = candidate_pairs(store)
                self.assertEqual(pairs["total_eligible_quotes"], 0)
                report = score_reviewed_pairs(store)
                self.assertEqual(report["manual_judgments_recorded"], 0)
                self.assertIsNone(report["lexical_accuracy"])
                self.assertIsNone(report["neural_accuracy"])
                for fn in (candidate_pairs, score_reviewed_pairs):
                    with self.assertRaises(ValueError):
                        fn(store, limit=0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
