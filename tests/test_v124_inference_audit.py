"""v1.24: diagnostic honesty—loaded weights are not necessarily a choice."""
import copy
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v120_grounded import grounded_answer
from evolve.v124_inference_audit import evaluate_inference


class InferenceAuditTests(unittest.TestCase):
    def test_competing_candidates_and_quote_integrity_are_not_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                for host, text, declared in (
                    ("first.example.org",
                     "A current is a continuous motion of seawater along the shore.",
                     "CC0-1.0"),
                    ("second.example.org",
                     "A current is a moving stream of ocean water driven by wind.",
                     "CC0-1.0"),
                    ("unknown.example.org",
                     "A current is water moving along a coastline at some speed.",
                     None),
                ):
                    url = f"https://{host}/facts/current"
                    filler = " ".join(
                        "Measurements of the ocean were recorded with careful "
                        "instruments during coastal surveys at site " + str(i) + "."
                        for i in range(15))
                    result = archive_page(
                        store, url,
                        "<html><body>" + text + " " + filler + "</body></html>",
                        text + " " + filler, title="What is a current?",
                        noai=False, declared_license=declared,
                        rights_url="fixture:self-authored" if declared else None)
                    self.assertEqual(result["status"], "archived")
                response = grounded_answer(store, "What is a current?")
                self.assertEqual(response["candidate_count"], 2)
                self.assertFalse(response["ranker_checkpoint_loaded"])
                self.assertFalse(response["neural_ranker_used"])
                self.assertFalse(response["neural_decision_changed_top_quote"])
                before = store.db.total_changes
                result = evaluate_inference(store, limit=20)
                self.assertEqual(store.db.total_changes, before)
                self.assertEqual(result["status"], "diagnostic_only")
                self.assertEqual(result["approved_definition_topics_total"], 1)
                self.assertEqual(result["approved_definition_quotes_total"], 2)
                self.assertEqual(result["topics_with_competing_quotes"], 1)
                self.assertEqual(result["actual_neural_ranking_competitions"], 0)
                self.assertEqual(result["distinct_source_quotes_integrity_checked"], 2)
                self.assertEqual(result["distinct_source_quotes_integrity_passed"], 2)
                self.assertEqual(result["quote_integrity_failures"], [])
                self.assertEqual(result["topics"][0]["candidate_quotes"], 2)
                self.assertFalse(result["limitations"][
                    "external_human_graded_answer_relevance"])
                self.assertFalse(result["limitations"]["fresh_blind_test_performed"])
                self.assertTrue(all(
                    probe["status"] == "no_definition" for probe in
                    result["absent_from_approved_inventory_probes"]))
                self.assertEqual(
                    result["unsupported_question_probe"]["status"], "unsupported")
            finally:
                store.close()

    def test_empty_corpus_reports_no_baseline_and_abstains(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                report = evaluate_inference(store)
                self.assertEqual(report["approved_definition_topics_total"], 0)
                self.assertEqual(report["candidate_question_variants_examined"], 0)
                self.assertEqual(report["topics_with_competing_quotes"], 0)
                self.assertEqual(report["loaded_checkpoint_question_cases"], 0)
                self.assertFalse(
                    report["last_ranker_comparison"]["checkpoint_recorded"])
                with self.assertRaises(ValueError):
                    evaluate_inference(store, limit=0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
