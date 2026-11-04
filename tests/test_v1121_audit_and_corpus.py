"""v1.12.1 regression: truthful checkpoint promotion and isolated corpus crawling."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from evolve.model_audit import audit_model_and_ownership
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import discover_web
from evolve.open_corpus import corpus_seed


class ReportingCorrections(unittest.TestCase):
    def test_general_promoted_model_is_not_falsely_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            artifact = Path(root) / "models" / "checkpoint.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"small fake checkpoint metadata fixture")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            store.insert_model(artifact, digest, 32, 1.6574, None, True)
            audit = audit_model_and_ownership(store)
            model = audit["saved_models"]["latest_general_promoted"]
            self.assertEqual(audit["saved_models"]["general_promoted_runs"], 1)
            self.assertEqual(model["checkpoint"]["status"], "sha256_verified")
            self.assertTrue(model["checkpoint"]["promoted"])
            self.assertIsNone(model["checkpoint"]["accepted_by_validation"])
            self.assertIn("Not recorded", model["checkpoint"]["acceptance_note"])
            self.assertEqual(model["trained_on_document_count_at_save"], 32)
            store.close()

    def test_specialized_acceptance_is_reported_only_when_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            file = Path(root) / "checkpoint.pt"
            file.write_bytes(b"specialized fixture")
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            store.put("gradient_lab_latest_candidate", {
                "path": str(file), "sha256": sha, "accepted": True
            })
            audit = audit_model_and_ownership(store)
            record = audit["saved_models"]["specialized_checkpoint_records"]["gradient_lab"]
            self.assertTrue(record["accepted_by_validation"])
            self.assertNotIn("promoted", record)
            store.close()

    def test_corpus_only_does_not_review_old_pending_questions_with_zero_quota(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = load_config()
            cfg["storage_root"] = root
            cfg["daily_request_budget"] = 0
            corpus_seed(store, ["https://example.org/"])
            store.add_concept_task(
                "volcano", "https://example.org/",
                max_total=20, priority=1.0, source_title="volcano")
            self.assertEqual(store.pending_concepts(), 1)
            first = discover_web(
                store, cfg, limit=3, review_questions=False,
                fetcher=lambda _: (_ for _ in ()).throw(
                    AssertionError("Quota zero; no fetch")))
            self.assertEqual(first["attempted"], 0)
            self.assertEqual(first["status"], "daily_budget_exhausted")
            self.assertEqual(first["questions_created"], 0)
            self.assertEqual(first["question_review_mode"], "disabled_for_corpus_only")
            self.assertEqual(store.pending_concepts(), 1)
            self.assertEqual(store.open_questions_count(), 0)
            self.assertEqual(store.db.execute(
                "SELECT COUNT(*) FROM web_corpus_texts").fetchone()[0], 0)
            # Research discovery deliberately retains its old behavior:
            second = discover_web(
                store, cfg, limit=3, review_questions=True,
                fetcher=lambda _: (_ for _ in ()).throw(
                    AssertionError("Quota zero; no fetch")))
            self.assertEqual(second["questions_created"], 1)
            self.assertEqual(second["question_review_mode"], "research_backlog_enabled")
            self.assertEqual(store.open_questions_count(), 1)
            store.close()

    def test_corpus_only_with_new_page_keeps_question_queue_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = load_config()
            cfg["storage_root"] = root
            cfg["daily_request_budget"] = 5
            cfg["crawl_delay_seconds"] = 0
            cfg["web"]["max_depth"] = 0
            corpus_seed(store, ["https://example.org/"])
            html = ("<html><head><title>Volcano education</title></head><body>"
                    "<h1>Volcano formation and geological activity</h1>"
                    "<p>Volcanoes arise from geological processes in the Earth's "
                    "crust and can release gases, ash and melted rock.</p>"
                    "<p>Scientists study volcano patterns to learn about "
                    "planetary geology and natural hazard events.</p>"
                    "</body></html>")
            report = discover_web(store, cfg, limit=1, review_questions=False,
                                  fetcher=lambda _: html,
                                  sleep_fn=lambda _: None)
            self.assertEqual(report["attempted"], 1)
            self.assertEqual(report["corpus_archived"], 1)
            self.assertEqual(report["questions_created"], 0)
            self.assertEqual(store.open_questions_count(), 0)
            self.assertGreater(store.pending_concepts(), 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
