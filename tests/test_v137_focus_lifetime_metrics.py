"""Durable lifetime focus metrics stay nonzero after worker session resets."""
import copy
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.store import Store
from evolve.settings import load_config
from evolve.v118_studio import LearningController

PROFILE = "saudi-history-ar"


def arabic():
    return ("تاريخ المملكة العربية السعودية والدولة السعودية الأولى "
            "والدرعية والملك عبدالعزيز ويوم التأسيس. " * 50)


def html():
    return "<html><title>تاريخ المملكة العربية السعودية</title><article>" + arabic() + "</article></html>"


class LifetimeFocusMetricsTests(unittest.TestCase):
    def test_persistent_archive_bytes_dedup_and_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                url1 = "https://saudipedia.com/first-article"
                url2 = "https://saudipedia.com/second-article"
                result1 = archive_page(store, url1, html(), arabic(),
                                       title="تاريخ المملكة العربية السعودية")
                result2 = archive_page(store, url2, html(), arabic(),
                                       title="تاريخ المملكة العربية السعودية")
                self.assertEqual(result1["clean_sha256"], result2["clean_sha256"])
                self.assertTrue(store.record_focus_archive(
                    PROFILE, url1, result1["clean_sha256"], source_questions=3))
                self.assertTrue(store.record_focus_archive(
                    PROFILE, url2, result2["clean_sha256"], source_questions=1))
                totals = store.focus_archive_totals(PROFILE)
                self.assertEqual(totals["source_urls"], 2)
                self.assertEqual(totals["unique_texts"], 1)
                self.assertEqual(totals["source_questions"], 4)
                self.assertEqual(
                    totals["unique_clean_utf8_bytes"],
                    len(arabic().encode("utf-8")))
                self.assertEqual(totals["training_bytes"], 0)
            finally:
                store.close()
            reopened = Store(folder)
            try:
                self.assertEqual(
                    reopened.focus_archive_totals(PROFILE)["source_urls"], 2)
            finally:
                reopened.close()

    def test_legacy_observations_backfill_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                url = "https://saudipedia.com/saudi-history"
                archived = archive_page(store, url, html(), arabic(),
                                        title="تاريخ المملكة العربية السعودية")
                items = [{
                    "url": url,
                    "source_sha256": archived["clean_sha256"],
                    "curiosity": [{"question": "متى التأسيس؟"},
                                  {"question": "أين الدرعية؟"}],
                }]
                self.assertEqual(store.backfill_focus_archives(PROFILE, items), 1)
                self.assertEqual(store.backfill_focus_archives(PROFILE, items), 0)
                totals = store.focus_archive_totals(PROFILE)
                self.assertEqual(totals["source_urls"], 1)
                self.assertEqual(totals["source_questions"], 2)
            finally:
                store.close()

    def test_zero_new_session_counts_do_not_erase_saved_research(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                url = "https://saudipedia.com/historical-article"
                archived = archive_page(store, url, html(), arabic(),
                                        title="تاريخ المملكة العربية السعودية")
                store.put("focus_observations_" + PROFILE, [{
                    "url": url, "source_sha256": archived["clean_sha256"],
                    "title": "تاريخ المملكة العربية السعودية",
                    "host": "saudipedia.com",
                    "rights": "UNKNOWN",
                    "curiosity": [{"question": "متى تأسست المملكة؟"}],
                }])
                store.put("focus_run_" + PROFILE, {
                    "run_id": "new-session-with-zero-work",
                    "status": "idle_waiting_for_allowed_work",
                    "pages_archived": 0, "pages_read": 0,
                    "errors": 2, "pages_attempted": 2,
                    "candidate_questions_created_this_run": 0,
                    "training_steps_this_run": 0,
                    "training_byte_tokens_processed": 0,
                    "training_status": "skipped_no_eligible_training_corpus",
                })
            finally:
                store.close()
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = folder
            ctrl = LearningController(cfg, interval=30)
            try:
                status = ctrl.status()
                focus = status["focus_agent"]
                self.assertEqual(focus["pages_read"], 0)
                self.assertEqual(focus["source_errors_this_run"], 2)
                self.assertEqual(focus["pages_attempted"], 2)
                saved = focus["saved_research_all_runs"]
                self.assertEqual(saved["source_urls"], 1)
                self.assertEqual(saved["source_questions"], 1)
                self.assertGreater(saved["unique_clean_utf8_bytes"], 0)
                self.assertEqual(focus["training_steps_this_run"], 0)
                inspector = ctrl.focus_inspector()
                self.assertEqual(
                    inspector["research"]["archived_research_records"], 1)
                self.assertGreater(inspector["research"]
                                   ["unique_clean_utf8_bytes_all_runs"], 0)
                self.assertIsNone(
                    inspector["training"]["last_accepted_checkpoint"])
            finally:
                ctrl.shutdown()


if __name__ == "__main__":
    unittest.main()
