"""Current batch progress must not reuse the last completed batch count."""
import tempfile
import unittest
from unittest.mock import patch

from evolve.store import Store
from evolve.v118_studio import _current_cycle_progress
from evolve.v118_studio import LearningController
from evolve.settings import load_config


class FocusMetricAccuracyTests(unittest.TestCase):
    def test_readiness_cache_tracks_corpus_approval_and_example_changes(self):
        from evolve.open_corpus import archive_page
        from evolve.training_approval import approve_current_corpus
        with tempfile.TemporaryDirectory() as root:
            settings = load_config()
            settings["storage_root"] = root
            ctrl = LearningController(settings)
            store = Store(root)
            try:
                with patch("evolve.v118_studio.qa_readiness", return_value={"status": "ready"}) as compute:
                    ctrl._qa_readiness_snapshot(store, [], [])
                    ctrl._qa_readiness_snapshot(store, [], [])
                    self.assertEqual(compute.call_count, 1)
                    store.put("focus_run_saudi-history-ar", {"heartbeat_at_utc": "new heartbeat"})
                    ctrl._qa_readiness_snapshot(store, [], [])
                    self.assertEqual(compute.call_count, 1)
                    archive_page(store, "https://example.org/article", "<p>source</p>", "A new source text.")
                    ctrl._qa_readiness_snapshot(store, [], [])
                    self.assertEqual(compute.call_count, 2)
                    approve_current_corpus(store, "Operator approval")
                    ctrl._qa_readiness_snapshot(store, [], [])
                    self.assertEqual(compute.call_count, 3)
                    ctrl._qa_readiness_snapshot(store, [{"question": "new"}], [])
                    self.assertEqual(compute.call_count, 4)
                    ctrl.qa_plan_cache["expires"] = 0
                    ctrl._qa_readiness_snapshot(store, [{"question": "new"}], [])
                    self.assertEqual(compute.call_count, 5)
            finally:
                ctrl.shutdown()
                store.close()

    def test_current_cycle_uses_matching_run_events_and_resets_after_completion(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                run = {"run_id": "current", "status": "running",
                       "last_cycle_pages_attempted": 25}
                self.assertEqual(_current_cycle_progress(store, run),
                                 {"current_cycle_attempts": 0, "current_cycle_active": False})
                counter = [0]
                def event(kind, run_id="current"):
                    counter[0] += 1
                    store.append_focus_activity("saudi-history-ar", run_id,
                        f"2026-11-06T00:00:{counter[0]:02d}+00:00", kind)
                event("search_check")
                event("fetch_started")
                event("fetch_started", "different-run")
                event("fetch_started")
                self.assertEqual(_current_cycle_progress(store, run),
                                 {"current_cycle_attempts": 2, "current_cycle_active": True})
                event("cycle_finished")
                self.assertEqual(_current_cycle_progress(store, run),
                                 {"current_cycle_attempts": 0, "current_cycle_active": False})
                event("search_check")
                event("fetch_started")
                self.assertEqual(_current_cycle_progress(store, run)["current_cycle_attempts"], 1)
                run["status"] = "stopped"
                self.assertFalse(_current_cycle_progress(store, run)["current_cycle_active"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
