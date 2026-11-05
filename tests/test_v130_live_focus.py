"""v1.30: durable Studio telemetry and refreshing does not stop its worker.

These tests use a real temporary SQLite store and an OS worker lock, without
opening a network browser or spawning a long-running daemon.
"""
import copy
from pathlib import Path
import tempfile
import unittest

from evolve.focus_control import acquire_lock, release_lock, worker_running
from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_studio import LearningController


class LiveDashboardTests(unittest.TestCase):
    def test_studio_reconnect_preserves_external_worker_and_session_start(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            lock = acquire_lock(store, "saudi-history-ar")
            controller = None
            try:
                started = "2026-11-05T01:42:00+00:00"
                state = {
                    "run_id": "focus-unchanged-on-page-refresh",
                    "started_at_utc": started,
                    "finished_at_utc": None,
                    "heartbeat_at_utc": "2026-11-05T01:45:00+00:00",
                    "status": "running",
                    "search_status": "browser_search_unavailable",
                    "last_search_diagnostic": {
                        "status": "browser_search_unavailable",
                        "provider": "duckduckgo",
                        "error_phase": "submitting_query",
                        "error_type": "TimeoutError",
                        "retry_after_seconds": 300,
                    },
                    "current_phase": "reading_article",
                    "current_url": "https://example.org/saudi-history",
                    "pages_read": 5,
                    "html_bytes_read": 19000,
                    "text_utf8_byte_tokens_extracted": 8000,
                    "unique_archived_utf8_byte_tokens": 4000,
                    "training_byte_tokens_processed": 0,
                    "current_training_steps_done": 0,
                    "current_training_steps_target": 0,
                    "current_training_loss": None,
                    "recent_activity": [{
                        "at_utc": "2026-11-05T01:45:00+00:00",
                        "kind": "article_read",
                        "url": "https://example.org/saudi-history",
                        "detail": "8000 UTF-8 byte tokens",
                    }],
                }
                store.put("focus_run_saudi-history-ar", state)
                controller = LearningController(cfg, interval=60)
                one = controller.status()
                two = controller.status()  # same operation a reload performs
                for result in (one, two):
                    f = result["focus_agent"]
                    self.assertTrue(f["process_running"])
                    self.assertEqual(f["run_id"], state["run_id"])
                    self.assertEqual(f["started_at_utc"], started)
                    self.assertEqual(f["pages_read"], 5)
                    self.assertEqual(f["text_utf8_byte_tokens_extracted"], 8000)
                    self.assertEqual(f["training_byte_tokens_processed"], 0)
                    self.assertEqual(f["current_phase"], "reading_article")
                    self.assertEqual(f["search_status"],
                                     "browser_search_unavailable")
                    self.assertEqual(f["search_diagnostic"]["error_phase"],
                                     "submitting_query")
                    self.assertEqual(f["search_diagnostic"]["error_type"],
                                     "TimeoutError")
                    self.assertEqual(len(f["recent_activity"]), 1)
                self.assertTrue(worker_running(store, "saudi-history-ar"))
                # Studio teardown is unrelated to the separate worker lock.
                controller.shutdown()
                controller = None
                self.assertTrue(worker_running(store, "saudi-history-ar"))
                self.assertEqual(
                    store.get("focus_run_saudi-history-ar")["run_id"],
                    "focus-unchanged-on-page-refresh")
            finally:
                if controller is not None:
                    controller.shutdown()
                release_lock(lock)
                store.close()

    def test_html_has_restored_timer_and_live_counters_not_restart_controls(self):
        html = (Path(__file__).resolve().parents[1] / "evolve" /
                "studio.html").read_text("utf-8")
        for control in (
            'id="focus-elapsed"', 'id="focus-started"',
            'id="focus-read-pages"', 'id="focus-extracted-tokens"',
            'id="focus-unique-tokens"', 'id="focus-trained-tokens"',
            'id="focus-current-url"', 'id="focus-recent-activity"',
            'id="focus-search-health"', 'f.search_diagnostic',
            'id="focus-start"', 'id="focus-stop"',
            'setInterval(focusTimer,1000)',
            'f.started_at_utc', 'f.finished_at_utc',
            'f.text_utf8_byte_tokens_extracted',
        ):
            self.assertIn(control, html)
        self.assertIn("browser never stops", html)
        self.assertNotIn("location.reload(true)", html)


if __name__ == "__main__":
    unittest.main()
