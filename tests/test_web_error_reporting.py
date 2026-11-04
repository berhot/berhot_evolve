"""Regressions: crawl failures must be visible, durable and non-destructive."""
import tempfile
import unittest

from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import WebSourceError, discover_web, register_exploration


SEED = "https://lessons.example.org/alphabet"


class WebErrorReportingTests(unittest.TestCase):
    def test_crawl_failure_reports_reason_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            settings = load_config()
            settings.update(storage_root=tmp, crawl_delay_seconds=0, daily_request_budget=25)
            register_exploration(store, SEED)

            def denied(_url):
                raise WebSourceError("Robots availability uncertain (HTTP 403)")

            report = discover_web(store, settings, limit=3, fetcher=denied,
                                  sleep_fn=lambda _: None)
            self.assertEqual(report["attempted"], 1)
            self.assertEqual(report["errors"], 1)
            self.assertEqual(report["error_details"], [
                {"url": SEED, "reason": "Robots availability uncertain (HTTP 403)"}
            ])
            self.assertEqual(store.status()["web_error_urls"], 1)
            self.assertEqual(store.status()["web_pending_urls"], 0)
            self.assertEqual(store.status()["concepts_discovered"], 0)
            self.assertEqual(store.recent_web_errors(1)[0]["last_error"],
                             "Robots availability uncertain (HTTP 403)")
            store.close()

            reopened = Store(tmp)
            self.assertEqual(reopened.recent_web_errors()[0]["url"], SEED)
            reopened.close()

    def test_no_errors_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            self.assertEqual(store.recent_web_errors(), [])
            self.assertEqual(store.status()["web_error_urls"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
