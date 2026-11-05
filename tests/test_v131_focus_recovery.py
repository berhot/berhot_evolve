"""Regression tests for autonomous overnight research recovery (all offline)."""
import copy
import tempfile
import unittest
from unittest.mock import patch

from evolve.focus_browser_search import headless_search
from evolve.focus_data import get_profile
from evolve.focus_discovery import _candidate_link, run_one_search
from evolve.focus_overnight import _run_page, init_state
from evolve.settings import load_config
from evolve.store import Store


class FocusRecoveryTests(unittest.TestCase):
    def test_seed_urls_are_direct_hyphenated_article_slugs(self):
        profile = get_profile("saudi-history-ar")
        urls = profile["seeds"]
        saudipedia = [url for url in urls if "saudipedia.com" in url]
        self.assertGreaterEqual(len(saudipedia), 4)
        for url in urls:
            self.assertEqual(_candidate_link(url), url)
            self.assertNotIn("%20", url)
            self.assertNotIn("[https://", url)
        self.assertTrue(any("saudipedia.com/" in url and "%D8" in url
                            for url in saudipedia))

    def test_failed_queries_are_not_exhausted_and_failure_is_diagnosed(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = folder
            profile = get_profile("saudi-history-ar")
            attempted = []

            def broken_search(query):
                attempted.append(query)
                raise TimeoutError("offline simulated backend outage")

            try:
                for i in range(5):
                    # Fake wall time enforces normal 180-second pacing.
                    with patch("evolve.focus_discovery.time.time",
                               return_value=1000.0 + i * 200):
                        result = run_one_search(
                            store, cfg, profile, [],
                            search_fn=broken_search, open_ended=True)
                    self.assertEqual(result["status"], "search_failed", result)
                    self.assertEqual(result["error_type"], "TimeoutError")
                    self.assertEqual(result["queries_attempted"], 1)
                    self.assertGreater(result["retry_after_seconds"], 0)
                self.assertEqual(len(set(attempted[:4])), 4)
                self.assertEqual(attempted[0], attempted[4])
                day = __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).strftime("%Y-%m-%d")
                self.assertEqual(
                    store.get("focus_search_success_" + day, []), [])
                self.assertGreaterEqual(len(
                    store.get("focus_search_failures_" +
                              profile["name_ar"]) or {}), 4)
            finally:
                store.close()

    def test_browser_timeout_has_safe_failure_phase(self):
        class BrokenPage:
            def goto(self, *args, **kwargs):
                raise TimeoutError("sensitive test path")

            def close(self):
                pass

        class BrokenBrowser:
            def new_page(self, **kwargs):
                return BrokenPage()

            def close(self):
                pass

        result = headless_search(
            "الدولة السعودية الأولى", "duckduckgo",
            robots_check=lambda _: True,
            browser_factory=lambda: BrokenBrowser())
        self.assertEqual(result["status"], "browser_search_unavailable")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertEqual(result["error_phase"], "opening_search_home")
        self.assertNotIn("sensitive test path", str(result))

    def test_off_topic_arabic_hub_can_yield_relevant_article_links(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = folder
            profile = get_profile("saudi-history-ar")
            state = init_state(store, "saudi-history-ar")
            state["minimum_host_gap_seconds"] = 0
            html = (
                "<html><title>الأخبار العامة</title><body>"
                "<article>" +
                ("يتحدث هذا التقرير عن الطقس والأجهزة والرياضة والتقنية. " * 12) +
                "</article>"
                '<a href="https://history.example.org/saudi-history">'
                "اقرأ التفاصيل</a>"
                "</body></html>"
            )
            try:
                result = _run_page(
                    store, cfg, state, profile, fetcher=lambda _: html)
                self.assertEqual(result["status"], "off_topic_or_non_arabic")
                self.assertEqual(result["reason"], "off_topic")
                self.assertGreaterEqual(result["related_links_queued"], 1)
                self.assertIn(
                    "https://history.example.org/saudi-history",
                    [item["url"] for item in state["_queue"]])
                self.assertEqual(state["pages_archived"], 0)
                self.assertEqual(state["training_steps_this_run"], 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
