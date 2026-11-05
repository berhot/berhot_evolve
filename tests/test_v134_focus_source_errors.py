"""Offline regression tests for honest worker source failures and delayed retries."""
import copy
import tempfile
import unittest

from evolve.focus_data import get_profile
from evolve.focus_discovery import relevant_links
from evolve.focus_overnight import (
    _run_page, _source_error_policy, _publish, cycle, init_state,
)
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import WebSourceError


PROFILE = "saudi-history-ar"


def source_html():
    article = (
        "تاريخ المملكة العربية السعودية والدرعية والدولة السعودية الأولى. "
        * 75
    )
    return (
        "<html><head><title>تاريخ المملكة العربية السعودية</title></head>"
        "<body><article>" + article + "</article></body></html>"
    )


class FocusSourceFailureTests(unittest.TestCase):
    def setup(self, root):
        store = Store(root)
        settings = copy.deepcopy(load_config())
        settings["storage_root"] = root
        settings["crawl_delay_seconds"] = 0
        state = init_state(store, PROFILE)
        state["until_stopped"] = True
        state["no_fixed_quotas"] = True
        state["minimum_host_gap_seconds"] = 0
        state["_queue"] = [
            {"url": "https://saudipedia.com/article-test", "depth": 0}
        ]
        state["_seen"] = set()
        state["_host_last_at"] = {}
        return store, settings, state

    def test_discovery_skips_home_and_tag_navigation(self):
        profile = get_profile(PROFILE)
        html = (
            '<html><body>'
            '<a href="https://www.spa.gov.sa/">تاريخ المملكة السعودية</a>'
            '<a href="https://www.darah.org.sa/">الدرعية تاريخ السعودية</a>'
            '<a href="/tag/history">تاريخ المملكة السعودية</a>'
            '<a href="/category/saudi">تاريخ المملكة السعودية</a>'
            '<a href="https://www.spa.gov.sa/ar/N2520902">'
            'تاريخ المملكة العربية السعودية الدرعية</a>'
            '</body></html>'
        )
        links = relevant_links(
            html, "https://saudipedia.com/saudi-history", profile)
        self.assertIn("https://www.spa.gov.sa/ar/N2520902", links)
        self.assertNotIn("https://www.spa.gov.sa/", links)
        self.assertNotIn("https://www.darah.org.sa/", links)
        self.assertFalse(any("/tag/" in url or "/category/" in url
                             for url in links))

    def test_classification_never_retries_denials_or_redirects(self):
        expected = {
            "HTTP 404": ("http_404_not_found", False),
            "HTTP 403": ("http_403_access_denied", False),
            "HTTP 301": ("redirect_not_followed", False),
            "Website disallows automated access to this URL":
                ("robots_disallowed", False),
            "HTTP 429": ("http_429_rate_limited", True),
            "HTTP 503": ("http_503_server_error", True),
            "Cannot verify robots.txt: test":
                ("robots_unverifiable", True),
        }
        for message, (reason, retry) in expected.items():
            with self.subTest(message=message):
                result = _source_error_policy(WebSourceError(message))
                self.assertEqual(result[0], reason)
                self.assertEqual(result[1], retry)

    def test_permanent_denial_logged_with_specific_reason_no_retry(self):
        with tempfile.TemporaryDirectory() as root:
            store, settings, state = self.setup(root)
            try:
                def denied(_url):
                    raise WebSourceError("Website disallows automated access to this URL")
                result = _run_page(
                    store, settings, state, get_profile(PROFILE),
                    fetcher=denied, time_fn=lambda: 1000.0)
                self.assertEqual(result["error"], "robots_disallowed")
                self.assertFalse(result["retry_scheduled"])
                self.assertEqual(state["_queue"], [])
                self.assertIn(result["url"], state["_seen"])
                self.assertEqual(store.focus_activity(PROFILE)["events"][-1]
                                 ["detail"],
                                 "robots_disallowed · not automatically retried")
            finally:
                store.close()

    def test_temporary_server_failure_retries_only_after_delay(self):
        with tempfile.TemporaryDirectory() as root:
            store, settings, state = self.setup(root)
            try:
                attempted = []
                def temporarily_bad(url):
                    attempted.append(url)
                    raise WebSourceError("HTTP 503")
                first = _run_page(
                    store, settings, state, get_profile(PROFILE),
                    fetcher=temporarily_bad, time_fn=lambda: 1000.0)
                self.assertEqual(first["error"], "http_503_server_error")
                self.assertTrue(first["retry_scheduled"])
                self.assertEqual(first["retry_after_seconds"], 300)
                self.assertEqual(state["_queue"][0]["retry_count"], 1)
                self.assertEqual(state["_queue"][0]["retry_at"], 1300)
                self.assertNotIn(first["url"], state["_seen"])
                deferred = _run_page(
                    store, settings, state, get_profile(PROFILE),
                    fetcher=lambda _: self.fail("cannot retry too early"),
                    time_fn=lambda: 1200.0)
                self.assertEqual(deferred["status"], "source_retry_wait")
                self.assertEqual(state["pages_attempted"], 1)
                allowed = _run_page(
                    store, settings, state, get_profile(PROFILE),
                    fetcher=lambda _: source_html(),
                    time_fn=lambda: 1301.0)
                self.assertEqual(allowed["status"], "archived_research_only")
                self.assertEqual(state["pages_attempted"], 2)
                self.assertEqual(len(attempted), 1)
            finally:
                store.close()

    def test_retry_exhaustion_and_queue_survives_worker_restart(self):
        with tempfile.TemporaryDirectory() as root:
            store, settings, state = self.setup(root)
            try:
                def bad(_):
                    raise WebSourceError("HTTP 502")
                for now in (1000.0, 1301.0, 1902.0):
                    _run_page(
                        store, settings, state, get_profile(PROFILE),
                        fetcher=bad, time_fn=lambda now=now: now)
                self.assertEqual(state["pages_attempted"], 3)
                self.assertEqual(state["_queue"], [])
                self.assertIn("https://saudipedia.com/article-test",
                              state["_seen"])
                self.assertFalse(state["last_errors"][-1]["retry_scheduled"])
                # A separate pending URL survives the saved frontier format.
                state["_queue"] = [{
                    "url": "https://saudipedia.com/article-retry",
                    "depth": 0, "retry_count": 1, "retry_at": 2400,
                }]
                store.put("focus_frontier_" + PROFILE, {
                    "day": state["pages_day_utc"],
                    "seen": sorted(state["_seen"]),
                    "pending": state["_queue"],
                    "host_counts": {},
                    "pages_attempted_today": state["pages_attempted_today"],
                    "host_last_at": state["_host_last_at"],
                })
                restored = init_state(store, PROFILE)
                self.assertTrue(any(
                    x["url"] == "https://saudipedia.com/article-retry"
                    and x.get("retry_count") == 1
                    and x.get("retry_at") == 2400
                    for x in restored["_queue"]))
            finally:
                store.close()

    def test_deferred_only_frontier_yields_to_next_cycle_and_live_count(self):
        with tempfile.TemporaryDirectory() as root:
            store, settings, state = self.setup(root)
            try:
                state["_queue"][0]["retry_at"] = 9000.0
                _publish(store, state, "ready")
                self.assertEqual(store.get("focus_run_" + PROFILE)[
                    "pending_research_urls"], 1)
                result = cycle(
                    store, settings, state, pages=30, train_steps=1,
                    fetcher=lambda _: self.fail("no premature retry"),
                    sleep_fn=lambda _: None, time_fn=lambda: 1000.0)
                self.assertEqual(result["pages_attempted_this_cycle"], 0)
                self.assertEqual(result["outcomes"][0]["status"],
                                 "source_retry_wait")
                self.assertEqual(result["pending_source_urls"], 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
