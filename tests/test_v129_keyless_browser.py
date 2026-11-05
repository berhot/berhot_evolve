"""Offline contract tests for keyless browser search and open-ended crawling."""
import copy
import tempfile
import unittest

from evolve.focus_browser_search import (
    headless_search, search_engines, _robots_allows,
)
from evolve.focus_data import get_profile
from evolve.focus_discovery import run_one_search
from evolve.focus_overnight import init_state, cycle
from evolve.settings import load_config
from evolve.store import Store


class _FakeLocator:
    def __init__(self, page, mode, index=0):
        self.page, self.mode, self.index = page, mode, index

    @property
    def first(self):
        return self

    def fill(self, value, timeout=None):
        self.page.entered = value

    def press(self, value, timeout=None):
        assert value == "Enter"
        self.page.url = "https://duckduckgo.com/?q=saudi"

    def wait_for(self, state="visible", timeout=None):
        pass

    def count(self):
        return len(self.page.results) if self.mode == "results" else 1

    def nth(self, index):
        return _FakeLocator(self.page, self.mode, index)

    def get_attribute(self, key, timeout=None):
        if self.mode != "results":
            return None
        return self.page.results[self.index]["url"]

    def inner_text(self, timeout=None):
        if self.mode == "body":
            return self.page.body
        if self.mode == "results":
            return self.page.results[self.index]["title"]
        return ""


class _FakePage:
    def __init__(self, challenge=False):
        self.url = "https://duckduckgo.com/"
        self.entered = ""
        self.body = "Verify you are human" if challenge else "Results for Saudi history"
        self.results = [
            {"url": "https://example.edu/saudi-history",
             "title": "تاريخ المملكة العربية السعودية الدرعية"},
            {"url": "https://another.org/diriyah",
             "title": "الدولة السعودية الأولى في الدرعية"},
        ]
        self.closed = False

    def goto(self, url, wait_until=None, timeout=None):
        self.url = url

    def locator(self, selector):
        if selector == "body":
            return _FakeLocator(self, "body")
        if selector.startswith("input"):
            return _FakeLocator(self, "form")
        return _FakeLocator(self, "results")

    def title(self):
        return "Challenge" if "Verify" in self.body else "Search results"

    def wait_for_load_state(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self, challenge=False):
        self.page = _FakePage(challenge)
        self.closed = False

    def new_page(self, **kwargs):
        assert kwargs["accept_downloads"] is False
        return self.page

    def close(self):
        self.closed = True


class BrowserPolicyTests(unittest.TestCase):
    def test_google_and_bing_blocked_no_stealth_or_api_key(self):
        status = search_engines()
        self.assertFalse(status["google"]["headless_supported"])
        self.assertFalse(status["bing"]["headless_supported"])
        self.assertTrue(status["duckduckgo"]["headless_supported"])
        for engine in ("google", "bing"):
            result = headless_search(
                "تاريخ السعودية", engine,
                robots_check=lambda url: self.fail("robots not queried"),
                browser_factory=lambda: self.fail("browser not launched"))
            self.assertEqual(
                result["status"], "search_engine_policy_disallows_automation")
            self.assertEqual(result["results"], [])

    def test_respects_live_robots_and_never_opens_browser_when_forbidden(self):
        r = headless_search(
            "تاريخ السعودية", "duckduckgo",
            robots_check=lambda url: False,
            browser_factory=lambda: self.fail("must not launch"))
        self.assertEqual(r["status"], "search_engine_robots_disallow_or_unverifiable")
        self.assertFalse(_robots_allows(
            "https://duckduckgo.com/html?q=x",
            fetch_robots=lambda _: "User-agent: *\nDisallow: /html"))
        self.assertTrue(_robots_allows(
            "https://duckduckgo.com/?q=x",
            fetch_robots=lambda _: "User-agent: *\nDisallow: /html\nAllow: /"))
        # DuckDuckGo currently disallows search-result URLs via wildcard.
        # urllib.robotparser omits '*' semantics, so enforce fail-closed.
        self.assertFalse(_robots_allows(
            "https://duckduckgo.com/?q=saudi",
            fetch_robots=lambda _: (
                "User-agent: *\n"
                "Disallow: /lite\n"
                "Disallow: /html\n"
                "Disallow: /*?\n"
                "Allow: /search.html\n"
            )))
        self.assertFalse(_robots_allows(
            "https://duckduckgo.com/?q=saudi",
            fetch_robots=lambda _: (
                "User-agent: *\nDisallow: /*?\n"
            )))

    def test_headless_browser_returns_only_links_not_facts(self):
        browser = _FakeBrowser()
        result = headless_search(
            "تاريخ الدولة السعودية الأولى", "duckduckgo",
            robots_check=lambda url: True,
            browser_factory=lambda: browser)
        self.assertEqual(result["status"], "search_completed", result)
        self.assertEqual(len(result["results"]), 2)
        self.assertFalse(result["browser_visible"])
        self.assertFalse(result["search_snippets_are_facts"])
        self.assertTrue(browser.closed)
        self.assertTrue(browser.page.closed)
        blocked = headless_search(
            "تاريخ السعودية", "duckduckgo",
            robots_check=lambda url: True,
            browser_factory=lambda: _FakeBrowser(challenge=True))
        self.assertEqual(blocked["status"], "search_blocked_or_challenged")
        self.assertEqual(blocked["results"], [])


class OpenEndedResearchTests(unittest.TestCase):
    def test_no_fixed_page_budget_but_training_still_requires_reviews(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                cfg = copy.deepcopy(load_config())
                cfg["storage_root"] = folder
                cfg["daily_request_budget"] = 0
                cfg["crawl_delay_seconds"] = 0
                state = init_state(store, "saudi-history-ar", max_pages=1)
                state["until_stopped"] = True
                state["no_fixed_quotas"] = True
                state["minimum_host_gap_seconds"] = 0
                seen = []
                text = (
                    "تاريخ المملكة العربية السعودية وتأسيس الدولة السعودية "
                    "الأولى في مدينة الدرعية وفترة حكم الملك عبدالعزيز. " * 28)
                def fetch(url):
                    seen.append(url)
                    return "<html><head><title>تاريخ المملكة العربية السعودية</title>" \
                           "</head><body><article>" + text + \
                           "</article></body></html>"
                outcome = cycle(
                    store, cfg, state, pages=3, train_steps=1,
                    fetcher=fetch, sleep_fn=lambda _: None)
                self.assertEqual(outcome["status"], "cycle_completed")
                self.assertEqual(state["pages_attempted"], 3)
                self.assertGreater(
                    state["pages_attempted_today"], state["max_pages_per_utc_day"])
                self.assertEqual(state["training_steps_this_run"], 0)
                self.assertEqual(len(seen), 3)
                self.assertEqual(state["search_status"],
                                 "offline_test_no_search_provider")
            finally:
                store.close()

    def test_open_ended_search_not_limited_by_small_shared_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = folder
            cfg["daily_request_budget"] = 0
            profile = get_profile("saudi-history-ar")
            try:
                r = run_one_search(
                    store, cfg, profile, [],
                    search_fn=lambda q: [{
                        "url": "https://edu.example.org/saudi",
                        "title": "تاريخ الدولة السعودية الأولى والدرعية",
                        "snippet": "تاريخ السعودية الملك عبدالعزيز",
                    }],
                    open_ended=True)
                self.assertEqual(r["status"], "search_completed")
                self.assertEqual(r["queries_attempted"], 1)
                self.assertTrue(r["url_candidates"])
                next_r = run_one_search(
                    store, cfg, profile, [], search_fn=lambda q: [],
                    open_ended=True)
                self.assertEqual(next_r["status"], "search_pacing_wait")
                self.assertEqual(next_r["queries_attempted"], 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
