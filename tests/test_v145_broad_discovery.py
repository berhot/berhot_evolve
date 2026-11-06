"""Broader discovery must reach independent origins without bypassing checks."""
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from evolve.focus_data import get_profile
from evolve.focus_discovery import relevant_links, publisher_key, _queries_from_observations
from evolve.focus_overnight import init_state, _run_page, cycle, run
from evolve.focus_sitemaps import discover_publisher_sitemaps
from evolve.focus_browser_search import headless_search
from evolve.focus_search_api import brave_search
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import extract

PROFILE = "saudi-history-ar"
TEXT = "تاريخ المملكة العربية السعودية والدولة السعودية الأولى والدرعية. " * 70
HTML = "<title>تاريخ السعودية</title><article>" + TEXT + "</article>"


class BroadDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name)
        self.settings = copy.deepcopy(load_config())
        self.settings.update(storage_root=self.tmp.name, crawl_delay_seconds=0)
        self.profile = get_profile(PROFILE)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def state(self):
        state = init_state(self.store, PROFILE)
        state.update(until_stopped=True, no_fixed_quotas=True,
                     minimum_host_gap_seconds=0)
        return state

    def test_external_opaque_citation_survives_dense_internal_links(self):
        local = "".join('<a href="/تاريخ-السعودية-%d">تاريخ الدولة السعودية الأولى</a>' % n
                        for n in range(70))
        external = ('<a href="https://new-publisher.example.org/article/123">المصدر</a>'
                    '<a href="https://other-history.example.net/saudi-history">تاريخ السعودية</a>'
                    '<a href="https://facebook.com/post/1">المصدر</a>'
                    '<a href="https://bad.example.org/login">المصدر</a>')
        result = relevant_links(HTML + local + external,
                                "https://saudipedia.com/start", self.profile, max_results=6)
        self.assertIn("https://new-publisher.example.org/article/123", result)
        self.assertIn("https://other-history.example.net/saudi-history", result)
        self.assertEqual(len(result), 6)
        self.assertFalse(any("facebook.com" in x or "/login" in x for x in result))
        self.assertEqual(relevant_links(external[:external.index('</a>')+4],
                                       "https://saudipedia.com/start", self.profile), [])

    def test_underrepresented_publisher_precedes_dominant_queue(self):
        state = self.state()
        state["_publisher_attempts"] = {"saudipedia.com": 100}
        state["_queue"] = [
            {"url": "https://saudipedia.com/saudi-history", "depth": 0},
            {"url": "https://independent.example.org/saudi-history", "depth": 0}]
        result = _run_page(self.store, self.settings, state, self.profile,
                           fetcher=lambda _: HTML)
        self.assertEqual(result["url"], "https://independent.example.org/saudi-history")
        self.assertEqual(state["_publisher_attempts"]["example.org"], 1)
        self.assertEqual(publisher_key("https://sites.moe.gov.sa/a"), "moe.gov.sa")

    def test_fresh_publisher_is_not_starved_by_cached_sitemap(self):
        profile = dict(self.profile, seeds=("https://saudipedia.com/saudi-history",
                                         "https://independent.example.org/saudi-history"))
        backlog_key = "focus_sitemap_backlog_" + profile["name_ar"]
        old = ["https://saudipedia.com/saudi-history-%d" % n for n in range(40)]
        self.store.put(backlog_key, old)
        self.store.put("focus_sitemap_checks_" + profile["name_ar"], {
            "saudipedia.com": {"checked_at": 1000, "scan_version": 2}})
        calls = []
        def probe(host):
            calls.append(host)
            return {"status": "sitemap_urls_found", "url_candidates": [
                "https://" + host + "/saudi-history-new"], "network_requests": 0}
        result = discover_publisher_sitemaps(
            self.store, profile, [], sitemap_fn=probe, now=1001,
            prefer_new_publishers=True)
        self.assertEqual(calls, ["independent.example.org"])
        self.assertIn("independent.example.org", result["url_candidates"][0])
        self.assertEqual(self.store.get(backlog_key), old)

    def test_retrying_queue_does_not_block_other_publisher_discovery(self):
        state = self.state()
        state["_queue"] = [{"url": "https://saudipedia.com/saudi-history",
                            "depth": 0, "retry_at": 5000}]
        state["_host_cooldown_until"] = {"saudipedia.com": 5000}
        calls = []
        def probe(host):
            calls.append(host)
            return {"status": "sitemap_urls_found", "network_requests": 0,
                    "url_candidates": ["https://" + host + "/saudi-history"]}
        result = cycle(self.store, self.settings, state, pages=1, train_steps=1,
                       fetcher=lambda _: HTML, sitemap_fn=probe,
                       time_fn=lambda: 1000, sleep_fn=lambda _: None)
        self.assertEqual(calls, ["www.spa.gov.sa"])
        self.assertEqual(result["pages_attempted_this_cycle"], 1)
        self.assertEqual(state["_queue"][0]["retry_at"], 5000)

    def test_dominant_publisher_generates_independent_queries(self):
        observations = [{"url": "https://saudipedia.com/a"}] * 5
        normal = _queries_from_observations(self.profile, observations)
        broad = _queries_from_observations(self.profile, observations, diversify=True)
        self.assertEqual(len(normal), 4)
        self.assertTrue(broad[0].endswith("-site:saudipedia.com"))
        self.assertEqual(broad[1], normal[0])

    def test_official_search_api_results_and_key_handling(self):
        class Response(io.BytesIO):
            def geturl(inner):
                return request[0].full_url
        class Opener:
            def open(inner, req, timeout):
                request.append(req)
                return Response(json.dumps({"web": {"results": [
                    {"url": "https://unseen.example.net/123", "title": "تاريخ السعودية"}]
                }}).encode())
        request = []
        result = brave_search("تاريخ السعودية", api_key="test-key", opener=Opener())
        self.assertEqual(result["status"], "search_completed")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(request[0].get_header("X-subscription-token"), "test-key")
        self.assertNotIn("test-key", request[0].full_url)
        self.assertEqual(brave_search("q", api_key="")["status"], "search_api_key_missing")

    def test_auto_uses_api_only_when_configured(self):
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "configured"}), patch(
                "evolve.focus_search_api.brave_search", return_value={"status": "search_completed"}) as api:
            headless_search("تاريخ السعودية", provider="auto")
            api.assert_called_once()
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}):
            result = headless_search("تاريخ السعودية", provider="auto", robots_check=lambda _: False)
            self.assertEqual(result["status"], "search_engine_robots_disallow_or_unverifiable")

    def test_worker_accepts_auto_and_api_provider_at_startup(self):
        # Advance beyond the finite duration before any cycle/network work.
        for provider in ('auto', 'brave'):
            clock = iter([0.0, 10.0, 10.0, 10.0])
            with patch('builtins.print'):
                run(self.store, self.settings, hours=0.0001,
                    search_provider=provider, monotonic_fn=lambda: next(clock, 10.0))
            saved = self.store.get('focus_run_' + PROFILE)
            self.assertEqual(saved['browser_search_provider'], provider)
            self.assertEqual(saved['pages_attempted'], 0)

    def test_publisher_server_form_keeps_article_but_skips_controls_and_scripts(self):
        html = ('<title>تاريخ السعودية</title><form method="post" id="aspnetForm">'
                '<input type="hidden" name="__VIEWSTATE" value="hidden-state">'
                '<article>' + TEXT + '</article><script>SECRET_SCRIPT</script>'
                '<form id="feedback">FEEDBACK_CONTROL</form>'
                '<p>نهاية المقال</p></form><footer>FOOTER_CONTROL</footer>')
        text, title, *_ = extract(html)
        self.assertIn(TEXT.strip(), text)
        self.assertIn('نهاية المقال', text)
        for excluded in ('hidden-state', 'SECRET_SCRIPT', 'FEEDBACK_CONTROL', 'FOOTER_CONTROL'):
            self.assertNotIn(excluded, text)
        ordinary = extract('<form method="post" id="aspnetForm">ordinary input form</form>')[0]
        self.assertNotIn('ordinary input form', ordinary)


if __name__ == "__main__":
    unittest.main()
