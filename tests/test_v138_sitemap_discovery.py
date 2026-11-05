"""Sitemap discovery is topical, policy-safe and does not grant training rights."""
import copy
import tempfile
import unittest
from xml.etree.ElementTree import ParseError

from evolve.focus_data import get_profile
from evolve.focus_overnight import cycle, init_state
from evolve.focus_sitemaps import (
    _xml_links, _select_topical, _same_host_https,
    _robots_denies_wildcard, discover_publisher_sitemaps,
)
from evolve.settings import load_config
from evolve.store import Store


PROFILE = "saudi-history-ar"


def article_html():
    text = ("تاريخ المملكة العربية السعودية والدرعية والدولة السعودية الأولى "
            "والملك عبدالعزيز ويوم التأسيس. " * 70)
    return "<html><title>تاريخ الدولة السعودية</title><article>" + text + "</article></html>"


class SitemapDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = Store(self.folder.name)
        self.settings = copy.deepcopy(load_config())
        self.settings["storage_root"] = self.folder.name
        self.settings["crawl_delay_seconds"] = 0
        self.profile = get_profile(PROFILE)
        self.state = init_state(self.store, PROFILE)
        self.state["_queue"] = []
        self.state["_seen"] = set()
        self.state["until_stopped"] = True
        self.state["no_fixed_quotas"] = True
        self.state["minimum_host_gap_seconds"] = 0

    def tearDown(self):
        self.store.close()
        self.folder.cleanup()

    def test_xml_parsing_accepts_sitemap_and_rejects_dtd(self):
        simple = (b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                  b'<url><loc>https://saudipedia.com/saudi-history</loc></url>'
                  b'</urlset>')
        kind, urls = _xml_links(simple)
        self.assertEqual(kind, "urlset")
        self.assertEqual(urls, ["https://saudipedia.com/saudi-history"])
        self.assertEqual(_xml_links(
            b'<sitemapindex><sitemap><loc>https://saudipedia.com/ar.xml'
            b'</loc></sitemap></sitemapindex>')[0], "sitemapindex")
        for invalid in (
            b'<!DOCTYPE x [ <!ENTITY a "unsafe"> ]><urlset/>',
            b'<not-a-sitemap/>',
        ):
            with self.assertRaises(ValueError):
                _xml_links(invalid)

    def test_candidate_selection_is_topical_without_allowlisting_hosts(self):
        selected = _select_topical([
            "https://saudipedia.com/الدولة-السعودية-الأولى",
            "https://saudipedia.com/other-topic",
            "https://saudipedia.com/tag/saudi",
            "http://saudipedia.com/saudi-history",
            "https://www.spa.gov.sa/saudi-history",
        ], self.profile)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all("saudi" in url or "السعود" in url for url in selected))
        self.assertIsNone(_same_host_https(
            "https://another.example/saudi-history", "saudipedia.com"))
        self.assertTrue(_robots_denies_wildcard(
            ["Disallow: /site*"], "/sitemap.xml"))
        self.assertFalse(_robots_denies_wildcard(
            ["Disallow: /api/*"], "/sitemap.xml"))

    def test_sitemap_provider_candidates_are_same_host_and_paced(self):
        now = 50_000
        calls = []
        def fake_sitemap(host):
            calls.append(host)
            return {
                "status": "sitemap_urls_found",
                "network_requests": 2,
                "sitemap_documents_checked": 1,
                "url_candidates": [
                    f"https://{host}/الدولة-السعودية-الأولى",
                    "https://evil.example/saudi-history",
                    f"https://{host}/not-related",
                ],
            }
        first = discover_publisher_sitemaps(
            self.store, self.profile, [], sitemap_fn=fake_sitemap,
            now=now)
        self.assertEqual(first["status"], "sitemap_urls_found")
        self.assertEqual(first["host"], "saudipedia.com")
        self.assertEqual(len(first["url_candidates"]), 1)
        self.assertTrue(first["candidate_urls_are_not_training_rights"])
        next_publisher = discover_publisher_sitemaps(
            self.store, self.profile, [], sitemap_fn=fake_sitemap,
            now=now + 1)
        self.assertEqual(next_publisher["host"], "www.spa.gov.sa")
        self.assertEqual(calls, ["saudipedia.com", "www.spa.gov.sa"])
        again = discover_publisher_sitemaps(
            self.store, self.profile, [], sitemap_fn=fake_sitemap,
            now=now + 2)
        self.assertEqual(again["host"], "www.mofa.gov.sa")
        cooling = discover_publisher_sitemaps(
            self.store, self.profile, [], sitemap_fn=fake_sitemap,
            now=now + 3)
        self.assertEqual(cooling["status"], "sitemap_publisher_cooldown")
        self.assertEqual(len(calls), 3)
        self.assertEqual(cooling["network_requests"], 0)

    def test_large_sitemap_drains_persistent_candidate_backlog(self):
        now = 50_000
        calls = []
        def sitemap_provider(host):
            calls.append(host)
            return {
                "status": "sitemap_urls_found",
                "network_requests": 2,
                "url_candidates": [
                    f"https://{host}/تاريخ-السعودية-مقال-{i}"
                    for i in range(56)
                ],
            }
        batches = []
        for delta in range(3):
            response = discover_publisher_sitemaps(
                self.store, self.profile, [],
                sitemap_fn=sitemap_provider, now=now + delta,
                max_candidates=25)
            batches.append(response)
        self.assertEqual([len(x["url_candidates"]) for x in batches],
                         [25, 25, 6])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [x["backlog_remaining"] for x in batches], [31, 6, 0])
        self.assertEqual(
            [x["network_requests"] for x in batches], [2, 0, 0])
        all_urls = [url for batch in batches
                    for url in batch["url_candidates"]]
        self.assertEqual(len(set(all_urls)), 56)
        fourth = discover_publisher_sitemaps(
            self.store, self.profile, [],
            sitemap_fn=sitemap_provider, now=now + 3,
            max_candidates=25)
        self.assertEqual(fourth["host"], "www.spa.gov.sa")
        self.assertEqual(len(calls), 2)

    def test_cycle_uses_sitemap_candidates_after_queue_is_empty(self):
        attempted = []
        def fake_sitemap(host):
            return {
                "status": "sitemap_urls_found",
                "url_candidates": [
                    f"https://{host}/الدولة-السعودية-الأولى"],
                "network_requests": 0,
            }
        def fetch_article(url):
            attempted.append(url)
            return article_html()
        result = cycle(
            self.store, self.settings, self.state,
            pages=1, train_steps=1, fetcher=fetch_article,
            sitemap_fn=fake_sitemap,
            sleep_fn=lambda seconds: None)
        self.assertEqual(result["pages_attempted_this_cycle"], 1)
        self.assertEqual(result["publisher_sitemap_discovery"]
                         ["new_urls_queued"], 1)
        self.assertEqual(len(attempted), 1)
        self.assertTrue(self.state["pages_read"])
        self.assertEqual(self.state["training_steps_this_run"], 0)
        self.assertTrue(any(
            item["kind"] == "publisher_sitemap_result"
            for item in self.store.focus_activity(PROFILE)["events"]))

    def test_offline_fixture_never_makes_sitemap_network_request(self):
        result = cycle(
            self.store, self.settings, self.state,
            pages=1, train_steps=1,
            fetcher=lambda url: self.fail("No URLs to fetch"),
            sleep_fn=lambda seconds: None)
        self.assertEqual(result["pages_attempted_this_cycle"], 0)
        self.assertIsNone(result["publisher_sitemap_discovery"])
        self.assertEqual(result["idle_reason"],
                         "no_pending_urls_and_search_produced_no_candidates")

    def test_sitemap_failure_is_paced_not_retried_each_cycle(self):
        called = []
        def fails(host):
            called.append(host)
            raise RuntimeError("network unavailable")
        first = discover_publisher_sitemaps(
            self.store, self.profile, [], sitemap_fn=fails, now=50_000)
        self.assertEqual(first["status"], "sitemap_unavailable_RuntimeError")
        self.assertEqual(first["network_requests"], 0)
        again = discover_publisher_sitemaps(
            self.store, self.profile, [], sitemap_fn=fails, now=50_001)
        self.assertEqual(again["host"], "www.spa.gov.sa")
        self.assertEqual(len(called), 2)


if __name__ == "__main__":
    unittest.main()
