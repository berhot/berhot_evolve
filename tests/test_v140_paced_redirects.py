"""Canonical publisher redirects are queued, access-checked and paced."""
import io
import tempfile
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote

from evolve import web_discovery
from evolve.focus_data import get_profile
from evolve.focus_discovery import _candidate_link
from evolve.focus_overnight import _run_page, init_state, retry_saved_redirects
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import SourceRedirect, WebSourceError, fetch_page
from test_v139_unicode_article_fetch import AsciiTransport, HTML


OLD = "https://saudipedia.com/article/123/history/old"
NEW = "https://saudipedia.com/المكتبات-في-عهد-الملك-عبدالعزيز"


class RedirectTransport(AsciiTransport):
    def __init__(self, target, robots="User-agent: *\nAllow: /\n"):
        super().__init__(robots)
        self.target = target

    def open(self, request, timeout):
        if request.full_url == OLD:
            self.requests.append(request.full_url)
            headers = Message()
            headers["Location"] = self.target
            raise HTTPError(OLD, 301, "Moved", headers, io.BytesIO())
        return super().open(request, timeout)


class PacedRedirectTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(self.folder.name)
        self.addCleanup(self.store.close)
        self.settings = load_config()
        self.settings["storage_root"] = self.folder.name
        self.state = init_state(self.store, "saudi-history-ar")
        self.state.update(no_fixed_quotas=True, until_stopped=True)
        self.state["_queue"] = [{"url": OLD, "depth": 0}]
        self.profile = get_profile("saudi-history-ar")
        for patcher in (patch.object(web_discovery, "check_public_dns"),
                        patch.dict(web_discovery._ROBOTS_CACHE, {}, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_actual_fetcher_redirect_then_pacing_then_archive(self):
        transport = RedirectTransport(NEW)
        with patch.object(web_discovery, "make_fetcher", return_value=transport):
            first = _run_page(self.store, self.settings, self.state, self.profile,
                              time_fn=lambda: 1000.0)
            self.assertEqual(first["status"], "redirect_queued")
            self.assertEqual(self.state["errors"], 0)
            self.assertEqual(len(transport.requests), 2)  # robots and original URL
            wait = _run_page(self.store, self.settings, self.state, self.profile,
                             time_fn=lambda: 1001.0)
            self.assertEqual(wait["status"], "publisher_pacing_wait")
            self.assertEqual(len(transport.requests), 2)
            last = _run_page(self.store, self.settings, self.state, self.profile,
                             time_fn=lambda: 1021.0)
        self.assertEqual(last["status"], "archived_research_only")
        self.assertEqual(last["url"], NEW)
        self.assertEqual(self.state["pages_read"], 1)

    def test_destination_robots_checked_and_cross_host_or_http_refused(self):
        transport = RedirectTransport(NEW, "User-agent: *\nDisallow: " + quote(NEW.split('.com')[1]) + "\n")
        with self.assertRaises(SourceRedirect):
            fetch_page(OLD, self.settings, opener=transport)
        with self.assertRaisesRegex(WebSourceError, "disallows automated access"):
            fetch_page(NEW, self.settings, opener=transport)
        for target in ("https://other.example.org/story", "http://saudipedia.com/story",
                       "https://127.0.0.1/secret", "https://saudipedia.com:8443/story",
                       "https://user:pw@saudipedia.com/story"):
            with self.subTest(target=target), self.assertRaises(WebSourceError):
                fetch_page(OLD, self.settings, opener=RedirectTransport(target))

    def test_known_unicode_alias_not_fetched_again_and_loops_are_bounded(self):
        self.state["_seen"].add(NEW)
        result = _run_page(self.store, self.settings, self.state, self.profile,
                          fetcher=lambda _: (_ for _ in ()).throw(SourceRedirect(OLD, quote(NEW, safe=':/'))),
                          time_fn=lambda: 1000.0)
        self.assertEqual(result["status"], "redirect_already_known")
        self.assertEqual(self.state["_queue"], [])
        self.state["_queue"] = [{"url": NEW, "depth": 0, "redirect_chain": [OLD]}]
        self.state["_seen"].discard(NEW)
        result = _run_page(self.store, self.settings, self.state, self.profile,
                          fetcher=lambda _: (_ for _ in ()).throw(SourceRedirect(NEW, OLD)),
                          time_fn=lambda: 1021.0)
        self.assertEqual(result["error"], "redirect_not_followed")
        self.assertEqual(self.state["_queue"], [])

    def test_whitespace_only_discovery_path_rejected(self):
        self.assertIsNone(_candidate_link("https://www.spa.gov.sa/%20"))
        self.assertIsNotNone(_candidate_link("https://www.spa.gov.sa/ar/article"))

    def test_saved_redirect_recovery_preserves_other_terminal_failures(self):
        denied = "https://saudipedia.com/private-history"
        self.store.put("focus_frontier_saudi-history-ar", {
            "seen": [OLD, denied], "pending": [], "host_last_at": {"saudipedia.com": 1000}})
        for url, reason in ((OLD, "redirect_not_followed"), (denied, "robots_disallowed")):
            self.store.append_focus_activity("saudi-history-ar", self.state["run_id"],
                self.state["started_at_utc"], "source_error", url=url,
                detail=reason + " · not automatically retried")
        result = retry_saved_redirects(self.store)
        self.assertEqual(result["redirect_sources_requeued"], 1)
        frontier = self.store.get("focus_frontier_saudi-history-ar")
        self.assertEqual(frontier["pending"], [{"url": OLD, "depth": 0}])
        self.assertEqual(frontier["seen"], [denied])
        self.assertEqual(frontier["host_last_at"], {"saudipedia.com": 1000})
        self.assertEqual(retry_saved_redirects(self.store)["redirect_sources_requeued"], 0)

    def test_raw_utf8_location_header_uses_urllib_normalized_target(self):
        headers = Message()
        headers["Location"] = NEW.encode("utf-8").decode("latin-1")
        request = web_discovery.urllib.request.Request(OLD)
        handler = web_discovery.NoRedirect()
        with self.assertRaises(web_discovery.RedirectResponse) as caught:
            handler.http_error_302(request, io.BytesIO(), 301, "Moved", headers)
        normalized = caught.exception
        self.assertEqual(normalized.target_url, quote(NEW, safe=":/"))
        class Transport(AsciiTransport):
            def open(self, request, timeout):
                if request.full_url == OLD:
                    raise normalized
                return super().open(request, timeout)
        with self.assertRaises(SourceRedirect) as redirect:
            fetch_page(OLD, self.settings, opener=Transport())
        self.assertEqual(redirect.exception.target_url, quote(NEW, safe=":/"))


if __name__ == "__main__":
    unittest.main()
