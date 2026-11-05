"""Exercise Unicode article URLs through the real fetch/worker code paths."""
import io
import tempfile
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.parse import quote, urlsplit

from evolve import web_discovery
from evolve.focus_data import get_profile
from evolve.focus_overnight import _run_page, init_state
from evolve.settings import load_config
from evolve.store import Store


ARTICLE = "https://saudipedia.com/المكتبات-في-عهد-الملك-عبدالعزيز"
HTML = (
    "<html><head><title>تاريخ المملكة العربية السعودية</title></head>"
    "<body><article>" +
    "تاريخ المملكة العربية السعودية والدرعية والدولة السعودية الأولى. " * 75 +
    "</article></body></html>"
)


class Response(io.BytesIO):
    def __init__(self, text, content_type):
        super().__init__(text.encode("utf-8"))
        self.headers = Message()
        self.headers["Content-Type"] = content_type


class AsciiTransport:
    """Model http.client's ASCII request-target requirement, without network."""
    def __init__(self, robots="User-agent: *\nAllow: /\n"):
        self.robots = robots
        self.requests = []

    def open(self, request, timeout):
        request.selector.encode("ascii")
        self.requests.append(request.full_url)
        if urlsplit(request.full_url).path == "/robots.txt":
            return Response(self.robots, "text/plain; charset=utf-8")
        return Response(HTML, "text/html; charset=utf-8")


class UnicodeArticleFetchTests(unittest.TestCase):
    def setUp(self):
        self.dns = patch.object(web_discovery, "check_public_dns")
        self.dns.start()
        self.addCleanup(self.dns.stop)
        self.cache = patch.dict(web_discovery._ROBOTS_CACHE, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_unicode_paths_encoded_once_for_http_transport(self):
        paths = (
            "/المكتبات-في-عهد-الملك-عبدالعزيز",
            "/ما-الفرق-بين-يوم-التأسيس-واليوم-الوطني؟",
            "/تاريخ%20السعودية",
            "/%D8%AA%D8%A7%D8%B1%D9%8A%D8%AE",
            "/history;year=1727/@article:one+two,three!$&'()*/",
        )
        for path in paths:
            with self.subTest(path=path):
                transport = AsciiTransport()
                url = "https://saudipedia.com" + path
                self.assertEqual(web_discovery.fetch_page(
                    url, load_config(), opener=transport), HTML)
                self.assertEqual(transport.requests[-1],
                    "https://saudipedia.com" +
                    quote(path, safe="/%:@!$&'()*+,;="))
                self.assertNotIn("%25", transport.requests[-1])
                self.assertEqual(web_discovery.canonical_url(url), url)

    def test_robots_disallow_still_blocks_unicode_and_encoded_paths(self):
        path = "/المكتبات-في-عهد-الملك-عبدالعزيز"
        for candidate in (path, quote(path)):
            with self.subTest(path=candidate):
                web_discovery._ROBOTS_CACHE.clear()
                transport = AsciiTransport(
                    "User-agent: *\nDisallow: " + quote(path) + "\n")
                with self.assertRaisesRegex(web_discovery.WebSourceError,
                                            "disallows automated access"):
                    web_discovery.fetch_page(
                        "https://saudipedia.com" + candidate,
                        load_config(), opener=transport)
                self.assertEqual(transport.requests,
                                 ["https://saudipedia.com/robots.txt"])

    def test_worker_reads_and_archives_unicode_article_with_real_fetcher(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            self.addCleanup(store.close)
            settings = load_config()
            settings["storage_root"] = root
            state = init_state(store, "saudi-history-ar")
            state["_queue"] = [{"url": ARTICLE, "depth": 0}]
            state["_seen"] = set()
            transport = AsciiTransport()
            with patch.object(web_discovery, "make_fetcher",
                              return_value=transport):
                result = _run_page(store, settings, state,
                                   get_profile("saudi-history-ar"))
            self.assertEqual(result["status"], "archived_research_only")
            self.assertEqual(state["pages_read"], 1)
            self.assertEqual(state["errors"], 0)
            self.assertGreater(state["html_bytes_read"], 0)
            saved = store.db.execute(
                "SELECT url,rights_status FROM web_corpus_sources"
            ).fetchone()
            self.assertEqual(saved["url"], ARTICLE)
            self.assertEqual(saved["rights_status"], "UNKNOWN")
            self.assertEqual(state["training_steps_this_run"], 0)


if __name__ == "__main__":
    unittest.main()
