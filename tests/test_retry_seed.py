"""Manual requeue for transient TLS errors without bypassing robots policy."""
import tempfile
import unittest

from evolve.store import Store
from evolve.web_discovery import canonical_url

URL = "https://lessons.example.org/alphabet"


class RetrySeedTests(unittest.TestCase):
    def test_retry_ssl_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.add_web_url(URL)
            store.mark_web_url(URL, "error", "Cannot verify robots.txt: SSL: CERTIFICATE_VERIFY_FAILED")
            result = store.retry_failed_web_url(URL)
            self.assertEqual(result["status"], "requeued")
            self.assertEqual(store.next_web_url()["url"], URL)
            self.assertEqual(store.db.execute("SELECT attempts FROM web_frontier WHERE url=?",
                                              (URL,)).fetchone()[0], 1)
            store.close()
            reopened = Store(tmp)
            self.assertEqual(reopened.next_web_url()["url"], URL)
            reopened.close()

    def test_never_retry_explicit_robots_disallow(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.add_web_url(URL)
            store.mark_web_url(URL, "error", "Website disallows automated access to this URL")
            result = store.retry_failed_web_url(URL)
            self.assertEqual(result["status"], "blocked_by_robots")
            self.assertIsNone(store.next_web_url())
            store.close()

    def test_nonfailed_and_unknown_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            self.assertEqual(store.retry_failed_web_url(URL)["status"], "not_found")
            store.add_web_url(URL)
            self.assertEqual(store.retry_failed_web_url(URL)["status"], "not_failed")
            store.close()

    def test_requires_safe_canonical_https_url(self):
        with self.assertRaises(ValueError):
            canonical_url("http://localhost/lesson")


if __name__ == "__main__":
    unittest.main()
