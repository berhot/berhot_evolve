"""Rights reviews never silently grant permission or overwrite old weights."""
import tempfile
import unittest
from pathlib import Path
from hashlib import sha256

from evolve.store import Store
from evolve.open_corpus import archive_page
from evolve.v117_rights import sources, review_rights
from evolve.v117 import plan, prepare


def article(store, host, n, noai=False):
    title = f"Scientific methods {n}"
    text = " ".join(
        f"Scientific observation {i} tests instrument {n+i} and compares "
        f"energy distribution, atmospheric motion, plant growth and "
        f"geological processes. Repeated experiments improve estimates."
        for i in range(180))
    url = f"https://{host}/research/science-{n}"
    archived = archive_page(
        store, url, "<html><title>" + title + "</title><p>" + text +
        "</p></html>", text, title=title, noai=noai)
    assert archived["status"] == "archived"
    return url


class RightsReviewTests(unittest.TestCase):
    def test_explicit_attestation_and_distinct_hosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                urls = [article(store, f"host-{i}.example.org", i)
                        for i in range(3)]
                old = sources(store)["sources"]
                self.assertEqual([r["rights_status"] for r in old],
                                 ["UNKNOWN"] * 3)
                self.assertEqual(plan(store)["eligible_article_count"], 0)
                for i, url in enumerate(urls):
                    with self.assertRaisesRegex(ValueError, "confirm"):
                        review_rights(store, url, "PUBLIC-DOMAIN",
                                      f"https://host-{i}.example.org/policy")
                    record = review_rights(
                        store, url, "PUBLIC-DOMAIN",
                        f"https://host-{i}.example.org/policy",
                        confirmed=True)
                    self.assertEqual(record["status"], "review_recorded")
                self.assertEqual(len(store.get("v117_rights_review_history")), 3)
                self.assertEqual(plan(store)["status"], "ready_for_strict_freeze")
                result = prepare(store)
                self.assertEqual(result["status"], "prepared", result)
                self.assertFalse(result["diagnostic_source_leakage"])
            finally:
                store.close()

    def test_noai_and_unknown_article_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                url = article(store, "blocked.example.org", 0, noai=True)
                blocked = review_rights(
                    store, url, "PUBLIC-DOMAIN",
                    "https://blocked.example.org/policy", confirmed=True)
                self.assertEqual(blocked["reason"], "publisher_noai_signal")
                self.assertEqual(sources(store)["sources"][0]["rights_status"],
                                 "UNKNOWN")
                missing = review_rights(
                    store, "https://missing.example.org/article",
                    "PUBLIC-DOMAIN", "https://missing.example.org/policy",
                    confirmed=True)
                self.assertEqual(missing["reason"], "article_not_archived")
                self.assertEqual(store.get("v117_rights_review_history", []), [])
            finally:
                store.close()

    def test_invalid_evidence_and_tampered_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                url = article(store, "alpha.example.org", 1)
                with self.assertRaisesRegex(ValueError, "normal HTTPS"):
                    review_rights(store, url, "PUBLIC-DOMAIN",
                                  "http://alpha.example.org/policy",
                                  confirmed=True)
                with self.assertRaisesRegex(ValueError, "Separate"):
                    review_rights(store, url, "PUBLIC-DOMAIN", url,
                                  confirmed=True)
                row = store.db.execute(
                    "SELECT t.clean_path FROM web_corpus_sources s "
                    "JOIN web_corpus_texts t ON t.sha256=s.sha256 "
                    "WHERE s.url=?", (url,)).fetchone()
                Path(row[0]).write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    review_rights(
                        store, url, "PUBLIC-DOMAIN",
                        "https://alpha.example.org/policy", confirmed=True)
                self.assertEqual(
                    sources(store)["sources"][0]["rights_status"], "UNKNOWN")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
