"""v1.22.3: explicitly targeted crawl without queue hopping or robots bypass."""
import tempfile
import unittest

from evolve.open_corpus import corpus_seed
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import discover_web, WebSourceError

HOST = "oceanservice.noaa.gov"
A = "https://" + HOST + "/facts/current.html"
B = "https://" + HOST + "/facts/estuary.html"
OLDER = "https://www.example.org/science"


def config(path):
    settings = load_config()
    settings["storage_root"] = path
    settings["daily_request_budget"] = 20
    settings["crawl_delay_seconds"] = 0
    settings["web"]["max_depth"] = 0
    return settings


def science_page(url):
    topic = "current" if "current" in url else "estuary"
    return (
        "<html><head><title>What is an " + topic + "?</title></head>"
        "<body><p>A " + topic + " is a physical ocean environment "
        "described through measurements and observations of water "
        "and moving substances in documented coastal science.</p></body></html>"
    )


class TargetedCrawlTests(unittest.TestCase):
    def test_only_pending_matching_host_attempted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                corpus_seed(store, [OLDER], cross_host=False)
                corpus_seed(store, [A, B], cross_host=False)
                called = []
                def fetcher(url):
                    called.append(url)
                    return science_page(url)
                result = discover_web(
                    store, config(tmp), limit=2, host=HOST,
                    fetcher=fetcher, sleep_fn=lambda _: None,
                    review_questions=False)
                self.assertEqual(called, [A, B])
                self.assertEqual(result["host_filter"], HOST)
                self.assertEqual(result["attempted_urls"], [A, B])
                self.assertEqual(result["archived_urls"], [A, B])
                self.assertEqual(result["corpus_archived"], 2)
                self.assertEqual(result["errors"], 0)
                self.assertIsNone(store.next_web_url(host=HOST))
                self.assertEqual(store.next_web_url()["url"], OLDER)
                self.assertEqual(
                    store.db.execute(
                        "SELECT COUNT(*) FROM web_corpus_sources "
                        "WHERE host=?", (HOST,)
                    ).fetchone()[0], 2)
                self.assertFalse(
                    store.db.execute(
                        "SELECT 1 FROM web_corpus_sources "
                        "WHERE url=?", (OLDER,)
                    ).fetchone())
                self.assertEqual(
                    [x[0] for x in store.db.execute(
                        "SELECT rights_status FROM web_corpus_sources "
                        "WHERE host=?", (HOST,)
                    )], ["UNKNOWN", "UNKNOWN"])
            finally:
                store.close()

    def test_host_filter_never_bypasses_denied_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                corpus_seed(store, [OLDER], cross_host=False)
                corpus_seed(store, [A], cross_host=False)
                def denied(_url):
                    raise WebSourceError("Robots availability uncertain (HTTP 403)")
                result = discover_web(
                    store, config(tmp), limit=2, host=HOST,
                    fetcher=denied, sleep_fn=lambda _: None,
                    review_questions=False)
                self.assertEqual(result["attempted_urls"], [A])
                self.assertEqual(result["archived_urls"], [])
                self.assertEqual(result["errors"], 1)
                self.assertEqual(result["error_details"][0]["url"], A)
                self.assertEqual(store.next_web_url()["url"], OLDER)
                self.assertIsNone(store.next_web_url(host=HOST))
                self.assertEqual(
                    store.db.execute("SELECT COUNT(*) FROM web_corpus_sources"
                                    ).fetchone()[0], 0)
            finally:
                store.close()

    def test_invalid_host_is_rejected_before_queue_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                corpus_seed(store, [A], cross_host=False)
                for host in ("https://oceanservice.noaa.gov",
                             "oceanservice.noaa.gov/anything",
                             "other.example.org@oceanservice.noaa.gov",
                             "NOAA.GOV", "a_b.example.org"):
                    with self.subTest(host=host), self.assertRaises(ValueError):
                        discover_web(store, config(tmp), host=host)
                self.assertIsNotNone(store.next_web_url(host=HOST))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
