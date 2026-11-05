"""No-network re-discovery from immutable archived Saudi-history HTML."""
import copy
import tempfile
import unittest
from pathlib import Path

from evolve.focus_data import get_profile
from evolve.focus_overnight import (
    cycle, init_state, recover_cached_source_links,
)
from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store


PROFILE = "saudi-history-ar"


def archived_html(target_url):
    body = ("تاريخ المملكة العربية السعودية والدرعية والدولة السعودية "
            "الأولى والثانية ويوم التأسيس. ") * 65
    return (
        "<html><head><title>تاريخ المملكة العربية السعودية</title>"
        "</head><body><article>" + body + "</article>"
        '<a href="' + target_url + '">الدولة السعودية تاريخ المملكة</a>'
        "</body></html>"
    )


class LocalLinkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = Store(self.folder.name)
        self.settings = copy.deepcopy(load_config())
        self.settings["storage_root"] = self.folder.name
        self.settings["crawl_delay_seconds"] = 0
        self.state = init_state(self.store, PROFILE)
        self.state["until_stopped"] = True
        self.state["no_fixed_quotas"] = True
        self.state["minimum_host_gap_seconds"] = 0
        self.state["_queue"] = []
        self.state["_seen"] = set()

    def tearDown(self):
        self.store.close()
        self.folder.cleanup()

    def archive(self, url, target):
        html = archived_html(target)
        outcome = archive_page(
            self.store, url, html,
            ("تاريخ المملكة العربية السعودية والدرعية. " * 80),
            title="تاريخ المملكة العربية السعودية")
        self.assertEqual(outcome["status"], "archived")
        self.store.put("focus_observations_" + PROFILE, [{
            "url": url, "host": "saudipedia.com",
            "title": "تاريخ الدولة السعودية",
            "rights": "UNKNOWN", "curiosity": [],
            "source_sha256": outcome["clean_sha256"],
        }])

    def test_cached_link_recovery_queues_candidate_without_network(self):
        source = "https://saudipedia.com/saudi-history-archive"
        candidate = "https://history.example.org/saudi-history-article"
        self.archive(source, candidate)
        output = recover_cached_source_links(
            self.store, self.state, PROFILE)
        self.assertEqual(output["status"], "cached_sources_scanned")
        self.assertEqual(output["source_snapshots_checked"], 1)
        self.assertEqual(output["new_candidate_urls"], 1)
        self.assertEqual(output["cross_publisher_candidates"], 1)
        self.assertEqual(output["network_requests"], 0)
        self.assertEqual(self.state["_queue"], [
            {"url": candidate, "depth": 1}])
        again = recover_cached_source_links(self.store, self.state, PROFILE)
        self.assertEqual(again["source_snapshots_checked"], 0)
        self.assertEqual(again["new_candidate_urls"], 0)

    def test_empty_queue_cycle_recovers_and_reads_candidate(self):
        source = "https://saudipedia.com/saudi-history-archive"
        candidate = "https://history.example.org/saudi-history-article"
        self.archive(source, candidate)
        calls = []
        def fetch(url):
            calls.append(url)
            return archived_html("https://example.com/founding-history")
        run = cycle(
            self.store, self.settings, self.state,
            pages=1, train_steps=1, fetcher=fetch, sleep_fn=lambda _: None)
        self.assertEqual(calls, [candidate])
        self.assertEqual(run["pages_attempted_this_cycle"], 1)
        self.assertEqual(run["archive_link_recovery"]["new_candidate_urls"], 1)
        self.assertEqual(self.state["pages_archived"], 1)
        self.assertEqual(self.state["training_steps_this_run"], 0)
        events = self.store.focus_activity(PROFILE)["events"]
        self.assertTrue(any(e["kind"] == "archive_link_recovery"
                            for e in events))

    def test_exhaustion_reported_honestly_and_never_cycles_perpetually(self):
        source = "https://saudipedia.com/saudi-history-archive"
        self.archive(source, "https://history.example.org/saudi-history-article")
        self.state["_seen"].add("https://history.example.org/saudi-history-article")
        first = cycle(
            self.store, self.settings, self.state,
            pages=30, train_steps=1,
            fetcher=lambda _: self.fail("No eligible URL"),
            sleep_fn=lambda _: None)
        self.assertEqual(first["pages_attempted_this_cycle"], 0)
        self.assertEqual(first["idle_reason"],
                         "no_pending_urls_and_search_produced_no_candidates")
        self.assertEqual(first["archive_link_recovery"]["new_candidate_urls"], 0)
        second = cycle(
            self.store, self.settings, self.state,
            pages=30, train_steps=1,
            fetcher=lambda _: self.fail("No eligible URL"),
            sleep_fn=lambda _: None)
        self.assertEqual(second["archive_link_recovery"]["status"],
                         "cached_sources_exhausted")
        self.assertEqual(second["pages_attempted_this_cycle"], 0)
        events = self.store.focus_activity(PROFILE)["events"]
        self.assertTrue(any(e["kind"] == "research_idle_reason"
                            for e in events))

    def test_cannot_read_untrusted_archived_raw_path(self):
        source = "https://saudipedia.com/saudi-history-archive"
        self.archive(source, "https://history.example.org/saudi-history-article")
        outside = Path(self.folder.name) / "private.html.gz"
        outside.write_bytes(b"not allowed")
        self.store.db.execute(
            "UPDATE web_corpus_sources SET raw_path=? WHERE url=?",
            (str(outside), source))
        self.store.db.commit()
        scanned = recover_cached_source_links(self.store, self.state, PROFILE)
        self.assertEqual(scanned["new_candidate_urls"], 0)
        self.assertEqual(self.state["_queue"], [])


if __name__ == "__main__":
    unittest.main()
