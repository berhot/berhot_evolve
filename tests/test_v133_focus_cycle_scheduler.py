"""Open-ended focus scheduling: actual fetch slots, fair host pacing, stop safety."""
import copy
import tempfile
import unittest

from evolve.focus_data import get_profile
from evolve.focus_overnight import _run_page, cycle, init_state
from evolve.settings import load_config
from evolve.store import Store


PROFILE = "saudi-history-ar"


def _page():
    article = ("تاريخ المملكة العربية السعودية والدرعية "
               "الدولة السعودية الأولى والثانية والتوحيد. ") * 85
    return ("<html><head><title>تاريخ المملكة العربية السعودية</title>"
            "</head><body><article>" + article + "</article></body></html>")


class CycleSchedulingTests(unittest.TestCase):
    def _context(self, folder):
        store = Store(folder)
        settings = copy.deepcopy(load_config())
        settings["storage_root"] = folder
        settings["crawl_delay_seconds"] = 0
        state = init_state(store, PROFILE)
        state["until_stopped"] = True
        state["no_fixed_quotas"] = True
        state["minimum_host_gap_seconds"] = 20
        state["_seen"] = set()
        state["_host_last_at"] = {}
        return store, settings, state

    def test_paced_host_is_skipped_in_favor_of_other_queued_publisher(self):
        with tempfile.TemporaryDirectory() as folder:
            store, settings, state = self._context(folder)
            try:
                first = "https://saudipedia.com/first-state"
                second = "https://www.spa.gov.sa/second-state"
                state["_queue"] = [
                    {"url": first, "depth": 0},
                    {"url": second, "depth": 0},
                ]
                state["_host_last_at"]["saudipedia.com"] = 997.0
                result = _run_page(
                    store, settings, state, get_profile(PROFILE),
                    fetcher=lambda url: _page(),
                    time_fn=lambda: 1000.0)
                self.assertEqual(result["status"], "archived_research_only", result)
                self.assertEqual(result["url"], second)
                self.assertEqual(state["pages_attempted"], 1)
                self.assertEqual(state["_queue"][0]["url"], first)
                self.assertAlmostEqual(state["_host_last_at"]["www.spa.gov.sa"],
                                       1000.0)
            finally:
                store.close()

    def test_pacing_does_not_consume_page_slots_and_waits_to_fill_cycle(self):
        with tempfile.TemporaryDirectory() as folder:
            store, settings, state = self._context(folder)
            try:
                first = "https://saudipedia.com/first-state"
                second = "https://saudipedia.com/second-state"
                state["_queue"] = [
                    {"url": first, "depth": 0},
                    {"url": second, "depth": 0},
                ]
                simulated = [1000.0]
                state["_host_last_at"]["saudipedia.com"] = 996.0
                sleeps = []
                def advance(seconds):
                    sleeps.append(seconds)
                    simulated[0] += seconds
                seen = []
                def fetch(url):
                    seen.append(url)
                    return _page()
                result = cycle(
                    store, settings, state, pages=2, train_steps=1,
                    fetcher=fetch, sleep_fn=advance,
                    time_fn=lambda: simulated[0])
                self.assertEqual(result["status"], "cycle_completed")
                self.assertEqual(result["pages_attempted_this_cycle"], 2)
                self.assertEqual(result["pages_target_this_cycle"], 2)
                self.assertEqual(state["last_cycle_pages_attempted"], 2)
                self.assertEqual(state["last_cycle_page_target"], 2)
                self.assertGreaterEqual(state["last_cycle_pacing_waits"], 2)
                self.assertGreaterEqual(result["publisher_pacing_waits_this_cycle"], 2)
                self.assertEqual(state["pages_attempted"], 2)
                self.assertEqual(seen, [first, second])
                self.assertGreaterEqual(sum(sleeps), 35)
                self.assertEqual(state["training_steps_this_run"], 0)
                events = store.focus_activity(PROFILE)["events"]
                self.assertTrue(any(x["kind"] == "publisher_pacing_wait"
                                    for x in events))
                self.assertTrue(any(x["kind"] == "cycle_research_summary"
                                    and "2/2" in x["detail"] for x in events))
            finally:
                store.close()

    def test_noop_fake_sleep_does_not_busy_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            store, settings, state = self._context(folder)
            try:
                state["_queue"] = [{
                    "url": "https://saudipedia.com/first-state", "depth": 0,
                }]
                state["_host_last_at"]["saudipedia.com"] = 999.0
                result = cycle(
                    store, settings, state, pages=20, train_steps=1,
                    fetcher=lambda _url: self.fail("must not fetch paced host"),
                    sleep_fn=lambda _seconds: None,
                    time_fn=lambda: 1000.0)
                self.assertEqual(state["pages_attempted"], 0)
                self.assertEqual(result["pages_attempted_this_cycle"], 0)
                self.assertEqual(result["outcomes"][0]["status"],
                                 "publisher_pacing_wait")
            finally:
                store.close()

    def test_per_cycle_target_accepts_up_to_50_not_51(self):
        with tempfile.TemporaryDirectory() as folder:
            store, settings, state = self._context(folder)
            try:
                # Empty queue returns immediately; a high target is allowed.
                result = cycle(
                    store, settings, state, pages=50, train_steps=1,
                    fetcher=lambda _: self.fail("queue empty"))
                self.assertEqual(result["pages_target_this_cycle"], 50)
                with self.assertRaisesRegex(ValueError, "1..50"):
                    cycle(store, settings, state, pages=51, train_steps=1,
                          fetcher=lambda _: self.fail("unreachable"))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
