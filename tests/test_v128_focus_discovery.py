"""v1.28: autonomous topic-based multi-publisher research, safely bounded.

Offline fake web search and self-authored articles; no license assertion for
the research. NO real HTTP credentials, rate bypass or automatic truth labels.
"""
import copy
import tempfile
import unittest
from unittest.mock import patch

from evolve.focus_data import get_profile
from evolve.focus_discovery import (
    relevant_links, run_one_search, _candidate_link,
)
from evolve.focus_overnight import init_state, cycle, report
from evolve.focus_control import (
    acquire_lock, release_lock, request_stop, worker_running,
    model_size_snapshot,
)
from evolve.settings import load_config
from evolve.store import Store

SAUDI = (
    "تأسست الدولة السعودية الأولى في مدينة الدرعية عام 1727م. "
    "يعرض تاريخ المملكة العربية السعودية التأسيس والدولة "
    "السعودية الأولى ومسيرة الملك عبدالعزيز في الدرعية. "
) * 20

def html_page(text, links=""):
    return (
        "<html><head><title>تاريخ المملكة العربية السعودية</title></head>"
        "<body><article>" + text + "</article>" + links + "</body></html>"
    )


class FocusDiscoveryTests(unittest.TestCase):
    def test_cross_publisher_links_are_not_restricted_to_seed_domains(self):
        profile = get_profile("saudi-history-ar")
        page = html_page(
            SAUDI,
            '<a href="https://history-new.example.org/saudi-history">'
            'تاريخ المملكة العربية السعودية والدرعية</a>'
            '<a href="https://www.spa.gov.sa/article">'
            'الدرعية وتاريخ الدولة السعودية الأولى</a>'
            '<a href="https://instagram.com/bad">'
            'تاريخ الدرعية</a>'
            '<a href="http://127.0.0.1/private">'
            'تاريخ الدرعية</a>'
            '<a href="https://weather.example.org/today">'
            'Weather forecast</a>'
        )
        links = relevant_links(
            page, "https://saudipedia.com/start", profile)
        self.assertIn(
            "https://history-new.example.org/saudi-history", links)
        self.assertIn("https://www.spa.gov.sa/article", links)
        self.assertFalse(any("instagram.com" in x for x in links))
        self.assertFalse(any("127.0.0.1" in x for x in links))
        self.assertFalse(any("weather.example.org" in x for x in links))
        self.assertIsNone(_candidate_link("http://localhost/secret"))
        self.assertIsNone(_candidate_link("https://127.0.0.1/secret"))

    def test_search_requires_key_or_explicit_injected_provider_and_daily_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = tmp
            settings["daily_request_budget"] = 8
            profile = get_profile("saudi-history-ar")
            fake_calls = []
            def search(q):
                fake_calls.append(q)
                return [
                    {"url": "https://independent-history.example.org/saudi",
                     "title": "تاريخ المملكة العربية السعودية",
                     "snippet": "الدرعية وتاريخ الدولة السعودية الأولى"},
                    {"url": "https://weather.example.org/weather",
                     "title": "Weather forecast",
                     "snippet": "Sunny weather."},
                ]
            try:
                with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}):
                    absent = run_one_search(store, settings, profile, [])
                self.assertEqual(absent["queries_attempted"], 0)
                self.assertEqual(absent["url_candidates"], [])
                for _ in range(4):
                    r = run_one_search(store, settings, profile, [],
                                       search_fn=search)
                    self.assertEqual(r["status"], "search_completed")
                    self.assertEqual(r["queries_attempted"], 1)
                    self.assertIn(
                        "https://independent-history.example.org/saudi",
                        r["url_candidates"])
                    self.assertNotIn(
                        "https://weather.example.org/weather",
                        r["url_candidates"])
                exhausted = run_one_search(
                    store, settings, profile, [], search_fn=search)
                self.assertEqual(exhausted["status"],
                                 "no_new_search_questions_today")
                self.assertEqual(len(fake_calls), 4)
                self.assertEqual(len(set(fake_calls)), 4)
                # No duplicate queries on the same UTC day, even if
                # unused API budget remains. New source-derived questions
                # become eligible in subsequent cycles.
            finally:
                store.close()

    def test_observed_questions_expand_search_and_resume_discovered_frontier(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = folder
            cfg["daily_request_budget"] = 12
            cfg["crawl_delay_seconds"] = 0
            try:
                state = init_state(store, "saudi-history-ar", max_pages=3)
                seen = []
                linked = (
                    '<a href="https://new-publisher.example.org/saudi-history">'
                    'تاريخ الدولة السعودية الأولى في الدرعية</a>'
                )
                def fetch(url):
                    seen.append(url)
                    return html_page(SAUDI, linked)
                def search(_question):
                    return [
                        {"url": "https://another-new.example.org/saudi-history",
                         "title": "تاريخ الدولة السعودية الأولى",
                         "snippet": "الدرعية وتاريخ السعودية"}
                    ]
                outcome = cycle(
                    store, cfg, state, pages=1, train_steps=1,
                    fetcher=fetch, search_fn=search,
                    sleep_fn=lambda _: None)
                self.assertEqual(outcome["status"], "cycle_completed")
                self.assertEqual(state["pages_attempted"], 1)
                self.assertGreaterEqual(state["links_queued_this_run"], 1)
                self.assertGreaterEqual(
                    state["cross_host_links_queued_this_run"], 1)
                self.assertEqual(state["search_queries_this_run"], 1)
                self.assertGreaterEqual(
                    state["search_results_queued_this_run"], 1)
                self.assertEqual(state["training_steps_this_run"], 0)
                self.assertIn("https://new-publisher.example.org/saudi-history",
                              [x["url"] for x in state["_queue"]])
                self.assertEqual(
                    report(store)["discovery"]["no_fixed_website_allowlist"],
                    True)
                new_state = init_state(store, "saudi-history-ar", max_pages=3)
                self.assertIn("https://new-publisher.example.org/saudi-history",
                              [x["url"] for x in new_state["_queue"]])
                self.assertIn(seen[0], new_state["_seen"])
                self.assertEqual(
                    store.db.execute(
                        "SELECT COUNT(*) FROM web_corpus_sources WHERE "
                        "rights_status='UNKNOWN'").fetchone()[0], 1)
            finally:
                store.close()


class FocusControlTests(unittest.TestCase):
    def test_single_worker_lock_and_local_stop_control(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                self.assertFalse(worker_running(store))
                lock = acquire_lock(store)
                try:
                    self.assertTrue(worker_running(store))
                    with self.assertRaises(RuntimeError):
                        acquire_lock(store)
                    store.put("focus_run_saudi-history-ar", {
                        "run_id": "run-abc", "status": "running"})
                    r = request_stop(store)
                    self.assertEqual(r["status"], "stop_requested")
                    self.assertEqual(r["run_id"], "run-abc")
                    self.assertEqual(
                        store.get("focus_stop_request_saudi-history-ar")[
                            "run_id"], "run-abc")
                finally:
                    release_lock(lock)
                self.assertFalse(worker_running(store))
                self.assertEqual(request_stop(store)["status"], "not_running")
            finally:
                store.close()

    def test_model_size_architectures_not_claimed_as_trained(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                sizes = model_size_snapshot(store)
                self.assertEqual(sizes["ranker"][
                    "architecture_parameters"], 109)
                self.assertEqual(sizes["saudi_history_arabic"][
                    "architecture_parameters"], 79168)
                self.assertFalse(
                    sizes["saudi_history_arabic"][
                        "accepted_checkpoint_recorded"])
                self.assertFalse(
                    sizes["saudi_history_arabic"][
                        "parameters_grew_from_more_articles"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
