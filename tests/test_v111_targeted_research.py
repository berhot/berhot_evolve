"""v1.11: exact-concept research targeting and offline source planning.

All sources are simulated locally. No live requests, rights assumptions, or
changes to the existing verified-source admission policy.
"""
import tempfile
import time
import unittest

from evolve.research import research_questions
from evolve.research_diagnostics import research_plan
from evolve.settings import load_config
from evolve.store import Store

A = "https://alpha.example.org/energy"
B = "https://beta.other.edu/reference/energy"
C = "https://alpha.example.org/population"
CC0 = '<link rel="license" href="https://creativecommons.org/publicdomain/zero/1.0/"/>'
CLAUSE = "a measurable capacity for doing work in physical systems"


def cfg(root, daily=60, cooldown=3600):
    settings = load_config()
    settings["storage_root"] = root
    settings["daily_request_budget"] = daily
    settings["crawl_delay_seconds"] = 0
    settings["research"] = {
        "questions_per_cycle": 5,
        "max_pages_per_question": 4,
        "retry_cooldown_seconds": cooldown,
    }
    return settings


def seed(store):
    store.save_observation(A, "Energy source", "Energy concept",
                           ["energy"], "research_only")
    store.save_observation(B, "Independent source", "Energy concept",
                           ["energy"], "research_only")
    store.save_observation(C, "Population source", "Population concept",
                           ["population"], "research_only")
    store.add_research_question("population", "What is population?", C)
    store.add_research_question("energy", "What is energy?", A)


class TargetedResearchTests(unittest.TestCase):
    def test_exact_concept_uses_only_matching_queue_and_true_source_rules(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            seed(store)
            seen = []
            pages = {
                A: f"<html><head>{CC0}</head><body>An energy is {CLAUSE}.</body></html>",
                B: f"<html><body>An energy is {CLAUSE}.</body></html>",
            }
            def fetch(url):
                seen.append(url)
                if url not in pages:
                    raise AssertionError("Unselected concept's source was fetched")
                return pages[url]
            report = research_questions(
                store, cfg(root), limit=1, concept="energy", fetcher=fetch)
            self.assertEqual(report["targeted_concept"], "energy")
            self.assertEqual(report["attempted_questions"], 1)
            self.assertEqual(report["verified_answers"], 1)
            self.assertEqual(report["training_examples_created"], 1)
            self.assertEqual(report["source_requests"], 2)
            self.assertEqual(set(seen), {A, B})
            answers = store.list_verified_answers(10)
            self.assertEqual(len(answers), 1)
            self.assertTrue(answers[0]["training_eligible"])
            self.assertEqual(answers[0]["concept"], "energy")
            self.assertEqual(store.db.execute(
                "SELECT attempts FROM research_questions "
                "WHERE concept='population'").fetchone()[0], 0)
            store.close()

    def test_no_matching_concept_does_not_touch_any_question(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            seed(store)
            report = research_questions(
                store, cfg(root), limit=3, concept="volcano",
                fetcher=lambda _: (_ for _ in ()).throw(AssertionError("No fetch")))
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["attempted_questions"], 0)
            self.assertEqual(report["stop_reason"],
                             "no_matching_supported_open_question")
            self.assertEqual(store.db.execute(
                "SELECT SUM(attempts) FROM research_questions").fetchone()[0], 0)
            self.assertEqual(research_plan(store, cfg(root), "volcano")["status"],
                             "not_found")
            store.close()

    def test_cooldown_and_quota_are_enforced_for_targeted_runs(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            seed(store)
            store.mark_question_attempt(2)
            report = research_questions(
                store, cfg(root), limit=1, concept="energy",
                fetcher=lambda _: (_ for _ in ()).throw(AssertionError("No fetch")))
            self.assertEqual(report["stop_reason"], "retry_cooldown_active")
            self.assertEqual(report["attempted_questions"], 0)
            store.put("web_requests_" + time.strftime("%Y-%m-%d", time.gmtime()), 60)
            report = research_questions(
                store, cfg(root), limit=1, concept="population",
                fetcher=lambda _: (_ for _ in ()).throw(AssertionError("No fetch")))
            self.assertEqual(report["stop_reason"], "daily_budget_exhausted")
            self.assertEqual(report["attempted_questions"], 0)
            self.assertEqual(store.db.execute(
                "SELECT attempts FROM research_questions "
                "WHERE concept='population'").fetchone()[0], 0)
            store.close()

    def test_plan_is_readonly_and_does_not_claim_validated_rights(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            seed(store)
            before = store.db.total_changes
            report = research_plan(store, cfg(root), "energy")
            after = store.db.total_changes
            self.assertEqual(before, after)
            self.assertEqual(report["status"], "planned")
            self.assertEqual(report["concept"], "energy")
            self.assertEqual(report["candidate_urls_total"], 2)
            self.assertEqual(report["candidate_groups_in_fetch_window"], 2)
            self.assertTrue(report["has_independent_candidate_publishers"])
            self.assertEqual({item["url"] for item in report["candidate_sources"]},
                             {A, B})
            self.assertTrue(all(
                item["training_rights"] == "unknown_unless_page_checked"
                for item in report["candidate_sources"]))
            self.assertIn("--concept energy", report["next_action"])
            self.assertEqual(len(store.list_verified_answers()), 0)
            store.close()

    def test_invalid_concept_is_rejected_before_any_network_activity(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            seed(store)
            for invalid in ("", "x", "energy; DROP TABLE", "outside/energy"):
                with self.assertRaises(ValueError):
                    research_plan(store, cfg(root), invalid)
                with self.assertRaises(ValueError):
                    research_questions(store, cfg(root), concept=invalid,
                        fetcher=lambda _: (_ for _ in ()).throw(
                            AssertionError("No fetch")))
            self.assertEqual(len(store.list_research_questions()), 2)
            store.close()


if __name__ == "__main__":
    unittest.main()
