"""v1.10: no wasted research retry attempts when budget or cooldown blocks work."""
from datetime import datetime, timezone, timedelta
import tempfile
import time
import unittest

from evolve.research import research_questions
from evolve.research_diagnostics import research_health
from evolve.settings import load_config
from evolve.store import Store

A = "https://alpha.example.org/river"
B = "https://independent.example.net/river"


def settings(root, daily=60, cooldown=3600):
    cfg = load_config()
    cfg["storage_root"] = root
    cfg["daily_request_budget"] = daily
    cfg["crawl_delay_seconds"] = 0
    cfg["research"] = {
        "questions_per_cycle": 5, "max_pages_per_question": 4,
        "retry_cooldown_seconds": cooldown,
    }
    return cfg


def add_question(store, concept="river", source=A):
    store.add_research_question(concept, f"What is {concept}?", source)


class ResearchQueueSafetyTests(unittest.TestCase):
    def test_quota_exhaustion_reports_no_attempts_and_preserves_retry_state(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = settings(root, daily=2)
            add_question(store)
            store.put("web_requests_" + time.strftime("%Y-%m-%d", time.gmtime()), 2)
            before = store.db.execute(
                "SELECT attempts, last_attempt_at FROM research_questions"
            ).fetchone()
            response = research_questions(
                store, cfg, limit=1,
                fetcher=lambda _: (_ for _ in ()).throw(
                    AssertionError("No requests should happen when budget is empty")))
            after = store.db.execute(
                "SELECT attempts, last_attempt_at FROM research_questions"
            ).fetchone()
            self.assertEqual(response["attempted_questions"], 0)
            self.assertEqual(response["source_requests"], 0)
            self.assertEqual(response["stop_reason"], "daily_budget_exhausted")
            self.assertEqual(response["deferred_due_to_budget"], 1)
            self.assertEqual(tuple(before), tuple(after))
            self.assertEqual(store.open_questions_count(), 1)
            store.close()

    def test_cooldown_defers_research_and_does_not_consume_requests(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = settings(root, daily=50, cooldown=3600)
            add_question(store)
            store.mark_question_attempt(1)
            before = store.db.execute(
                "SELECT attempts, last_attempt_at FROM research_questions WHERE id=1"
            ).fetchone()
            output = research_questions(
                store, cfg, limit=1,
                fetcher=lambda _: (_ for _ in ()).throw(
                    AssertionError("Cooldown should prevent fetch")))
            after = store.db.execute(
                "SELECT attempts, last_attempt_at FROM research_questions WHERE id=1"
            ).fetchone()
            self.assertEqual(output["attempted_questions"], 0)
            self.assertEqual(output["stop_reason"], "retry_cooldown_active")
            self.assertEqual(tuple(before), tuple(after))
            store.close()

    def test_mid_batch_budget_exhaustion_does_not_consume_other_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = settings(root, daily=1, cooldown=0)
            add_question(store, "river")
            add_question(store, "planet")
            store.save_observation(B, "Other source", "river crosscheck",
                                   ["river"], "research_only")
            hits = []
            def fetch(url):
                hits.append(url)
                return "<html><body>An river is a natural flowing stream of water moving through the land.</body></html>"
            result = research_questions(store, cfg, limit=2, fetcher=fetch)
            self.assertEqual(result["attempted_questions"], 1)
            self.assertEqual(result["source_requests"], 1)
            self.assertEqual(result["deferred_due_to_budget"], 1)
            self.assertEqual(result["stop_reason"], "daily_budget_exhausted")
            self.assertEqual(len(hits), 1)
            attempts = [x[0] for x in store.db.execute(
                "SELECT attempts FROM research_questions ORDER BY id").fetchall()]
            self.assertEqual(attempts, [1, 0])
            store.close()

    def test_snapshot_reads_quota_and_cooldown_without_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = settings(root, daily=60, cooldown=3600)
            add_question(store)
            store.add_research_question(
                "river", "What concepts relate to river?", A)
            store.mark_question_attempt(1)
            observed = datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)
            timestamp = observed.timestamp()
            store.db.execute(
                "UPDATE research_questions SET last_attempt_at=? WHERE id=1",
                (timestamp - 100,))
            store.db.commit()
            store.put("web_requests_2026-11-04", 60)
            before_attempts = [tuple(row) for row in store.db.execute(
                "SELECT id, attempts, last_attempt_at FROM research_questions ORDER BY id")]
            output = research_health(store, cfg, now=timestamp)
            self.assertEqual(output["status"], "blocked")
            self.assertEqual(output["reason"], "daily_budget_exhausted")
            self.assertEqual(output["request_budget"]["used"], 60)
            self.assertEqual(output["request_budget"]["remaining"], 0)
            self.assertEqual(
                output["request_budget"]["reset_at_utc"],
                "2026-11-05T00:00:00+00:00")
            self.assertEqual(output["question_queue"]["supported_definition_questions"], 1)
            self.assertEqual(output["question_queue"]["unsupported_open_questions"], 1)
            self.assertEqual(output["question_queue"]["in_cooldown"], 1)
            self.assertEqual(output["question_queue"]["ready_after_cooldown"], 0)
            self.assertEqual(output["question_queue"]["next_retry_at_utc"],
                             (observed + timedelta(seconds=3500)).isoformat())
            self.assertEqual(output["source_hosts_in_supported_queue"],
                             {"alpha.example.org": 1})
            self.assertEqual(output["training_admission"]["verified_answers"], 0)
            after_attempts = [tuple(row) for row in store.db.execute(
                "SELECT id, attempts, last_attempt_at FROM research_questions ORDER BY id")]
            self.assertEqual(before_attempts, after_attempts)
            store.close()

    def test_snapshot_when_budget_open_but_question_on_cooldown(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = settings(root, daily=60)
            add_question(store)
            store.mark_question_attempt(1)
            snapshot = research_health(store, cfg, now=time.time())
            self.assertEqual(snapshot["status"], "blocked")
            self.assertEqual(snapshot["reason"], "retry_cooldown_active")
            self.assertEqual(snapshot["question_queue"]["in_cooldown"], 1)
            self.assertEqual(snapshot["question_queue"]["ready_after_cooldown"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
