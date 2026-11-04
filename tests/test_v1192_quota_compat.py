"""Regressions: UI/backend protocol mismatch and daily-budget cache recovery.

Fixture pages are self-authored. No network and no model API used.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest

from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_studio import LearningController, serve
from evolve.v119_curiosity import budget_status, research, smart_lookup


class QuotaRetryTests(unittest.TestCase):
    def test_cached_budget_failure_retries_when_quota_is_available(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            settings["daily_request_budget"] = 1
            settings["crawl_delay_seconds"] = 0
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = "web_requests_" + today
            finding = "v119_finding:" + hashlib.sha256(b"moon").hexdigest()
            store.put(finding, {
                "status": "unresolved", "topic": "moon", "answer":
                    "Old stale quota failure", "sources": [],
                "reason": "daily_budget_exhausted",
                "retry_after_epoch": time.time() + 10000,
            })
            store.put(key, 1)
            calls = []
            def fake_fetch(url):
                calls.append(url)
                return (
                    "<html><title>Moon science</title><body><p>"
                    "The Moon is Earth's natural satellite and it orbits "
                    "the planet while scientists examine craters, rocks, "
                    "and changes in its surface over time."
                    "</p></body></html>"
                )
            try:
                blocked = smart_lookup(store, "moon", settings)
                self.assertEqual(blocked["status"], "research_budget_exhausted")
                self.assertIn("1/1", blocked["answer"])
                self.assertEqual(budget_status(store, settings)["remaining"], 0)
                still = research(store, settings, "moon", fetcher=fake_fetch)
                self.assertEqual(still["status"], "cached_unresolved")
                self.assertEqual(calls, [])

                # UTC day rollover opens a new storage counter. Simulating the
                # now-empty counter verifies that the cache no longer blocks
                # retries until its old generic cooldown has elapsed.
                store.put(key, 0)
                self.assertEqual(smart_lookup(store, "moon", settings)["status"],
                                 "needs_research")
                success = research(store, settings, "moon", fetcher=fake_fetch)
                self.assertEqual(success["status"], "researched", success)
                self.assertEqual(len(calls), 1)
                self.assertIn("Moon", success["answer"])
                self.assertEqual(store.get(key), 1)
                self.assertEqual(smart_lookup(store, "moon", settings)["status"],
                                 "answered")
                self.assertIsNone(store.get("v117_latest"))
            finally:
                store.close()

    def test_unseen_topic_reports_quota_without_scheduling_network(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            settings["daily_request_budget"] = 2
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            store.put("web_requests_" + today, 2)
            try:
                blocked = smart_lookup(store, "sun", settings)
                self.assertEqual(blocked["status"], "research_budget_exhausted")
                self.assertIn("2/2", blocked["answer"])
                self.assertIn("reset_at_utc", blocked["research_budget"])
            finally:
                store.close()

    def test_nonquota_unresolved_respects_cooldown(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            settings["daily_request_budget"] = 3
            finding = "v119_finding:" + hashlib.sha256(b"moon").hexdigest()
            store.put(finding, {
                "status": "unresolved", "topic": "moon", "answer": "No quote",
                "sources": [], "reason": "no_supported_passage",
                "retry_after_epoch": time.time() + 1000,
            })
            try:
                self.assertEqual(smart_lookup(store, "moon", settings)["status"],
                                 "unresolved")
            finally:
                store.close()


class StudioCompatibilityTests(unittest.TestCase):
    def test_backend_advertises_modes_and_rejects_bounded_quota_honestly(self):
        with tempfile.TemporaryDirectory() as root:
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            settings["daily_request_budget"] = 0
            controller = LearningController(settings, interval=60)
            try:
                status = controller.status()
                self.assertEqual(status["studio_api_version"], 2)
                self.assertEqual(status["studio_backend_version"], "1.19.2")
                self.assertIn("auto", status["supported_chat_modes"])
                self.assertEqual(status["research_budget"]["remaining"], 0)
                answer = controller.talk("sun", "auto")
                self.assertEqual(answer["status"], "research_budget_exhausted")
                self.assertNotIn("Unknown chat mode", answer["answer"])
            finally:
                controller.shutdown()

    def test_port_conflict_is_explained(self):
        with tempfile.TemporaryDirectory() as root:
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            try:
                port = listener.getsockname()[1]
                if port <= 1023:
                    self.skipTest("Cannot test reserved local port")
                with self.assertRaisesRegex(RuntimeError, "already used"):
                    serve(settings, port=port, interval=60)
            finally:
                listener.close()

    def test_page_explains_stale_old_backend(self):
        html = (Path(__file__).resolve().parents[1] / "evolve" /
                "studio.html").read_text("utf-8")
        self.assertIn('const STUDIO_API_VERSION=2;', html)
        self.assertIn("Old Studio backend detected", html)
        self.assertIn('id="research-budget"', html)


if __name__ == "__main__":
    unittest.main()
