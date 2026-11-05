"""Persistent focus-worker activity journal and local live-monitor API tests."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from evolve.focus_overnight import init_state, _publish
from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_studio import LearningController, ThreadingHTTPServer, _handler


PROFILE = "saudi-history-ar"


class FocusActivityHistoryTests(unittest.TestCase):
    def test_unbounded_history_and_bidirectional_cursor_pages(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                for i in range(510):
                    ok = store.append_focus_activity(
                        PROFILE, "run-one", "2026-11-05T04:00:00Z",
                        "source_event", phase="reading_article",
                        detail=f"source-{i}", url=f"https://example.org/{i}")
                    self.assertTrue(ok)
                self.assertFalse(store.append_focus_activity(
                    PROFILE, "run-one", "2026-11-05T04:00:00Z",
                    "source_event", phase="reading_article",
                    detail="source-0", url="https://example.org/0"))
                latest = store.focus_activity(PROFILE, limit=150)
                self.assertEqual(latest["total_events"], 510)
                self.assertIsNone(latest["history_retention_limit"])
                self.assertEqual(len(latest["events"]), 150)
                self.assertTrue(latest["has_more"])
                self.assertEqual(latest["events"][-1]["detail"], "source-509")

                all_events = list(latest["events"])
                before = latest["events"][0]["id"]
                while True:
                    older = store.focus_activity(PROFILE, before=before, limit=150)
                    self.assertTrue(all(e["id"] < before for e in older["events"]))
                    all_events = older["events"] + all_events
                    if not older["has_more"]:
                        break
                    before = older["events"][0]["id"]
                self.assertEqual(len(all_events), 510)
                self.assertEqual(len({x["id"] for x in all_events}), 510)
                self.assertEqual(all_events[0]["detail"], "source-0")
                after = store.focus_activity(PROFILE, after=509, limit=150)
                self.assertEqual([e["detail"] for e in after["events"]],
                                 ["source-509"])
                self.assertFalse(after["has_more"])
            finally:
                store.close()

    def test_legacy_snapshot_backfill_is_idempotent_and_incremental(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                state = {
                    "run_id": "legacy-worker",
                    "recent_activity": [{
                        "at_utc": "2026-11-05T04:00:00Z",
                        "kind": "worker_started",
                        "detail": "Started",
                        "url": None,
                    }],
                }
                store.put("focus_run_" + PROFILE, state)
                store.backfill_focus_activity(PROFILE)
                store.backfill_focus_activity(PROFILE)
                self.assertEqual(
                    store.focus_activity(PROFILE)["total_events"], 1)
                state["recent_activity"].append({
                    "at_utc": "2026-11-05T04:00:01Z",
                    "kind": "article_read",
                    "detail": "new article",
                    "url": "https://example.org/one",
                })
                store.put("focus_run_" + PROFILE, state)
                store.backfill_focus_activity(PROFILE)
                self.assertEqual(
                    store.focus_activity(PROFILE)["total_events"], 2)
                store.backfill_focus_activity(PROFILE)
                self.assertEqual(
                    store.focus_activity(PROFILE)["total_events"], 2)
            finally:
                store.close()

    def test_actual_worker_publish_records_durable_event(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                state = init_state(store, PROFILE)
                _publish(store, state, "reading_article",
                         url="https://example.org/saudi",
                         event="article_read", detail="real bytes measured")
                journal = store.focus_activity(PROFILE)
                self.assertEqual(journal["total_events"], 1)
                self.assertEqual(journal["events"][0]["phase"],
                                 "reading_article")
                self.assertEqual(journal["events"][0]["detail"],
                                 "real bytes measured")
                self.assertEqual(journal["events"][0]["run_id"],
                                 state["run_id"])
            finally:
                store.close()

    def test_invalid_cursor_does_not_read_data(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                for query in (
                    {"after": -1}, {"before": 0}, {"after": 1, "before": 2},
                    {"limit": 251}, {"limit": 0},
                ):
                    with self.assertRaises(ValueError):
                        store.focus_activity(PROFILE, **query)
            finally:
                store.close()


class FocusActivityStudioTests(unittest.TestCase):
    def test_http_activity_feed_live_history_and_bad_cursors(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            db = Store(root)
            db.append_focus_activity(
                PROFILE, "run-test", "2026-11-05T04:00:00Z",
                "worker_started", detail="real source")
            db.close()
            controller = LearningController(cfg, interval=30)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), _handler(
                    controller, "secret", "http://127.0.0.1:0"))
            server.RequestHandlerClass = _handler(
                controller, "secret",
                f"http://127.0.0.1:{server.server_port}")
            thread = threading.Thread(
                target=server.serve_forever, daemon=True)
            thread.start()
            try:
                def read(path):
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", server.server_port, timeout=5)
                    conn.request("GET", path)
                    response = conn.getresponse()
                    status, payload = response.status, json.loads(response.read())
                    conn.close()
                    return status, payload

                status, data = read("/api/focus/activity?limit=20")
                self.assertEqual(status, 200)
                self.assertEqual(data["total_events"], 1)
                self.assertEqual(data["events"][0]["detail"], "real source")
                status, data = read("/api/focus/activity?after=1&limit=20")
                self.assertEqual(status, 200)
                self.assertEqual(data["events"], [])
                for path in (
                    "/api/focus/activity?after=-2",
                    "/api/focus/activity?before=0",
                    "/api/focus/activity?limit=99999",
                    "/api/focus/activity?after=1&before=2",
                ):
                    status, payload = read(path)
                    self.assertEqual(status, 400, path)
                    self.assertEqual(payload["error"],
                                     "invalid_activity_cursor")
            finally:
                server.shutdown()
                server.server_close()
                controller.shutdown()
                thread.join(timeout=3)

    def test_ui_auto_polls_and_never_slices_event_history(self):
        html = (Path(__file__).resolve().parents[1] /
                "evolve" / "studio.html").read_text("utf-8")
        for item in (
            'id="focus-recent-activity"', 'id="focus-load-earlier"',
            'id="focus-activity-connection"',
            'setInterval(pollFocusActivity,1000)',
            '/api/focus/activity?after=', '/api/focus/activity?before=',
            'loadOlderFocusActivity', 'focus-activity-count',
        ):
            self.assertIn(item, html)
        self.assertNotIn('renderFocusActivity(f.recent_activity)', html)
        self.assertNotIn('[...events].reverse().slice(0,10)', html)


if __name__ == "__main__":
    unittest.main()
