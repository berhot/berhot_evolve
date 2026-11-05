"""Evidence-only learning inspector: no invented learning, claims or network."""
import copy
import hashlib
import http.client
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
import threading
import unittest

from evolve.focus_control import acquire_lock, release_lock, worker_running
from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_studio import LearningController, ThreadingHTTPServer, _handler


PROFILE = "saudi-history-ar"


class InspectorTests(unittest.TestCase):
    def _settings(self, root):
        settings = copy.deepcopy(load_config())
        settings["storage_root"] = root
        return settings

    def test_research_questions_dont_count_as_answers_or_training(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self._settings(root)
            store = Store(root)
            controller = None
            lock = None
            try:
                url = "https://example.org/saudi-history"
                store.put("focus_observations_" + PROFILE, [{
                    "title": "تاريخ الدولة السعودية",
                    "url": url, "host": "example.org",
                    "rights": "UNKNOWN", "noai": False,
                    "source_sha256": "a" * 64,
                    "curiosity": [{
                        "question": "متى تأسست الدولة السعودية؟",
                        "subject": "تأسيس الدولة السعودية",
                        "year_mentions": ["1727"],
                        "exact_excerpt": "تذكر الوثيقة تاريخ تأسيس الدولة السعودية.",
                        "status": "unverified_research_question",
                    }],
                }])
                store.put("focus_run_" + PROFILE, {
                    "run_id": "unit-inspector", "status": "running",
                    "heartbeat_at_utc": "2026-11-05T04:35:00+00:00",
                    "current_phase": "waiting_for_next_cycle",
                    "pages_read": 1, "pages_archived": 1,
                    "training_steps_this_run": 0,
                    "training_runs_with_new_weights": 0,
                    "training_status": "skipped_no_eligible_training_corpus",
                    "last_cycle_pages_attempted": 0,
                    "last_cycle_page_target": 30,
                    "search_status": "search_pacing_wait",
                    "pending_research_urls": 0,
                    "last_cycle_idle_reason":
                        "no_pending_urls_and_search_produced_no_candidates",
                    "last_archive_recovery": {
                        "source_snapshots_checked": 5,
                        "new_candidate_urls": 0,
                        "network_requests": 0,
                    },
                })
                store.log("focus_cycle", "completed", {
                    "run_id": "unit-inspector", "cycles": 1,
                    "outcomes": [{"status": "queue_empty"}],
                    "training_status": "skipped_no_eligible_training_corpus",
                })
                lock = acquire_lock(store, PROFILE)
                controller = LearningController(cfg, interval=30)
                result = controller.focus_inspector()
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["worker"]["process_running"])
                self.assertEqual(result["worker"]["search_status"],
                                 "search_pacing_wait")
                self.assertEqual(result["latest_cycle"]["last_reason"],
                                 "queue_empty")
                self.assertEqual(result["worker"]["last_cycle_idle_reason"],
                                 "no_pending_urls_and_search_produced_no_candidates")
                self.assertEqual(result["worker"]["last_archive_recovery"]
                                 ["source_snapshots_checked"], 5)
                self.assertEqual(
                    result["curiosity"]["unverified_question_count"], 1)
                self.assertEqual(
                    result["curiosity"]["verified_autonomous_answers"], 0)
                self.assertIsNone(result["curiosity"]["latest_questions"]
                                  [0]["model_answer"])
                self.assertEqual(result["research"]["archived_research_records"], 1)
                self.assertEqual(result["training"]["approved_pages"], 0)
                self.assertIsNone(
                    result["training"]["last_accepted_checkpoint"])
                self.assertIsNone(result["training"]["last_attempt"])
                self.assertFalse(
                    result["training"]["weights_updated_in_current_run"])
                self.assertFalse(
                    result["capabilities"]["focus_model_fluent_qa_implemented"])
                self.assertFalse(
                    result["capabilities"]["independently_verifies_historical_claims"])
                self.assertTrue(worker_running(store, PROFILE))
            finally:
                if controller is not None:
                    controller.shutdown()
                if lock is not None:
                    release_lock(lock)
                store.close()

    def test_separate_verified_source_answer_is_not_focus_model_output(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self._settings(root)
            store = Store(root)
            url = "https://example.org/saudi-history"
            try:
                store.put("focus_observations_" + PROFILE, [{
                    "title": "History", "url": url, "host": "example.org",
                    "rights": "UNKNOWN", "noai": False,
                    "source_sha256": "f" * 64, "curiosity": [],
                }])
                store.add_research_question(
                    "History", "What is history?", url, "History")
                row = store.db.execute(
                    "SELECT id FROM research_questions WHERE source_url=?",
                    (url,)).fetchone()
                store.save_verified_answer(
                    row["id"], "History", "A record of past events",
                    evidence=[
                        {"publisher": "example.org", "license": "UNKNOWN"},
                        {"publisher": "different.example.net",
                         "license": "UNKNOWN"},
                    ],
                    method="two_source_definition_overlap_v1")
                ctrl = LearningController(cfg, interval=30)
                try:
                    result = ctrl.focus_inspector()
                    q = result["curiosity"]
                    self.assertEqual(
                        q["separately_recorded_verified_answer_count"], 1)
                    self.assertEqual(q["verified_autonomous_answers"], 0)
                    self.assertEqual(q["model_generated_answers"], 0)
                    entry = q["separately_recorded_verified_answers"][0]
                    self.assertFalse(entry["focus_model_generated"])
                    self.assertFalse(entry["training_eligible"])
                    self.assertEqual(entry["answer"], "A record of past events")
                    self.assertEqual(entry["source_url"], url)
                finally:
                    ctrl.shutdown()
            finally:
                store.close()

    def test_last_attempt_and_accepted_checkpoint_not_conflated(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self._settings(root)
            store = Store(root)
            try:
                checkpoint_dir = (Path(root) / "models" / "focus" / PROFILE)
                checkpoint_dir.mkdir(parents=True)
                valid_file = checkpoint_dir / "accepted.pt"
                saved_bytes = b"checkpoint bytes used by integrity test"
                valid_file.write_bytes(saved_bytes)
                valid_digest = hashlib.sha256(saved_bytes).hexdigest()
                store.put("focus_train_attempted_" + PROFILE, {
                    "status": "trained", "accepted": False,
                    "steps": 12, "validation_before": 3.1,
                    "validation_after": 3.5, "sha256": "a" * 64,
                    "checkpoint": str(checkpoint_dir / "not-present.pt"),
                })
                store.put("focus_train_latest_" + PROFILE, {
                    "status": "trained", "accepted": True,
                    "steps": 10, "steps_total": 10,
                    "validation_before": 4.2,
                    "validation_after": 3.9,
                    "sha256": valid_digest,
                    "checkpoint": str(valid_file),
                })
                ctrl = LearningController(cfg, interval=30)
                try:
                    result = ctrl.focus_inspector()
                    self.assertFalse(
                        result["training"]["last_attempt"]["accepted"])
                    self.assertTrue(
                        result["training"]["last_accepted_checkpoint"]
                        ["accepted"])
                    self.assertNotEqual(
                        result["training"]["last_attempt"]["sha256"],
                        result["training"]["last_accepted_checkpoint"]["sha256"])
                    self.assertEqual(
                        result["training"]["last_accepted_checkpoint"]
                        ["steps_total"], 10)
                    self.assertTrue(
                        result["training"]["last_accepted_checkpoint"]
                        ["checkpoint_sha256_verified"])
                    self.assertFalse(
                        result["training"]["last_attempt"]
                        ["checkpoint_sha256_verified"])
                finally:
                    ctrl.shutdown()
            finally:
                store.close()

    def test_local_inspector_get_is_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self._settings(root)
            controller = LearningController(cfg, interval=30)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                _handler(controller, "local-key", "http://127.0.0.1:0"))
            server.RequestHandlerClass = _handler(
                controller, "local-key",
                f"http://127.0.0.1:{server.server_port}")
            thread = threading.Thread(
                target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=5)
                conn.request("GET", "/api/focus/inspector")
                response = conn.getresponse()
                body = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(body["status"], "ok")
                self.assertEqual(body["training"]["approved_pages"], 0)
                self.assertFalse(body["worker"]["process_running"])
                conn.close()
                check = Store(root)
                try:
                    self.assertIsNone(check.get("focus_run_" + PROFILE))
                finally:
                    check.close()
            finally:
                server.shutdown()
                server.server_close()
                controller.shutdown()
                thread.join(timeout=3)

    def test_modal_exists_and_polls_without_page_reload(self):
        html = (Path(__file__).resolve().parents[1] /
                "evolve" / "studio.html").read_text("utf-8")
        for part in (
            'id="open-inspector"', 'id="inspector-panel"',
            'role="dialog" aria-modal="true"',
            'id="inspector-close"',
            'id="inspector-tab-overview"',
            'id="inspector-tab-research"',
            'id="inspector-tab-curiosity"',
            'id="inspector-tab-training"',
            'id="inspector-content"',
            'setInterval(refreshInspector,3000)',
            '"/api/focus/inspector"',
            "inspectorData=d" if False else "inspectorData=result",
            'document.body.classList.add("inspector-open")',
            'if(e.key==="Escape")',
        ):
            self.assertIn(part, html)
        self.assertNotIn("location.reload(true)", html)
        self.assertIn("textContent", html)
        self.assertIn("full worker event history", html)
        self.assertIn("separately_recorded_verified_answers", html)
        self.assertIn("last_cycle_idle_reason", html)
        self.assertIn("focus-recent-activity", html)
        self.assertIn("Gradually load all saved events on scroll", html)

    def test_ui_javascript_is_syntactically_valid_when_node_installed(self):
        binary = shutil.which("node")
        if not binary:
            self.skipTest("Node not installed on this test runner")
        html = (Path(__file__).resolve().parents[1] /
                "evolve" / "studio.html").read_text("utf-8")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "inspector-script.js"
            file.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [binary, "--check", str(file)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
