"""Local-only BERHOT Learning Studio: status, train jobs, supervised QA and chat.

No external frontend backend, no remote model API, no unattended crawler by
default. The Python process must remain running for any scheduled checks.
"""
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import RLock, Thread, Event
import html
import json
import math
import secrets
import time
from urllib.parse import urlsplit

from .store import Store
from .open_corpus import corpus_seed
from .web_discovery import discover_web
from .v117 import train as train_language, prepare
from .v118_core import (ask, add_qa_example, train_qa, qa_readiness,
                        curriculum)

HOST = "127.0.0.1"
MAX_BODY = 16384


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Expected integer between {minimum} and {maximum}")
    return value


class LearningController:
    def __init__(self, settings, interval=3600):
        self.settings = settings
        self.interval = _safe_int(interval, 10, 86400)
        self.lock = RLock()
        self.active = None
        self.auto_enabled = False
        self.next_check_at = None
        self.last_check_at = None
        self.stop_event = Event()
        self.started_at = time.monotonic()
        self.monitor = Thread(target=self._monitor_loop, daemon=True)
        self.monitor.start()

    def _store(self):
        return Store(self.settings["storage_root"])

    def shutdown(self):
        self.stop_event.set()
        with self.lock:
            self.auto_enabled = False

    def status(self):
        with self.lock:
            auto_enabled = self.auto_enabled
            current_job = self.active is not None and self.active.is_alive()
            next_check_at = self.next_check_at
            last_check_at = self.last_check_at
        store = self._store()
        try:
            rows = store.db.execute(
                "SELECT COUNT(*) AS pages, COALESCE(SUM(characters),0) AS chars "
                "FROM web_corpus_texts").fetchone()
            approved = store.db.execute(
                "SELECT COUNT(*) AS articles,COUNT(DISTINCT host) AS hosts "
                "FROM web_corpus_sources WHERE noai=0 AND "
                "rights_status IN ('CC0-1.0','PUBLIC-DOMAIN')").fetchone()
            pending = store.db.execute(
                "SELECT COUNT(*) FROM web_frontier WHERE status='pending'"
            ).fetchone()[0]
            qa_examples = store.get("v118_qa_examples", []) or []
            v117 = store.get("v117_latest")
            qa_model = store.get("v118_qa_latest")
            qa_plan = qa_readiness(store) if qa_examples else {
                "status": "needs_reviewed_examples"}
            job = store.get("v118_last_job")
            history = store.get("v118_job_history", []) or []
            if job and job["status"] == "running":
                # If the process restarted, persisted 'running' is stale.
                if not current_job:
                    job = dict(job, status="interrupted",
                               note="A past server process ended during this job")
                else:
                    started = datetime.fromisoformat(job["started_at_utc"])
                    elapsed = max(0.0, (datetime.now(timezone.utc) -
                                        started).total_seconds())
                    job = dict(job, elapsed_seconds=round(elapsed, 1))
                    done = job.get("completed_steps", 0)
                    target = job.get("target_steps", 0)
                    if 0 < done < target and elapsed >= 2:
                        job["estimated_remaining_seconds"] = round(
                            elapsed * (target - done) / done, 1)
            return {
                "status": "online", "time_utc": _utc(),
                "server_uptime_seconds": int(time.monotonic() - self.started_at),
                "archived_pages": int(rows["pages"]),
                "archived_characters": int(rows["chars"]),
                "rights_eligible_articles": int(approved["articles"]),
                "rights_eligible_hosts": int(approved["hosts"]),
                "pending_crawl_urls": int(pending),
                "human_reviewed_qa_examples": len(qa_examples),
                "qa_training_readiness": qa_plan,
                "v117_models": {
                    k: {"sha256": v["sha256"],
                        "steps": v["actual_training_steps"],
                        "val_bits_per_byte": v["validation_bits_per_byte_after"],
                        "accepted_by_validation": v["accepted_by_validation"]}
                    for k, v in (v117.get("candidates", {}) if v117 else {}).items()
                },
                "v117_dataset_sha256": v117.get("dataset_sha256") if v117 else None,
                "v118_qa_model": {
                    "sha256": qa_model["sha256"],
                    "steps": qa_model["actual_training_steps"],
                    "val_answer_loss": qa_model["validation_answer_loss_after"]
                } if qa_model else None,
                "job": job,
                "job_history": history[-12:][::-1],
                "auto": {
                    "enabled": auto_enabled, "next_check_utc": next_check_at,
                    "last_check_utc": last_check_at, "interval_seconds": self.interval,
                    "minimum_training_articles": 12,
                    "minimum_training_bytes": 60000,
                    "network_crawl_in_auto_mode": False,
                    "rights_auto_approval": False,
                    "new_dataset_required": True,
                    "note": "Only while this local server runs. Checks new reviewed data and starts bounded experiments; not independently chosen research goals.",
                },
            }
        finally:
            store.close()

    def queue(self, urls):
        if (not isinstance(urls, list) or
            not 1 <= len(urls) <= 10 or
            any(not isinstance(x, str) or len(x) > 2048 for x in urls)):
            raise ValueError("Submit 1..10 HTTPS article URLs")
        store = self._store()
        try:
            return corpus_seed(store, urls, cross_host=False)
        finally:
            store.close()

    def lesson(self, question, answer, url):
        store = self._store()
        try:
            return add_qa_example(store, question, answer, url)
        finally:
            store.close()

    def talk(self, question, mode):
        with self.lock:
            if self.active and self.active.is_alive() and mode != "sources":
                raise ValueError("Neural chat is disabled while training; use source lookup")
        store = self._store()
        try:
            return ask(store, question, mode)
        finally:
            store.close()

    def plan(self):
        store = self._store()
        try:
            return curriculum(store)
        finally:
            store.close()

    def set_auto(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Auto enabled must be true or false")
        with self.lock:
            self.auto_enabled = enabled
            if enabled:
                self.next_check_at = _utc()
            else:
                self.next_check_at = None
        return {"status": "updated", "auto_enabled": enabled,
                "warning": "Only detects new eligible datasets and trains bounded experiments. Does not approve rights or gather new sites by itself."}

    def submit(self, kind, steps=None, automatic=False):
        if kind not in ("language", "qa", "crawl"):
            raise ValueError("Allowed jobs: language, qa, crawl")
        if steps is None:
            steps = 120 if kind == "language" else (60 if kind == "qa" else 5)
        steps = _safe_int(steps, 1, 500 if kind != "crawl" else 10)
        with self.lock:
            if self.active is not None and self.active.is_alive():
                return {"status": "busy", "reason": "A training/crawl job is running"}
            job = {
                "id": secrets.token_hex(8), "kind": kind,
                "status": "running", "phase": "starting",
                "started_at_utc": _utc(), "automatic": bool(automatic),
                "completed_steps": 0,
                "target_steps": steps * 2 if kind == "language" else steps,
                "training_loss": None,
            }
            t = Thread(target=self._worker, args=(job, steps), daemon=True)
            store = self._store()
            try:
                store.put("v118_last_job", job)
            finally:
                store.close()
            self.active = t
            t.start()
        return {"status": "started", "job_id": job["id"], "kind": kind,
                "target_steps": job["target_steps"]}

    def _worker(self, job, steps):
        store = self._store()
        started = time.monotonic()
        last_saved = {"at": 0.0}

        def progress(phase, step, target, loss):
            job["phase"] = phase
            job["completed_steps"] = step
            job["target_steps"] = target
            if loss is not None:
                job["training_loss"] = loss
            now = time.monotonic()
            if now - last_saved["at"] >= 0.2 or step == target:
                store.put("v118_last_job", job)
                last_saved["at"] = now
        try:
            if job["kind"] == "language":
                progress("data_validation", 0, 2 * steps, None)
                result = train_language(store, self.settings,
                                        steps=steps, progress=progress)
            elif job["kind"] == "qa":
                progress("qa_readiness", 0, steps, None)
                result = train_qa(store, self.settings,
                                  steps=steps, progress=progress)
            else:
                progress("robots_and_crawl", 0, steps, None)
                result = discover_web(store, self.settings, limit=steps,
                                      review_questions=False)
            job["result"] = result
            job["status"] = (
                "completed" if result.get("status") in (
                    "trained", "daily_budget_exhausted", "completed")
                else result.get("status", "completed"))
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"[:350]
        finally:
            job["finished_at_utc"] = _utc()
            job["duration_seconds"] = round(time.monotonic() - started, 2)
            job["phase"] = "finished"
            # Avoid storing very large model reports in the status history.
            if isinstance(job.get("result"), dict):
                res = job["result"]
                job["result_summary"] = {
                    "status": res.get("status"),
                    "reason": res.get("reason"),
                    "dataset_sha256": res.get("dataset_sha256"),
                    "previous_checkpoints_unchanged":
                        res.get("previous_checkpoints_unchanged"),
                }
                del job["result"]
            try:
                history = store.get("v118_job_history", []) or []
                history.append(job)
                store.put("v118_job_history", history[-100:])
                store.put("v118_last_job", job)
                store.log("v118_studio_job", job["status"], job)
            finally:
                store.close()

    def _monitor_loop(self):
        # Wake every second so enabling a previously idle monitor takes
        # effect promptly even when the configured interval is an hour.
        while not self.stop_event.wait(1):
            with self.lock:
                if not self.auto_enabled or (
                    self.active is not None and self.active.is_alive()):
                    continue
                next_check = self.next_check_at
            now = datetime.now(timezone.utc)
            if next_check:
                try:
                    if now < datetime.fromisoformat(next_check):
                        continue
                except (TypeError, ValueError):
                    pass
            with self.lock:
                self.last_check_at = _utc()
                from datetime import timedelta
                self.next_check_at = (now + timedelta(seconds=self.interval)).isoformat()
            store = self._store()
            try:
                # Automatic learning never allows UNKNOWN rights, same-source
                # diagnostic splits, or recycling an already trained dataset.
                info = curriculum(store)
                new_sha = info.get("frozen_dataset_sha256")
                last = store.get("v117_latest")
                eligible = info["eligible_articles"]
                train_bytes = info["bytes_per_split"].get("train", 0)
                if (info["ready_for_independent_training"] and
                    new_sha and (not last or last["dataset_sha256"] != new_sha)
                    and eligible >= 12 and train_bytes >= 60000):
                    store.log("v118_auto_check", "eligible",
                              {"dataset_sha256": new_sha, "articles": eligible})
                    # The worker will independently recompute rights and
                    # SHA-verified data before training.
                    store.close()
                    store = None
                    self.submit("language", steps=120, automatic=True)
                else:
                    store.log("v118_auto_check", "not_ready", {
                        "eligible_articles": eligible,
                        "train_bytes": train_bytes,
                        "new_dataset": bool(new_sha and (not last or
                                                      last["dataset_sha256"] != new_sha)),
                        "required_articles": 12,
                        "required_train_bytes": 60000,
                    })
            except Exception as exc:
                if store:
                    store.log("v118_auto_check", "error",
                              {"error": str(exc)[:300]})
            finally:
                if store:
                    store.close()


def _handler(controller, token, origin):
    class StudioHandler(BaseHTTPRequestHandler):
        server_version = "BERHOT-Studio/1.18"
        def _headers(self, code, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
                             "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()

        def _host_ok(self):
            # Reject DNS rebinding, malformed absolute-URI requests and
            # explicitly remote hosts even if an attacker reaches loopback.
            host = self.headers.get("Host", "")
            return host == urlsplit(origin).netloc and self.path.startswith("/")

        def _reply(self, result, code=200):
            data = json.dumps(result, ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
            self._headers(code, "application/json; charset=utf-8")
            self.wfile.write(data)

        def do_GET(self):
            if not self._host_ok():
                return self._reply({"error": "host_not_allowed"}, 403)
            if self.path == "/":
                template = (Path(__file__).resolve().parent /
                            "studio.html").read_text(encoding="utf-8")
                page = template.replace("__EVOLVE_TOKEN__",
                                        html.escape(token, quote=True))
                self._headers(200, "text/html; charset=utf-8")
                self.wfile.write(page.encode("utf-8"))
            elif self.path == "/api/status":
                self._reply(controller.status())
            elif self.path == "/api/plan":
                self._reply(controller.plan())
            else:
                self._reply({"error": "not_found"}, 404)

        def do_POST(self):
            if not self._host_ok():
                return self._reply({"error": "host_not_allowed"}, 403)
            if self.headers.get("Origin") != origin:
                return self._reply({"error": "origin_not_allowed"}, 403)
            if not secrets.compare_digest(self.headers.get("X-Evolve-Key", ""),
                                          token):
                return self._reply({"error": "csrf_token_required"}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Request body too large or missing")
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("Expected JSON object")
                if self.path == "/api/chat":
                    result = controller.talk(body.get("question"), body.get("mode", "sources"))
                elif self.path == "/api/lesson":
                    result = controller.lesson(body.get("question"), body.get("answer"),
                                               body.get("url"))
                elif self.path == "/api/queue":
                    result = controller.queue(body.get("urls"))
                elif self.path == "/api/job":
                    result = controller.submit(body.get("kind"),
                                               steps=body.get("steps"))
                elif self.path == "/api/auto":
                    result = controller.set_auto(body.get("enabled"))
                else:
                    return self._reply({"error": "not_found"}, 404)
                self._reply(result, 200 if result.get("status") != "busy" else 409)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._reply({"status": "error", "error": str(exc)[:400]}, 400)
            except Exception as exc:
                self._reply({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]},
                            500)

        def log_message(self, fmt, *args):
            # Do not log private questions or excerpts into process stdout.
            if self.path.startswith("/api/"):
                return
            super().log_message(fmt, *args)
    return StudioHandler


def serve(settings, port=8787, interval=3600):
    port = _safe_int(port, 1024, 65535)
    control = LearningController(settings, interval=interval)
    token = secrets.token_hex(32)
    origin = f"http://{HOST}:{port}"
    server = ThreadingHTTPServer((HOST, port), _handler(control, token, origin))
    print(f"BERHOT Learning Studio: {origin}", flush=True)
    print("Local-only. Press Ctrl+C to stop; automatic training runs only while Studio remains open.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        control.shutdown()
        server.server_close()
