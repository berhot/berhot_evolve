"""Local-only BERHOT Learning Studio: status, train jobs, supervised QA and chat.

No external frontend backend, no remote model API, no unattended crawler by
default. The Python process must remain running for any scheduled checks.
"""
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import RLock, Thread, Event
import html
import errno
import json
import os
import math
import hashlib
import secrets
import time
from urllib.parse import urlsplit

from .store import Store
from .open_corpus import corpus_seed
from .web_discovery import discover_web
from .v117 import train as train_language, prepare
from .v118_core import (ask, add_qa_example, train_qa, qa_readiness,
                        curriculum)
from .v119_curiosity import (smart_lookup, research as research_topic,
                             sentence_lessons, inventory as curiosity_inventory,
                             budget_status)
from .v120_grounded import (grounded_answer, train_ranker,
                            readiness as ranker_readiness)
from .v121_autonomy import cycle as autonomous_cycle
from .focus_control import (PROFILE as FOCUS_PROFILE, model_size_snapshot,
                            worker_running, request_stop, start_detached)

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
        self.focus_launch_at = 0.0
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
            auto_lessons = store.get("v119_auto_lessons", []) or []
            v117 = store.get("v117_latest")
            qa_model = store.get("v118_qa_latest")
            grounded_ranker = store.get("v120_ranker_latest")
            focus_run = store.get("focus_run_" + FOCUS_PROFILE) or {}
            focus_live = worker_running(store, FOCUS_PROFILE)
            focus_observations = store.get("focus_observations_" + FOCUS_PROFILE, []) or []
            focus_model = store.get("focus_train_latest_" + FOCUS_PROFILE) or {}
            size = model_size_snapshot(store)
            previous_autonomy = store.get("v121_last_cycle") or {}
            qa_plan = qa_readiness(store) if (qa_examples or auto_lessons) else {
                "status": "needs_source_grounded_examples"}
            job = store.get("v118_last_job")
            history = store.get("v118_job_history", []) or []
            curiosity = curiosity_inventory(store)
            budget = budget_status(store, self.settings)
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
                "studio_backend_version": "1.28.0",
                "studio_api_version": 4,
                "supported_chat_modes": ["auto", "grounded", "sources", "char", "byte_bpe", "qa"],
                "research_budget": budget,
                "server_uptime_seconds": int(time.monotonic() - self.started_at),
                "archived_pages": int(rows["pages"]),
                "archived_characters": int(rows["chars"]),
                "rights_eligible_articles": int(approved["articles"]),
                "rights_eligible_hosts": int(approved["hosts"]),
                "pending_crawl_urls": int(pending),
                "human_reviewed_qa_examples": len(qa_examples),
                "source_derived_lessons": curiosity["source_derived_lessons"],
                "research_attempts": curiosity["research_attempts"],
                "recent_learning_events": curiosity["recent_learning_events"],
                "qa_training_readiness": qa_plan,
                "v121_autonomous_learning": {
                    "last_status": previous_autonomy.get("status"),
                    "questions_asked": previous_autonomy.get("curiosity_question_count", 0),
                    "research_requests": previous_autonomy.get("new_requests_this_cycle", 0),
                    "source_quiz": {
                        "pass": (previous_autonomy.get("local_quiz_audit") or {}).get("pass"),
                        "tested": (previous_autonomy.get("local_quiz_audit") or {}).get("tested"),
                    },
                    "neural_training": (previous_autonomy.get("neural_training") or {}).get("status"),
                    "understanding_proven": False,
                },
                "v120_grounded_ranker": {
                    "sha256": grounded_ranker["sha256"],
                    "steps": grounded_ranker["actual_training_steps"],
                    "validation_pair_accuracy":
                        grounded_ranker["validation_after"]["pair_accuracy"],
                    "validation_loss": grounded_ranker["validation_after"]["loss"],
                    "note": "Neural evidence ranking, not generated explanations",
                } if grounded_ranker else None,
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
                "model_size": size,
                "focus_agent": {
                    "profile": FOCUS_PROFILE,
                    "focus": "تاريخ المملكة العربية السعودية",
                    "status": (
                        focus_run.get("status") if focus_live
                        else "stopped_or_interrupted" if focus_run.get(
                            "status") in ("running", "idle_waiting_for_allowed_work")
                        else focus_run.get("status", "not_started")),
                    "process_running": focus_live,
                    "stop_requested": bool(
                        store.get("focus_stop_request_" + FOCUS_PROFILE, {}).get(
                            "run_id") == focus_run.get("run_id")
                    ) if focus_run.get("run_id") else False,
                    "worker_pid": focus_run.get("worker_pid"),
                    "run_id": focus_run.get("run_id"),
                    "heartbeat_at_utc": focus_run.get("heartbeat_at_utc"),
                    "until_stopped": bool(focus_run.get("until_stopped", False)),
                    "pages_attempted": focus_run.get("pages_attempted", 0),
                    "pages_archived": focus_run.get("pages_archived", 0),
                    "unique_pages_added": focus_run.get("new_unique_pages", 0),
                    "exact_duplicates": focus_run.get("exact_duplicate_pages", 0),
                    "pages_today": focus_run.get("pages_attempted_today", 0),
                    "max_pages_per_day": focus_run.get("max_pages_per_utc_day", 24),
                    "candidate_questions": focus_run.get(
                        "candidate_questions_created_this_run", 0),
                    "training_steps_this_run": focus_run.get(
                        "training_steps_this_run", 0),
                    "accepted_training_steps": focus_model.get("steps_total", 0),
                    "training_status": focus_run.get("training_status", "not_started"),
                    "observed_research_articles": len(focus_observations),
                    "distinct_research_hosts": len({
                        item.get("host") for item in focus_observations
                        if item.get("host")}),
                    "pending_research_urls": focus_run.get(
                        "pending_research_urls", 0),
                    "search_queries_this_run": focus_run.get(
                        "search_queries_this_run", 0),
                    "cross_publisher_links_queued": focus_run.get(
                        "cross_host_links_queued_this_run", 0),
                    "keyless_browser_search": True,
                    "browser_search_provider": focus_run.get(
                        "browser_search_provider", "duckduckgo"),
                    "no_fixed_research_quotas": bool(
                        focus_run.get("no_fixed_quotas", False)),
                    "source_comparison_leads": len(
                        focus_run.get("research_review_leads", [])),
                    "research_only_is_not_neural_training": True,
                    "training_needs_manual_source_rights_review": True,
                    "last_errors": focus_run.get("last_errors", [])[-3:],
                },
                "job": job,
                "job_history": history[-12:][::-1],
                "auto": {
                    "enabled": auto_enabled, "next_check_utc": next_check_at,
                    "last_check_utc": last_check_at, "interval_seconds": self.interval,
                    "minimum_training_articles": 12,
                    "minimum_training_bytes": 60000,
                    "network_crawl_in_auto_mode": True,
                    "network_crawl_limit_per_daily_autonomous_cycle": 6,
                    "source_study_when_new_approved_articles": True,
                    "grounded_ranker_on_new_approved_sources": True,
                    "network_research_triggered_by_auto_chat": True,
                    "rights_auto_approval": False,
                    "new_dataset_required": True,
                    "note": "Only while this local server runs. Operator-enabled automated cycles choose source-observed research words; max 6 public HTTPS requests per daily cycle, within shared budget and robots policy; rights are never auto-approved.",
                },
            }
        finally:
            store.close()

    def start_focus(self):
        with self.lock:
            # UI double-clicks cannot spawn unlimited detached workers
            # while an earlier one is still acquiring its OS lock.
            if time.monotonic() - self.focus_launch_at < 10.0:
                return {"status": "launching_or_running"}
            store = self._store()
            try:
                result = start_detached(
                    store, self.settings["_project_root"], FOCUS_PROFILE)
                if result.get("status") == "launched":
                    self.focus_launch_at = time.monotonic()
                return result
            finally:
                store.close()

    def stop_focus(self):
        store = self._store()
        try:
            return request_stop(store, FOCUS_PROFILE)
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
            if self.active and self.active.is_alive() and mode not in (
                    "sources", "auto", "grounded"):
                raise ValueError("Neural chat is disabled while training; use source lookup")
        store = self._store()
        try:
            if mode in ("auto", "grounded"):
                result = grounded_answer(store, question)
                if mode == "auto" and result.get("status") == "no_definition":
                    # The earlier Curiosity source lookup still searches broader
                    # topics and may launch bounded web research when missing.
                    result = smart_lookup(store, question, self.settings)
            else:
                result = ask(store, question, mode)
        finally:
            store.close()
        if mode == "auto" and result.get("status") == "needs_research":
            # Explicit Auto chat request authorizes a bounded background
            # read-only web investigation. No rights auto-admission.
            started = self.submit("research", steps=2, question=question,
                                  automatic=True)
            if started["status"] == "started":
                return {"status": "research_queued", "mode": "auto",
                        "answer": "Searching permitted public science sources for " +
                        result["topic"] + ". I will save citations, not invent an answer.",
                        "job_id": started["job_id"], "sources": [],
                        "training_updated": False}
            return {"status": "research_waiting", "mode": "auto",
                    "answer": "Another training/research job is running. Try again after it completes.",
                    "sources": [], "training_updated": False}
        return result

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

    def submit(self, kind, steps=None, automatic=False, question=None):
        if kind not in ("language", "qa", "crawl", "research", "self-study", "grounded-ranker", "curiosity-cycle"):
            raise ValueError("Allowed jobs: language, qa, crawl, research, self-study, grounded-ranker, curiosity-cycle")
        if kind == "research" and (not isinstance(question, str) or
                                    not 1 <= len(question.strip()) <= 180):
            raise ValueError("Research requires a short question")
        if steps is None:
            steps = (120 if kind in ("language", "grounded-ranker")
                     else 60 if kind in ("qa", "self-study")
                     else 2 if kind == "research" else 4 if kind == "curiosity-cycle" else 5)
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
            if kind == "research":
                job["topic"] = question.strip()[:180]
            t = Thread(target=self._worker, args=(job, steps, question), daemon=True)
            store = self._store()
            try:
                store.put("v118_last_job", job)
            finally:
                store.close()
            self.active = t
            t.start()
        return {"status": "started", "job_id": job["id"], "kind": kind,
                "target_steps": job["target_steps"]}

    def _worker(self, job, steps, question=None):
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
            elif job["kind"] == "research":
                progress("searching_public_sources", 0, 2, None)
                result = research_topic(store, self.settings, question,
                                        progress=progress)
            elif job["kind"] == "grounded-ranker":
                progress("definition_training_readiness", 0, steps, None)
                result = train_ranker(store, self.settings, steps=steps,
                                      progress=progress)
            elif job["kind"] == "curiosity-cycle":
                progress("discovering_source_concepts", 0, steps, None)
                result = autonomous_cycle(
                    store, self.settings, network=True,
                    max_topics=min(12, steps), max_requests=min(6, steps),
                    steps=120, progress=progress)
            elif job["kind"] == "self-study":
                progress("extracting_source_exercises", 0, steps, None)
                mined = sentence_lessons(store)
                ready = qa_readiness(store)
                if ready.get("status") == "ready":
                    result = train_qa(store, self.settings, steps=steps,
                                      progress=progress)
                    result["source_derived_extraction"] = mined
                else:
                    result = {"status": "skipped",
                              "reason": "not_enough_approved_source_exercises",
                              "readiness": ready,
                              "source_derived_extraction": mined}
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
                    "quiz_pass": (res.get("local_quiz_audit") or {}).get("pass"),
                    "quiz_total": (res.get("local_quiz_audit") or {}).get("tested"),
                    "neural_training_status":
                        (res.get("neural_training") or {}).get("status"),
                    "research_requests": res.get("new_requests_this_cycle"),
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
                # Self-study uses ONLY local approved source text. One bounded
                # attempt per rights-approved content revision; never retry a
                # rejected model forever on the exact same examples.
                revision_rows = store.db.execute(
                    "SELECT url,sha256 FROM web_corpus_sources WHERE noai=0 "
                    "AND rights_status IN ('CC0-1.0','PUBLIC-DOMAIN') "
                    "ORDER BY url").fetchall()
                revision = hashlib.sha256(json.dumps(
                    [[r["url"], r["sha256"]] for r in revision_rows]
                ).encode("utf-8")).hexdigest()
                if (len(revision_rows) >= 3 and
                    revision != store.get("v119_auto_study_revision")):
                    store.put("v119_auto_study_revision", revision)
                    store.log("v119_auto_check", "source_study_ready",
                              {"approved_sources": len(revision_rows),
                               "revision_sha256": revision})
                    store.close()
                    store = None
                    self.submit("self-study", steps=60, automatic=True)
                    continue
                # Attempt one validated evidence-ranking experiment on each
                # revised approved corpus. Insufficient facts lead to an
                # explicit skipped job, never repeated on the same revision.
                if (len(revision_rows) >= 3 and
                    revision != store.get("v120_auto_ranker_revision")):
                    store.put("v120_auto_ranker_revision", revision)
                    store.log("v120_auto_check", "ranking_study_requested",
                              {"approved_sources": len(revision_rows),
                               "revision_sha256": revision})
                    store.close()
                    store = None
                    self.submit("grounded-ranker", steps=120, automatic=True)
                    continue
                # Opt-in Studio monitor: once per UTC day, ask questions
                # derived from stored observations and attempt a bounded
                # network-curiosity study. No automatic rights admission.
                day = now.strftime("%Y-%m-%d")
                if store.get("v121_auto_cycle_day") != day:
                    store.put("v121_auto_cycle_day", day)
                    store.log("v121_auto_check", "curiosity_cycle_requested",
                              {"utc_day": day, "max_requests": 6})
                    store.close()
                    store = None
                    self.submit("curiosity-cycle", steps=6, automatic=True)
                    continue
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
        server_version = "BERHOT-Studio/1.28"
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
                elif self.path == "/api/focus/start":
                    result = controller.start_focus()
                elif self.path == "/api/focus/stop":
                    result = controller.stop_focus()
                elif self.path == "/api/study":
                    result = controller.submit("self-study", steps=body.get("steps", 60))
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
    try:
        server = ThreadingHTTPServer((HOST, port), _handler(control, token, origin))
    except OSError as exc:
        control.shutdown()
        if getattr(exc, "errno", None) == errno.EADDRINUSE:
            raise RuntimeError(
                f"Port {port} is already used by another process. "
                f"Stop the previous BERHOT Studio with Ctrl+C, or inspect "
                f"its PID with: lsof -nP -iTCP:{port} -sTCP:LISTEN. "
                f"Do not run two Studio servers against the same storage."
            ) from exc
        raise
    print(f"BERHOT Learning Studio: {origin}", flush=True)
    print("Backend 1.28.0 / API 4; focus agent can run until manually stopped.", flush=True)
    print("Local-only. Press Ctrl+C to stop; automatic training runs only while Studio remains open.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        control.shutdown()
        server.server_close()
