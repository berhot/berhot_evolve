"""Bounded overnight research supervisor for opt-in Arabic Saudi history.

This is an *on-device process* the user starts; ChatGPT does not operate the
user's Mac in the background. It never searches arbitrarily all domains,
bypasses robots, grants licenses, or confuses fetched pages with model updates.
"""
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlsplit, unquote
import hashlib
import json
import signal
import time

from .focus_data import get_profile, topic_relevance, arabic_curiosity, review_leads
from .focus_training import train_if_eligible, readiness as training_readiness
from .open_corpus import archive_page
from .v119_curiosity import budget_status
from .web_discovery import (
    canonical_url, extract, fetch_page, detect_declared_rights,
)

_KEY = "focus_run_"
_HISTORY = "focus_observations_"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _public_state(state):
    return {key: value for key, value in state.items()
            if key not in ("_queue", "_seen", "_seeded", "_last_errors")}


def _enqueue(state, url, depth=0):
    try:
        url = canonical_url(url)
    except (ValueError, AttributeError):
        return False
    if urlsplit(url).hostname not in state["allowed_hosts"]:
        return False
    if url in state["_seen"] or any(item["url"] == url
                                   for item in state["_queue"]):
        return False
    state["_queue"].append({"url": url, "depth": int(depth)})
    return True


def plan(profile_name="saudi-history-ar", extra_urls=None):
    profile = get_profile(profile_name)
    seeds = list(profile["seeds"])
    for url in extra_urls or []:
        canonical = canonical_url(url)
        if urlsplit(canonical).hostname not in profile["allowed_hosts"]:
            raise ValueError(
                "Extra seed host requires a separately reviewed profile; "
                "overnight mode cannot crawl arbitrary new domains")
        seeds.append(canonical)
    return {
        "status": "ready_for_user_initiated_overnight_run",
        "profile": profile_name, "focus": profile["name_ar"],
        "language": profile["language"],
        "allowed_hosts": list(profile["allowed_hosts"]),
        "seed_urls": list(dict.fromkeys(seeds)),
        "no_training_rights_inferred": True,
        "research_only_until_per_article_rights_review": True,
        "arabic_curiosity_from_real_sentences": True,
        "self_correction": "cross_publisher_date_review_leads_only_no_auto_truth",
        "network_requests_without_explicit_run": 0,
    }


def init_state(store, profile_name, extra_urls=None, max_pages=24):
    if not 1 <= int(max_pages) <= 80:
        raise ValueError("max_pages must be 1..80")
    profile = get_profile(profile_name)
    base = plan(profile_name, extra_urls)
    previous = store.get(_KEY + profile_name) or {}
    now = _utc()
    state = {
        "version": "focus-overnight-1.0",
        "profile": profile_name, "focus": profile["name_ar"],
        "allowed_hosts": list(profile["allowed_hosts"]),
        "run_id": now.replace(":", "").replace("-", ""),
        "started_at_utc": now,
        "updated_at_utc": now,
        "finished_at_utc": None,
        "status": "initialized",
        "max_pages_this_run": int(max_pages),
        "pages_attempted": 0, "pages_archived": 0,
        "new_unique_pages": 0, "exact_duplicate_pages": 0,
        "skipped_off_topic_or_non_arabic": 0,
        "errors": 0,
        "articles_with_unknown_rights": 0,
        "articles_with_explicit_rights_signal": 0,
        "candidate_questions_created_this_run": 0,
        "hours_elapsed": 0.0,
        "cycles_completed": 0,
        "training_runs_with_new_weights": 0,
        "training_steps_this_run": 0,
        "training_status": "not_checked",
        "latest_training": None,
        "last_sources": [],
        "last_errors": [],
        "last_candidate_questions": [],
        "research_review_leads": [],
        "research_records_saved": 0,
        "previous_run_id": previous.get("run_id"),
        "_queue": [], "_seen": set(),
        "_seeded": base["seed_urls"],
        "_last_errors": [],
    }
    # Read-only separation from other EVOLVE frontier and learning queues.
    for url in base["seed_urls"]:
        _enqueue(state, url)
    store.put(_KEY + profile_name, _public_state(state))
    return state


def _topic_link(url, profile):
    raw = unquote(urlsplit(url).path).casefold()
    # Do not follow broad navigation indexes, login, policies or file assets.
    if any(x in raw for x in ("/footer/", "/login", "/account/",
                              ".pdf", ".jpg", ".png", ".zip",
                              "/tags/", "/category/", "/search/")):
        return False
    if len(raw) > 220:
        return False
    return any(x.casefold() in raw for x in profile["url_terms"])


def _run_page(store, settings, state, profile, fetcher=None,
              sleep_fn=time.sleep):
    if not state["_queue"]:
        return {"status": "queue_empty"}
    item = state["_queue"].pop(0)
    url, depth = item["url"], item["depth"]
    if url in state["_seen"]:
        return {"status": "already_seen", "url": url}
    state["_seen"].add(url)
    if state["pages_attempted"] >= state["max_pages_this_run"]:
        return {"status": "run_cap_reached"}
    quota = budget_status(store, settings)
    if quota["remaining"] <= 0:
        # Keep the candidate for tomorrow after the shared UTC reset.
        state["_seen"].remove(url)
        state["_queue"].insert(0, item)
        return {"status": "daily_budget_exhausted"}
    key = "web_requests_" + quota["day_utc"]
    store.put(key, quota["used"] + 1)
    state["pages_attempted"] += 1
    try:
        # fetch_page already enforces TLS, DNS public-only, robots, MIME,
        # response size and refusal to follow redirects.
        html = fetcher(url) if fetcher else fetch_page(url, settings)
        content, title, links, noai = extract(html)
        quality = topic_relevance(content, title, profile)
        if not quality["language_ok"] or not quality["topical"] or len(content) < 400:
            state["skipped_off_topic_or_non_arabic"] += 1
            return {"status": "off_topic_or_non_arabic",
                    "url": url, "language_ratio_ar": quality["language_ratio_ar"]}
        # A re-fetched source might have an earlier human license review;
        # don't silently replace that review with UNKNOWN.
        existing = store.db.execute(
            "SELECT sha256,rights_status,noai FROM web_corpus_sources "
            "WHERE url=?", (url,)).fetchone()
        if existing and existing["rights_status"] != "UNKNOWN":
            # Cannot refresh a previously approved source without a
            # re-review of its new content; use only existing immutable data.
            return {"status": "already_rights_reviewed_no_overwrite", "url": url}
        declared, evidence = detect_declared_rights(html, url)
        outcome = archive_page(store, url, html, content, title=title,
                               noai=noai, declared_license=declared,
                               rights_url=evidence)
        if outcome["status"] != "archived":
            return {"status": "skipped_archive", "url": url}
        state["pages_archived"] += 1
        state["new_unique_pages"] += int(outcome["new_unique_text"])
        state["exact_duplicate_pages"] += int(not outcome["new_unique_text"])
        state["articles_with_unknown_rights"] += int(
            outcome["rights_status"] == "UNKNOWN")
        state["articles_with_explicit_rights_signal"] += int(
            outcome["rights_status"] != "UNKNOWN")
        saved = {
            "url": url, "host": urlsplit(url).hostname, "title": title[:200],
            "source_sha256": outcome["clean_sha256"],
            "rights": outcome["rights_status"], "noai": bool(noai),
            "arabic_ratio": quality["language_ratio_ar"],
            "keyword_hits": quality["keyword_hits"],
            "curiosity": arabic_curiosity(
                content, url, outcome["clean_sha256"], profile),
        }
        # New quote labels remain unverified; no rights are granted.
        memory_key = _HISTORY + state["profile"]
        records = store.get(memory_key, []) or []
        old = next((x for x in records if x["url"] == url), None)
        if old is None or old["source_sha256"] != saved["source_sha256"]:
            records = [x for x in records if x["url"] != url] + [saved]
            store.put(memory_key, records[-500:])
            state["candidate_questions_created_this_run"] += len(
                saved["curiosity"])
        state["research_records_saved"] = len(records)
        state["last_sources"] = (
            state["last_sources"] + [{
                "url": url, "rights": saved["rights"],
                "new_unique_text": outcome["new_unique_text"],
                "candidate_questions": len(saved["curiosity"]),
            }])[-12:]
        state["last_candidate_questions"] = (
            state["last_candidate_questions"] + [
                {"question": q["question"], "source_url": url,
                 "status": q["status"]}
                for q in saved["curiosity"]])[-12:]
        # At most one internal hop, within explicitly allowed hosts, on
        # topical paths only. No general search-engine crawling.
        if depth == 0:
            added = 0
            for ref in links:
                if added >= 3:
                    break
                try:
                    new = canonical_url(urljoin(url, ref))
                except (ValueError, AttributeError):
                    continue
                if _topic_link(new, profile) and _enqueue(state, new, depth + 1):
                    added += 1
        return {
            "status": "archived_research_only", "url": url,
            "new_unique_text": outcome["new_unique_text"],
            "rights": outcome["rights_status"],
            "source_questions": len(saved["curiosity"]),
            "new_links_queued": added if depth == 0 else 0,
        }
    except Exception as exc:
        state["errors"] += 1
        state["last_errors"] = (state["last_errors"] + [{
            "url": url, "error": str(exc)[:250]}])[-10:]
        return {"status": "error", "url": url,
                "error": type(exc).__name__ + ": " + str(exc)[:220]}


def cycle(store, settings, state, pages=3, train_steps=40,
          fetcher=None, sleep_fn=time.sleep):
    if not 1 <= int(pages) <= 8:
        raise ValueError("pages per cycle must be 1..8")
    profile = get_profile(state["profile"])
    outcomes = []
    for i in range(int(pages)):
        if state["pages_attempted"] >= state["max_pages_this_run"]:
            break
        result = _run_page(store, settings, state, profile, fetcher=fetcher)
        if result["status"] in ("queue_empty", "daily_budget_exhausted"):
            outcomes.append(result)
            break
        outcomes.append(result)
        if (fetcher is None and i < int(pages) - 1 and
                float(settings.get("crawl_delay_seconds", 2)) > 0):
            sleep_fn(max(2.0, float(settings.get("crawl_delay_seconds", 2))))
    observations = store.get(_HISTORY + state["profile"], []) or []
    state["research_review_leads"] = review_leads(observations)[:20]
    training = train_if_eligible(store, state["profile"], steps=train_steps)
    state["training_status"] = training["status"]
    state["latest_training"] = {
        key: training.get(key) for key in (
            "status", "reason", "steps", "accepted", "checkpoint",
            "sha256", "validation_before", "validation_after",
            "trainable_parameters", "unique_approved_training_bytes",
        ) if key in training}
    if training.get("status") == "trained":
        state["training_runs_with_new_weights"] += 1
        state["training_steps_this_run"] += training["steps"]
    state["cycles_completed"] += 1
    state["updated_at_utc"] = _utc()
    store.put(_KEY + state["profile"], _public_state(state))
    store.log("focus_cycle", "completed", {
        "profile": state["profile"], "run_id": state["run_id"],
        "cycles": state["cycles_completed"], "outcomes": outcomes[:10],
        "training_status": training["status"],
    })
    return {
        "status": "cycle_completed",
        "cycle": state["cycles_completed"],
        "pages_attempted_total": state["pages_attempted"],
        "pages_archived_total": state["pages_archived"],
        "new_unique_pages_total": state["new_unique_pages"],
        "new_questions_total": state["candidate_questions_created_this_run"],
        "training": state["latest_training"],
        "outcomes": outcomes,
    }


def report(store, profile_name="saudi-history-ar"):
    profile = get_profile(profile_name)
    run = store.get(_KEY + profile_name)
    observations = store.get(_HISTORY + profile_name, []) or []
    model = store.get("focus_train_latest_" + profile_name) or {}
    approved = training_readiness(store, profile_name)
    approved.pop("_pages", None)
    approved.pop("_assignments", None)
    source_hosts = sorted({entry.get("host") for entry in observations})
    cumulative_research = len(observations)
    cumulative_questions = sum(len(x.get("curiosity", []))
                               for x in observations)
    return {
        "status": "run_found" if run else "not_started",
        "profile": profile_name,
        "focus": profile["name_ar"],
        "last_run": run,
        "observed_source_hosts": source_hosts,
        "cumulative_relevant_research_articles": cumulative_research,
        "cumulative_source_derived_unverified_questions": cumulative_questions,
        "potential_cross_publisher_review_leads": review_leads(observations),
        "rights_eligible_arabic_training": approved,
        "latest_accepted_focus_model": {
            key: model.get(key) for key in (
                "sha256", "checkpoint", "steps_total",
                "trainable_parameters", "unique_approved_training_bytes",
                "validation_after", "resumed_accepted_weights",
            ) if key in model
        } if model else None,
        "notes": {
            "cached_articles_are_not_neural_training": True,
            "model_parameter_count_does_not_grow_from_reading": True,
            "unverified_history_is_not_learned_truth": True,
            "article_rights_never_auto_approved": True,
            "automatic_fact_correction_performed": False,
            "daemon_currently_running": "not_detectable_from_saved_status",
            "mac_can_sleep_and_pause_the_worker": True,
        },
    }


def run(store, settings, profile_name="saudi-history-ar", hours=8.0,
        interval_minutes=15, max_pages=24, pages_per_cycle=3,
        train_steps=40, extra_urls=None, fetcher=None, sleep_fn=time.sleep,
        monotonic_fn=time.monotonic):
    if not 0 < float(hours) <= 12:
        raise ValueError("hours must be within (0, 12]")
    if not 1 <= int(interval_minutes) <= 60:
        raise ValueError("interval-minutes must be 1..60")
    state = init_state(store, profile_name, extra_urls, max_pages=max_pages)
    duration = float(hours)*3600
    start = monotonic_fn()
    end = start + duration
    stop = False
    previous_signals = {}
    def on_stop(_sig, _frame):
        nonlocal stop
        stop = True
    # Signal registration is possible only in the main thread.
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_signals[signum] = signal.getsignal(signum)
            signal.signal(signum, on_stop)
        except ValueError:
            continue
    state["status"] = "running"
    print(json.dumps({
        "status": "running", "profile": profile_name,
        "run_id": state["run_id"],
        "ends_after_max_hours": float(hours),
        "max_pages": max_pages,
        "note": "Keep Mac on power, lid open; sleep stops the worker.",
    }, ensure_ascii=False), flush=True)
    try:
        while not stop and monotonic_fn() < end:
            result = cycle(store, settings, state, pages=pages_per_cycle,
                           train_steps=train_steps, fetcher=fetcher,
                           sleep_fn=sleep_fn)
            state["hours_elapsed"] = round(
                (monotonic_fn() - start) / 3600, 3)
            store.put(_KEY + profile_name, _public_state(state))
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if state["pages_attempted"] >= state["max_pages_this_run"]:
                break
            remaining = end - monotonic_fn()
            if remaining <= 0:
                break
            # Responsive to SIGTERM without long full-interval sleeps.
            cadence = min(remaining, int(interval_minutes)*60)
            while cadence > 0 and not stop:
                chunk = min(cadence, 5)
                sleep_fn(chunk)
                cadence -= chunk
    finally:
        state["finished_at_utc"] = _utc()
        state["status"] = "stopped" if stop else "completed"
        state["hours_elapsed"] = round(
            max(0.0, monotonic_fn() - start)/3600, 3)
        state["updated_at_utc"] = _utc()
        store.put(_KEY + profile_name, _public_state(state))
        for sig, prior in previous_signals.items():
            signal.signal(sig, prior)
        print(json.dumps({
            "status": state["status"], "profile": profile_name,
            "pages_archived": state["pages_archived"],
            "unique_pages": state["new_unique_pages"],
            "question_candidates": state["candidate_questions_created_this_run"],
            "training_steps": state["training_steps_this_run"],
        }, ensure_ascii=False), flush=True)
    return _public_state(state)
