"""Bounded overnight research supervisor for opt-in Arabic Saudi history.

This is an *on-device process* the user starts; ChatGPT does not operate the
user's Mac in the background. It never searches arbitrarily all domains,
bypasses robots, grants licenses, or confuses fetched pages with model updates.
"""
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlsplit, unquote
import hashlib
import json
import os
import signal
import time

from .focus_control import acquire_lock, release_lock, stop_requested

from .focus_data import get_profile, topic_relevance, arabic_curiosity, review_leads
from .focus_discovery import _candidate_link, relevant_links, run_one_search
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
            if key not in ("_queue", "_seen", "_seeded", "_last_errors",
                           "_host_day_used", "_host_last_at")}


def _publish(store, state, phase, *, url=None, event=None, detail=None):
    """Persist factual per-operation progress for the independent Studio UI.

    The browser only observes this state; refreshing it never controls the
    actual OS worker. All counters are real measured operations, not estimates.
    """
    now = _utc()
    state["current_phase"] = phase
    state["current_url"] = url
    state["updated_at_utc"] = now
    state["heartbeat_at_utc"] = now
    if event:
        payload = {
            "at_utc": now,
            "kind": event,
            "detail": str(detail or "")[:500],
            "url": url,
            "phase": phase,
        }
        # The full history is in the append-only SQLite event journal.
        # The small snapshot is retained only for older Studio clients.
        store.append_focus_activity(
            state["profile"], state["run_id"], now, event,
            phase=phase, detail=payload["detail"], url=url,
        )
        events = state.setdefault("recent_activity", [])
        events.append(payload)
        state["recent_activity"] = events[-16:]
    store.put(_KEY + state["profile"], _public_state(state))


def _enqueue(state, url, depth=0):
    # The topic, not a fixed website list, determines exploration.
    # All fetched pages remain subject to the strict existing network/rights
    # checks. Keep the pending frontier finite in memory/on disk.
    url = _candidate_link(url)
    if not url or not 0 <= int(depth) <= 3:
        return False
    if len(state["_queue"]) >= 500:
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
        if not _candidate_link(canonical):
            raise ValueError("Extra seed is not an eligible public HTTPS article URL")
        seeds.append(canonical)
    return {
        "status": "ready_for_user_initiated_overnight_run",
        "profile": profile_name, "focus": profile["name_ar"],
        "language": profile["language"],
        "initial_seed_hosts": list(profile["initial_seed_hosts"]),
        "no_permanent_website_allowlist": True,
        "discovery": "cross_publisher_topical_links_and_keyless_headless_search",
        "search_provider": "duckduckgo (Google/Bing automated result access blocked by current policies)",
        "search_api_key_required": False,
        "optional_browser_install_needed": True,
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
    # Preserve any activity still visible from a pre-journal worker before
    # creating a new run; older evicted events cannot be reconstructed.
    store.backfill_focus_activity(profile_name, previous)
    now = _utc()
    state = {
        "version": "focus-overnight-1.0",
        "profile": profile_name, "focus": profile["name_ar"],
        "initial_seed_hosts": list(profile["initial_seed_hosts"]),
        "no_permanent_website_allowlist": True,
        "run_id": now.replace(":", "").replace("-", ""),
        "started_at_utc": now,
        "updated_at_utc": now,
        "finished_at_utc": None,
        "status": "initialized",
        "max_pages_this_run": int(max_pages),
        "max_pages_per_utc_day": int(max_pages),
        "pages_attempted_today": 0,
        "pages_day_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "until_stopped": False,
        "no_fixed_quotas": False,
        "browser_search_provider": "duckduckgo",
        "minimum_host_gap_seconds": 20,
        "_host_last_at": {},
        "worker_pid": None,
        "heartbeat_at_utc": now,
        "current_phase": "not_started",
        "current_url": None,
        "recent_activity": [],
        "pages_read": 0,
        "html_bytes_read": 0,
        "text_characters_extracted": 0,
        "text_words_extracted": 0,
        "text_utf8_byte_tokens_extracted": 0,
        "unique_archived_utf8_byte_tokens": 0,
        "training_byte_tokens_processed": 0,
        "current_training_steps_done": 0,
        "current_training_steps_target": 0,
        "current_training_loss": None,
        "stop_requested": False,
        "pages_attempted": 0, "pages_archived": 0,
        "new_unique_pages": 0, "exact_duplicate_pages": 0,
        "skipped_off_topic_or_non_arabic": 0,
        "errors": 0,
        "articles_with_unknown_rights": 0,
        "articles_with_explicit_rights_signal": 0,
        "candidate_questions_created_this_run": 0,
        "hours_elapsed": 0.0,
        "cycles_completed": 0,
        "links_queued_this_run": 0,
        "cross_host_links_queued_this_run": 0,
        "search_queries_this_run": 0,
        "search_results_queued_this_run": 0,
        "search_status": "not_checked",
        "last_search_diagnostic": {},
        "distinct_research_hosts": 0,
        "pending_research_urls": 0,
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
        "_seeded": base["seed_urls"], "_host_day_used": {},
        "_last_errors": [],
    }
    # Restore discovered links across process restarts. A new run does not
    # wipe out the curiosity frontier or re-request already visited URLs.
    saved_frontier = store.get("focus_frontier_" + profile_name) or {}
    state["_seen"] = set(saved_frontier.get("seen", [])[-6000:])
    state["_queue"] = [
        item for item in saved_frontier.get("pending", [])[:500]
        if isinstance(item, dict) and _candidate_link(item.get("url"))
        and 0 <= item.get("depth", -1) <= 3
    ]
    if saved_frontier.get("day") == state["pages_day_utc"]:
        state["_host_day_used"] = dict(saved_frontier.get("host_counts", {}))
        state["pages_attempted_today"] = int(
            saved_frontier.get("pages_attempted_today", 0))
    state["_host_last_at"] = dict(
        saved_frontier.get("host_last_at", {}))
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


def _rollover_if_new_utc_day(state):
    """Track the UTC day, retaining discovered links and visited hashes.

    Legacy bounded runs reset their per-day counters. Open-ended research
    does not stop at a numeric count but still obeys robots and pacing.
    """
    if not state.get("until_stopped"):
        return False
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["pages_day_utc"] == today:
        return False
    state["pages_day_utc"] = today
    state["pages_attempted_today"] = 0
    state["_host_day_used"] = {}
    # Keep learned links and visited URLs across days; don't waste nightly
    # quotas on the same unchanged pages. Once weekly re-explore original
    # entry points for new topical links (never grant rights automatically).
    if datetime.now(timezone.utc).toordinal() % 7 == 0:
        for seed in state["_seeded"]:
            state["_seen"].discard(seed)
            _enqueue(state, seed)
    return True


def _run_page(store, settings, state, profile, fetcher=None,
              sleep_fn=time.sleep):
    _rollover_if_new_utc_day(state)
    if (state.get("until_stopped") and
            not state.get("no_fixed_quotas") and
            state["pages_attempted_today"] >= state["max_pages_per_utc_day"]):
        return {"status": "daily_page_cap_reached"}
    if not state["_queue"]:
        return {"status": "queue_empty"}
    item = state["_queue"].pop(0)
    url, depth = item["url"], item["depth"]
    host = urlsplit(url).hostname
    # Fair use: no single site consumes the whole day. This is a request
    # quota, NOT a permanent list of permitted websites.
    if not state.get("no_fixed_quotas") and (
            state["_host_day_used"].get(host, 0) >= 4):
        state["_queue"].append(item)
        return {"status": "publisher_daily_quota", "host": host}
    if state.get("no_fixed_quotas"):
        last = float(state["_host_last_at"].get(host, 0) or 0)
        wait = int(state.get("minimum_host_gap_seconds", 20))
        elapsed = time.time() - last
        if 0 <= elapsed < wait:
            state["_queue"].append(item)
            return {"status": "publisher_pacing_wait", "host": host,
                    "retry_after_seconds": round(wait - elapsed, 2)}
    if url in state["_seen"]:
        return {"status": "already_seen", "url": url}
    state["_seen"].add(url)
    if (not state.get("until_stopped") and
            state["pages_attempted"] >= state["max_pages_this_run"]):
        return {"status": "run_cap_reached"}
    quota = budget_status(store, settings)
    if not state.get("no_fixed_quotas") and quota["remaining"] <= 0:
        # Legacy bounded mode: defer until the shared UTC counter resets.
        state["_seen"].remove(url)
        state["_queue"].insert(0, item)
        return {"status": "daily_budget_exhausted"}
    key = "web_requests_" + quota["day_utc"]
    store.put(key, quota["used"] + 1)
    state["_host_last_at"][host] = time.time()
    state["pages_attempted"] += 1
    state["pages_attempted_today"] += 1
    state["_host_day_used"][host] = state["_host_day_used"].get(host, 0) + 1
    _publish(store, state, "reading_article", url=url,
             event="fetch_started")
    try:
        # fetch_page already enforces TLS, DNS public-only, robots, MIME,
        # response size and refusal to follow redirects.
        html = fetcher(url) if fetcher else fetch_page(url, settings)
        content, title, links, noai = extract(html)
        raw_bytes = len(html) if isinstance(html, bytes) else len(
            html.encode("utf-8"))
        utf8_tokens = len(content.encode("utf-8"))
        state["pages_read"] += 1
        state["html_bytes_read"] += raw_bytes
        state["text_characters_extracted"] += len(content)
        state["text_words_extracted"] += len(content.split())
        # The current Arabic focus model is byte-level; each UTF-8 byte is
        # exactly one potential token ID (0..255), NOT a JSM/BPE token.
        state["text_utf8_byte_tokens_extracted"] += utf8_tokens
        _publish(store, state, "checking_language_and_relevance", url=url,
                 event="article_read", detail=(
                     str(utf8_tokens) + " UTF-8 byte tokens extracted"))
        quality = topic_relevance(content, title, profile)
        if not quality["language_ok"] or not quality["topical"] or len(content) < 400:
            state["skipped_off_topic_or_non_arabic"] += 1
            reason = (
                "non_arabic" if not quality["language_ok"] else
                "off_topic" if not quality["topical"] else
                "short_extracted_article"
            )
            # An Arabic hub may link to genuine historical articles even if
            # the hub itself is too short or too broad to archive as evidence.
            # Link hints are never trained on, and every target is separately
            # screened by fetch_page and the article admission policy.
            related_links = 0
            if quality["language_ok"] and depth < 3:
                for target in relevant_links(html, url, profile):
                    if _enqueue(state, target, depth + 1):
                        related_links += 1
                        state["links_queued_this_run"] += 1
                        state["cross_host_links_queued_this_run"] += int(
                            urlsplit(target).hostname != host)
            _publish(store, state, "skipped_irrelevant_page", url=url,
                     event="page_skipped", detail=reason + " · " +
                     str(related_links) + " related links queued")
            return {
                "status": "off_topic_or_non_arabic", "url": url,
                "reason": reason,
                "language_ratio_ar": quality["language_ratio_ar"],
                "extracted_characters": len(content),
                "keyword_hits": quality["keyword_hits"],
                "title_keyword_hits": quality["title_keyword_hits"],
                "related_links_queued": related_links,
            }
        # A re-fetched source might have an earlier human license review;
        # don't silently replace that review with UNKNOWN.
        existing = store.db.execute(
            "SELECT sha256,rights_status,noai FROM web_corpus_sources "
            "WHERE url=?", (url,)).fetchone()
        if existing and existing["rights_status"] != "UNKNOWN":
            # Cannot refresh a previously approved source without a
            # re-review of its new content; use only existing immutable data.
            _publish(store, state, "preserving_existing_rights_review",
                     url=url, event="page_skipped",
                     detail="approved snapshot protected against overwrite")
            return {"status": "already_rights_reviewed_no_overwrite", "url": url}
        declared, evidence = detect_declared_rights(html, url)
        outcome = archive_page(store, url, html, content, title=title,
                               noai=noai, declared_license=declared,
                               rights_url=evidence)
        if outcome["status"] != "archived":
            return {"status": "skipped_archive", "url": url}
        state["pages_archived"] += 1
        if outcome["new_unique_text"]:
            state["unique_archived_utf8_byte_tokens"] += utf8_tokens
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
        added, external = 0, 0
        if depth < 3:
            # Follow topic-bearing anchor links across NEW publisher hosts.
            # Link text/path are only discovery hints; every new article
            # must independently pass Arabic relevance and access checks.
            for target in relevant_links(html, url, profile):
                if _enqueue(state, target, depth + 1):
                    added += 1
                    external += int(
                        urlsplit(target).hostname != urlsplit(url).hostname)
        state["links_queued_this_run"] += added
        state["cross_host_links_queued_this_run"] += external
        _publish(
            store, state, "research_article_archived", url=url,
            event="article_archived", detail=(
                ("unique" if outcome["new_unique_text"] else "duplicate")
                + " · " + str(len(saved["curiosity"]))
                + " source-derived questions"))
        return {
            "status": "archived_research_only", "url": url,
            "new_unique_text": outcome["new_unique_text"],
            "rights": outcome["rights_status"],
            "source_questions": len(saved["curiosity"]),
            "new_links_queued": added, "new_publisher_links": external,
        }
    except Exception as exc:
        state["errors"] += 1
        state["last_errors"] = (state["last_errors"] + [{
            "url": url, "error": str(exc)[:250]}])[-10:]
        _publish(store, state, "source_failed", url=url,
                 event="source_error", detail=type(exc).__name__)
        return {"status": "error", "url": url,
                "error": type(exc).__name__ + ": " + str(exc)[:220]}


def cycle(store, settings, state, pages=3, train_steps=40,
          fetcher=None, sleep_fn=time.sleep, search_fn=None):
    if not 1 <= int(pages) <= 8:
        raise ValueError("pages per cycle must be 1..8")
    profile = get_profile(state["profile"])
    observations = store.get(_HISTORY + state["profile"], []) or []
    _publish(store, state, "searching_research_sources",
             event="search_check", detail="Starting source discovery cycle")
    if stop_requested(store, state["run_id"], state["profile"]):
        search = {"status": "skipped_stop_requested",
                  "queries_attempted": 0, "url_candidates": []}
    elif fetcher is not None and search_fn is None:
        # Offline fixture injection must NEVER unexpectedly make real
        # browser or search-engine requests, even when Playwright is installed.
        search = {"status": "offline_test_no_search_provider",
                  "queries_attempted": 0, "url_candidates": []}
    else:
        search = run_one_search(
            store, settings, profile, observations, search_fn=search_fn,
            open_ended=bool(state.get("no_fixed_quotas")),
            provider=state.get("browser_search_provider", "duckduckgo"))
    queued_search = 0
    for found in search.get("url_candidates", []):
        if _enqueue(state, found):
            queued_search += 1
    state["search_status"] = search.get("status", "unknown")
    state["last_search_diagnostic"] = {
        key: search[key] for key in (
            "status", "provider", "error_type", "error_phase",
            "hint", "retry_after_seconds",
        ) if key in search and search[key] is not None
    }
    state["search_queries_this_run"] += search.get("queries_attempted", 0)
    state["search_results_queued_this_run"] += queued_search
    if queued_search:
        _publish(store, state, "source_links_discovered",
                 event="source_links_queued",
                 detail=str(queued_search) + " search-discovered candidate URLs")
    _publish(store, state, "search_finished", event="search_result",
             detail=(state["search_status"] + " · " + str(queued_search)
                     + " new source URLs" +
                     (" · " + str(search["error_type"])
                      if search.get("error_type") else "")))
    outcomes = []
    for i in range(int(pages)):
        if stop_requested(store, state["run_id"], state["profile"]):
            outcomes.append({"status": "stop_requested"})
            break
        if (not state.get("until_stopped") and
                state["pages_attempted"] >= state["max_pages_this_run"]):
            break
        result = _run_page(store, settings, state, profile, fetcher=fetcher)
        if result["status"] in (
                "queue_empty", "daily_budget_exhausted",
                "daily_page_cap_reached"):
            outcomes.append(result)
            break
        outcomes.append(result)
        if result["status"] in (
                "publisher_pacing_wait", "publisher_daily_quota",
                "queue_empty", "already_seen", "daily_budget_exhausted",
                "daily_page_cap_reached", "skipped_archive",
                "run_cap_reached"):
            _publish(
                store, state, "source_request_deferred",
                event=result["status"], url=result.get("url"),
                detail=(str(result.get("host") or "") + " " +
                        str(result.get("retry_after_seconds") or "")).strip(),
            )
        if (fetcher is None and i < int(pages) - 1 and
                float(settings.get("crawl_delay_seconds", 2)) > 0):
            sleep_fn(max(2.0, float(settings.get("crawl_delay_seconds", 2))))
    observations = store.get(_HISTORY + state["profile"], []) or []
    state["research_review_leads"] = review_leads(observations)[:20]
    base_steps = int(state["training_steps_this_run"])
    base_tokens = int(state["training_byte_tokens_processed"])
    state["current_training_steps_done"] = 0
    state["current_training_steps_target"] = 0
    state["current_training_loss"] = None

    def neural_progress(step, target, loss):
        # Exactly BATCH=8 windows * CONTEXT=48 UTF-8 byte token IDs
        # per optimizer update in the existing byte-level GRU.
        from .focus_training import BATCH, CONTEXT
        state["current_training_steps_done"] = step
        state["current_training_steps_target"] = target
        state["current_training_loss"] = round(float(loss), 6)
        state["training_steps_this_run"] = base_steps + step
        state["training_byte_tokens_processed"] = (
            base_tokens + step * BATCH * CONTEXT)
        _publish(
            store, state, "training_neural_weights",
            event="neural_optimizer_step",
            detail=(str(step) + "/" + str(target) +
                    " · actual optimizer update · loss " +
                    str(round(float(loss), 6))),
        )

    if stop_requested(store, state["run_id"], state["profile"]):
        training = {"status": "skipped_stop_requested", "steps": 0,
                    "weights_updated": False}
    else:
        _publish(store, state, "checking_training_eligibility",
                 event="training_eligibility_check")
        training = train_if_eligible(
            store, state["profile"], steps=train_steps,
            progress=neural_progress)
    state["training_status"] = training["status"]
    state["latest_training"] = {
        key: training.get(key) for key in (
            "status", "reason", "steps", "accepted", "checkpoint",
            "sha256", "validation_before", "validation_after",
            "trainable_parameters", "unique_approved_training_bytes",
        ) if key in training}
    if training.get("status") == "trained":
        state["training_runs_with_new_weights"] += 1
        state["training_steps_this_run"] = (
            base_steps + int(training["steps"]))
        state["training_byte_tokens_processed"] = (
            base_tokens + int(training.get("training_window_bytes_seen", 0)))
    _publish(store, state, "training_result", event="training_result",
             detail=(training["status"] + " · " +
                     str(training.get("reason") or "")[:200]))
    _publish(store, state, "cycle_finished", event="cycle_finished",
             detail=("research · training " + training["status"]))
    state["cycles_completed"] += 1
    state["updated_at_utc"] = _utc()
    state["pending_research_urls"] = len(state["_queue"])
    observations = store.get(_HISTORY + state["profile"], []) or []
    state["distinct_research_hosts"] = len({
        item.get("host") for item in observations if item.get("host")})
    store.put("focus_frontier_" + state["profile"], {
        "day": state["pages_day_utc"],
        "seen": sorted(state["_seen"])[-6000:],
        "pending": state["_queue"][:500],
        "host_counts": state["_host_day_used"],
        "pages_attempted_today": state["pages_attempted_today"],
        "host_last_at": state["_host_last_at"],
    })
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
        "search": {
            "status": state["search_status"],
            "queries_attempted": search.get("queries_attempted", 0),
            "urls_queued": queued_search,
            "error_type": search.get("error_type"),
            "error_phase": search.get("error_phase"),
            "retry_after_seconds": search.get("retry_after_seconds"),
            "hint": search.get("hint"),
        },
        "pending_source_urls": len(state["_queue"]),
        "discovered_publisher_hosts_total":
            state["distinct_research_hosts"],
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
        "discovery": {
            "cross_publisher_topic_links_enabled": True,
            "no_fixed_website_allowlist": True,
            "browser_search_requires_api_key": False,
            "search_provider": (run or {}).get(
                "browser_search_provider", "duckduckgo"),
            "no_fixed_quotas": bool(
                (run or {}).get("no_fixed_quotas", False)),
            "search_query_count_last_run": (run or {}).get(
                "search_queries_this_run", 0),
            "pending_links": len(
                (store.get("focus_frontier_" + profile_name) or {}).get(
                    "pending", [])),
            "research_hosts": source_hosts,
        },
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
        monotonic_fn=time.monotonic, until_stopped=False,
        search_fn=None, no_fixed_quotas=False,
        search_provider="duckduckgo"):
    """User-started local process; continuous mode ends ONLY on manual stop.

    Source queue and page caps reset at UTC midnight, but network permissions,
    robots, TLS, license review and unchanged-dataset training restrictions do
    not reset. With no new authorized work, it remains awake but idle.
    """
    if not until_stopped and not 0 < float(hours) <= 12:
        raise ValueError("hours must be within (0, 12]")
    if no_fixed_quotas and not until_stopped:
        raise ValueError("--no-fixed-quotas requires --until-stopped")
    if search_provider not in ("duckduckgo", "google", "bing"):
        raise ValueError("Unknown browser search provider")
    if not 1 <= int(interval_minutes) <= 60:
        raise ValueError("interval-minutes must be 1..60")
    if not 1 <= int(pages_per_cycle) <= 8:
        raise ValueError("pages-per-cycle must be 1..8")
    lock_handle = acquire_lock(store, profile_name)
    try:
        state = init_state(store, profile_name, extra_urls, max_pages=max_pages)
        state["until_stopped"] = bool(until_stopped)
        state["no_fixed_quotas"] = bool(no_fixed_quotas)
        state["browser_search_provider"] = search_provider
        state["worker_pid"] = os.getpid()
        duration = float(hours) * 3600 if not until_stopped else None
        start = monotonic_fn()
        end = start + duration if duration is not None else None
        stop = False
        previous_signals = {}

        def on_stop(_sig, _frame):
            nonlocal stop
            stop = True

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, on_stop)
            except ValueError:
                continue

        def operator_stop():
            return bool(
                stop or stop_requested(store, state["run_id"], profile_name))

        state["status"] = "running"
        _publish(store, state, "starting", event="worker_started",
                 detail="Learning process started independently of Studio")
        print(json.dumps({
            "status": "running", "profile": profile_name,
            "run_id": state["run_id"],
            "stops_only_when_operator_requests": bool(until_stopped),
            "ends_after_max_hours": None if until_stopped else float(hours),
            "max_pages_per_utc_day": None if no_fixed_quotas else max_pages,
            "no_fixed_quotas": bool(no_fixed_quotas),
            "search_provider": search_provider,
            "note": (
                "Continuous discovery runs until manual stop. There are no "
                "arbitrary daily page/search counts in open-ended mode; "
                "per-host pacing, robots, access and model rights still apply."
                if no_fixed_quotas else
                "Daily quotas still apply. Mac must stay awake."
            ),
        }, ensure_ascii=False), flush=True)
        try:
            while not operator_stop() and (
                    end is None or monotonic_fn() < end):
                _rollover_if_new_utc_day(state)
                result = cycle(store, settings, state, pages=pages_per_cycle,
                               train_steps=train_steps, fetcher=fetcher,
                               sleep_fn=sleep_fn, search_fn=search_fn)
                state["hours_elapsed"] = round(
                    (monotonic_fn() - start) / 3600, 3)
                state["heartbeat_at_utc"] = _utc()
                if result["outcomes"] and result["outcomes"][0].get(
                        "status") in (
                            "queue_empty", "daily_budget_exhausted",
                            "daily_page_cap_reached", "publisher_pacing_wait",
                            "publisher_daily_quota"):
                    state["status"] = "idle_waiting_for_allowed_work"
                else:
                    state["status"] = "running"
                if state["status"] == "idle_waiting_for_allowed_work":
                    _publish(
                        store, state, "idle_waiting_for_allowed_work",
                        event="worker_idle",
                        detail="No allowed work this cycle; next interval pending",
                    )
                else:
                    _publish(
                        store, state, "waiting_for_next_cycle",
                        event="worker_waiting_next_cycle",
                        detail=str(interval_minutes) + "-minute cadence",
                    )
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if not until_stopped and (
                        state["pages_attempted"] >= state["max_pages_this_run"]):
                    break
                if operator_stop():
                    break
                remaining = (
                    (end - monotonic_fn()) if end is not None
                    else float(interval_minutes * 60))
                if remaining <= 0:
                    break
                cadence = min(remaining, int(interval_minutes) * 60)
                while cadence > 0 and not operator_stop():
                    chunk = min(cadence, 5)
                    sleep_fn(chunk)
                    cadence -= chunk
                    # Keep the UI heartbeat fresh while idle, without
                    # doing any additional network fetches or training.
                    state["heartbeat_at_utc"] = _utc()
                    state["updated_at_utc"] = _utc()
                    store.put(_KEY + profile_name, _public_state(state))
        finally:
            requested = operator_stop()
            state["finished_at_utc"] = _utc()
            state["status"] = "stopped" if requested else "completed"
            state["stop_requested"] = requested
            state["hours_elapsed"] = round(
                max(0.0, monotonic_fn() - start) / 3600, 3)
            state["updated_at_utc"] = _utc()
            state["heartbeat_at_utc"] = _utc()
            _publish(store, state, state["status"],
                     event="worker_" + state["status"])
            for sig, prior in previous_signals.items():
                signal.signal(sig, prior)
            print(json.dumps({
                "status": state["status"], "profile": profile_name,
                "pages_archived": state["pages_archived"],
                "unique_pages": state["new_unique_pages"],
                "candidate_questions": state["candidate_questions_created_this_run"],
                "training_steps": state["training_steps_this_run"],
            }, ensure_ascii=False), flush=True)
        return _public_state(state)
    finally:
        release_lock(lock_handle)
