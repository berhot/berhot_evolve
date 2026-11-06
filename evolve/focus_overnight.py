"""Bounded overnight research supervisor for opt-in Arabic Saudi history.

This is an *on-device process* the user starts; ChatGPT does not operate the
user's Mac in the background. It never searches arbitrarily all domains,
bypasses robots, grants licenses, or confuses fetched pages with model updates.
"""
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlsplit, unquote
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from .focus_control import acquire_lock, release_lock, stop_requested

from .focus_data import get_profile, topic_relevance, arabic_curiosity, review_leads
from .focus_discovery import (_candidate_link, relevant_links, run_one_search,
                              publisher_key, arabic_candidate_priority)
from .focus_training import train_if_eligible, readiness as training_readiness
from .focus_sitemaps import discover_publisher_sitemaps
from .open_corpus import archive_page
from .plain_corpus import (begin_extraction_run, capture_extraction,
                           finish_collection_cycle, prepare_focus_dataset)
from .v119_curiosity import budget_status
from .web_discovery import (
    canonical_url, extract, fetch_page, detect_declared_rights,
    WebSourceError, SourceRedirect,
)

_KEY = "focus_run_"
_HISTORY = "focus_observations_"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _source_error_policy(exc):
    """Safe, actionable failure classification; never weaken fetch_page gates.

    Only transient connection/5xx/robots-unverifiable conditions are eligible
    for bounded, delayed retry. HTTP 4xx denial, robots disallow, redirects,
    and invalid or non-HTML pages remain terminal for this URL.
    """
    if isinstance(exc, WebSourceError):
        reason = str(exc)
        if reason.startswith("HTTP "):
            status = reason[5:].strip().split()[0]
            if status.isdigit():
                code = int(status)
                if code == 429:
                    return ("http_429_rate_limited", True, 900)
                if code in (500, 502, 503, 504):
                    return ("http_" + status + "_server_error", True, 300)
                if code in (301, 302, 303, 307, 308):
                    return ("redirect_not_followed", False, 0)
                if code in (401, 403):
                    return ("http_" + status + "_access_denied", False, 0)
                if code in (404, 410):
                    return ("http_" + status + "_not_found", False, 0)
                return ("http_" + status, False, 0)
        if reason.startswith("Redirect refused"):
            # NoRedirect explicitly raises this for HTTP 3xx. The old
            # fallback wrongly labelled valid canonical-URL redirects as
            # source_policy_or_format_rejected.
            return ("redirect_not_followed", False, 0)
        if reason.startswith("Website disallows automated access"):
            return ("robots_disallowed", False, 0)
        if reason.startswith("Oversized robots.txt"):
            return ("robots_oversized", False, 0)
        if (reason.startswith("Cannot verify robots.txt") or
                reason.startswith("Robots availability uncertain")):
            return ("robots_unverifiable", True, 900)
        if reason.startswith("Unable to resolve source host"):
            return ("dns_resolution_failed", True, 600)
        if reason.startswith("Only HTML pages"):
            return ("non_html_source", False, 0)
        if reason.startswith("Page too large"):
            return ("page_oversized", False, 0)
        if reason.startswith("Resolved host points to a non-public"):
            return ("non_public_address_blocked", False, 0)
        return ("source_policy_or_format_rejected", False, 0)
    if isinstance(exc, (TimeoutError, OSError)):
        return ("network_error_or_timeout", True, 300)
    return ("unexpected_source_exception", False, 0)


def _public_state(state):
    return {key: value for key, value in state.items()
            if key not in ("_queue", "_seen", "_seeded", "_last_errors",
                           "_host_day_used", "_host_last_at",
                           "_host_cooldown_until", "_publisher_attempts")}


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
    if "_queue" in state:
        state["pending_research_urls"] = len(state["_queue"])
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
    # Publisher redirects and sitemap links mix Unicode and escaped paths.
    # Treat those representations as one resource rather than refetching both.
    identity = _url_identity(url)
    if any(_url_identity(known) == identity for known in state["_seen"]) or any(
            _url_identity(item["url"]) == identity for item in state["_queue"]):
        return False
    state["_queue"].append({"url": url, "depth": int(depth)})
    return True


def _url_identity(url):
    parts = urlsplit(url)
    try:
        path = unquote(parts.path, errors="strict")
    except UnicodeError:
        path = parts.path
    return (parts.hostname, path, parts.query)


def retry_saved_redirects(store, profile_name="saudi-history-ar"):
    """Explicit recovery after upgrading redirect support; keep failures logged."""
    from .focus_control import worker_running
    if worker_running(store, profile_name):
        raise ValueError("Stop the focus worker before repairing its saved frontier")
    run = store.get(_KEY + profile_name) or {}
    frontier = store.get("focus_frontier_" + profile_name) or {}
    seen = set(frontier.get("seen", []))
    pending = list(frontier.get("pending", []))
    known = {_url_identity(item["url"]) for item in pending}
    rows = store.db.execute(
        "SELECT DISTINCT url FROM focus_activity_events WHERE profile=? "
        "AND run_id=? AND kind='source_error' AND detail LIKE 'redirect_not_followed%' "
        "ORDER BY id", (profile_name, run.get("run_id", ""))).fetchall()
    added = 0
    for row in rows:
        url = _candidate_link(row["url"])
        if (not url or url not in seen or _url_identity(url) in known
                or len(pending) >= 500):
            continue
        pending.append({"url": url, "depth": 0})
        known.add(_url_identity(url))
        seen.discard(url)
        added += 1
    if added:
        frontier.update(seen=sorted(seen), pending=pending)
        store.put("focus_frontier_" + profile_name, frontier)
        run["pending_research_urls"] = len(pending)
        store.put(_KEY + profile_name, run)
        store.append_focus_activity(
            profile_name, run["run_id"], _utc(), "redirect_recovery_queued",
            phase="worker_stopped", detail=str(added) +
            " previously refused redirect URLs queued for paced access checks")
    return {"status": "queued" if added else "no_redirects_to_recover",
            "redirect_sources_requeued": added, "pending_urls": len(pending),
            "worker_started": False}


def recover_cached_source_links(store, state, profile_name, scan_limit=6):
    """Discover additional candidate links in previously archived HTML.

    This makes no network requests and never upgrades source rights. Each
    archived URL and immutable raw HTML digest is scanned once per profile;
    later changed snapshots may be scanned again. The existing crawler still
    enforces per-target TLS, public DNS, robots, MIME and size checks.
    """
    if not 1 <= int(scan_limit) <= 20:
        raise ValueError("scan_limit must be 1..20")
    # Revisit immutable snapshots once after the broader citation extractor
    # upgrade; earlier scans missed opaque external reference URLs.
    key = "focus_archive_link_scans_v2_" + profile_name
    reviewed = store.get(key, {}) or {}
    observations = store.get(_HISTORY + profile_name, []) or []
    root = (store.root / "web_corpus" / "raw").resolve()
    examined = 0
    queued = 0
    cross_host = 0
    attempts = 0
    # Keep bounded even if historical observations contain many records.
    for record in observations[-500:]:
        if attempts >= int(scan_limit) or len(state["_queue"]) >= 500:
            break
        if not isinstance(record, dict) or not isinstance(record.get("url"), str):
            continue
        url = record["url"]
        row = store.db.execute(
            "SELECT raw_path,raw_html_sha256 FROM web_corpus_sources WHERE url=?",
            (url,),
        ).fetchone()
        if row is None:
            continue
        digest = row["raw_html_sha256"]
        if reviewed.get(url) == digest:
            continue
        attempts += 1
        raw = Path(row["raw_path"]).resolve()
        if not raw.is_relative_to(root) or not raw.is_file():
            # Mark this specific immutable snapshot as checked; an updated
            # source SHA may be examined in a later run.
            reviewed[url] = digest
            continue
        try:
            with gzip.open(raw, "rb") as stream:
                data = stream.read(2_000_001)
            if len(data) > 2_000_000:
                reviewed[url] = digest
                continue
            html = data.decode("utf-8", errors="replace")
            targets = relevant_links(html, url, get_profile(profile_name),
                                     max_results=30)
        except (OSError, EOFError, UnicodeError, ValueError):
            reviewed[url] = digest
            continue
        examined += 1
        reviewed[url] = digest
        for target in targets:
            if _enqueue(state, target, depth=1):
                queued += 1
                cross_host += int(
                    urlsplit(target).hostname != urlsplit(url).hostname)
    store.put(key, reviewed)
    state["links_queued_this_run"] += queued
    state["cross_host_links_queued_this_run"] += cross_host
    return {
        "status": "cached_sources_scanned" if attempts else "cached_sources_exhausted",
        "source_snapshots_checked": attempts,
        "readable_cached_sources": examined,
        "new_candidate_urls": queued,
        "cross_publisher_candidates": cross_host,
        "all_sources_scanned_for_current_snapshot": attempts == 0,
        "network_requests": 0,
        "candidate_urls_are_not_training_rights": True,
    }


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
        "discovery": "publisher_balanced_links_citations_sitemaps_and_optional_web_search",
        "search_provider": "auto (official Brave API when configured, otherwise policy-checked DuckDuckGo)",
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
        "browser_search_provider": "auto",
        "minimum_host_gap_seconds": 20,
        "_host_last_at": {},
        "_host_cooldown_until": {},
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
        "last_cycle_pages_attempted": 0,
        "last_cycle_page_target": 0,
        "last_cycle_pacing_waits": 0,
        "last_cycle_idle_reason": None,
        "last_archive_recovery": None,
        "last_sitemap_discovery": None,
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
    state["_host_cooldown_until"] = dict(
        saved_frontier.get("host_cooldown_until", {}))
    for url in base["seed_urls"]:
        _enqueue(state, url)
    # Preserve historical representation across restarts. This is a queue
    # priority, independent of daily quotas and strict per-origin pacing.
    state["_publisher_attempts"] = {}
    for record in store.get(_HISTORY + profile_name, []) or []:
        key = publisher_key(record.get("url", ""))
        if key:
            state["_publisher_attempts"][key] = (
                state["_publisher_attempts"].get(key, 0) + 1)
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
              sleep_fn=time.sleep, time_fn=time.time):
    _rollover_if_new_utc_day(state)
    if (state.get("until_stopped") and
            not state.get("no_fixed_quotas") and
            state["pages_attempted_today"] >= state["max_pages_per_utc_day"]):
        return {"status": "daily_page_cap_reached"}
    if not state["_queue"]:
        return {"status": "queue_empty"}
    if state.get("no_fixed_quotas"):
        # Examine the pending frontier ONCE. A temporarily paced publisher
        # must not waste a page slot while another host has eligible work.
        # Rotate paced entries to the tail without discarding or fetching them.
        earliest = None
        item = None
        now = time_fn()
        autonomous=(store.get('focus_execution_policy') or {}).get('autonomous_pipeline',False)
        exploration_turn=state['pages_attempted']%5==4
        def scheduling_rank(candidate):
            fairness=state.get('_publisher_attempts',{}).get(publisher_key(candidate['url']),0)
            if not autonomous:return (fairness,)
            language=arabic_candidate_priority(candidate['url'])
            # Most requests favor likely Arabic articles; reserve every fifth
            # attempt for unfamiliar publishers so discovery remains open.
            return (fairness,language) if exploration_turn else (language,fairness)
        state['_queue'].sort(key=scheduling_rank)
        for _ in range(len(state["_queue"])):
            candidate = state["_queue"].pop(0)
            candidate_host = urlsplit(candidate["url"]).hostname
            last = float(state["_host_last_at"].get(candidate_host, 0) or 0)
            gap = max(0, int(state.get("minimum_host_gap_seconds", 20)))
            elapsed = now - last
            host_wait = max(0.0, gap - elapsed) if 0 <= elapsed < gap else 0.0
            retry_wait = max(
                0.0, float(candidate.get("retry_at", 0) or 0) - now)
            # HTTP 429 applies to the entire publisher, not just one URL.
            # Defer ALL URLs from that host until its cooldown expires.
            cooldown = max(0.0, float(
                state.get("_host_cooldown_until", {}).get(candidate_host, 0)
                or 0) - now)
            retry_wait = max(retry_wait, cooldown)
            remaining = max(host_wait, retry_wait)
            if remaining > 0:
                state["_queue"].append(candidate)
                status = ("source_retry_wait" if retry_wait >= host_wait
                          else "publisher_pacing_wait")
                if earliest is None or remaining < earliest[0]:
                    earliest = (remaining, candidate_host, status)
                continue
            item = candidate
            break
        if item is None:
            return {
                "status": earliest[2],
                "host": earliest[1],
                "retry_after_seconds": round(earliest[0], 2),
            }
    else:
        item = state["_queue"].pop(0)
    url, depth = item["url"], item["depth"]
    host = urlsplit(url).hostname
    # Finite legacy runs retain the publisher daily quota. Open-ended runs
    # instead respect a minimum delay between requests to each host.
    if not state.get("no_fixed_quotas") and (
            state["_host_day_used"].get(host, 0) >= 4):
        state["_queue"].append(item)
        return {"status": "publisher_daily_quota", "host": host}
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
    state["_host_last_at"][host] = time_fn()
    state["pages_attempted"] += 1
    state["pages_attempted_today"] += 1
    state["_host_day_used"][host] = state["_host_day_used"].get(host, 0) + 1
    publisher = publisher_key(url)
    attempts = state.setdefault("_publisher_attempts", {})
    attempts[publisher] = attempts.get(publisher, 0) + 1
    _publish(store, state, "reading_article", url=url,
             event="fetch_started")
    try:
        # fetch_page already enforces TLS, DNS public-only, robots, MIME,
        # response size and refusal to follow redirects.
        html = fetcher(url) if fetcher else fetch_page(url, settings)
        content, title, links, noai, raw_text = extract(html, include_raw=True)
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
        enhanced=(store.get('focus_execution_policy') or {}).get('article_body_training',False)
        quality_text=content
        capture_options={}
        if enhanced:
            from .owned_corpus import article_body
            from .open_corpus import normalize_web_text
            quality_text=article_body(html)
            capture_options={'clean_text':quality_text,'approval_parent_sha256':
                hashlib.sha256(normalize_web_text(content).encode('utf-8')).hexdigest()}
        quality = topic_relevance(quality_text, title, profile)
        if state.get("current_extraction_run_id"):
            begin_extraction_run(store, state["current_extraction_run_id"], state["profile"])
            capture_extraction(store, state["current_extraction_run_id"], state["pages_attempted"],
                url, title, html, raw_text, noai=noai,
                profile_match=quality["language_ok"] and quality["topical"] and len(quality_text) >= 400,
                **capture_options)
        if not quality["language_ok"] or not quality["topical"] or len(quality_text) < 400:
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
        # Persist focus membership independently of the bounded UI
        # observation snapshot. This survives restarts and allows Studio to
        # show meaningful all-run totals even when this-run counters reset.
        store.record_focus_archive(
            state["profile"], url, outcome["clean_sha256"],
            source_questions=len(saved["curiosity"]))
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
        if isinstance(exc, SourceRedirect):
            chain = list(item.get("redirect_chain", [])) + [url]
            target = _candidate_link(exc.target_url)
            if (target and len(chain) <= 5 and
                    _url_identity(target) not in {_url_identity(u) for u in chain}):
                queued = _enqueue(state, target, depth)
                if queued:
                    state["_queue"][-1]["redirect_chain"] = chain
                _publish(store, state, "redirect_destination_scheduled", url=url,
                         event="source_redirect", detail=(
                             str(exc.status_code) + " · " +
                             ("destination queued" if queued else "destination already known") +
                             " · " + target))
                return {"status": "redirect_queued" if queued else "redirect_already_known",
                        "url": url, "redirect_url": target}
            exc = WebSourceError("Redirect refused; loop, hop limit or ineligible destination")
        state["errors"] += 1
        reason, transient, delay = _source_error_policy(exc)
        if reason == "http_429_rate_limited":
            state.setdefault("_host_cooldown_until", {})[host] = (
                time_fn() + delay)
        count = max(0, int(item.get("retry_count", 0) or 0))
        retry_queued = False
        retry_delay = 0
        if (transient and state.get("no_fixed_quotas") and count < 2
                and len(state["_queue"]) < 500):
            retry_delay = min(3600, delay * (2 ** count))
            retry_item = dict(item, retry_count=count + 1,
                              retry_at=time_fn() + retry_delay)
            state["_seen"].discard(url)
            state["_queue"].append(retry_item)
            retry_queued = True
        state["last_errors"] = (state["last_errors"] + [{
            "url": url, "error": reason, "error_type": type(exc).__name__,
            "message": str(exc)[:300],
            "retry_scheduled": retry_queued,
            "retry_after_seconds": retry_delay if retry_queued else None,
        }])[-10:]
        detail = reason
        if reason == "unexpected_source_exception":
            detail += " · " + type(exc).__name__
        if retry_queued:
            detail += " · delayed retry in " + str(retry_delay) + "s"
        else:
            detail += " · not automatically retried"
        _publish(store, state, "source_failed", url=url,
                 event="source_error", detail=detail)
        return {
            "status": "error", "url": url,
            "error": reason, "error_type": type(exc).__name__,
            "retry_scheduled": retry_queued,
            "retry_after_seconds": retry_delay if retry_queued else None,
        }


def cycle(store, settings, state, pages=3, train_steps=40,
          fetcher=None, sleep_fn=time.sleep, search_fn=None,
          time_fn=time.time, sitemap_fn=None):
    if not 1 <= int(pages) <= 50:
        raise ValueError("pages per cycle must be 1..50")
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
            "status", "provider", "query", "error_type", "error_phase",
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
                     (' · query: '+str(search['query']) if search.get('query') else '') +
                     (" · " + str(search["error_type"])
                      if search.get("error_type") else "")))
    recovery = None
    if state.get("no_fixed_quotas") and not stop_requested(
            store, state["run_id"], state["profile"]):
        # Search engines can lawfully disallow automated results. Re-examine
        # previously archived HTML *locally* for relevant candidate links.
        # This is not web search, fact verification or model training.
        recovery = recover_cached_source_links(
            store, state, state["profile"], scan_limit=6)
        state["last_archive_recovery"] = recovery
        _publish(store, state, "archived_source_link_discovery",
                 event="archive_link_recovery",
                 detail=(str(recovery["source_snapshots_checked"]) +
                         " cached sources scanned · " +
                         str(recovery["new_candidate_urls"]) +
                         " new candidate URLs · 0 network requests"))
    # An existing queue can be dominated by one publisher or held on retry.
    # Neither should prevent discovery of other independently checked sites.
    now = time_fn()
    blocked_hosts = {
        host for host, until in state.get("_host_cooldown_until", {}).items()
        if float(until or 0) > now}
    blocked_hosts.update(
        host for host, last in state.get("_host_last_at", {}).items()
        if 0 <= now - float(last or 0) < state.get("minimum_host_gap_seconds", 20))
    ready_publishers = {
        publisher_key(item["url"]) for item in state["_queue"]
        if float(item.get("retry_at", 0) or 0) <= now
        and urlsplit(item["url"]).hostname not in blocked_hosts}
    if (state.get("no_fixed_quotas") and len(ready_publishers) < 3 and
            (fetcher is None or sitemap_fn is not None) and
            not stop_requested(store, state["run_id"], state["profile"])):
        # If both browser search and local archive links yield nothing, try
        # one due publisher's own public sitemap. The sitemap reader checks
        # public DNS, robots, same-origin HTTPS, TLS and redirects. This is
        # link discovery only; actual page fetches still use fetch_page.
        sitemap = discover_publisher_sitemaps(
            store, profile, observations, sitemap_fn=sitemap_fn,
            prefer_new_publishers=True,
            candidate_urls=[item["url"] for item in state["_queue"]],
            excluded_hosts=blocked_hosts,
            already_seen=state["_seen"] | {
                item["url"] for item in state["_queue"]})
        queued_sitemap = 0
        for candidate in sitemap.get("url_candidates", []):
            queued_sitemap += int(_enqueue(state, candidate))
        state["last_sitemap_discovery"] = {
            key: sitemap.get(key) for key in (
                "status", "host", "network_requests",
                "sitemap_documents_checked", "candidate_urls_found",
                "sitemap_indexes_parsed", "sitemap_urlsets_parsed",
                "sitemap_locations_listed", "topical_urls_after_filter",
                "sitemap_failures", "sitemap_more_documents_unchecked",
                "backlog_remaining", "retry_after_seconds",
            )
        }
        state["last_sitemap_discovery"]["new_urls_queued"] = queued_sitemap
        # A sitemap fetch is a real publisher request. Prevent an article
        # fetch against that same host until the 20-second gap has elapsed.
        sitemap_host = sitemap.get("host")
        if sitemap_host and sitemap.get("network_requests", 0):
            state["_host_last_at"][sitemap_host] = time_fn()
        state["links_queued_this_run"] += queued_sitemap
        _publish(
            store, state, "publisher_sitemap_discovery",
            event="publisher_sitemap_result",
            detail=(str(sitemap.get("host") or "all publishers paced") +
                    " · " + str(sitemap.get("status")) + " · " +
                    str(queued_sitemap) + " new candidate URLs · " +
                    str(sitemap.get("network_requests", 0)) +
                    " discovery requests · " +
                    str(sitemap.get("sitemap_indexes_parsed", 0)) +
                    " XML indexes · " +
                    str(sitemap.get("sitemap_urlsets_parsed", 0)) +
                    " URL sets · " +
                    str(sitemap.get("sitemap_locations_listed", 0)) +
                    " listed URLs" +
                    (" · " + ",".join(sitemap.get("sitemap_failures", []))
                     if sitemap.get("sitemap_failures") else "")))
    outcomes = []
    pages_at_cycle_start = int(state["pages_attempted"])
    extraction_id = state["run_id"] + "-batch-" + str(state["cycles_completed"] + 1).zfill(6)
    state["current_extraction_run_id"] = extraction_id
    pacing_waits = 0
    checks = 0
    # Guard against an unexpectedly malformed/duplicate frontier. In
    # open-ended mode ONLY actual network fetch attempts use page slots.
    # Existing bounded mode still performs at most 'pages' selections.
    max_checks = (max(100, len(state["_queue"]) * 2, int(pages) * 20)
                  if state.get("no_fixed_quotas") else int(pages))
    while checks < max_checks:
        if (state.get("no_fixed_quotas") and
                state["pages_attempted"] - pages_at_cycle_start >= int(pages)):
            break
        checks += 1
        if stop_requested(store, state["run_id"], state["profile"]):
            outcomes.append({"status": "stop_requested"})
            break
        if (not state.get("until_stopped") and
                state["pages_attempted"] >= state["max_pages_this_run"]):
            break
        result = _run_page(
            store, settings, state, profile,
            fetcher=fetcher, sleep_fn=sleep_fn, time_fn=time_fn)
        status = result["status"]
        if status in (
                "queue_empty", "daily_budget_exhausted",
                "daily_page_cap_reached", "run_cap_reached"):
            outcomes.append(result)
            break
        if status == "source_retry_wait":
            _publish(
                store, state, "waiting_for_source_retry",
                event="source_retry_wait", detail=(
                    str(result.get("host") or "") + " · " +
                    str(result.get("retry_after_seconds", 0)) +
                    "s until an eligible temporary-failure retry"),
            )
            # Long connection backoffs should yield to the normal cycle
            # cadence, rather than holding the worker inside one cycle.
            outcomes.append(result)
            break
        if status == "publisher_pacing_wait":
            pacing_waits += 1
            _publish(
                store, state, "waiting_for_publisher_pacing",
                event="publisher_pacing_wait", detail=(
                    str(result.get("host") or "") + " · " +
                    str(result.get("retry_after_seconds", 0)) +
                    "s until next allowed request"),
            )
            if not state.get("no_fixed_quotas"):
                outcomes.append(result)
                continue
            # All queued publishers are paced. Sleep only until the earliest
            # eligible host; stop requests are checked every <=2 seconds.
            # The 15-minute cycle interval is NOT applied between pages.
            delay = max(0, float(result.get("retry_after_seconds", 0)))
            if delay < 0.01:
                outcomes.append(result)
                break
            started_wait = time_fn()
            remaining = delay + 0.1
            while remaining > 0:
                if stop_requested(store, state["run_id"], state["profile"]):
                    break
                chunk = min(2.0, remaining)
                sleep_fn(chunk)
                remaining -= chunk
            if stop_requested(store, state["run_id"], state["profile"]):
                outcomes.append({"status": "stop_requested"})
                break
            if time_fn() - started_wait < delay * 0.5:
                # A dry-run/fake sleep that does not advance clock time must
                # not leave a test or the worker in an infinite busy loop.
                outcomes.append(result)
                break
            continue
        outcomes.append(result)
        if status in ("publisher_daily_quota", "already_seen",
                      "skipped_archive"):
            _publish(
                store, state, "source_request_deferred", event=status,
                url=result.get("url"),
                detail=str(result.get("host") or ""),
            )
        # Respect the cross-request crawl delay, but do not sleep after
        # pure queue/skipped decisions that made no network request.
        if (fetcher is None and
                state["pages_attempted"] > pages_at_cycle_start and
                status not in ("publisher_daily_quota", "already_seen") and
                float(settings.get("crawl_delay_seconds", 2)) > 0 and
                (state.get("no_fixed_quotas") or checks < int(pages))):
            sleep_fn(max(2.0, float(settings.get("crawl_delay_seconds", 2))))
    completed_page_attempts = state["pages_attempted"] - pages_at_cycle_start
    if completed_page_attempts == 0:
        if outcomes and outcomes[0].get("status") == "queue_empty":
            state["last_cycle_idle_reason"] = (
                "no_pending_urls_and_search_produced_no_candidates"
                if not state["_queue"] else "no_fetchable_url")
        elif outcomes and outcomes[0].get("status") == "source_retry_wait":
            state["last_cycle_idle_reason"] = "pending_sources_in_retry_cooldown"
        elif outcomes and outcomes[0].get("status") == "publisher_pacing_wait":
            state["last_cycle_idle_reason"] = "publisher_access_pacing"
        else:
            state["last_cycle_idle_reason"] = (
                outcomes[0].get("status") if outcomes else "no_fetch_attempt")
        _publish(
            store, state, "research_idle_explained",
            event="research_idle_reason",
            detail=(state["last_cycle_idle_reason"] + " · search: " +
                    state["search_status"] +
                    " · pending: " + str(len(state["_queue"]))),
        )
    else:
        state["last_cycle_idle_reason"] = None
    state["last_cycle_pages_attempted"] = completed_page_attempts
    state["last_cycle_page_target"] = int(pages)
    state["last_cycle_pacing_waits"] = pacing_waits
    _publish(
        store, state, "research_cycle_pages_complete",
        event="cycle_research_summary",
        detail=(str(completed_page_attempts) + "/" + str(pages) +
                " actual page attempts · " + str(pacing_waits) +
                " publisher pacing waits"),
    )
    observations = store.get(_HISTORY + state["profile"], []) or []
    state["research_review_leads"] = review_leads(observations)[:20]
    _publish(store, state, "preparing_plain_corpus", event="corpus_preparation_started")
    enhanced=(store.get('focus_execution_policy') or {}).get('article_body_training',False)
    if enhanced:
        from .owned_corpus import prepare_owned_corpus
        dataset=prepare_owned_corpus(store,state['profile'])
        if (store.get('focus_execution_policy') or {}).get('supervised_qa'):
            from .supervised_corpus import prepare_supervised_corpus
            qa_dataset=prepare_supervised_corpus(store,dataset)
            state['supervised_qa_dataset']=qa_dataset
            _publish(store,state,'supervised_qa_prepared',event='supervised_qa_prepared',detail=qa_dataset['directory'])
    else:
        prepared = training_readiness(store, state["profile"])
        dataset = prepare_focus_dataset(store, state["profile"], prepared) if prepared["status"] == "ready" else None
    bundle = finish_collection_cycle(store, extraction_id, state["profile"], dataset,
        reason=state.get("last_cycle_idle_reason"), attempted=completed_page_attempts)
    summary = {k: bundle.get(k) for k in (
        "status", "directory", "extracted_pages", "unique_clean_texts", "duplicate_clean_texts",
        "training_eligible_unique_texts", "training_dataset", "reason", "audit_log")}
    state["last_cycle_extraction"] = summary
    if bundle["extracted_pages"]:
        state["last_extraction_bundle"] = summary
    elif not state.get("last_extraction_bundle"):
        state["last_extraction_bundle"] = store.get("plain_extraction_latest_populated_" + state["profile"])
    state["current_extraction_run_id"] = None
    _publish(store, state, "plain_corpus_prepared" if bundle["extracted_pages"] else "no_text_extracted",
             event="corpus_batch_prepared" if bundle["extracted_pages"] else "corpus_batch_empty",
             detail=(str(bundle["extracted_pages"]) + " extracted · " +
                     str(bundle["unique_clean_texts"]) + " unique clean texts · " +
                     (("raw.txt / clean.txt saved · "+str(bundle.get('training_eligible_unique_texts',0))+" eligible texts in corpus.txt"+
                       (' · training uses previously saved corpus' if not bundle.get('training_eligible_unique_texts') else '')) if bundle["extracted_pages"] else
                      "no article text read; no empty dataset created; " + str(bundle.get("reason")))))
    base_steps = int(state["training_steps_this_run"])
    base_tokens = int(state["training_byte_tokens_processed"])
    state["current_training_steps_done"] = 0
    state["current_training_steps_target"] = 0
    state["current_training_loss"] = None
    language_start=(store.get('owned_training_cursor_'+state['profile']) or {}).get('attempted_training_bytes',0)
    qa_start=(store.get('focus_supervised_qa_cursor') or {}).get('input_bytes',0)

    def neural_progress(step, target, loss):
        # Exactly BATCH=8 windows * CONTEXT=48 UTF-8 byte token IDs
        # per optimizer update in the existing byte-level GRU.
        from .focus_training import BATCH, CONTEXT
        state["current_training_steps_done"] = step
        state["current_training_steps_target"] = target
        state["current_training_loss"] = round(float(loss), 6)
        state['current_training_task']=(store.get('focus_training_last_step') or {}).get('task','language')
        state["training_steps_this_run"] = base_steps + step
        state["training_byte_tokens_processed"] = (
            base_tokens + max(0,(store.get('owned_training_cursor_'+state['profile']) or {}).get('attempted_training_bytes',0)-language_start)
            +max(0,(store.get('focus_supervised_qa_cursor') or {}).get('input_bytes',0)-qa_start)
            if enhanced else base_tokens + step * BATCH * CONTEXT)
        _publish(
            store, state, "training_neural_weights",
            event="neural_optimizer_step",
            detail=(str(step) + "/" + str(target) +
                    " · " + state['current_training_task'] + " optimizer update · loss " +
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
            "plain_dataset_directory",
            "supervised_qa_steps", "qa_validation_loss_before", "qa_validation_loss_after",
        ) if key in training}
    if training.get("status") == "trained":
        state["training_runs_with_new_weights"] += 1
        state["training_steps_this_run"] = (
            base_steps + int(training["steps"]))
        state["training_byte_tokens_processed"] = (
            base_tokens + int(training.get("training_window_bytes_seen", 0)))
    _publish(store, state, "training_result", event="training_result",
             detail=(training["status"] + " · " +
                     (('checkpoint accepted' if training.get('accepted') else 'candidate rejected; accepted checkpoint retained')
                      if training.get('status')=='trained' else str(training.get("reason") or "")[:200]) +
                     (' · '+str(training['supervised_qa_steps'])+' supervised QA steps · held-out answer loss '+
                      str(round(training['qa_validation_loss_before'],4))+' → '+str(round(training['qa_validation_loss_after'],4))
                      if training.get('qa_validation_loss_after') is not None else '')))
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
        "host_cooldown_until": state.get("_host_cooldown_until", {}),
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
        "pages_attempted_this_cycle": completed_page_attempts,
        "pages_target_this_cycle": int(pages),
        "idle_reason": state["last_cycle_idle_reason"],
        "archive_link_recovery": recovery,
        "publisher_sitemap_discovery": state.get("last_sitemap_discovery"),
        "publisher_pacing_waits_this_cycle": pacing_waits,
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
        "extraction_bundle": state["last_cycle_extraction"],
    }


def report(store, profile_name="saudi-history-ar"):
    profile = get_profile(profile_name)
    run = store.get(_KEY + profile_name)
    observations = store.get(_HISTORY + profile_name, []) or []
    model = store.get("focus_train_latest_" + profile_name) or {}
    approved = training_readiness(store, profile_name)
    approved = {k: v for k, v in approved.items() if not k.startswith("_")}
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
            "browser_search_requires_api_key": (
                (run or {}).get("browser_search_provider") == "brave" or (
                    (run or {}).get("browser_search_provider") == "auto" and
                    bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()))),
            "official_web_search_api_available": True,
            "official_web_search_api_configured": bool(
                os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()),
            "independent_publisher_priority_enabled": True,
            "external_citation_discovery_enabled": True,
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
        interval_minutes=2, max_pages=24, pages_per_cycle=20,
        train_steps=40, extra_urls=None, fetcher=None, sleep_fn=time.sleep,
        monotonic_fn=time.monotonic, until_stopped=False,
        search_fn=None, no_fixed_quotas=False,
        search_provider="auto"):
    """User-started local process; continuous mode ends ONLY on manual stop.

    Per-cycle page targets count real permitted fetch attempts, not publisher
    pacing waits. Publisher request delays and access rights remain mandatory.
    Source queue and page caps reset at UTC midnight, but network permissions,
    robots, TLS, license review and unchanged-dataset training restrictions do
    not reset. With no new authorized work, it remains awake but idle.
    """
    policy=store.get('focus_execution_policy') or {}
    if policy.get('engine')=='pytorch' and policy.get('article_body_training') and not policy.get('autonomous_pipeline'):
        from .pytorch_learning import run_pytorch
        return run_pytorch(store,steps=int(train_steps),context=256,until_stopped=until_stopped)
    if policy.get('engine')=='first_party_c':
        from .owned_learning import run_owned
        return run_owned(store,steps=int(train_steps),context=256,until_stopped=until_stopped)
    if not until_stopped and not 0 < float(hours) <= 12:
        raise ValueError("hours must be within (0, 12]")
    if no_fixed_quotas and not until_stopped:
        raise ValueError("--no-fixed-quotas requires --until-stopped")
    if search_provider not in ("catalog", "auto", "brave", "duckduckgo", "google", "bing"):
        raise ValueError("Unknown browser search provider")
    if not 1 <= int(interval_minutes) <= 60:
        raise ValueError("interval-minutes must be 1..60")
    if not 1 <= int(pages_per_cycle) <= 50:
        raise ValueError("pages-per-cycle must be 1..50")
    lock_handle = acquire_lock(store, profile_name)
    try:
        state = init_state(store, profile_name, extra_urls, max_pages=max_pages)
        state["until_stopped"] = bool(until_stopped)
        state["no_fixed_quotas"] = bool(no_fixed_quotas)
        state["browser_search_provider"] = search_provider
        state['training_engine']='pytorch' if policy.get('engine')=='pytorch' else 'legacy_pytorch'
        state['network_enabled']=True
        state["pages_per_cycle_target"] = int(pages_per_cycle)
        state["cycle_interval_minutes"] = int(interval_minutes)
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
            "pages_per_cycle_target": int(pages_per_cycle),
            "cycle_interval_minutes": int(interval_minutes),
            "note": (
                "Continuous discovery runs until manual stop. There are no "
                "arbitrary daily page/search counts in open-ended mode; "
                "per-host pacing, robots, access and model rights still apply."
                if no_fixed_quotas else
                "Daily quotas still apply. Mac must stay awake."
            ),
        }, ensure_ascii=False), flush=True)
        try:
            if policy.get('autonomous_pipeline'):
                from .supervised_corpus import prepare_supervised_corpus,read_supervised_corpus,plan_curiosity
                from .pytorch_learning import TorchGRU
                qa_descriptor=prepare_supervised_corpus(store)
                qa_examples=read_supervised_corpus(qa_descriptor)
                accepted=store.get('focus_train_latest_'+profile_name) or {}
                planner,_=TorchGRU.load(store.root,accepted['checkpoint'],accepted['sha256'])
                tasks=plan_curiosity(store,planner,qa_examples['train']);planner.close()
                _publish(store,state,'autonomous_questions_planned',event='curiosity_questions_planned',
                         detail=str(len(tasks))+' source questions ranked by your local model')
            while not operator_stop() and (
                    end is None or monotonic_fn() < end):
                if policy.get('autonomous_pipeline'):
                    import shutil
                    if shutil.disk_usage(store.root).free<2_000_000_000:
                        _publish(store,state,'storage_capacity_wait',event='autonomous_storage_wait',
                                 detail='Less than 2GB free; preserving saved data; collection/training waits for free space')
                        for _ in range(30):
                            if operator_stop():break
                            sleep_fn(2)
                        continue
                _rollover_if_new_utc_day(state)
                try:
                    result = cycle(store, settings, state, pages=pages_per_cycle,
                                   train_steps=train_steps, fetcher=fetcher,
                                   sleep_fn=sleep_fn, search_fn=search_fn)
                except (OSError,ValueError,RuntimeError) as exc:
                    if not policy.get('autonomous_pipeline'):raise
                    control=store.get('focus_autonomous_learning_control') or {}
                    control['learning_rate']=max(.00005,float(control.get('learning_rate',.0003))*.5)
                    control['last_error_type']=type(exc).__name__;store.put('focus_autonomous_learning_control',control)
                    _publish(store,state,'autonomous_recovery_wait',event='autonomous_retry',
                             detail=type(exc).__name__+': '+str(exc)[:250]+'; retained accepted checkpoint; retry in 60 seconds')
                    for _ in range(30):
                        if operator_stop():break
                        sleep_fn(2)
                    continue
                state["hours_elapsed"] = round(
                    (monotonic_fn() - start) / 3600, 3)
                state["heartbeat_at_utc"] = _utc()
                if result["outcomes"] and result["outcomes"][0].get(
                        "status") in (
                            "queue_empty", "daily_budget_exhausted",
                            "daily_page_cap_reached", "publisher_pacing_wait",
                            "source_retry_wait", "publisher_daily_quota"):
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
