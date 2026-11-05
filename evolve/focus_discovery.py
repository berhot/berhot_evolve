"""Topic-driven discovery across publishers without a permanent URL/host allowlist.

Discovery hints are untrusted; only fetch_page() may fetch result pages and it
checks DNS, HTTPS, robots, access restrictions, MIME, and response size.
Headless search is keyless and honors search-engine robots/policy restrictions.
It only supplies candidate LINKS, not training rights or factual assertions.
"""
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
import hashlib
import re
import time

from .web_discovery import canonical_url
from .focus_data import topic_relevance

_MAX_SEARCHES_PER_UTC_DAY = 6  # legacy finite run compatibility
_MIN_GAP_BETWEEN_OPEN_ENDED_SEARCHES_SEC = 180
_MAX_LINKS_PER_PAGE = 12
_DISALLOWED_URL_PARTS = (
    "/login", "/signin", "/register", "/checkout", "/account",
    "/search?", "/privacy", "/terms", "/cookie",
    ".pdf", ".jpg", ".png", ".gif", ".zip", ".mp4", ".exe",
)
_AVOID_HOSTS = (
    "facebook.com", "instagram.com", "tiktok.com", "x.com",
    "twitter.com", "youtube.com", "youtu.be", "linkedin.com",
    "pinterest.com", "t.me", "wa.me",
)


def _utc_day():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _candidate_link(url):
    """Return sanitized public HTTPS URL without hostname allowlisting."""
    try:
        safe = canonical_url(url)
    except (ValueError, AttributeError):
        return None
    parts = urlsplit(safe)
    host = parts.hostname or ""
    if any(host == bad or host.endswith("." + bad) for bad in _AVOID_HOSTS):
        return None
    lower = unquote(parts.path).casefold()
    if any(item in lower for item in _DISALLOWED_URL_PARTS):
        return None
    if len(safe) > 1300 or len(lower) > 380:
        return None
    return safe


class _AnchorCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.current = None
        self.entries = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        props = dict(attrs)
        url = props.get("href")
        self.current = {"href": url, "text": []} if url else None

    def handle_data(self, data):
        if self.current:
            self.current["text"].append(data[:300])

    def handle_endtag(self, tag):
        if tag == "a" and self.current:
            self.entries.append((
                self.current["href"],
                " ".join(" ".join(self.current["text"]).split())[:240]))
            self.current = None


def relevant_links(html, source_url, profile, max_results=_MAX_LINKS_PER_PAGE):
    """Follow relevant links to ANY domain, based on anchor/URL hints.

    The target article must independently pass the Arabic/content filter.
    Link snippets are never source evidence, and robots are checked on fetch.
    """
    collector = _AnchorCollector()
    collector.feed(html)
    candidates, seen = [], set()
    for ref, anchor in collector.entries[:800]:
        link = _candidate_link(urljoin(source_url, ref or ""))
        if not link or link in seen or link == source_url:
            continue
        seen.add(link)
        uri_text = unquote(urlsplit(link).path).replace("-", " ").replace("_", " ")
        quality = topic_relevance(
            (anchor + " " + uri_text)[:900],
            anchor + " " + uri_text, profile)
        query_signal = (
            any(word.casefold() in (anchor + " " + uri_text).casefold()
                for word in profile["url_terms"])
        )
        if not quality["topical"] and not query_signal:
            continue
        rank = (
            4 * len(quality["title_keyword_hits"]) +
            2 * len(quality["keyword_hits"]) +
            int(urlsplit(link).hostname != urlsplit(source_url).hostname)
        )
        candidates.append((rank, link))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [url for _, url in candidates[:min(int(max_results), 30)]]


def _queries_from_observations(profile, observations):
    """The follow-up search questions come from observed source evidence."""
    initial = (
        "تاريخ المملكة العربية السعودية مصادر تاريخية",
        "الدولة السعودية الأولى الدرعية تاريخ",
        "توحيد المملكة العربية السعودية التاريخ",
        "تأسيس الدولة السعودية الثانية مصادر",
    )
    output, seen = [], set()
    from_sources = []
    for entry in observations[-2000:]:
        for q in entry.get("curiosity", []):
            phrase = q.get("question", "").strip()
            subject = q.get("subject", "").strip()
            if phrase:
                from_sources.append(phrase + " تاريخ مصادر مستقلة")
            if subject:
                from_sources.append(subject + " وثائق تاريخية مستقلة")
        title = entry.get("title", "").strip()
        if title and 8 <= len(title) <= 110:
            from_sources.append(title + " مصادر تاريخية")
    for query in from_sources + list(initial):
        query = " ".join(query.split())[:150]
        if not query or query in seen:
            continue
        seen.add(query)
        output.append(query)
    return output


def discover_search_urls(store, profile, observations, *, search_fn=None,
                         open_ended=False):
    """Plan a query about actual article evidence, independent of an API key.

    A finite legacy run retains its old small budget; open-ended focus never
    stops due to an arbitrary page/search count. It still has time pacing.
    """
    day = _utc_day()
    used = int(store.get("focus_search_used_" + day, 0) or 0)
    if not open_ended and used >= _MAX_SEARCHES_PER_UTC_DAY:
        return {"status": "search_daily_limit",
                "queries_attempted": 0, "url_candidates": []}
    if open_ended:
        last = float(store.get("focus_browser_search_last_at", 0) or 0)
        elapsed = time.time() - last
        if 0 <= elapsed < _MIN_GAP_BETWEEN_OPEN_ENDED_SEARCHES_SEC:
            return {
                "status": "search_pacing_wait",
                "retry_after_seconds":
                    int(_MIN_GAP_BETWEEN_OPEN_ENDED_SEARCHES_SEC - elapsed),
                "queries_attempted": 0, "url_candidates": [],
            }
    # Only successful browser searches count as done. Earlier versions
    # incorrectly recorded failures as completed queries and could leave an
    # otherwise healthy worker idle with no new sources for the entire day.
    done = set(store.get("focus_search_success_" + day, []) or [])
    queries = _queries_from_observations(profile, observations)
    historical = store.get("focus_search_history_" + profile["name_ar"]) or {}
    failures = store.get("focus_search_failures_" + profile["name_ar"]) or {}
    now = time.time()
    eligible, retry_times = [], []
    for i, query in enumerate(queries):
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if query_hash in done:
            continue
        prior_failure = failures.get(query_hash) or {}
        retry_at = float(prior_failure.get("retry_at", 0) or 0)
        if retry_at > now:
            retry_times.append(retry_at)
            continue
        eligible.append((
            historical.get(query_hash, ""),
            float(prior_failure.get("last_attempt_at", 0) or 0),
            i, query,
        ))
    eligible.sort(key=lambda candidate: candidate[:3])
    selected = eligible[0][3] if eligible else None
    if not selected:
        if retry_times:
            return {
                "status": "search_retry_wait",
                "retry_after_seconds": max(1, int(min(retry_times) - now)),
                "queries_attempted": 0, "url_candidates": [],
            }
        return {"status": "no_new_search_questions_today",
                "queries_attempted": 0, "url_candidates": []}
    return {
        "status": "query_ready", "query": selected,
        "query_hash": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        "queries_attempted": 0, "url_candidates": [],
    }


def run_one_search(store, settings, profile, observations,
                   search_fn=None, open_ended=False,
                   provider="duckduckgo"):
    """Browser search without API keys, respecting policies and stop limits.

    Google/Bing are explicitly unsupported for unattended results access
    while their robots/terms disallow it; no disguising or CAPTCHA bypass.
    """
    planned = discover_search_urls(
        store, profile, observations, search_fn=search_fn,
        open_ended=open_ended)
    if planned["status"] != "query_ready":
        return planned
    day = _utc_day()
    shared_key = "web_requests_" + day
    used = int(store.get(shared_key, 0) or 0)
    if (not open_ended and
            used >= int(settings.get("daily_request_budget", 50))):
        return {"status": "shared_network_daily_budget_exhausted",
                "queries_attempted": 0, "url_candidates": []}
    # Count the attempted request, but do NOT claim the query was completed
    # until the search backend actually succeeded.
    store.put(shared_key, used + 1)
    store.put("focus_search_used_" + day,
              int(store.get("focus_search_used_" + day, 0) or 0) + 1)
    if open_ended:
        store.put("focus_browser_search_last_at", time.time())

    failure_key = "focus_search_failures_" + profile["name_ar"]

    def fail(status, **details):
        failures = store.get(failure_key) or {}
        previous = failures.get(planned["query_hash"]) or {}
        count = int(previous.get("count", 0) or 0) + 1
        # Avoid hammering an unavailable browser or search provider;
        # retain the query for a later retry rather than marking it done.
        retry_delay = min(3600, 300 * (2 ** min(count - 1, 4)))
        now = time.time()
        failures[planned["query_hash"]] = {
            "count": count,
            "last_attempt_at": now,
            "retry_at": now + retry_delay,
        }
        store.put(failure_key, failures)
        return {
            "status": status, "queries_attempted": 1,
            "provider": provider, "retry_after_seconds": retry_delay,
            "url_candidates": [], **details,
        }

    try:
        if search_fn is not None:
            raw_result = search_fn(planned["query"])
            result = {"status": "search_completed", "results": raw_result}
        else:
            from .focus_browser_search import headless_search
            result = headless_search(planned["query"], provider=provider)
    except Exception as exc:
        return fail("search_failed", error_type=type(exc).__name__)

    if result.get("status") not in ("search_completed", "search_no_results"):
        return fail(
            result.get("status", "search_failed"),
            error_type=result.get("error_type"),
            error_phase=result.get("error_phase"),
            hint=result.get("hint"),
        )

    success_key = "focus_search_success_" + day
    done = store.get(success_key, []) or []
    store.put(success_key, (done + [planned["query_hash"]])[-5000:])
    history_key = "focus_search_history_" + profile["name_ar"]
    history = store.get(history_key) or {}
    history[planned["query_hash"]] = day
    store.put(history_key, history)
    failures = store.get(failure_key) or {}
    if planned["query_hash"] in failures:
        failures.pop(planned["query_hash"], None)
        store.put(failure_key, failures)

    candidates = []
    for item in result.get("results", [])[:80]:
        if not isinstance(item, dict):
            continue
        safe = _candidate_link(item.get("url"))
        if not safe:
            continue
        hint = str(item.get("title", "")) + " " + str(
            item.get("snippet", ""))
        quality = topic_relevance(hint, hint, profile)
        if quality["topical"] or quality["language_ok"]:
            candidates.append(safe)
    return {
        "status": result["status"],
        "queries_attempted": 1,
        "query": planned["query"],
        "provider": provider,
        "url_candidates": list(dict.fromkeys(candidates))[:30],
        "result_snippets_trusted_as_fact": False,
        "search_api_key_required": False,
        "browser_visible": False,
    }
