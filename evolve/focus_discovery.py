"""Topic-driven discovery across publishers without a permanent URL/host allowlist.

Discovery hints are untrusted; only fetch_page() may fetch result pages and it
checks DNS, HTTPS, robots, access restrictions, MIME, and response size.
Optional Brave Web Search uses a user-provided key (never stored in SQLite).
It only supplies candidate LINKS, not training rights or factual assertions.
"""
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote, urlencode
import hashlib
import json
import os
import re
import urllib.request

from .web_discovery import canonical_url, make_fetcher
from .focus_data import topic_relevance

_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_MAX_SEARCHES_PER_UTC_DAY = 6
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
    for query in list(initial) + [
        q.get("question", "") + " تاريخ مصادر مستقلة"
        for entry in observations[-100:]
        for q in entry.get("curiosity", [])[:8]
    ]:
        query = " ".join(query.split())[:150]
        if not query or query in seen:
            continue
        if len(output) >= 60:
            break
        seen.add(query)
        output.append(query)
    return output


def _brave_search(query, key, opener=None):
    """Use Brave's documented structured Web Search endpoint; links only."""
    if not isinstance(query, str) or not 1 <= len(query) <= 150:
        raise ValueError("Search question length invalid")
    if not key or not isinstance(key, str):
        raise ValueError("BRAVE_SEARCH_API_KEY required")
    args = urlencode({"q": query, "count": 10,
                      "country": "SA", "search_lang": "ar"})
    request = urllib.request.Request(
        _SEARCH_ENDPOINT + "?" + args,
        headers={"X-Subscription-Token": key,
                 "Accept": "application/json",
                 "User-Agent": "BERHOT-EVOLVE-Research/0.4"},
    )
    fetcher = opener or make_fetcher()
    with fetcher.open(request, timeout=18) as response:
        raw = response.read(350_001)
    if len(raw) > 350_000:
        raise ValueError("Search response too large")
    doc = json.loads(raw.decode("utf-8"))
    results = doc.get("web", {}).get("results", [])
    if not isinstance(results, list):
        return []
    out = []
    for item in results[:20]:
        if not isinstance(item, dict):
            continue
        link = _candidate_link(item.get("url"))
        if link:
            out.append({
                "url": link,
                "title": str(item.get("title", ""))[:220],
                "snippet": str(item.get("description", ""))[:450],
            })
    return out


def discover_search_urls(store, profile, observations, *, search_fn=None):
    """At most 6 opted-in search requests per UTC day; no search-key bypass.

    Uses the same quota counter as page fetches. The API key is never logged.
    The search service may charge separately under the operator's account.
    """
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if search_fn is None and not key:
        return {
            "status": "search_key_not_configured_link_discovery_only",
            "queries_attempted": 0, "url_candidates": [],
        }
    day = _utc_day()
    used = store.get("focus_search_used_" + day, 0) or 0
    done = set(store.get("focus_search_done_" + day, []) or [])
    remaining = max(0, _MAX_SEARCHES_PER_UTC_DAY - int(used))
    if remaining <= 0:
        return {
            "status": "search_daily_limit", "queries_attempted": 0,
            "url_candidates": [],
        }
    queries = _queries_from_observations(profile, observations)
    historical = store.get("focus_search_history_" + profile["name_ar"]) or {}
    # Select previously unseen observed questions first; otherwise revisit
    # the least-recently-searched question on a later day. Without this
    # rotation the first four seeds would monopolize every UTC day's budget.
    eligible = [
        (historical.get(hashlib.sha256(q.encode("utf-8")).hexdigest(), ""),
         i, q)
        for i, q in enumerate(queries)
        if hashlib.sha256(q.encode("utf-8")).hexdigest() not in done
    ]
    eligible.sort(key=lambda triple: (triple[0], triple[1]))
    selected = eligible[0][2] if eligible else None
    if not selected:
        return {"status": "no_new_search_questions_today",
                "queries_attempted": 0, "url_candidates": []}
    day_key = "web_requests_" + day
    used_web = int(store.get(day_key, 0))
    # The absolute daily budget is set in the main settings, separately
    # validated by the caller. This function needs a passed-in max.
    return {
        "status": "query_ready", "query": selected,
        "query_hash": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        "queries_attempted": 0, "url_candidates": [],
    }


def run_one_search(store, settings, profile, observations,
                   search_fn=None):
    planned = discover_search_urls(
        store, profile, observations, search_fn=search_fn)
    if planned["status"] != "query_ready":
        return planned
    day = _utc_day()
    shared_key = "web_requests_" + day
    used = int(store.get(shared_key, 0))
    if used >= int(settings.get("daily_request_budget", 50)):
        return {"status": "shared_network_daily_budget_exhausted",
                "queries_attempted": 0, "url_candidates": []}
    done_key = "focus_search_done_" + day
    done = store.get(done_key, []) or []
    # Count the API request against both quotas BEFORE the HTTP operation.
    store.put(shared_key, used + 1)
    store.put("focus_search_used_" + day,
              int(store.get("focus_search_used_" + day, 0)) + 1)
    store.put(done_key, (done + [planned["query_hash"]])[-100:])
    history_key = "focus_search_history_" + profile["name_ar"]
    history = store.get(history_key) or {}
    history[planned["query_hash"]] = day
    store.put(history_key, history)
    try:
        results = (
            search_fn(planned["query"]) if search_fn is not None
            else _brave_search(
                planned["query"], os.environ["BRAVE_SEARCH_API_KEY"])
        )
    except Exception as exc:
        return {
            "status": "search_failed",
            "queries_attempted": 1,
            "error_type": type(exc).__name__,
            # Do not leak the API key, request headers, or raw URL exceptions.
            "url_candidates": [],
        }
    candidates = []
    for item in results[:20]:
        if not isinstance(item, dict):
            continue
        safe = _candidate_link(item.get("url"))
        if not safe:
            continue
        hint = item.get("title", "") + " " + item.get("snippet", "")
        quality = topic_relevance(hint, hint, profile)
        # Search returns are discovery hints only, not science citations.
        if quality["topical"] or quality["language_ok"]:
            candidates.append(safe)
    return {
        "status": "search_completed",
        "queries_attempted": 1,
        "query": planned["query"],
        "url_candidates": list(dict.fromkeys(candidates))[:10],
        "result_snippets_trusted_as_fact": False,
        "third_party_search_may_charge": True,
    }
