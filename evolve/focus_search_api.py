"""Optional official web search API; returns links, never training text.

Only the fixed Brave API endpoint receives the key. Redirects are refused so
credentials cannot follow a response to another origin. No secrets are logged.
"""
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode
import urllib.request

from .web_discovery import USER_AGENT, make_fetcher


def brave_search(query, *, api_key=None, opener=None):
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "") if api_key is None else api_key
    if not key.strip():
        return {"status": "search_api_key_missing", "engine": "brave",
                "results": [], "hint": "Set BRAVE_SEARCH_API_KEY in the worker environment"}
    query = " ".join(str(query).split())[:150]
    if len(query) < 2:
        return {"status": "invalid_query", "engine": "brave", "results": []}
    url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({
        "q": query, "count": 20, "country": "SA", "search_lang": "ar"})
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json",
        "X-Subscription-Token": key.strip()})
    try:
        client = opener if opener is not None else make_fetcher()
        with client.open(request, timeout=15) as response:
            if response.geturl() != url:
                return {"status": "search_redirect_disallowed", "engine": "brave", "results": []}
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("Oversized search response")
            payload = json.loads(raw)
        entries = payload.get("web", {}).get("results", [])
        results = [{"url": row.get("url"), "title": row.get("title", ""),
                    "snippet": row.get("description", "")}
                   for row in entries[:40] if isinstance(row, dict)]
        return {"status": "search_completed" if results else "search_no_results",
                "engine": "brave", "results": results,
                "search_api_key_required": True}
    except HTTPError as exc:
        return {"status": ("search_api_rate_limited" if exc.code == 429 else
                           "search_api_authentication_failed" if exc.code in (401, 403)
                           else "search_api_http_error"),
                "engine": "brave", "results": [], "http_status": exc.code}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"status": "search_api_failed", "engine": "brave",
                "results": [], "error_type": type(exc).__name__}
