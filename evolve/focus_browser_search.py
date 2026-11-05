"""Keyless research discovery through a headless browser search form.

This is *not* a stealth browser or an automation-bypass tool. Search engines
must explicitly allow the results route in their current robots policy and
terms. Google and Bing disallow unattended /search results access, so they are
recorded as unavailable, not accessed by impersonating a human.

DuckDuckGo's current robots policy forbids automated search result query URLs
(Disallow: /*?). The built-in urllib.robotparser does not implement robots
wildcard matching, so explicit wildcard-query restrictions must fail closed.
Future robots policy changes must still pass live checks; CAPTCHA, 403/429 or
challenges are never bypassed.
All search text/results are merely discovery hints, never training data.
"""
from urllib.parse import quote, urlsplit
from urllib.robotparser import RobotFileParser
import re
import urllib.request

from .web_discovery import USER_AGENT, canonical_url

_ENGINES = {
    "duckduckgo": {
        "home": "https://duckduckgo.com/",
        "results": "https://duckduckgo.com/?q={query}",
        "selector": "input[name='q']",
        "results_selector": "a[data-testid='result-title-a'], .result__a, h2 a[href]",
        "automated_results_allowed": True,
    },
    "google": {
        "home": "https://www.google.com/",
        "results": "https://www.google.com/search?q={query}",
        "automated_results_allowed": False,
        "reason": "Google prohibits unauthorized automated search queries and robots disallows /search",
    },
    "bing": {
        "home": "https://www.bing.com/",
        "results": "https://www.bing.com/search?q={query}",
        "automated_results_allowed": False,
        "reason": "Bing robots.txt disallows /search for generic automated crawlers",
    },
}

_BLOCK_TEXT = (
    "captcha", "automated traffic", "unusual traffic",
    "verify you are human", "access denied", "rate limit",
)


def search_engines():
    return {
        name: {
            "headless_supported": bool(spec["automated_results_allowed"]),
            "no_api_key_needed": True,
            "reason": spec.get("reason", "Requires live robots permission and no blocking challenge"),
        }
        for name, spec in _ENGINES.items()
    }


def _robots_allows(url, *, fetch_robots=None):
    """Fail closed for unavailable, unreadable, or restrictive robots.txt."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        return False
    endpoint = f"https://{parts.hostname}/robots.txt"
    try:
        if fetch_robots is None:
            request = urllib.request.Request(
                endpoint, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=12) as response:
                if response.status != 200:
                    return False
                raw = response.read(200_001)
        else:
            raw = fetch_robots(endpoint)
        if isinstance(raw, bytes):
            if len(raw) > 200_000:
                return False
            robots = raw.decode("utf-8", errors="strict")
        else:
            robots = str(raw)
        # RobotFileParser ignores the '*' wildcard in patterns like /*?,
        # which can incorrectly allow DuckDuckGo result URLs. Treat such
        # global query restrictions conservatively rather than bypass them.
        # This guard can overblock, but never grants access.
        if parts.query and re.search(
                r"(?im)^\\s*Disallow:\\s*/\\*\\?[^\\r\\n]*$", robots):
            return False
        parser = RobotFileParser()
        parser.parse(robots.splitlines())
        return parser.can_fetch(USER_AGENT, url)
    except (OSError, ValueError, UnicodeError):
        return False


def _search_results(page, provider, limit=12):
    """Extract result titles/URLs; never treat snippets as facts."""
    from .focus_discovery import _candidate_link
    spec = _ENGINES[provider]
    found, seen = [], set()
    locator = page.locator(spec["results_selector"])
    count = min(locator.count(), 80)
    for i in range(count):
        entry = locator.nth(i)
        try:
            href = entry.get_attribute("href", timeout=500)
            title = entry.inner_text(timeout=700).strip()[:220]
        except Exception:
            continue
        if not href or not title:
            continue
        # Resolve only genuine external article addresses. Do not decode
        # tracking redirect payloads or follow sponsored redirect links.
        try:
            link = _candidate_link(href)
        except (ValueError, AttributeError):
            continue
        if not link or link in seen:
            continue
        if urlsplit(link).hostname in (
                "duckduckgo.com", "www.google.com", "www.bing.com"):
            continue
        seen.add(link)
        found.append({"url": link, "title": title, "snippet": ""})
        if len(found) >= int(limit):
            break
    return found


def headless_search(query, provider="duckduckgo", *, robots_check=None,
                    browser_factory=None):
    """Search using an invisible Chromium page, not an API or fake identity.

    robots_check and browser_factory may be injected by OFFLINE unit tests.
    The default browser requires the optional playwright dependency and
    installed Chromium binaries. None of the user's browser cookies/login
    sessions or data are loaded.
    """
    if provider not in _ENGINES:
        return {"status": "unknown_search_engine", "results": [],
                "engine": provider}
    spec = _ENGINES[provider]
    if not spec["automated_results_allowed"]:
        return {"status": "search_engine_policy_disallows_automation",
                "reason": spec["reason"], "results": [],
                "engine": provider}
    query = " ".join(str(query).split())[:150]
    if len(query) < 2:
        return {"status": "invalid_query", "results": [], "engine": provider}
    target = spec["results"].format(query=quote(query, safe=""))
    check = robots_check if robots_check is not None else _robots_allows
    if not check(target):
        return {"status": "search_engine_robots_disallow_or_unverifiable",
                "engine": provider, "results": []}
    if browser_factory is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {"status": "playwright_not_installed", "engine": provider,
                    "results": [], "hint": (
                        "Install: python3 -m pip install -e '.[browser]' && "
                        "python3 -m playwright install chromium")}
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception:
                return {"status": "chromium_not_installed_or_unavailable",
                        "engine": provider, "results": [],
                        "hint": "Run python3 -m playwright install chromium"}
            try:
                return _run_browser(browser, spec, query, provider,
                                    target, check)
            finally:
                browser.close()
    browser = browser_factory()
    try:
        return _run_browser(browser, spec, query, provider, target, check)
    finally:
        browser.close()


def _run_browser(browser, spec, query, provider, target, check):
    # Report a safe failure phase instead of hiding all browser failures
    # behind a generic status. Never expose browser paths, cookies or HTML.
    phase = "creating_page"
    page = None
    try:
        page = browser.new_page(
            locale="ar-SA", viewport={"width": 1180, "height": 900},
            accept_downloads=False,
        )
        phase = "opening_search_home"
        page.goto(spec["home"], wait_until="domcontentloaded",
                  timeout=15000)
        phase = "submitting_query"
        search = page.locator(spec["selector"]).first
        search.fill(query, timeout=7000)
        search.press("Enter", timeout=7000)
        phase = "waiting_for_results"
        try:
            page.wait_for_load_state("domcontentloaded", timeout=12000)
        except Exception:
            pass
        current = page.url
        if (urlsplit(current).hostname != urlsplit(target).hostname or
                not check(current)):
            return {"status": "search_redirect_disallowed",
                    "engine": provider, "results": []}
        phase = "checking_for_challenges"
        content = (page.title() + " " +
                   page.locator("body").inner_text(timeout=5000)[:4500]
                   ).casefold()
        if any(token in content for token in _BLOCK_TEXT):
            return {"status": "search_blocked_or_challenged",
                    "engine": provider, "results": []}
        phase = "extracting_result_links"
        try:
            page.locator(spec["results_selector"]).first.wait_for(
                state="visible", timeout=7000)
        except Exception:
            pass
        results = _search_results(page, provider)
        return {
            "status": "search_completed" if results else "search_no_results",
            "engine": provider, "results": results,
            "browser_visible": False,
            "captcha_bypassed": False, "search_snippets_are_facts": False,
        }
    except Exception as exc:
        return {
            "status": "browser_search_unavailable", "engine": provider,
            "error_type": type(exc).__name__, "error_phase": phase,
            "results": [],
        }
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
