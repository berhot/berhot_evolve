"""Opt-in discovery of topical article URLs from publisher-advertised XML sitemaps.

Sitemaps are *candidate link sources*, not permission to fetch, archive, train
or reproduce a page.  Never bypass HTTPS, public DNS, robots, access controls
or redirects.  Every candidate is checked again by the normal page fetcher.
"""
from datetime import datetime, timezone
from urllib.parse import urlsplit, unquote
from urllib.robotparser import RobotFileParser
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree
import re
import time
import urllib.request

from .focus_discovery import _candidate_link
from .web_discovery import (
    USER_AGENT, canonical_url, check_public_dns, make_fetcher, WebSourceError,
)

_MAX_XML_BYTES = 5_000_000
_MAX_ROBOTS_BYTES = 250_000
_HOST_GAP_SECONDS = 6 * 3600
_SKIP_PATHS = ("tag/", "tags/", "category/", "categories/", "search/",
               "privacy/", "terms/", "footer/", "login/")


def _local_tag(element):
    return str(element.tag).rsplit("}", 1)[-1]


def _same_host_https(link, host):
    try:
        link = canonical_url(link)
    except (ValueError, AttributeError):
        return None
    return link if urlsplit(link).hostname == host else None


def _xml_links(data, *, max_entries=20000):
    """Parse a bounded XML document without accepting DTD or entity tricks."""
    if not isinstance(data, bytes) or len(data) > _MAX_XML_BYTES:
        raise ValueError("XML sitemap too large or not bytes")
    if b"<!doctype" in data.lower() or b"<!entity" in data.lower():
        raise ValueError("DTD and entity declarations forbidden")
    root = ElementTree.fromstring(data)
    tag = _local_tag(root)
    if tag not in ("urlset", "sitemapindex"):
        raise ValueError("Not a sitemap XML document")
    result = []
    for item in root:
        if len(result) >= max_entries:
            break
        if _local_tag(item) != ("url" if tag == "urlset" else "sitemap"):
            continue
        for part in item:
            if _local_tag(part) == "loc" and part.text:
                result.append(part.text.strip())
                break
    return tag, result


def _select_topical(urls, profile, max_urls=30):
    """URL path is a discovery hint.  Full text is independently screened."""
    results, seen = [], set()
    terms = tuple(t.casefold() for t in profile["url_terms"])
    for url in urls:
        link = _candidate_link(url)
        if not link or link in seen:
            continue
        path = unquote(urlsplit(link).path).strip("/").casefold()
        if path in ("", "ar", "en", "index.html", "home.aspx"):
            continue
        if path.startswith(_SKIP_PATHS):
            continue
        if not any(term in path for term in terms):
            continue
        seen.add(link)
        results.append(link)
        if len(results) >= max_urls:
            break
    return results


def _robots_denies_wildcard(lines, target_path):
    # urllib.robotparser ignores wildcard patterns; fail closed for wildcard
    # rules that match a sitemap path. False positives are safer than bypass.
    for line in lines:
        stripped = line.partition("#")[0].strip()
        if not stripped.lower().startswith("disallow:"):
            continue
        pattern = stripped.partition(":")[2].strip()
        if "*" not in pattern:
            continue
        expr = re.escape(pattern.rstrip("$")).replace(r"\*", ".*")
        if re.match("^" + expr, target_path):
            return True
    return False


def _fetch_publisher_sitemaps(host):
    """Read one public publisher's robots.txt and up to 3 same-host XMLs.

    Redirects, auth, TLS failures, unknown robots, private DNS and large XML
    fail closed.  No browser impersonation or hidden endpoint guessing.
    """
    check_public_dns(host)
    opener = make_fetcher()
    requests = 0

    def read(url, cap):
        nonlocal requests
        requests += 1
        with opener.open(urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT,
                              "Accept": "application/xml,text/xml,text/plain"}),
                timeout=12) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            payload = response.read(cap + 1)
            if len(payload) > cap:
                raise WebSourceError("Sitemap or robots response too large")
            return payload, content_type

    robots_url = "https://" + host + "/robots.txt"
    try:
        robots, _ = read(robots_url, _MAX_ROBOTS_BYTES)
        rules = robots.decode("utf-8", "replace").splitlines()
    except HTTPError as exc:
        if exc.code in (404, 410):
            rules = []
        else:
            raise WebSourceError(
                "Cannot verify publisher robots.txt (HTTP " +
                str(exc.code) + ")") from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise WebSourceError("Cannot verify publisher robots.txt") from exc

    parser = RobotFileParser()
    parser.parse(rules)
    advertised = []
    for line in rules:
        if line.lower().lstrip().startswith("sitemap:"):
            candidate = line.partition(":")[2].strip()
            allowed = _same_host_https(candidate, host)
            if allowed:
                advertised.append(allowed)
    # robots-declared sitemap paths first, conventional XML last.
    endpoints = list(dict.fromkeys(
        advertised[:3] + ["https://" + host + "/sitemap.xml"]))
    pending = endpoints[:3]
    seen = set()
    urls = []
    failures = []
    while pending and len(seen) < 3:
        sitemap = pending.pop(0)
        if sitemap in seen or not _same_host_https(sitemap, host):
            continue
        seen.add(sitemap)
        target_path = urlsplit(sitemap).path or "/"
        if (not parser.can_fetch(USER_AGENT, sitemap) or
                _robots_denies_wildcard(rules, target_path)):
            failures.append("sitemap_disallowed_by_robots")
            continue
        try:
            data, typ = read(sitemap, _MAX_XML_BYTES)
            if not (any(x in typ for x in
                        ("xml", "text/plain", "application/octet-stream"))):
                failures.append("sitemap_not_xml")
                continue
            kind, links = _xml_links(data)
        except (HTTPError, URLError, OSError, TimeoutError,
                ValueError, ElementTree.ParseError) as exc:
            failures.append("sitemap_unavailable_" + type(exc).__name__)
            continue
        if kind == "urlset":
            urls.extend(
                url for url in links if _same_host_https(url, host))
        else:
            # Do not scan huge sitemap indices.  Prefer children that mention
            # Arabic/article/history, then scan at most the remaining budget.
            choices = [x for x in links if _same_host_https(x, host)]
            choices.sort(key=lambda x: (not any(
                signal in x.casefold() for signal in
                ("ar", "article", "post", "page", "history")), x))
            for child in choices[:2]:
                if child not in seen and child not in pending:
                    pending.append(child)
    return {
        "status": "sitemap_urls_found" if urls else
                  (failures[0] if failures else "sitemap_no_article_urls"),
        "url_candidates": urls[:20000],
        "network_requests": requests,
        "sitemap_documents_checked": len(seen),
    }


def discover_publisher_sitemaps(store, profile, observations, *,
                                sitemap_fn=None, now=None,
                                max_candidates=25, already_seen=None):
    """Try one due source publisher each time the research frontier runs dry.

    Rotates hosts deterministically across cycles, with durable 6h cooldown.
    The publisher list grows only from profile seeds and successfully archived
    sources, not arbitrary URLs embedded in scraped page text.
    """
    now = time.time() if now is None else float(now)
    known = set(already_seen or ())
    backlog_key = "focus_sitemap_backlog_" + profile["name_ar"]
    backlog = store.get(backlog_key) or []
    if backlog:
        deliver = []
        leftover = []
        for link in backlog:
            if link in known:
                continue
            if len(deliver) < max_candidates:
                deliver.append(link)
            else:
                leftover.append(link)
        store.put(backlog_key, leftover)
        if deliver:
            return {
                "status": "sitemap_cached_candidate_links",
                "host": urlsplit(deliver[0]).hostname,
                "url_candidates": deliver,
                "candidate_urls_found": len(deliver),
                "backlog_remaining": len(leftover),
                "network_requests": 0,
                "candidate_urls_are_not_training_rights": True,
            }

    key = "focus_sitemap_checks_" + profile["name_ar"]
    history = store.get(key) or {}
    hosts = []
    for url in profile["seeds"]:
        host = urlsplit(url).hostname
        if host and host not in hosts:
            hosts.append(host)
    for record in observations[-500:]:
        if not isinstance(record, dict):
            continue
        host = urlsplit(record.get("url", "")).hostname
        if host and host not in hosts:
            hosts.append(host)
    due = [
        host for host in hosts if now -
        float((history.get(host) or {}).get("checked_at", 0)) >=
        _HOST_GAP_SECONDS
    ]
    if not due:
        soonest = min(
            (_HOST_GAP_SECONDS - (now -
              float((entry or {}).get("checked_at", 0))))
            for entry in history.values()
        ) if history else _HOST_GAP_SECONDS
        return {
            "status": "sitemap_publisher_cooldown",
            "host": None, "url_candidates": [],
            "network_requests": 0,
            "retry_after_seconds": max(1, int(soonest)),
        }
    host = due[0]
    # Persist the probe attempt even if it fails, preventing request loops.
    history[host] = {"checked_at": now, "status": "attempting"}
    store.put(key, history)
    try:
        result = (sitemap_fn(host) if sitemap_fn is not None
                  else _fetch_publisher_sitemaps(host))
        if not isinstance(result, dict):
            raise ValueError("Unexpected sitemap provider result")
    except Exception as exc:
        result = {
            "status": "sitemap_unavailable_" + type(exc).__name__,
            "url_candidates": [], "network_requests": 0,
        }
    # Restrict candidate URLs to the responding host even in offline tests.
    candidates = [
        url for url in result.get("url_candidates", [])
        if _same_host_https(url, host)
    ]
    all_topical = _select_topical(candidates, profile, max_urls=2000)
    unseen = [url for url in all_topical if url not in known]
    selected, remainder = (
        unseen[:max_candidates], unseen[max_candidates:2000])
    store.put(backlog_key, remainder)
    history[host] = {
        "checked_at": now, "status": result.get("status", "unknown"),
        "candidate_count": len(unseen),
    }
    store.put(key, history)
    count = int(result.get("network_requests", 0))
    if count:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        request_key = "web_requests_" + day
        store.put(request_key, int(store.get(request_key, 0) or 0) + count)
    return {
        "status": result.get("status", "unknown"),
        "host": host,
        "url_candidates": selected,
        "network_requests": count,
        "sitemap_documents_checked":
            result.get("sitemap_documents_checked", 0),
        "candidate_urls_found": len(selected),
        "backlog_remaining": len(remainder),
        "candidate_urls_are_not_training_rights": True,
        "retry_after_seconds": _HOST_GAP_SECONDS,
    }
