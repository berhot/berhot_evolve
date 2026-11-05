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
_SITEMAP_SCAN_VERSION = 2  # One fresh scan after correcting locale/index traversal.
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


_LOCALES = frozenset(("ar", "en", "fr", "de", "ru", "ch", "zh", "es"))
_MAX_SITEMAPS_PER_PROBE = 4


def _locale_relevant(url, language="ar"):
    """Prefer base-language and requested-language sitemaps, not foreign ones."""
    segments = urlsplit(url).path.strip("/").lower().split("/")
    return not (segments[0] in _LOCALES and segments[0] != language)


def _fetch_publisher_sitemaps(host, *, language="ar",
                              document_reader=None,
                              dns_checker=check_public_dns,
                              sleep_fn=time.sleep, time_fn=time.monotonic):
    """Read robots.txt and bounded linked XML sitemap documents.

    A sitemapindex is NOT a list of articles. Follow its child XML before
    spending the document budget on other-language sitemap indexes. Each
    document independently needs robots approval and same-host HTTPS.
    """
    dns_checker(host)
    opener = make_fetcher() if document_reader is None else None
    requests = 0
    last_requested_at = None
    request_gap_seconds = 20

    def read(url, cap):
        nonlocal requests, last_requested_at
        # The normal article worker has a 20-second publisher gap. Sitemap
        # discovery must honor the same spacing, even within one probe.
        # The injected reader uses no network and no delays in unit tests.
        if document_reader is None and last_requested_at is not None:
            elapsed = time_fn() - last_requested_at
            if elapsed < request_gap_seconds:
                sleep_fn(request_gap_seconds - max(0, elapsed))
        requests += 1
        last_requested_at = time_fn()
        if document_reader is not None:
            return document_reader(url, cap)
        with opener.open(urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT,
                              "Accept": "application/xml,text/xml,text/plain"}),
                timeout=12) as response:
            typ = response.headers.get("Content-Type", "").lower()
            payload = response.read(cap + 1)
            if len(payload) > cap:
                raise WebSourceError("Sitemap or robots response too large")
            return payload, typ

    robots_url = "https://" + host + "/robots.txt"
    try:
        robots, _ = read(robots_url, _MAX_ROBOTS_BYTES)
        if isinstance(robots, bytes) and len(robots) > _MAX_ROBOTS_BYTES:
            raise WebSourceError("Oversized robots.txt")
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
            link = _same_host_https(line.partition(":")[2].strip(), host)
            if link and _locale_relevant(link, language):
                advertised.append(link)
    main = "https://" + host + "/sitemap.xml"
    # Root-language index and requested-language declarations first; generic
    # fallback only when no publisher declaration exists.
    endpoints = list(dict.fromkeys(advertised + [main]))
    endpoints.sort(key=lambda u: (
        0 if u == main else 1 if urlsplit(u).path.startswith(
            "/" + language + "/") else 2, u))
    pending = endpoints
    seen = set()
    urls = []
    failures = []
    parsed_indexes = 0
    parsed_urlsets = 0
    listed_locations = 0
    while pending and len(seen) < _MAX_SITEMAPS_PER_PROBE:
        sitemap = pending.pop(0)
        if (sitemap in seen or not _same_host_https(sitemap, host)
                or not _locale_relevant(sitemap, language)):
            continue
        seen.add(sitemap)
        target_path = urlsplit(sitemap).path or "/"
        if (not parser.can_fetch(USER_AGENT, sitemap) or
                _robots_denies_wildcard(rules, target_path)):
            failures.append("sitemap_disallowed_by_robots")
            continue
        try:
            data, typ = read(sitemap, _MAX_XML_BYTES)
            if not isinstance(data, bytes) or len(data) > _MAX_XML_BYTES:
                failures.append("sitemap_oversized")
                continue
            if not any(x in typ for x in
                       ("xml", "text/plain", "application/octet-stream")):
                failures.append("sitemap_not_xml")
                continue
            kind, links = _xml_links(data)
        except (HTTPError, URLError, OSError, TimeoutError,
                ValueError, ElementTree.ParseError) as exc:
            failures.append("sitemap_unavailable_" + type(exc).__name__)
            continue
        if kind == "urlset":
            parsed_urlsets += 1
            listed_locations += len(links)
            for url in links:
                link = _same_host_https(url, host)
                if link:
                    urls.append(link)
                if len(urls) >= 20000:
                    break
        else:
            parsed_indexes += 1
            choices = [x for x in links if _same_host_https(x, host)
                       and _locale_relevant(x, language)]
            choices.sort(key=lambda x: (
                0 if any(word in x.casefold() for word in
                         ("/ar/", "article", "history", "post", "page"))
                else 1, x))
            # Prefer child XMLs ahead of alternative language/root indexes.
            children = [link for link in choices[:_MAX_SITEMAPS_PER_PROBE]
                        if link not in seen and link not in pending]
            pending = children + pending
        if len(urls) >= 20000:
            break
    return {
        "status": ("sitemap_urls_found" if urls else
                   "sitemap_parsed_but_no_article_urls"
                   if parsed_indexes or parsed_urlsets else
                   failures[0] if failures else "sitemap_no_article_urls"),
        "url_candidates": urls[:20000],
        "network_requests": requests,
        "sitemap_documents_checked": len(seen),
        "sitemap_indexes_parsed": parsed_indexes,
        "sitemap_urlsets_parsed": parsed_urlsets,
        "sitemap_locations_listed": listed_locations,
        "sitemap_failures": failures[:5],
        "sitemap_more_documents_unchecked":
            bool(pending and len(seen) >= _MAX_SITEMAPS_PER_PROBE),
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
        host for host in hosts
        if ((history.get(host) or {}).get("scan_version") !=
            _SITEMAP_SCAN_VERSION or
            now - float((history.get(host) or {}).get("checked_at", 0)) >=
            _HOST_GAP_SECONDS)
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
    history[host] = {"checked_at": now, "status": "attempting",
                     "scan_version": _SITEMAP_SCAN_VERSION}
    store.put(key, history)
    try:
        result = (sitemap_fn(host) if sitemap_fn is not None
                  else _fetch_publisher_sitemaps(
                      host, language=profile.get("language", "ar")))
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
        "scan_version": _SITEMAP_SCAN_VERSION,
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
        "topical_urls_after_filter": len(all_topical),
        "sitemap_indexes_parsed": result.get("sitemap_indexes_parsed", 0),
        "sitemap_urlsets_parsed": result.get("sitemap_urlsets_parsed", 0),
        "sitemap_locations_listed": result.get("sitemap_locations_listed", 0),
        "sitemap_failures": result.get("sitemap_failures", []),
        "sitemap_more_documents_unchecked":
            result.get("sitemap_more_documents_unchecked", False),
        "backlog_remaining": len(remainder),
        "candidate_urls_are_not_training_rights": True,
        "retry_after_seconds": _HOST_GAP_SECONDS,
    }
