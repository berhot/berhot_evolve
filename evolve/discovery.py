"""Constrained, source-aware autonomous discovery of CC0 Wikidata entities.

No general-purpose website scraping: only a documented, fixed-host, fixed-format
source is admitted. Additional media/social connectors require separate review.
"""
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .settings import user_agent

ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json?flavor=dump"
RELATED_PROPERTIES = ("P31", "P279", "P361", "P527")
MAX_METADATA_CHARS = 400


class DiscoveryError(Exception):
    pass


class RateLimited(DiscoveryError):
    pass


def clean(value):
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    return " ".join(value.split())[:MAX_METADATA_CHARS]


def get_entity_json(entity_id, contact_ua, max_bytes=2_000_000):
    if not re.fullmatch(r"Q[1-9]\d*", entity_id):
        raise DiscoveryError("Invalid Wikidata entity identifier")
    url = ENTITY_URL.format(entity_id=entity_id)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": contact_ua, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=18) as response:
            resolved = urllib.parse.urlparse(response.geturl())
            if resolved.scheme != "https" or resolved.hostname != "www.wikidata.org":
                raise DiscoveryError("Unexpected redirect host")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise DiscoveryError("Entity response exceeds configured size cap")
    except urllib.error.HTTPError as error:
        if error.code in (429, 503):
            raise RateLimited(f"Upstream rate-limited ({error.code}); stop this cycle") from error
        raise DiscoveryError(f"Upstream HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise DiscoveryError(f"Upstream connection failed: {error.reason}") from error
    try:
        return json.loads(data)
    except json.JSONDecodeError as error:
        raise DiscoveryError("Upstream returned invalid JSON") from error


def parse_entity(payload, entity_id, link_limit=6):
    entity = payload.get("entities", {}).get(entity_id, {})
    if not entity or entity.get("type") != "item":
        raise DiscoveryError("Entity missing or not an item")
    languages = ("en", "ar")
    entries = []
    for lang in languages:
        label = clean(entity.get("labels", {}).get(lang, {}).get("value", ""))
        description = clean(entity.get("descriptions", {}).get(lang, {}).get("value", ""))
        if label:
            entries.append(f"[{lang}] {label} ({entity_id}). {description}".strip())
        aliases = entity.get("aliases", {}).get(lang, [])[:4]
        alt = [clean(alias.get("value", "")) for alias in aliases]
        alt = [alias for alias in alt if alias]
        if label and alt:
            entries.append(f"[{lang}] Alternative names for {label}: {', '.join(alt)}.")
    text = "\n".join(entries).strip()
    linked = []
    for property_id in RELATED_PROPERTIES:
        for claim in entity.get("claims", {}).get(property_id, [])[:link_limit]:
            if claim.get("rank") == "deprecated":
                continue
            snak = claim.get("mainsnak", {})
            value = snak.get("datavalue", {}).get("value", {})
            if isinstance(value, dict) and value.get("entity-type") == "item":
                qid = value.get("id")
                if qid and re.fullmatch(r"Q[1-9]\d*", qid) and qid != entity_id:
                    linked.append(qid)
    unique_links = list(dict.fromkeys(linked))[:link_limit]
    return text, unique_links


def discover(store, settings, limit=None, fetcher=None, sleep_fn=time.sleep):
    """Execute one bounded discovery cycle; leaves frontier state persisted."""
    contact_ua = user_agent() if fetcher is None else "BERHOT-EVOLVE-TEST/0.1"
    cap = int(limit if limit is not None else settings["discovery_per_cycle"])
    if cap < 1 or cap > 1000:
        raise ValueError("Discovery limit must be between 1 and 1000")
    for entity_id in settings["seed_entities"]:
        store.add_frontier(entity_id, depth=0, maximum=settings["frontier_max"])
    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_key = "discovery_requests_" + date_key
    used = store.get(day_key, 0)
    stats = {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0, "linked_added": 0}
    fetcher = fetcher or get_entity_json
    for i in range(cap):
        if used >= settings["daily_request_budget"]:
            stats["budget_exhausted"] = True
            break
        item = store.next_entity()
        if item is None:
            break
        qid = item["entity_id"]
        # Count before the request, including failures, to enforce the daily budget.
        used += 1
        store.put(day_key, used)
        stats["attempted"] += 1
        try:
            payload = fetcher(qid, contact_ua, settings["max_response_bytes"])
            text, links = parse_entity(payload, qid, settings["max_linked_entities_per_source"])
            if len(text) < 12:
                stats["skipped"] += 1
            else:
                source_url = f"https://www.wikidata.org/wiki/{qid}"
                topic = "wikidata:linked-entity"
                saved = store.save_doc(qid, text, source_url, "CC0-1.0", topic)
                if saved:
                    stats["saved"] += 1
                    # Cache source snapshot for transparency, not for indiscriminate re-use.
                    raw_path = store.root / "raw" / "wikidata"
                    raw_path.mkdir(parents=True, exist_ok=True)
                    tmp = raw_path / f"{qid}.json.tmp"
                    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    tmp.replace(raw_path / f"{qid}.json")
                else:
                    stats["skipped"] += 1
            store.mark_entity(qid, "done")
            for linked in links:
                if store.add_frontier(linked, item["depth"] + 1, settings["frontier_max"]):
                    stats["linked_added"] += 1
        except RateLimited as exc:
            # Don't hammer a service that has declined more requests.
            store.mark_entity(qid, "error", str(exc))
            stats["failed"] += 1
            stats["rate_limited"] = True
            break
        except (DiscoveryError, ValueError, KeyError) as exc:
            store.mark_entity(qid, "error", str(exc))
            stats["failed"] += 1
        if i != cap - 1:
            sleep_fn(float(settings["crawl_delay_seconds"]))
    store.log("discover", "completed", stats)
    return stats
