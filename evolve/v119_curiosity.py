"""BERHOT EVOLVE v1.19: bounded curiosity, source memory and self-study.

Network research is SOURCE LOOKUP, never neural knowledge. Eligible self-study
examples are mechanically generated from already rights-cleared local articles,
without any invented answers, with independent host partitions and provenance.
"""
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit
import hashlib
import json
import re
import time

from .web_discovery import canonical_url, fetch_page, extract, detect_declared_rights
from .open_corpus import archive_page
from .v118_core import _approved_text, _words, _sha, answer_from_sources, _qa_xy
from .web_lab import _canonical_bytes

VERSION = "source_grounded_curiosity_v119"
_REJECT_START = (
    "image ", "media ", "photo ", "video ", "caption ", "see also ",
    "read more", "click here", "advertisement", "copyright ", "references",
    "navigation ", "language ", "figure ", "skip to ", "contents ",
)
_REJECT_TEXT = ("ignore previous", "system prompt", "api key",
                "subscribe to", "cookie policy", "privacy policy")
_STOP = {"the", "a", "an", "of", "please", "tell", "me", "about",
         "what", "is", "are", "does", "do", "how", "why", "explain", "describe"}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _topic(question):
    """Support plain topics and elementary 'what is ...' queries, never pretend
    to handle open-ended reasoning questions with a one-page search."""
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 180:
        raise ValueError("A research question must be 1..180 characters")
    raw = question.strip().rstrip("?!؟. ").strip()
    raw = re.sub(r"^(?:(?:what is|what are|tell me about|describe|explain)\s+)",
                 "", raw, flags=re.I)
    raw = re.sub(r"^(?:the|a|an)\s+", "", raw, flags=re.I)
    if not re.fullmatch(r"[A-Za-z][A-Za-z\- ]{2,50}", raw):
        return None
    if not 1 <= len(raw.split()) <= 3:
        return None
    return raw.casefold()


def candidates(topic):
    # The first candidate is a subject-specific public-domain *research*
    # destination, not a blanket rights declaration. Other topics can use a
    # public encyclopedia article (CC BY-SA; never auto-admit to model training).
    specialized = {
        "moon": ["https://science.nasa.gov/moon/facts/"],
        "earth": ["https://science.nasa.gov/earth/facts/"],
        "sun": ["https://science.nasa.gov/sun/facts/"],
    }
    values = list(specialized.get(topic, []))
    slug = quote("_".join(word.capitalize() for word in topic.split()), safe="_")
    values.append("https://en.wikipedia.org/wiki/" + slug)
    return values[:2]


def _sentences(raw):
    return re.split(r"(?<=[.!?؟])\s+|\n+", raw)


def best_passage(text, topic, max_len=260):
    """Rank informative sentences, discourage captions and navigation.

    This is a lexical heuristic, not a semantic truth/QA verifier.
    """
    words = set(re.findall(r"[A-Za-z]+", topic.casefold()))
    if not words:
        return None
    scored = []
    for sentence in _sentences(text):
        s = " ".join(sentence.split()).strip()
        if not 45 <= len(s) <= 390 or len(s.split()) < 9:
            continue
        lower = s.casefold()
        if lower.startswith(_REJECT_START) or any(p in lower for p in _REJECT_TEXT):
            continue
        present = set(re.findall(r"[a-z]+", lower))
        overlap = present & words
        if not overlap:
            continue
        head = " ".join(s.casefold().split()[:12])
        definitional = bool(re.search(
            r"\b" + re.escape(topic) + r"\b.{0,32}\b(?:is|are|means)\b",
            lower))
        score = len(overlap) * 3 + (9 if definitional else 0) + (
            4 if topic in head else 0) + (
            2 if 80 <= len(s) <= 260 else 0) - (
            8 if "image:" in lower or "media " in lower else 0)
        scored.append((score, abs(len(s)-150), s))
    if not scored:
        return None
    scored.sort(key=lambda v: (-v[0], v[1], v[2]))
    return scored[0][2][:max_len].strip()


def _key(topic):
    return "v119_finding:" + _sha(topic.encode("utf-8"))


def cached(store, question):
    topic = _topic(question)
    return store.get(_key(topic)) if topic else None


def smart_lookup(store, question):
    """Return local approved quotes / recorded research, or an unresolved task."""
    topic = _topic(question)
    local = answer_from_sources(store, question)
    if local["sources"]:
        # Choose a complete, explanatory sentence rather than an image caption.
        ranked = []
        for source in local["sources"]:
            try:
                _src, text = _approved_text(store, source["url"])
            except ValueError:
                continue
            excerpt = best_passage(text, topic or question)
            if excerpt:
                ranked.append({"title": source["title"], "url": source["url"],
                               "excerpt": excerpt})
        if ranked:
            return {
                "mode": "auto_local_source_not_neural", "status": "answered",
                "answer": ranked[0]["excerpt"], "sources": ranked,
                "training_updated": False,
                "warning": "An exact passage from previously rights-approved saved text. No neural weights changed.",
            }
    saved = cached(store, question)
    if saved:
        if saved["status"] == "researched":
            return {
                "mode": "research_memory_not_neural", "status": "answered",
                "answer": saved["answer"], "sources": saved["sources"],
                "training_updated": False,
                "research_at_utc": saved["finished_at_utc"],
                "warning": (
                    "Previously fetched and cached source excerpt; not independently "
                    "verified as fact, not neural memory or neural training."
                ),
            }
        return {
            "mode": "curiosity", "status": saved["status"],
            "answer": saved["answer"],
            "sources": saved.get("sources", []),
            "training_updated": False,
        }
    if topic is None:
        return {
            "mode": "curiosity", "status": "unsupported",
            "answer": "I can investigate a short English subject such as 'moon' or 'What is gravity?'. I cannot yet autonomously solve arbitrary multi-step questions.",
            "sources": [], "training_updated": False,
        }
    return {
        "mode": "curiosity", "status": "needs_research",
        "answer": f"I don't have a usable source passage about {topic} yet. I can research permitted public pages and save what I find.",
        "topic": topic, "sources": [], "training_updated": False,
    }


def research(store, settings, question, fetcher=None, progress=None,
             sleep_fn=time.sleep):
    """Bounded at most two public HTTPS fetches, shared daily web budget.

    A fetched page may be archived with its observed license, which defaults
    to UNKNOWN and is NEVER a training grant. Cite short verbatim excerpts.
    """
    topic = _topic(question)
    if topic is None:
        return {"status": "unsupported_question", "question": str(question)[:180]}
    previous = cached(store, topic)
    if previous:
        return {
            "status": "cached_" + previous["status"],
            "topic": topic, "answer": previous["answer"],
            "sources": previous.get("sources", []),
        }
    used_key = "web_requests_" + time.strftime("%Y-%m-%d", time.gmtime())
    budget = max(0, int(settings.get("daily_request_budget", 60)))
    max_urls = candidates(topic)
    results, attempted, errors = [], 0, []
    for i, url in enumerate(max_urls):
        used = int(store.get(used_key, 0))
        if used >= budget:
            break
        store.put(used_key, used + 1)
        attempted += 1
        if progress:
            progress("research_source", i, len(max_urls), None)
        try:
            html = fetcher(url) if fetcher else fetch_page(url, settings)
            text, title, _links, noai = extract(html)
            passage = best_passage(text, topic)
            if not passage:
                errors.append({"url": url, "reason": "no_informative_passage"})
                continue
            rights, proof = detect_declared_rights(html, url)
            # Research archive only, unknown rights remain UNKNOWN. Avoid
            # overwriting an earlier explicit rights review for this URL.
            existing = store.db.execute(
                "SELECT 1 FROM web_corpus_sources WHERE url=?", (url,)
            ).fetchone()
            if not existing:
                archive_page(store, url, html, text, title=title, noai=noai,
                             declared_license=rights, rights_url=proof)
            results.append({
                "url": url, "title": title[:160] or topic.title(),
                "host": urlsplit(url).hostname, "excerpt": passage[:260],
                "source_rights": rights or "UNKNOWN",
                "publisher_noai": bool(noai),
                "training_admitted": bool(rights in ("CC0-1.0", "PUBLIC-DOMAIN")
                                          and not noai),
            })
        except Exception as exc:
            errors.append({"url": url, "reason": str(exc)[:200]})
        if fetcher is None and i < len(max_urls)-1 and settings.get("crawl_delay_seconds", 0):
            sleep_fn(min(5.0, float(settings["crawl_delay_seconds"])))
    if progress:
        progress("research_review", attempted, len(max_urls), None)
    if results:
        selected = results[0]
        record = {
            "status": "researched", "topic": topic,
            "answer": selected["excerpt"], "sources": results,
            "finished_at_utc": _utc(),
            "verification": "single_or_multiple_sources_not_truth_verified",
            "neural_weights_updated": False,
            "next_step": "Manual rights/source review if you want this newly archived text used to train neural weights.",
        }
        store.put(_key(topic), record)
        store.log("v119_research", "completed", {
            "topic": topic, "source_urls": [x["url"] for x in results],
            "source_count": len(results), "training_admitted": False,
            "network_requests": attempted,
        })
        return {"status": "researched", **record}
    # Do not cache budget/network failures indefinitely. Record cooldown to
    # prevent an uncontrolled retry loop and manual flood.
    reason = ("daily_budget_exhausted" if attempted == 0 or
              int(store.get(used_key, 0)) >= budget and attempted < len(max_urls)
              else "no_supported_passage")
    record = {
        "status": "unresolved", "topic": topic,
        "answer": ("I couldn't find a trustworthy enough excerpt in the two "
                   "permitted research candidates. No answer was invented."),
        "sources": [], "finished_at_utc": _utc(),
        "reason": reason, "error_details": errors[:2],
        "retry_after_epoch": time.time() + 3600,
        "neural_weights_updated": False,
    }
    store.put(_key(topic), record)
    store.log("v119_research", "unresolved", {
        "topic": topic, "reason": reason, "attempted": attempted})
    return {"status": "unresolved", **record}


def sentence_lessons(store, per_host=18, max_sources=150):
    """Generate *reading continuation* examples from already rights-cleared
    passages; not invented question-answer facts and not model-generated truth.

    The source host later decides train/val/test. No repeated/shuffled copies
    are allowed to cross split boundaries.
    """
    if not 2 <= int(per_host) <= 40:
        raise ValueError("per_host must be 2..40")
    rows = store.db.execute(
        "SELECT url,host FROM web_corpus_sources WHERE noai=0 AND "
        "rights_status IN ('CC0-1.0','PUBLIC-DOMAIN') ORDER BY url LIMIT ?",
        (int(max_sources),)).fetchall()
    old = store.get("v119_auto_lessons", []) or []
    seen = {x["fingerprint"] for x in old}
    # Per-host cap keeps one publisher from dominating the generated curriculum.
    host_counts = {}
    for item in old:
        host_counts[item["host"]] = host_counts.get(item["host"], 0) + 1
    new = []
    for row in rows:
        url = row["url"]
        source, text = _approved_text(store, url)
        host = source["host"]
        if host_counts.get(host, 0) >= per_host:
            continue
        for sentence in _sentences(text):
            s = " ".join(sentence.split()).strip()
            if not 64 <= len(s) <= 195:
                continue
            lower = s.casefold()
            if (lower.startswith(_REJECT_START) or
                any(tag in lower for tag in _REJECT_TEXT) or
                len(s.split()) < 12 or
                len(set(s.casefold().split())) < 9):
                continue
            words = s.split()
            # Predict the remaining literal source passage, not an invented
            # explanatory answer. This trains reading/continuation only.
            leading = " ".join(words[:min(7, max(4, len(words)//3))])
            tail = s[len(leading):].strip()
            q = "Continue: " + leading
            if not 15 <= len(tail) <= 180:
                continue
            item = {
                "question": q, "answer": tail, "url": url, "host": host,
                "source_sha256": source["sha256"], "created_at_utc": _utc(),
                "source_provenance": "auto_exact_source_continuation_not_verified_qa",
            }
            if _qa_xy(item, 256) is None:
                continue
            item["fingerprint"] = _sha(_canonical_bytes([
                q.casefold(), tail, url]))
            if item["fingerprint"] in seen:
                continue
            seen.add(item["fingerprint"])
            new.append(item)
            host_counts[host] = host_counts.get(host, 0) + 1
            if host_counts[host] >= per_host:
                break
    if new:
        store.put("v119_auto_lessons", old + new)
        store.log("v119_self_study", "created", {
            "new_examples": len(new), "source_hosts": sorted(
                set(x["host"] for x in new)),
            "training_updated": False,
        })
    return {
        "status": "generated" if new else "unchanged",
        "new_lessons": len(new), "total_source_derived_lessons": len(old)+len(new),
        "by_host": host_counts,
        "source_provenance": "approved_local_articles_exact_continuation",
        "neural_weights_updated": False,
        "warning": "Automatically constructed cloze/reading exercises, NOT independently fact-checked Q&A. Training needs source-host holdout and validation.",
    }


def inventory(store):
    rows = store.db.execute(
        "SELECT action,status,created_at FROM runs WHERE action IN "
        "('v119_research','v119_self_study','v118_qa_train') "
        "ORDER BY id DESC LIMIT 12").fetchall()
    return {
        "source_derived_lessons": len(store.get("v119_auto_lessons", []) or []),
        "research_attempts": int(store.db.execute(
            "SELECT COUNT(*) FROM runs WHERE action='v119_research'").fetchone()[0]),
        "recent_learning_events": [dict(x) for x in rows],
        "training_status_note": "Research memory and generated reading exercises are not neural weight changes.",
    }
