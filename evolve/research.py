"""Bounded source-backed fact research and cautious corroboration.

This is a lexical, English-definition verifier. Two hosts agreeing does not
establish scientific truth; the audit trail and limitations are explicit.
Untrusted web data is never executed or treated as system instructions.
"""
from collections import Counter
from urllib.parse import quote, urlsplit
import hashlib
import re
import time

from .web_discovery import (
    canonical_url, detect_declared_rights, extract, fetch_page, WebSourceError,
)

_DEFINITION_REJECT = (
    "javascript", "ignore previous", "system prompt", "instructions:",
    "click here", "subscribe", "password", "apikey", "api key",
)
_STOP = set(
    "a an the of to for in on by with at and or as is are was were "
    "it they this that from which who how used use means refers "
    "part consists includes can be".split()
)
_RESEARCH_PATTERNS = (r"^What is (.+)\?$",)


def _tokens(text):
    return set(w for w in re.findall(r"[a-z]{3,24}", text.casefold()) if w not in _STOP)


def extract_definition(text, concept):
    """Return a concise definition clause only if explicitly stated in text."""
    term = concept.strip().casefold()
    if not re.fullmatch(r"[a-z]{3,28}(?: [a-z]{3,28})?", term):
        return None
    pattern = (r"\b(?:an?\s+|the\s+)?" + re.escape(term) +
               r"\s+(?:is|are|means|refers to)\s+([^.!?;\n]{18,200})[.!?;]")
    matches = re.finditer(pattern, text, flags=re.I)
    for match in matches:
        clause = " ".join(match.group(1).split()).strip(" ,:-")
        if len(clause) < 18 or any(marker in clause.casefold() for marker in _DEFINITION_REJECT):
            continue
        # Reject obvious HTML instruction-like data, numbers without context,
        # and fragments without enough informational words to compare.
        if len(_tokens(clause)) >= 4:
            return clause
    return None


def _same_claim(clause_a, clause_b):
    a, b = _tokens(clause_a), _tokens(clause_b)
    if min(len(a), len(b)) < 4:
        return False
    shared = len(a & b)
    if shared < 3 or shared / len(a | b) < 0.55:
        return False
    # Numeric and explicit negation disagreements require further review.
    numbers_a = set(re.findall(r"\b\d+(?:\.\d+)?\b", clause_a))
    numbers_b = set(re.findall(r"\b\d+(?:\.\d+)?\b", clause_b))
    if numbers_a != numbers_b:
        return False
    not_a = bool(re.search(r"\b(?:no|not|never|without)\b", clause_a, re.I))
    not_b = bool(re.search(r"\b(?:no|not|never|without)\b", clause_b, re.I))
    return not_a == not_b


def _publisher_hint(host):
    """Conservative grouping of common subdomains; not a formal publisher ID."""
    parts = host.casefold().split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac", "edu"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) > 1 else host


def candidate_urls(store, concept, source_url, cross_host=False):
    """Prioritize observed evidence, then one public encyclopedia candidate.

    No search-engine scraping, paywall avoidance, or unbounded URL discovery.
    """
    seen = set()
    urls = []
    def add(raw):
        try:
            url = canonical_url(raw)
        except WebSourceError:
            return
        if url not in seen:
            seen.add(url)
            urls.append(url)
    add(source_url)
    for item in store.concept_sources(concept):
        add(item["url"])
    if cross_host:
        wiki_title = quote(concept.replace(" ", "_").title(), safe="_")
        add("https://en.wikipedia.org/wiki/" + wiki_title)
    return urls


def research_questions(store, settings, limit=None, fetcher=None, sleep_fn=time.sleep):
    """Fetch sources, corroborate strict definitions, persist citeable answers.

    Web budget is shared with discover_web through web_requests_YYYY-MM-DD.
    Question attempts have a cooldown to avoid repeatedly pounding failures.
    """
    cfg = settings.get("research", {})
    limit = int(limit if limit is not None else cfg.get("questions_per_cycle", 5))
    if not 1 <= limit <= 50:
        raise ValueError("Question budget must be between 1 and 50")
    max_pages = max(2, min(int(cfg.get("max_pages_per_question", 4)), 8))
    cooldown = max(0, int(cfg.get("retry_cooldown_seconds", 3600)))
    selected = store.next_open_questions(limit, cooldown_seconds=cooldown)
    profile = store.get("web_source_profile", {})
    cross_host = bool(profile.get("cross_host", False))
    report = {"attempted_questions": 0, "verified_answers": 0,
              "training_examples_created": 0, "unresolved_questions": 0,
              "unsupported_questions": 0, "source_requests": 0,
              "source_errors": [], "answers": []}
    cache = {}
    used_key = "web_requests_" + time.strftime("%Y-%m-%d", time.gmtime())
    daily = int(settings.get("daily_request_budget", 60))
    for question in selected:
        question_id = question["id"]
        store.mark_question_attempt(question_id)
        report["attempted_questions"] += 1
        match = re.fullmatch(r"What is (.+)\?", question["question"], re.I)
        if not match:
            report["unsupported_questions"] += 1
            continue
        concept = match.group(1).casefold()
        candidates = []
        sources = candidate_urls(store, concept, question["source_url"], cross_host)
        for url in sources[:max_pages]:
            if url not in cache:
                consumed = int(store.get(used_key, 0))
                if consumed >= daily:
                    break
                store.put(used_key, consumed + 1)
                report["source_requests"] += 1
                try:
                    html = fetcher(url) if fetcher is not None else fetch_page(url, settings)
                    text, _title, _links, noai = extract(html)
                    license_id, rights_url = detect_declared_rights(html, url)
                    cache[url] = (text, noai, license_id, rights_url)
                except Exception as exc:
                    cache[url] = None
                    if len(report["source_errors"]) < 10:
                        report["source_errors"].append({"url": url, "reason": str(exc)[:220]})
                if fetcher is None and settings.get("crawl_delay_seconds", 0):
                    sleep_fn(float(settings["crawl_delay_seconds"]))
            evidence = cache.get(url)
            if evidence is None:
                continue
            text, noai, license_id, rights_url = evidence
            clause = extract_definition(text, concept)
            if clause is None:
                continue
            host = urlsplit(url).hostname
            candidates.append({
                "url": url, "publisher": _publisher_hint(host), "definition": clause,
                "license": license_id if not noai else None,
                "rights_url": rights_url if not noai else None,
            })
        matched = None
        for i, first in enumerate(candidates):
            for second in candidates[i + 1:]:
                if first["publisher"] == second["publisher"]:
                    continue
                if _same_claim(first["definition"], second["definition"]):
                    matched = (first, second)
                    break
            if matched:
                break
        if not matched:
            report["unresolved_questions"] += 1
            continue
        first, second = matched
        licensed = next((x for x in matched if x["license"] in ("CC0-1.0", "PUBLIC-DOMAIN")), None)
        # Never copy unlicensed material into the neural corpus. The answer
        # uses only an independently corroborated, explicitly reusable clause.
        chosen = licensed or first
        answer = f"{concept.capitalize()} is {chosen['definition']}."
        proof = [{
            "url": item["url"], "publisher": item["publisher"],
            "definition_sha256": hashlib.sha256(item["definition"].encode()).hexdigest(),
            "license": item["license"], "rights_url": item["rights_url"],
        } for item in matched]
        if store.save_verified_answer(question_id, concept, answer, proof,
                                      method="two_source_definition_overlap_v1",
                                      training_eligible=bool(licensed)):
            report["verified_answers"] += 1
            if licensed:
                # One compact example per independently corroborated fact.
                # No raw/research-only page content is admitted to training.
                lesson = f"Question: {question['question']}\nAnswer: {answer}\n"
                if store.save_doc(f"verified-definition:{question_id}", lesson,
                                  licensed["url"], "SELF-GENERATED-VERIFIED",
                                  topic="verified:definition", modality="text"):
                    report["training_examples_created"] += 1
            report["answers"].append({
                "concept": concept, "answer": answer,
                "source_urls": [item["url"] for item in matched],
                "training_eligible": bool(licensed),
            })
    report["open_questions"] = store.open_questions_count()
    return report
