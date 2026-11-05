"""BERHOT EVOLVE 1.26: read-only, SHA-grounded evidence expansion triage.

Never grant rights, train, edit a corpus, or promote heuristic science claims.
Prioritize already archived *source text* for human licensing and fact review.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import re

from .v119_curiosity import _sentences
from .v120_grounded import (
    _inventory, _navigation_fragment, _term_forms, extract_definition,
    title_evidence, title_question,
)
from .web_lab import _within

_APPROVED = frozenset(("PUBLIC-DOMAIN", "CC0-1.0"))
# These are structural/web-writing labels, not standalone scientific topics.
_SUSPECT = frozenset((
    "currents tutorial", "estuaries tutorial", "science explorer",
    "learn more", "information", "news", "facts", "privacy policy",
    "active links", "page", "home", "tutorial", "read more",
))


def _review_type(topic):
    low = topic.casefold()
    if (low in _SUSPECT or low.endswith(" tutorial") or
            low.startswith(("page ", "navigation ", "click "))):
        return "suspected_page_heading_or_navigation"
    return "unreviewed_explicit_definition"


def _quotes_from_text(text, title, max_candidates=16):
    """Only exact contiguous candidates; headings alone add NO quotation."""
    quotes, seen = [], set()
    for sentence in _sentences(text)[:500]:
        quote = " ".join(sentence.split()).strip()
        if not quote or _navigation_fragment(quote):
            continue
        topic = extract_definition(quote)
        if not topic or (topic, quote) in seen:
            continue
        seen.add((topic, quote))
        quotes.append((topic, quote, "sentence_definitional_heuristic"))
        if len(quotes) >= max_candidates:
            return quotes
    title_topic = title_question(title)
    if title_topic and len(quotes) < max_candidates:
        quote = title_evidence(text, title_topic)
        if quote and (title_topic, quote) not in seen:
            quotes.append((title_topic, quote,
                           "question_title_and_subject_anchored_quote"))
    return quotes


def expansion_report(store, host=None, limit=80, quote_limit=3):
    """Rank existing unreviewed archives by actual target-topic coverage.

    Topics and quote snippets may be inspected without legally asserting a
    training license. All bytes come from SHA-verified local archives.
    """
    if not 1 <= int(limit) <= 200:
        raise ValueError("limit must be 1..200")
    if not 1 <= int(quote_limit) <= 10:
        raise ValueError("quote-limit must be 1..10")
    if host is not None:
        host = host.strip().casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{2,251}", host):
            raise ValueError("host must be an exact public hostname")
    approved_facts = _inventory(store)
    existing_topics = {f["topic"] for f in approved_facts}
    existing_hosts = defaultdict(set)
    for item in approved_facts:
        existing_hosts[item["topic"]].add(item["host"])

    sql = (
        "SELECT s.url,s.host,s.title,s.rights_status,s.rights_evidence_url,"
        "s.noai,s.sha256,t.clean_path,t.words FROM web_corpus_sources s "
        "JOIN web_corpus_texts t ON s.sha256=t.sha256 "
    )
    params = []
    if host:
        sql += "WHERE s.host=? "
        params.append(host)
    sql += "ORDER BY s.url LIMIT ?"
    params.append(int(limit))
    rows = store.db.execute(sql, params).fetchall()

    entries = []
    noai_count, unknown_count, approved_count = 0, 0, 0
    potential_same_topic_sources = 0
    for row in rows:
        if row["noai"]:
            noai_count += 1
            continue
        is_approved = row["rights_status"] in _APPROVED
        approved_count += int(is_approved)
        unknown_count += int(row["rights_status"] == "UNKNOWN")
        # No rights promotion. Reading clean bytes here is diagnostic only.
        archived = _within(
            store.root / "web_corpus" / "clean", row["clean_path"])
        data = archived.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(
                "Archived source SHA256 mismatch; refusing source review")
        content = data.decode("utf-8", errors="strict")
        extracted = _quotes_from_text(content, row["title"])
        overlaps = sorted({t for t, _q, _kind in extracted if t in existing_topics})
        new_topics = sorted({t for t, _q, _kind in extracted
                             if t not in existing_topics})
        meaningful = [t for t in new_topics if _review_type(t) !=
                      "suspected_page_heading_or_navigation"]
        suspect = [t for t in new_topics if _review_type(t) ==
                   "suspected_page_heading_or_navigation"]
        # Only give positive labels for actual quote identity, never title
        # similarity. Even overlaps are NOT independent confirmed facts.
        same_topic_new_hosts = sorted(
            {t for t in overlaps if row["host"] not in existing_hosts[t]})
        if overlaps and not is_approved:
            potential_same_topic_sources += 1
        if overlaps:
            reason = "same_topic_candidate_needs_evidence_review"
        elif meaningful:
            reason = "new_topic_candidate_needs_evidence_review"
        elif suspect:
            reason = "suspected_navigation_heading_not_a_science_definition"
        else:
            reason = "no_explicit_definition_candidate"
        samples = []
        for topic, quote, method in extracted:
            if len(samples) >= int(quote_limit):
                break
            samples.append({
                "topic": topic, "exact_quote": quote,
                "candidate_method": method,
                "quote_sha256": hashlib.sha256(
                    quote.casefold().encode("utf-8")).hexdigest(),
                "quote_contained_in_archived_text": quote in content,
                "requires_human_fact_review": True,
                "suspected_structural_topic": _review_type(topic) ==
                    "suspected_page_heading_or_navigation",
            })
        entries.append({
            "url": row["url"], "host": row["host"],
            "title": row["title"], "words": int(row["words"]),
            "rights": row["rights_status"],
            "rights_evidence_url": row["rights_evidence_url"],
            "rights_approved_for_training": is_approved,
            "article_noai": False,
            "candidate_quotes": len(extracted),
            "topics_already_present": overlaps,
            "topics_not_yet_present": meaningful,
            "suspected_navigation_topics": suspect,
            "topics_from_other_independent_hosts": same_topic_new_hosts,
            "triage_reason": reason,
            "exact_source_preview": samples,
            "does_not_approve_training_rights": True,
        })

    # Priority is a queue for HUMAN inspection, not a quality/relevance score.
    # Existing, approved facts are separate; only rank unknown-rights pages
    # when we are recommending licensing review.
    unknown = [e for e in entries if e["rights"] == "UNKNOWN"]
    ranking = {
        "same_topic_candidate_needs_evidence_review": 0,
        "new_topic_candidate_needs_evidence_review": 1,
        "suspected_navigation_heading_not_a_science_definition": 2,
        "no_explicit_definition_candidate": 3,
    }
    unknown.sort(key=lambda e: (ranking[e["triage_reason"]],
                                e["host"], e["url"]))
    approved = [e for e in entries if e["rights_approved_for_training"]]
    return {
        "status": "research_review_only_no_training",
        "filtered_host": host,
        "sources_examined": len(rows),
        "approved_science_topics_before": len(existing_topics),
        "approved_science_quotes_before": len(approved_facts),
        "unknown_rights_sources_examined": unknown_count,
        "noai_sources_excluded": noai_count,
        "already_approved_sources_examined": approved_count,
        "unapproved_sources_with_same_topic_candidates":
            potential_same_topic_sources,
        "queue_to_review": unknown,
        "already_approved_diagnostics": approved[:20],
        "reviewed_training_rights_automatically": False,
        "new_training_pairs_created": 0,
        "model_weights_updated": False,
        "future_test_split_reassignment_possible": True,
        "note": (
            "Only inspect sources with explicit real text quotes. An exact "
            "quotation, a publisher title or government hostname never "
            "establishes factual correctness, public-domain licensing, "
            "permission for model training, or an independent evaluation. "
            "Review each article and rights statement separately. "
            "Wikipedia/other open licenses are not automatically compatible "
            "with the limited CC0/PUBLIC-DOMAIN admission policy."
        ),
        "next_action": (
            "Review the top archived candidates' actual quote and rights "
            "statement. Skip page-heading noise; then re-run v125-pairs "
            "to see whether distinct same-topic evidence really increased."
        ),
    }
