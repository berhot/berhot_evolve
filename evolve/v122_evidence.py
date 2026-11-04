"""BERHOT EVOLVE v1.22: read-only rights-aware evidence qualification.

A source's license metadata and a mechanically detected sentence do not
prove factual truth or legal training permission. Never approve sources here.
"""
from collections import Counter
import hashlib

from .v119_curiosity import _sentences
from .v120_grounded import extract_definition, title_question, title_evidence
from .web_lab import _within

_RIGHTS = ("CC0-1.0", "PUBLIC-DOMAIN")


def evidence_report(store, limit=80):
    """Audit saved article quality and candidate learning gaps, no writes/net.

    Research-only articles may be inspected for *counts*, but are never added
    to the approved dataset or quoted as licensed neural supervision.
    """
    if not 1 <= int(limit) <= 200:
        raise ValueError("limit must be between 1 and 200")
    rows = store.db.execute(
        "SELECT s.url,s.host,s.title,s.rights_status,s.noai,s.sha256,"
        "s.rights_evidence_url,t.clean_path,t.words FROM web_corpus_sources s "
        "JOIN web_corpus_texts t ON s.sha256=t.sha256 "
        "ORDER BY s.url LIMIT ?", (int(limit),)).fetchall()
    existing = store.get("v121_last_cycle") or {}
    readiness = existing.get("heldout_readiness") or {}
    expected = readiness.get("source_hosts") or {}
    by_host = {}
    entries = []
    for row in rows:
        host = row["host"]
        rights = row["rights_status"]
        approved = rights in _RIGHTS and not bool(row["noai"])
        file = _within(store.root / "web_corpus" / "clean",
                       row["clean_path"])
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Source SHA256 mismatch; refusing evidence preview")
        sentences = _sentences(data.decode("utf-8", errors="strict"))
        examples = []
        for sentence in sentences[:500]:
            topic = extract_definition(sentence)
            if topic and topic not in examples:
                examples.append(topic)
            if len(examples) >= 20:
                break
        heading = title_question(row["title"])
        if heading and heading not in examples and title_evidence(
                data.decode("utf-8", errors="strict"), heading):
            examples.append(heading)
        if bool(row["noai"]):
            eligibility = "excluded_publisher_noai"
        elif not approved:
            eligibility = "research_only_requires_article_rights_review"
        elif len(examples) == 0:
            eligibility = "approved_but_no_extracted_definitions"
        else:
            eligibility = "approved_source_has_heuristic_definitions"

        group = by_host.setdefault(host, {
            "host": host, "archived_articles": 0,
            "approved_articles": 0, "unknown_rights_articles": 0,
            "publisher_noai_articles": 0,
            "candidate_definition_topics": set(),
            "approved_definition_topics": set(),
        })
        group["archived_articles"] += 1
        group["approved_articles"] += int(approved)
        group["unknown_rights_articles"] += int(rights == "UNKNOWN")
        group["publisher_noai_articles"] += int(bool(row["noai"]))
        group["candidate_definition_topics"].update(examples)
        if approved:
            group["approved_definition_topics"].update(examples)
        entries.append({
            "url": row["url"], "title": row["title"], "host": host,
            "rights": rights, "noai": bool(row["noai"]),
            "rights_evidence_url": row["rights_evidence_url"],
            "words": int(row["words"]), "status": eligibility,
            "extracted_candidate_topics": examples[:12],
            "potential_qa_examples": len(examples),
            "candidate_not_human_fact_checked": True,
            "source_rights_eligible_for_qa_extraction": approved,
            "neural_training_admitted_by_this_report": False,
        })
    summary = []
    for host, row in sorted(by_host.items()):
        split = next((part for part, hosts in expected.items()
                      if host in hosts), None)
        summary.append({
            "host": host, "articles": row["archived_articles"],
            "rights_approved_articles": row["approved_articles"],
            "unknown_rights_articles": row["unknown_rights_articles"],
            "noai_articles": row["publisher_noai_articles"],
            "heuristic_topic_candidates": len(row["candidate_definition_topics"]),
            "approved_heuristic_topic_candidates":
                len(row["approved_definition_topics"]),
            "last_cycle_host_partition": split,
        })
    blocked = [e for e in entries
               if e["status"] != "approved_source_has_heuristic_definitions"]
    return {
        "status": "review_only", "sources_examined": len(rows),
        "hosts": summary, "sources": entries,
        "needs_manual_rights_review": [
            {"url": e["url"], "host": e["host"],
             "candidate_topics": e["extracted_candidate_topics"]}
            for e in entries
            if e["status"] == "research_only_requires_article_rights_review"
            and e["potential_qa_examples"] > 0
        ][:20],
        "approved_pages_without_definitions": [
            {"url": e["url"], "host": e["host"], "title": e["title"]}
            for e in entries
            if e["status"] == "approved_but_no_extracted_definitions"
        ][:20],
        "last_cycle_source_hosts": expected,
        "last_cycle_training_gap": {
            part: max(0, minimum - (readiness.get("counts") or {}).get(part, 0))
            for part, minimum in (("train", 4), ("val", 2), ("test", 2))
        } if readiness else None,
        "source_count_by_status": dict(Counter(
            e["status"] for e in entries)),
        "claims": {
            "source_data_modified": False,
            "rights_approved_automatically": False,
            "unknown_rights_examples_admitted": False,
            "unverified_extraction_is_truth": False,
            "last_cycle_partitions_are_current_guarantee": False,
        },
        "next_action": (
            "Review actual article rights for suitable existing unknown-rights "
            "sources; acquire additional independent publisher material when "
            "needed. A new host may change the deterministic source split."
        ),
    }
