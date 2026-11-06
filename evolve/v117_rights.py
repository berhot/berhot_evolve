"""v1.17.1 explicit per-article training-rights review for stored web archives.

Collector metadata is not proof of rights. Default remains UNKNOWN.
No automatic classification of government domains; the operator must read
the particular article and authoritative rights policy, and attest.
"""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json

from .web_lab import _within


ALLOWED = ("CC0-1.0", "PUBLIC-DOMAIN")


def sources(store):
    from .training_approval import operator_approvals, approved_for_training
    approvals = operator_approvals(store)
    rows = store.db.execute(
        "SELECT s.url,s.host,s.title,s.rights_status,s.rights_evidence_url,"
        "s.noai,s.sha256,t.words,t.characters FROM web_corpus_sources s "
        "JOIN web_corpus_texts t ON t.sha256=s.sha256 ORDER BY s.url"
    ).fetchall()
    return {
        "status": "listed",
        "sources": [
            {
                "url": r["url"], "host": r["host"], "title": r["title"],
                "rights_status": r["rights_status"],
                "rights_evidence_url": r["rights_evidence_url"],
                "publisher_noai": bool(r["noai"]),
                "clean_sha256": r["sha256"], "word_count": r["words"],
                "characters": r["characters"],
                "operator_training_approved": (r["url"], r["sha256"]) in approvals,
                "potentially_eligible": (
                    approved_for_training(r, approvals)
                    and int(r["words"]) >= 300),
            }
            for r in rows
        ],
        "note": "Publicly accessible does not imply public domain, CC0, or permission for model training.",
    }


def review_rights(store, url, license_id, evidence_url, confirmed=False):
    if not confirmed:
        raise ValueError("Explicit --confirm-manual-review is required")
    if license_id not in ALLOWED:
        raise ValueError("Only CC0-1.0 and PUBLIC-DOMAIN are supported")
    parsed = urlsplit(url)
    evidence = urlsplit(evidence_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
        parsed.password or parsed.fragment or
        evidence.scheme != "https" or not evidence.hostname or evidence.username or
        evidence.password or evidence.fragment):
        raise ValueError("Both article and evidence must be normal HTTPS URLs")
    if evidence_url == url:
        raise ValueError("Separate policy or rights evidence URL required")
    row = store.db.execute(
        "SELECT s.url,s.title,s.host,s.sha256,s.rights_status,s.noai,"
        "s.raw_html_sha256,s.rights_evidence_url,t.clean_path "
        "FROM web_corpus_sources s JOIN web_corpus_texts t "
        "ON s.sha256=t.sha256 WHERE s.url=?", (url,)
    ).fetchone()
    if row is None:
        return {"status": "skipped", "reason": "article_not_archived", "url": url}
    if row["noai"]:
        return {"status": "skipped", "reason": "publisher_noai_signal", "url": url}
    if row["rights_status"] != "UNKNOWN":
        return {"status": "skipped", "reason": "not_an_unknown_rights_record",
                "url": url, "existing_rights_status": row["rights_status"]}
    filepath = _within(store.root / "web_corpus" / "clean", row["clean_path"])
    if (not filepath.is_file() or
        hashlib.sha256(filepath.read_bytes()).hexdigest() != row["sha256"]):
        raise ValueError("Cleaned source text is missing or SHA256 mismatched")
    review = {
        "url": url, "host": row["host"], "title": row["title"],
        "clean_sha256": row["sha256"], "raw_html_sha256": row["raw_html_sha256"],
        "previous_rights_status": "UNKNOWN",
        "reviewed_rights": license_id, "rights_evidence_url": evidence_url,
        "reviewer_attested_per_page": True,
        "review_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "A human has attested to reviewing this specific article and the "
            "linked rights statement. The software does not verify ownership "
            "or legal applicability. Third-party works and usage restrictions "
            "must still be checked; no publisher no-AI signs may be ignored."
        ),
    }
    # Update rights, review history and provenance in ONE SQLite transaction.
    history = store.get("v117_rights_review_history", []) or []
    history.append(review)
    with store.db:
        cur = store.db.execute(
            "UPDATE web_corpus_sources SET rights_status=?,rights_evidence_url=? "
            "WHERE url=? AND sha256=? AND rights_status='UNKNOWN' AND noai=0",
            (license_id, evidence_url, url, row["sha256"]),
        )
        if cur.rowcount != 1:
            raise ValueError("Source rights record changed during review")
        store.db.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("v117_rights_review_history", json.dumps(history, ensure_ascii=False)))
        store.db.execute(
            "INSERT INTO runs(action,status,details,created_at) VALUES(?,?,?,?)",
            ("v117_manual_rights_review", "attested",
             json.dumps(review, ensure_ascii=False),
             review["review_at_utc"]))
    return {"status": "review_recorded", **review}
