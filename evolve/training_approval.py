"""Operator training decisions bound to exact archived snapshots, not licenses."""
from datetime import datetime, timezone
import hashlib
import json

POLICY_KEY = "corpus_future_training_approval"


def future_approval_policy(store):
    return store.get(POLICY_KEY, {}) or {}


def approve_future_snapshot(store, url, clean_sha256, noai, when):
    """Called inside the archive transaction after immutable text is written."""
    policy = future_approval_policy(store)
    if not policy.get("enabled") or noai:
        return False
    store.db.execute(
        "INSERT OR IGNORE INTO corpus_training_approvals "
        "(url,clean_sha256,approved_at_utc,approval_reason) VALUES (?,?,?,?)",
        (url, clean_sha256, when, "Standing operator approval: " + policy["reason"]))
    return True


def training_condition(alias="web_corpus_sources"):
    """Internal SQL predicate shared by corpus consumers; alias is code-owned."""
    return (
        f"{alias}.noai=0 AND ({alias}.rights_status IN ('CC0-1.0','PUBLIC-DOMAIN') "
        "OR EXISTS (SELECT 1 FROM corpus_training_approvals AS approval "
        f"WHERE approval.url={alias}.url AND approval.clean_sha256={alias}.sha256))"
    )


def operator_approvals(store):
    return {(row["url"], row["clean_sha256"]) for row in store.db.execute(
        "SELECT url,clean_sha256 FROM corpus_training_approvals")}


def approved_for_training(row, approvals):
    return bool(row and not row["noai"] and (
        row["rights_status"] in ("CC0-1.0", "PUBLIC-DOMAIN") or
        (row["url"], row["sha256"]) in approvals))


def approve_current_corpus(store, reason, include_future=None):
    """Apply explicit operator approval to existing, verified snapshots only.

    Future approval requires an explicit persistent operator policy.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A recorded operator approval reason is required")
    if include_future is not None and type(include_future) is not bool:
        raise ValueError("include_future must be a boolean")
    from .web_lab import _within
    rows = store.db.execute(
        "SELECT s.url,s.sha256,s.noai,t.clean_path FROM web_corpus_sources s "
        "JOIN web_corpus_texts t ON t.sha256=s.sha256 ORDER BY s.url").fetchall()
    selected = []
    for row in rows:
        if row["noai"]:
            continue
        path = _within(store.root / "web_corpus" / "clean", row["clean_path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Missing or changed archive: " + row["url"])
        selected.append(row)
    now = datetime.now(timezone.utc).isoformat()
    with store.db:
        before = store.db.total_changes
        store.db.executemany(
            "INSERT OR IGNORE INTO corpus_training_approvals "
            "(url,clean_sha256,approved_at_utc,approval_reason) VALUES (?,?,?,?)",
            [(r["url"], r["sha256"], now, reason.strip()) for r in selected])
        added = store.db.total_changes - before
        policy = future_approval_policy(store)
        if include_future is not None:
            policy = {"enabled": include_future, "reason": reason.strip(),
                      "configured_at_utc": now}
            store.db.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (POLICY_KEY, json.dumps(policy)))
        result = {"status": "approved", "approved_current_sources": len(selected),
                  "new_approvals": added, "publisher_noai_skipped": len(rows)-len(selected),
                  "future_sources_auto_approved": bool(policy.get("enabled")),
                  "publisher_license_metadata_changed": False}
        store.db.execute(
            "INSERT INTO runs(action,status,details,created_at) VALUES(?,?,?,?)",
            ("corpus_operator_training_approval", "approved",
             json.dumps(dict(result, reason=reason.strip())), now))
    return result
