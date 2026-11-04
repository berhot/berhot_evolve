"""Read-only triage of the existing source-backed research queue.

A diagnostics command must never fetch a page, reset a budget, increment
attempts, change question status or manufacture training rights.
"""
from datetime import datetime, timedelta, timezone
from collections import Counter
from urllib.parse import urlsplit
import re
import time


def _utc(when):
    return datetime.fromtimestamp(float(when), timezone.utc).isoformat()


def research_health(store, settings, now=None):
    now = time.time() if now is None else float(now)
    utc = datetime.fromtimestamp(now, timezone.utc)
    day = utc.strftime("%Y-%m-%d")
    used = int(store.get(f"web_requests_{day}", 0))
    daily_limit = int(settings.get("daily_request_budget", 60))
    if daily_limit < 0:
        raise ValueError("Daily request budget cannot be negative")
    cooldown = max(0, int(settings.get("research", {}).get(
        "retry_cooldown_seconds", 3600)))
    rows = store.db.execute(
        "SELECT id, question, source_url, attempts, last_attempt_at "
        "FROM research_questions WHERE status = 'open' ORDER BY id"
    ).fetchall()
    supported = []
    unsupported_count = 0
    ready = 0
    on_cooldown = 0
    next_retry = None
    hosts = Counter()
    for row in rows:
        if not re.fullmatch(r"What is (.+)\?", row["question"], flags=re.I):
            unsupported_count += 1
            continue
        supported.append(row)
        host = urlsplit(row["source_url"]).hostname or "(invalid URL)"
        hosts[host] += 1
        last = row["last_attempt_at"]
        if last is None or now >= float(last) + cooldown:
            ready += 1
        else:
            on_cooldown += 1
            due = float(last) + cooldown
            next_retry = min(due, next_retry) if next_retry is not None else due
    left = max(0, daily_limit - used)
    next_midnight = datetime(
        utc.year, utc.month, utc.day, tzinfo=timezone.utc) + timedelta(days=1)
    reason = (
        "daily_budget_exhausted" if left == 0 else
        "retry_cooldown_active" if ready == 0 and on_cooldown > 0 else
        "no_supported_open_questions" if ready == 0 else
        "ready"
    )
    return {
        "status": "ready" if reason == "ready" else "blocked",
        "reason": reason,
        "checked_at_utc": utc.isoformat(),
        "request_budget": {
            "used": used, "limit": daily_limit, "remaining": left,
            "reset_at_utc": next_midnight.isoformat(),
            "scope": "shared_with_web_discovery",
        },
        "question_queue": {
            "open_total": len(rows),
            "supported_definition_questions": len(supported),
            "unsupported_open_questions": unsupported_count,
            "ready_after_cooldown": ready,
            "in_cooldown": on_cooldown,
            "next_retry_at_utc": _utc(next_retry) if next_retry is not None else None,
        },
        "source_hosts_in_supported_queue": dict(hosts.most_common(8)),
        "training_admission": {
            "verified_answers": int(store.db.execute(
                "SELECT COUNT(*) FROM verified_answers").fetchone()[0]),
            "verified_reusable_answers": int(store.db.execute(
                "SELECT COUNT(*) FROM verified_answers "
                "WHERE training_eligible=1").fetchone()[0]),
            "verified_definition_training_documents": int(store.db.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE topic='verified:definition' AND split='train' "
                "AND license='SELF-GENERATED-VERIFIED'").fetchone()[0]),
        },
        "action": (
            "Do not retry or raise the quota to circumvent limits; request budget "
            "resets on the next UTC day."
            if reason == "daily_budget_exhausted" else
            "No supported questions are eligible until their retry cooldown ends."
            if reason == "retry_cooldown_active" else
            "Gather additional supported, source-backed definition questions."
            if reason == "no_supported_open_questions" else
            "Use a small research batch, then inspect qualification outcomes."
        ),
        "note": (
            "Queue counts and stored URLs only. No fetches or database writes. "
            "A discovered page does not establish independent corroboration or "
            "reusable training rights."
        ),
    }
