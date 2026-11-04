"""BERHOT EVOLVE v1.21: bounded curiosity -> evidence -> questions -> neural ranker -> audit.

This is an inspectable autonomous *learning experiment*, NOT an LLM,
self-updating general intelligence, autonomous fact verifier or a permission
to treat arbitrary crawled text as neural-training material.

Each cycle chooses observed source-derived words; asks short questions;
optionally researches public HTTPS pages (existing robots/quota code);
creates SHA-locked, rights-cleared literal QA exercises; performs a source
integrity quiz; and trains the v1.20 PyTorch evidence ranker ONLY if strict
3-host train/validation/test conditions are met. No trust/rights shortcuts.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

from .v118_core import _approved_text
from .v119_curiosity import research as research_topic, budget_status
from .v120_grounded import (
    _inventory, grounded_answer, readiness, topic_from_question,
    title_question, train_ranker,
)

VERSION = "berhot-autonomous-curiosity-v121"
HISTORY = "v121_cycle_history"
_QUIZ_KEY = "v121_quiz_bank_latest"
_ATTEMPTS_KEY = "v121_network_attempts"
_RECENT_MAX = 80

# Narrow learner is studying scientific *concepts*, not organization names,
# vague adjectives or website navigation. A real noun phrase containing one
# of these words (e.g. "natural selection") remains eligible.
_NON_CONCEPT_TOPICS = {
    "nasa", "noaa", "usgs", "faq", "urls", "http", "https",
    "natural", "general", "latest", "recent", "global", "local",
    "national", "federal", "different", "available", "important",
    "official", "free", "new", "old", "updated", "several",
    "resources", "overview", "related", "about", "information",
}
_SINGULAR_ENDING_S = {
    "gas", "mass", "glass", "grass", "news", "physics",
    "mathematics", "statistics", "series", "species",
}


def question_for_topic(topic):
    """Conservative English singular/plural framing, not a POS tagger."""
    topic = _topic_ok(topic)
    if topic is None:
        return None
    last = topic.split()[-1]
    plural = (
        last.endswith("s") and
        not last.endswith(("ss", "us", "is")) and
        last not in _SINGULAR_ENDING_S
    )
    return ("What are " if plural else "What is ") + topic + "?"



def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _source_revision(store):
    rows = store.db.execute(
        "SELECT url,sha256,rights_status,noai FROM web_corpus_sources "
        "WHERE noai=0 AND rights_status IN ('CC0-1.0','PUBLIC-DOMAIN') "
        "ORDER BY url").fetchall()
    return _hash(_canonical([list(row) for row in rows]))


def _topic_ok(raw):
    # Only source-observed, short English nouns; no generative goal invention.
    try:
        topic = topic_from_question("What is " + str(raw).strip() + "?")
    except ValueError:
        return None
    if not topic or len(topic) > 48 or not re.fullmatch(
            r"[a-z][a-z\- ]{2,48}", topic):
        return None
    # Prevent names of UI elements and policy terms from becoming research tasks.
    if (topic in _NON_CONCEPT_TOPICS or
            topic in {"cookie policy", "privacy policy", "submit", "subscribe",
                      "sign in", "login", "menu", "latest news", "navigation"}):
        return None
    return topic


def curiosity_queue(store, limit=8):
    """Progress through observed concepts without revisiting completed research.

    Every candidate originates from archived metadata or an existing user
    research question; no LLM or random seed words. Completed source lookups
    cannot silently become training examples, and unresolved failures honor
    their cooldown. The query is read-only.
    """
    if not 1 <= int(limit) <= 20:
        raise ValueError("limit must be 1..20")
    from .v119_curiosity import cached
    import time

    known = {f["topic"] for f in _inventory(store)}
    seen, pending = set(), []
    deferred = {
        "already_has_approved_definition": 0,
        "previous_research_available": 0,
        "already_attempted_today": 0,
        "retry_cooldown_active": 0,
        "invalid_or_unsupported_topic": 0,
    }
    attempts = store.get(_ATTEMPTS_KEY, {}) or {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def add(term, url, origin, priority):
        topic = _topic_ok(term)
        if topic is None:
            deferred["invalid_or_unsupported_topic"] += 1
            return
        if topic in seen:
            return
        seen.add(topic)
        if topic in known:
            deferred["already_has_approved_definition"] += 1
            return
        previous = cached(store, topic)
        if previous and previous.get("status") == "researched":
            deferred["previous_research_available"] += 1
            return
        if today + ":" + topic in attempts:
            deferred["already_attempted_today"] += 1
            return
        if (previous and previous.get("status") == "unresolved" and
                (previous.get("reason") != "daily_budget_exhausted" or
                 (previous.get("finished_at_utc") or "")[:10] == today) and
                time.time() < float(previous.get("retry_after_epoch", float("inf")))):
            deferred["retry_cooldown_active"] += 1
            return
        if not isinstance(url, str) or not url.startswith("https://"):
            deferred["invalid_or_unsupported_topic"] += 1
            return
        pending.append({
            "topic": topic,
            "question": question_for_topic(topic),
            "observed_at": url,
            "origin": origin,
            "priority": round(float(priority), 3),
            "answer_known": False,
        })

    for row in store.db.execute(
        "SELECT concept,source_url FROM research_questions "
        "WHERE status='open' AND question LIKE 'What is %?' "
        "ORDER BY id LIMIT 300"):
        add(row["concept"], row["source_url"], "unanswered_saved_question", 5)
    for row in store.db.execute(
        "SELECT term,source_url,priority FROM concept_tasks "
        "WHERE status='pending' ORDER BY priority DESC,term LIMIT 300"):
        add(row["term"], row["source_url"],
            "concept_from_observed_page",
            3 + min(1, max(0, float(row["priority"])) / 100))
    for row in store.db.execute(
        "SELECT url,title FROM web_corpus_sources ORDER BY url LIMIT 200"):
        topic = title_question(row["title"])
        if topic:
            add(topic, row["url"], "question_from_article_title", 4)

    pending.sort(key=lambda row: (-row["priority"], row["topic"],
                                  row["observed_at"]))
    return {
        "status": "questions_available" if pending else "no_new_eligible_questions",
        "questions": pending[:int(limit)],
        "pending_distinct_topics": len(pending),
        "deferred_by_reason": deferred,
        "previously_researched_questions_are_not_reused": True,
        "network_requests_made": 0,
        "note": ("Previously fetched research is evidence memory, not model "
                 "training. Source-discovered terms remain unverified questions."),
    }


def source_questions(store, limit=8):
    return curiosity_queue(store, limit=limit)["questions"]


def _quiz_questions(topic):
    # These are verifiable *literal evidence-matching questions*, not
    # paraphrase labels checked by a language model.
    return ["What is " + topic + "?", "Define " + topic]


def build_lesson_bank(store, max_lessons=120, write=False):
    """Create immutable, cited question/quote examples from approved text.

    The exact same source-derived topic/definition data supplies the v1.20
    pairwise evidence-ranking trainer. No neural memorization is implied.
    """
    if not 1 <= int(max_lessons) <= 500:
        raise ValueError("max_lessons must be 1..500")
    lessons = []
    for fact in _inventory(store):
        source, text = _approved_text(store, fact["url"])
        if source["sha256"] != fact["source_sha256"]:
            raise ValueError("Source revision changed while creating lessons")
        if fact["answer"] not in text:
            raise ValueError("Expected source sentence not found in verified archive")
        for question in _quiz_questions(fact["topic"]):
            if topic_from_question(question) != fact["topic"]:
                raise ValueError("Question normalization drift")
            lessons.append({
                "question": question, "expected_quote": fact["answer"],
                "topic": fact["topic"], "url": fact["url"],
                "host": fact["host"], "source_sha256": fact["source_sha256"],
                "sentence_sha256": fact["sentence_sha256"],
                "provenance": fact["provenance"],
                "label_type": "mechanically_source_derived_not_fact_verified",
            })
            if len(lessons) >= int(max_lessons):
                break
        if len(lessons) >= int(max_lessons):
            break
    document = {
        "format": VERSION + "-quiz-v1",
        "revision_sha256": _source_revision(store),
        "source_host_count": len({f["host"] for f in lessons}),
        "rights_policy": "existing_archived_CC0_or_public_domain_no_noai",
        "excludes_unknown_rights": True,
        "questions": lessons,
    }
    digest = _hash(_canonical(document))
    output = {
        "status": "created" if lessons else "no_eligible_evidence",
        "questions_created": len(lessons),
        "unique_topics": len({f["topic"] for f in lessons}),
        "independent_source_hosts": document["source_host_count"],
        "manifest_sha256": digest,
        "revision_sha256": document["revision_sha256"],
        "not_independently_fact_verified": True,
        "new_neural_weights": False,
        "note": ("Source-based quiz labels. Trainer uses the same approved "
                 "definition inventory with strict source-host holdout. "
                 "The bank itself is not a fluent generative model."),
    }
    if write and lessons:
        directory = store.root / "learning" / "v121"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("quiz-" + digest + ".json")
        payload = _canonical(document)
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Immutable quiz bank hash collision or modification")
        if not path.exists():
            tmp = directory / ("." + digest + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
        output["manifest_path"] = str(path)
        store.put(_QUIZ_KEY, output)
    return output


def audit_quiz(store, maximum=24):
    """Ask existing evidence questions and validate literal provenance.

    This is retrieval integrity evaluation, NOT generalization or a true
    held-out model test; the same source library is available to lookup.
    """
    if not 1 <= int(maximum) <= 100:
        raise ValueError("maximum must be 1..100")
    distinct = {}
    for fact in _inventory(store):
        distinct.setdefault(fact["topic"], []).append(fact)
    successes, outcomes = 0, []
    for topic in sorted(distinct)[:int(maximum)]:
        q = "Define " + topic
        result = grounded_answer(store, q)
        sources = result.get("sources") or []
        try:
            proof = False
            if result.get("status") == "answered" and sources:
                selected = sources[0]
                approved, text = _approved_text(store, selected["url"])
                proof = (
                    approved["sha256"] == selected["source_sha256"] and
                    selected["excerpt"] == result["answer"] and
                    result["answer"] in text and
                    result["answer"] in {f["answer"] for f in distinct[topic]})
            successes += int(proof)
            outcomes.append({"question": q, "topic": topic,
                             "source_quote_verified": proof,
                             "source_url": sources[0]["url"] if sources else None})
        except (ValueError, OSError, KeyError):
            outcomes.append({"question": q, "topic": topic,
                             "source_quote_verified": False,
                             "source_url": None})
    return {
        "status": "tested" if outcomes else "no_questions",
        "evaluation_kind": "source_quote_integrity_NOT_neural_generalization",
        "tested": len(outcomes), "pass": successes,
        "pass_fraction": (round(successes / len(outcomes), 4)
                          if outcomes else None),
        "outcomes": outcomes,
        "independent_host_holdout": False,
        "proves_factual_understanding": False,
    }


def plan(store, settings, max_topics=6):
    progress = curiosity_queue(store, limit=max_topics)
    questions = progress["questions"]
    approved = _inventory(store)
    ready = readiness(store)
    return {
        "status": "planned", "version": VERSION,
        "curiosity_questions": questions,
        "curiosity_triage": {
            "status": progress["status"],
            "pending_distinct_topics": progress["pending_distinct_topics"],
            "deferred_by_reason": progress["deferred_by_reason"],
        },
        "observed_topics_already_with_definitions":
            sorted({row["topic"] for row in approved})[:50],
        "source_derived_definition_count": len(approved),
        "question_bank_preview": build_lesson_bank(store, write=False),
        "ranker_readiness": ready,
        "network_budget": budget_status(store, settings),
        "network_requests_made": 0, "neural_weights_updated": False,
        "rights_automatically_approved": False,
    }


def cycle(store, settings, network=False, max_topics=6, max_requests=6,
          steps=120, seed=31415, research_fn=None, progress=None):
    """Perform one bounded study cycle, optionally doing robots-compliant research.

    The only place new neural weights are calculated is train_ranker(), and
    only when independent hosts with qualifying evidence exist. Existing
    source rights records are never changed by this orchestrator.
    """
    if not 1 <= int(max_topics) <= 12:
        raise ValueError("max_topics must be 1..12")
    if not 0 <= int(max_requests) <= 24:
        raise ValueError("max_requests must be 0..24")
    if not 1 <= int(steps) <= 500:
        raise ValueError("steps must be 1..500")
    started = _utc()
    before = plan(store, settings, max_topics=max_topics)
    revision_before = _source_revision(store)
    questions = before["curiosity_questions"]
    quiz = build_lesson_bank(store, write=True)
    quiz_result = audit_quiz(store)
    work = []
    attempts = store.get(_ATTEMPTS_KEY, {}) or {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    remaining = int(max_requests) if network else 0
    for item in questions:
        topic = item["topic"]
        attempt_key = today + ":" + topic
        if not network or remaining <= 0:
            work.append({"question": item["question"],
                         "status": "planned_offline" if not network
                         else "network_limit_reached"})
            continue
        if attempt_key in attempts:
            work.append({"question": item["question"],
                         "status": "already_attempted_today"})
            continue
        before_requests = budget_status(store, settings)
        if before_requests["remaining"] <= 0:
            work.append({"question": item["question"],
                         "status": "shared_daily_budget_exhausted"})
            break
        # Limit the *existing* research engine's budget to the remaining
        # request allocation, without modifying the persistent budget counter.
        scoped_settings = dict(settings)
        scoped_settings["daily_request_budget"] = min(
            int(settings["daily_request_budget"]),
            before_requests["used"] + remaining)
        if progress:
            progress("curiosity_research", len(work), len(questions), None)
        # The stored source/candidate network remains subject to robots.txt,
        # TLS certificate verification, SSRF prevention and access restrictions.
        try:
            outcome = (
                research_fn(store, scoped_settings, item["question"])
                if research_fn is not None else
                research_topic(store, scoped_settings, item["question"]))
            state = outcome.get("status", "unknown")
            note = outcome.get("reason")
        except Exception as error:
            state = "error"
            note = (type(error).__name__ + ": " + str(error))[:180]
        after_requests = budget_status(store, settings)
        spent = max(0, after_requests["used"] - before_requests["used"])
        remaining = max(0, remaining - spent)
        attempts[attempt_key] = {
            "question": item["question"], "status": state,
            "requests_used": spent, "at_utc": _utc(),
        }
        work.append({
            "question": item["question"], "status": state,
            "requests_used": spent, "reason": note,
            "new_training_rights_granted_by_cycle": False,
        })
    if network:
        # Only retain a short finite budget history, never an infinite KV.
        attempts = {k: attempts[k] for k in sorted(attempts)[-300:]}
        store.put(_ATTEMPTS_KEY, attempts)
    after = readiness(store)
    train = {"status": "skipped", "reason": after["status"],
             "neural_weights_changed": False,
             "independent_heldout_test_available": False}
    # Freeze per-source rights revisions, not the research-only cache.
    revision_after = _source_revision(store)
    last_trained_revision = store.get("v121_last_ranker_revision")
    if after["status"] == "ready":
        if revision_after == last_trained_revision:
            train = {
                "status": "unchanged", "reason": "same_approved_data_as_last_attempt",
                "neural_weights_changed": False,
            }
        else:
            if progress:
                progress("neural_ranker_training", 0, int(steps), None)
            train = train_ranker(store, settings, steps=steps, seed=seed,
                                 progress=progress)
            store.put("v121_last_ranker_revision", revision_after)
    existing = store.get("v120_ranker_latest")
    summary = {
        "status": "completed" if train.get("status") in ("trained", "unchanged")
                  else "completed_with_training_blocker",
        "version": VERSION,
        "started_at_utc": started, "finished_at_utc": _utc(),
        "network_opt_in": bool(network),
        "curiosity_question_count": len(questions),
        "curiosity_questions": questions,
        "curiosity_triage": before["curiosity_triage"],
        "research_outcomes": work,
        "new_requests_this_cycle": sum(x.get("requests_used", 0) for x in work),
        "source_derived_quiz_bank": quiz,
        "local_quiz_audit": quiz_result,
        "neural_training": train,
        "heldout_readiness": after,
        "approved_corpus_revision_before": revision_before,
        "approved_corpus_revision_after": revision_after,
        "accepted_neural_ranker_sha256":
            (existing or {}).get("sha256"),
        "new_neural_weights_trained":
            train.get("status") == "trained",
        "accepted_neural_weights":
            bool(train.get("accepted_by_validation", False)),
        "new_public_research_is_auto_authorized_for_training": False,
        "understanding_proven": False,
        "scientific_fact_truth_independently_verified": False,
        "warning": (
            "A local exact-source quiz does not test independent understanding; "
            "a neural ranking experiment does not generate or understand text. "
            "A proper neural test requires held-out publisher evidence. "
            "Research-only web pages are not automatic model training rights."
        ),
    }
    history = store.get(HISTORY, []) or []
    history.append({
        "started_at_utc": summary["started_at_utc"],
        "status": summary["status"],
        "question_count": summary["curiosity_question_count"],
        "requests_used": summary["new_requests_this_cycle"],
        "quiz_pass": summary["local_quiz_audit"]["pass"],
        "quiz_tested": summary["local_quiz_audit"]["tested"],
        "training_status": train.get("status"),
        "accepted_neural_weights": summary["accepted_neural_weights"],
        "approved_revision": revision_after,
    })
    store.put(HISTORY, history[-_RECENT_MAX:])
    store.put("v121_last_cycle", summary)
    store.log("v121_curiosity_cycle", summary["status"], {
        "question_count": len(questions),
        "requests_used": summary["new_requests_this_cycle"],
        "training_status": train.get("status"),
        "source_quiz_pass": quiz_result["pass"],
        "source_quiz_tested": quiz_result["tested"]})
    return summary


def history(store):
    return {
        "status": "available",
        "recent_runs": store.get(HISTORY, []) or [],
        "last_cycle": store.get("v121_last_cycle"),
        "latest_quiz_manifest": store.get(_QUIZ_KEY),
        "warning": "Neural learned knowledge cannot be inferred from source retrieval.",
    }


def learning_report(store):
    """Compact and honest progress dashboard for the last actual local cycle.

    This is read-only and cannot consume research requests or train weights.
    """
    latest = store.get("v121_last_cycle")
    if not latest:
        return {
            "status": "no_cycle_yet",
            "next_command": "python3 -m evolve v121-cycle --network --max-topics 6 --max-requests 6 --steps 120",
            "new_neural_weights": False,
        }
    evaluation = latest.get("local_quiz_audit") or {}
    training = latest.get("neural_training") or {}
    # Use the *last completed cycle's* readiness snapshot. Recomputing
    # readiness would call freeze_dataset() and write files, violating this
    # command's read-only contract.
    ready = latest.get("heldout_readiness") or {}
    missing = {}
    for split in ("train", "val", "test"):
        count = (ready.get("counts") or {}).get(split, 0)
        topics = (ready.get("unique_topics") or {}).get(split, 0)
        missing[split] = {
            "examples": max(0, {"train": 4, "val": 2, "test": 2}[split] - count),
            "distinct_topics": max(0, 2 - topics),
        }
    results = latest.get("research_outcomes") or []
    return {
        "status": latest.get("status"),
        "cycle_utc": latest.get("started_at_utc"),
        "curiosity_questions": [
            item.get("question") for item in
            (latest.get("curiosity_questions") or [])[:12]],
        "curiosity_triage": latest.get("curiosity_triage"),
        "research_results": [
            {"question": item.get("question"),
             "status": item.get("status"),
             "requests_used": item.get("requests_used", 0),
             "reason": item.get("reason")}
            for item in results[:12]
        ],
        "network_requests_in_last_cycle":
            latest.get("new_requests_this_cycle", 0),
        "source_lessons": (latest.get("source_derived_quiz_bank") or {}).get(
            "questions_created", 0),
        "source_quiz": {
            "passed": evaluation.get("pass"),
            "tested": evaluation.get("tested"),
            "meaning": "exact saved-source quote and SHA integrity; NOT general knowledge",
        },
        "ranker": {
            "training_status": training.get("status"),
            "reason": training.get("reason"),
            "candidate_weights_trained":
                bool(latest.get("new_neural_weights_trained")),
            "accepted_after_validation":
                bool(latest.get("accepted_neural_weights")),
            "heldout_test": training.get("test_after"),
            "neural_vs_lexical_heldout_benchmark":
                training.get("benchmark"),
            "outperformed_lexical_on_test":
                (training.get("benchmark") or {}).get(
                    "outperformed_lexical_on_unseen_test_hosts"),
            "requires_three_independent_host_partitions": True,
        },
        "missing_to_train_at_last_cycle": missing,
        "source_hosts_by_split_at_last_cycle": ready.get("source_hosts", {}),
        "understanding_proven": False,
        "new_web_research_automatically_training_eligible": False,
        "next_action": (
            "Acquire and individually rights-review usable science definitions "
            "from at least one independent test-source host; rerun readiness."
            if ready.get("status") != "ready" else
            "Read heldout test and validation metrics; inspect incorrect and unanswerable queries."
        ),
    }
