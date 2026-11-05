"""EVOLVE v1.25 — explicitly HUMAN-reviewed same-topic ranking diagnostics.

Operator labels never enter the v120 neural training or validation dataset.
The current source inventory is tiny, often a single quote per topic. This
workflow makes the deficiency visible, not hidden behind synthetic accuracy.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re

from .v120_grounded import (
    _accepted_model, _features, _identity_priority, _inventory,
    _lexical_score, topic_from_question,
)

_KEY = "v125_manual_pair_judgments"
_SHA = re.compile(r"^[a-f0-9]{64}$")
_SCOPE = "development_diagnostic_not_blind"


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _key_of(fact):
    return (fact["url"], fact["sentence_sha256"])


def _sorted_groups(facts):
    groups = defaultdict(list)
    for f in facts:
        groups[f["topic"]].append(f)
    for topic in groups:
        groups[topic].sort(key=lambda x: (
            x["url"], x["sentence_sha256"]))
    return dict(sorted(groups.items()))


def candidate_pairs(store, limit=40):
    """Display only real, currently eligible, same-topic source excerpts."""
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be 1..100")
    facts = _inventory(store)
    groups = _sorted_groups(facts)
    topics = []
    for topic, rows in groups.items():
        if len(rows) < 2:
            continue
        topics.append({
            "topic": topic,
            "question_example": "What is " + topic + "?",
            "quote_count": len(rows),
            "distinct_hosts": len(set(f["host"] for f in rows)),
            "possible_pairs_unlabeled": len(rows) * (len(rows) - 1) // 2,
            "candidates": [{
                "url": f["url"], "host": f["host"],
                "sentence_sha256": f["sentence_sha256"],
                "source_sha256": f["source_sha256"],
                "quote": f["answer"],
                "already_human_fact_checked": f["fact_checked"],
            } for f in rows],
        })
    reviewed = store.get(_KEY, []) or []
    return {
        "status": "review_candidates_only",
        "total_eligible_topics": len(groups),
        "total_eligible_quotes": len(facts),
        "topics_with_two_or_more_quotes": len(topics),
        "topics_with_one_quote": sum(len(v) == 1 for v in groups.values()),
        "unlabeled_potential_same_topic_pairs": sum(
            x["possible_pairs_unlabeled"] for x in topics),
        "human_judgments_saved": len(reviewed),
        "shown_topics": topics[:int(limit)],
        "truncated": len(topics) > int(limit),
        "no_labels_inferred_by_software": True,
        "note": ("These citations are exact source excerpts, not proven "
                 "semantic preferences. Review them manually before "
                 "providing any judgment. This is development-only, "
                 "not an independent blinded test."),
    }


def review_pair(store, question, better_url, better_sha,
                worse_url, worse_sha, confirm=False):
    """Append immutable operator-attested preference after corpus verification."""
    if not confirm:
        raise ValueError("An explicit --confirm-human-review is required")
    topic = topic_from_question(question)
    if topic is None:
        raise ValueError("Only supported same-topic definition questions accepted")
    if not all(isinstance(h, str) and _SHA.fullmatch(h)
               for h in (better_sha, worse_sha)):
        raise ValueError("Quote identifiers must be full SHA256 digests")
    if better_url == worse_url and better_sha == worse_sha:
        raise ValueError("Cannot compare a source quote against itself")
    facts = _inventory(store)  # Already rights-checked, text SHA-verified.
    matches = {_key_of(f): f for f in facts if f["topic"] == topic}
    good = matches.get((better_url, better_sha))
    bad = matches.get((worse_url, worse_sha))
    if not good or not bad:
        raise ValueError(
            "One or both SHA-verified quotes are not eligible definitions "
            "for this exact topic; UNKNOWN/noai cannot be reviewed here")
    identity = sorted([list(_key_of(good)), list(_key_of(bad))])
    pair_id = _digest({"question": question.strip(), "topic": topic,
                       "unordered_refs": identity})
    saved = store.get(_KEY, []) or []
    if any(x["pair_id"] == pair_id for x in saved):
        raise ValueError("Pair already reviewed; immutable labels cannot be rewritten")
    record = {
        "pair_id": pair_id,
        "scope": _SCOPE,
        "question": question.strip(),
        "topic": topic,
        "better": {
            "url": good["url"], "sentence_sha256": good["sentence_sha256"],
            "source_sha256": good["source_sha256"],
        },
        "worse": {
            "url": bad["url"], "sentence_sha256": bad["sentence_sha256"],
            "source_sha256": bad["source_sha256"],
        },
        "operator_attested_better_answers_question": True,
        "operator_identity_independently_verified": False,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "not_training_data": True,
        "blind_holdout_claimed": False,
    }
    # Append-only relative to this supported CLI. Source files untouched.
    store.put(_KEY, saved + [record])
    return {
        "status": "manual_judgment_saved",
        "pair_id": pair_id,
        "scope": _SCOPE,
        "question": question.strip(),
        "better_quote": good["answer"],
        "worse_quote": bad["answer"],
        "reviewed_judgments_total": len(saved) + 1,
        "fed_to_neural_training": False,
        "not_an_independent_blind_test": True,
    }


def score_reviewed_pairs(store, limit=100):
    """Read-only source-verified review audit of exact production scores."""
    if not 1 <= int(limit) <= 1000:
        raise ValueError("limit must be 1..1000")
    records = store.get(_KEY, []) or []
    facts = _inventory(store)
    index = {_key_of(f): f for f in facts}
    model, checkpoint = (None, None)
    try:
        model, checkpoint = _accepted_model(store, facts=facts)
    except (ValueError, OSError, RuntimeError, KeyError):
        pass
    torch = None
    if model is not None and records:
        from .brain import require_torch
        torch, _ = require_torch()
    outcomes, invalid = [], []
    for record in records[:int(limit)]:
        better = record["better"]
        worse = record["worse"]
        pos = index.get((better["url"], better["sentence_sha256"]))
        neg = index.get((worse["url"], worse["sentence_sha256"]))
        valid = (
            pos is not None and neg is not None and
            pos["topic"] == neg["topic"] == record["topic"] and
            pos["source_sha256"] == better["source_sha256"] and
            neg["source_sha256"] == worse["source_sha256"] and
            topic_from_question(record["question"]) == record["topic"] and
            record.get("scope") == _SCOPE and
            record.get("operator_attested_better_answers_question") is True
        )
        if not valid:
            invalid.append({
                "pair_id": record.get("pair_id"),
                "reason": "evidence_or_approval_changed_or_invalid_label",
            })
            continue
        question = record["question"]
        pos_score = _lexical_score(question, pos) + _identity_priority(
            pos["topic"], pos["answer"])
        neg_score = _lexical_score(question, neg) + _identity_priority(
            neg["topic"], neg["answer"])
        lex_delta = pos_score - neg_score
        neural_delta = None
        if model is not None:
            with torch.no_grad():
                def neural_score(f):
                    vector = torch.tensor([_features(question, f)],
                                          dtype=torch.float32)
                    val = float(model(vector)[0, 0].item()) * 0.1
                    return max(-0.5, min(0.5, val))
                neural_delta = (
                    pos_score + neural_score(pos) -
                    neg_score - neural_score(neg)
                )
        outcomes.append({
            "pair_id": record["pair_id"], "topic": record["topic"],
            "question": question,
            "better_url": pos["url"], "worse_url": neg["url"],
            "lexical_prefers_reviewed_better": lex_delta > 1e-9,
            "lexical_tie": abs(lex_delta) <= 1e-9,
            "neural_prefers_reviewed_better":
                neural_delta > 1e-9 if neural_delta is not None else None,
            "neural_tie":
                abs(neural_delta) <= 1e-9 if neural_delta is not None else None,
        })
    evaluated = len(outcomes)
    lex_ok = sum(x["lexical_prefers_reviewed_better"] for x in outcomes)
    neural_count = sum(
        x["neural_prefers_reviewed_better"] is not None for x in outcomes)
    neural_ok = sum(
        x["neural_prefers_reviewed_better"] is True for x in outcomes)
    return {
        "status": "manual_development_diagnostic",
        "scope": _SCOPE,
        "manual_judgments_recorded": len(records),
        "reviewed_pairs_evaluated": evaluated,
        "invalid_or_stale_pairs": invalid,
        "label_manifest_sha256": _digest(records),
        "neural_checkpoint_loaded": model is not None,
        "checkpoint_sha256": checkpoint["sha256"] if checkpoint else None,
        "lexical_correct": lex_ok,
        "lexical_accuracy": round(lex_ok / evaluated, 4) if evaluated else None,
        "neural_pairs_evaluated": neural_count,
        "neural_correct": neural_ok,
        "neural_accuracy": (
            round(neural_ok / neural_count, 4) if neural_count else None
        ),
        "neural_minus_lexical_accuracy": (
            round(neural_ok / neural_count - lex_ok / evaluated, 4)
            if neural_count and neural_count == evaluated else None
        ),
        "evaluations": outcomes,
        "limitations": {
            "development_judgments_not_blind_test": True,
            "reviewer_accuracy_independently_verified": False,
            "training_uses_these_judgments": False,
            "test_scores_used_for_checkpoint_selection": False,
            "new_neural_weights_created": False,
            "read_only": True,
            "semantic_generalization_proven": False,
        },
        "next_action": (
            "Collect more independent, rights-reviewed source explanations "
            "for the SAME topics, have operators compare real competing "
            "answers, then create a separate locked, previously-unseen "
            "publisher holdout dataset before claiming generalization."
        ),
    }
