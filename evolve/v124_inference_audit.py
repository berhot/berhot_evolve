"""EVOLVE v1.24 — read-only inference diagnostics, NOT scientific grading.

This examines actual source-backed answers and the potential to distinguish
competing quotations. It cannot establish semantic understanding from
self-generated questions, a citation's SHA, or synthetic easy negatives.
"""
from collections import Counter

from .v118_core import _approved_text
from .v120_grounded import (
    _inventory, _navigation_fragment, grounded_answer, topic_from_question,
)

_ABSENT_PROBES = (
    "quantum entanglement", "mycorrhiza", "quasiparticle",
    "transcription factor", "nucleosynthesis",
)


def evaluate_inference(store, limit=30):
    """Inspect current approved evidence and current inference without writes.

    Source previews and citations are SHA-checked by _approved_text.
    The experiment has NO external human relevance labels and avoids making
    accuracy or generalization claims.
    """
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be 1..100")
    facts = _inventory(store)
    by_topic = {}
    for fact in facts:
        by_topic.setdefault(fact["topic"], []).append(fact)

    rows = []
    status_counts = Counter()
    integrity_failures = []
    competition = 0
    model_loaded = 0
    neural_competitions = 0
    neural_changed = 0
    verified_quotes = 0
    checked_quote_hashes = set()

    for topic in sorted(by_topic)[:int(limit)]:
        group = by_topic[topic]
        # Two surface forms of the SAME known concept, not novel semantics.
        questions = ("What is " + topic + "?", "Define " + topic)
        variants = []
        for question in questions:
            response = grounded_answer(store, question, limit=3)
            status = response["status"]
            status_counts[status] += 1
            if response.get("ranker_checkpoint_loaded"):
                model_loaded += 1
            if response.get("neural_ranker_used"):
                neural_competitions += 1
            if response.get("neural_decision_changed_top_quote"):
                neural_changed += 1
            for quote in response.get("sources", []):
                url = quote["url"]
                try:
                    source, text = _approved_text(store, url)
                    ok = (
                        quote["source_sha256"] == source["sha256"] and
                        quote["excerpt"] in text and
                        not _navigation_fragment(quote["excerpt"])
                    )
                except (ValueError, OSError, KeyError):
                    ok = False
                marker = (url, quote["excerpt"])
                if marker not in checked_quote_hashes:
                    checked_quote_hashes.add(marker)
                    if ok:
                        verified_quotes += 1
                    else:
                        integrity_failures.append({
                            "url": url, "topic": topic,
                            "reason": "quote_not_exact_or_navigation_noise",
                        })
            variants.append({
                "question": question, "status": status,
                "answer": response.get("answer"),
                "sources": [s["url"] for s in response.get("sources", [])],
                "candidate_count": response.get("candidate_count", 0),
                "ranker_checkpoint_loaded":
                    response.get("ranker_checkpoint_loaded", False),
                "neural_ranker_used": response.get("neural_ranker_used", False),
                "neural_changed_top_quote":
                    response.get("neural_decision_changed_top_quote", False),
            })

        distinct = len({f["answer"] for f in group})
        if distinct > 1:
            competition += 1
        rows.append({
            "topic": topic,
            "candidate_quotes": len(group),
            "distinct_quotes": distinct,
            "independent_hosts_with_topic": len({f["host"] for f in group}),
            "can_compare_same_topic_quotes": distinct >= 2,
            "variants": variants,
        })

    # Deliberately choose only topics absent from today's approved inventory.
    # A correct 'no_definition' is thus trivial dictionary membership, not
    # a calibrated answerability/knowledge test.
    absent_topics = [t for t in _ABSENT_PROBES if t not in by_topic][:3]
    abstention = []
    for topic in absent_topics:
        result = grounded_answer(store, "What is " + topic + "?")
        abstention.append({
            "question": "What is " + topic + "?",
            "status": result["status"],
            "empty_sources": not result["sources"],
            "expected_by_inventory_lookup_only": "no_definition",
        })
    unsupported = grounded_answer(store, "Why does the Sun shine?")
    previous = store.get("v120_ranker_latest") or {}
    benchmark = previous.get("benchmark") or {}

    return {
        "status": "diagnostic_only",
        "method": "self_derived_surface_forms_and_quote_integrity",
        "approved_definition_topics_total": len(by_topic),
        "approved_definition_quotes_total": len(facts),
        "topics_examined": len(rows),
        "candidate_question_variants_examined": sum(
            len(x["variants"]) for x in rows),
        "answers_by_status": dict(status_counts),
        "topics_with_competing_quotes": competition,
        "topics_without_competing_quotes": len(rows) - competition,
        "loaded_checkpoint_question_cases": model_loaded,
        "actual_neural_ranking_competitions": neural_competitions,
        "neural_changed_top_quote_count": neural_changed,
        "distinct_source_quotes_integrity_checked": len(checked_quote_hashes),
        "distinct_source_quotes_integrity_passed": verified_quotes,
        "quote_integrity_failures": integrity_failures[:20],
        "unsupported_question_probe": {
            "question": "Why does the Sun shine?",
            "status": unsupported["status"],
            "empty_sources": not unsupported["sources"],
            "meaning": "parser scope rejection, not scientific reasoning",
        },
        "absent_from_approved_inventory_probes": abstention,
        "last_ranker_comparison": {
            "checkpoint_recorded": bool(previous),
            "lexical_heldout_pair_accuracy":
                (benchmark.get("lexical_test") or {}).get("pair_accuracy"),
            "neural_heldout_pair_accuracy":
                benchmark.get("neural_test_pair_accuracy"),
            "neural_outperformed_lexical":
                benchmark.get("outperformed_lexical_on_unseen_test_hosts"),
            "different_topic_easy_negative_pairs":
                benchmark.get("different_topic_easy_negative_pairs"),
        },
        "topics": rows,
        "limitations": {
            "read_only": True,
            "external_human_graded_answer_relevance": False,
            "unseen_semantics_tested": False,
            "fresh_blind_test_performed": False,
            "same_topic_correct_answer_labels_provided": False,
            "novel_scientific_understanding_proven": False,
            "zero_source_claims_fabricated": True,
            "checkpoint_weights_modified": False,
            "paraphrases_are_only_syntactic_variants": True,
            "missing_topic_abstention_is_trivial_membership": True,
        },
        "next_action": (
            "For a meaningful ranker evaluation, collect many independent "
            "rights-cleared definitions, have a reviewer label difficult "
            "same-topic alternatives and unanswerable prompts, and lock "
            "publisher-held-out test material before training or tuning."
        ),
    }
