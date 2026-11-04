"""BERHOT EVOLVE 1.20 — cautious source-grounded definitions and neural evidence ranking.

Source extraction != factual verification; PyTorch ranking != generative reasoning.
No API, network, rights approval or writes to older neural registries.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import random
import re
import time

from .brain import require_torch, sha256_file
from .v118_core import _approved_text, _sha, _words
from .v117 import prepare
from .v119_curiosity import _sentences

FORMAT = "berhot_evidence_ranker_v120"
_SPLITS = ("train", "val", "test")
_BAD_START = ("media ", "image ", "photo ", "video ", "see also ", "references",
              "advertisement ", "read more ", "copyright ", "subscribe ",
              "click here", "skip to ", "caption ", "figure ")
_BAD_BODY = ("ignore previous", "system prompt", "sign in", "cookie policy",
             "privacy policy", "read our full", "your browser")
_BAD_TOPIC = {
    "this", "that", "these", "those", "it", "there", "here", "they",
    "which", "what", "many", "most", "some", "more", "other", "each",
    "our", "all", "one", "two", "three", "world", "people", "today",
    "scientists", "researchers", "article", "information",
}
_PREDICATE = re.compile(
    r"^(?:an?\s+|the\s+|one\s+of\s+|part\s+of\s+|"
    r"made\s+of\s+|composed\s+of\s+|formed\s+from\s+|"
    r"located\s+|known\s+as\s+|called\s+|"
    r"a\s+type\s+of\s+|the\s+name\s+of\s+)", re.I
)
_DEFINE = re.compile(
    r"^(?:(?:The|A|An|Our)\s+)?"
    r"(?P<topic>[A-Za-z][A-Za-z-]{2,32}"
    r"(?:\s+[A-Za-z][A-Za-z-]{2,32})?)"
    r"\s+(?P<verb>is|are|means|refers\s+to)\s+"
    r"(?P<predicate>[^!?;]{6,250})[.!?]?$",
    flags=re.I,
)
_QUESTION = re.compile(
    r"^(?:(?:what\s+(?:is|are)|define|describe|explain|tell\s+me\s+about)"
    r"\s+)?(?:(?:the|an?|our)\s+)?"
    r"(?P<topic>[a-z][a-z-]{2,32}(?:\s+[a-z][a-z-]{2,32})?)\??$",
    re.I,
)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def topic_from_question(question):
    if not isinstance(question, str) or not 2 <= len(question.strip()) <= 180:
        raise ValueError("Question must contain 2..180 characters")
    q = " ".join(question.strip().rstrip("?!. ").split())
    m = _QUESTION.fullmatch(q)
    if not m:
        return None
    topic = m["topic"].casefold()
    if topic in _BAD_TOPIC:
        return None
    if topic.startswith(("why ", "where ", "when ", "how ", "who ")):
        return None
    return topic


def extract_definition(sentence):
    """Explicit sentence-level, *heuristic* definitional claim, not proven fact."""
    s = " ".join(sentence.split()).strip()
    if not 18 <= len(s) <= 320 or not 4 <= len(s.split()) <= 62:
        return None
    lower = s.casefold()
    if lower.startswith(_BAD_START) or any(bad in lower for bad in _BAD_BODY):
        return None
    match = _DEFINE.match(s)
    if not match:
        return None
    topic = match["topic"].casefold()
    if topic in _BAD_TOPIC or any(w.casefold() in _BAD_TOPIC for w in topic.split()):
        return None
    predicate = match["predicate"].strip()
    # Don't interpret an arbitrary adverb ("roughly", "sometimes") as
    # a noun definition. Explicit capitalized possessives such as
    # "Earth's natural satellite" are allowed, case-sensitively.
    defining_predicate = (_PREDICATE.match(predicate) is not None or
                          re.match(r"^[A-Z][a-z]+'s\s+", predicate) is not None)
    if match["verb"].casefold() in ("is", "are") and not defining_predicate:
        return None
    # Don't turn arbitrary text/navigation into supposed scientific claims.
    if len(set(re.findall(r"[a-z]{3,}", predicate.casefold()))) < 1:
        return None
    return topic


def title_question(topic_title):
    """Read only explicit public article titles like 'What is an estuary?'.

    A page title is a question, not automatically a correct answer.
    """
    if not isinstance(topic_title, str):
        return None
    match = re.match(
        r"^\\s*(what\\s+(?:is|are)\\s+"
        r"(?:(?:a|an|the)\\s+)?"
        r"[A-Za-z][A-Za-z-]{2,32}"
        r"(?:\\s+[A-Za-z][A-Za-z-]{2,32})?\\?)",
        topic_title.strip(), flags=re.I)
    if not match:
        return None
    return topic_from_question(match.group(1))


def _term_forms(topic):
    """Conservative singular/plural matching; never infer a new fact."""
    if " " in topic:
        return [topic]
    words = {topic, topic + "s", topic + "es"}
    if topic.endswith("y"):
        words.add(topic[:-1] + "ies")
    return sorted(words)


def title_evidence(text, topic):
    """Find one exact on-topic *definition-like* sentence near a Q title.

    Does not paraphrase, cite unrelated sentences, or add content to the
    source. A title alone is insufficient for a training example.
    """
    term = re.compile(r"\\b(?:" +
                      "|".join(re.escape(x) for x in _term_forms(topic)) +
                      r")\\b", re.I)
    # Avoid selecting incidental narrative such as 'Our Sun is roughly midway'.
    pattern = re.compile(
        r"\\b(?:is|are)\\s+(?:an?|the|one\\s+of)\\b|"
        r"\\b(?:is|are)\\s+(?:\\w+\\s+){0,3}"
        r"(?:bodies|areas|regions|types|systems|places|"
        r"formations|structures|objects|habitats|masses)\\s+of\\b|"
        r"\\b(?:means|refers\\s+to|defined\\s+as|describes)\\b",
        re.I)
    for sentence in _sentences(text)[:24]:
        candidate = " ".join(sentence.split()).strip()
        lower = candidate.casefold()
        if not 28 <= len(candidate) <= 320 or len(candidate.split()) < 6:
            continue
        if lower.startswith(_BAD_START) or any(x in lower for x in _BAD_BODY):
            continue
        if term.search(candidate) and pattern.search(candidate):
            return candidate
    return None


def _inventory(store, limit_sources=200, per_source=18):
    """Only read the SHA-verified, already rights-approved local corpus.

    Never use crawled UNKNOWN-rights pages or noai signals for training.
    """
    rows = store.db.execute(
        "SELECT url FROM web_corpus_sources WHERE rights_status IN "
        "('CC0-1.0','PUBLIC-DOMAIN') AND noai=0 ORDER BY url LIMIT ?",
        (limit_sources,)).fetchall()
    facts, seen = [], set()
    for row in rows:
        source, text = _approved_text(store, row["url"])
        count = 0
        for sentence in _sentences(text):
            topic = extract_definition(sentence)
            if topic is None:
                continue
            s = " ".join(sentence.split()).strip()
            digest = _sha(s.casefold().encode("utf-8"))
            if digest in seen:
                continue
            seen.add(digest)
            facts.append({
                "topic": topic, "question": "What is " + topic + "?",
                "answer": s, "url": source["url"], "title": source["title"],
                "host": source["host"], "source_sha256": source["sha256"],
                "sentence_sha256": digest,
                "provenance": "automatically_extracted_exact_source_definition",
                "fact_checked": False, "human_reviewed": False,
            })
            count += 1
            if count >= per_source:
                break
        # A genuine publisher heading such as 'What is a current?' can
        # supply the question; an exact, nearby source sentence supplies
        # the answer. Neither is silently promoted to verified science.
        heading_topic = title_question(source["title"])
        if heading_topic and count < per_source:
            sentence = title_evidence(text, heading_topic)
            if sentence is not None:
                digest = _sha(sentence.casefold().encode("utf-8"))
                key = (heading_topic, digest)
                if digest not in seen and not any(
                        f["topic"] == heading_topic and f["url"] == source["url"]
                        for f in facts):
                    seen.add(digest)
                    facts.append({
                        "topic": heading_topic,
                        "question": "What is " + heading_topic + "?",
                        "answer": sentence, "url": source["url"],
                        "title": source["title"], "host": source["host"],
                        "source_sha256": source["sha256"],
                        "sentence_sha256": digest,
                        "provenance": "publisher_question_title_exact_source_sentence",
                        "fact_checked": False, "human_reviewed": False,
                    })
    return facts


def _features(question, fact):
    topic = topic_from_question(question)
    qwords = _words(question)
    sentence = fact["answer"]
    fwords = _words(sentence)
    term = fact["topic"]
    exact = float(topic == term) if topic else 0.0
    overlap = len(qwords & fwords) / max(1, len(qwords))
    term_coverage = len(set(term.split()) & qwords) / max(1, len(term.split()))
    early = float(term in sentence[:70].casefold())
    # Smooth numeric features independent of a particular publisher/topic.
    length = 1.0 - min(abs(len(sentence) - 150) / 300, 1.0)
    title = len(qwords & _words(fact["title"])) / max(1, len(qwords))
    return [exact, overlap, term_coverage, early, length, title, 1.0]


def _lexical_score(question, fact):
    f = _features(question, fact)
    return 8*f[0] + 3*f[1] + 3*f[2] + f[3] + 0.5*f[4] + 0.5*f[5]


def _model():
    torch, nn = require_torch()
    return nn.Sequential(nn.Linear(7, 12), nn.Tanh(), nn.Linear(12, 1))


def _accepted_model(store):
    record = store.get("v120_ranker_latest")
    if not record or not record.get("accepted_by_validation"):
        return None, None
    file = Path(record.get("checkpoint", ""))
    # Store paths are untrusted metadata; validate containment and SHA.
    base = (store.root / "models" / "v120").resolve()
    if not file.resolve().is_relative_to(base):
        return None, None
    if not file.is_file() or sha256_file(file) != record["sha256"]:
        return None, None
    torch, _ = require_torch()
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT or checkpoint.get("feature_size") != 7:
        return None, None
    model = _model()
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model, record


def grounded_answer(store, question, limit=3):
    """Cite the exact full source definition, never synthesize a causal claim."""
    topic = topic_from_question(question)
    if topic is None:
        return {
            "mode": "grounded_qa_not_generative", "status": "unsupported",
            "answer": ("I can answer simple definitions supported by explicit "
                       "source sentences, such as 'What is the Sun?'. "
                       "I cannot yet establish a reliable explanation for "
                       "open-ended how/why questions."),
            "sources": [], "neural_weights_updated": False,
        }
    facts = _inventory(store)
    candidates = [fact for fact in facts if topic == fact["topic"]]
    if not candidates:
        return {
            "mode": "grounded_qa_not_generative", "status": "no_definition",
            "answer": (f"I haven't found an explicit, approved-source definition "
                       f"of {topic}. I won't invent one."),
            "sources": [], "neural_weights_updated": False,
        }
    model, record = (None, None)
    try:
        model, record = _accepted_model(store)
    except (RuntimeError, ValueError, OSError, KeyError):
        pass
    scores = []
    torch = None
    if model is not None:
        torch, _ = require_torch()
    for fact in candidates:
        base = _lexical_score(question, fact)
        neural = 0.0
        if model is not None:
            with torch.no_grad():
                features = torch.tensor([_features(question, fact)],
                                        dtype=torch.float32)
                neural = float(model(features)[0, 0].item())
        # Neural scoring is a bounded tiebreaker; it never invents evidence.
        scores.append((base + max(-0.5, min(0.5, neural * 0.1)), fact))
    scores.sort(key=lambda x: (-x[0], x[1]["url"], x[1]["sentence_sha256"]))
    selected = [x[1] for x in scores[:max(1, min(limit, 3))]]
    return {
        "mode": "grounded_qa_not_generative", "status": "answered",
        "answer": selected[0]["answer"],
        "sources": [{
            "title": f["title"], "url": f["url"],
            "excerpt": f["answer"], "host": f["host"],
            "source_sha256": f["source_sha256"],
        } for f in selected],
        "neural_ranker_used": model is not None,
        "ranker_checkpoint_sha256": record["sha256"] if record else None,
        "neural_weights_updated": False,
        "training_examples_not_neural_answers": True,
        "warning": (
            "Exact sentence from an approved local article, selected using "
            "definition extraction" + (" and a small neural evidence ranker" if model
            else " and lexical relevance") +
            ". The source's factual accuracy is not independently verified. "
            "Not a neural-generated explanation."
        ),
    }


def _partitions(store):
    frozen = prepare(store)
    if frozen["status"] != "prepared":
        return None, frozen
    host_split = {host: split for split in _SPLITS
                  for host in frozen["source_hosts"][split]}
    urls_by_split = {split: set(frozen["source_urls"][split]) for split in _SPLITS}
    grouped = {split: [] for split in _SPLITS}
    for row in _inventory(store):
        split = host_split.get(row["host"])
        if split and row["url"] in urls_by_split[split]:
            grouped[split].append(row)
    # Explicit source-host holdout. QA must be derived from distinct URLs.
    for a in _SPLITS:
        for b in _SPLITS:
            if a != b and (set(x["host"] for x in grouped[a])
                           & set(x["host"] for x in grouped[b])):
                raise ValueError("Source host leaked between QA partitions")
    return grouped, frozen


def readiness(store):
    parts, frozen = _partitions(store)
    if parts is None:
        return {"status": "needs_sources", "detail": frozen}
    counts = {split: len(parts[split]) for split in _SPLITS}
    unique = {split: len(set(f["topic"] for f in parts[split]))
              for split in _SPLITS}
    ready = all(counts[x] >= n for x, n in
                (("train", 4), ("val", 2), ("test", 2))) and all(
                    unique[x] >= 2 for x in _SPLITS)
    return {
        "status": "ready" if ready else "needs_diverse_definitions",
        "counts": counts, "unique_topics": unique,
        "minimum": {"train": 4, "val": 2, "test": 2},
        "minimum_topics_per_split": 2,
        "source_hosts": frozen["source_hosts"],
        "note": ("Extraction from explicit definitional sentences; not "
                 "human-verified factual QA. Held-out hosts and test untouched "
                 "during model selection."),
    }


def _pairs(facts, cap=150):
    # Deterministic balanced negative candidates: distinct topics, same split.
    result = []
    ordered = sorted(facts, key=lambda f: (f["topic"], f["url"],
                                            f["sentence_sha256"]))
    for i, positive in enumerate(ordered[:cap]):
        alternatives = [negative for negative in ordered
                        if negative["topic"] != positive["topic"]]
        if not alternatives:
            continue
        negative = alternatives[(i * 7 + 3) % len(alternatives)]
        question = positive["question"]
        result.append((_features(question, positive),
                       _features(question, negative)))
    return result


def _rank_eval(model, pairs, torch):
    if not pairs:
        return {"loss": None, "pair_accuracy": None, "n": 0}
    model.eval()
    with torch.no_grad():
        pos = torch.tensor([row[0] for row in pairs], dtype=torch.float32)
        neg = torch.tensor([row[1] for row in pairs], dtype=torch.float32)
        diff = model(pos).squeeze(-1) - model(neg).squeeze(-1)
        loss = torch.nn.functional.softplus(-diff).mean().item()
        accuracy = (diff > 0).float().mean().item()
    return {"loss": round(loss, 6), "pair_accuracy": round(accuracy, 4),
            "n": len(pairs)}


def train_ranker(store, settings, steps=120, seed=31415, progress=None):
    """Train a small learned evidence *ranker*, not a text generation model.

    Train-only examples, independent validation acceptance, untouched test
    and SHA-checked immutable checkpoint. Existing models are never touched.
    """
    if not 1 <= int(steps) <= 500:
        raise ValueError("Steps must be 1..500")
    parts, frozen = _partitions(store)
    if parts is None:
        return {"status": "skipped", "reason": "strict_source_host_holdout_missing"}
    check = readiness(store)
    if check["status"] != "ready":
        return {"status": "skipped", "reason": check["status"],
                "readiness": check}
    data = {split: _pairs(parts[split]) for split in _SPLITS}
    if any(len(data[split]) < n for split, n in
           (("train", 4), ("val", 2), ("test", 2))):
        return {"status": "skipped", "reason": "insufficient_pairwise_examples"}
    torch, _ = require_torch()
    torch.manual_seed(seed)
    if torch.get_num_threads() > 2:
        torch.set_num_threads(2)
    model = _model()
    val_before = _rank_eval(model, data["val"], torch)
    # Fixed train-only samples; no test access in optimization.
    pos = torch.tensor([r[0] for r in data["train"]], dtype=torch.float32)
    neg = torch.tensor([r[1] for r in data["train"]], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    t0, actual, loss_value = time.monotonic(), 0, None
    for index in range(int(steps)):
        model.train()
        margin = model(pos).squeeze(-1) - model(neg).squeeze(-1)
        loss = torch.nn.functional.softplus(-margin).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        actual += 1
        loss_value = float(loss.detach().item())
        if progress and (actual == 1 or actual % 10 == 0 or actual == steps):
            progress("evidence_ranking", actual, int(steps), round(loss_value, 6))
        if time.monotonic() - t0 > min(120, float(
                settings.get("train", {}).get("max_seconds", 600))):
            break
    val_after = _rank_eval(model, data["val"], torch)
    test_after = _rank_eval(model, data["test"], torch)
    # Accept only if genuinely improved on *held-out validation* and better
    # than random ranker pair accuracy; never select based on the test score.
    accepted = (
        val_after["loss"] + 1e-5 < val_before["loss"] and
        val_after["pair_accuracy"] >= 0.50
    )
    snapshot = {
        "format": FORMAT,
        "dataset_sha256": frozen["dataset_sha256"],
        "source_hosts": frozen["source_hosts"],
        "splits": {k: [{
            "question": item["question"], "url": item["url"],
            "source_sha256": item["source_sha256"],
            "sentence_sha256": item["sentence_sha256"],
            "provenance": item["provenance"],
        } for item in parts[k]] for k in _SPLITS},
        "training_eligible_rights": ["CC0-1.0", "PUBLIC-DOMAIN"],
        "human_verification_claimed": False,
    }
    canonical = json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    base = store.root / "models" / "v120"
    base.mkdir(parents=True, exist_ok=True)
    manifest = base / ("manifest-" + digest + ".json")
    if manifest.is_file() and manifest.read_bytes() != canonical:
        raise ValueError("Immutable manifest content mismatch")
    if not manifest.exists():
        manifest.write_bytes(canonical)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint = base / ("ranker-" + stamp + ".pt")
    if checkpoint.exists():
        raise ValueError("Refusing to overwrite a previous v120 checkpoint")
    torch.save({
        "format": FORMAT, "feature_size": 7, "seed": int(seed),
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "manifest_sha256": digest, "steps": actual,
    }, checkpoint)
    record = {
        "status": "trained", "model_type": "neural_evidence_ranking_not_generation",
        "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
        "manifest_path": str(manifest), "manifest_sha256": digest,
        "dataset_sha256": frozen["dataset_sha256"],
        "source_hosts": frozen["source_hosts"],
        "actual_training_steps": actual, "accepted_by_validation": accepted,
        "validation_before": val_before, "validation_after": val_after,
        "test_after": test_after,
        "lexical_baseline_note": (
            "Keyword/topic lexical ranking remains available; validation "
            "measures only synthetic source-derived pairwise comparisons."),
        "question_answer_understanding_proven": False,
        "previous_checkpoints_unchanged": True,
        "source_derived_examples": {k: len(parts[k]) for k in _SPLITS},
        "created_at_utc": _utc(),
    }
    history = store.get("v120_ranker_history", []) or []
    history.append(record)
    store.put("v120_ranker_history", history[-100:])
    store.put("v120_ranker_candidate", record)
    if accepted:
        store.put("v120_ranker_latest", record)
    store.log("v120_ranker_train", "completed", {
        "sha256": record["sha256"], "accepted": accepted,
        "val": val_after, "test": test_after})
    return record


def inventory(store):
    facts = _inventory(store)
    model = store.get("v120_ranker_latest")
    return {
        "status": "available", "source_derived_definitions": len(facts),
        "unique_topics": len(set(x["topic"] for x in facts)),
        "approved_source_hosts": len(set(x["host"] for x in facts)),
        "neural_ranker_accepted": bool(model),
        "neural_ranker_sha256": model.get("sha256") if model else None,
        "qa_definition_generation": "exact source quotation, not generative",
        "unapproved_sources_in_train": False,
        "heldout_qa_quality_not_proven_by_extraction": True,
    }
