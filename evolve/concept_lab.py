"""v1.9: neural answer recall from independently corroborated, reusable definitions.

This deliberately narrow *text* learning experiment consumes the existing
source-verification/rights pipeline's admitted definition records only.
A byte-level causal Transformer is trained with answer-token loss, then
tested on question phrasings never used for gradient updates or checkpoint
selection. Greedy predictions come from the neural checkpoint, WITHOUT a
definition lookup, retrieval augmentation or correct-answer substitution.

Scope: same-concept paraphrase generalization, NOT unseen-fact knowledge,
general reasoning, or proof that two agreeing websites are always correct.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import random
import re
import time

from .brain import (
    build_model, get_device, load_checkpoint, require_torch, sha256_file,
)

FORMAT = "verified_definition_answer_only_byte_transformer_v1"
TRAIN_TEMPLATES = (
    "What is {concept}?",
    "Define {concept}.",
    "Explain the meaning of {concept}.",
)
VALIDATION_TEMPLATE = "How would you define {concept}?"
TEST_TEMPLATES = (
    "Describe {concept}.",
    "What does {concept} mean?",
)
NEGATIVE_CONCEPT = "quendrel"
MAX_ANSWER_BYTES = 192


def _inventory(store, max_concepts=8):
    """Admit only facts with database provenance, rights and exact source docs."""
    limit = int(max_concepts)
    if not 2 <= limit <= 32:
        raise ValueError("max_concepts must be between 2 and 32")
    sql = (
        "SELECT v.question_id, v.concept, v.answer, v.evidence_json, "
        "v.verification_method, v.training_eligible, "
        "q.question AS source_question, d.text AS document_text, "
        "d.topic, d.license, d.modality, d.source_id "
        "FROM verified_answers v "
        "JOIN research_questions q ON q.id = v.question_id "
        "JOIN documents d ON d.source_id = ('verified-definition:' || v.question_id) "
        "WHERE v.training_eligible = 1 "
        "AND v.verification_method = 'two_source_definition_overlap_v1' "
        "AND d.topic = 'verified:definition' "
        "AND d.license = 'SELF-GENERATED-VERIFIED' "
        "AND d.modality = 'text' ORDER BY v.question_id ASC"
    )
    items = []
    seen = set()
    rows = store.db.execute(sql).fetchall()
    for row in rows:
        concept, answer = str(row["concept"]).strip(), str(row["answer"]).strip()
        if not (re.fullmatch(r"[a-z]{3,28}(?: [a-z]{3,28})?", concept)
                and 16 <= len(answer.encode("utf-8")) <= MAX_ANSWER_BYTES):
            continue
        if concept.casefold() in seen:
            continue
        try:
            proof = json.loads(row["evidence_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(proof, list):
            continue
        publishers = {p.get("publisher") for p in proof if isinstance(p, dict)
                      and p.get("publisher")}
        eligible = any(
            p.get("license") in ("CC0-1.0", "PUBLIC-DOMAIN")
            for p in proof if isinstance(p, dict))
        if len(publishers) < 2 or not eligible:
            continue
        lesson = f"Question: {row['source_question']}\nAnswer: {answer}\n"
        if row["document_text"] != lesson:
            continue
        if row["source_question"].casefold() != f"What is {concept}?".casefold():
            continue
        if NEGATIVE_CONCEPT in concept.casefold():
            continue
        sources = sorted(str(p.get("definition_sha256", ""))
                         for p in proof if isinstance(p, dict))
        items.append({
            "question_id": int(row["question_id"]),
            "concept": concept,
            "answer": answer,
            "proof_hashes": sources,
        })
        seen.add(concept.casefold())
        if len(items) >= limit:
            break
    return items


def concept_inventory(store, max_concepts=8):
    selected = _inventory(store, max_concepts=max_concepts)
    return {
        "status": "available" if len(selected) >= 2 else "insufficient_verified_facts",
        "eligible_concepts": len(selected),
        "required_concepts": 2,
        "concepts": [c["concept"] for c in selected],
        "source": "corroborated_licensed_verified_definition_documents_only",
        "note": ("Concepts shown here passed two-publisher, reusable-rights and "
                 "exact training-document checks. No web pages are fetched."),
    }


def _dataset_digest(records):
    serial = [{
        "id": c["question_id"], "concept": c["concept"],
        "answer": c["answer"], "evidence": c["proof_hashes"],
    } for c in records]
    return hashlib.sha256(json.dumps(
        serial, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _model_cfg(settings):
    cfg = dict(settings["model"])
    # Keep all verified English definitions within a fixed context; never
    # truncate the target answer because that could inflate exact match.
    cfg["block_size"] = max(256, int(cfg["block_size"]))
    return cfg


def _prompt(question):
    return f"Question: {question}\nAnswer:"


def _example(question, answer, block):
    """Padded batch targets are masked; loss never trains on question bytes."""
    context = _prompt(question).encode("utf-8")
    completion = (" " + answer + "\n").encode("utf-8")
    sequence = context + completion
    if len(sequence) > block:
        return None
    x = list(sequence[:-1])
    y = list(sequence[1:])
    y[:len(context) - 1] = [-100] * (len(context) - 1)
    return x, y


def _batch_loss(model, rows, torch, device):
    if not rows:
        raise ValueError("No training examples")
    max_len = max(len(x) for x, _ in rows)
    x = torch.tensor([
        arr + [10] * (max_len - len(arr)) for arr, _ in rows
    ], dtype=torch.long, device=device)
    y = torch.tensor([
        arr + [-100] * (max_len - len(arr)) for _, arr in rows
    ], dtype=torch.long, device=device)
    logits = model(x)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100)


def _val_loss(model, rows, torch, device):
    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            values.append(float(_batch_loss(model, [row], torch, device).item()))
    return sum(values) / len(values)


def _completion(model, question, block, device="cpu", max_new_bytes=128):
    """Greedy neural tokens only. The answer never comes from a record lookup."""
    torch, _ = require_torch()
    prefix = _prompt(question).encode("utf-8")
    if len(prefix) >= block:
        raise ValueError("Question exceeds model context")
    generated = list(prefix)
    output = bytearray()
    model.eval()
    with torch.no_grad():
        for _ in range(int(max_new_bytes)):
            x = torch.tensor([generated[-block:]], dtype=torch.long, device=device)
            predicted = int(model(x)[0, -1].argmax().item())
            if predicted == 10:
                break
            generated.append(predicted)
            output.append(predicted)
    return bytes(output).decode("utf-8", errors="replace").strip()


def _normalize(s):
    return re.sub(r"\s+", " ", str(s).casefold().strip()).rstrip(". ")


def _word_overlap(prediction, expected):
    """A diagnostic, not a substitute for answer correctness."""
    p, e = re.findall(r"[a-z]+", prediction.casefold()), re.findall(
        r"[a-z]+", expected.casefold())
    if not p or not e:
        return 0.0
    from collections import Counter
    common = sum((Counter(p) & Counter(e)).values())
    return 2 * common / (len(p) + len(e))


def _test_cases(items):
    return [(c["concept"], template.format(concept=c["concept"]), c["answer"])
            for c in items for template in TEST_TEMPLATES]


def _evaluate(model, items, cfg, device, max_new_bytes):
    answers = {c["concept"]: _normalize(c["answer"]) for c in items}
    examples = []
    for concept, question, expected in _test_cases(items):
        predicted = _completion(model, question, cfg["block_size"], device,
                                max_new_bytes)
        norm = _normalize(predicted)
        examples.append({
            "concept": concept, "question": question,
            "raw_completion": predicted,
            "exact_match": norm == _normalize(expected),
            "token_f1": round(_word_overlap(predicted, expected), 4),
            "another_concept_answer": (
                next((other for other, value in answers.items()
                      if other != concept and norm == value), None)),
        })
    size = len(examples)
    correct = sum(int(x["exact_match"]) for x in examples)
    by_concept = {
        c["concept"]: {
            "correct": sum(int(row["exact_match"]) for row in examples
                           if row["concept"] == c["concept"]),
            "total": len(TEST_TEMPLATES),
        } for c in items
    }
    return {
        "questions": size, "correct": correct,
        "exact_accuracy": round(correct / size, 4),
        "mean_token_f1": round(sum(row["token_f1"] for row in examples) / size, 4),
        "cross_concept_confusions": sum(bool(x["another_concept_answer"])
                                        for x in examples),
        "distinct_completions": len({_normalize(x["raw_completion"]) for x in examples}),
        "by_concept": by_concept,
        "examples": examples,
    }


def _load_model(store, candidate=False):
    registry = "concept_lab_latest_candidate" if candidate else "concept_lab_latest"
    record = store.get(registry)
    if record is None:
        return None
    path = Path(record["path"]).resolve()
    root = (store.root / "models" / "concept_lab").resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("Concept checkpoint missing or outside experiment folder")
    if sha256_file(path) != record["sha256"]:
        raise ValueError("Concept checkpoint SHA256 mismatch")
    model, data = load_checkpoint(path)
    if (data.get("objective") != FORMAT or
        data.get("dataset_sha256") != record["dataset_sha256"] or
        data.get("model_config") != record["model_config"] or
        data.get("concepts") != record["concepts"]):
        raise ValueError("Concept checkpoint format or metadata mismatch")
    model.eval()
    return model, record


def predict_concept_lab(store, question, candidate=False, max_new_bytes=128):
    """No calls to documents, evidence, lexical retrieval, or verified answers."""
    if not isinstance(question, str) or not 3 <= len(question.strip()) <= 180:
        raise ValueError("Question length must be 3..180")
    if not 1 <= int(max_new_bytes) <= 192:
        raise ValueError("max_new_bytes must be 1..192")
    loaded = _load_model(store, candidate=candidate)
    if loaded is None:
        return {"status": "skipped", "reason": "No accepted neural concept checkpoint"}
    model, ref = loaded
    answer = _completion(model, question.strip(), ref["model_config"]["block_size"],
                         "cpu", int(max_new_bytes))
    return {
        "status": "predicted", "question": question, "raw_completion": answer,
        "checkpoint_status": "accepted" if ref["accepted"] else "unaccepted_candidate",
        "checkpoint": ref["sha256"], "experiment": FORMAT,
        "initialized_from": ref.get("initialized_from"),
        "note": ("Greedy tokens from a trained neural checkpoint, with no "
                 "retrieval/answer substitution. Factual correctness is not guaranteed."),
    }


def train_concept_lab(store, settings, steps=300, seed=31415, max_concepts=8,
                      force=False, max_new_bytes=128):
    """Train on admitted answers, select by heldout prompt validation, then test."""
    steps, seed, max_new_bytes = int(steps), int(seed), int(max_new_bytes)
    if not 1 <= steps <= 3000:
        raise ValueError("Training steps must be 1..3000")
    if not 16 <= max_new_bytes <= 192:
        raise ValueError("max_new_bytes must be 16..192")
    items = _inventory(store, max_concepts)
    if len(items) < 2:
        return {
            "status": "skipped",
            "reason": "Need at least 2 independently corroborated, licensed, persisted definitions",
            "eligible_concepts": len(items),
            "action": "Run existing source-grounded discovery/research and verify reuse rights first.",
        }
    cfg = _model_cfg(settings)
    digest = _dataset_digest(items)
    previous = store.get("concept_lab_latest")
    if previous and previous.get("dataset_sha256") == digest and not force:
        return {"status": "skipped", "reason": "No new eligible concept records since accepted checkpoint",
                "dataset_sha256": digest}

    all_train = [
        _example(template.format(concept=c["concept"]),
                 c["answer"], cfg["block_size"])
        for c in items for template in TRAIN_TEMPLATES
    ]
    all_val = [
        _example(VALIDATION_TEMPLATE.format(concept=c["concept"]),
                 c["answer"], cfg["block_size"]) for c in items
    ]
    if any(item is None for item in all_train + all_val):
        return {"status": "skipped", "reason": "Definition/prompt exceeds model context; no truncation",
                "block_size": cfg["block_size"]}
    # No heldout question template appears in training or validation prompts.
    train_questions = {
        template.format(concept=c["concept"]) for c in items
        for template in TRAIN_TEMPLATES
    }
    val_questions = {
        VALIDATION_TEMPLATE.format(concept=c["concept"]) for c in items
    }
    test_questions = {q for _, q, _ in _test_cases(items)}
    if train_questions & test_questions or val_questions & test_questions:
        raise AssertionError("Heldout question template leakage")

    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = None
    initialized_from = "random_weights"
    old_checkpoint = None
    if previous:
        prior_model, ref = _load_model(store)
        if ref["model_config"] == cfg:
            model = prior_model
            initialized_from = "previous_accepted_concept_checkpoint"
            old_checkpoint = ref["sha256"]
    if model is None:
        model = build_model(cfg)
    model.to(device)
    initial_weights = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    train_cfg = settings["train"]
    batch_size = max(1, min(int(train_cfg.get("batch_size", 8)), len(all_train), 16))
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(train_cfg.get("learning_rate", .0007)))
    start_loss = _val_loss(model, all_val, torch, device)
    before_training_loss = _val_loss(model, all_train, torch, device)
    best_loss, best_step, best_weights = float("inf"), 0, None
    actual_steps, misses = 0, 0
    interval = max(5, min(25, steps // 12))
    min_delta = max(1e-5, float(train_cfg.get("min_improvement", .001)))
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        picked = [all_train[rng.randrange(len(all_train))]
                  for _ in range(batch_size)]
        loss = _batch_loss(model, picked, torch, device)
        if not torch.isfinite(loss):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_steps = step
        if step % interval == 0 or step == steps:
            val = _val_loss(model, all_val, torch, device)
            if math.isfinite(val) and val + min_delta < best_loss:
                best_loss, best_step, misses = val, step, 0
                best_weights = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                misses += 1
                if misses >= 12:
                    break
        if time.monotonic() - start >= float(train_cfg.get("max_seconds", 600)):
            break
    if best_weights is None:
        return {"status": "rejected", "reason": "No finite validation checkpoint"}
    model.load_state_dict(best_weights)
    accepted = best_loss + min_delta < start_loss
    after_train = _val_loss(model, all_train, torch, device)
    after_val = _val_loss(model, all_val, torch, device)

    # Held-out test prompts and negative-control output are first evaluated
    # after the validation-based checkpoint is fixed. No test label updates.
    initial = build_model(cfg).to(device)
    initial.load_state_dict(initial_weights)
    before_test = _evaluate(initial, items, cfg, device, max_new_bytes)
    after_test = _evaluate(model, items, cfg, device, max_new_bytes)
    negative_prompt = f"What does {NEGATIVE_CONCEPT} mean?"
    negative_completion = _completion(model, negative_prompt,
                                      cfg["block_size"], device, max_new_bytes)

    out = store.root / "models" / "concept_lab"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out / f"concept-seed{seed}-{stamp}.pt"
    torch.save({
        "format_version": 1, "objective": FORMAT, "model_config": cfg,
        "weights": best_weights, "seed": seed,
        "dataset_sha256": digest, "concepts": [c["concept"] for c in items],
        "source_question_ids": [c["question_id"] for c in items],
        "train_templates": TRAIN_TEMPLATES,
        "validation_template": VALIDATION_TEMPLATE,
        "test_templates": TEST_TEMPLATES,
        "best_step": best_step, "best_val_answer_loss": best_loss,
        "initialized_from_checkpoint": old_checkpoint,
        "rights_scope": "independently_corroborated_reusable_verified_only",
    }, path)
    sha = sha256_file(path)
    record = {
        "path": str(path), "sha256": sha, "accepted": accepted,
        "seed": seed, "dataset_sha256": digest,
        "model_config": cfg, "concepts": [c["concept"] for c in items],
        "initialized_from": initialized_from, "objective": FORMAT,
    }
    store.put("concept_lab_latest_candidate", record)
    if accepted:
        store.put("concept_lab_latest", record)

    report = {
        "status": "trained", "experiment": FORMAT,
        "source": "verified_training_eligible_definition_records_only",
        "concepts": [c["concept"] for c in items],
        "concept_count": len(items), "training_prompt_templates": list(TRAIN_TEMPLATES),
        "validation_prompt_template": VALIDATION_TEMPLATE,
        "heldout_prompt_templates": list(TEST_TEMPLATES),
        "train_examples": len(all_train), "validation_examples": len(all_val),
        "heldout_question_count": len(_test_cases(items)),
        "initialized_from": initialized_from, "previous_checkpoint": old_checkpoint,
        "train_answer_loss_before": round(before_training_loss, 6),
        "train_answer_loss_after": round(after_train, 6),
        "validation_answer_loss_before": round(start_loss, 6),
        "validation_answer_loss_after": round(after_val, 6),
        "best_step": best_step, "actual_steps": actual_steps,
        "before_heldout": before_test, "after_heldout": after_test,
        "unknown_concept_control": {
            "question": negative_prompt, "raw_completion": negative_completion,
            "note": "No true answer is provided; this tests unsupported-answer behavior, not accuracy.",
        },
        "seed": seed, "device": device, "accepted": accepted,
        "dataset_sha256": digest, "checkpoint": str(path), "sha256": sha,
        "notes": [
            "Paraphrases of the SAME learned concepts are withheld, NOT new fact identities.",
            "Answer generation uses only Transformer weights; retrieval is not used.",
            "Selection uses validation answer-token loss, not heldout exact accuracy.",
            "Acceptance means improved validation loss, not proven concept recall.",
            "Source agreement and license checks do not guarantee factual truth.",
        ],
    }
    store.log("concept_lab", "completed", {
        **{k: v for k, v in report.items()
           if k not in ("before_heldout", "after_heldout")},
        "before_heldout": {k: v for k, v in before_test.items()
                           if k != "examples"},
        "after_heldout": {k: v for k, v in after_test.items()
                          if k != "examples"},
    })
    return report
