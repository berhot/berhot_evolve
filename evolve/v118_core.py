"""BERHOT EVOLVE v1.18: inspectable data, supervised lessons and honest chat.

No external LLM API. Retrieval is explicitly labeled retrieval; neural output
is from local BERHOT weights and may be gibberish. All trainable examples require
a previously archived rights-reviewed source and human-provided QA.
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
import unicodedata

from .brain import build_model, load_checkpoint, require_torch, sha256_file, sample_text
from .store import Store
from .v117 import (plan as source_plan, prepare, _read_frozen, _load_tokenizer,
                   _generate, _within, FORMAT as V117_FORMAT)
from .web_lab import _canonical_bytes

QA_FORMAT = "berhot_supervised_answer_v118"
STOPWORDS = frozenset(
    "what which where when why how who does did do can is are was were "
    "a an the to for of and or in at on from about explain please tell me "
    "hello hi thank you what is describe".split())


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _words(value):
    return set(re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)) - STOPWORDS


def _approved_text(store, url):
    row = store.db.execute(
        "SELECT s.url,s.title,s.host,s.rights_status,s.noai,s.sha256,"
        "t.clean_path,t.words,t.characters FROM web_corpus_sources s "
        "JOIN web_corpus_texts t ON s.sha256=t.sha256 WHERE s.url=?",
        (url,)).fetchone()
    if not row or row["rights_status"] not in ("CC0-1.0", "PUBLIC-DOMAIN") or row["noai"]:
        raise ValueError("Source is not archived and explicitly rights-reviewed")
    file = _within(store.root / "web_corpus" / "clean", row["clean_path"])
    data = file.read_bytes()
    if _sha(data) != row["sha256"]:
        raise ValueError("Archived source text failed SHA256 verification")
    return dict(row), data.decode("utf-8", errors="strict")


def curriculum(store):
    """Shows full rights and host diversity; never counts pages as understanding."""
    source = source_plan(store)
    rows = store.db.execute(
        "SELECT s.url,s.title,s.host,s.rights_status,s.noai,t.words,t.characters "
        "FROM web_corpus_sources s JOIN web_corpus_texts t "
        "ON s.sha256=t.sha256 ORDER BY s.url").fetchall()
    eligible = [dict(r) for r in rows if
                r["rights_status"] in ("PUBLIC-DOMAIN", "CC0-1.0")
                and not r["noai"]]
    hosts = sorted(set(x["host"] for x in eligible))
    count = len(eligible)
    staged = prepare(store) if len(source["eligible_article_hosts"]) >= 3 else {
        "status": "skipped", "reason": "needs_more_eligible_hosts"}
    return {
        "status": "planned", "archived_articles": len(rows),
        "eligible_articles": count, "eligible_hosts": hosts,
        "approved_text_characters": sum(r["characters"] for r in eligible),
        "unreviewed_articles": len(rows) - count,
        "ready_for_independent_training": staged["status"] == "prepared",
        "frozen_dataset_sha256": staged.get("dataset_sha256"),
        "split_source_hosts": staged.get("source_hosts", {}),
        "bytes_per_split": staged.get("bytes", {}),
        "manual_source_review_required": True,
        "progress_goals": {
            "pilot_hosts": {"current": len(hosts), "target": 3},
            "diverse_articles": {"current": count, "target": 100},
            "training_text_characters": {
                "current": sum(r["characters"] for r in eligible),
                "target": 1000000},
            "human_reviewed_qa": {
                "current": len(store.get("v118_qa_examples", []) or []),
                "target": 20},
        },
        "note": "100 articles and 1M characters are learning-roadmap targets, not proof of fluent language or sufficient data for a production model.",
    }


def answer_from_sources(store, question, limit=3):
    """Literal, cited excerpts from allowed stored articles; NOT neural QA."""
    terms = _words(question)
    if not terms:
        return {
            "mode": "source_lookup_not_neural", "answer":
                "I can search my approved science articles. Ask about stars, volcanoes or corals.",
            "sources": [], "learning": False}
    matches = []
    rows = store.db.execute(
        "SELECT url FROM web_corpus_sources "
        "WHERE rights_status IN ('CC0-1.0','PUBLIC-DOMAIN') AND noai=0 "
        "ORDER BY url LIMIT 250").fetchall()
    for row in rows:
        source, text = _approved_text(store, row["url"])
        # Sentence boundaries are approximate on unstructured HTML excerpts.
        sentences = re.split(r"(?<=[.!?؟])\s+|\n+", text)
        ranked = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not 35 <= len(sentence) <= 900:
                continue
            overlap = terms & _words(sentence)
            if overlap:
                ranked.append((len(overlap), sentence))
        if ranked:
            score, best = max(ranked, key=lambda x: (x[0], -len(x[1])))
            matches.append((score, source, best))
    matches.sort(key=lambda v: (-v[0], v[1]["url"]))
    citations = [
        {"title": src["title"], "url": src["url"], "host": src["host"],
         "excerpt": text[:300], "matched_terms": score}
        for score, src, text in matches[:max(1, min(int(limit), 3))]]
    if not citations:
        return {
            "mode": "source_lookup_not_neural",
            "answer": "I could not find a matching passage in the approved local articles. This is not a neural answer.",
            "sources": [], "learning": False}
    return {
        "mode": "source_lookup_not_neural",
        "answer": citations[0]["excerpt"],
        "sources": citations, "learning": False,
        "warning": "An extract from an approved archived source, not a generated explanation or proof that BERHOT understands it.",
    }


def neural_response(store, prompt, kind="byte_bpe", seed=42):
    """Exactly load and SHA-check the model created by v1.17 before sampling."""
    if kind not in ("char", "byte_bpe"):
        raise ValueError("Only the two local v1.17 neural families are available")
    report = store.get("v117_latest")
    if not report:
        return {"mode": "neural_raw", "status": "no_model",
                "answer": "No v1.17 neural checkpoint exists yet."}
    record = report["candidates"][kind]
    file = _within(store.root / "models" / "v117", record["checkpoint"])
    if sha256_file(file) != record["sha256"]:
        raise ValueError("Neural checkpoint SHA256 mismatch")
    model, meta = load_checkpoint(file, map_location="cpu")
    if (meta.get("objective") != V117_FORMAT or
        meta.get("kind") != kind or
        meta.get("dataset_sha256") != report["dataset_sha256"] or
        meta.get("tokenizer_sha256") != record["tokenizer_sha256"]):
        raise ValueError("Neural checkpoint metadata mismatch")
    manifest, streams = _read_frozen(store, record["dataset_sha256"])
    tok = _load_tokenizer(store, record, kind, streams["train"])
    result, missing = _generate(model, kind, tok, prompt,
                                record["model_config"], seed=seed,
                                new_tokens=48)
    return {
        "mode": "neural_raw", "status": "generated", "model": kind,
        "answer": result, "prompt": prompt,
        "checkpoint_sha256": record["sha256"],
        "unknown_prompt_characters": missing,
        "learning": False,
        "warning": "Actual BERHOT neural continuation, not source retrieval. Current models are tiny and text is not expected to be coherent or factually correct.",
    }


def add_qa_example(store, question, answer, url):
    """Only literal source-backed answers, paired with operator-written questions."""
    if not isinstance(question, str) or not 8 <= len(question.strip()) <= 180:
        raise ValueError("Question must be 8..180 characters")
    if not isinstance(answer, str) or not 15 <= len(answer.strip()) <= 220:
        raise ValueError("Answer must be 15..220 characters")
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Source URL required")
    source, text = _approved_text(store, url)
    if unicodedata.normalize("NFC", answer.strip()) not in unicodedata.normalize("NFC", text):
        raise ValueError("Answer must be an exact quotation from the approved archived article")
    example = {
        "question": question.strip(), "answer": answer.strip(),
        "url": url, "host": source["host"],
        "source_sha256": source["sha256"], "reviewed_at_utc": _utc(),
        "source_provenance": "human_supplied_question_exact_archived_quote",
    }
    examples = store.get("v118_qa_examples", []) or []
    fingerprint = _sha(_canonical_bytes([
        example["question"].casefold(), example["answer"], example["url"]]))
    if any(e.get("fingerprint") == fingerprint for e in examples):
        return {"status": "duplicate", "fingerprint": fingerprint}
    if len(examples) >= 5000:
        raise ValueError("QA review queue full")
    example["fingerprint"] = fingerprint
    examples.append(example)
    store.put("v118_qa_examples", examples)
    store.log("v118_qa_approval", "added", {"url": url, "fingerprint": fingerprint})
    return {"status": "added", "fingerprint": fingerprint,
            "total_reviewed_qa": len(examples), "host": source["host"]}


def _qa_partitions(store):
    frozen = prepare(store)
    if frozen["status"] != "prepared":
        return None, frozen
    split_hosts = {h: split for split, hosts in frozen["source_hosts"].items()
                   for h in hosts}
    parts = {"train": [], "val": [], "test": []}
    for row in store.get("v118_qa_examples", []) or []:
        src, text = _approved_text(store, row["url"])
        if src["sha256"] != row["source_sha256"] or row["answer"] not in text:
            raise ValueError("Review-locked QA source changed")
        split = split_hosts.get(row["host"])
        if split:
            parts[split].append(row)
    return parts, frozen


def qa_readiness(store):
    parts, frozen = _qa_partitions(store)
    if parts is None:
        return {"status": "needs_sources", "detail": frozen}
    counts = {k: len(v) for k, v in parts.items()}
    ready = counts["train"] >= 8 and counts["val"] >= 2 and counts["test"] >= 2
    return {
        "status": "ready" if ready else "needs_reviewed_examples",
        "counts": counts, "minimum": {"train": 8, "val": 2, "test": 2},
        "source_hosts": frozen["source_hosts"],
        "warning": "Human-written questions and exact source answers are training examples, not verified semantic understanding.",
    }


def _qa_xy(row, block):
    prefix = ("Question: " + row["question"] + "\nAnswer: ").encode("utf-8")
    full = prefix + row["answer"].encode("utf-8") + b"\n"
    if len(full) > block + 1:
        return None
    # Predictions for only the answer suffix, not for the prompt.
    x = list(full[:-1]) + [0] * (block + 1 - len(full))
    y = [-100] * block
    for pos in range(len(prefix) - 1, len(full) - 1):
        y[pos] = full[pos + 1]
    return x, y


def _qa_loss(model, examples, torch, device):
    xs = torch.tensor([e[0] for e in examples], device=device)
    ys = torch.tensor([e[1] for e in examples], device=device)
    logits = model(xs)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), ys.reshape(-1), ignore_index=-100)


def train_qa(store, settings, steps=80, seed=31415, progress=None):
    """Separate new tiny QA model: no legacy weights or source text modifications."""
    if not 1 <= int(steps) <= 500:
        raise ValueError("QA training steps must be 1..500")
    parts, frozen = _qa_partitions(store)
    if parts is None:
        return {"status": "skipped", "reason": "strict_host_holdout_missing"}
    ready = qa_readiness(store)
    if ready["status"] != "ready":
        return {"status": "skipped", "reason": "insufficient_human_verified_qa",
                "readiness": ready}
    block = 256
    samples = {split: [
        encoded for row in rows if (encoded := _qa_xy(row, block)) is not None]
        for split, rows in parts.items()}
    if any(len(samples[k]) < {"train": 8, "val": 2, "test": 2}[k] for k in samples):
        return {"status": "skipped",
                "reason": "qa_examples_exceed_256_byte_context", "usable": {
                    k: len(v) for k, v in samples.items()}}
    qa_manifest = {
        "format": "human_reviewed_qa_split_by_source_host_v118",
        "examples": {k: [{f: r[f] for f in (
            "question", "answer", "url", "source_sha256")}
            for r in parts[k]] for k in parts},
        "dataset_sha256": frozen["dataset_sha256"],
        "split_method": "source_host_holdout",
    }
    manifest_bytes = _canonical_bytes(qa_manifest)
    fingerprint = _sha(manifest_bytes)
    manifest_dir = store.root / "datasets" / "v118_qa" / fingerprint
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Frozen QA dataset manifest changed")
    else:
        manifest_path.write_bytes(manifest_bytes)
    torch, _ = require_torch()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(int(seed))
    rng = random.Random(int(seed))
    cfg = {"block_size": block, "vocab_size": 256, "embedding_dim": 48,
           "heads": 4, "layers": 2, "dropout": 0.0}
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007)

    def score(rows):
        model.eval()
        with torch.no_grad():
            return float(_qa_loss(model, rows, torch, "cpu").item())

    start = time.monotonic()
    val_before = score(samples["val"])
    actual = 0
    for step in range(int(steps)):
        model.train()
        minibatch = [samples["train"][rng.randrange(len(samples["train"]))]
                     for _ in range(4)]
        loss = _qa_loss(model, minibatch, torch, "cpu")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual += 1
        if progress and (actual == 1 or actual % 10 == 0 or actual == steps):
            progress("qa", actual, int(steps), round(float(loss.item()), 5))
        if time.monotonic() - start >= float(settings["train"].get("max_seconds", 600)):
            break
    val_after = score(samples["val"])
    accepted = math.isfinite(val_after) and val_after < val_before - 0.0001
    # Inspect the held-out test only after validation acceptance is decided.
    test_after = score(samples["test"])
    folder = store.root / "models" / "v118_qa"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint = folder / ("qa-" + stamp + ".pt")
    torch.save({
        "format_version": 1, "objective": QA_FORMAT, "weights": {
            k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": cfg, "qa_manifest_sha256": fingerprint,
        "dataset_sha256": frozen["dataset_sha256"],
        "optimizer_state": optimizer.state_dict(), "sampler_rng_state": rng.getstate(),
        "actual_training_steps": actual, "seed": int(seed),
    }, checkpoint)
    result = {
        "status": "trained", "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint), "actual_training_steps": actual,
        "qa_manifest_sha256": fingerprint,
        "qa_manifest_path": str(manifest_path),
        "source_dataset_sha256": frozen["dataset_sha256"],
        "train_examples": len(samples["train"]),
        "val_examples": len(samples["val"]), "test_examples": len(samples["test"]),
        "validation_answer_loss_before": round(val_before, 5),
        "validation_answer_loss_after": round(val_after, 5),
        "test_answer_loss_after": round(test_after, 5),
        "accepted_by_validation": accepted,
        "source_hosts": frozen["source_hosts"],
        "test_used_for_selection": False,
        "previous_model_families_untouched": True,
    }
    # Checkpoint selection is validation-only; if rejected, do not make it the
    # default chat model but keep immutable candidate and history.
    history = store.get("v118_qa_history", []) or []
    history.append(result)
    store.put("v118_qa_history", history)
    store.put("v118_qa_latest_candidate", result)
    if accepted:
        store.put("v118_qa_latest", result)
    store.log("v118_qa_train", "completed", result)
    return result


def qa_neural_response(store, question):
    record = store.get("v118_qa_latest")
    if not record:
        return {"mode": "qa_neural_raw", "status": "no_accepted_qa_model",
                "answer": "No accepted supervised QA model yet. Add reviewed lessons and train QA."}
    path = _within(store.root / "models" / "v118_qa",
                   record["checkpoint"])
    qa_manifest_file = _within(store.root / "datasets" / "v118_qa",
                               record["qa_manifest_path"])
    raw_manifest = qa_manifest_file.read_bytes()
    if _sha(raw_manifest) != record["qa_manifest_sha256"]:
        raise ValueError("QA dataset manifest SHA256 mismatch")
    if sha256_file(path) != record["sha256"]:
        raise ValueError("QA checkpoint SHA256 mismatch")
    model, meta = load_checkpoint(path, map_location="cpu")
    if (meta.get("objective") != QA_FORMAT or
        meta.get("qa_manifest_sha256") != record["qa_manifest_sha256"] or
        meta.get("dataset_sha256") != record["source_dataset_sha256"]):
        raise ValueError("QA checkpoint metadata mismatch")
    prompt = "Question: " + question + "\nAnswer: "
    if len(prompt.encode("utf-8")) > meta["model_config"]["block_size"] - 2:
        raise ValueError("Question is too long for tiny QA context")
    generated = sample_text(
        model, prompt, meta["model_config"]["block_size"],
        max_new_bytes=65, seed=42)
    continuation = generated[len(prompt):].split("\n")[0]
    return {
        "mode": "qa_neural_raw", "status": "generated",
        "answer": continuation, "checkpoint_sha256": record["sha256"],
        "learning": False,
        "warning": "Raw locally trained supervised model; no guarantee of coherent or truthful answers. Default source lookup remains the reliable display mode.",
    }


def ask(store, question, mode="sources"):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 280:
        raise ValueError("Question must be 1..280 characters")
    question = question.strip()
    if mode == "sources":
        return answer_from_sources(store, question)
    if mode in ("char", "byte_bpe"):
        return neural_response(store, question, kind=mode)
    if mode == "qa":
        return qa_neural_response(store, question)
    raise ValueError("Unknown chat mode")
