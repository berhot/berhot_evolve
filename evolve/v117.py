"""v1.17 independent-host language experiment, isolated from legacy checkpoints.

Five milestones: strict multi-host preparation, rights-aware archive reuse,
train-only char/BPE, qualitative + normalized evaluation, immutable history.
All models are randomly initialized and trained using PyTorch (external).
"""
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import math
import random
import re
import tempfile
import time

from .brain import build_model, load_checkpoint, get_device, require_torch, sha256_file
from .byte_bpe import (fit_byte_bpe, encode_bpe, decode_bpe, token_bytes,
                       canonical_bytes, digest, validate_bpe)
from .char_lab import build_tokenizer, encode_chars, _safe_generation
from .open_corpus import archive_page
from .web_lab import freeze_dataset, _read_frozen, _within, web_inventory

FORMAT = "berhot_host_heldout_language_v117"
KINDS = ("char", "byte_bpe")
PROMPTS = ("Energy moves through", "Describe a distant forest", "What causes rain?")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _archive_snapshot(store):
    """Hashes and index values of all earlier model families, before and after."""
    keys = ("char_lab_latest", "char_lab_latest_candidate",
            "web_lab_latest", "web_lab_latest_candidate")
    snapshots = {}
    for key in keys:
        record = store.get(key)
        if record:
            path = Path(record["checkpoint"])
            path = _within(store.root / "models", path)
            if not path.is_file():
                raise ValueError("Missing legacy model artifact: " + key)
            if sha256_file(path) != record["sha256"]:
                raise ValueError("Legacy model hash mismatch: " + key)
            snapshots[key] = {"record": record, "sha256": sha256_file(path)}
    main = store.latest_promoted()
    if main and Path(main["path"]).is_file():
        snapshots["general_promoted"] = sha256_file(main["path"])
    return snapshots


def plan(store):
    inventory = web_inventory(store, include_unknown=False)
    rows = store.db.execute(
        "SELECT rights_status, COUNT(*) FROM web_corpus_sources "
        "GROUP BY rights_status ORDER BY rights_status").fetchall()
    hosts = set()
    for item in inventory["articles"]:
        from urllib.parse import urlsplit
        hosts.add(urlsplit(item["url"]).hostname)
    return {
        "status": "ready_for_strict_freeze" if len(hosts) >= 3 else "needs_sources",
        "eligible_article_hosts": sorted(hosts),
        "eligible_article_count": inventory["candidate_article_count"],
        "required_distinct_hosts": 3,
        "missing_host_count": max(0, 3 - len(hosts)),
        "rights_declarations": {str(row[0]): int(row[1]) for row in rows},
        "excluded_by_reason": inventory["excluded_by_reason"],
        "training_rights_policy": "Only declared CC0-1.0 or PUBLIC-DOMAIN without no-AI signals; declarations require independent human rights review.",
        "example_domains_count_as_real_sources": False,
        "hint": "Queue and crawl real permitted articles with corpus-seed and corpus-crawl. An archive is not training authorization.",
    }


def prepare(store):
    result = freeze_dataset(store, include_unknown=False, diagnostic=False)
    if result["status"] != "frozen":
        return {"status": "skipped", "reason": result["reason"],
                "plan": plan(store), "no_neural_training": True}
    manifest, streams = _read_frozen(store, result["dataset_sha256"])
    if (manifest["diagnostic_source_leakage"] or
        manifest["split_method"] != "heldout_source_hosts"):
        raise ValueError("v1.17 refuses same-source diagnostic splits")
    hosts = [set(manifest["partitions"][s]["source_hosts"])
             for s in ("train", "val", "test")]
    if any(hosts[i] & hosts[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Host leakage across frozen splits")
    if any(x["rights"] not in ("CC0-1.0", "PUBLIC-DOMAIN")
           for x in manifest["eligible_articles"]):
        raise ValueError("Noneligible declared training rights")
    return {
        "status": "prepared", "dataset_sha256": result["dataset_sha256"],
        "manifest_sha256_verified": True,
        "split_method": manifest["split_method"],
        "diagnostic_source_leakage": False,
        "source_hosts": {s: sorted(hosts[i])
                         for i, s in enumerate(("train", "val", "test"))},
        "source_urls": {s: manifest["partitions"][s]["source_urls"]
                        for s in ("train", "val", "test")},
        "shard_sha256": {s: manifest["partitions"][s]["sha256"]
                         for s in ("train", "val", "test")},
        "bytes": {s: len(streams[s]) for s in ("train", "val", "test")},
        "rights_warning": "Recorded rights metadata is not independent legal verification.",
    }


def _tokenizer(store, dataset_sha, kind, train_bytes, bpe_vocab):
    if kind == "char":
        tok = build_tokenizer(train_bytes, vocab_limit=512)
    else:
        tok = fit_byte_bpe(train_bytes, vocab_size=bpe_vocab)
    content = canonical_bytes(tok)
    tok_sha = _hash(content)
    folder = store.root / "datasets" / "v117" / dataset_sha
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (kind + "-" + tok_sha + ".json")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Immutable tokenizer content changed")
    else:
        path.write_bytes(content)
    return tok, tok_sha, path


def _load_tokenizer(store, record, kind, train_bytes):
    root = store.root / "datasets" / "v117"
    path = _within(root, record["tokenizer_path"])
    content = path.read_bytes()
    if _hash(content) != record["tokenizer_sha256"]:
        raise ValueError("v1.17 tokenizer digest mismatch")
    tok = json.loads(content.decode("utf-8"))
    if kind == "byte_bpe":
        validate_bpe(tok)
        if tok["train_sha256"] != _hash(train_bytes):
            raise ValueError("BPE trained on mismatched source")
    else:
        expected = build_tokenizer(train_bytes, vocab_limit=512)
        if tok != expected:
            raise ValueError("Character tokenizer is not derived from TRAIN")
    if tok["vocab_size"] != record["model_config"]["vocab_size"]:
        raise ValueError("Model/tokenizer vocabulary mismatch")
    return tok


def _ids(raw, kind, tok):
    return (encode_chars(raw.decode("utf-8"), tok) if kind == "char"
            else encode_bpe(raw, tok))


def _eval_bytes(model, raw, kind, tok, cfg, device):
    """Deterministic prefix NLL per UTF-8 target byte, with boundary caveat.

    Both families see the same first <=2048 complete UTF-8 bytes of each shard.
    The first target token differs by tokenizer and is excluded; unlike
    per-token CE, normalized rates are directionally comparable.
    """
    torch, _ = require_torch()
    prefix = raw[:2048].decode("utf-8", errors="ignore")
    original = prefix.encode("utf-8")
    ids = _ids(original, kind, tok)
    if kind == "char":
        widths = [len(char.encode("utf-8")) for char in prefix]
        unknown = sum(value == 0 for value in ids)
    else:
        vocabulary = token_bytes(tok)
        widths = [len(vocabulary[value]) for value in ids]
        unknown = 0
    if len(ids) < 2:
        raise ValueError("Too few tokens for language evaluation")
    total_loss, count, byte_count = 0.0, 0, 0
    block = cfg["block_size"]
    model.eval()
    with torch.no_grad():
        start = 0
        while start < len(ids) - 1:
            end = min(len(ids), start + block + 1)
            x = torch.tensor([ids[start:end-1]], dtype=torch.long, device=device)
            y = torch.tensor(ids[start+1:end], dtype=torch.long, device=device)
            logits = model(x)[0]
            loss = torch.nn.functional.cross_entropy(
                logits, y, reduction="sum")
            total_loss += float(loss.item())
            count += len(y)
            byte_count += sum(widths[start+1:end])
            start = end - 1
    return {
        "cross_entropy_nats_per_token": round(total_loss / count, 5),
        "estimated_bits_per_utf8_target_byte": round(
            total_loss / (math.log(2) * byte_count), 5),
        "tokens_scored": count, "utf8_target_bytes": byte_count,
        "unknown_characters": unknown,
        "input_prefix_sha256": _hash(original),
    }


def quality_proxy(text):
    """Mechanical checks only: never label these grammar, truth or coherence."""
    lower = text.casefold()
    chunks = [lower[i:i+3] for i in range(max(0, len(lower)-2))]
    ngrams = [lower[i:i+8] for i in range(max(0, len(lower)-7))]
    return {
        "characters": len(text),
        "unique_character_fraction": round(len(set(text)) / max(len(text), 1), 4),
        "repeated_3gram_fraction": round(
            (len(chunks) - len(set(chunks))) / max(len(chunks), 1), 4),
        "distinct_8gram_fraction": round(
            len(set(ngrams)) / max(len(ngrams), 1), 4),
        "word_like_spans_not_dictionary_validated": len(re.findall(
            r"[A-Za-z]{2,}", text)),
        "terminal_punctuation_count": len(re.findall(r"[.!?؟]", text)),
        "replacement_unicode_count": text.count("\ufffd"),
        "human_coherence_required": True,
    }


def _generate(model, kind, tok, prompt, cfg, seed=42, new_tokens=48):
    torch, _ = require_torch()
    if kind == "char":
        result = _safe_generation(
            model, tok, prompt, cfg["block_size"],
            new_chars=new_tokens, temperature=0.85, seed=seed,
            top_k=min(30, tok["vocab_size"]-1))
        return result["continuation"], result["unknown_prompt_characters"]
    seq = encode_bpe(prompt.encode("utf-8"), tok)
    torch.manual_seed(seed)
    model.eval().to("cpu")
    output = []
    with torch.no_grad():
        for _ in range(new_tokens):
            x = torch.tensor([seq[-cfg["block_size"]:]], dtype=torch.long)
            logits = model(x)[0, -1] / 0.85
            values, indices = torch.topk(logits, min(30, tok["vocab_size"]))
            pick = int(torch.multinomial(torch.softmax(values, dim=-1), 1).item())
            new_id = int(indices[pick].item())
            seq.append(new_id)
            output.append(new_id)
    return decode_bpe(output, tok).decode("utf-8", errors="replace"), 0


def _report_for_model(model, streams, kind, tok, cfg, seed=42):
    model.eval().to("cpu")
    scores = {split: _eval_bytes(model, streams[split], kind, tok,
                               cfg, "cpu")
              for split in ("train", "val", "test")}
    generations = []
    for i, prompt in enumerate(PROMPTS):
        generated, unseen = _generate(model, kind, tok, prompt, cfg,
                                      seed=seed + i)
        generations.append({
            "prompt": prompt, "continuation": generated,
            "unknown_prompt_characters": unseen,
            "mechanical_quality_not_semantic": quality_proxy(generated),
        })
    return {"scores": scores, "unseen_prompt_probes": generations}


def train(store, settings, steps=120, seed=31415, device="cpu",
          bpe_vocab=320, context=64, embedding_dim=32,
          layers=1, heads=2):
    if not 1 <= int(steps) <= 5000:
        raise ValueError("steps must be 1..5000")
    if not 64 <= int(context) <= 256:
        raise ValueError("context must be 64..256")
    if (not 16 <= int(embedding_dim) <= 128 or
        not 1 <= int(layers) <= 4 or
        not 1 <= int(heads) <= 8 or int(embedding_dim) % int(heads)):
        raise ValueError("Unsupported model dimensions")
    before = _archive_snapshot(store)
    ready = prepare(store)
    if ready["status"] != "prepared":
        return ready
    manifest, streams = _read_frozen(store, ready["dataset_sha256"])
    torch, _ = require_torch()
    chosen = get_device(device)
    if chosen == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    from .training import random_batch
    candidates = {}
    for kind in KINDS:
        tok, tok_sha, tok_path = _tokenizer(
            store, ready["dataset_sha256"], kind,
            streams["train"], bpe_vocab)
        data = {split: _ids(streams[split], kind, tok)
                for split in ("train", "val", "test")}
        if any(len(data[split]) < context + 2 for split in data):
            return {"status": "skipped", "reason": "too_few_tokens_after_tokenization",
                    "kind": kind, "no_legacy_checkpoint_modified": True}
        cfg = {"vocab_size": tok["vocab_size"], "block_size": int(context),
               "embedding_dim": int(embedding_dim), "layers": int(layers),
               "heads": int(heads), "dropout": 0.0}
        torch.manual_seed(int(seed))
        rng = random.Random(int(seed))
        model = build_model(cfg).to(chosen)
        val_before = _eval_bytes(model, streams["val"], kind, tok,
                                 cfg, chosen)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["train"].get("learning_rate", 0.0007)))
        start = time.monotonic()
        actual = 0
        model.train()
        for _ in range(int(steps)):
            x, y = random_batch(data["train"], cfg["block_size"], 8,
                                torch, chosen, rng)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            actual += 1
            if time.monotonic() - start >= float(
                    settings["train"].get("max_seconds", 600)):
                break
        val_after = _eval_bytes(model, streams["val"], kind, tok,
                                cfg, chosen)
        accepted = (val_after["estimated_bits_per_utf8_target_byte"] <
                    val_before["estimated_bits_per_utf8_target_byte"] - 0.00001)
        folder = store.root / "models" / "v117"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / (kind + "-" + stamp + ".pt")
        if path.exists():
            raise ValueError("Refusing to overwrite an immutable v1.17 checkpoint")
        torch.save({
            "format_version": 1, "objective": FORMAT,
            "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": cfg, "dataset_sha256": ready["dataset_sha256"],
            "tokenizer_sha256": tok_sha, "kind": kind,
            "total_training_steps": actual, "seed": int(seed),
            "optimizer_state": optimizer.state_dict(),
            "sampler_rng_state": rng.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
        }, path)
        record = {
            "kind": kind, "checkpoint": str(path),
            "sha256": sha256_file(path), "tokenizer_path": str(tok_path),
            "tokenizer_sha256": tok_sha, "model_config": cfg,
            "dataset_sha256": ready["dataset_sha256"],
            "actual_training_steps": actual,
            "accepted_by_validation": accepted,
            "validation_bits_per_byte_before": val_before[
                "estimated_bits_per_utf8_target_byte"],
            "validation_bits_per_byte_after": val_after[
                "estimated_bits_per_utf8_target_byte"],
        }
        candidates[kind] = record
    if before != _archive_snapshot(store):
        raise ValueError("Existing model checkpoint or registry changed")
    report = {
        "status": "trained", "objective": FORMAT,
        "dataset_sha256": ready["dataset_sha256"],
        "strict_host_holdout": True, "diagnostic_source_leakage": False,
        "source_hosts": ready["source_hosts"],
        "tokenizer_training_split": "train_only",
        "candidates": candidates,
        "previous_checkpoints_unchanged": True,
        "selection": "independent per-family validation score; test never selects",
        "normalized_comparison_warning": (
            "Bits per UTF-8 target byte is a length-normalized approximation "
            "on matching source prefixes; token boundaries and OOV encoding "
            "differ, and compute per update is not matched."),
        "human_sentence_coherence_not_proven": True,
    }
    history = store.get("v117_history", []) or []
    history.append(report)
    store.put("v117_history", history)
    store.put("v117_latest", report)
    store.log("v117_train", "completed", report)
    return report


def evaluate(store):
    report = store.get("v117_latest")
    if not report:
        return {"status": "skipped", "reason": "no_v117_training_run"}
    metadata, streams = _read_frozen(store, report["dataset_sha256"])
    if metadata["diagnostic_source_leakage"] or metadata["split_method"] != "heldout_source_hosts":
        raise ValueError("v1.17 checkpoint has no strict host holdout")
    output = {}
    for kind in KINDS:
        record = report["candidates"][kind]
        path = _within(store.root / "models" / "v117",
                       record["checkpoint"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError("v1.17 checkpoint hash mismatch: " + kind)
        model, meta = load_checkpoint(path, map_location="cpu")
        if (meta.get("objective") != FORMAT or
            meta.get("kind") != kind or
            meta.get("tokenizer_sha256") != record["tokenizer_sha256"] or
            meta.get("dataset_sha256") != report["dataset_sha256"] or
            meta.get("model_config") != record["model_config"]):
            raise ValueError("v1.17 checkpoint metadata mismatch")
        tok = _load_tokenizer(store, record, kind, streams["train"])
        output[kind] = {
            "checkpoint_sha256": record["sha256"],
            "tokenizer_sha256": record["tokenizer_sha256"],
            "actual_training_steps": record["actual_training_steps"],
            "accepted_by_validation": record["accepted_by_validation"],
            **_report_for_model(model, streams, kind, tok,
                                record["model_config"]),
        }
    return {
        "status": "evaluated", "dataset_sha256": report["dataset_sha256"],
        "split_method": metadata["split_method"],
        "diagnostic_source_leakage": False,
        "source_hosts": report["source_hosts"],
        "models": output,
        "selection_used_test": False,
        "no_neural_training_in_evaluate": True,
        "warning": "Generated text requires human coherence and factual evaluation; automatic pattern scores are not semantic understanding. Both tokenizers cover slightly different initial prediction boundaries.",
    }


def smoke(settings, steps=8):
    """Completely isolated self-authored synthetic three-host CI/demo corpus."""
    topics = (
        ("physics", "energy kinetic motion radiation temperature electricity"),
        ("botany", "leaves roots soil stems forests oxygen chlorophyll"),
        ("geography", "rivers deserts mountains rainfall valleys climate"),
    )
    with tempfile.TemporaryDirectory(prefix="berhot-v117-smoke-") as folder:
        from .store import Store
        store = Store(folder)
        try:
            for i, (subject, terms) in enumerate(topics):
                words = terms.split()
                paragraphs = []
                for n in range(160):
                    rotation = words[n % len(words):] + words[:n % len(words)]
                    paragraphs.append(
                        f"{subject.title()} lesson {n} observes " +
                        " ".join(rotation) +
                        f" with measurement set {n*n+11}. " +
                        f"Careful investigators compare sample {n+1} " +
                        "against independent evidence, then report uncertain results.")
                article = " ".join(paragraphs)
                archive_page(
                    store, f"https://fictional-host-{i}.example.org/articles/{subject}",
                    "<html><title>" + subject + "</title><body>" +
                    article + "</body></html>", article,
                    title=subject, declared_license="CC0-1.0",
                    rights_url="fixture:self-authored")
            cfg = copy.deepcopy(settings)
            cfg["storage_root"] = folder
            cfg["train"]["max_seconds"] = 100
            pre = prepare(store)
            if pre["status"] != "prepared":
                return {"status": "skipped", "synthetic_fixture_only": True,
                        "reason": pre}
            trained = train(store, cfg, steps=int(steps), device="cpu")
            tested = evaluate(store) if trained["status"] == "trained" else trained
            return {
                "status": "smoke_test_complete",
                "synthetic_fixture_only": True,
                "real_world_generalization_demonstrated": False,
                "frozen_dataset_sha256": pre["dataset_sha256"],
                "strict_host_holdout": True,
                "training": trained,
                "evaluation": tested,
                "previous_user_model_files_touched": False,
            }
        finally:
            store.close()
