"""BERHOT EVOLVE v1.15: own Unicode character tokenizer + resumable neural lab.

This is deliberately a separate experiment from the existing BYTE web lab.
The tokenizer is pure Python standard library, fitted from TRAIN SPLIT ONLY.
The neural layers / autograd continue to use the explicitly declared PyTorch
dependency; no claim of BERHOT-owned GPU/tensor/autodiff implementation.

The design addresses invalid UTF-8 byte completions, not the harder unsolved
problem of language understanding. Decoding samples entire Unicode characters
and suppresses unknown/control tokens. It cannot guarantee fluent sentences.

The source-host and diagnostic same-article evaluation semantics, rights
metadata and frozen dataset are inherited from web_lab. No networking here.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import random
import time
import unicodedata

from .brain import (
    build_model, get_device, load_checkpoint, require_torch, sha256_file,
)
from .web_lab import (
    _canonical_bytes, _read_frozen, _within, freeze_dataset,
)

FORMAT = "berhot_unicode_char_lm_v1"
TOKENIZER_FORMAT = "berhot_train_only_unicode_chars_v1"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def build_tokenizer(train_bytes, vocab_limit=512):
    """A reproducible train-only Unicode character vocabulary.

    id 0 is a reserved unknown token; there is no proprietary tokenizer
    dependency or hidden use of a third-party pretrained vocabulary.
    """
    vocab_limit = int(vocab_limit)
    if not 32 <= vocab_limit <= 2048:
        raise ValueError("vocab_limit must be 32..2048")
    text = train_bytes.decode("utf-8", errors="strict")
    if not text:
        raise ValueError("Empty train split")
    freqs = Counter(text)
    by_frequency = sorted(freqs, key=lambda char: (-freqs[char], ord(char)))
    chars = by_frequency[:vocab_limit - 1]
    # Stable IDs for a particular frozen corpus, independent of dict order.
    tokenizer = {
        "format": TOKENIZER_FORMAT,
        "unk_id": 0,
        "chars": chars,
        "vocab_size": len(chars) + 1,
        "vocab_limit": vocab_limit,
        "train_chars_seen": len(freqs),
        "train_chars_retained": len(chars),
    }
    return tokenizer


def encode_chars(text, tokenizer):
    if not isinstance(text, str):
        raise TypeError("Expected Unicode text")
    mapping = {character: i + 1 for i, character
               in enumerate(tokenizer["chars"])}
    result = [mapping.get(char, 0) for char in text]
    return result


def decode_chars(token_ids, tokenizer):
    chars = tokenizer["chars"]
    return "".join(chars[token - 1] if 1 <= token <= len(chars)
                   else "\ufffd" for token in token_ids)


def _tokenizer_path(store, dataset_sha, tokenizer):
    serialized = _canonical_bytes(tokenizer)
    tok_sha = _digest(serialized)
    base = (store.root / "datasets" / "char_lab" / dataset_sha).resolve()
    base.mkdir(parents=True, exist_ok=True)
    path = base / ("tokenizer-" + tok_sha + ".json")
    if path.exists():
        if path.read_bytes() != serialized:
            raise ValueError("Existing immutable tokenizer changed")
    else:
        path.write_bytes(serialized)
    return str(path), tok_sha


def _load_tokenizer(store, dataset_sha, tokenizer_sha):
    if (not isinstance(dataset_sha, str) or len(dataset_sha) != 64 or
        not isinstance(tokenizer_sha, str) or len(tokenizer_sha) != 64 or
        any(c not in "0123456789abcdef" for c in dataset_sha + tokenizer_sha)):
        raise ValueError("Malformed dataset/tokenizer SHA256")
    base = store.root / "datasets" / "char_lab"
    path = _within(base, base / dataset_sha /
                   ("tokenizer-" + tokenizer_sha + ".json"))
    raw = path.read_bytes()
    if _digest(raw) != tokenizer_sha:
        raise ValueError("Tokenizer SHA256 mismatch")
    tokenizer = json.loads(raw.decode("utf-8"))
    if (tokenizer.get("format") != TOKENIZER_FORMAT
            or tokenizer.get("unk_id") != 0
            or tokenizer.get("vocab_size") != len(tokenizer.get("chars", [])) + 1
            or len(set(tokenizer["chars"])) != len(tokenizer["chars"])):
        raise ValueError("Invalid train-only Unicode tokenizer metadata")
    return tokenizer


def _cfg(settings, tokenizer, context=128, embedding_dim=64, layers=2, heads=4):
    context = int(context)
    embedding_dim = int(embedding_dim)
    layers = int(layers)
    heads = int(heads)
    if not 64 <= context <= 256:
        raise ValueError("context must be 64..256 Unicode characters")
    if not 1 <= layers <= 4 or not 1 <= heads <= 8:
        raise ValueError("layers must be 1..4, heads 1..8")
    if not 16 <= embedding_dim <= 128 or embedding_dim % heads:
        raise ValueError("embedding_dim must be 16..128 and divisible by heads")
    return {
        "block_size": context, "embedding_dim": embedding_dim,
        "heads": heads, "layers": layers, "dropout": 0.0,
        "vocab_size": tokenizer["vocab_size"],
    }


def _eval(model, ids, cfg, device, seed=1901, batches=3):
    torch, _ = require_torch()
    model.eval()
    from .training import random_batch
    rng = random.Random(seed)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = random_batch(
                ids, cfg["block_size"], 4, torch, device, rng)
            logits = model(x)
            losses.append(float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1)
            ).item()))
    return sum(losses) / len(losses)


def _unigram(ids_train, ids_eval, cfg, seed=1901, batches=3):
    from collections import Counter
    from math import log
    counts = Counter(ids_train)
    total = len(ids_train) + cfg["vocab_size"]
    costs = [
        -log((counts.get(k, 0) + 1) / total)
        for k in range(cfg["vocab_size"])
    ]
    rng = random.Random(seed)
    block = cfg["block_size"]
    max_start = len(ids_eval) - block - 1
    loss = 0.0
    n = 0
    for _ in range(batches):
        for _ in range(4):
            start = rng.randint(0, max_start)
            for token in ids_eval[start + 1:start + block + 1]:
                loss += costs[token]
                n += 1
    return loss / n


def _encode_all(tokenizer, streams, context):
    values = {}
    unknown = {}
    for split in ("train", "val", "test"):
        raw = streams[split].decode("utf-8", errors="strict")
        values[split] = encode_chars(raw, tokenizer)
        if len(values[split]) < context + 2:
            raise ValueError(f"{split} requires >= context+2 characters")
        missing = sum(i == 0 for i in values[split])
        unknown[split] = {
            "unknown_characters": missing,
            "unknown_fraction": round(
                missing / len(values[split]), 5),
            "total_characters": len(values[split]),
        }
    return values, unknown


def _record_key(candidate):
    return "char_lab_latest_candidate" if candidate else "char_lab_latest"


def _load_char_model(store, candidate=True):
    report = store.get(_record_key(candidate))
    if report is None:
        return None
    path = _within(store.root / "models" / "char_lab", report["checkpoint"])
    if not path.is_file() or sha256_file(path) != report["sha256"]:
        raise ValueError("Unicode model checkpoint SHA256 mismatch")
    tokenizer = _load_tokenizer(
        store, report["dataset_sha256"], report["tokenizer_sha256"])
    model, metadata = load_checkpoint(path, map_location="cpu")
    if (metadata.get("objective") != FORMAT
            or metadata.get("dataset_sha256") != report["dataset_sha256"]
            or metadata.get("tokenizer_sha256") != report["tokenizer_sha256"]
            or metadata.get("model_config", {}).get("vocab_size")
            != tokenizer["vocab_size"]):
        raise ValueError("Unicode model metadata mismatch")
    return model, metadata, report, tokenizer


def train_char_lab(store, settings, steps=400, seed=31415, device="cpu",
                   include_unknown=False, diagnostic=False, resume=False,
                   context=128, embedding_dim=64, layers=2, heads=4,
                   vocab_limit=512):
    steps = int(steps)
    if not 1 <= steps <= 5000:
        raise ValueError("steps must be 1..5000")
    # Snapshot the existing GENERAL model; this separate lab must never
    # mutate it. Do not confuse a newly trained Unicode model with old ones.
    existing_main = store.latest_promoted()
    main_sha_before = (
        sha256_file(existing_main["path"]) if
        existing_main and Path(existing_main["path"]).is_file() else None)
    frozen = freeze_dataset(store, include_unknown, diagnostic)
    if frozen["status"] != "frozen":
        return frozen
    manifest, streams = _read_frozen(store, frozen["dataset_sha256"])
    tokenizer = build_tokenizer(streams["train"], vocab_limit=vocab_limit)
    tokenizer_path, tokenizer_sha = _tokenizer_path(
        store, frozen["dataset_sha256"], tokenizer)
    cfg = _cfg(settings, tokenizer, context, embedding_dim, layers, heads)
    ids, unknown = _encode_all(tokenizer, streams, cfg["block_size"])
    torch, _ = require_torch()
    chosen = get_device(device)
    if chosen == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(int(seed))
    rng = random.Random(int(seed))
    previous = None
    if resume:
        previous = _load_char_model(store, candidate=False)
        if previous is None:
            return {"status": "skipped", "reason": "no_accepted_char_checkpoint_to_resume"}
        model, old_meta, old_record, old_tokenizer = previous
        if (old_meta["model_config"] != cfg
                or old_record["dataset_sha256"] != frozen["dataset_sha256"]
                or old_record["tokenizer_sha256"] != tokenizer_sha):
            return {
                "status": "skipped",
                "reason": "incompatible_previous_dataset_or_model_configuration",
            }
        old_checkpoint_sha = old_record["sha256"]
        model = model.to(chosen)
    else:
        model = build_model(cfg).to(chosen)
        old_checkpoint_sha = None

    train_before = _eval(model, ids["train"], cfg, chosen)
    val_before = _eval(model, ids["val"], cfg, chosen)
    base_val = _unigram(ids["train"], ids["val"], cfg)
    base_test = _unigram(ids["train"], ids["test"], cfg, seed=398)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(
            settings["train"].get("learning_rate", 0.0007)))
    from .training import random_batch
    actual = 0
    start = time.monotonic()
    model.train()
    for _ in range(steps):
        x, y = random_batch(
            ids["train"], cfg["block_size"], 8, torch, chosen, rng)
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
    train_after = _eval(model, ids["train"], cfg, chosen)
    val_after = _eval(model, ids["val"], cfg, chosen)
    # Only the validation split is involved in candidate acceptance.
    accepted = bool(
        math.isfinite(val_after) and
        val_after + 0.0001 < val_before)
    output = store.root / "models" / "char_lab"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint = output / f"candidate-{stamp}.pt"
    torch.save({
        "format_version": 1, "objective": FORMAT,
        "weights": {
            k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": cfg,
        "dataset_sha256": frozen["dataset_sha256"],
        "tokenizer_sha256": tokenizer_sha,
        "actual_training_steps": actual,
        "total_training_steps": actual + (
            int(previous[2]["total_training_steps"]) if previous else 0),
        "seed": int(seed),
    }, checkpoint)
    sha = sha256_file(checkpoint)
    # Withheld test is scored AFTER the choice to accept or reject.
    test_after = _eval(model, ids["test"], cfg, chosen, seed=398)
    main_sha_after = (
        sha256_file(existing_main["path"]) if
        existing_main and Path(existing_main["path"]).is_file() else None)
    if main_sha_before != main_sha_after:
        raise ValueError("Existing general model checkpoint changed during char lab")
    report = {
        "status": "trained", "objective": FORMAT,
        "checkpoint": str(checkpoint), "sha256": sha,
        "dataset_sha256": frozen["dataset_sha256"],
        "frozen_manifest": frozen["manifest"],
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_path": tokenizer_path,
        "tokenizer_owner": "BERHOT custom Python Unicode characters",
        "neural_backend": "PyTorch external backend",
        "vocab_size": tokenizer["vocab_size"],
        "model_config": cfg,
        "actual_steps": actual,
        "total_training_steps": actual + (
            int(previous[2]["total_training_steps"]) if previous else 0),
        "resumed_from_sha256": old_checkpoint_sha,
        "accepted_by_validation": accepted,
        "split_method": manifest["split_method"],
        "diagnostic_source_leakage": bool(
            manifest["diagnostic_source_leakage"]),
        "unknown_character_rates": unknown,
        "train_char_loss_before": round(train_before, 5),
        "train_char_loss_after": round(train_after, 5),
        "validation_char_loss_before": round(val_before, 5),
        "validation_char_loss_after": round(val_after, 5),
        "heldout_test_char_loss_after": round(test_after, 5),
        "unigram_validation_char_loss": round(base_val, 5),
        "unigram_test_char_loss": round(base_test, 5),
        "beats_unigram_on_validation": val_after < base_val,
        "beats_unigram_on_test": test_after < base_test,
        "test_used_for_selection": False,
        "existing_general_checkpoint_unchanged": (
            main_sha_before == main_sha_after
            if main_sha_before is not None else "no_existing_general_checkpoint"),
        "existing_arithmetic_checkpoints_modified": "not_accessed_by_char_lab",
        "note": (
            "Cross-entropy is measured per Unicode character, not per "
            "byte. Do NOT directly compare this number with v1.14 byte "
            "cross-entropy; no claim of factual knowledge."
        ),
        "warning": (
            "WITHIN-ARTICLE DIAGNOSTIC: splits share an article; not "
            "independent generalization."
            if manifest["diagnostic_source_leakage"] else
            "Source-host holdout: not a factual QA benchmark."
        ),
    }
    if main_sha_after is not None:
        report["existing_main_checkpoint_sha256"] = main_sha_after
    store.put("char_lab_latest_candidate", report)
    if accepted:
        store.put("char_lab_latest", report)
    store.log("char_lab_training", "completed", report)
    return report


def _safe_generation(model, tokenizer, prompt, block, new_chars=100,
                     temperature=0.85, seed=42, top_k=30, greedy=False):
    torch, _ = require_torch()
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be nonempty")
    if not 1 <= int(new_chars) <= 300:
        raise ValueError("new_chars must be 1..300")
    if not 0.05 <= float(temperature) <= 2.0:
        raise ValueError("temperature must be 0.05..2.0")
    if not 1 <= int(top_k) <= tokenizer["vocab_size"]:
        raise ValueError("top_k outside vocabulary")
    start = encode_chars(prompt, tokenizer)
    device = "cpu"
    torch.manual_seed(int(seed))
    model.eval().to(device)
    tokens = list(start)
    new = []
    # Never sample unknown token or Unicode controls; newline is allowed.
    forbidden = {0}
    for i, symbol in enumerate(tokenizer["chars"], 1):
        if ((not symbol.isprintable() and symbol not in ("\n", "\t"))
                or symbol == "\ufffd"):
            forbidden.add(i)
    with torch.no_grad():
        for _ in range(int(new_chars)):
            x = torch.tensor([tokens[-block:]],
                             dtype=torch.long, device=device)
            logits = model(x)[0, -1].clone()
            for bad in forbidden:
                logits[bad] = -float("inf")
            logits = logits / float(temperature)
            if greedy:
                token = int(logits.argmax().item())
            else:
                count = min(int(top_k), int(logits.numel()) - len(forbidden))
                top_values, top_ids = torch.topk(logits, count)
                probabilities = torch.softmax(top_values, dim=-1)
                sample = int(torch.multinomial(probabilities, 1).item())
                token = int(top_ids[sample].item())
            tokens.append(token)
            new.append(token)
    return {
        "text": prompt + decode_chars(new, tokenizer),
        "unknown_prompt_characters": sum(i == 0 for i in start),
        "generated_characters": len(new),
        "unique_generated_characters": len(set(new)),
        "continuation": decode_chars(new, tokenizer),
    }


def predict_char_lab(store, prompt, candidate=True, new_chars=100,
                     temperature=0.85, seed=42, top_k=30, greedy=False):
    loaded = _load_char_model(store, candidate)
    if loaded is None:
        return {"status": "skipped", "reason": "No char-model checkpoint yet"}
    model, meta, record, tokenizer = loaded
    gen = _safe_generation(
        model, tokenizer, prompt, meta["model_config"]["block_size"],
        new_chars=new_chars, temperature=temperature, seed=seed,
        top_k=top_k, greedy=greedy)
    return {
        "status": "predicted",
        "checkpoint_sha256": record["sha256"],
        "tokenizer_sha256": record["tokenizer_sha256"],
        "dataset_sha256": record["dataset_sha256"],
        "decoding": "greedy" if greedy else "top_k_sample",
        "source_lookup": False,
        **gen,
        "warning": (
            "Generated Unicode characters are valid but not necessarily "
            "meaningful words, correct facts or grammatical English."
        ),
    }


def evaluate_char_lab(store, candidate=True):
    loaded = _load_char_model(store, candidate)
    if loaded is None:
        return {"status": "skipped", "reason": "No char-model checkpoint yet"}
    model, meta, record, tokenizer = loaded
    frozen, streams = _read_frozen(store, record["dataset_sha256"])
    cfg = meta["model_config"]
    ids, unknown = _encode_all(tokenizer, streams, cfg["block_size"])
    return {
        "status": "evaluated", "checkpoint_sha256": record["sha256"],
        "tokenizer_sha256": record["tokenizer_sha256"],
        "dataset_sha256": record["dataset_sha256"],
        "split_method": frozen["split_method"],
        "diagnostic_source_leakage": bool(
            frozen["diagnostic_source_leakage"]),
        "character_cross_entropy_nats": {
            split: round(_eval(model, ids[split], cfg, "cpu",
                               seed=398 if split == "test" else 1901), 5)
            for split in ("train", "val", "test")
        },
        "unknown_character_rates": unknown,
        "note": (
            "Characters are Unicode code points, not bytes. Cannot "
            "compare raw CE with byte-level v1.14. These are language-"
            "model diagnostics, not verified factual answers."
        ),
    }
