"""BERHOT v1.13: one-command corpus -> frozen dataset -> real neural learning.

Uses existing BERHOT byte Transformer / PyTorch, NOT a native BERHOT tensor
backend. Readable public HTML was already collected; no network calls here.
Unknown-rights text requires explicit local research opt-in. This makes no
claim of ownership, lawful redistribution, independently verified facts or
general reasoning.

STRICT evaluation holds out source HOSTS. With fewer than three distinct
article hosts, --diagnostic-within-document permits a SMALL within-article
smoke test; such scores are explicitly not independent generalization.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import json
import math
import random
import re
import time

from .brain import (
    build_model, get_device, load_checkpoint, require_torch, sample_text,
    sha256_file,
)
from .training import evaluate_loss, random_batch

FORMAT = "berhot_web_byte_lm_v1"
DATASET_VERSION = "web_corpus_curated_sharded_v1"
EXCLUDE_TITLES = re.compile(
    r"(?i)^(main page|contents|current events|portal|index of|home page)$"
)
EXCLUDE_URL = re.compile(
    r"(?i)(?:^|/)(?:main_page|portal:|special:|wikipedia:|"
    r"help:|category:|template:|file:|talk:|current_events|contents)(?:$|/)"
)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _canonical_bytes(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _within(root, candidate):
    root = Path(root).resolve()
    file = Path(candidate).resolve()
    if not file.is_relative_to(root):
        raise ValueError("Archive/checkpoint path escapes its trusted storage root")
    return file


def _language(text):
    """Script hint, not statistical language detection."""
    ascii_letters = sum("a" <= c.casefold() <= "z" for c in text)
    arabic_letters = sum("\u0600" <= c <= "\u06ff" for c in text)
    if arabic_letters >= max(80, ascii_letters * 2):
        return "arabic_script"
    if ascii_letters >= max(100, arabic_letters * 3):
        return "latin_script"
    return "undetermined"


def _article(url, title, words):
    parsed = urlsplit(url)
    path = unquote(parsed.path)
    # Navigation pages are not educational training articles.
    if EXCLUDE_TITLES.search(title.strip()) or EXCLUDE_URL.search(path):
        return False, "navigation_or_dynamic_index"
    if "wikipedia.org" in (parsed.hostname or ""):
        if not path.startswith("/wiki/") or ":" in path:
            return False, "non_article_wiki_namespace"
    elif path in ("", "/"):
        return False, "site_homepage"
    if words < 300:
        return False, "too_few_words"
    return True, "article_candidate"


def _chunks(text, target=900):
    """Build non-overlapping sentence groups with explicit length bounds."""
    if not 256 <= target <= 2048:
        raise ValueError("Chunk target must be 256..2048 characters")
    sentences = re.split(r"(?<=[.!?؟。])\s+", text)
    blocks, current = [], ""
    for sentence in sentences:
        # Long sentences are chunked on words, never by overlapping windows.
        words = sentence.split()
        pieces = []
        if len(sentence) > 2 * target:
            current_piece = []
            for w in words:
                if sum(len(x) + 1 for x in current_piece) + len(w) > target and current_piece:
                    pieces.append(" ".join(current_piece))
                    current_piece = []
                current_piece.append(w)
            if current_piece:
                pieces.append(" ".join(current_piece))
        else:
            pieces = [sentence]
        for piece in pieces:
            if len(current) + len(piece) + 1 > target and current:
                if len(current) >= 192:
                    blocks.append(current.strip())
                current = ""
            current = (current + " " + piece).strip()
    if len(current) >= 192:
        blocks.append(current.strip())
    # Exact duplicate chunks should never be sent to different splits.
    seen, unique = set(), []
    for block in blocks:
        content = block.encode("utf-8")
        h = _sha(content)
        if h not in seen and len(content) >= 256:
            unique.append({"text": block, "sha256": h})
            seen.add(h)
    return unique


def _shingles(text, n=5):
    # Content-token n-grams are an inexpensive near-copy signal; not semantic.
    tokens = re.findall(r"\w+", text.casefold(), re.UNICODE)
    return set(tuple(tokens[i:i+n])
               for i in range(max(0, len(tokens) - n + 1)))


def _remove_near_copies(articles, threshold=0.86):
    """Drop duplicated or highly overlapping chunks BEFORE any split assignment."""
    all_seen = []
    dropped = 0
    for article in articles:
        clean = []
        for chunk in article["chunks"]:
            shingles = _shingles(chunk["text"])
            near = False
            if shingles:
                for existing in all_seen:
                    overlap = len(shingles & existing)
                    if overlap / len(shingles | existing) >= threshold:
                        near = True
                        break
            if near:
                dropped += 1
            else:
                all_seen.append(shingles)
                clean.append(chunk)
        article["chunks"] = clean
    return dropped


def _source_inventory(store, include_unknown=False):
    sql = (
        "SELECT s.url,s.host,s.title,s.rights_status,s.noai,"
        "s.sha256,t.clean_path,t.words,t.characters,t.quality_score "
        "FROM web_corpus_sources s JOIN web_corpus_texts t "
        "ON t.sha256=s.sha256 ORDER BY s.url"
    )
    candidates, excluded = [], []
    seen = set()
    for row in store.db.execute(sql):
        url, rights = str(row["url"]), str(row["rights_status"])
        reason = None
        if int(row["noai"]):
            reason = "publisher_noai_signal"
        elif rights == "UNKNOWN" and not include_unknown:
            reason = "unknown_rights_requires_explicit_research_opt_in"
        elif rights not in ("CC0-1.0", "PUBLIC-DOMAIN", "UNKNOWN"):
            reason = "unrecognized_rights_metadata"
        else:
            allowed, why = _article(url, str(row["title"]), int(row["words"]))
            if not allowed:
                reason = why
        if row["sha256"] in seen:
            reason = "duplicate_cleaned_article"
        if reason:
            excluded.append({"url": url, "reason": reason})
            continue
        seen.add(str(row["sha256"]))
        path = _within(store.root / "web_corpus" / "clean", row["clean_path"])
        if not path.is_file():
            excluded.append({"url": url, "reason": "missing_cleaned_archive_file"})
            continue
        raw = path.read_bytes()
        if _sha(raw) != row["sha256"]:
            raise ValueError(f"Corpus text hash mismatch at {url}")
        text = raw.decode("utf-8")
        chunks = _chunks(text)
        if len(chunks) < 3:
            excluded.append({"url": url, "reason": "too_few_content_chunks"})
            continue
        candidates.append({
            "url": url, "host": str(row["host"]).casefold(),
            "title": str(row["title"]), "sha256": str(row["sha256"]),
            "rights": rights, "script_hint": _language(text),
            "words": int(row["words"]), "quality_heuristic": float(row["quality_score"]),
            "chunks": chunks,
        })
    # Near-copy filtering occurs across article sources before partitioning.
    # A duplicate phrase cannot be learned in train and scored in test.
    dropped = _remove_near_copies(candidates)
    if dropped:
        excluded.append({
            "url": "(cross-source chunks)",
            "reason": "near_duplicate_chunks_dropped",
            "count": dropped,
        })
    still_valid = []
    for article in candidates:
        if len(article["chunks"]) < 3:
            excluded.append({
                "url": article["url"], "reason": "insufficient_unique_content_chunks"
            })
        else:
            still_valid.append(article)
    return still_valid, excluded


def web_inventory(store, include_unknown=False):
    selected, exclusions = _source_inventory(store, include_unknown)
    by_reason = dict(Counter(item["reason"] for item in exclusions))
    return {
        "status": "eligible_for_experiment" if selected else "no_article_candidates",
        "candidate_article_count": len(selected),
        "candidate_hosts": len({x["host"] for x in selected}),
        "candidate_chunks": sum(len(x["chunks"]) for x in selected),
        "articles": [{
            "url": x["url"], "words": x["words"],
            "script_hint": x["script_hint"], "rights": x["rights"],
            "quality_heuristic_not_factual_accuracy": x["quality_heuristic"],
            "chunks": len(x["chunks"]),
        } for x in selected],
        "excluded_by_reason": by_reason, "excluded": exclusions[:20],
        "unknown_rights_opted_in": bool(include_unknown),
        "warning": (
            "These are content candidates only. No rights, truth or learning "
            "is proved by discovery. Strict heldout evaluation needs at least "
            "three independent source hosts."
        ),
    }


def _choose_splits(articles, diagnostic=False):
    hosts = sorted({x["host"] for x in articles})
    if len(hosts) >= 3:
        # Deterministic order; one complete host per heldout partition.
        ranked = sorted(hosts, key=lambda host: _sha(host.encode()))
        train_n = max(1, len(ranked) - 2)
        selected = {
            host: ("train" if i < train_n else
                   "val" if i == train_n else "test")
            for i, host in enumerate(ranked)
        }
        partitions = {s: [] for s in ("train", "val", "test")}
        for article in articles:
            partitions[selected[article["host"]]].extend([
                (article, item) for item in article["chunks"]])
        return partitions, "heldout_source_hosts", False
    if not diagnostic:
        return None, "strict_source_holdout_unavailable", False
    # Diagnostic only: pieces of the SAME article are not independent.
    # Holdout assignment never repeats identical chunk bytes across splits.
    parts = {s: [] for s in ("train", "val", "test")}
    unique_hashes = set()
    all_chunks = []
    for article in articles:
        for item in article["chunks"]:
            if item["sha256"] not in unique_hashes:
                unique_hashes.add(item["sha256"])
                all_chunks.append((article, item))
    ranked = sorted(all_chunks, key=lambda item: item[1]["sha256"])
    if len(ranked) < 12:
        return None, "diagnostic_requires_at_least_12_unique_chunks", True
    n = len(ranked)
    for i, row in enumerate(ranked):
        split = "train" if i < n - 2 * max(2, n // 10) else (
            "val" if i < n - max(2, n // 10) else "test")
        parts[split].append(row)
    return parts, "within_document_diagnostic_not_independent", True


def freeze_dataset(store, include_unknown=False, diagnostic=False):
    """Freeze deterministic byte shards; SHA-validate all existing files."""
    articles, exclusions = _source_inventory(store, include_unknown)
    partitions, mode, leakage = _choose_splits(articles, diagnostic=diagnostic)
    if partitions is None:
        return {
            "status": "skipped", "reason": mode,
            "candidate_articles": len(articles),
            "candidate_hosts": len({x["host"] for x in articles}),
            "excluded_by_reason": dict(Counter(x["reason"] for x in exclusions)),
            "hint": (
                "For the existing single Energy article only, use "
                "--diagnostic-within-document. Those results are exploratory "
                "and NOT an independent generalization benchmark."
            ),
        }
    streams = {}
    parts = {}
    for name in ("train", "val", "test"):
        rows = partitions[name]
        # Never concatenate content from different partitions.
        data = b"\n\n".join(x["text"].encode("utf-8") for _, x in rows)
        if len(data) < 514:
            return {"status": "skipped", "reason": f"{name}_data_too_short"}
        streams[name] = data
        parts[name] = {
            "bytes": len(data), "chunks": len(rows),
            "sha256": _sha(data),
            "source_urls": sorted({x["url"] for x, _ in rows}),
            "source_hosts": sorted({x["host"] for x, _ in rows}),
            "chunk_sha256": [x["sha256"] for _, x in rows],
        }
    # Invariant: each chunk's original content hash belongs to one partition.
    pairs = [set(parts[name]["chunk_sha256"]) for name in ("train", "val", "test")]
    if any(pairs[i] & pairs[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Exact document chunk crossed a dataset split")
    meta = {
        "format": DATASET_VERSION, "split_method": mode,
        "diagnostic_source_leakage": leakage,
        "unknown_rights_opted_in": bool(include_unknown),
        "eligible_articles": [{
            "url": a["url"], "source_sha256": a["sha256"],
            "host": a["host"], "rights": a["rights"],
            "script_hint": a["script_hint"], "chunks": len(a["chunks"]),
        } for a in articles],
        "excluded_by_reason": dict(Counter(x["reason"] for x in exclusions)),
        "partitions": parts,
        "note": (
            "Content-addressed immutable dataset. UNKNOWN rights opt-in "
            "is not a legal permission determination. No withheld test bytes "
            "are used to train or select checkpoint weights."
        ),
    }
    digest = _sha(_canonical_bytes(meta))
    base = store.root / "datasets" / "web_lab" / digest
    base.mkdir(parents=True, exist_ok=True)
    for name, data in streams.items():
        path = base / (name + ".bin")
        if path.exists():
            if _sha(path.read_bytes()) != parts[name]["sha256"]:
                raise ValueError("Previously frozen dataset shard has changed")
        else:
            path.write_bytes(data)
    manifest_path = base / "manifest.json"
    canonical = _canonical_bytes(meta)
    if manifest_path.exists():
        if manifest_path.read_bytes() != canonical:
            raise ValueError("Existing immutable dataset manifest changed")
    else:
        manifest_path.write_bytes(canonical)
    return {
        "status": "frozen", "dataset_sha256": digest,
        "dataset_dir": str(base), "manifest": str(manifest_path),
        "split_method": mode, "diagnostic_source_leakage": leakage,
        "partitions": {k: {key: val for key, val in v.items()
                             if key != "chunk_sha256"} for k, v in parts.items()},
        "candidate_articles": len(articles),
        "unknown_rights_opted_in": bool(include_unknown),
        "excluded_by_reason": meta["excluded_by_reason"],
    }


def _read_frozen(store, dataset_sha):
    if not re.fullmatch(r"[a-f0-9]{64}", str(dataset_sha)):
        raise ValueError("Dataset ID must be a SHA256 digest")
    base = (store.root / "datasets" / "web_lab" / dataset_sha).resolve()
    root = (store.root / "datasets" / "web_lab").resolve()
    if not base.is_relative_to(root):
        raise ValueError("Dataset path escapes root")
    meta = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if _sha(_canonical_bytes(meta)) != dataset_sha or meta.get("format") != DATASET_VERSION:
        raise ValueError("Frozen manifest SHA256/version mismatch")
    streams = {}
    for split in ("train", "val", "test"):
        data = (base / (split + ".bin")).read_bytes()
        if _sha(data) != meta["partitions"][split]["sha256"]:
            raise ValueError("Frozen shard SHA256 mismatch")
        streams[split] = data
    return meta, streams


def _lab_cfg(settings):
    # Tiny separate lab: does not change main checkpoint registry.
    model = dict(settings["model"])
    model.update(block_size=64, embedding_dim=32, heads=4, layers=1, dropout=0.0)
    return model


def _eval(model, data, cfg, device, seed=1901):
    return evaluate_loss(model, data, {
        "block_size": cfg["block_size"], "eval_batches": 3,
    }, device, seed=seed)


def _arithmetic_bytes(store):
    rows = store.db.execute(
        "SELECT text FROM documents WHERE topic='verified:arithmetic' "
        "AND split='val' ORDER BY id LIMIT 24").fetchall()
    return b"\n".join(row["text"].encode("utf-8") for row in rows)


def train_web_lab(store, settings, steps=60, seed=31415,
                  include_unknown=False, diagnostic=False, device="cpu"):
    """One-shot real experiment; no network activity or base-model mutation."""
    steps = int(steps)
    if not 1 <= steps <= 1000:
        raise ValueError("Steps must be between 1 and 1000")
    frozen = freeze_dataset(store, include_unknown, diagnostic)
    if frozen["status"] != "frozen":
        return frozen
    torch, _ = require_torch()
    chosen = get_device(device)
    if chosen not in ("cpu", "mps", "cuda"):
        raise ValueError("Unsupported device")
    torch.manual_seed(int(seed))
    rng = random.Random(int(seed))
    # Keep default CPU test latency predictable.
    if chosen == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    cfg = _lab_cfg(settings)
    meta, streams = _read_frozen(store, frozen["dataset_sha256"])
    if any(len(streams[k]) < cfg["block_size"] + 2 for k in streams):
        return {"status": "skipped", "reason": "not_enough_bytes_for_block"}
    model = build_model(cfg).to(chosen)
    train_before = _eval(model, streams["train"], cfg, chosen)
    val_before = _eval(model, streams["val"], cfg, chosen)
    old_math = _arithmetic_bytes(store)
    math_before = (_eval(model, old_math, cfg, chosen, seed=295)
                   if len(old_math) >= cfg["block_size"] + 2 else None)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(settings["train"].get("learning_rate", 0.0007)))
    start = time.monotonic()
    actual = 0
    model.train()
    for i in range(steps):
        x, y = random_batch(streams["train"], cfg["block_size"], 8,
                            torch, chosen, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual += 1
        if time.monotonic() - start >= float(settings["train"].get("max_seconds", 600)):
            break
    train_after = _eval(model, streams["train"], cfg, chosen)
    val_after = _eval(model, streams["val"], cfg, chosen)
    math_after = (_eval(model, old_math, cfg, chosen, seed=295)
                  if math_before is not None else None)
    # Select ONLY with validation. The held-out test is evaluated after the
    # candidate checkpoint and selection decision have been determined.
    accepted = (math.isfinite(val_after) and val_after < val_before)
    output = store.root / "models" / "web_lab"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint = output / f"candidate-{stamp}.pt"
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "format_version": 1, "objective": FORMAT,
        "weights": weights, "model_config": cfg,
        "dataset_sha256": frozen["dataset_sha256"], "steps": actual,
        "seed": int(seed), "validation_before": val_before,
        "validation_after": val_after,
        "source_leakage_warning": meta["diagnostic_source_leakage"],
    }, checkpoint)
    sha = sha256_file(checkpoint)
    # NOW evaluate truly withheld test bytes, without further weight updates.
    test_after = _eval(model, streams["test"], cfg, chosen, seed=398)
    report = {
        "status": "trained", "objective": FORMAT, "actual_steps": actual,
        "model_backend": "PyTorch byte-level causal Transformer",
        "checkpoint": str(checkpoint), "sha256": sha,
        "accepted_by_validation": accepted,
        "dataset_sha256": frozen["dataset_sha256"],
        "frozen_manifest": frozen["manifest"],
        "split_method": frozen["split_method"],
        "diagnostic_source_leakage": frozen["diagnostic_source_leakage"],
        "source_rights": [
            {"url": a["url"], "rights": a["rights"]}
            for a in meta["eligible_articles"]],
        "train_byte_loss_before": round(train_before, 5),
        "train_byte_loss_after": round(train_after, 5),
        "validation_byte_loss_before": round(val_before, 5),
        "validation_byte_loss_after": round(val_after, 5),
        "heldout_test_byte_loss_after": round(test_after, 5),
        "test_used_for_checkpoint_selection": False,
        "arithmetic_retention_proxy_byte_loss_before": (
            round(math_before, 5) if math_before is not None else None),
        "arithmetic_retention_proxy_byte_loss_after": (
            round(math_after, 5) if math_after is not None else None),
        "arithmetic_accuracy_tested": False,
        "trained_on_verified_facts": False,
        "warning": (
            "WITHIN-ARTICLE DIAGNOSTIC: the same article appears in multiple "
            "splits. These numbers do NOT establish independent generalization."
            if meta["diagnostic_source_leakage"] else
            "Heldout source-host test is separate from training and validation. "
            "Lower loss alone does not prove factual understanding."
        ),
    }
    store.put("web_lab_latest_candidate", report)
    if accepted:
        store.put("web_lab_latest", report)
    store.log("web_lab_train", "completed", report)
    return report


def _load_web_model(store, candidate=True):
    record = store.get("web_lab_latest_candidate" if candidate else "web_lab_latest")
    if record is None:
        return None
    path = _within(store.root / "models" / "web_lab", record["checkpoint"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError("Web neural checkpoint SHA256 mismatch")
    model, meta = load_checkpoint(path, map_location="cpu")
    if meta.get("objective") != FORMAT or meta.get("dataset_sha256") != record["dataset_sha256"]:
        raise ValueError("Web neural checkpoint objective/dataset mismatch")
    return model, meta, record


def evaluate_web_lab(store, candidate=True):
    loaded = _load_web_model(store, candidate)
    if loaded is None:
        return {"status": "skipped", "reason": "No web neural checkpoint yet"}
    model, meta, record = loaded
    frozen, streams = _read_frozen(store, record["dataset_sha256"])
    cfg = meta["model_config"]
    # Recalculation can be run after training, never trains or selects weights.
    result = {
        "status": "evaluated", "checkpoint_sha256": record["sha256"],
        "dataset_sha256": record["dataset_sha256"],
        "split_method": frozen["split_method"],
        "diagnostic_source_leakage": frozen["diagnostic_source_leakage"],
        "loss": {
            split: round(_eval(model, streams[split], cfg, "cpu",
                               seed=398 if split == "test" else 1901), 5)
            for split in ("train", "val", "test")
        },
        "note": "Byte prediction only; not a QA or human knowledge benchmark.",
    }
    return result


def predict_web_lab(store, prompt, candidate=True, max_new_bytes=120):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if not 1 <= int(max_new_bytes) <= 300:
        raise ValueError("Max new bytes must be 1..300")
    loaded = _load_web_model(store, candidate)
    if loaded is None:
        return {"status": "skipped", "reason": "No web neural checkpoint yet"}
    model, meta, record = loaded
    block = meta["model_config"]["block_size"]
    result = sample_text(model, prompt, block,
                         max_new_bytes=int(max_new_bytes), device="cpu",
                         greedy=True)
    return {
        "status": "predicted", "prompt": prompt, "neural_generation": result,
        "checkpoint_sha256": record["sha256"],
        "dataset_sha256": record["dataset_sha256"],
        "method": "checkpoint_weights_only_no_source_lookup",
        "warning": "Text continuation is not verified factual question answering.",
    }
