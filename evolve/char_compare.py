"""Read-only historical checkpoint comparison for the Unicode char experiment.

Reference SHA is REQUIRED. Both checkpoints must be local under char_lab,
hash-valid and compatible with the SAME frozen dataset and tokenizer.
No training or network requests. Test scores are descriptive diagnostics,
never used to choose weights.
"""
from pathlib import Path
import re

from .brain import load_checkpoint, sha256_file
from .char_lab import (
    FORMAT, _encode_all, _eval, _load_char_model, _safe_generation,
)
from .web_lab import _read_frozen, _within


def _find_reference(store, sha):
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("reference_sha256 must be exactly 64 lowercase hex characters")
    root = (store.root / "models" / "char_lab").resolve()
    if not root.is_dir():
        raise ValueError("No Unicode checkpoint directory")
    for candidate in root.glob("candidate-*.pt"):
        safe = _within(root, candidate)
        if safe.is_file() and sha256_file(safe) == sha:
            return safe
    raise ValueError("Reference checkpoint not found (or SHA256 differs)")


def _text_stats(text):
    return {
        "characters": len(text),
        "unique_characters": len(set(text)),
        "whitespace_fraction": round(
            sum(char.isspace() for char in text) / len(text), 4)
        if text else 0.0,
        "word_like_spans_not_validated": len(
            re.findall(r"[A-Za-z]{2,}", text)),
    }


def compare_char_checkpoints(store, reference_sha256, prompt="Energy is",
                             new_chars=100, seed=42, temperature=0.85,
                             top_k=30, candidate=True):
    current = _load_char_model(store, candidate=candidate)
    if current is None:
        return {"status": "skipped", "reason": "no_latest_char_checkpoint"}
    current_model, current_meta, current_report, tokenizer = current
    ref_path = _find_reference(store, reference_sha256)
    reference_model, reference_meta = load_checkpoint(ref_path, map_location="cpu")
    for field in ("objective", "dataset_sha256", "tokenizer_sha256", "model_config"):
        if reference_meta.get(field) != current_meta.get(field):
            raise ValueError(f"Reference and latest checkpoint mismatch: {field}")
    if current_meta.get("objective") != FORMAT:
        raise ValueError("Not a BERHOT Unicode checkpoint")
    _, frozen = _read_frozen(store, current_report["dataset_sha256"])
    cfg = current_meta["model_config"]
    ids, unknown = _encode_all(tokenizer, frozen, cfg["block_size"])
    evals = {}
    for split in ("train", "val", "test"):
        seed_for_split = 398 if split == "test" else 1901
        old = _eval(reference_model, ids[split], cfg, "cpu", seed=seed_for_split)
        new = _eval(current_model, ids[split], cfg, "cpu", seed=seed_for_split)
        evals[split] = {
            "reference_loss": round(old, 5),
            "latest_loss": round(new, 5),
            "latest_minus_reference_nats_per_character": round(new - old, 5),
            "improved": new < old,
        }
    old_generation = _safe_generation(
        reference_model, tokenizer, prompt, cfg["block_size"],
        new_chars=new_chars, temperature=temperature, seed=seed,
        top_k=top_k)
    new_generation = _safe_generation(
        current_model, tokenizer, prompt, cfg["block_size"],
        new_chars=new_chars, temperature=temperature, seed=seed,
        top_k=top_k)
    return {
        "status": "compared",
        "reference_checkpoint_sha256": reference_sha256,
        "reference_training_steps": reference_meta.get("total_training_steps"),
        "latest_checkpoint_sha256": current_report["sha256"],
        "latest_training_steps": current_meta.get("total_training_steps"),
        "latest_resumed_from_sha256": current_report.get("resumed_from_sha256"),
        "dataset_sha256": current_report["dataset_sha256"],
        "tokenizer_sha256": current_report["tokenizer_sha256"],
        "same_model_and_frozen_dataset": True,
        "diagnostic_source_leakage": current_report["diagnostic_source_leakage"],
        "evaluation": evals,
        "generation": {
            "prompt": prompt, "seed": seed, "temperature": temperature,
            "top_k": top_k,
            "reference_continuation": old_generation["continuation"],
            "latest_continuation": new_generation["continuation"],
            "reference_diagnostics": _text_stats(old_generation["continuation"]),
            "latest_diagnostics": _text_stats(new_generation["continuation"]),
        },
        "unknown_character_rates": unknown,
        "no_weights_modified": True,
        "test_used_for_checkpoint_selection": False,
        "warning": (
            "Same-source chunks do NOT give independent generalization. "
            "Word-like spans are NOT a dictionary/semantic accuracy metric. "
            "Comparing repeated heldout test results informally can bias "
            "later development decisions."
        ),
    }
