"""v1.14: honest token-frequency baseline and decoding-collapse diagnostics.

All evidence comes from an existing SHA-verified web model + frozen shards.
No web requests or gradient updates. Baseline probabilities are computed ONLY
from the training partition, then scored on identical evaluation windows
as the neural loss metric. This is an actual lower-complexity control, not
a random-model baseline. Eval is still SAME-ARTICLE if so labeled in manifest.
"""
from collections import Counter
import math
import random

from .brain import require_torch, sample_text
from .web_lab import _load_web_model, _read_frozen, _eval


def unigram_baseline(train_data, evaluation_data, block=64, seed=1901,
                     eval_batches=3, batch_size=4):
    """Smoothed training-only unigram CE in nats on original LM eval windows."""
    if len(train_data) < 1 or len(evaluation_data) <= block + 1:
        raise ValueError("Insufficient input for unigram evaluation")
    frequencies = Counter(train_data)
    total = len(train_data) + 256  # Laplace smoothing across 256 byte types
    costs = [
        -math.log((frequencies.get(byte, 0) + 1) / total)
        for byte in range(256)
    ]
    # Mirror random_batch() EXACTLY: its randint includes len(data)-block-1,
    # target positions are k+1 .. k+block. One evaluation entry per byte.
    rng = random.Random(seed)
    max_start = len(evaluation_data) - block - 1
    loss = 0.0
    num_bytes = 0
    for _ in range(eval_batches):
        for _ in range(batch_size):
            start = rng.randint(0, max_start)
            for byte in evaluation_data[start + 1:start + block + 1]:
                loss += costs[byte]
                num_bytes += 1
    return loss / num_bytes


def _generation_stats(generated_text, prompt):
    # sample_text returns prompt + continuation. Only measure continuation.
    suffix = generated_text[len(prompt):]
    data = suffix.encode("utf-8", errors="replace")
    if not data:
        return {
            "generated_bytes": 0, "unique_bytes": 0,
            "whitespace_fraction": 0.0, "most_common_byte_fraction": 0.0,
            "longest_identical_byte_run": 0,
        }
    counts = Counter(data)
    longest = 0
    previous = None
    run = 0
    for byte in data:
        run = run + 1 if byte == previous else 1
        previous = byte
        longest = max(longest, run)
    return {
        "generated_bytes": len(data),
        "unique_bytes": len(counts),
        "whitespace_fraction": round(
            sum(int(chr(byte).isspace()) for byte in data) / len(data), 4),
        "most_common_byte_fraction": round(max(counts.values()) / len(data), 4),
        "longest_identical_byte_run": longest,
    }


def diagnose_web_lab(store, candidate=True, prompt="Energy is",
                     max_new_bytes=80, seed=42, temperature=0.9):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be blank")
    if not 1 <= int(max_new_bytes) <= 200:
        raise ValueError("Max new bytes must be 1..200")
    if not 0.05 <= float(temperature) <= 2.0:
        raise ValueError("Temperature must be 0.05..2.0")
    loaded = _load_web_model(store, candidate)
    if loaded is None:
        return {"status": "skipped", "reason": "No web checkpoint to diagnose"}
    model, checkpoint_meta, record = loaded
    manifest, streams = _read_frozen(store, record["dataset_sha256"])
    cfg = checkpoint_meta["model_config"]
    block = int(cfg["block_size"])
    train_data = streams["train"]

    neural = {
        split: _eval(model, streams[split], cfg, "cpu",
                     seed=398 if split == "test" else 1901)
        for split in ("train", "val", "test")
    }
    unigram = {
        split: unigram_baseline(
            train_data, streams[split], block=block,
            seed=398 if split == "test" else 1901)
        for split in ("train", "val", "test")
    }
    # Greedy vs multinomial changes OUTPUT sampling only: weights fixed.
    greedy = sample_text(model, prompt, block, max_new_bytes=int(max_new_bytes),
                         device="cpu", seed=int(seed), greedy=True)
    sampled = sample_text(model, prompt, block, max_new_bytes=int(max_new_bytes),
                          device="cpu", seed=int(seed), temperature=float(temperature),
                          greedy=False)
    comparisons = {}
    for split in ("train", "val", "test"):
        n, u = neural[split], unigram[split]
        comparisons[split] = {
            "neural_byte_ce_nats": round(n, 5),
            "train_unigram_byte_ce_nats": round(u, 5),
            "neural_minus_unigram_nats": round(n - u, 5),
            "neural_better_than_unigram": n < u,
            "neural_perplexity": round(math.exp(n), 2),
            "unigram_perplexity": round(math.exp(u), 2),
        }
    return {
        "status": "diagnosed",
        "checkpoint_sha256": record["sha256"],
        "dataset_sha256": record["dataset_sha256"],
        "split_method": manifest["split_method"],
        "diagnostic_source_leakage": manifest["diagnostic_source_leakage"],
        "train_only_fitted_baseline": "Laplace-smoothed 256-byte unigram",
        "evaluation": comparisons,
        "generation": {
            "prompt": prompt,
            "greedy": {
                "text": greedy,
                "collapse_stats": _generation_stats(greedy, prompt),
            },
            "sampled": {
                "text": sampled, "seed": int(seed),
                "temperature": float(temperature),
                "collapse_stats": _generation_stats(sampled, prompt),
            },
            "model_weights_changed": False,
        },
        "interpretation": (
            "If neural CE is HIGHER than a training-only unigram baseline, "
            "it has not yet beaten a simple byte-frequency predictor. "
            "Sampling can make output look more varied without improving "
            "learned weights, understanding, or factual accuracy."
        ),
        "warning": (
            "Evaluation partitions share article content; results are "
            "not independent generalization."
            if manifest["diagnostic_source_leakage"]
            else "Source-host heldout partitions; still not a factual QA benchmark."
        ),
    }
