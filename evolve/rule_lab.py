"""Supervised neural arithmetic without a built-in summation operation.

A generic neural classifier learns from pairs of digit tokens. Its forward
pass has NO addition operator and no exact calculator; 'checked_answer' is
used ONLY for supervised labels and post-training evaluation.

Scope is intentionally narrow: single-digit addition. The input range
(0..9), output classes (0..18) and operand-order symmetry are supplied priors.
This is neither the general language model nor self-discovered mathematics.
"""
from datetime import datetime, timezone
import math
import random
import time

from .brain import require_torch, get_device, sha256_file
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

FORMAT = "neural_pair_classifier_addition_v1"


def _build_model(hidden=64):
    torch, nn = require_torch()
    hidden = int(hidden)
    if hidden < 8 or hidden > 256:
        raise ValueError("Hidden size must be between 8 and 256")

    class GenericPairClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.digit = nn.Embedding(10, 16)
            self.classifier = nn.Sequential(
                nn.Linear(32, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 19),
            )

        def forward(self, a, b):
            # An unordered pair is represented canonically. This is a
            # commutativity prior, not an arithmetic computation.
            left = torch.minimum(a, b)
            right = torch.maximum(a, b)
            features = torch.cat([self.digit(left), self.digit(right)], dim=-1)
            return self.classifier(features)

    return GenericPairClassifier()


def _data(pairs, torch, device):
    left = torch.tensor([a for a, b in pairs], dtype=torch.long, device=device)
    right = torch.tensor([b for a, b in pairs], dtype=torch.long, device=device)
    # The verifier is used only in labels, never to produce predictions.
    target = torch.tensor([
        checked_answer(a, b, "+") for a, b in pairs
    ], dtype=torch.long, device=device)
    return left, right, target


def _metrics(model, batch, torch):
    model.eval()
    with torch.no_grad():
        a, b, target = batch
        logits = model(a, b)
        loss = float(torch.nn.functional.cross_entropy(logits, target).item())
        predicted = logits.argmax(-1)
        correct = int((predicted == target).sum().item())
    return {
        "total": int(target.shape[0]),
        "correct": correct,
        "accuracy": round(correct / target.shape[0], 4),
        "cross_entropy": round(loss, 6),
    }


def _existing_additive_baseline(store, partitions, seed):
    """Read a matching v0.9 model; never rewrite its checkpoint or registry."""
    record = store.get("numeric_lab_latest")
    if not record or record.get("seed") != seed:
        return {"status": "not_available_for_this_seed",
                "how_to_compare": f"Run numeric-lab --steps 400 --seed {seed} before rule-lab"}
    try:
        torch, _ = require_torch()
        from .numeric_lab import _build_model as additive_model, _tensors, _metrics as add_metrics
        data = torch.load(record["path"], map_location="cpu", weights_only=True)
        if data.get("splits") != partitions or data.get("format") != "structured_single_digit_addition_v1":
            return {"status": "mismatched_split_or_format"}
        model = additive_model()
        model.load_state_dict(data["weights"])
        heldout = _tensors(partitions["test"], torch, "cpu")
        return {
            "status": "comparable",
            "architecture": "learned_scalar_digit_embeddings_plus_explicit_addition",
            "checkpoint_sha256": record["sha256"],
            "test": add_metrics(model, heldout, torch),
        }
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"status": "could_not_read_checkpoint", "error": str(exc)[:180]}


def run_rule_lab(store, settings, steps=800, seed=31415, hidden=64):
    """Fit a black-box pair model; select on validation, inspect test after."""
    steps = int(steps)
    if not 1 <= steps <= 3000:
        raise ValueError("Steps must be between 1 and 3000")
    seed = int(seed)
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(seed)
    random.seed(seed)
    partitions = split_addition_pairs(seed)
    batches = {name: _data(pairs, torch, device)
               for name, pairs in partitions.items()}
    model = _build_model(hidden).to(device)
    initial = {name: val.detach().cpu().clone()
               for name, val in model.state_dict().items()}
    before_train = _metrics(model, batches["train"], torch)
    before_val = _metrics(model, batches["validation"], torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01)
    a, b, labels = batches["train"]
    best_weights = None
    best_step = 0
    best_loss = float("inf")
    wait = 0
    actual_steps = 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        logits = model(a, b)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_steps = step
        if step % 20 == 0 or step == steps:
            val_loss = _metrics(model, batches["validation"], torch)["cross_entropy"]
            if math.isfinite(val_loss) and val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_step = step
                best_weights = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 12:
                    break
        if time.monotonic() - start > float(settings["train"].get("max_seconds", 600)):
            break
    if best_weights is None:
        return {"status": "rejected", "reason": "No finite validation checkpoint"}
    model.load_state_dict(best_weights)
    after_train = _metrics(model, batches["train"], torch)
    after_val = _metrics(model, batches["validation"], torch)
    accepted = (after_val["cross_entropy"] + 1e-4 <
                before_val["cross_entropy"])
    # The test split is first measured AFTER validation has chosen the weights.
    original = _build_model(hidden).to(device)
    original.load_state_dict(initial)
    before_test = _metrics(original, batches["test"], torch)
    after_test = _metrics(model, batches["test"], torch)
    model.eval()
    with torch.no_grad():
        a_test, b_test, targets = batches["test"]
        logits = model(a_test, b_test)
        predictions = logits.argmax(-1).cpu().tolist()
        confidence = logits.softmax(-1).max(-1).values.cpu().tolist()
    samples = [{
        "question": f"{a} + {b}",
        "expected": checked_answer(a, b, "+"),
        "model_answer": int(predictions[i]),
        "correct": int(predictions[i]) == checked_answer(a, b, "+"),
        "confidence": round(float(confidence[i]), 4),
    } for i, (a, b) in enumerate(partitions["test"])]
    output = store.root / "models" / "rule_lab"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output / f"pair-net-{stamp}.pt"
    torch.save({
        "format": FORMAT, "hidden": int(hidden),
        "weights": best_weights, "seed": seed,
        "partitions": partitions, "best_step": best_step,
        "validation_cross_entropy": after_val["cross_entropy"],
    }, path)
    digest = sha256_file(path)
    if accepted:
        store.put("rule_lab_latest", {
            "path": str(path), "sha256": digest,
            "format": FORMAT, "seed": seed,
            "hidden": int(hidden), "test_accuracy": after_test["accuracy"],
        })
    comparison = _existing_additive_baseline(store, partitions, seed)
    report = {
        "status": "trained",
        "experiment": FORMAT,
        "architecture": "learned_digit_embeddings + generic MLP classifier",
        "prior": "Only input digit domain, output classes 0..18, and unordered-pair symmetry; no arithmetic operation in forward pass",
        "task": "single-digit addition only",
        "seed": seed, "device": device,
        "split_policy": "35/10/10 unordered pairs; no reverse-order leakage",
        "train_groups": len(partitions["train"]),
        "validation_groups": len(partitions["validation"]),
        "test_groups": len(partitions["test"]),
        "before": {"train": before_train, "validation": before_val, "test": before_test},
        "after": {"train": after_train, "validation": after_val, "test": after_test},
        "best_step": best_step, "actual_steps": actual_steps,
        "accepted": accepted, "checkpoint": str(path),
        "sha256": digest, "test_examples": samples,
        "explicit_additive_baseline": comparison,
        "note": "Test accuracy measures a black-box learner on held-out pairs; no accuracy level is guaranteed. This is not self-directed web knowledge.",
    }
    store.log("rule_lab", "completed", {
        key: val for key, val in report.items() if key != "test_examples"
    })
    return report


def predict_rule_lab(store, a, b):
    """Infer using only weights of the generic neural classifier."""
    a, b = int(a), int(b)
    if not (0 <= a <= 9 and 0 <= b <= 9):
        raise ValueError("Rule lab supports operands 0..9 only")
    checkpoint = store.get("rule_lab_latest")
    if not checkpoint:
        return {"status": "skipped", "reason": "No accepted rule-lab checkpoint"}
    torch, _ = require_torch()
    data = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if data["format"] != FORMAT:
        raise ValueError("Unrecognized rule-lab format")
    model = _build_model(data["hidden"])
    model.load_state_dict(data["weights"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([a]), torch.tensor([b]))
        predicted = int(logits.argmax(-1).item())
        confidence = float(logits.softmax(-1).max().item())
    return {
        "status": "predicted",
        "question": f"{a} + {b}",
        "model_answer": predicted,
        "confidence": round(confidence, 4),
        "checkpoint": checkpoint["sha256"],
        "architecture": FORMAT,
        "note": "Prediction is from a learned classifier; checked_answer was not called.",
    }
