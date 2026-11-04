"""Structured neural addition experiment with leakage-free pair-group splits.

This explicitly assumes compositional addition: a trainable scalar representation
for each digit, and a sum of two learned representations. It is NOT a general
language model, discovered arithmetic algorithm, or website-derived knowledge.
The model must learn each digit value from examples. The true answer is used
only to build training labels and independently check held-out predictions.
"""
from datetime import datetime, timezone
from pathlib import Path
import random
import time

from .brain import get_device, require_torch, sha256_file
from .practice import checked_answer

FORMAT = "structured_single_digit_addition_v1"


def split_addition_pairs(seed=42):
    """All 55 unordered pairs; neither reversed ordering can leak across splits."""
    pairs = [(a, b) for a in range(10) for b in range(a, 10)]
    random.Random(int(seed)).shuffle(pairs)
    train, validation, test = pairs[:35], pairs[35:45], pairs[45:]
    return {"train": train, "validation": validation, "test": test}


def _build_model():
    torch, nn = require_torch()

    class CompositionalDigitAdder(nn.Module):
        def __init__(self):
            super().__init__()
            # These ten numbers are PARAMETERS, not initialized to digit values.
            self.digit_representation = nn.Embedding(10, 1)

        def forward(self, a, b):
            return (self.digit_representation(a) +
                    self.digit_representation(b)).squeeze(-1)

    return CompositionalDigitAdder()


def _tensors(pairs, torch, device):
    a = torch.tensor([left for left, _ in pairs], device=device, dtype=torch.long)
    b = torch.tensor([right for _, right in pairs], device=device, dtype=torch.long)
    expected = torch.tensor([
        checked_answer(left, right, "+") for left, right in pairs
    ], device=device, dtype=torch.float32)
    return a, b, expected


def _metrics(model, tensors, torch):
    a, b, expected = tensors
    model.eval()
    with torch.no_grad():
        raw = model(a, b)
        mse = float(torch.mean((raw - expected) ** 2).item())
        predicted = torch.round(raw).to(torch.long)
        correct = int((predicted == expected.to(torch.long)).sum().item())
    return {"total": len(expected), "correct": correct,
            "accuracy": round(correct / len(expected), 4),
            "mse": round(mse, 6)}


def run_numeric_lab(store, settings, steps=400, seed=42):
    """Train on 35 groups, select by 10 val groups, test 10 groups exactly once."""
    steps = int(steps)
    if not 1 <= steps <= 3000:
        raise ValueError("Steps must be 1..3000")
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    seed = int(seed)
    torch.manual_seed(seed)
    partitions = split_addition_pairs(seed)
    tensors = {
        split: _tensors(pairs, torch, device)
        for split, pairs in partitions.items()
    }
    model = _build_model().to(device)
    before = {
        name: _metrics(model, tensors[name], torch)
        for name in ("train", "validation", "test")
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)
    train_a, train_b, train_y = tensors["train"]
    best_state = None
    best_step = 0
    best_val = float("inf")
    no_improve = 0
    actual_steps = 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        pred = model(train_a, train_b)
        loss = torch.nn.functional.mse_loss(pred, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        actual_steps = step
        if step % 25 == 0 or step == steps:
            score = _metrics(model, tensors["validation"], torch)["mse"]
            if score + 1e-5 < best_val:
                best_val = score
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 8:
                    break
        if time.monotonic() - start > float(settings["train"].get("max_seconds", 600)):
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        best_step = actual_steps
    model.load_state_dict(best_state)
    # Test is consulted only after validation has selected the candidate.
    after = {
        name: _metrics(model, tensors[name], torch)
        for name in ("train", "validation", "test")
    }
    accepted = after["validation"]["mse"] + 0.0001 < before["validation"]["mse"]
    output = store.root / "models" / "numeric_lab"
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = output / f"digit-add-{timestamp}.pt"
    torch.save({
        "format": FORMAT,
        "weights": best_state,
        "seed": seed,
        "best_step": best_step,
        "splits": partitions,
        "validation_mse": after["validation"]["mse"],
        "test_accuracy": after["test"]["accuracy"],
    }, filename)
    digest = sha256_file(filename)
    if accepted:
        store.put("numeric_lab_latest", {
            "path": str(filename), "sha256": digest,
            "format": FORMAT, "seed": seed,
            "test_accuracy": after["test"]["accuracy"],
        })
    with torch.no_grad():
        learned_values = [round(float(v), 3) for v in
                          model.digit_representation.weight.cpu().flatten()]
        sample = []
        a, b, true = tensors["test"]
        guesses = model(a, b).cpu().tolist()
        for (left, right), expected, guess in zip(
                partitions["test"][:5], true.cpu().tolist()[:5], guesses[:5]):
            sample.append({"question": f"{left} + {right}",
                           "correct": int(expected),
                           "model_prediction": int(round(guess)),
                           "raw_prediction": round(float(guess), 3)})
    report = {
        "status": "trained", "experiment": FORMAT,
        "architecture": "learned_scalar_digit_embeddings_plus_composition",
        "explicit_prior": "The embeddings are combined by addition; this is not general arithmetic reasoning",
        "source": "Python-generated, independently checked arithmetic examples",
        "split_policy": "disjoint unordered operand groups; reverse pairs never cross splits",
        "train_groups": len(partitions["train"]),
        "validation_groups": len(partitions["validation"]),
        "test_groups": len(partitions["test"]),
        "before": before, "after": after,
        "learned_digit_values": learned_values,
        "best_step": best_step, "actual_steps": actual_steps,
        "device": device, "accepted": accepted,
        "checkpoint": str(filename), "sha256": digest,
        "unseen_examples": sample,
        "note": "This tests a narrow structural prior and neural parameter learning; it is not EVOLVE's byte-level language-model capability.",
    }
    store.log("numeric_lab", "completed", {
        key: value for key, value in report.items() if key != "unseen_examples"
    })
    return report


def predict_numeric_lab(store, a, b):
    """Use only learned parameters to predict; never call the arithmetic verifier."""
    a, b = int(a), int(b)
    if not (0 <= a <= 9 and 0 <= b <= 9):
        raise ValueError("This specialist supports single-digit operands 0..9 only")
    prior = store.get("numeric_lab_latest")
    if prior is None:
        return {"status": "skipped", "reason": "No accepted structured addition checkpoint"}
    torch, _ = require_torch()
    data = torch.load(prior["path"], weights_only=True, map_location="cpu")
    if data.get("format") != FORMAT:
        raise ValueError("Incompatible numeric lab checkpoint")
    model = _build_model()
    model.load_state_dict(data["weights"])
    model.eval()
    with torch.no_grad():
        result = float(model(
            torch.tensor([a]), torch.tensor([b])).item())
    return {
        "status": "predicted", "question": f"{a} + {b}",
        "model_answer": int(round(result)),
        "raw_prediction": round(result, 4),
        "checkpoint": prior["sha256"],
        "architecture": FORMAT,
        "note": "This output is from the neural digit embeddings; no arithmetic-verifier value is substituted.",
    }
