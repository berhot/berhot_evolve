"""Numerical-magnitude ablation, without a hard-coded arithmetic operation.

Supervised digits enter the network as ordered continuous values, rather than
arbitrary categorical embeddings. The forward pass never adds the operands:
a generic MLP predicts a scalar. This retains STRONG ordinal, scaling and
commutativity priors, and is not general reasoning or website knowledge.

35 train, 10 validation, 10 test unordered pairs. Test outcomes are
computed only after validation checkpoint selection. Multi-seed study reports
ALL requested results; no selecting a lucky seed by its test accuracy.
"""
from datetime import datetime, timezone
import math
import random
import statistics
import time

from .brain import get_device, require_torch, sha256_file
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

FORMAT = "magnitude_mlp_addition_v1"


def _build_model(hidden=32):
    torch, nn = require_torch()
    hidden = int(hidden)
    if not 8 <= hidden <= 256:
        raise ValueError("Hidden size must be 8..256")

    class MagnitudePairRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, a, b):
            # Canonical unordered numeric inputs in [0, 1].
            # NO addition of operand values or embeddings in this forward.
            lo = torch.minimum(a, b).float() / 9.0
            hi = torch.maximum(a, b).float() / 9.0
            features = torch.stack((lo, hi), dim=-1)
            return self.layers(features).squeeze(-1)

    return MagnitudePairRegressor()


def _batch(pairs, torch, device):
    a = torch.tensor([x for x, _ in pairs], dtype=torch.long, device=device)
    b = torch.tensor([y for _, y in pairs], dtype=torch.long, device=device)
    target = torch.tensor(
        [checked_answer(x, y, "+") / 18.0 for x, y in pairs],
        dtype=torch.float32, device=device)
    return a, b, target


def _metrics(model, batch, torch):
    model.eval()
    with torch.no_grad():
        a, b, target = batch
        predicted = model(a, b) * 18.0
        truth = target * 18.0
        mse = float(torch.mean((predicted - truth) ** 2).item())
        correct = int((torch.round(predicted) == torch.round(truth)).sum().item())
        total = len(a)
    return {"total": total, "correct": correct,
            "accuracy": round(correct / total, 4),
            "mse": round(mse, 6)}


def _categorical_comparison(store, split, seed):
    """Re-evaluate the previous token-embedding candidate on the SAME test."""
    ref = store.get("rule_lab_latest_candidate")
    if not ref or ref.get("seed") != seed:
        return {"status": "not_available_for_this_seed",
                "how_to_compare": f"Run rule-lab --steps 800 --seed {seed}"}
    try:
        from .rule_lab import _build_model as build_old, _metrics as old_metrics
        from .rule_lab import _data as old_batch, FORMAT as old_format
        torch, _ = require_torch()
        record = torch.load(ref["path"], map_location="cpu", weights_only=True)
        if record.get("format") != old_format or record.get("partitions") != split:
            return {"status": "mismatched_split_or_format"}
        if sha256_file(ref["path"]) != ref["sha256"]:
            return {"status": "checkpoint_hash_mismatch"}
        model = build_old(record["hidden"])
        model.load_state_dict(record["weights"])
        batch = old_batch(split["test"], torch, "cpu")
        return {"status": "comparable", "test": old_metrics(model, batch, torch),
                "accepted": bool(ref["accepted"]),
                "checkpoint_sha256": ref["sha256"],
                "architecture": "categorical_digit_embeddings_mlp_classifier"}
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return {"status": "could_not_read_checkpoint", "reason": str(error)[:150]}


def run_magnitude_lab(store, settings, steps=800, seed=31415, hidden=32):
    steps = int(steps)
    if not 1 <= steps <= 3000:
        raise ValueError("Steps must be 1..3000")
    seed = int(seed)
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(seed)
    random.seed(seed)
    split = split_addition_pairs(seed)
    data = {key: _batch(pairs, torch, device) for key, pairs in split.items()}
    model = _build_model(hidden).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    before = {
        key: _metrics(model, data[key], torch)
        for key in ("train", "validation")
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    a, b, target = data["train"]
    best_mse = float("inf")
    best_weights = None
    best_step = 0
    no_improve = 0
    actual_steps = 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        loss = torch.nn.functional.mse_loss(model(a, b), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        actual_steps = step
        if step % 10 == 0 or step == steps:
            validation = _metrics(model, data["validation"], torch)["mse"]
            if math.isfinite(validation) and validation + 0.000001 < best_mse:
                best_mse = validation
                best_step = step
                best_weights = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 15:
                    break
        if time.monotonic() - start > float(settings["train"].get("max_seconds", 600)):
            break
    if best_weights is None:
        return {"status": "rejected", "reason": "No finite validation checkpoint"}
    model.load_state_dict(best_weights)
    after = {
        key: _metrics(model, data[key], torch)
        for key in ("train", "validation")
    }
    accepted = (math.isfinite(after["validation"]["mse"]) and
                after["validation"]["mse"] + 0.0001 < before["validation"]["mse"])

    # Test is accessed only AFTER the checkpoint has been selected.
    initial_model = _build_model(hidden).to(device)
    initial_model.load_state_dict(initial)
    before["test"] = _metrics(initial_model, data["test"], torch)
    after["test"] = _metrics(model, data["test"], torch)
    model.eval()
    with torch.no_grad():
        ta, tb, ty = data["test"]
        outputs = (model(ta, tb) * 18.0).tolist()
    examples = [
        {"question": f"{a} + {b}", "expected": checked_answer(a, b, "+"),
         "model_answer": int(round(outputs[i])),
         "raw_prediction": round(float(outputs[i]), 4),
         "correct": int(round(outputs[i])) == checked_answer(a, b, "+")}
        for i, (a, b) in enumerate(split["test"])
    ]
    out = store.root / "models" / "magnitude_lab"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out / f"magnitude-seed{seed}-{stamp}.pt"
    torch.save({
        "format": FORMAT, "weights": best_weights,
        "hidden": int(hidden), "seed": seed, "splits": split,
        "best_step": best_step,
        "validation_mse": after["validation"]["mse"],
    }, path)
    digest = sha256_file(path)
    record = {
        "path": str(path), "sha256": digest,
        "seed": seed, "hidden": int(hidden),
        "format": FORMAT, "accepted": accepted,
        "test_accuracy": after["test"]["accuracy"],
    }
    store.put("magnitude_lab_latest_candidate", record)
    if accepted:
        store.put("magnitude_lab_latest", record)
    from .rule_lab import _existing_additive_baseline
    additive = _existing_additive_baseline(store, split, seed)
    categorical = _categorical_comparison(store, split, seed)
    report = {
        "status": "trained", "experiment": FORMAT,
        "architecture": "normalized_ordinal_inputs_generic_mlp_regression",
        "explicit_prior": "Numeric order and scale, unordered pair symmetry; NO summation layer",
        "task": "supervised_single_digit_addition",
        "seed": seed, "device": device,
        "split_policy": "35 train / 10 validation / 10 test disjoint unordered pairs",
        "before": before, "after": after,
        "train_groups": 35, "validation_groups": 10, "test_groups": 10,
        "best_step": best_step, "actual_steps": actual_steps,
        "accepted": accepted, "checkpoint": str(path), "sha256": digest,
        "test_examples": examples,
        "categorical_embedding_baseline": categorical,
        "explicit_additive_baseline": additive,
        "note": "Numeric magnitudes and regression are strong priors. Test metrics do not prove general arithmetic, web learning or autonomous skill invention.",
    }
    store.log("magnitude_lab", "completed", {
        key: value for key, value in report.items() if key != "test_examples"
    })
    return report


def run_magnitude_study(store, settings, seeds=(31415, 42, 1001), steps=800, hidden=32):
    seeds = tuple(int(x) for x in seeds)
    if not 2 <= len(seeds) <= 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply 2..5 distinct seeds, fixed before seeing test metrics")
    results = [run_magnitude_lab(store, settings, steps=steps,
                                 seed=seed, hidden=hidden) for seed in seeds]
    scores = [r["after"]["test"]["accuracy"] for r in results
              if r.get("status") == "trained"]
    report = {
        "status": "evaluated",
        "experiment": FORMAT,
        "seeds": list(seeds),
        "results": [
            {"seed": r["seed"], "accepted": r["accepted"],
             "train_accuracy": r["after"]["train"]["accuracy"],
             "validation_accuracy": r["after"]["validation"]["accuracy"],
             "test_accuracy": r["after"]["test"]["accuracy"],
             "checkpoint": r["checkpoint"]}
            for r in results if r.get("status") == "trained"
        ],
        "test_mean": round(statistics.mean(scores), 4) if scores else None,
        "test_min": min(scores) if scores else None,
        "test_max": max(scores) if scores else None,
        "caveat": "No best-seed selection. Validation chooses weights per seed; these repeated tests are a research diagnostic, not an untouched final confirmatory evaluation.",
    }
    store.log("magnitude_study", "completed", report)
    return report


def predict_magnitude_lab(store, a, b, candidate=False):
    a, b = int(a), int(b)
    if not (0 <= a <= 9 and 0 <= b <= 9):
        raise ValueError("Magnitude lab accepts only digits 0..9")
    key = "magnitude_lab_latest_candidate" if candidate else "magnitude_lab_latest"
    record = store.get(key)
    if record is None:
        return {"status": "skipped", "reason": "No matching magnitude checkpoint"}
    torch, _ = require_torch()
    if sha256_file(record["path"]) != record["sha256"]:
        raise ValueError("Magnitude checkpoint hash mismatch")
    data = torch.load(record["path"], map_location="cpu", weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError("Unrecognized magnitude checkpoint")
    model = _build_model(data["hidden"])
    model.load_state_dict(data["weights"])
    model.eval()
    with torch.no_grad():
        raw = float((model(torch.tensor([a]), torch.tensor([b])) * 18.0).item())
    return {
        "status": "predicted",
        "question": f"{a} + {b}", "model_answer": int(round(raw)),
        "raw_prediction": round(raw, 4),
        "checkpoint_status": "accepted" if record["accepted"] else "unaccepted_candidate",
        "checkpoint": record["sha256"],
        "architecture": FORMAT,
        "note": "Prediction from trained MLP and numeric input prior, not an arithmetic calculator.",
    }
