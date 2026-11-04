"""v1.4: A single supervised neural network for +, -, * with honest holdouts.

One shared numerical representation and three learned operator output heads.
The forward pass contains NO arithmetic formulas (no a+b, a-b or a*b);
Python's checked_answer is used exclusively for labels and evaluation.

Supplied priors: ordered continuous operands, one-hot operation codes,
per-operation output scales, and a smooth MLP. This is a *controlled study*,
not general mathematical reasoning, autonomous discovery, or web learning.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import random
import statistics
import time

from .brain import get_device, require_torch, sha256_file
from .extrapolation import RANGES, select_ood_pairs, _evaluate_group
from .linear_lab import _read_linear
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

FORMAT = "shared_multiop_numeric_mlp_v1"
OPS = ("+", "-", "*")
SCALES = (18.0, 9.0, 81.0)  # Known output ranges are task priors.


def _build_model(hidden=64):
    torch, nn = require_torch()
    hidden = int(hidden)
    if not 8 <= hidden <= 256:
        raise ValueError("Hidden size must be 8..256")

    class SharedMultiOp(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(5, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 3),
            )

        def forward(self, a, b, operations):
            op_features = torch.nn.functional.one_hot(operations, num_classes=3).float()
            # Preserve operand order: a-b and b-a are different.
            features = torch.cat((
                a.float().unsqueeze(1) / 9.0,
                b.float().unsqueeze(1) / 9.0,
                op_features,
            ), dim=1)
            outputs = self.body(features)
            return outputs.gather(1, operations.unsqueeze(1)).squeeze(1)

    return SharedMultiOp()


def _examples(pairs):
    """Both operand orders stay in the same split for all three operators."""
    for a, b in pairs:
        orientations = ((a, b),) if a == b else ((a, b), (b, a))
        for left, right in orientations:
            for operator in OPS:
                yield (left, right, operator)


def _tensor_batch(examples, torch, device):
    items = list(examples)
    a = torch.tensor([left for left, _, _ in items], dtype=torch.long, device=device)
    b = torch.tensor([right for _, right, _ in items], dtype=torch.long, device=device)
    ops = torch.tensor([OPS.index(op) for _, _, op in items],
                       dtype=torch.long, device=device)
    targets = torch.tensor([
        checked_answer(left, right, op) / SCALES[OPS.index(op)]
        for left, right, op in items
    ], dtype=torch.float32, device=device)
    return a, b, ops, targets


def _scored(model, batch, torch, sample_limit=3):
    """Evaluate each operator separately; do not conflate low loss with skill."""
    a, b, ops, targets = batch
    model.eval()
    with torch.no_grad():
        normalized = model(a, b, ops)
        normalizers = torch.tensor(SCALES, dtype=torch.float32, device=a.device)[ops]
        predicted = normalized * normalizers
        truth = targets * normalizers
        rounded = predicted.round()
        errors = (predicted - truth).detach().cpu().tolist()
        predicted_values = predicted.detach().cpu().tolist()
        truth_values = truth.detach().cpu().tolist()
        raw_a, raw_b, raw_ops = a.cpu().tolist(), b.cpu().tolist(), ops.cpu().tolist()
    aggregate = {}
    for idx, op in enumerate(OPS):
        rows = [i for i, item in enumerate(raw_ops) if item == idx]
        correct = sum(int(round(predicted_values[i]) == round(truth_values[i])) for i in rows)
        count = len(rows)
        mse_normalized = sum((errors[i] / SCALES[idx]) ** 2 for i in rows) / count
        samples = [{
            "question": f"{raw_a[i]} {op} {raw_b[i]}",
            "expected": int(round(truth_values[i])),
            "prediction": int(round(predicted_values[i])),
            "raw_prediction": round(predicted_values[i], 4),
            "correct": round(predicted_values[i]) == round(truth_values[i]),
        } for i in rows[:sample_limit]]
        aggregate[op] = {
            "questions": count,
            "correct": correct,
            "accuracy": round(correct / count, 4),
            "normalized_mse": round(mse_normalized, 6),
            "mean_absolute_error": round(statistics.mean(abs(errors[i]) for i in rows), 4),
            "mean_signed_error": round(statistics.mean(errors[i] for i in rows), 4),
            "examples": samples,
        }
    return aggregate


def _validation_score(model, batch, torch):
    result = _scored(model, batch, torch, sample_limit=0)
    return sum(result[op]["normalized_mse"] for op in OPS) / len(OPS)


def _safe_record(store, path, digest, seed, hidden, accepted):
    return {
        "path": str(path), "sha256": digest, "seed": int(seed),
        "hidden": int(hidden), "accepted": bool(accepted), "format": FORMAT,
    }


def run_multiop_lab(store, settings, seed=31415, steps=800,
                    hidden=64, count=20, probe_seed=20261104):
    steps = int(steps)
    hidden = int(hidden)
    count = int(count)
    if not 1 <= steps <= 3000:
        raise ValueError("Steps must be 1..3000")
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    if not 8 <= hidden <= 256:
        raise ValueError("Hidden must be 8..256")
    seed = int(seed)
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(seed)
    random.seed(seed)
    split = split_addition_pairs(seed)
    batches = {
        name: _tensor_batch(_examples(pairs), torch, device)
        for name, pairs in split.items()
    }
    model = _build_model(hidden).to(device)
    initial_weights = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    initial_train = _scored(model, batches["train"], torch, sample_limit=0)
    initial_validation = _scored(model, batches["validation"], torch, sample_limit=0)
    before_score = _validation_score(model, batches["validation"], torch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0001)
    a, b, op, targets = batches["train"]
    best_score = float("inf")
    best_weights = None
    best_step = 0
    no_improve = 0
    actual_steps = 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        output = model(a, b, op)
        loss = torch.nn.functional.mse_loss(output, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_steps = step
        if step % 20 == 0 or step == steps:
            score = _validation_score(model, batches["validation"], torch)
            if math.isfinite(score) and score + 1e-6 < best_score:
                best_score = score
                best_step = step
                best_weights = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 12:
                    break
        if time.monotonic() - start > float(settings["train"].get("max_seconds", 600)):
            break
    if best_weights is None:
        return {"status": "rejected", "reason": "No finite validation checkpoint"}
    model.load_state_dict(best_weights)
    after_train = _scored(model, batches["train"], torch)
    after_validation = _scored(model, batches["validation"], torch)
    accepted = (best_score + 0.0001 < before_score)

    # Test and OOD are NOT read until the validation checkpoint is frozen.
    initial_model = _build_model(hidden).to(device)
    initial_model.load_state_dict(initial_weights)
    before_test = _scored(initial_model, batches["test"], torch, sample_limit=0)
    after_test = _scored(model, batches["test"], torch)

    cpu = _build_model(hidden)
    cpu.load_state_dict({key: value.cpu() for key, value in best_weights.items()})
    ood = {}
    for group in RANGES:
        pairs = select_ood_pairs(group, count=count, seed=probe_seed)
        ood[group] = _scored(
            cpu, _tensor_batch(_examples(pairs), torch, "cpu"), torch)
    linear_comparison = {"status": "no_matching_seed_accepted_checkpoint"}
    baseline = _read_linear(store, seed=seed)
    if baseline is not None:
        baseline_model, baseline_record, _ = baseline
        linear_comparison = {
            "status": "addition_only_comparable",
            "checkpoint": baseline_record["sha256"],
            "accepted": baseline_record["accepted"],
            "groups": {
                group: {
                    k: v for k, v in _evaluate_group(
                        baseline_model, torch,
                        select_ood_pairs(group, count=count, seed=probe_seed)
                    ).items() if k not in ("examples", "mistakes")
                } for group in RANGES
            },
            "caveat": "v1.3 linear baseline was trained ONLY on addition. Other operations have no corresponding v1.3 baseline.",
        }

    folder = store.root / "models" / "multiop_lab"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"multiop-seed{seed}-{stamp}.pt"
    torch.save({
        "format": FORMAT, "seed": seed, "hidden": hidden,
        "weights": best_weights, "split": split, "best_step": best_step,
        "validation_normalized_mse": best_score,
        "scales": SCALES, "operations": OPS,
    }, path)
    digest = sha256_file(path)
    record = _safe_record(store, path, digest, seed, hidden, accepted)
    store.put(f"multiop_lab_seed_{seed}", record)
    store.put("multiop_lab_latest_candidate", record)
    if accepted:
        store.put("multiop_lab_latest", record)
    report = {
        "status": "trained", "experiment": FORMAT,
        "architecture": "shared_mlp_with_operator_features_and_separate_output_heads",
        "explicit_priors": "Known operand magnitudes, operator codes, per-operation output scales; no arithmetic formulas in neural forward",
        "training_domain": "operands 0..9, three supervised operations",
        "seed": seed, "device": device,
        "train_groups": len(split["train"]), "validation_groups": len(split["validation"]),
        "test_groups": len(split["test"]),
        "train_examples": len(batches["train"][0]),
        "validation_examples": len(batches["validation"][0]),
        "test_examples": len(batches["test"][0]),
        "before": {"train": initial_train, "validation": initial_validation,
                   "test": before_test},
        "after": {"train": after_train, "validation": after_validation,
                  "test": after_test},
        "ood": ood,
        "frozen_addition_linear_baseline": linear_comparison,
        "accepted": accepted, "best_step": best_step, "actual_steps": actual_steps,
        "checkpoint": str(path), "sha256": digest,
        "note": "A shared supervised three-operator model with numeric magnitude priors; performance is measured, not guaranteed. OOD groups repeat earlier stress sets.",
    }
    store.log("multiop_lab", "completed", {
        **report, "ood": {
            group: {
                op: {key: value for key, value in d.items() if key != "examples"}
                for op, d in records.items()
            } for group, records in ood.items()
        },
    })
    return report


def predict_multiop_lab(store, a, b, operator, seed=None, candidate=False):
    a, b = int(a), int(b)
    if operator not in OPS:
        raise ValueError("Operator must be +, - or *")
    if not (0 <= a <= 99 and 0 <= b <= 99):
        raise ValueError("Operands must be nonnegative integers 0..99")
    record = (store.get(f"multiop_lab_seed_{int(seed)}") if seed is not None
              else store.get("multiop_lab_latest_candidate" if candidate
                             else "multiop_lab_latest"))
    if record is None:
        return {"status": "skipped", "reason": "No matching shared arithmetic checkpoint"}
    path = Path(record["path"]).resolve()
    root = (store.root / "models" / "multiop_lab").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Checkpoint missing or outside multiop model directory")
    if sha256_file(path) != record["sha256"]:
        raise ValueError("Shared arithmetic checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format") != FORMAT or saved.get("split") != split_addition_pairs(int(saved["seed"]))
            or tuple(saved.get("operations", ())) != OPS or tuple(saved.get("scales", ())) != SCALES):
        raise ValueError("Incompatible shared arithmetic checkpoint")
    model = _build_model(saved["hidden"])
    model.load_state_dict(saved["weights"])
    model.eval()
    with torch.no_grad():
        result = model(torch.tensor([a]), torch.tensor([b]),
                       torch.tensor([OPS.index(operator)]))
        raw = float(result.item()) * SCALES[OPS.index(operator)]
    return {
        "status": "predicted",
        "question": f"{a} {operator} {b}",
        "model_answer": int(round(raw)),
        "raw_prediction": round(raw, 4),
        "checkpoint_status": "accepted" if record["accepted"] else "unaccepted_candidate",
        "checkpoint": record["sha256"],
        "experiment": FORMAT,
        "note": "Predicted by one shared neural network, not by Python arithmetic.",
    }


def run_multiop_study(store, settings, seeds=(31415, 42, 1001), steps=800,
                      count=20, probe_seed=20261104, hidden=64):
    seeds = tuple(int(x) for x in seeds)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Specify 2..10 distinct seeds before evaluating test results")
    runs = [run_multiop_lab(store, settings, seed=seed, steps=steps,
                            count=count, probe_seed=probe_seed, hidden=hidden)
            for seed in seeds]
    outcomes = []
    for run in runs:
        if run["status"] != "trained":
            outcomes.append({"seed": run.get("seed"), "status": run["status"]})
            continue
        outcomes.append({
            "seed": run["seed"], "accepted": run["accepted"],
            "in_range_accuracy": {
                op: run["after"]["test"][op]["accuracy"] for op in OPS
            },
            "ood_accuracy": {
                group: {op: run["ood"][group][op]["accuracy"] for op in OPS}
                for group in RANGES
            },
            "checkpoint": run["checkpoint"],
        })
    summary = {}
    for group in RANGES:
        summary[group] = {}
        for op in OPS:
            values = [row["ood_accuracy"][group][op] for row in outcomes
                      if "ood_accuracy" in row]
            summary[group][op] = {
                "mean": round(statistics.mean(values), 4) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    return {
        "status": "evaluated" if any("ood_accuracy" in r for r in outcomes) else "skipped",
        "experiment": FORMAT,
        "seeds": list(seeds),
        "runs": outcomes,
        "ood_summary": summary,
        "caveat": "The same OOD questions are reused across model seeds, so counts are not independent. No model selection by final test or OOD accuracy.",
    }
