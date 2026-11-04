"""v1.5: shared three-operation network with learned bilinear interactions.

This experiment adds a *multiplicative interaction operator between learned
hidden activations*. That is an important architectural prior: it makes the
multiplication relation representable. No operation's correct answer is
computed in forward(); true arithmetic is used only for supervised targets
and final scoring. This is NOT autonomous arithmetic-rule discovery.

Selection is based strictly on held-out in-range validation. OOD diagnostics
are evaluated once after selecting the weights, including a new 30..99 probe
range. Existing v1.4 checkpoints are loaded read-only and never promoted.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import random
import statistics
import time

from .brain import get_device, require_torch, sha256_file
from .extrapolation import RANGES, select_ood_pairs
from .multiop_lab import (
    OPS, SCALES, FORMAT as V14_FORMAT, _build_model as v14_model,
    _examples, _scored, _tensor_batch,
)
from .numeric_lab import split_addition_pairs

FORMAT = "shared_learned_bilinear_multiop_v1"
OOD_GROUPS = (*RANGES.keys(), "both_30_to_49", "both_50_to_99")


def _build_model(rank=8):
    torch, nn = require_torch()
    rank = int(rank)
    if not 2 <= rank <= 64:
        raise ValueError("Interaction rank must be 2..64")

    class LearnedInteractionNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Two trainable projections followed by elementwise interaction.
            # Multiplying hidden activations is a supplied bilinear prior.
            self.left = nn.Linear(1, rank)
            self.right = nn.Linear(1, rank)
            self.head = nn.Linear(rank + 2, len(OPS))

        def forward(self, a, b, operations):
            x = a.float().unsqueeze(1) / 9.0
            y = b.float().unsqueeze(1) / 9.0
            cross = self.left(x) * self.right(y)
            features = torch.cat((x, y, cross), dim=-1)
            scores = self.head(features)
            return scores.gather(1, operations.unsqueeze(1)).squeeze(1)

    return LearnedInteractionNet()


def _ood_pairs(group, count, seed):
    count = int(count)
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    if group in RANGES:
        return select_ood_pairs(group, count=count, seed=seed)
    if group == "both_30_to_49":
        low, high, salt = 30, 49, 1147
    elif group == "both_50_to_99":
        low, high, salt = 50, 99, 2173
    else:
        raise ValueError("Unknown OOD group")
    pairs = [(a, b) for a in range(low, high + 1)
             for b in range(a, high + 1)]
    random.Random(int(seed) + salt).shuffle(pairs)
    return pairs[:count]


def _v14_reference(store, seed, split, count, probe_seed):
    ref = store.get(f"multiop_lab_seed_{int(seed)}")
    if not ref:
        return {"status": "missing_matching_seed_v14_checkpoint"}
    path = Path(ref["path"]).resolve()
    root = (store.root / "models" / "multiop_lab").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return {"status": "missing_or_outside_model_folder"}
    if sha256_file(path) != ref["sha256"]:
        return {"status": "sha256_mismatch"}
    torch, _ = require_torch()
    data = torch.load(path, weights_only=True, map_location="cpu")
    if (data.get("format") != V14_FORMAT or data.get("split") != split
            or int(data.get("seed", -1)) != int(seed)
            or tuple(data.get("operations", ())) != OPS
            or tuple(data.get("scales", ())) != SCALES):
        return {"status": "mismatched_seed_split_or_format"}
    model = v14_model(int(data["hidden"]))
    model.load_state_dict(data["weights"])
    model.eval()
    return {
        "status": "comparable",
        "checkpoint_sha256": ref["sha256"],
        "checkpoint_status": "accepted" if ref["accepted"] else "unaccepted_candidate",
        "ood": {
            group: {
                op: {k: v for k, v in vals.items() if k != "examples"}
                for op, vals in _scored(
                    model, _tensor_batch(
                        _examples(_ood_pairs(group, count, probe_seed)),
                        torch, "cpu"), torch, sample_limit=0).items()
            } for group in OOD_GROUPS
        },
    }


def _load_checkpoint(store, seed=None, candidate=False):
    if seed is not None:
        ref = store.get(f"interaction_lab_seed_{int(seed)}")
    else:
        ref = store.get("interaction_lab_latest_candidate" if candidate
                        else "interaction_lab_latest")
    if ref is None:
        return None
    path = Path(ref["path"]).resolve()
    root = (store.root / "models" / "interaction_lab").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Interaction checkpoint missing/outside model storage")
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("Interaction checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("format") != FORMAT or
        int(data["seed"]) != int(ref["seed"]) or
        data["split"] != split_addition_pairs(int(ref["seed"])) or
        tuple(data.get("operations", ())) != OPS or
        tuple(data.get("scales", ())) != SCALES):
        raise ValueError("Incompatible interaction checkpoint")
    model = _build_model(int(data["rank"]))
    model.load_state_dict(data["weights"])
    model.eval()
    return model, ref


def run_interaction_lab(store, settings, seed=31415, steps=800,
                        rank=8, count=20, probe_seed=20261104):
    steps, seed, rank, count = int(steps), int(seed), int(rank), int(count)
    if not 1 <= steps <= 3000:
        raise ValueError("Steps must be 1..3000")
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    if not 2 <= rank <= 64:
        raise ValueError("Rank must be 2..64")
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(seed)
    split = split_addition_pairs(seed)
    batches = {name: _tensor_batch(_examples(pairs), torch, device)
               for name, pairs in split.items()}
    model = _build_model(rank).to(device)
    before_train = _scored(model, batches["train"], torch, sample_limit=0)
    before_val = _scored(model, batches["validation"], torch, sample_limit=0)
    initial_val_score = sum(before_val[op]["normalized_mse"] for op in OPS) / 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    a, b, ops, targets = batches["train"]
    best_weights, best_step, best_loss = None, 0, float("inf")
    checks_without_improvement, actual_steps = 0, 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        loss = torch.nn.functional.mse_loss(model(a, b, ops), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_steps = step
        if step % 20 == 0 or step == steps:
            current = _scored(model, batches["validation"], torch, sample_limit=0)
            val_loss = sum(current[op]["normalized_mse"] for op in OPS) / 3
            if math.isfinite(val_loss) and val_loss + 1e-8 < best_loss:
                best_loss = val_loss
                best_step = step
                best_weights = {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()}
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= 12:
                    break
        if time.monotonic() - start >= float(settings["train"].get("max_seconds", 600)):
            break
    if best_weights is None:
        return {"status": "rejected", "reason": "No finite validation checkpoint",
                "seed": seed}
    model.load_state_dict(best_weights)
    after_train = _scored(model, batches["train"], torch)
    after_val = _scored(model, batches["validation"], torch)
    accepted = math.isfinite(best_loss) and best_loss + 1e-5 < initial_val_score

    # Held-out test and OOD are not consulted for checkpoint selection.
    test = _scored(model, batches["test"], torch)
    frozen = _build_model(rank)
    frozen.load_state_dict(best_weights)
    frozen.eval()
    ood = {
        group: _scored(
            frozen, _tensor_batch(_examples(_ood_pairs(group, count, probe_seed)),
                                  torch, "cpu"), torch)
        for group in OOD_GROUPS
    }
    baseline = _v14_reference(store, seed, split, count, probe_seed)
    folder = store.root / "models" / "interaction_lab"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint_path = folder / f"interaction-seed{seed}-{stamp}.pt"
    torch.save({
        "format": FORMAT, "seed": seed, "rank": rank,
        "split": split, "weights": best_weights,
        "scales": SCALES, "operations": OPS,
        "best_step": best_step,
        "validation_normalized_mse": best_loss,
    }, checkpoint_path)
    digest = sha256_file(checkpoint_path)
    record = {
        "path": str(checkpoint_path), "sha256": digest,
        "seed": seed, "rank": rank,
        "accepted": accepted, "format": FORMAT,
    }
    store.put(f"interaction_lab_seed_{seed}", record)
    store.put("interaction_lab_latest_candidate", record)
    if accepted:
        store.put("interaction_lab_latest", record)
    report = {
        "status": "trained", "experiment": FORMAT,
        "architecture": "trainable_bilinear_hidden_interactions_shared_three_op_heads",
        "explicit_priors": (
            "Hidden activation multiplication (bilinearity), numeric magnitudes, "
            "operator labels and known output scales; true formulas NOT in forward"
        ),
        "seed": seed, "device": device, "rank": rank,
        "training_operand_range": [0, 9],
        "train_groups": 35, "validation_groups": 10, "test_groups": 10,
        "training_examples": len(batches["train"][0]),
        "validation_examples": len(batches["validation"][0]),
        "test_examples": len(batches["test"][0]),
        "before_train": before_train, "before_validation": before_val,
        "after_train": after_train, "after_validation": after_val,
        "in_range_test": test, "ood": ood,
        "frozen_v14_reference": baseline,
        "accepted": accepted, "best_step": best_step, "actual_steps": actual_steps,
        "checkpoint": str(checkpoint_path), "sha256": digest,
        "note": (
            "Strong bilinear architectural bias makes multiplication representable. "
            "High OOD accuracy would not imply autonomous discovery or general reasoning. "
            "Repeated probe sets are developmental, not untouched confirmatory evaluation."
        ),
    }
    store.log("interaction_lab", "completed", {
        **{k: v for k, v in report.items() if k != "ood"},
        "ood": {
            group: {op: {k: v for k, v in measures.items()
                         if k != "examples"} for op, measures in scores.items()}
            for group, scores in ood.items()
        },
    })
    return report


def predict_interaction_lab(store, a, b, operator, seed=None, candidate=False):
    if operator not in OPS:
        raise ValueError("Operator must be +, -, or *")
    a, b = int(a), int(b)
    if not (0 <= a <= 99 and 0 <= b <= 99):
        raise ValueError("Supported operand domain is 0..99")
    selected = _load_checkpoint(store, seed=seed, candidate=candidate)
    if selected is None:
        return {"status": "skipped", "reason": "No matching interaction checkpoint"}
    model, record = selected
    torch, _ = require_torch()
    with torch.no_grad():
        normalized = float(model(
            torch.tensor([a]), torch.tensor([b]),
            torch.tensor([OPS.index(operator)])
        ).item())
    raw = normalized * SCALES[OPS.index(operator)]
    return {
        "status": "predicted", "question": f"{a} {operator} {b}",
        "model_answer": int(round(raw)),
        "raw_prediction": round(raw, 4),
        "checkpoint_status": "accepted" if record["accepted"] else "unaccepted_candidate",
        "checkpoint": record["sha256"], "experiment": FORMAT,
        "note": "Prediction uses learned bilinear hidden interactions, not a calculator.",
    }


def run_interaction_study(store, settings, seeds=(31415, 42, 1001),
                          steps=800, rank=8, count=20, probe_seed=20261104):
    seeds = tuple(int(s) for s in seeds)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply 2..10 distinct seeds")
    trials = [run_interaction_lab(
        store, settings, seed=s, steps=steps, rank=rank, count=count,
        probe_seed=probe_seed) for s in seeds]
    reports = []
    for t in trials:
        if t["status"] != "trained":
            reports.append({"seed": t["seed"], "status": t["status"]})
            continue
        reports.append({
            "seed": t["seed"], "accepted": t["accepted"],
            "in_range_accuracy": {
                op: t["in_range_test"][op]["accuracy"] for op in OPS
            },
            "ood_accuracy": {
                group: {op: t["ood"][group][op]["accuracy"] for op in OPS}
                for group in OOD_GROUPS
            },
            "v14_reference": {
                group: {op: t["frozen_v14_reference"]["ood"][group][op]["accuracy"]
                        for op in OPS}
                for group in OOD_GROUPS
            } if t["frozen_v14_reference"]["status"] == "comparable" else None,
            "checkpoint": t["checkpoint"],
        })
    aggregate = {}
    for group in OOD_GROUPS:
        aggregate[group] = {}
        for op in OPS:
            values = [r["ood_accuracy"][group][op] for r in reports
                      if "ood_accuracy" in r]
            aggregate[group][op] = {
                "mean": round(statistics.mean(values), 4) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    return {
        "status": "evaluated" if any("ood_accuracy" in r for r in reports) else "skipped",
        "experiment": FORMAT, "seeds": list(seeds),
        "runs": reports, "ood_summary": aggregate,
        "caveat": (
            "All seeds share fixed OOD probes; counts are not independent. "
            "Bilinear hidden interactions are a strong supplied prior, not discovered mathematics."
        ),
    }
