"""v1.7: calibrate ONLY the output head of an existing v1.5 bilinear model.

Keep saved v1.5 learned projections (left/right) frozen. Refit the SAME
operator-specific output heads by CPU float64 least squares using ONLY the
original 35 unordered training operand groups. No polynomial feature basis,
ideal coefficients, test labels or OOD labels are supplied to this fit.

The architecture is bilinear, which is a strong supplied inductive bias.
This is supervised calibration with a numerical solver, NOT new gradient
training, autonomous discovery, or general-language-model reasoning.
"""
from datetime import datetime, timezone
from pathlib import Path
import statistics

from .brain import require_torch, sha256_file
from .interaction_lab import (
    _load_checkpoint as load_v15, FORMAT as SOURCE_FORMAT,
    _build_model as build_v15,
)
from .basis_lab import GROUPS, _ood_pairs, _score, _effective_v15_coefficients
from .multiop_lab import OPS, SCALES, _examples
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

FORMAT = "frozen_bilinear_hidden_head_calibration_v1"


def _build_model(rank=8):
    """Same layers and normalization as v1.5, with float64 inputs/weights.

    v1.5 calls float() on integer operands (float32); converting its module
    to double therefore requires the dtype-aware input version below.
    """
    torch, nn = require_torch()
    rank = int(rank)
    if not 2 <= rank <= 64:
        raise ValueError("Invalid hidden rank")

    class FrozenInteractionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(1, rank, dtype=torch.float64)
            self.right = nn.Linear(1, rank, dtype=torch.float64)
            self.head = nn.Linear(rank + 2, len(OPS), dtype=torch.float64)

        def features(self, a, b):
            x = a.to(torch.float64).reshape(-1, 1) / 9.0
            y = b.to(torch.float64).reshape(-1, 1) / 9.0
            cross = self.left(x) * self.right(y)
            return torch.cat((x, y, cross), dim=-1)

        def forward(self, a, b, operations):
            values = self.head(self.features(a, b))
            return values.gather(1, operations.unsqueeze(1)).squeeze(1)

    return FrozenInteractionModel()


def _frozen_features(model, examples):
    torch, _ = require_torch()
    items = list(examples)
    a = torch.tensor([left for left, _, _ in items], dtype=torch.long)
    b = torch.tensor([right for _, right, _ in items], dtype=torch.long)
    with torch.no_grad():
        features = model.features(a, b)
        return torch.cat((
            torch.ones((len(items), 1), dtype=torch.float64), features
        ), dim=1)


def _fit_head(model, train_pairs):
    """Fit the existing v1.5 head on its saved training pair groups only."""
    torch, _ = require_torch()
    examples = list(_examples(train_pairs))
    matrices = {}
    diagnostics = {}
    for idx, operation in enumerate(OPS):
        train_examples = [(a,b,op) for a,b,op in examples if op == operation]
        matrix = _frozen_features(model, train_examples)
        # All the head features are polynomials of degree <= 2; expected
        # matrix rank is 4. We do NOT introduce the explicit [1,x,y,xy]
        # basis as model features or substitute any ideal coefficients.
        numeric_rank = int(torch.linalg.matrix_rank(matrix).item())
        if numeric_rank != 4:
            raise ValueError(
                f"Frozen features for {operation} have rank {numeric_rank}, not 4")
        targets = torch.tensor([
            checked_answer(a, b, operation) / SCALES[idx]
            for a, b, _ in train_examples
        ], dtype=torch.float64)
        solution = torch.linalg.lstsq(matrix, targets).solution
        residual = float((matrix @ solution - targets).abs().max().item())
        if not torch.isfinite(solution).all():
            raise ArithmeticError("Non-finite fitted head")
        matrices[operation] = solution
        diagnostics[operation] = {
            "feature_rank": numeric_rank,
            "training_rows": len(train_examples),
            "max_train_normalized_residual": round(residual, 12),
        }
    with torch.no_grad():
        for idx, operation in enumerate(OPS):
            solution = matrices[operation]
            model.head.bias[idx].copy_(solution[0])
            model.head.weight[idx].copy_(solution[1:])
    return diagnostics


def _only_metrics(results):
    return {op: {key: value for key, value in stats.items() if key != "examples"}
            for op, stats in results.items()}


def _load_calibrated(store, seed=None):
    ref = (store.get(f"calibration_lab_seed_{int(seed)}") if seed is not None
           else store.get("calibration_lab_latest"))
    if ref is None:
        return None
    path = Path(ref["path"]).resolve()
    root = (store.root / "models" / "calibration_lab").resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("Calibrated checkpoint missing or outside model directory")
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("Calibrated checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format") != FORMAT or
        int(saved["seed"]) != int(ref["seed"]) or
        saved["split"] != split_addition_pairs(int(ref["seed"])) or
        tuple(saved.get("operations", ())) != OPS or
        tuple(saved.get("scales", ())) != SCALES or
        saved.get("source_sha256") != ref["source_sha256"]):
        raise ValueError("Incompatible calibrated checkpoint")
    model = _build_model(saved["rank"])
    model.load_state_dict(saved["weights"])
    model.eval()
    return model, ref


def run_calibration_lab(store, seed=31415, count=20, probe_seed=20261104):
    seed, count = int(seed), int(count)
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    loaded = load_v15(store, seed=seed)
    if loaded is None:
        return {"status": "skipped", "seed": seed,
                "reason": "No saved v1.5 model for this seed"}
    source, source_ref = loaded
    torch, _ = require_torch()
    split = split_addition_pairs(seed)
    source_path = Path(source_ref["path"])
    source_hash = sha256_file(source_path)
    if source_hash != source_ref["sha256"]:
        raise ValueError("Original v1.5 checkpoint SHA256 mismatch")

    # Initialize exactly from the saved model's trained hidden parameters.
    model = _build_model(source.left.out_features)
    model.load_state_dict(source.state_dict())
    model.eval()
    before_hidden = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith("left.") or key.startswith("right.")
    }
    original = _score(model, _examples(split["validation"]), sample_limit=0)

    diagnostics = _fit_head(model, split["train"])
    model.eval()
    after = _score(model, _examples(split["validation"]), sample_limit=0)
    # The candidate must improve mean validation MAE; no OOD or test labels
    # influence promotion or fitting.
    initial_validation = statistics.mean(
        original[op]["mean_absolute_error"] for op in OPS)
    current_validation = statistics.mean(
        after[op]["mean_absolute_error"] for op in OPS)
    accepted = current_validation + 1e-7 < initial_validation
    for key, value in before_hidden.items():
        if not torch.equal(value, model.state_dict()[key]):
            raise AssertionError(f"Original hidden projection mutated: {key}")
    # The saved original source checkpoint must still be untouched.
    if sha256_file(source_path) != source_hash:
        raise AssertionError("Original v1.5 checkpoint modified")

    # Test and OOD are consulted ONLY after the training-only fit and
    # validation acceptance rule have been fixed.
    test = _score(model, _examples(split["test"]))
    source_test = _score(source, _examples(split["test"]))
    ood, source_ood = {}, {}
    for group in GROUPS:
        probes = list(_examples(_ood_pairs(group, count, probe_seed)))
        ood[group] = _score(model, probes)
        source_ood[group] = _score(source, probes)

    folder = store.root / "models" / "calibration_lab"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (
        f"calibration-seed{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.pt")
    torch.save({
        "format": FORMAT, "seed": seed, "rank": source.left.out_features,
        "split": split, "operations": OPS, "scales": SCALES,
        "source_sha256": source_hash,
        "weights": {k: v.detach().cpu().clone()
                    for k,v in model.state_dict().items()},
    }, path)
    digest = sha256_file(path)
    record = {
        "path": str(path), "sha256": digest,
        "seed": seed, "accepted": accepted,
        "source_sha256": source_hash, "format": FORMAT,
    }
    store.put(f"calibration_lab_seed_{seed}", record)
    store.put("calibration_lab_latest_candidate", record)
    if accepted:
        store.put("calibration_lab_latest", record)
    report = {
        "status": "calibrated",
        "experiment": FORMAT,
        "method": "frozen_saved_v15_hidden_projections_double_precision_head_least_squares",
        "training_data": "original_35_unordered_pairs_0_to_9_only",
        "explicit_prior": "The original saved v1.5 hidden multiplication stays intact; no manually supplied ideal coefficients",
        "seed": seed, "source_checkpoint_sha256": source_hash,
        "source_status": "accepted" if source_ref["accepted"] else "unaccepted_candidate",
        "hidden_layers_unchanged": True,
        "train_group_count": len(split["train"]),
        "validation_group_count": len(split["validation"]),
        "test_group_count": len(split["test"]),
        "fit_diagnostics": diagnostics,
        "validation_before": _only_metrics(original),
        "validation_after": _only_metrics(after),
        "mean_validation_mae_before": round(initial_validation, 8),
        "mean_validation_mae_after": round(current_validation, 8),
        "test": _only_metrics(test),
        "source_test": _only_metrics(source_test),
        "ood": ood,
        "source_ood": {group: _only_metrics(data) for group,data in source_ood.items()},
        "effective_coefficients": _effective_v15_coefficients(model),
        "accepted": accepted,
        "checkpoint": str(path), "sha256": digest,
        "note": "This is supervised output-head refitting of a pre-existing bilinear model, not new feature discovery or SGD neural learning. Large-operand probes are developmental diagnostics.",
    }
    store.log("calibration_lab", "completed", {
        **{k:v for k,v in report.items() if k not in ("ood","source_ood")},
        "ood": {group: _only_metrics(data) for group,data in ood.items()},
        "source_ood": source_ood,
    })
    return report


def predict_calibration_lab(store, a, b, operator, seed=31415):
    if operator not in OPS:
        raise ValueError("Operator must be +, - or *")
    a, b = int(a), int(b)
    if not (0 <= a <= 999 and 0 <= b <= 999):
        raise ValueError("Supported operands are 0..999")
    loaded = _load_calibrated(store, seed=seed)
    if loaded is None:
        return {"status": "skipped", "reason": "No calibrated model for requested seed"}
    model, ref = loaded
    torch, _ = require_torch()
    with torch.no_grad():
        result = model(
            torch.tensor([a]), torch.tensor([b]),
            torch.tensor([OPS.index(operator)]))
    raw = float(result.item()) * SCALES[OPS.index(operator)]
    return {
        "status": "predicted", "question": f"{a} {operator} {b}",
        "model_answer": int(round(raw)), "raw_prediction": round(raw, 6),
        "checkpoint_status": "accepted" if ref["accepted"] else "unaccepted_candidate",
        "source_checkpoint_sha256": ref["source_sha256"],
        "checkpoint": ref["sha256"],
        "experiment": FORMAT,
        "note": "Prediction comes only from the calibrated v1.5 bilinear network, with its original frozen hidden layers.",
    }


def run_calibration_study(store, seeds=(31415,42,1001), count=20, probe_seed=20261104):
    seeds = tuple(int(s) for s in seeds)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply 2..10 distinct seeds")
    runs = [run_calibration_lab(store, seed=s, count=count, probe_seed=probe_seed)
            for s in seeds]
    summaries = {}
    for group in GROUPS:
        summaries[group] = {}
        for op in OPS:
            valid = [r for r in runs if r.get("status")=="calibrated"]
            scores = [r["ood"][group][op]["accuracy"] for r in valid]
            source_scores = [r["source_ood"][group][op]["accuracy"] for r in valid]
            summaries[group][op] = {
                "seeds_evaluated": len(valid),
                "calibrated_mean": round(statistics.mean(scores), 4) if scores else None,
                "calibrated_min": min(scores) if scores else None,
                "calibrated_max": max(scores) if scores else None,
                "frozen_v15_mean": round(statistics.mean(source_scores), 4) if source_scores else None,
            }
    return {
        "status": "evaluated" if any(r.get("status")=="calibrated" for r in runs) else "skipped",
        "experiment": FORMAT,
        "seeds": list(seeds),
        "runs": [{
            "seed": r["seed"], "status": r["status"],
            "accepted": r.get("accepted"),
            "validation_before": r.get("mean_validation_mae_before"),
            "validation_after": r.get("mean_validation_mae_after"),
            "test_accuracy": {
                op: r["test"][op]["accuracy"] for op in OPS
            } if r["status"]=="calibrated" else None,
            "source_checkpoint_sha256": r.get("source_checkpoint_sha256"),
            "calibrated_checkpoint": r.get("checkpoint"),
            "reason": r.get("reason"),
        } for r in runs],
        "ood_summary": summaries,
        "caveat": "Only the output head was refitted with supervised labels and a supplied least-squares solver. Shared OOD sets across seeds are developmental, not new independent tests.",
    }
