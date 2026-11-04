"""v1.6: read-only v1.5 coefficient audit and supervised polynomial-basis reference.

Important: this experiment PROVIDES the exact functional basis [1, x, y, xy],
where x=a/9 and y=b/9. It is a strong inductive bias that spans addition,
subtraction and multiplication. A linear neural head is fitted with DOUBLE
PRECISION least squares from train examples ONLY (no gradient GPU training).
Targets come from Python arithmetic; its forward/inference does not call the
verifier. This is not discovered mathematics or the general EVOLVE model.
"""
from datetime import datetime, timezone
from pathlib import Path
import random
import statistics

from .brain import require_torch, sha256_file
from .interaction_lab import (
    _load_checkpoint as load_v15, _ood_pairs as v15_ood_pairs, OOD_GROUPS,
)
from .multiop_lab import OPS, SCALES, _examples
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

FORMAT = "supervised_polynomial_basis_least_squares_v1"
BASIS = ("constant", "left", "right", "left_times_right")
GROUPS = (*OOD_GROUPS, "both_100_to_199", "both_500_to_999")
IDEAL = {
    "+": (0.0, 0.5, 0.5, 0.0),
    "-": (0.0, 1.0, -1.0, 0.0),
    "*": (0.0, 0.0, 0.0, 1.0),
}

def _reference_value(a, b, operation):
    """Bounded evaluation oracle; never called by the neural model's forward.

    Preserve the production practice verifier's deliberately small 0..99
    domain. Larger synthetic OOD probes use an isolated, independently
    inverse-checked scoring path, with a strict 0..999 limit.
    """
    a, b = int(a), int(b)
    if not (0 <= a <= 999 and 0 <= b <= 999):
        raise ValueError("Evaluation oracle only supports operands 0..999")
    if a <= 99 and b <= 99:
        return checked_answer(a, b, operation)
    if operation == "+":
        answer = a + b
        valid = (answer - a == b and answer - b == a)
    elif operation == "-":
        answer = a - b
        valid = (answer + b == a and a - answer == b)
    elif operation == "*":
        answer = a * b
        # Independent repeated-addition check is bounded to 999 terms.
        valid = (answer == sum([a] * b)
                 and (a == 0 or answer // a == b)
                 and (b == 0 or answer // b == a))
    else:
        raise ValueError("Evaluation oracle supports only +, - and *")
    if not valid:
        raise ArithmeticError("OOD arithmetic inverse check failed")
    return answer



def _build_model():
    torch, nn = require_torch()

    class PolynomialHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, len(OPS), bias=False, dtype=torch.float64)

        def forward(self, a, b, operations):
            x = a.to(torch.float64) / 9.0
            y = b.to(torch.float64) / 9.0
            features = torch.stack((torch.ones_like(x), x, y, x * y), dim=-1)
            return self.head(features).gather(1, operations.unsqueeze(1)).squeeze(1)

    return PolynomialHead()


def _tensor_inputs(examples, torch):
    items = list(examples)
    a = torch.tensor([x for x, _, _ in items], dtype=torch.long)
    b = torch.tensor([y for _, y, _ in items], dtype=torch.long)
    ops = torch.tensor([OPS.index(op) for _, _, op in items], dtype=torch.long)
    return a, b, ops, items


def _score(model, examples, sample_limit=3):
    torch, _ = require_torch()
    a, b, operations, items = _tensor_inputs(examples, torch)
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(a, b, operations).detach().cpu().tolist()
    output = {}
    for index, operator in enumerate(OPS):
        rows = [i for i, item in enumerate(items) if item[2] == operator]
        if not rows:
            raise ValueError("Each scored group requires all three operations")
        samples = []
        errors = []
        hits = 0
        for i in rows:
            left, right, _ = items[i]
            truth = _reference_value(left, right, operator)
            raw = float(predicted_normalized[i]) * SCALES[index]
            guess = int(round(raw))
            errors.append(raw - truth)
            correct = guess == truth
            hits += int(correct)
            if len(samples) < sample_limit:
                samples.append({
                    "question": f"{left} {operator} {right}",
                    "expected": truth,
                    "prediction": guess, "raw_prediction": round(raw, 6),
                    "correct": correct,
                })
        output[operator] = {
            "questions": len(rows), "correct": hits,
            "accuracy": round(hits / len(rows), 4),
            "mean_absolute_error": round(statistics.mean(abs(e) for e in errors), 8),
            "mean_signed_error": round(statistics.mean(errors), 8),
            "max_absolute_error": round(max(abs(e) for e in errors), 8),
            "examples": samples,
        }
    return output


def _fit_from_train_pairs(pairs):
    torch, _ = require_torch()
    model = _build_model()
    batch = list(_examples(pairs))
    # One common numerical feature matrix across three operations.
    for idx, operator in enumerate(OPS):
        training = [(a, b) for a, b, op in batch if op == operator]
        a = torch.tensor([a for a, _ in training], dtype=torch.float64)
        b = torch.tensor([b for _, b in training], dtype=torch.float64)
        x, y = a / 9.0, b / 9.0
        matrix = torch.stack((torch.ones_like(x), x, y, x * y), dim=-1)
        targets = torch.tensor(
            [checked_answer(left, right, operator) / SCALES[idx]
             for left, right in training], dtype=torch.float64)
        solution = torch.linalg.lstsq(matrix, targets).solution
        if int(torch.linalg.matrix_rank(matrix).item()) != 4:
            raise ValueError(f"Insufficient numerical rank in train split for {operator}")
        with torch.no_grad():
            model.head.weight[idx].copy_(solution)
    return model


def _ood_pairs(group, count=20, seed=20261104):
    count = int(count)
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    if group in OOD_GROUPS:
        return v15_ood_pairs(group, count, seed)
    if group == "both_100_to_199":
        low, high, salt = 100, 199, 31337
    elif group == "both_500_to_999":
        low, high, salt = 500, 999, 99331
    else:
        raise ValueError("Unknown OOD group")
    rng = random.Random(int(seed) + salt)
    # For 500..999, avoid constructing all 125250 pairs just to pick 20.
    selected = set()
    while len(selected) < count:
        x = rng.randint(low, high)
        y = rng.randint(low, high)
        selected.add(tuple(sorted((x, y))))
    return sorted(selected)


def _coef_dict(values):
    return {name: round(float(value), 10) for name, value in zip(BASIS, values)}


def _effective_v15_coefficients(model):
    """Algebraically expand the existing bilinear hidden layer's exact function."""
    torch, _ = require_torch()
    with torch.no_grad():
        lw = model.left.weight[:, 0].to(torch.float64)
        lb = model.left.bias.to(torch.float64)
        rw = model.right.weight[:, 0].to(torch.float64)
        rb = model.right.bias.to(torch.float64)
        hw = model.head.weight.to(torch.float64)
        hb = model.head.bias.to(torch.float64)
        result = {}
        for idx, op in enumerate(OPS):
            row = hw[idx]
            hidden = row[2:]
            coefficients = [
                hb[idx] + (hidden * lb * rb).sum(),
                row[0] + (hidden * lw * rb).sum(),
                row[1] + (hidden * lb * rw).sum(),
                (hidden * lw * rw).sum(),
            ]
            result[op] = [float(x.item()) for x in coefficients]
        return result


def audit_interaction_coefficients(store, seed=31415, a=27, b=28):
    """Use only v1.5 weights; never fit, save, promote or mutate any checkpoint."""
    a, b, seed = int(a), int(b), int(seed)
    if not 0 <= a <= 999 or not 0 <= b <= 999:
        raise ValueError("Audit operands must be 0..999")
    found = load_v15(store, seed=seed)
    if found is None:
        return {"status": "skipped", "reason": f"No v1.5 checkpoint for seed {seed}"}
    model, ref = found
    values = _effective_v15_coefficients(model)
    x, y = a / 9.0, b / 9.0
    terms = [1.0, x, y, x*y]
    ops = {}
    for idx, op in enumerate(OPS):
        coeffs = values[op]
        ideal = IDEAL[op]
        raw = sum(t*c for t, c in zip(terms, coeffs)) * SCALES[idx]
        expected = _reference_value(a, b, op)
        ops[op] = {
            "coefficients": _coef_dict(coeffs),
            "reference_coefficients": _coef_dict(ideal),
            "coefficient_errors": _coef_dict(
                [coef - truth for coef, truth in zip(coeffs, ideal)]),
            "question": f"{a} {op} {b}",
            "expected": expected,
            "model_prediction": int(round(raw)),
            "raw_prediction": round(raw, 6),
            "signed_error": round(raw - expected, 6),
            "error_contributions": _coef_dict(
                [(coef - truth) * term * SCALES[idx]
                 for coef, truth, term in zip(coeffs, ideal, terms)]),
        }
    return {
        "status": "audited",
        "evaluation": "frozen_bilinear_coefficient_audit_v1",
        "seed": seed, "checkpoint": ref["sha256"],
        "accepted": ref["accepted"],
        "weights_updated": False,
        "input_normalization": "x=a/9, y=b/9",
        "effective_basis": list(BASIS),
        "operations": ops,
        "note": "Reference coefficient patterns are evaluation knowledge, not new training labels. Small coefficient errors grow with x and xy outside 0..9.",
    }


def _registered_model(store, seed=None):
    ref = store.get(f"basis_lab_seed_{int(seed)}") if seed is not None else store.get("basis_lab_latest")
    if ref is None:
        return None
    path = Path(ref["path"]).resolve()
    root = (store.root / "models" / "basis_lab").resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("Basis checkpoint missing or outside its model directory")
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("Basis checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("format") != FORMAT or data.get("seed") != int(ref["seed"])
        or data.get("split") != split_addition_pairs(int(ref["seed"]))
        or tuple(data.get("scales", ())) != SCALES
        or tuple(data.get("operations", ())) != OPS):
        raise ValueError("Incompatible basis checkpoint")
    model = _build_model()
    model.load_state_dict(data["weights"])
    model.eval()
    return model, ref


def run_basis_lab(store, seed=31415, count=20, probe_seed=20261104):
    """Fit once on 0..9 training data; selection never accesses test/OOD."""
    seed = int(seed)
    count = int(count)
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    split = split_addition_pairs(seed)
    # Fit ONLY the 35 train groups; the 10 validation groups are not fit.
    model = _fit_from_train_pairs(split["train"])
    train = _score(model, _examples(split["train"]))
    validation = _score(model, _examples(split["validation"]))
    # A single deterministic LS fit; no hyperparameters are selected on OOD.
    accepted = all(
        validation[op]["mean_absolute_error"] < 0.5 for op in OPS)
    # Final test & all OOD probes are scored only after fitting/acceptance.
    heldout = _score(model, _examples(split["test"]))
    ood = {
        group: _score(model, _examples(_ood_pairs(group, count, probe_seed)))
        for group in GROUPS
    }
    current = load_v15(store, seed=seed)
    baseline = {
        "status": "no_matching_seed_v15_checkpoint"
    }
    if current is not None:
        old_model, old_ref = current
        baseline = {
            "status": "comparable_read_only",
            "checkpoint_sha256": old_ref["sha256"],
            "accepted": old_ref["accepted"],
            "ood": {
                group: {op: {k: v for k, v in scores.items() if k != "examples"}
                        for op, scores in _score(old_model,
                            _examples(_ood_pairs(group, count, probe_seed))).items()}
                for group in GROUPS
            },
        }
    folder = store.root / "models" / "basis_lab"
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / (
        f"basis-seed{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.pt")
    torch, _ = require_torch()
    torch.save({
        "format": FORMAT, "seed": seed, "split": split,
        "operations": OPS, "scales": SCALES,
        "weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "solver": "torch.linalg.lstsq float64",
        "validation": validation,
    }, filename)
    sha = sha256_file(filename)
    record = {
        "path": str(filename), "sha256": sha,
        "format": FORMAT, "seed": seed, "accepted": accepted,
    }
    store.put(f"basis_lab_seed_{seed}", record)
    store.put("basis_lab_latest_candidate", record)
    if accepted:
        store.put("basis_lab_latest", record)
    report = {
        "status": "fitted", "experiment": FORMAT,
        "fitting_method": "double_precision_supervised_least_squares_not_gradient_training",
        "architecture": "one_shared_neural_linear_head_on_supplied_polynomial_features",
        "explicit_prior": "The basis includes x, y and x*y; this is a strong supplied mathematical assumption.",
        "seed": seed, "train_groups": 35, "validation_groups": 10,
        "test_groups": 10, "training_operand_range": [0, 9],
        "train": train, "validation": validation, "in_range_test": heldout,
        "learned_coefficients": {
            op: _coef_dict(model.head.weight[idx].detach().cpu().tolist())
            for idx, op in enumerate(OPS)
        },
        "ood": ood,
        "frozen_v15_baseline": baseline,
        "accepted": accepted, "checkpoint": str(filename), "sha256": sha,
        "note": "This analytic reference identifies optimization/precision error in v1.5. It is not emergent reasoning, and larger-number scores are developmental diagnostics.",
    }
    store.log("basis_lab", "completed", {
        **{key: value for key, value in report.items() if key != "ood"},
        "ood": {
            group: {op: {k: v for k, v in stats.items() if k != "examples"}
                    for op, stats in ops.items()} for group, ops in ood.items()
        },
    })
    return report


def predict_basis_lab(store, a, b, operator, seed=None):
    if operator not in OPS:
        raise ValueError("Operator must be +, - or *")
    a, b = int(a), int(b)
    if not (0 <= a <= 999 and 0 <= b <= 999):
        raise ValueError("Supported operands 0..999")
    found = _registered_model(store, seed=seed)
    if found is None:
        return {"status": "skipped", "reason": "No matching fitted basis checkpoint"}
    model, ref = found
    torch, _ = require_torch()
    with torch.no_grad():
        value = model(torch.tensor([a]), torch.tensor([b]),
                      torch.tensor([OPS.index(operator)]))
    raw = float(value.item()) * SCALES[OPS.index(operator)]
    return {
        "status": "predicted", "question": f"{a} {operator} {b}",
        "model_answer": int(round(raw)), "raw_prediction": round(raw, 6),
        "checkpoint_status": "accepted" if ref["accepted"] else "unaccepted_candidate",
        "checkpoint": ref["sha256"], "experiment": FORMAT,
        "note": "Prediction uses only fitted floating-point polynomial coefficients; not a calculator. The basis explicitly includes multiplication.",
    }


def run_basis_study(store, seeds=(31415, 42, 1001), count=20, probe_seed=20261104):
    seeds = tuple(int(s) for s in seeds)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Select 2..10 distinct seeds")
    runs = [run_basis_lab(store, seed=seed, count=count,
                          probe_seed=probe_seed) for seed in seeds]
    summary = {}
    for group in GROUPS:
        summary[group] = {}
        for op in OPS:
            values = [run["ood"][group][op]["accuracy"] for run in runs]
            summary[group][op] = {
                "mean": round(statistics.mean(values), 4),
                "min": min(values), "max": max(values),
            }
    return {
        "status": "evaluated", "experiment": FORMAT,
        "seeds": list(seeds), "results": [{
            "seed": run["seed"], "accepted": run["accepted"],
            "test_accuracy": {
                op: run["in_range_test"][op]["accuracy"] for op in OPS},
            "ood_accuracy": {
                group: {op: run["ood"][group][op]["accuracy"] for op in OPS}
                for group in GROUPS
            },
            "v15_baseline": {
                group: {op: run["frozen_v15_baseline"]["ood"][group][op]["accuracy"]
                        for op in OPS}
                for group in GROUPS
            } if run["frozen_v15_baseline"]["status"] == "comparable_read_only" else None,
            "checkpoint": run["checkpoint"],
        } for run in runs], "ood_summary": summary,
        "caveat": "Exact supervised fit on a supplied polynomial basis; seeds use overlapping problems and OOD probes. No model selection by test accuracy.",
    }
