"""v1.8: head-only gradient descent on frozen saved v1.5 neural features.

Unlike v1.7, the optimization path has NO least-squares solver, polynomial
feature injection, ideal coefficient targets, OOD training or test selection.
The existing bilinear v1.5 hidden projections are copied and frozen; only
the ORIGINAL output head is trained with Adam or SGD and supervised 0..9
training examples. Compare against saved v1.7 analytical calibration
read-only, using identical heldout and OOD samples.

The multiplication of learned hidden features is a strong supplied prior,
and all labels are produced by a Python verifier, not discovered from the web.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import statistics
import time

from .brain import require_torch, sha256_file
from .interaction_lab import _load_checkpoint as load_v15
from .calibration_lab import (
    _build_model, _load_calibrated, _only_metrics,
)
from .basis_lab import (
    GROUPS, _ood_pairs, _score, _effective_v15_coefficients,
)
from .multiop_lab import OPS, SCALES, _examples
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

FORMAT = "frozen_bilinear_head_gradient_descent_v1"
OPTIMIZERS = ("adam", "sgd")


def _inputs(model, pairs):
    """Return frozen features and normalized supervised labels, all float64."""
    torch, _ = require_torch()
    examples = list(_examples(pairs))
    left = torch.tensor([a for a, _, _ in examples], dtype=torch.long)
    right = torch.tensor([b for _, b, _ in examples], dtype=torch.long)
    ops = torch.tensor([OPS.index(op) for _, _, op in examples], dtype=torch.long)
    targets = torch.tensor([
        checked_answer(a, b, op) / SCALES[OPS.index(op)]
        for a, b, op in examples
    ], dtype=torch.float64)
    with torch.no_grad():
        frozen = model.features(left, right)
    return frozen, ops, targets


def _mse(model, batch):
    torch, _ = require_torch()
    features, ops, targets = batch
    output = model.head(features).gather(1, ops.unsqueeze(1)).squeeze(1)
    return torch.nn.functional.mse_loss(output, targets)


def _reference(store, seed):
    found = _load_calibrated(store, seed=seed)
    if found is None:
        return None, {"status": "missing_saved_v17_reference"}
    model, ref = found
    return model, {
        "status": "same_seed_saved_v17_reference",
        "checkpoint_sha256": ref["sha256"],
        "source_sha256": ref["source_sha256"],
        "accepted": bool(ref["accepted"]),
    }


def _load_gradient(store, seed, optimizer):
    optimizer = str(optimizer).lower()
    if optimizer not in OPTIMIZERS:
        raise ValueError("Optimizer must be adam or sgd")
    ref = store.get(f"gradient_lab_seed_{int(seed)}_{optimizer}")
    if ref is None:
        return None
    path = Path(ref["path"]).resolve()
    folder = (store.root / "models" / "gradient_lab").resolve()
    if not path.is_file() or not path.is_relative_to(folder):
        raise ValueError("Gradient checkpoint missing or outside model storage")
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("Gradient checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    data = torch.load(path, weights_only=True, map_location="cpu")
    if (data.get("format") != FORMAT or
            data.get("seed") != int(seed) or
            data.get("optimizer") != optimizer or
            data.get("source_sha256") != ref["source_sha256"] or
            data.get("split") != split_addition_pairs(int(seed)) or
            tuple(data.get("operations", ())) != OPS or
            tuple(data.get("scales", ())) != SCALES):
        raise ValueError("Incompatible gradient checkpoint")
    model = _build_model(data["rank"])
    model.load_state_dict(data["weights"])
    model.eval()
    return model, ref


def run_gradient_lab(store, seed=31415, optimizer="adam", steps=2000,
                     learning_rate=None, count=20, probe_seed=20261104,
                     patience=16, max_seconds=120):
    seed, steps, count, patience = (int(seed), int(steps), int(count), int(patience))
    optimizer = str(optimizer).lower()
    if optimizer not in OPTIMIZERS:
        raise ValueError("Optimizer must be adam or sgd")
    if not 1 <= steps <= 10000:
        raise ValueError("Steps must be 1..10000")
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    if not 1 <= patience <= 100:
        raise ValueError("Patience must be 1..100")
    lr = float(learning_rate if learning_rate is not None else
               (0.01 if optimizer == "adam" else 0.05))
    if not math.isfinite(lr) or not 0 < lr <= 0.5:
        raise ValueError("Learning rate must be finite and 0 < lr <= 0.5")

    loaded = load_v15(store, seed=seed)
    if loaded is None:
        return {"status": "skipped", "seed": seed,
                "optimizer": optimizer, "reason": "No v1.5 checkpoint for this seed"}
    source, source_ref = loaded
    source_hash = sha256_file(source_ref["path"])
    if source_hash != source_ref["sha256"]:
        raise ValueError("Original source SHA256 mismatch")
    torch, _ = require_torch()
    model = _build_model(source.left.out_features)
    model.load_state_dict(source.state_dict())  # source float32 -> float64
    for parameter in model.left.parameters():
        parameter.requires_grad_(False)
    for parameter in model.right.parameters():
        parameter.requires_grad_(False)

    frozen_layers = {
        key: value.detach().clone() for key, value in model.state_dict().items()
        if key.startswith(("left.", "right."))
    }
    split = split_addition_pairs(seed)
    train_batch = _inputs(model, split["train"])
    validation_batch = _inputs(model, split["validation"])
    with torch.no_grad():
        initial_loss = float(_mse(model, validation_batch).item())
    before = _only_metrics(_score(model, _examples(split["validation"]), sample_limit=0))

    if optimizer == "adam":
        stepper = torch.optim.Adam(model.head.parameters(), lr=lr)
    else:
        stepper = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=0)

    best_state, best_step = None, 0
    best_loss = float("inf")
    no_improve, actual_steps = 0, 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        loss = _mse(model, train_batch)
        if not torch.isfinite(loss):
            break
        stepper.zero_grad(set_to_none=True)
        loss.backward()
        stepper.step()
        actual_steps = step
        if step % 25 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_loss = float(_mse(model, validation_batch).item())
            if math.isfinite(val_loss) and val_loss + 1e-13 < best_loss:
                best_loss, best_step, no_improve = val_loss, step, 0
                best_state = {
                    key: val.detach().cpu().clone()
                    for key, val in model.head.state_dict().items()
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if time.monotonic() - start >= float(max_seconds):
            break
    if best_state is None:
        return {"status": "rejected", "seed": seed, "optimizer": optimizer,
                "reason": "No finite validation checkpoint"}
    model.head.load_state_dict(best_state)
    model.eval()
    after = _only_metrics(_score(model, _examples(split["validation"]), sample_limit=0))
    accepted = best_loss + 1e-10 < initial_loss
    with torch.no_grad():
        final_train_loss = float(_mse(model, train_batch).item())
        best_validation_loss = float(_mse(model, validation_batch).item())
    for key, weight in frozen_layers.items():
        if not torch.equal(weight, model.state_dict()[key]):
            raise AssertionError(f"Frozen hidden projection changed: {key}")
    if sha256_file(source_ref["path"]) != source_hash:
        raise AssertionError("Original v1.5 source checkpoint modified")

    # Only NOW inspect test, OOD and saved v1.7 reference diagnostics.
    ref_model, reference = _reference(store, seed=seed)
    test = _only_metrics(_score(model, _examples(split["test"]), sample_limit=0))
    original_test = _only_metrics(_score(source, _examples(split["test"]), sample_limit=0))
    reference_test = (_only_metrics(_score(ref_model, _examples(split["test"]), sample_limit=0))
                      if ref_model is not None else None)
    ood = {}
    for group in GROUPS:
        samples = list(_examples(_ood_pairs(group, count, probe_seed)))
        ood[group] = {
            "gradient": _only_metrics(_score(model, samples, sample_limit=0)),
            "original_v15": _only_metrics(_score(source, samples, sample_limit=0)),
            "least_squares_v17": (
                _only_metrics(_score(ref_model, samples, sample_limit=0))
                if ref_model is not None else None),
        }
    coeffs = _effective_v15_coefficients(model)
    reference_coeffs = (_effective_v15_coefficients(ref_model)
                        if ref_model is not None else None)
    coeff_delta = ({
        op: {
            "max_abs_difference": round(
                max(abs(x - y) for x, y in zip(coeffs[op], reference_coeffs[op])), 10),
            "l2_distance": round(
                math.sqrt(sum((x-y)**2 for x,y in zip(coeffs[op], reference_coeffs[op]))), 10),
        } for op in OPS
    } if reference_coeffs is not None else None)

    folder = store.root / "models" / "gradient_lab"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (
        f"gradient-{optimizer}-seed{seed}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.pt")
    torch.save({
        "format": FORMAT, "seed": seed, "optimizer": optimizer,
        "learning_rate": lr, "rank": source.left.out_features,
        "source_sha256": source_hash,
        "weights": {key: val.detach().cpu().clone()
                    for key, val in model.state_dict().items()},
        "split": split, "operations": OPS, "scales": SCALES,
        "best_step": best_step, "best_validation_normalized_mse": best_loss,
    }, path)
    digest = sha256_file(path)
    record = {
        "path": str(path), "sha256": digest, "seed": seed,
        "optimizer": optimizer, "accepted": bool(accepted),
        "source_sha256": source_hash, "format": FORMAT,
    }
    store.put(f"gradient_lab_seed_{seed}_{optimizer}", record)
    store.put("gradient_lab_latest_candidate", record)
    if accepted:
        store.put("gradient_lab_latest", record)
    report = {
        "status": "trained", "experiment": FORMAT, "seed": seed,
        "optimizer": optimizer, "learning_rate": lr,
        "steps_requested": steps, "actual_steps": actual_steps,
        "best_step": best_step, "patience_checks": patience,
        "device": "cpu", "dtype": "float64",
        "method": "torch_autograd_optimizer_only_saved_v15_output_head",
        "frozen_layers_unchanged": True,
        "source_checkpoint_sha256": source_hash,
        "training_operand_range": [0, 9],
        "train_groups": 35, "validation_groups": 10, "test_groups": 10,
        "initial_validation_normalized_mse": initial_loss,
        "best_validation_normalized_mse": best_validation_loss,
        "final_training_normalized_mse": final_train_loss,
        "validation_before": before,
        "validation_after": after,
        "test": test, "source_test": original_test,
        "saved_v17_reference": reference,
        "v17_reference_test": reference_test,
        "ood": ood, "coefficient_distance_from_v17": coeff_delta,
        "accepted": accepted, "checkpoint": str(path), "sha256": digest,
        "caveat": (
            "Gradient descent updates ONLY the output head. No least-squares "
            "solution is used for optimization; v1.7 is read only after validation. "
            "Bilinear hidden interactions and numeric magnitudes remain supplied priors. "
            "Repeated OOD problems are developmental, not untouched confirmation."
        ),
    }
    store.log("gradient_lab", "completed", {
        **{key: val for key,val in report.items() if key != "ood"},
        "ood": {group: {
            name: {op: {"accuracy": stats["accuracy"],
                        "mean_absolute_error": stats["mean_absolute_error"]}
                   for op,stats in data.items()}
            if data is not None else None
            for name,data in group_results.items()}
            for group,group_results in ood.items()},
    })
    return report


def predict_gradient_lab(store, a, b, operator, seed=31415, optimizer="adam"):
    if operator not in OPS:
        raise ValueError("Operator must be +, - or *")
    a, b = int(a), int(b)
    if not (0 <= a <= 999 and 0 <= b <= 999):
        raise ValueError("Supported operands are 0..999")
    loaded = _load_gradient(store, seed, optimizer)
    if loaded is None:
        return {"status": "skipped", "reason": "No matching saved gradient candidate"}
    model, ref = loaded
    torch, _ = require_torch()
    model.eval()
    with torch.no_grad():
        val = model(torch.tensor([a]), torch.tensor([b]),
                    torch.tensor([OPS.index(operator)]))
    raw = float(val.item()) * SCALES[OPS.index(operator)]
    return {
        "status": "predicted",
        "question": f"{a} {operator} {b}",
        "model_answer": int(round(raw)),
        "raw_prediction": round(raw, 6),
        "optimizer": optimizer,
        "checkpoint_status": ("accepted" if ref["accepted"] else "unaccepted_candidate"),
        "source_checkpoint_sha256": ref["source_sha256"],
        "checkpoint": ref["sha256"],
        "experiment": FORMAT,
        "note": "Prediction uses learned head gradients and frozen v1.5 hidden layers; no arithmetic verifier is called.",
    }


def run_gradient_study(store, seeds=(31415, 42, 1001),
                       optimizers=("adam", "sgd"), steps=2000,
                       count=20, probe_seed=20261104, patience=16,
                       max_seconds=120):
    seeds = tuple(int(seed) for seed in seeds)
    optimizers = tuple(str(x).lower() for x in optimizers)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Select 2..10 distinct seeds")
    if not optimizers or len(optimizers) > 2 or len(set(optimizers)) != len(optimizers):
        raise ValueError("Select one or both distinct optimizers")
    if any(op not in OPTIMIZERS for op in optimizers):
        raise ValueError("Unknown optimizer")
    results = [run_gradient_lab(
        store, seed=seed, optimizer=optimizer, steps=steps,
        count=count, probe_seed=probe_seed, patience=patience,
        max_seconds=max_seconds)
        for seed in seeds for optimizer in optimizers]
    rows = []
    for r in results:
        if r["status"] != "trained":
            rows.append({
                "seed": r.get("seed"), "optimizer": r.get("optimizer"),
                "status": r["status"], "reason": r.get("reason")})
            continue
        rows.append({
            "seed": r["seed"], "optimizer": r["optimizer"],
            "status": r["status"], "accepted": r["accepted"],
            "best_step": r["best_step"],
            "validation_mse": r["best_validation_normalized_mse"],
            "in_range_accuracy": {op: r["test"][op]["accuracy"] for op in OPS},
            "hardest_ood_accuracy": {
                op: r["ood"]["both_500_to_999"]["gradient"][op]["accuracy"]
                for op in OPS},
            "hardest_ood_reference": ({
                op: r["ood"]["both_500_to_999"]["least_squares_v17"][op]["accuracy"]
                for op in OPS
            } if r["ood"]["both_500_to_999"]["least_squares_v17"] else None),
            "checkpoint": r["checkpoint"],
        })
    comparisons = {}
    for name in GROUPS:
        comparisons[name] = {}
        for op in OPS:
            comparisons[name][op] = {}
            for optimizer in optimizers:
                valid = [r for r in results if r["status"] == "trained"
                         and r["optimizer"] == optimizer]
                values = [r["ood"][name]["gradient"][op]["accuracy"] for r in valid]
                refs = [r["ood"][name]["least_squares_v17"][op]["accuracy"]
                        for r in valid
                        if r["ood"][name]["least_squares_v17"] is not None]
                comparisons[name][op][optimizer] = {
                    "models": len(values),
                    "mean_accuracy": round(statistics.mean(values), 4) if values else None,
                    "min_accuracy": min(values) if values else None,
                    "max_accuracy": max(values) if values else None,
                    "v17_mean_accuracy": round(statistics.mean(refs), 4) if refs else None,
                }
    return {
        "status": "evaluated" if any(r["status"] == "trained" for r in results) else "skipped",
        "experiment": FORMAT, "seeds": list(seeds), "optimizers": list(optimizers),
        "steps_requested": int(steps), "runs": rows, "ood_summary": comparisons,
        "caveat": (
            "All runs use train-only gradient descent. v1.7 least-squares is a "
            "read-only reference. Predeclared seeds and all failures are shown, "
            "without choosing a winner based on OOD scores."
        ),
    }
