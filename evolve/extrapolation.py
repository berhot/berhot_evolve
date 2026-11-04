"""Read-only out-of-distribution stress tests for v1.1 magnitude checkpoints.

Train/validation data contain only operands 0..9. These audits use new
operand magnitudes and never update weights, tune hyperparameters, select
checkpoints, or promote a model. They measure whether an interpolating
numeric-input MLP can extrapolate beyond its supervised domain.
"""
from pathlib import Path
import random
import statistics

from .brain import require_torch, sha256_file
from .magnitude_lab import FORMAT, _build_model
from .numeric_lab import split_addition_pairs
from .practice import checked_answer

RANGES = {
    "one_new_operand": ((0, 9), (10, 19)),
    "both_10_to_19": ((10, 19), (10, 19)),
    "both_20_to_29": ((20, 29), (20, 29)),
}


def select_ood_pairs(group, count=20, seed=20261104):
    """Fixed, non-overlapping-with-training problems for each OOD category."""
    if group not in RANGES:
        raise ValueError(f"Unknown probe group: {group}")
    count = int(count)
    if not 1 <= count <= 50:
        raise ValueError("Use 1..50 questions per probe group")
    (lo1, hi1), (lo2, hi2) = RANGES[group]
    pairs = sorted({
        tuple(sorted((left, right)))
        for left in range(lo1, hi1 + 1)
        for right in range(lo2, hi2 + 1)
    })
    # All categories have >=55 distinct unordered pairs.
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    return pairs[:count]


def _read_checkpoint(store, model_seed):
    """Choose a registered seed checkpoint or a labeled historical local archive.

    Archived v1.1 files predate seed-keyed registrations. Their hash is
    computed but cannot be independently cross-checked against an old
    registry; the report says so rather than inventing acceptance status.
    """
    if model_seed is None:
        record = store.get("magnitude_lab_latest_candidate")
        kind = "latest_registered_candidate"
        if record is None:
            return None, "No v1.1 magnitude checkpoint has been registered"
    else:
        seed = int(model_seed)
        record = store.get(f"magnitude_lab_seed_{seed}")
        kind = "registered_seed_checkpoint"
        if record is None:
            latest = store.get("magnitude_lab_latest_candidate")
            if latest and latest.get("seed") == seed:
                record = latest
                kind = "latest_registered_candidate"
        if record is None:
            root = store.root / "models" / "magnitude_lab"
            candidates = sorted(root.glob(f"magnitude-seed{seed}-*.pt"))
            if not candidates:
                return None, f"No saved magnitude checkpoint for seed {seed}"
            path = candidates[-1]
            record = {
                "path": str(path), "sha256": sha256_file(path),
                "seed": seed, "accepted": None,
            }
            kind = "local_archive_hash_not_registry_verified"
    path = Path(record["path"]).resolve()
    allowed = (store.root / "models" / "magnitude_lab").resolve()
    if not path.is_file() or not path.is_relative_to(allowed):
        raise ValueError("Magnitude checkpoint is missing or outside model storage")
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise ValueError("Magnitude checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError("Magnitude checkpoint has an incompatible format")
    seed = int(data["seed"])
    if model_seed is not None and seed != int(model_seed):
        raise ValueError("Checkpoint seed does not match requested model seed")
    if data.get("splits") != split_addition_pairs(seed):
        raise ValueError("Saved data partitions do not match checkpoint seed")
    model = _build_model(int(data["hidden"]))
    model.load_state_dict(data["weights"])
    model.eval()
    return {
        "model": model, "record": record, "seed": seed,
        "path": str(path), "sha256": digest,
        "source": kind, "checkpoint_status": (
            "accepted" if record.get("accepted") is True
            else "unaccepted_candidate" if record.get("accepted") is False
            else "historical_acceptance_unknown"),
    }, None


def _evaluate_group(model, torch, pairs):
    left = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    right = torch.tensor([b for _, b in pairs], dtype=torch.long)
    with torch.no_grad():
        raw = (model(left, right) * 18.0).tolist()
    samples = []
    errors = []
    correct_count = 0
    within_one = 0
    for (a, b), value in zip(pairs, raw):
        answer = checked_answer(a, b, "+")
        estimate = int(round(value))
        error = float(value) - answer
        errors.append(error)
        right = estimate == answer
        correct_count += int(right)
        within_one += int(abs(estimate - answer) <= 1)
        samples.append({
            "question": f"{a} + {b}",
            "expected": answer,
            "prediction": estimate,
            "raw_prediction": round(float(value), 4),
            "correct": right,
        })
    return {
        "questions": len(pairs),
        "correct": correct_count,
        "accuracy": round(correct_count / len(pairs), 4),
        "within_one": within_one,
        "within_one_rate": round(within_one / len(pairs), 4),
        "mean_absolute_error": round(statistics.mean(abs(e) for e in errors), 4),
        "mean_signed_error": round(statistics.mean(errors), 4),
        "worst_absolute_error": round(max(abs(e) for e in errors), 4),
        "examples": samples[:8],
        "mistakes": [sample for sample in samples if not sample["correct"]][:8],
    }


def magnitude_extrapolate(store, model_seed=None, count=20, probe_seed=20261104):
    """Audit existing weights only; this function performs NO gradient steps."""
    selected, error = _read_checkpoint(store, model_seed)
    if selected is None:
        return {"status": "skipped", "reason": error, "model_seed": model_seed}
    torch, _ = require_torch()
    results = {}
    for group in RANGES:
        problems = select_ood_pairs(group, count=count, seed=probe_seed)
        results[group] = _evaluate_group(selected["model"], torch, problems)
    report = {
        "status": "evaluated",
        "evaluation": "unseen_magnitude_extrapolation_v1",
        "model_seed": selected["seed"],
        "probe_seed": int(probe_seed),
        "checkpoint_sha256": selected["sha256"],
        "checkpoint_source": selected["source"],
        "checkpoint_status": selected["checkpoint_status"],
        "weights_updated": False,
        "training_operand_range": [0, 9],
        "probe_operand_ranges": {
            name: [list(bounds[0]), list(bounds[1])]
            for name, bounds in RANGES.items()
        },
        "results": results,
        "note": "No training, promotion or fine-tuning on OOD questions. Numeric magnitude and output scaling remain supplied priors. A repeated stress set is a development diagnostic, not an untouched future final test.",
    }
    # The evaluator may write a diagnostic event, but never a model version.
    store.log("magnitude_extrapolation", "completed", {
        **report, "results": {
            name: {key: value for key, value in data.items()
                   if key not in ("examples", "mistakes")}
            for name, data in results.items()
        },
    })
    return report


def magnitude_extrapolation_study(store, model_seeds=(31415, 42, 1001),
                                  count=20, probe_seed=20261104):
    seeds = tuple(int(x) for x in model_seeds)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Specify 2..10 distinct model seeds")
    trials = [magnitude_extrapolate(store, model_seed=seed, count=count,
                                   probe_seed=probe_seed) for seed in seeds]
    aggregates = {}
    for group in RANGES:
        valid = [trial["results"][group] for trial in trials
                 if trial["status"] == "evaluated"]
        values = [result["accuracy"] for result in valid]
        aggregates[group] = {
            "models_evaluated": len(valid),
            "mean_exact_accuracy": round(statistics.mean(values), 4) if values else None,
            "min_exact_accuracy": min(values) if values else None,
            "max_exact_accuracy": max(values) if values else None,
            "total_correct": sum(v["correct"] for v in valid),
            "total_questions": sum(v["questions"] for v in valid),
        }
    return {
        "status": "evaluated" if any(t["status"] == "evaluated" for t in trials)
                  else "skipped",
        "evaluation": "unseen_magnitude_extrapolation_multiseed_v1",
        "seeds_requested": list(seeds),
        "probe_seed": int(probe_seed),
        "count_per_group_per_model": int(count),
        "models": [
            {
                "seed": t["model_seed"],
                "status": t["status"],
                "checkpoint_sha256": t.get("checkpoint_sha256"),
                "checkpoint_source": t.get("checkpoint_source"),
                "error": t.get("reason"),
                "group_accuracies": {
                    group: t["results"][group]["accuracy"] for group in RANGES
                } if t["status"] == "evaluated" else {},
            }
            for t in trials
        ],
        "aggregate": aggregates,
        "caveat": "The same OOD operand sets are reused across model seeds; aggregate counts are not independent problems. No model is selected or trained using these scores.",
    }
