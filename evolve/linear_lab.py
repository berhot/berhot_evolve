"""v1.3: train a non-saturating numerical model on 0..9 and audit OOD.

A 2-input, 1-output trainable linear neural layer receives normalized numeric
magnitudes. There is no hard-coded addition operator in its forward pass, but
LINEARITY is an extremely strong architectural prior for addition. This
experiment isolates the cause of tanh MLP's out-of-range saturation; it is
not general reasoning, autonomous skill discovery, or web-based learning.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import random
import statistics
import time

from .brain import get_device, require_torch, sha256_file
from .numeric_lab import split_addition_pairs
from .practice import checked_answer
from .extrapolation import RANGES, select_ood_pairs, _evaluate_group, _read_checkpoint

FORMAT = "linear_magnitude_addition_v1"


def _build_model():
    torch, nn = require_torch()

    class LearnedLinearMagnitude(nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = nn.Linear(2, 1)

        def forward(self, a, b):
            lo = torch.minimum(a, b).float() / 9.0
            hi = torch.maximum(a, b).float() / 9.0
            # Learn a linear combination of the two input magnitudes.
            # Neither the addition operation nor its coefficients are
            # supplied in the forward pass.
            return self.mapping(torch.stack((lo, hi), dim=-1)).squeeze(-1)

    return LearnedLinearMagnitude()


def _data(pairs, torch, device):
    a = torch.tensor([x for x, _ in pairs], dtype=torch.long, device=device)
    b = torch.tensor([y for _, y in pairs], dtype=torch.long, device=device)
    targets = torch.tensor(
        [checked_answer(x, y, "+") / 18.0 for x, y in pairs],
        device=device, dtype=torch.float32)
    return a, b, targets


def _metrics(model, batch, torch):
    a, b, expected = batch
    model.eval()
    with torch.no_grad():
        raw = model(a, b) * 18.0
        true = expected * 18.0
        mse = float(((raw - true) ** 2).mean().item())
        correct = int((raw.round() == true.round()).sum().item())
    return {"total": len(a), "correct": correct,
            "accuracy": round(correct / len(a), 4),
            "mse": round(mse, 8)}


def _read_linear(store, seed=None):
    registry_key = f"linear_lab_seed_{int(seed)}" if seed is not None else "linear_lab_latest"
    record = store.get(registry_key)
    if record is None:
        return None
    path = Path(record["path"]).resolve()
    root = (store.root / "models" / "linear_lab").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Linear checkpoint missing or outside model directory")
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise ValueError("Linear checkpoint SHA256 mismatch")
    torch, _ = require_torch()
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError("Unsupported linear checkpoint format")
    model_seed = int(data["seed"])
    if seed is not None and model_seed != int(seed):
        raise ValueError("Linear checkpoint seed mismatch")
    if data["split"] != split_addition_pairs(model_seed):
        raise ValueError("Linear checkpoint split mismatch")
    model = _build_model()
    model.load_state_dict(data["weights"])
    model.eval()
    return model, record, model_seed


def _baseline_ood(store, seed, count, probe_seed):
    """Use pre-existing magnitude checkpoint; never mutate its weights."""
    saved, error = _read_checkpoint(store, seed)
    if saved is None:
        return {"status": "not_available", "reason": error}
    torch, _ = require_torch()
    return {
        "status": "comparable_frozen_checkpoint",
        "checkpoint_sha256": saved["sha256"],
        "checkpoint_source": saved["source"],
        "checkpoint_status": saved["checkpoint_status"],
        "ood": {
            group: _summary(_evaluate_group(saved["model"], torch,
                                            select_ood_pairs(group, count, probe_seed)))
            for group in RANGES
        },
    }


def _summary(result):
    return {key: value for key, value in result.items()
            if key not in ("examples", "mistakes")}


def run_linear_lab(store, settings, seed=31415, steps=500, count=20,
                   probe_seed=20261104):
    steps = int(steps)
    count = int(count)
    if not 1 <= steps <= 3000:
        raise ValueError("Steps must be 1..3000")
    if not 1 <= count <= 50:
        raise ValueError("Count must be 1..50")
    seed = int(seed)
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(seed)
    random.seed(seed)
    split = split_addition_pairs(seed)
    datasets = {name: _data(pairs, torch, device) for name, pairs in split.items()}
    model = _build_model().to(device)
    initial = {key: value.detach().cpu().clone()
               for key, value in model.state_dict().items()}
    before = {name: _metrics(model, datasets[name], torch)
              for name in ("train", "validation")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0)
    train_a, train_b, targets = datasets["train"]
    best_loss = float("inf")
    best_step = 0
    best_state = None
    no_improve = 0
    actual_steps = 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        prediction = model(train_a, train_b)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        actual_steps = step
        if step % 10 == 0 or step == steps:
            validation = _metrics(model, datasets["validation"], torch)["mse"]
            if math.isfinite(validation) and validation + 1e-9 < best_loss:
                best_loss = validation
                best_step = step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 25:
                    break
        if time.monotonic() - start > float(settings["train"].get("max_seconds", 600)):
            break
    if best_state is None:
        return {"status": "rejected", "reason": "No finite validation checkpoint"}
    model.load_state_dict(best_state)
    after = {name: _metrics(model, datasets[name], torch)
             for name in ("train", "validation")}
    accepted = (after["validation"]["mse"] + 1e-5 <
                before["validation"]["mse"])
    initial_model = _build_model().to(device)
    initial_model.load_state_dict(initial)
    # Test and OOD labels are consulted ONLY AFTER validation selected weights.
    before["test"] = _metrics(initial_model, datasets["test"], torch)
    after["test"] = _metrics(model, datasets["test"], torch)
    cpu_model = _build_model()
    cpu_model.load_state_dict({k: v.cpu() for k, v in best_state.items()})
    cpu_model.eval()
    ood = {}
    for group in RANGES:
        pairs = select_ood_pairs(group, count=count, seed=probe_seed)
        ood[group] = _evaluate_group(cpu_model, torch, pairs)
    output = store.root / "models" / "linear_lab"
    output.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output / f"linear-seed{seed}-{name}.pt"
    torch.save({
        "format": FORMAT, "weights": best_state,
        "seed": seed, "split": split, "best_step": best_step,
        "validation_mse": after["validation"]["mse"],
        "prior": "normalized_continuous_operands_plus_linear_function_class",
    }, path)
    digest = sha256_file(path)
    record = {"path": str(path), "sha256": digest,
              "seed": seed, "accepted": accepted,
              "format": FORMAT}
    store.put(f"linear_lab_seed_{seed}", record)
    store.put("linear_lab_latest_candidate", record)
    if accepted:
        store.put("linear_lab_latest", record)
    weights = cpu_model.mapping.weight.detach().flatten().tolist()
    bias = float(cpu_model.mapping.bias.detach().item())
    baseline = _baseline_ood(store, seed, count, probe_seed)
    result = {
        "status": "trained", "experiment": FORMAT,
        "architecture": "learned_affine_map_on_normalized_numeric_operands",
        "explicit_prior": "Linear function class and known continuous magnitudes; no hard-coded arithmetic coefficients",
        "task": "supervised_single_digit_addition_then_frozen_10_to_29_extrapolation",
        "seed": seed, "device": device,
        "train_groups": 35, "validation_groups": 10, "test_groups": 10,
        "before": before, "after": after,
        "learned_coefficients": [round(x, 6) for x in weights],
        "learned_bias": round(bias, 6),
        "best_step": best_step, "actual_steps": actual_steps,
        "accepted": accepted, "checkpoint": str(path), "sha256": digest,
        "probe_seed": int(probe_seed),
        "ood": ood,
        "frozen_tanh_mlp_baseline": baseline,
        "note": "A linear model can extrapolate the linear operation because linearity is provided by architecture. This is not general mathematical reasoning.",
    }
    store.log("linear_lab", "completed", {
        **result,
        "ood": {name: _summary(value) for name, value in ood.items()},
    })
    return result


def run_linear_study(store, settings, seeds=(31415, 42, 1001),
                     steps=500, count=20, probe_seed=20261104):
    seeds = tuple(int(s) for s in seeds)
    if not 2 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply 2..10 distinct seeds")
    results = [run_linear_lab(store, settings, seed=seed, steps=steps,
                              count=count, probe_seed=probe_seed)
               for seed in seeds]
    summary = {}
    for group in RANGES:
        accuracies = [r["ood"][group]["accuracy"] for r in results
                      if r.get("status") == "trained"]
        summary[group] = {
            "mean": round(statistics.mean(accuracies), 4) if accuracies else None,
            "min": min(accuracies) if accuracies else None,
            "max": max(accuracies) if accuracies else None,
        }
    return {
        "status": "evaluated",
        "experiment": FORMAT,
        "seeds": list(seeds),
        "results": [{
            "seed": r["seed"], "accepted": r["accepted"],
            "in_range_test_accuracy": r["after"]["test"]["accuracy"],
            "ood_accuracy": {group: r["ood"][group]["accuracy"] for group in RANGES},
            "baseline_ood_accuracy": ({
                group: r["frozen_tanh_mlp_baseline"]["ood"][group]["accuracy"]
                for group in RANGES
            } if r["frozen_tanh_mlp_baseline"]["status"] == "comparable_frozen_checkpoint"
              else None),
            "checkpoint": r["checkpoint"],
        } for r in results if r.get("status") == "trained"],
        "ood_summary": summary,
        "caveat": "The same probe pairs are used for all model seeds; seed results are not independent questions. This study tests an explicit linear prior, not universal arithmetic reasoning.",
    }


def predict_linear_lab(store, a, b, seed=None, candidate=False):
    a, b = int(a), int(b)
    if not (0 <= a <= 10000 and 0 <= b <= 10000):
        raise ValueError("Only nonnegative integers 0..10000 are supported")
    if seed is None and candidate:
        ref = store.get("linear_lab_latest_candidate")
        if ref is None:
            return {"status": "skipped", "reason": "No linear candidate"}
        seed = ref["seed"]
    result = _read_linear(store, seed=seed)
    if result is None:
        return {"status": "skipped", "reason": "No matching linear checkpoint"}
    model, record, _ = result
    torch, _ = require_torch()
    with torch.no_grad():
        value = float((model(torch.tensor([a]), torch.tensor([b])) * 18.0).item())
    return {
        "status": "predicted",
        "question": f"{a} + {b}", "model_answer": int(round(value)),
        "raw_prediction": round(value, 4),
        "model_status": "accepted" if record["accepted"] else "unaccepted_candidate",
        "checkpoint": record["sha256"],
        "architecture": FORMAT,
        "note": "Result from a learned linear function; not proof of general reasoning or carry handling.",
    }
