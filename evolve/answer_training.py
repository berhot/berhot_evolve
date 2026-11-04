"""Answer-focused arithmetic tuning using independently checked QA examples.

Unlike generic byte-stream pretraining, only the numeric answer (and terminator)
contributes to the loss. This remains a tiny neural sequence model and does not
guarantee arithmetic generalization. Held-out examples never enter gradients.

This uses a separate checkpoint track and DOES NOT overwrite the general model
registry, so future generic pretraining remains comparable.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import random
import re
import time

from .brain import build_model, get_device, load_checkpoint, require_torch, sha256_file
from .practice import checked_answer

MATH_FORMAT = re.compile(
    r"\AQuestion: (\d{1,2}) ([+*\-]) (\d{1,2}) = \?\nAnswer: (-?\d+)\n\Z"
)
OBJECTIVE = "arithmetic_answer_only_v1"


def _parse(row, block):
    match = MATH_FORMAT.fullmatch(row["text"])
    if match is None:
        return None
    a, operator, b, numeric = match.groups()
    if checked_answer(int(a), int(b), operator) != int(numeric):
        return None
    prompt = f"Question: {a} {operator} {b} = ?\nAnswer:"
    full = prompt + f" {numeric}\n"
    raw = full.encode("utf-8")
    if len(raw) > block + 1:
        return None
    # x[t] predicts raw[t+1]. Only the suffix AFTER 'Answer:' is scored.
    start = len(prompt.encode("utf-8")) - 1
    x = list(raw[:-1]) + [0] * (block - len(raw) + 1)
    y = [-100] * block
    for i in range(start, len(raw) - 1):
        y[i] = raw[i + 1]
    return x, y


def _rows(store, split, skill):
    rows = store.db.execute(
        "SELECT source_id,text FROM documents WHERE split=? "
        "AND topic='verified:arithmetic' AND license='SELF-GENERATED-VERIFIED' "
        "ORDER BY id", (split,)
    ).fetchall()
    if skill == "single-digit-add":
        return [row for row in rows if row["source_id"].startswith(
            "verified:arithmetic:single-digit-add:")]
    if skill == "mixed":
        return [row for row in rows if not row["source_id"].startswith(
            "verified:arithmetic:single-digit-add:")]
    raise ValueError("Skill must be 'mixed' or 'single-digit-add'")


def _loss(model, batch, torch, device):
    x = torch.tensor([x for x, _ in batch], dtype=torch.long, device=device)
    y = torch.tensor([y for _, y in batch], dtype=torch.long, device=device)
    logits = model(x)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100
    )


def _validation_loss(model, valid, torch, device, batch_size=16):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            group = valid[start:start + batch_size]
            values.append((float(_loss(model, group, torch, device).item()), len(group)))
    return sum(v * n for v, n in values) / sum(n for _, n in values)


def train_math_answerer(store, settings, skill="mixed", steps=400, force=False):
    """Continue answer-only fine-tuning from a separate accepted math checkpoint."""
    steps = int(steps)
    if not 1 <= steps <= 3000:
        raise ValueError("Math fine-tuning steps must be 1..3000")
    cfg = dict(settings["model"])
    block = int(cfg["block_size"])
    train_rows = _rows(store, "train", skill)
    val_rows = _rows(store, "val", skill)
    if len(train_rows) < 4 or len(val_rows) < 1:
        return {
            "status": "skipped",
            "reason": "Need at least 4 training and 1 held-out arithmetic examples",
            "skill": skill, "train_examples": len(train_rows),
            "heldout_examples": len(val_rows),
        }
    train = [item for row in train_rows if (item := _parse(row, block)) is not None]
    valid = [item for row in val_rows if (item := _parse(row, block)) is not None]
    if len(train) < 4 or not valid:
        return {"status": "skipped", "reason": "No usable verified QA sequences within block_size"}
    marker = "math_last_trained_docs_" + skill
    total = len(train_rows) + len(val_rows)
    new = total - int(store.get(marker, 0))
    if not force and new < int(settings["minimum_new_documents_for_training"]):
        return {"status": "skipped", "reason": f"Only {new} new arithmetic documents", "skill":skill}
    torch, _ = require_torch()
    device = get_device(settings["train"].get("device", "auto"))
    torch.manual_seed(int(settings["train"]["seed"]))
    rng = random.Random(int(settings["train"]["seed"]) + total)
    prior = store.get("math_answerer_checkpoint_" + skill)
    baseline = None
    initialized_from = "random_weights"
    if prior:
        prior_model, meta = load_checkpoint(prior["path"])
        if meta["model_config"] != cfg or meta.get("objective") != OBJECTIVE:
            return {"status": "skipped", "reason": "Math checkpoint configuration differs from current model"}
        model = prior_model.to(device)
        baseline = _validation_loss(model, valid, torch, device)
        initialized_from = "previous_math_checkpoint"
    else:
        active = store.latest_promoted()
        if active:
            old_model, old_meta = load_checkpoint(active["path"])
            if old_meta["model_config"] == cfg:
                model = old_model.to(device)
                initialized_from = "general_checkpoint_warm_start"
            else:
                model = build_model(cfg).to(device)
        else:
            model = build_model(cfg).to(device)
    before_train = _validation_loss(model, train, torch, device)
    before_val = _validation_loss(model, valid, torch, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["train"]["learning_rate"]),
    )
    batch_size = max(1, min(64, int(settings["train"]["batch_size"])))
    max_seconds = max(1, float(settings["train"]["max_seconds"]))
    start = time.monotonic()
    actual_steps = 0
    for _ in range(steps):
        batch = [train[rng.randrange(len(train))] for _ in range(batch_size)]
        loss = _loss(model, batch, torch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_steps += 1
        if time.monotonic() - start > max_seconds:
            break
    after_train = _validation_loss(model, train, torch, device)
    after_val = _validation_loss(model, valid, torch, device)
    accepted = math.isfinite(after_val) and (
        baseline is None or after_val + float(settings["train"]["min_improvement"]) < baseline)
    directory = store.root / "models" / "math"
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = directory / f"answer-{skill}-{name}.pt"
    weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({
        "format_version": 1,
        "weights": weights, "model_config": cfg,
        "verified_only": True, "objective": OBJECTIVE, "skill": skill,
        "train_examples": len(train), "heldout_examples": len(valid),
        "pre_train_loss": before_train, "post_train_loss": after_train,
        "pre_val_loss": before_val, "val_loss": after_val,
        "training_steps": actual_steps, "seed": int(settings["train"]["seed"]),
    }, filename)
    digest = sha256_file(filename)
    if accepted:
        store.put("math_answerer_checkpoint_" + skill, {
            "path": str(filename), "sha256": digest,
            "val_loss": after_val, "objective": OBJECTIVE,
            "skill": skill, "train_examples": len(train),
            "heldout_examples": len(valid),
        })
    store.put(marker, total)
    report = {
        "status": "trained",
        "skill": skill, "objective": OBJECTIVE,
        "initialized_from": initialized_from,
        "steps": actual_steps, "device": device,
        "train_examples": len(train), "heldout_examples": len(valid),
        "pre_train_answer_loss": round(before_train, 4),
        "post_train_answer_loss": round(after_train, 4),
        "pre_val_answer_loss": round(before_val, 4),
        "post_val_answer_loss": round(after_val, 4),
        "previous_checkpoint_val_loss": round(baseline, 4) if baseline is not None else None,
        "accepted": accepted, "candidate_path": str(filename),
        "sha256": digest,
        "note": "Answer-only token loss measures training objective; arithmetic skill requires heldout exact-match evaluation.",
    }
    store.log("arithmetic_answer_finetuning", "completed", report)
    return report
