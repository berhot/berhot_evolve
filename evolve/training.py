"""Candidate training, independent heldout loss, comparison and safe promotion."""
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .brain import build_model, get_device, load_checkpoint, require_torch, sha256_file


def corpus_bytes(store, split, verified_only=False):
    rows = store.documents(split, verified_only=verified_only)
    return b"\n".join(row["text"].encode("utf-8") for row in rows)


def random_batch(stream, block, batch_size, torch, device, rng):
    max_start = len(stream) - block - 1
    if max_start < 0:
        raise ValueError(f"Not enough {len(stream)} bytes to train/evaluate block_size {block}")
    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    inputs = torch.tensor([list(stream[k:k+block]) for k in starts], dtype=torch.long, device=device)
    targets = torch.tensor([list(stream[k+1:k+block+1]) for k in starts], dtype=torch.long, device=device)
    return inputs, targets


def evaluate_loss(model, val_bytes, cfg, device, seed=77):
    torch, _ = require_torch()
    model.eval()
    rng = random.Random(seed)
    block = int(cfg["block_size"])
    batch_size = 4
    losses = []
    with torch.no_grad():
        for _ in range(int(cfg.get("eval_batches", 6))):
            x, y = random_batch(val_bytes, block, batch_size, torch, device, rng)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def train_candidate(store, settings, force=False, verified_only=False):
    torch, _ = require_torch()
    documents = store.documents(verified_only=verified_only)
    threshold = int(settings["minimum_training_documents"])
    if len(documents) < threshold:
        return {"status": "skipped", "reason": f"Need >= {threshold} documents; have {len(documents)}"}
    count_key = "last_train_verified_document_count" if verified_only else "last_train_document_count"
    last_count = int(store.get(count_key, 0))
    added = len(documents) - last_count
    if not force and added < int(settings["minimum_new_documents_for_training"]):
        return {"status": "skipped", "reason": f"Only {added} new documents since last training"}
    cfg = dict(settings["model"])
    train_cfg = settings["train"]
    device = get_device(train_cfg.get("device", "auto"))
    block = int(cfg["block_size"])
    train_bytes = corpus_bytes(store, "train", verified_only=verified_only)
    val_bytes = corpus_bytes(store, "val", verified_only=verified_only)
    if min(len(train_bytes), len(val_bytes)) < block + 2:
        return {"status": "skipped", "reason": "Train or heldout document text is too short for configured block_size"}
    torch.manual_seed(int(train_cfg["seed"]))
    rng = random.Random(int(train_cfg["seed"]))
    active = store.latest_promoted()
    baseline = None
    if active:
        active_model, old = load_checkpoint(active["path"])
        if old["model_config"] != cfg:
            return {"status": "skipped", "reason": "Model config changed; use a new experiment or reset registry"}
        model = active_model.to(device)
        baseline = evaluate_loss(model, val_bytes, train_cfg | {"block_size":block}, device)
    else:
        model = build_model(cfg).to(device)
    pre_train_loss = evaluate_loss(model, train_bytes, train_cfg | {"block_size":block}, device)
    pre_val_loss = evaluate_loss(model, val_bytes, train_cfg | {"block_size":block}, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    steps = int(train_cfg["steps"])
    start = time.monotonic()
    actual_steps = 0
    for step in range(steps):
        x, y = random_batch(train_bytes, block, int(train_cfg["batch_size"]), torch, device, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_steps += 1
        if time.monotonic() - start > float(train_cfg["max_seconds"]):
            break
    val_loss = evaluate_loss(model, val_bytes, train_cfg | {"block_size":block}, device)
    post_train_loss = evaluate_loss(model, train_bytes, train_cfg | {"block_size":block}, device)
    promoted = math.isfinite(val_loss) and (
        baseline is None or val_loss + float(train_cfg["min_improvement"]) < baseline)
    output = store.root / "models"
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output / f"candidate-{timestamp}.pt"
    cpu_weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"format_version":1, "weights":cpu_weights, "model_config":cfg,
                "train_documents":len(documents), "val_loss":val_loss,
                "training_steps":actual_steps, "seed":int(train_cfg["seed"]),
                "verified_only":bool(verified_only), "pre_train_loss":pre_train_loss,
                "post_train_loss":post_train_loss, "pre_val_loss":pre_val_loss}, path)
    digest = sha256_file(path)
    store.insert_model(path, digest, len(documents), val_loss, baseline, promoted)
    store.put(count_key, len(documents))
    result = {"status": "trained", "candidate_path":str(path), "device":device,
              "dataset_scope": "verified_only" if verified_only else "eligible_corpus",
              "training_documents": len(documents),
              "pre_train_loss": round(pre_train_loss, 4),
              "post_train_loss": round(post_train_loss, 4),
              "pre_val_loss": round(pre_val_loss, 4),
              "actual_steps": actual_steps, "val_loss":round(val_loss, 4),
              "baseline_val_loss":round(baseline, 4) if baseline is not None else None,
              "promoted":promoted, "sha256":digest}
    store.log("training", "completed", result)
    return result
