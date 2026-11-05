"""Scoped, bounded byte-language-model training on manually approved Arabic text.

No research-only/UNKNOWN/noai text enters training. Test sources are distinct
publishers and never drive checkpoint acceptance. This small GRU is a language
model experiment, not an Arabic historian, fact checker or assistant.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import random

from .focus_data import get_profile, topic_relevance
from .v118_core import _approved_text

FORMAT = "berhot_focus_byte_gru_v1"
MIN_BYTES = {"train": 1600, "val": 600, "test": 600}
CONTEXT = 48
BATCH = 8


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _approved_pages(store, profile):
    rows = store.db.execute(
        "SELECT url,host,rights_status,noai FROM web_corpus_sources "
        "WHERE rights_status IN ('PUBLIC-DOMAIN','CC0-1.0') AND noai=0 "
        "ORDER BY host,url"
    ).fetchall()
    approved = []
    for row in rows:
        if row["host"] not in profile["allowed_hosts"]:
            continue
        source, text = _approved_text(store, row["url"])
        quality = topic_relevance(text, source["title"], profile)
        if quality["language_ok"] and quality["topical"]:
            approved.append({
                "url": row["url"], "host": row["host"],
                "sha256": source["sha256"],
                "rights": row["rights_status"],
                "data": text.encode("utf-8"),
            })
    return approved


def readiness(store, profile_name):
    profile = get_profile(profile_name)
    pages = _approved_pages(store, profile)
    hosts = sorted({p["host"] for p in pages})
    # Persisting the split on FIRST eligible training avoids changing
    # the role of old publishers after adding new sources.
    assignments = store.get("focus_host_split_" + profile_name) or {}
    if not assignments and len(hosts) >= 3:
        assignments = {host: ("train" if i < len(hosts) - 2
                             else "val" if i == len(hosts) - 2
                             else "test") for i, host in enumerate(hosts)}
    elif assignments:
        assignments = dict(assignments)
        for host in hosts:
            assignments.setdefault(host, "train")
    sizes = {k: sum(len(p["data"]) for p in pages
                    if assignments.get(p["host"]) == k)
             for k in ("train", "val", "test")}
    eligible = (
        len(hosts) >= 3 and all(sizes[k] >= MIN_BYTES[k]
                                for k in ("train", "val", "test"))
    )
    manifest = sorted({
        (p["url"], p["sha256"], p["rights"], assignments.get(p["host"]))
        for p in pages
    })
    digest = _sha(_canonical({
        "model_format": FORMAT, "profile": profile_name,
        "sources": manifest,
    }))
    previous = store.get("focus_train_latest_" + profile_name) or {}
    attempted = store.get("focus_train_attempted_" + profile_name) or {}
    return {
        "status": "ready" if eligible else "insufficient_approved_arabic_publishers_or_text",
        "rights_eligible_arabic_pages": len(pages),
        "rights_eligible_arabic_hosts": hosts,
        "source_hosts_by_split": {
            split: sorted(k for k, v in assignments.items() if v == split)
            for split in ("train", "val", "test")},
        "bytes_by_split": sizes,
        "minimum_bytes_by_split": dict(MIN_BYTES),
        "dataset_revision_sha256": digest,
        "new_training_data_since_last_attempt":
            eligible and attempted.get("dataset_revision_sha256") != digest,
        "last_accepted_checkpoint_sha256": previous.get("sha256"),
        "last_accepted_training_steps": previous.get("steps_total", 0),
        "trainable_model_type": "small_causal_byte_GRU_language_model",
        "training_rights_auto_approved": False,
        "readiness_is_factual_understanding": False,
        "_assignments": assignments, "_pages": pages,
    }


def _network():
    from .brain import require_torch
    torch, nn = require_torch()

    class ByteGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, 48)
            self.gru = nn.GRU(48, 96, batch_first=True)
            self.head = nn.Linear(96, 256)

        def forward(self, ids):
            sequence, _ = self.gru(self.embedding(ids))
            return self.head(sequence)

    return torch, ByteGRU()


def _window_batch(torch, docs, generator):
    xs, ys = [], []
    usable = [d for d in docs if len(d) > CONTEXT + 1]
    if not usable:
        raise ValueError("Too few source bytes for causal windows")
    for _ in range(BATCH):
        raw = usable[generator.randrange(len(usable))]
        start = generator.randrange(0, len(raw) - CONTEXT)
        seq = list(raw[start:start + CONTEXT + 1])
        xs.append(seq[:-1])
        ys.append(seq[1:])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def _measure(torch, nn_model, documents, seed):
    import torch.nn.functional as F
    generator = random.Random(seed)
    nn_model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(3):
            x, y = _window_batch(torch, documents, generator)
            logits = nn_model(x)
            losses.append(float(F.cross_entropy(
                logits.reshape(-1, 256), y.reshape(-1)).item()))
    return round(sum(losses)/len(losses), 6)


def train_if_eligible(store, profile_name, steps=60, seed=31415):
    """Train once per changed rights-reviewed dataset, never each timer tick."""
    if not 1 <= int(steps) <= 120:
        raise ValueError("steps must be 1..120")
    check = readiness(store, profile_name)
    if check["status"] != "ready":
        return {
            "status": "skipped_no_eligible_training_corpus",
            "reason": check["status"],
            "rights_eligible_arabic_pages": check["rights_eligible_arabic_pages"],
            "rights_eligible_arabic_hosts": check["rights_eligible_arabic_hosts"],
            "weights_updated": False, "steps": 0,
        }
    if not check["new_training_data_since_last_attempt"]:
        return {
            "status": "skipped_unchanged_approved_data",
            "dataset_revision_sha256": check["dataset_revision_sha256"],
            "weights_updated": False, "steps": 0,
        }
    torch, model = _network()
    torch.manual_seed(seed)
    # Recreate after seeding. CPU-only avoids surprising laptop GPU loads.
    torch, model = _network()
    sources = check["_pages"]
    splits = {part: [p["data"] for p in sources
                     if check["_assignments"][p["host"]] == part]
              for part in ("train", "val", "test")}
    last = store.get("focus_train_latest_" + profile_name) or {}
    root = (store.root / "models" / "focus" / profile_name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resumed = False
    if last.get("format") == FORMAT and last.get("accepted") is True:
        candidate = Path(last.get("checkpoint", ""))
        if (candidate.resolve().is_relative_to(root) and
                candidate.is_file() and _sha(candidate.read_bytes()) ==
                last.get("sha256")):
            state = torch.load(candidate, map_location="cpu",
                               weights_only=True)
            if state.get("format") == FORMAT:
                model.load_state_dict(state["weights"])
                resumed = True
    pre = _measure(torch, model, splits["val"], seed + 1000)
    import torch.nn.functional as F
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    prng = random.Random(seed)
    model.train()
    actual_steps, last_loss = 0, None
    for _ in range(int(steps)):
        x, y = _window_batch(torch, splits["train"], prng)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss; abandoning candidate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        actual_steps += 1
        last_loss = float(loss.item())
    val = _measure(torch, model, splits["val"], seed + 1000)
    # Evaluate test for diagnostic reporting only, NEVER select on test.
    test = _measure(torch, model, splits["test"], seed + 2000)
    accepted = val + 0.001 < pre
    all_params = sum(p.numel() for p in model.parameters())
    snapshot = {
        "format": FORMAT,
        "dataset_revision_sha256": check["dataset_revision_sha256"],
        "rights_policy": "per_page_manual_CC0_or_public_domain_no_noai",
        "hosts": check["source_hosts_by_split"],
        "source_hashes": sorted(
            [p["url"], p["sha256"], p["rights"]] for p in sources),
        "training_steps": actual_steps,
        "pre_validation_loss": pre,
        "post_validation_loss": val,
        "test_loss_for_diagnostics_only": test,
        "selected_on_test": False,
    }
    digest = _sha(_canonical(snapshot))
    manifest = root / ("manifest-" + digest + ".json")
    if not manifest.exists():
        manifest.write_bytes(_canonical(snapshot))
    checkpoint = root / ("candidate-" + digest + ".pt")
    if checkpoint.exists():
        raise ValueError("Refusing overwrite of an existing immutable model")
    torch.save({
        "format": FORMAT, "weights": {
            k: value.detach().cpu() for k, value in model.state_dict().items()},
        "manifest_sha256": digest,
    }, checkpoint)
    record = {
        "status": "trained", "format": FORMAT,
        "accepted": bool(accepted), "steps": actual_steps,
        "steps_total": (int(last.get("steps_total", 0)) if resumed else 0) +
        actual_steps,
        "resumed_accepted_weights": resumed,
        "trainable_parameters": all_params,
        "new_parameters_added": 0 if resumed else all_params,
        "training_window_bytes_seen": actual_steps * BATCH * CONTEXT,
        "unique_approved_training_bytes": check["bytes_by_split"]["train"],
        "dataset_revision_sha256": check["dataset_revision_sha256"],
        "checkpoint": str(checkpoint), "sha256": _sha(checkpoint.read_bytes()),
        "validation_before": pre, "validation_after": val,
        "heldout_publisher_test_loss_diagnostic": test,
        "train_loss_last_batch": round(last_loss, 6),
        "source_hosts_by_split": check["source_hosts_by_split"],
        "training_rights_auto_approved": False,
        "selected_on_test": False,
        "understands_saudi_history": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # The split assignments are set only after a real eligible training run.
    store.put("focus_host_split_" + profile_name, check["_assignments"])
    store.put("focus_train_attempted_" + profile_name, record)
    if accepted:
        store.put("focus_train_latest_" + profile_name, record)
    return record
