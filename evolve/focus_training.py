"""Scoped, bounded byte-language-model training on explicitly approved Arabic text.

UNKNOWN text needs snapshot-specific operator approval. No noai text is used.
Test sources are distinct
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
from .training_approval import training_condition, operator_approvals
from .training_approval import future_approval_policy

FORMAT = "berhot_focus_byte_gru_v1"
MIN_BYTES = {"train": 1600, "val": 600, "test": 600}
CONTEXT = 48
BATCH = 8
HOST_SPLIT = "heldout_source_hosts"
ARTICLE_SPLIT = "heldout_articles_diagnostic"
LEGACY_MODEL_CONFIG = {"embedding_dim": 48, "hidden_size": 96}


def _validate_model_config(config):
    result = {}
    for key, default in LEGACY_MODEL_CONFIG.items():
        value = config.get(key, default)
        if type(value) is not int or not 16 <= value <= 1024:
            raise ValueError(key + " must be an integer from 16 to 1024")
        result[key] = value
    return result


def model_config(store, profile_name="saudi-history-ar"):
    return _validate_model_config(store.get("focus_model_config_" + profile_name) or {})


def parameter_count(config):
    config = _validate_model_config(config)
    embed, hidden = config["embedding_dim"], config["hidden_size"]
    return 256 * embed + 3 * hidden * embed + 3 * hidden * hidden + 6 * hidden + hidden * 256 + 256


def set_model_capacity(store, profile_name, embedding_dim, hidden_size):
    from .focus_control import worker_running
    get_profile(profile_name)
    if worker_running(store, profile_name):
        raise ValueError("Stop the focus worker before changing model capacity")
    config = _validate_model_config({"embedding_dim": embedding_dim, "hidden_size": hidden_size})
    previous = model_config(store, profile_name)
    if any(config[k] < previous[k] for k in previous):
        raise ValueError("Capacity changes must preserve or increase each dimension")
    store.put("focus_model_config_" + profile_name, config)
    result = {"model_config": config, "architecture_parameters": parameter_count(config),
              "previous_parameters": parameter_count(previous),
              "existing_checkpoints_preserved": True}
    store.log("focus_model_capacity", "configured", dict(profile=profile_name, **result))
    return result


def set_article_holdout_policy(store, profile_name, enabled):
    get_profile(profile_name)
    if type(enabled) is not bool:
        raise ValueError("Article holdout policy must be a boolean")
    policy = {"allow_article_holdout": enabled,
              "configured_at_utc": datetime.now(timezone.utc).isoformat(),
              "independent_publisher_evaluation": not enabled}
    store.put("focus_training_policy_" + profile_name, policy)
    store.log("focus_training_policy", "configured", dict(profile=profile_name, **policy))
    return policy


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _approved_pages(store, profile):
    rows = store.db.execute(
        "SELECT url,host,rights_status,noai FROM web_corpus_sources "
        "WHERE " + training_condition() + " "
        "ORDER BY host,url"
    ).fetchall()
    approved = []
    review_history = store.get("v117_rights_review_history", []) or []
    reviewed = {
        (r.get("url"), r.get("clean_sha256"), r.get("reviewed_rights"))
        for r in review_history if r.get("reviewer_attested_per_page") is True
    }
    approvals = operator_approvals(store)
    for row in rows:
        # Focus is defined by topic and Arabic text, not a fixed publisher
        # allowlist. New independently sourced hosts require identical manual
        # per-page rights attestation before training admission.
        source, text = _approved_text(store, row["url"])
        # A publisher signal alone is insufficient: require either a rights
        # review or an explicit operator decision for this exact digest.
        if ((row["url"], source["sha256"], row["rights_status"]) not in reviewed
                and (row["url"], source["sha256"]) not in approvals):
            continue
        quality = topic_relevance(text, source["title"], profile)
        if quality["language_ok"] and quality["topical"]:
            approved.append({
                "url": row["url"], "host": row["host"],
                "sha256": source["sha256"],
                "rights": row["rights_status"],
                "operator_training_approved": (row["url"], source["sha256"]) in approvals,
                "data": text.encode("utf-8"),
            })
    # A duplicated full source at two publishers is not independent
    # validation evidence. Keep a clean digest at most once across hosts,
    # before any publisher split is assigned.
    unique, seen_digests = [], set()
    for page in approved:
        if page["sha256"] in seen_digests:
            continue
        seen_digests.add(page["sha256"])
        unique.append(page)
    return unique


def readiness(store, profile_name):
    profile = get_profile(profile_name)
    pages = _approved_pages(store, profile)
    config = model_config(store, profile_name)
    hosts = sorted({p["host"] for p in pages})
    policy = store.get("focus_training_policy_" + profile_name, {}) or {}
    diagnostic = bool(policy.get("allow_article_holdout"))
    method = ARTICLE_SPLIT if diagnostic else HOST_SPLIT
    assignment_key = ("focus_article_split_" if diagnostic else "focus_host_split_") + profile_name
    assignments = dict(store.get(assignment_key) or {})
    if diagnostic:
        # Group exact copies by cleaned digest. Whole documents stay in one
        # partition, and saved assignments never move as future pages arrive.
        initial = not assignments
        for i, page in enumerate(sorted(pages, key=lambda p: p["sha256"])):
            bucket = i % 10 if initial else int(page["sha256"][:8], 16) % 10
            assignments.setdefault(page["sha256"], "val" if bucket == 0 else
                                   "test" if bucket == 1 else "train")
    elif not assignments and len(hosts) >= 3:
        assignments = {host: ("train" if i < len(hosts) - 2
                             else "val" if i == len(hosts) - 2
                             else "test") for i, host in enumerate(hosts)}
    elif assignments:
        for host in hosts:
            assignments.setdefault(host, "train")
    def partition(page):
        return assignments.get(page["sha256"] if diagnostic else page["host"])
    sizes = {k: sum(len(p["data"]) for p in pages
                    if partition(p) == k)
             for k in ("train", "val", "test")}
    eligible = (
        (len(pages) >= 3 if diagnostic else len(hosts) >= 3) and all(sizes[k] >= MIN_BYTES[k]
                                for k in ("train", "val", "test"))
    )
    manifest = sorted({
        (p["url"], p["sha256"], p["rights"], partition(p))
        for p in pages
    })
    digest = _sha(_canonical({
        "model_format": FORMAT, "profile": profile_name,
        "sources": manifest, "split_method": method, "model_config": config,
    }))
    previous = store.get("focus_train_latest_" + profile_name) or {}
    attempted = store.get("focus_train_attempted_" + profile_name) or {}
    return {
        "status": "ready" if eligible else "insufficient_approved_arabic_publishers_or_text",
        "rights_eligible_arabic_pages": len(pages),
        "operator_approved_arabic_pages": sum(p["operator_training_approved"] for p in pages),
        "required_independent_hosts": 3,
        "split_method": method,
        "independent_publisher_evaluation": not diagnostic,
        "article_holdout_enabled": diagnostic,
        "documents_by_split": {k: sum(partition(p) == k for p in pages)
                               for k in ("train", "val", "test")},
        "accepts_explicit_operator_approval": True,
        "requires_human_per_page_rights_review": True,
        "exact_content_sha_dedup_before_host_split": True,
        "rights_eligible_arabic_hosts": hosts,
        "source_hosts_by_split": {
            split: sorted({p["host"] for p in pages if partition(p) == split})
            for split in ("train", "val", "test")},
        "bytes_by_split": sizes,
        "minimum_bytes_by_split": dict(MIN_BYTES),
        "dataset_revision_sha256": digest,
        "new_training_data_since_last_attempt":
            eligible and attempted.get("dataset_revision_sha256") != digest,
        "last_accepted_checkpoint_sha256": previous.get("sha256"),
        "last_accepted_training_steps": previous.get("steps_total", 0),
        "trainable_model_type": "small_causal_byte_GRU_language_model",
        "model_config": config,
        "architecture_parameters": parameter_count(config),
        "training_rights_auto_approved": bool(future_approval_policy(store).get("enabled")),
        "readiness_is_factual_understanding": False,
        "_assignments": assignments, "_assignment_key": assignment_key,
        "_partitions": {p["sha256"]: partition(p) for p in pages}, "_pages": pages,
    }


def _network(config=None):
    from .brain import require_torch
    torch, nn = require_torch()
    config = _validate_model_config(config or {})

    class ByteGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, config["embedding_dim"])
            self.gru = nn.GRU(config["embedding_dim"], config["hidden_size"], batch_first=True)
            self.head = nn.Linear(config["hidden_size"], 256)

        def forward(self, ids):
            sequence, _ = self.gru(self.embedding(ids))
            return self.head(sequence)

    return torch, ByteGRU()


def _expand_accepted_weights(torch, model, weights, old_config, new_config):
    """Widen a GRU while initially preserving the old prediction function.

    Old units cannot see new input/hidden dimensions initially. New units
    supply additional features; their output connections start at zero and
    learn normally, so old learned predictions survive the capacity change.
    """
    old_config, new_config = _validate_model_config(old_config), _validate_model_config(new_config)
    if any(new_config[k] < old_config[k] for k in old_config):
        raise ValueError("Cannot warm-start a smaller model from larger weights")
    old_e, old_h = old_config["embedding_dim"], old_config["hidden_size"]
    new_h = new_config["hidden_size"]
    with torch.no_grad():
        model.embedding.weight[:, :old_e].copy_(weights["embedding.weight"])
        for gate in range(3):
            old_rows = slice(gate * old_h, (gate + 1) * old_h)
            new_rows = slice(gate * new_h, gate * new_h + old_h)
            model.gru.weight_ih_l0[new_rows].zero_()
            model.gru.weight_ih_l0[new_rows, :old_e].copy_(weights["gru.weight_ih_l0"][old_rows])
            model.gru.weight_hh_l0[new_rows].zero_()
            model.gru.weight_hh_l0[new_rows, :old_h].copy_(weights["gru.weight_hh_l0"][old_rows])
            for name in ("bias_ih_l0", "bias_hh_l0"):
                getattr(model.gru, name)[new_rows].copy_(weights["gru." + name][old_rows])
        model.head.weight.zero_()
        model.head.weight[:, :old_h].copy_(weights["head.weight"])
        model.head.bias.copy_(weights["head.bias"])


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


def train_if_eligible(store, profile_name, steps=60, seed=31415, progress=None):
    """Record dataset lifecycle without moving any immutable shard/checkpoint."""
    policy=store.get('focus_execution_policy') or {}
    if policy.get('engine')=='pytorch' and policy.get('article_body_training'):
        from .pytorch_learning import train_pytorch
        return train_pytorch(store,steps=int(steps),context=256,seed=seed,progress=progress)
    if policy.get('engine')=='first_party_c':
        from .owned_learning import train_owned
        return train_owned(store,steps=int(steps),context=256,seed=seed,progress=progress)
    from .dataset_catalog import finish_dataset_training
    lifecycle = {}
    try:
        record = _train_if_eligible(store, profile_name, steps, seed, progress, lifecycle)
    except Exception as exc:
        if lifecycle:
            finish_dataset_training(store, lifecycle["dataset"], lifecycle["receipt"], error=exc)
        raise
    if lifecycle:
        finish_dataset_training(store, lifecycle["dataset"], lifecycle["receipt"], record=record)
        if record.get("accepted"):
            from .dataset_catalog import sync_saved_training_records
            sync_saved_training_records(store, profile_name)
    return record


def _train_if_eligible(store, profile_name, steps, seed, progress, lifecycle):
    """Train only approved data; optionally emit measured live PyTorch steps.

    progress(completed_steps, target_steps, actual_batch_loss) is invoked
    only AFTER a real optimizer update, never during skipped training.
    """
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
    from .plain_corpus import prepare_focus_dataset, read_focus_dataset
    plain_dataset = prepare_focus_dataset(store, profile_name, check)
    if not check["new_training_data_since_last_attempt"]:
        return {
            "status": "skipped_unchanged_approved_data",
            "dataset_revision_sha256": check["dataset_revision_sha256"],
            "weights_updated": False, "steps": 0,
            "plain_dataset_directory": plain_dataset["directory"],
        }
    from .dataset_catalog import begin_dataset_training
    lifecycle.update(dataset=plain_dataset, receipt=begin_dataset_training(
        store, plain_dataset, profile_name, steps))
    from .brain import require_torch
    torch, _ = require_torch()
    torch.manual_seed(seed)
    # CPU-only avoids surprising laptop GPU loads.
    torch, model = _network(check["model_config"])
    sources = check["_pages"]
    splits = read_focus_dataset(plain_dataset, check)
    last = store.get("focus_train_latest_" + profile_name) or {}
    root = (store.root / "models" / "focus" / profile_name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resumed = False
    expanded = False
    parent_parameters = 0
    if (last.get("format") == FORMAT and last.get("accepted") is True and
            last.get("split_method", HOST_SPLIT) == check["split_method"]):
        candidate = Path(last.get("checkpoint", ""))
        if (candidate.resolve().is_relative_to(root) and
                candidate.is_file() and _sha(candidate.read_bytes()) ==
                last.get("sha256")):
            state = torch.load(candidate, map_location="cpu",
                               weights_only=True)
            if (state.get("format") == FORMAT and
                    state.get("split_method", HOST_SPLIT) == check["split_method"]):
                old_config = _validate_model_config(state.get("model_config") or last.get("model_config") or {})
                parent_parameters = parameter_count(old_config)
                if old_config == check["model_config"]:
                    model.load_state_dict(state["weights"])
                else:
                    _expand_accepted_weights(torch, model, state["weights"], old_config, check["model_config"])
                    expanded = True
                resumed = True
    pre = _measure(torch, model, splits["val"], seed + 1000)
    import torch.nn.functional as F
    learning_rate = 0.0003 if parameter_count(check["model_config"]) >= 1_000_000 else 0.001
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
        if progress is not None:
            progress(actual_steps, int(steps), last_loss)
    val = _measure(torch, model, splits["val"], seed + 1000)
    # Evaluate test for diagnostic reporting only, NEVER select on test.
    test = _measure(torch, model, splits["test"], seed + 2000)
    accepted = val + 0.001 < pre
    all_params = sum(p.numel() for p in model.parameters())
    snapshot = {
        "format": FORMAT,
        "context_bytes": CONTEXT,
        "training_objective": "next_byte_prediction",
        "validation_metric": "next_UTF8_byte_cross_entropy",
        "model_config": check["model_config"],
        "dataset_revision_sha256": check["dataset_revision_sha256"],
        "plain_dataset_format": "berhot_plain_focus_dataset_v1",
        "rights_policy": "reviewed_rights_or_operator_approval_no_noai",
        "split_method": check["split_method"],
        "independent_publisher_evaluation": check["independent_publisher_evaluation"],
        "source_hashes_by_split": {part: sorted(
            p["sha256"] for p in sources if check["_partitions"][p["sha256"]] == part)
            for part in splits},
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
        "split_method": check["split_method"],
        "model_config": check["model_config"],
        "context_bytes": CONTEXT,
    }, checkpoint)
    record = {
        "status": "trained", "format": FORMAT,
        "context_bytes": CONTEXT,
        "training_objective": "next_byte_prediction",
        "validation_metric": "next_UTF8_byte_cross_entropy",
        "history_qa_evaluated": False,
        "accepted": bool(accepted), "steps": actual_steps,
        "steps_total": (int(last.get("steps_total", 0)) if resumed else 0) +
        actual_steps,
        "resumed_accepted_weights": resumed,
        "expanded_accepted_weights": expanded,
        "parent_checkpoint_sha256": last.get("sha256") if resumed else None,
        "model_config": check["model_config"],
        "learning_rate": learning_rate,
        "trainable_parameters": all_params,
        "new_parameters_added": all_params - parent_parameters if resumed else all_params,
        "training_window_bytes_seen": actual_steps * BATCH * CONTEXT,
        "unique_approved_training_bytes": check["bytes_by_split"]["train"],
        "plain_dataset_directory": plain_dataset["directory"],
        "plain_dataset_manifest": plain_dataset["manifest"],
        "dataset_revision_sha256": check["dataset_revision_sha256"],
        "checkpoint": str(checkpoint), "sha256": _sha(checkpoint.read_bytes()),
        "validation_before": pre, "validation_after": val,
        "heldout_test_loss_diagnostic": test,
        "heldout_publisher_test_loss_diagnostic": test if check["independent_publisher_evaluation"] else None,
        "train_loss_last_batch": round(last_loss, 6),
        "source_hosts_by_split": check["source_hosts_by_split"],
        "split_method": check["split_method"],
        "independent_publisher_evaluation": check["independent_publisher_evaluation"],
        "documents_by_split": check["documents_by_split"],
        "source_hashes_by_split": snapshot["source_hashes_by_split"],
        "training_rights_auto_approved": check["training_rights_auto_approved"],
        "selected_on_test": False,
        "understands_saudi_history": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # The split assignments are set only after a real eligible training run.
    store.put(check["_assignment_key"], check["_assignments"])
    store.put("focus_train_attempted_" + profile_name, record)
    if accepted:
        history = store.get("focus_train_accepted_history_" + profile_name, []) or []
        if last and not any(r.get("sha256") == last.get("sha256") for r in history):
            history.append(last)
        history.append(record)
        store.put("focus_train_accepted_history_" + profile_name, history)
        store.put("focus_train_latest_" + profile_name, record)
    return record
