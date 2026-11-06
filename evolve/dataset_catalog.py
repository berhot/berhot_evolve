"""Mutable training receipts beside immutable datasets and readable folder views."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import uuid

from .plain_corpus import _atomic_write, _json

STATES = ("ready", "training", "trained/accepted", "trained/rejected",
          "trained/unclassified", "failed")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _directory(store, descriptor):
    revision = descriptor.get("dataset_revision_sha256", "")
    if not re.fullmatch(r"[a-f0-9]{64}", revision):
        raise ValueError("Invalid dataset revision")
    base = (store.root / "datasets" / "focus_plain").resolve()
    directory = base / revision
    if (Path(descriptor["directory"]).resolve() != directory or
            not directory.is_dir() or directory.is_symlink()):
        raise ValueError("Dataset must be an immutable revision inside focus_plain")
    return base, directory, revision


def _link(target, link):
    link.parent.mkdir(parents=True, exist_ok=True)
    # macOS /var aliases /private/var; calculate relative links using one
    # resolved namespace so dataset views survive either spelling.
    target = target.resolve()
    link = link.parent.resolve() / link.name
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() and not link.is_symlink():
        raise ValueError("Refusing to replace a dataset folder with a link")
    temporary = link.with_name(".link-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        temporary.replace(link)
    finally:
        temporary.unlink(missing_ok=True)


def _state(store, descriptor, status, **details):
    if status not in STATES:
        raise ValueError("Unknown dataset state")
    base, directory, revision = _directory(store, descriptor)
    previous = directory / "training_status.json"
    result = json.loads(previous.read_text()) if previous.is_file() else {}
    result.update(format="berhot_dataset_training_status_v1",
                  dataset_revision_sha256=revision, status=status,
                  updated_at_utc=_now(), **details)
    _atomic_write(previous, _json(result))
    # Folder views are links; immutable shards stay at their original path.
    _link(directory, base / status / revision)
    for other in STATES:
        link = base / other / revision
        if other != status and link.is_symlink():
            link.unlink()
    return result


def register_dataset(store, descriptor):
    base, directory, _ = _directory(store, descriptor)
    for state in STATES:
        (base / state).mkdir(parents=True, exist_ok=True)
    readme = base / "README.txt"
    if not readme.exists():
        _atomic_write(readme, (
            "PLAIN DATASETS AND TRAINING RECORDS\n\n"
            "ready/: prepared with no completed training result recorded\n"
            "training/: optimizer run in progress\n"
            "trained/accepted/: trained candidate accepted by validation\n"
            "trained/rejected/: trained candidate rejected; not active weights\n"
            "trained/unclassified/: historical optimizer evidence; promotion unknown\n"
            "failed/: a training attempt failed\n"
            "active/saudi-history-ar/: dataset used for the active accepted checkpoint\n\n"
            "Folder views are links to the original SHA-256 revision directories.\n"
            "Do not edit or move the original text shards.\n"
            "train.txt is used for optimizer updates; validation.txt and test.txt are held out.\n"
            "training_status.json and training_receipts/ record actual attempts.\n"
            "Different cumulative revisions can contain overlapping documents.\n\n"
            "Commands: python3 -m evolve dataset-status\n"
            "          python3 -m evolve dataset-organize\n"
            "Guide: docs/check-your-data-and-chat.md in the project.\n"
        ).encode("utf-8"))
    status_file = directory / "training_status.json"
    if not status_file.exists():
        _state(store, descriptor, "ready")
    else:
        # Repair missing folder views without relabeling previously trained
        # data as ready when the same immutable revision is reused.
        _state(store, descriptor, json.loads(status_file.read_text())["status"])


def begin_dataset_training(store, descriptor, profile, steps):
    _, directory, revision = _directory(store, descriptor)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    path = directory / "training_receipts" / (attempt + ".json")
    receipt = {"format": "berhot_dataset_training_receipt_v1", "attempt_id": attempt,
               "profile": profile, "dataset_revision_sha256": revision,
               "status": "training", "started_at_utc": _now(),
               "target_optimizer_steps": int(steps), "optimizer_reads": "train.txt",
               "held_out_files": ["validation.txt", "test.txt"]}
    _atomic_write(path, _json(receipt))
    _state(store, descriptor, "training", current_attempt=str(path))
    return str(path)


def finish_dataset_training(store, descriptor, receipt_path, record=None, error=None):
    _, directory, _ = _directory(store, descriptor)
    path = Path(receipt_path).resolve()
    if path.parent != directory / "training_receipts":
        raise ValueError("Training receipt escapes dataset")
    receipt = json.loads(path.read_text())
    if record is not None:
        status = "trained/accepted" if record.get("accepted") else "trained/rejected"
        receipt.update(status=status, finished_at_utc=_now(), result=record)
    else:
        status = "failed"
        receipt.update(status=status, finished_at_utc=_now(),
                       error_type=type(error).__name__, checkpoint_accepted=False)
    _atomic_write(path, _json(receipt))
    _state(store, descriptor, status, current_attempt=None,
           last_completed_attempt=str(path))
    return receipt


def sync_saved_training_records(store, profile="saudi-history-ar"):
    """Backfill only actual saved results; never infer learning from extraction."""
    history = store.get("focus_train_accepted_history_" + profile, []) or []
    latest = store.get("focus_train_latest_" + profile) or {}
    attempted = store.get("focus_train_attempted_" + profile) or {}
    records = {r["sha256"]: r for r in history + [latest, attempted] if r.get("sha256")}
    base = store.root / "datasets" / "focus_plain"
    copied = 0
    for record in sorted(records.values(), key=lambda r: r.get("created_at_utc", "")):
        revision = record.get("dataset_revision_sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", revision) or not (base / revision / "manifest.json").is_file():
            continue
        descriptor = {"directory": str(base / revision), "dataset_revision_sha256": revision}
        _, directory, _ = _directory(store, descriptor)
        path = directory / "training_receipts" / ("saved-" + record["sha256"] + ".json")
        # New attempts already have a receipt and need no duplicate import.
        recorded = any(json.loads(p.read_text()).get("result", {}).get("sha256") == record["sha256"]
                       for p in (directory / "training_receipts").glob("*.json"))
        if not recorded:
            status = "trained/accepted" if record.get("accepted") else "trained/rejected"
            receipt = {"format": "berhot_dataset_training_receipt_v1", "profile": profile,
                       "dataset_revision_sha256": revision, "status": status,
                       "recorded_from": "saved_training_result", "imported_at_utc": _now(),
                       "finished_at_utc": record.get("created_at_utc"), "result": record,
                       "optimizer_reads": "train.txt", "held_out_files": ["validation.txt", "test.txt"]}
            _atomic_write(path, _json(receipt))
            _state(store, descriptor, status, current_attempt=None, last_completed_attempt=str(path))
            copied += 1
    # Missing historical acceptance records must not turn an optimizer run
    # into either an accepted checkpoint or a never-trained dataset.
    known_paths = {r.get("checkpoint") for r in records.values()}
    models = store.root / "models" / "focus" / profile
    for manifest_path in sorted(models.glob("manifest-*.json")):
        snapshot = json.loads(manifest_path.read_text())
        revision = snapshot.get("dataset_revision_sha256", "")
        checkpoint = manifest_path.with_name(manifest_path.name.replace("manifest-", "candidate-").replace(".json", ".pt"))
        if (str(checkpoint) in known_paths or not checkpoint.is_file() or
                not re.fullmatch(r"[a-f0-9]{64}", revision) or
                not (base / revision / "manifest.json").is_file() or
                not snapshot.get("training_steps")):
            continue
        descriptor = {"directory": str(base / revision), "dataset_revision_sha256": revision}
        path = base / revision / "training_receipts" / ("legacy-" + manifest_path.stem + ".json")
        if path.exists():
            continue
        receipt = {"format": "berhot_dataset_training_receipt_v1", "profile": profile,
                   "dataset_revision_sha256": revision, "status": "trained/unclassified",
                   "recorded_from": "immutable_candidate_and_training_manifest",
                   "imported_at_utc": _now(), "optimizer_reads": "train.txt",
                   "held_out_files": ["validation.txt", "test.txt"],
                   "result": {"steps": snapshot["training_steps"], "accepted": None,
                              "checkpoint": str(checkpoint), "training_manifest": str(manifest_path),
                              "validation_before": snapshot.get("pre_validation_loss"),
                              "validation_after": snapshot.get("post_validation_loss")}}
        _atomic_write(path, _json(receipt))
        state_file = base / revision / "training_status.json"
        current = json.loads(state_file.read_text()).get("status") if state_file.is_file() else None
        if current not in ("training", "trained/accepted", "trained/rejected"):
            _state(store, descriptor, "trained/unclassified", last_completed_attempt=str(path))
        copied += 1
    if base.exists():
        for directory in base.iterdir():
            if not re.fullmatch(r"[a-f0-9]{64}", directory.name) or not (directory / "manifest.json").is_file():
                continue
            register_dataset(store, {"directory": str(directory), "dataset_revision_sha256": directory.name})
    revision = latest.get("dataset_revision_sha256", "")
    if re.fullmatch(r"[a-f0-9]{64}", revision) and (base / revision).is_dir():
        _link((base / revision).resolve(), base / "active" / profile)
    return {"receipts_imported": copied, **dataset_inventory(store, profile)}


def dataset_inventory(store, profile="saudi-history-ar"):
    """Read-only inventory. Prepared, attempted and accepted stay distinct."""
    base = store.root / "datasets" / "focus_plain"
    latest = store.get("focus_train_latest_" + profile) or {}
    items = []
    for directory in sorted(base.iterdir()) if base.is_dir() else []:
        if not re.fullmatch(r"[a-f0-9]{64}", directory.name) or directory.is_symlink():
            continue
        manifest_file = directory / "manifest.json"
        if not manifest_file.is_file():
            continue
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("profile") != profile:
            continue
        status_file = directory / "training_status.json"
        status = json.loads(status_file.read_text()) if status_file.is_file() else {"status": "ready"}
        receipts = [json.loads(p.read_text()) for p in sorted((directory / "training_receipts").glob("*.json"))]
        receipt_paths = sorted((directory / "training_receipts").glob("*.json"))
        attempts = [{"status": r["status"], "steps": r.get("result", {}).get("steps", 0),
                     "checkpoint_sha256": r.get("result", {}).get("sha256"),
                     "accepted": r.get("result", {}).get("accepted"),
                     "finished_at_utc": r.get("finished_at_utc"),
                     "receipt": str(path)} for r, path in zip(receipts, receipt_paths)]
        items.append({"revision": directory.name, "directory": str(directory),
                      "status": status["status"], "created_at_utc": manifest.get("created_at_utc"),
                      "documents_by_split": manifest.get("documents_by_split", {}),
                      "files": manifest["files"], "training_attempts": attempts,
                      "active_model_dataset": directory.name == latest.get("dataset_revision_sha256"),
                      "folder_view": str(base / status["status"] / directory.name)})
    return {"profile": profile, "root": str(base), "datasets": items,
            "counts": {state: sum(item["status"] == state for item in items) for state in STATES},
            "active_checkpoint_sha256": latest.get("sha256"),
            "note": "Trained means optimizer work was recorded on train.txt; held-out files were evaluated, not trained."}
