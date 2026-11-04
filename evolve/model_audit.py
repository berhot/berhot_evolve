"""Read-only honest inventory of EVOLVE learning, training and dependencies.

No PyTorch import needed: the audit can run before installing ML libraries.
The presence of a dataset, checkpoint or lower loss is NOT proof of reasoning.
"""
from importlib.util import find_spec
from pathlib import Path
import hashlib

from .settings import ROOT, load_config


def _digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _checkpoint(record, root):
    if not record:
        return {"status": "missing"}
    raw = record.get("path") or record.get("checkpoint")
    path = Path(raw).resolve() if raw else None
    # Explicitly say when an old index references a different machine's path.
    if not path or not path.is_file():
        return {"status": "recorded_checkpoint_missing_on_disk",
                "seed": record.get("seed"),
                "accepted": record.get("accepted")}
    # Checkpoint refs are untrusted store metadata: only report verified hashes.
    expected = record.get("sha256")
    result = {
        "status": "sha256_verified" if expected and _digest(path) == expected
        else "hash_missing_or_mismatch",
        "seed": record.get("seed"),
        "sha256": expected,
    }
    # The general registry stores 'promoted', not 'accepted'; absence of
    # a field must never be presented as a negative decision.
    if "promoted" in record:
        result["promoted"] = bool(record["promoted"])
    if "accepted" in record:
        result["accepted_by_validation"] = bool(record["accepted"])
    elif "accepted_by_validation" in record:
        result["accepted_by_validation"] = bool(record["accepted_by_validation"])
    else:
        result["accepted_by_validation"] = None
        result["acceptance_note"] = "No acceptance decision field recorded"
    return result


def audit_model_and_ownership(store, settings=None):
    settings = settings or load_config()
    definitions = int(store.db.execute(
        "SELECT COUNT(*) FROM verified_answers WHERE training_eligible=1"
    ).fetchone()[0])
    ready_definitions = int(store.db.execute(
        "SELECT COUNT(*) FROM documents "
        "WHERE topic='verified:definition' AND split='train' "
        "AND license='SELF-GENERATED-VERIFIED'").fetchone()[0])
    rows = store.db.execute(
        "SELECT COUNT(*) AS total, SUM(CASE WHEN promoted=1 THEN 1 ELSE 0 END)"
        " AS accepted FROM model_versions").fetchone()
    latest = store.latest_promoted()
    specialized = {
        name: _checkpoint(store.get(key), store.root)
        for name, key in (
            ("numeric_lab", "numeric_lab_latest"),
            ("rule_lab", "rule_lab_latest_candidate"),
            ("magnitude_lab", "magnitude_lab_latest_candidate"),
            ("linear_lab", "linear_lab_latest_candidate"),
            ("multiop_lab", "multiop_lab_latest_candidate"),
            ("interaction_lab", "interaction_lab_latest_candidate"),
            ("basis_lab", "basis_lab_latest_candidate"),
            ("calibration_lab", "calibration_lab_latest_candidate"),
            ("gradient_lab", "gradient_lab_latest_candidate"),
            ("concept_lab", "concept_lab_latest_candidate"),
        )
    }
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    return {
        "status": "audited",
        "source_code": {
            "repository": "berhot/berhot_evolve",
            "license_file": "MIT" if license_text.startswith("MIT License")
            else "other_or_custom",
            "exclusive_proprietary_license": False if license_text.startswith(
                "MIT License") else "unverified",
            "authorship_and_ownership": (
                "Repository content alone cannot prove chain of title, "
                "exclusive ownership or rights to external source data."
            ),
        },
        "datasets": {
            "verified_arithmetic_examples": int(store.db.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE topic='verified:arithmetic'").fetchone()[0]),
            "verified_text_facts_with_training_eligibility": definitions,
            "eligible_definition_train_documents": ready_definitions,
            "research_page_observations": int(store.db.execute(
                "SELECT COUNT(*) FROM page_observations").fetchone()[0]),
            "web_corpus_unique_cleaned_pages": int(store.db.execute(
                "SELECT COUNT(*) FROM web_corpus_texts").fetchone()[0]),
        },
        "saved_models": {
            "general_training_candidate_runs": int(rows["total"] or 0),
            "general_promoted_runs": int(rows["accepted"] or 0),
            "latest_general_promoted": (
                {"status": "found", "checkpoint": _checkpoint(dict(latest), store.root),
                 "trained_on_document_count_at_save": int(latest["train_documents"]),
                 "validation_byte_loss": float(latest["val_loss"])}
                if latest else {"status": "none"}),
            "specialized_checkpoint_records": specialized,
            "unicode_char_lab_checkpoint": _checkpoint(
                store.get("char_lab_latest_candidate"), store.root),
            "unicode_char_lab_run": ({
                "dataset_sha256": store.get("char_lab_latest_candidate")["dataset_sha256"],
                "tokenizer_sha256": store.get("char_lab_latest_candidate")["tokenizer_sha256"],
                "total_training_steps": store.get("char_lab_latest_candidate")["total_training_steps"],
                "resumed_from_sha256": store.get("char_lab_latest_candidate")["resumed_from_sha256"],
                "validation_char_loss_after": store.get("char_lab_latest_candidate")["validation_char_loss_after"],
                "diagnostic_source_leakage": store.get("char_lab_latest_candidate")["diagnostic_source_leakage"],
            } if store.get("char_lab_latest_candidate") else None),
            "web_lab_checkpoint": _checkpoint(
                store.get("web_lab_latest_candidate"), store.root),
            "web_lab_run": ({
                "dataset_sha256": store.get("web_lab_latest_candidate")["dataset_sha256"],
                "actual_steps": store.get("web_lab_latest_candidate")["actual_steps"],
                "diagnostic_source_leakage": store.get("web_lab_latest_candidate")[
                    "diagnostic_source_leakage"],
                "validation_byte_loss_after": store.get("web_lab_latest_candidate")[
                    "validation_byte_loss_after"],
            } if store.get("web_lab_latest_candidate") else None),
        },
        "autonomy": {
            "manual_commands": "research, discover, train and specialized labs",
            "automatic_cycle_available": "python3 -m evolve daemon",
            "configured_cycle_seconds": max(60, int(settings["daemon"]["poll_seconds"])),
            "daemon_running_now": "unknown_from_database",
            "neural_weights_update_when_pages_are_saved": False,
            "neural_weights_update_when_training_runs": True,
            "independently_chosen_research_goals": False,
            "self_generated_general_knowledge": "not_demonstrated",
        },
        "implementation": {
            "collector_external_python_packages_required": [],
            "neural_backend": "PyTorch torch.nn / torch.optim / autograd",
            "declared_optional_training_dependencies": ["torch>=2.2", "numpy>=1.26"],
            "torch_installed": find_spec("torch") is not None,
            "numpy_installed": find_spec("numpy") is not None,
            "python_stdlib_used": ["sqlite3", "urllib", "html.parser", "hashlib", "json"],
            "is_zero_external_dependencies": False,
            "is_fully_original_neural_framework": False,
        },
        "assessment": (
            "Exact counts come from the current local SQLite database and "
            "checkpoint indexes, not GitHub CI fixtures. Arithmetic precision "
            "must be taken from separately measured evaluations. No count "
            "of facts or model parameters equals proved knowledge."
        ),
    }
