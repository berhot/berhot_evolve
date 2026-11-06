"""Local process control and transparent model-size statistics for EVOLVE focus.

Mac/Linux local-only. A lock prevents duplicate long-running focus workers.
Stop requests target a specific run ID, not all future workers.
"""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache

PROFILE = "saudi-history-ar"


def _key(profile=PROFILE):
    return "focus_stop_request_" + profile


def _lock_path(store, profile=PROFILE):
    return Path(store.root) / ("focus-worker-" + profile + ".lock")


def acquire_lock(store, profile=PROFILE):
    path = _lock_path(store, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("A focus worker is already active for this profile")
    return handle


def release_lock(handle):
    if handle is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def worker_running(store, profile=PROFILE):
    path = _lock_path(store, profile)
    if not path.exists():
        return False
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
    finally:
        handle.close()


def stop_requested(store, run_id, profile=PROFILE):
    request = store.get(_key(profile)) or {}
    return bool(run_id and request.get("run_id") == run_id)


def request_stop(store, profile=PROFILE):
    run = store.get("focus_run_" + profile) or {}
    if not run.get("run_id") or not worker_running(store, profile):
        return {
            "status": "not_running",
            "message": "No active focus worker; saved records remain available",
        }
    record = {
        "run_id": run["run_id"],
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_via": "local_operator",
    }
    store.put(_key(profile), record)
    return {
        "status": "stop_requested",
        "run_id": run["run_id"],
        "note": "A current source fetch or training step may finish before shutdown.",
    }


def start_detached(store, project_root, profile=PROFILE):
    """Studio starts a SEPARATE process, independent of the browser window.

    The CLI must be explicitly authorized by a local CSRF-protected button.
    Never add a remote host or train unknown-rights sources.
    """
    if worker_running(store, profile):
        return {"status": "already_running"}
    # v1.27 used nohup/caffeinate without an OS flock. A PID file from that
    # legacy worker must not silently permit a second concurrent crawler.
    # Do NOT terminate any arbitrary process by PID; require operator review.
    legacy = Path(store.root) / "focus-overnight.pid"
    if legacy.is_file():
        try:
            old_pid = int(legacy.read_text().strip())
            os.kill(old_pid, 0)
        except (OSError, ValueError):
            pass
        else:
            inspect = subprocess.run(
                ["ps", "-p", str(old_pid), "-o", "command="],
                capture_output=True, text=True, timeout=3, check=False)
            command = inspect.stdout.strip()
            if ("focus-overnight" in command or
                    ("caffeinate" in command and "-m evolve" in command)):
                return {
                    "status": "legacy_worker_detected",
                    "pid": old_pid,
                    "message": ("An older eight-hour focus worker is still "
                                "running. Stop it from its terminal or review "
                                "the PID before starting continuous mode."),
                }
    root = Path(project_root).resolve()
    logdir = Path(store.root) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / (profile + "-continuous.log")
    cmd = [sys.executable, "-u", "-m", "evolve",
           "focus-overnight", "--network", "--profile", profile,
           "--until-stopped", "--no-fixed-quotas",
           "--browser-search-provider", "duckduckgo",
           "--pages-per-cycle", "20", "--interval-minutes", "2",
           "--train-steps", "40"]
    owned=(store.get('focus_execution_policy') or {}).get('engine')=='first_party_c'
    if owned:
        cmd=[sys.executable,'-S','-u','-m','evolve','owned-learn','--steps','120',
             '--context-bytes','256','--until-stopped']
    pytorch=(store.get('focus_execution_policy') or {}).get('engine')=='pytorch'
    if pytorch:
        cmd=[sys.executable,'-u','-m','evolve','focus-learn','--steps','120',
             '--context-bytes','256','--until-stopped']
        if (store.get('focus_execution_policy') or {}).get('autonomous_pipeline'):
            cmd=[sys.executable,'-u','-m','evolve','focus-autonomous','--network','--until-stopped',
                 '--pages-per-cycle','5','--train-steps','120','--interval-minutes','2']
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", *cmd]
    with logfile.open("ab") as output:
        proc = subprocess.Popen(
            cmd, cwd=str(root), stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return {
        "status": "launched",
        "pid": proc.pid,
        "profile": profile,
        "log_path": str(logfile),
        "note": (
            "Owned C/Python learning process; network, external model APIs and site packages disabled."
            if owned else "PyTorch learning for your own model and data; no external model APIs or pretrained weights."
            if pytorch else
            "Separate local process. It remains active after this browser "
            "closes while the Mac stays awake. Publisher pacing, robots "
            "and rights checks still apply."
        ),
    }


@lru_cache(maxsize=8)
def _focus_parameter_count(embedding_dim, hidden_size):
    from .focus_training import _network
    _, architecture = _network({"embedding_dim": embedding_dim, "hidden_size": hidden_size})
    return sum(int(p.numel()) for p in architecture.parameters())


def model_size_snapshot(store):
    """Read metadata for trained status; parameter counts from real code shapes.

    Building an architecture here creates temporary initialized weights only.
    The returned parameter count is NOT a claim of model training or growth.
    """
    # Keep the Studio status endpoint functional on Python installs that
    # have not installed optional PyTorch yet. These formulas correspond to
    # the exact current nn.Linear/Embedding/GRU module dimensions.
    ranker_params = 7 * 12 + 12 + 12 * 1 + 1
    from .focus_training import model_config, parameter_count, LEGACY_MODEL_CONFIG
    config = model_config(store, PROFILE)
    focus_params = parameter_count(config)
    computed_from_pytorch = False
    if (store.get('focus_execution_policy') or {}).get('engine')!='first_party_c':
        try:
            from .v120_grounded import _model
            ranker_params = sum(int(p.numel()) for p in _model().parameters())
            focus_params = _focus_parameter_count(config["embedding_dim"], config["hidden_size"])
            computed_from_pytorch = True
        except RuntimeError as exc:
            if "Install PyTorch first" not in str(exc):
                raise
    ranker = store.get("v120_ranker_latest") or {}
    focus = store.get("focus_train_latest_" + PROFILE) or {}
    candidate = store.get("focus_train_attempted_" + PROFILE) or {}
    bytes_row = store.db.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(characters),0) AS chars "
        "FROM web_corpus_texts"
    ).fetchone()
    return {
        "ranker": {
            "architecture_parameters": ranker_params,
            "checkpoint_recorded": bool(ranker.get("accepted_by_validation")),
            "checkpoint_sha256": ranker.get("sha256"),
            "training_steps_recorded": ranker.get("actual_training_steps", 0),
        },
        "saudi_history_arabic": {
            "architecture_parameters": focus_params,
            "model_config": config,
            "accepted_checkpoint_parameters": focus.get("trainable_parameters", 0),
            "accepted_checkpoint_matches_architecture": bool(focus.get("accepted")) and
                (focus.get("model_config") or LEGACY_MODEL_CONFIG) == config,
            "accepted_checkpoint_recorded": bool(focus.get("accepted")),
            "accepted_checkpoint_sha256": focus.get("sha256"),
            "accepted_training_steps_recorded": focus.get("steps_total", 0),
            "latest_training_attempt_steps": candidate.get("steps", 0),
            "parameters_grew_from_more_articles": False,
        },
        "architecture_counts_from_installed_pytorch": computed_from_pytorch,
        "archived_unique_cleaned_pages": int(bytes_row["n"]),
        "archived_cleaned_characters": int(bytes_row["chars"]),
        "does_not_count_architecture_without_training_as_a_trained_model": True,
        "note": "Architecture size is fixed until code/config changes; retrieval and raw archived texts are not learned neural weights.",
    }
