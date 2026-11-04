"""Durable cycle orchestration. Process sleeps; model does not need to keep thinking."""
import json
import signal
import time
from datetime import datetime, timezone

from .discovery import discover
from .curriculum import expand_math_curriculum
from .lab import run_math_experiments
from .store import Store
from .training import train_candidate


def once(settings, discovery_limit=None, network=True, train=True, lab=True, fetcher=None,
         sleep_fn=time.sleep, practice=False):
    store = Store(settings["storage_root"])
    report = {"cycle_started_utc":datetime.now(timezone.utc).isoformat()}
    try:
        if network:
            try:
                if store.get('web_source_profile'):
                    from .web_discovery import discover_web
                    report["discovery"] = discover_web(store,settings,limit=discovery_limit,fetcher=fetcher,sleep_fn=sleep_fn)
                elif fetcher is not None:
                    report['discovery'] = discover(store,settings,limit=discovery_limit,fetcher=fetcher,sleep_fn=sleep_fn)
                else:
                    report["discovery"] = {"status":"not_configured","message":"Register a web seed using explore or web-start"}
            except Exception as error:
                report["discovery"] = {"status":"error", "message":str(error)}
        if network and store.get("web_source_profile"):
            try:
                from .research import research_questions
                report["research"] = research_questions(store, settings, fetcher=fetcher,
                                                        sleep_fn=sleep_fn)
            except Exception as error:
                report["research"] = {"status":"error", "message":str(error)}
        try:
            report["curriculum"] = expand_math_curriculum(store, settings)
        except Exception as error:
            report["curriculum"] = {"status":"error", "message":str(error)}
        if practice or settings.get("practice", {}).get("auto_enabled", False):
            try:
                from .practice import practice_arithmetic
                report["practice"] = practice_arithmetic(
                    store, limit=settings.get("practice", {}).get("batch_per_cycle", 32))
            except Exception as error:
                report["practice"] = {"status": "error", "message": str(error)}
        if train:
            try:
                profile = store.get("web_source_profile", {})
                verified_only = profile.get("mode") == "research_only" or practice or settings.get("practice", {}).get("auto_enabled", False)
                report["training"] = train_candidate(store, settings,
                                                    verified_only=verified_only)
            except Exception as error:
                report["training"] = {"status":"error", "message":str(error)}
        if lab:
            try:
                report["lab"] = run_math_experiments(store, settings)
            except Exception as error:
                report["lab"] = {"status":"error", "message":str(error)}
        report["status"] = store.status()
        store.log("cycle", "completed", report)
    finally:
        store.close()
    return report


def daemon(settings):
    """Run cycles on a cadence, stopped cleanly by SIGTERM or Ctrl+C."""
    keep_running = True

    def stop(signum, frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    interval = max(60, int(settings["daemon"]["poll_seconds"]))
    print("EVOLVE daemon running. Send SIGTERM/Ctrl+C to stop.", flush=True)
    while keep_running:
        print(json.dumps(once(settings), ensure_ascii=False, indent=2), flush=True)
        for _ in range(interval):
            if not keep_running:
                break
            time.sleep(1)
    print("EVOLVE daemon stopped; memory and model checkpoints remain saved.", flush=True)
