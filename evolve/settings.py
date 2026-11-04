"""Configuration and cautious operating limits."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    config_path = Path(path) if path else ROOT / "config" / "settings.json"
    with config_path.open(encoding="utf-8") as file:
        settings = json.load(file)
    settings["_project_root"] = str(config_path.resolve().parent.parent)
    root = Path(settings["storage_root"])
    if not root.is_absolute():
        root = Path(settings["_project_root"]) / root
    settings["storage_root"] = str(root.resolve())
    return settings


def user_agent():
    """Contact info is mandatory for live Wikimedia API access."""
    contact = os.environ.get("EVOLVE_CONTACT", "").strip()
    if not contact or contact == "replace@example.com":
        raise RuntimeError(
            "Set EVOLVE_CONTACT to your real email or project contact URL before online discovery. "
            "For example: export EVOLVE_CONTACT='mailto:research@example.org'"
        )
    if not ("@" in contact or contact.startswith("https://")):
        raise RuntimeError("EVOLVE_CONTACT must contain an email address or HTTPS contact URL")
    return f"BERHOT-EVOLVE/0.1 ({contact}) Python-urllib"
