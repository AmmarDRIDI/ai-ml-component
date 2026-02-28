"""Utility helpers: directory creation and JSON I/O."""
import json
import os


def ensure_dir(path: str) -> None:
    """Create *path* (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str) -> None:
    """Serialise *data* to *path* as pretty-printed JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: str) -> dict:
    """Load and return the JSON object stored at *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
