import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    """
    Create directory (and parents) if it doesn't exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

