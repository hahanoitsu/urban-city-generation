from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return path, payload


def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent.parent / path).resolve()
