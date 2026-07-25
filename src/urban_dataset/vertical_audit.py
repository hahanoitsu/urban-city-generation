from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .extract import TRANSPORT_VERTICAL_TAGS
from .prepared import load_city_gpkg
from .vertical import vertical_mode_name


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in {"", "none", "nan", "null"}


def _tag_summary(frame, tag: str, *, examples: int) -> dict[str, Any]:
    if tag not in frame.columns:
        return {"features": 0, "coverage": 0.0, "common_values": []}
    values = [str(value).strip() for value in frame[tag] if _present(value)]
    counts = Counter(values)
    return {
        "features": len(values),
        "coverage": len(values) / max(len(frame), 1),
        "common_values": [
            {"value": value, "features": count}
            for value, count in counts.most_common(max(0, examples))
        ],
    }


def audit_vertical_tags(
    city_path: str | Path,
    *,
    examples: int = 8,
) -> dict[str, Any]:
    city_path = Path(city_path).expanduser().resolve()
    layers, metadata = load_city_gpkg(city_path)
    result: dict[str, Any] = {
        "city": str(city_path),
        "city_id": metadata.get("city_id"),
        "source_snapshot": metadata.get("source_snapshot"),
        "transport_tags": list(TRANSPORT_VERTICAL_TAGS),
    }

    for name, frame in (("roads", layers.roads), ("rail", layers.rail)):
        modes = Counter(vertical_mode_name(row) for _index, row in frame.iterrows())
        result[name] = {
            "features": len(frame),
            "vertical_modes": dict(sorted(modes.items())),
            "tags": {
                tag: _tag_summary(frame, tag, examples=examples)
                for tag in TRANSPORT_VERTICAL_TAGS
            },
        }
    return result
