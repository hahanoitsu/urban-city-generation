from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from urban_dataset.prepared import read_city_metadata
from urban_dataset.vertical import vertical_mode_name

VERTICAL_TAGS = [
    "bridge", "bridge:structure", "bridge:movable", "tunnel", "location",
    "layer", "level", "incline", "ele", "height", "min_height", "depth",
    "maxheight", "maxheight:physical", "embankment", "cutting",
]
FALSE = {"", "no", "false", "0", "none", "nan", "<na>"}
NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def clean(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().lower()


def number(value) -> float | None:
    match = NUMBER.search(clean(value))
    if not match:
        return None
    try:
        result = float(match.group(0).replace(",", "."))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def positive(value) -> bool:
    return clean(value) not in FALSE


def length_series(frame: gpd.GeoDataFrame) -> pd.Series:
    if frame.crs is None or not frame.crs.is_projected:
        return pd.Series(np.nan, index=frame.index)
    return frame.geometry.length.astype(float)


def top_values(frame, lengths, column, limit=40):
    if column not in frame.columns:
        return []
    values = frame[column].map(clean)
    data = pd.DataFrame({"value": values, "length_m": lengths})
    data = data[data.value != ""]
    if data.empty:
        return []
    grouped = (
        data.groupby("value")
        .agg(features=("value", "size"), length_m=("length_m", "sum"))
        .reset_index()
        .sort_values(["length_m", "features"], ascending=False)
        .head(limit)
    )
    return grouped.to_dict(orient="records")


def issues_for(row) -> list[str]:
    mode = vertical_mode_name(row)
    tunnel = clean(row.get("tunnel"))
    location = clean(row.get("location"))
    layer = number(row.get("layer"))
    level = number(row.get("level"))
    railway = clean(row.get("railway"))

    underground = (
        tunnel not in FALSE and tunnel not in {"building_passage", "covered"}
    ) or location in {"underground", "subsurface", "below_ground"}
    elevated = positive(row.get("bridge")) or location in {
        "overground", "overhead", "elevated", "above_ground"
    }

    issues = []
    if underground and elevated:
        issues.append("explicit_underground_and_elevated")
    if underground and layer is not None and layer > 0:
        issues.append("underground_with_positive_layer")
    if elevated and layer is not None and layer < 0:
        issues.append("elevated_with_negative_layer")
    if mode == "unknown":
        issues.append("classifier_unknown")
    if (
        mode == "unknown"
        and layer is not None
        and abs(layer) > 1e-9
        and not underground
        and not elevated
    ):
        issues.append("layer_only_positive" if layer > 0 else "layer_only_negative")
    if railway == "subway" and mode == "surface":
        issues.append("subway_classified_surface")
    if railway == "subway" and mode == "unknown":
        issues.append("subway_classified_unknown")
    if railway == "subway" and not underground and not elevated:
        issues.append("subway_without_explicit_vertical_tag")
    if clean(row.get("bridge:structure")) and not positive(row.get("bridge")):
        issues.append("bridge_structure_without_bridge")
    if clean(row.get("bridge:movable")) and not positive(row.get("bridge")):
        issues.append("bridge_movable_without_bridge")
    if clean(row.get("depth")) and mode != "underground":
        issues.append("depth_tag_not_underground")
    if clean(row.get("min_height")) and mode != "elevated":
        issues.append("min_height_tag_not_elevated")
    if level is not None and abs(level) > 1e-9 and mode == "surface":
        issues.append("nonzero_level_classified_surface")
    if clean(row.get("ele")):
        issues.append("ele_preserved_but_current_profile_ignores_it")
    if clean(row.get("level")):
        issues.append("level_preserved_but_current_vertical_logic_ignores_it")
    return issues


def audit_transport(frame: gpd.GeoDataFrame, name: str, out: Path, suspicious_gpkg: Path):
    frame = frame.copy()
    lengths = length_series(frame)
    frame["_length_m"] = lengths
    recomputed = frame.apply(vertical_mode_name, axis=1)
    frame["_recomputed_vertical_mode"] = recomputed

    total_length = float(lengths.fillna(0).sum())
    modes = []
    for mode in ["surface", "underground", "elevated", "unknown"]:
        mask = recomputed == mode
        km = float(lengths[mask].fillna(0).sum()) / 1000.0
        modes.append({
            "mode": mode,
            "features": int(mask.sum()),
            "length_km": km,
            "length_fraction": (km * 1000.0 / total_length) if total_length else 0.0,
        })
    pd.DataFrame(modes).to_csv(out / f"{name}-vertical-modes.csv", index=False)

    column_report = {}
    for column in frame.columns:
        if column in {frame.geometry.name, "_length_m"}:
            continue
        values = frame[column].map(clean)
        active = values != ""
        active_length = float(lengths[active].fillna(0).sum())
        column_report[column] = {
            "dtype": str(frame[column].dtype),
            "nonempty_features": int(active.sum()),
            "nonempty_fraction": float(active.mean()) if len(frame) else 0.0,
            "nonempty_length_km": active_length / 1000.0,
            "top_values": top_values(frame, lengths, column),
        }
    (out / f"{name}-columns-values.json").write_text(
        json.dumps(column_report, indent=2, default=str) + "\n"
    )

    combo_cols = (
        ["railway"] if name == "rail" else ["highway", "road_class"]
    ) + ["_recomputed_vertical_mode", "bridge", "tunnel", "location", "layer", "level"]
    combo = pd.DataFrame(index=frame.index)
    for column in combo_cols:
        combo[column] = (
            frame[column].map(clean).replace("", "<blank>")
            if column in frame.columns
            else "<missing>"
        )
    combo["_length_m"] = lengths.fillna(0)
    (
        combo.groupby(combo_cols, dropna=False)
        .agg(features=("_length_m", "size"), length_m=("_length_m", "sum"))
        .reset_index()
        .sort_values(["length_m", "features"], ascending=False)
        .head(500)
        .to_csv(out / f"{name}-vertical-combinations-top500.csv", index=False)
    )

    issue_rows = []
    for idx, row in frame.iterrows():
        for issue in issues_for(row):
            issue_rows.append({"index": idx, "issue": issue})
    issues = pd.DataFrame(issue_rows)

    if issues.empty:
        issue_summary = pd.DataFrame(columns=["issue", "features", "length_km"])
    else:
        unique = issues.drop_duplicates(["index", "issue"]).copy()
        unique["length_m"] = unique["index"].map(lengths.to_dict()).fillna(0)
        issue_summary = (
            unique.groupby("issue")
            .agg(features=("index", "nunique"), length_m=("length_m", "sum"))
            .reset_index()
            .sort_values(["length_m", "features"], ascending=False)
        )
        issue_summary["length_km"] = issue_summary.pop("length_m") / 1000.0
    issue_summary.to_csv(out / f"{name}-issues-summary.csv", index=False)

    if not issues.empty:
        grouped = (
            issues.groupby("index")["issue"]
            .apply(lambda x: ";".join(sorted(set(x))))
            .rename("_audit_issues")
        )
        suspicious = frame.loc[grouped.index].copy()
        suspicious["_audit_issues"] = grouped
        keep = [
            c for c in [
                "id", "osm_type", "name", "highway", "road_class", "railway",
                "vertical_mode", "_recomputed_vertical_mode", *VERTICAL_TAGS,
                "_length_m", "_audit_issues", frame.geometry.name
            ] if c in suspicious.columns
        ]
        suspicious = suspicious[keep]
        suspicious.to_file(
            suspicious_gpkg,
            layer=f"{name}_suspicious",
            driver="GPKG",
            engine="pyogrio",
            mode="a" if suspicious_gpkg.exists() else "w",
        )
        suspicious.drop(columns=[frame.geometry.name]).to_csv(
            out / f"{name}-suspicious-features.csv", index=False
        )

    if name == "rail" and "railway" in frame.columns:
        pd.crosstab(
            frame["railway"].map(clean).replace("", "<blank>"),
            recomputed,
        ).to_csv(out / "railway-by-vertical-mode.csv")
    if name == "roads" and "road_class" in frame.columns:
        pd.crosstab(
            frame["road_class"].map(clean).replace("", "<blank>"),
            recomputed,
        ).to_csv(out / "road-class-by-vertical-mode.csv")

    return {
        "features": int(len(frame)),
        "total_length_km": total_length / 1000.0,
        "modes": modes,
        "issues": issue_summary.to_dict(orient="records"),
        "columns": list(frame.columns),
    }


def audit_general(frame: gpd.GeoDataFrame, name: str, out: Path):
    result = {
        "features": int(len(frame)),
        "crs": str(frame.crs),
        "columns": list(frame.columns),
        "geometry_types": {
            str(k): int(v)
            for k, v in frame.geometry.geom_type.value_counts().items()
        },
    }
    values = {}
    dummy_lengths = pd.Series(np.nan, index=frame.index)
    for column in frame.columns:
        if column == frame.geometry.name:
            continue
        vals = frame[column].map(clean)
        values[column] = {
            "nonempty_features": int((vals != "").sum()),
            "top_values": top_values(frame, dummy_lengths, column, limit=30),
        }
    (out / f"{name}-columns-values.json").write_text(
        json.dumps(values, indent=2, default=str) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gpkg = args.gpkg.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    suspicious_gpkg = out / "suspicious-features.gpkg"
    if suspicious_gpkg.exists():
        suspicious_gpkg.unlink()

    available = [str(name) for name, _ in pyogrio.list_layers(gpkg)]
    metadata = read_city_metadata(gpkg)
    summary = {
        "file": str(gpkg),
        "size_bytes": gpkg.stat().st_size,
        "layers": available,
        "metadata": metadata,
        "layer_summaries": {},
    }

    for name in available:
        if name == "urban_metadata":
            continue
        frame = pyogrio.read_dataframe(gpkg, layer=name)
        if name in {"roads", "rail"}:
            summary["layer_summaries"][name] = audit_transport(
                frame, name, out, suspicious_gpkg
            )
        else:
            summary["layer_summaries"][name] = audit_general(frame, name, out)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    lines = [
        "SINGAPORE GPKG SOURCE AUDIT",
        "=" * 32,
        f"file: {gpkg}",
        f"layers: {', '.join(available)}",
        "",
    ]
    for name in ["roads", "rail"]:
        if name not in summary["layer_summaries"]:
            continue
        item = summary["layer_summaries"][name]
        lines += [
            name.upper(),
            f"features: {item['features']}",
            f"total length: {item['total_length_km']:.3f} km",
        ]
        for mode in item["modes"]:
            lines.append(
                f"  {mode['mode']:12s} {mode['length_km']:9.3f} km "
                f"({mode['length_fraction'] * 100:6.2f}%)"
            )
        lines.append("issues:")
        for issue in item["issues"][:30]:
            lines.append(
                f"  {issue['issue']:55s} "
                f"{int(issue['features']):5d} features "
                f"{float(issue['length_km']):9.3f} km"
            )
        lines.append("")

    report = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(report)
    print(report)
    print(f"Detailed audit: {out}")
    if suspicious_gpkg.exists():
        print(f"Suspicious feature map: {suspicious_gpkg}")


if __name__ == "__main__":
    main()
