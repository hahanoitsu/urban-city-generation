from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .morphology import (
    MORPHOLOGY_CONTROL_FEATURES,
    MORPHOLOGY_FEATURES,
    describe_tile,
)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _feature_summary(frame: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        if values.empty:
            continue
        summary[name] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "p10": float(values.quantile(0.10)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
        }
    return summary


def _control_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    control = frame[features].apply(pd.to_numeric, errors="coerce").astype(float).copy()

    # Mode shares only have meaning when rail exists. Treating absent rail as a
    # zero underground/elevated share would make several PCA dimensions encode
    # the same rail-presence fact already captured by rail length.
    if "rail_present" in frame.columns:
        absent = pd.to_numeric(frame["rail_present"], errors="coerce").fillna(0) <= 0
        for name in ("rail_underground_share", "rail_elevated_share"):
            if name in control.columns:
                control.loc[absent, name] = np.nan

    return control


def _pca(
    frame: pd.DataFrame,
    features: list[str],
    components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").astype(float)
    medians = matrix.median(axis=0)
    matrix = matrix.fillna(medians)

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    usable = scales[scales > 1e-12].index.tolist()
    if not usable:
        return pd.DataFrame(index=frame.index), pd.DataFrame(), {
            "features": [],
            "explained_variance_ratio": [],
            "cumulative_explained_variance": [],
            "top_loadings": {},
        }

    values = (matrix[usable] - means[usable]) / scales[usable]
    u, singular, vt = np.linalg.svd(values.to_numpy(), full_matrices=False)
    count = min(max(1, int(components)), vt.shape[0])
    names = [f"PC{index + 1}" for index in range(count)]

    scores = pd.DataFrame(
        u[:, :count] * singular[:count],
        columns=names,
        index=frame.index,
    )
    loadings = pd.DataFrame(
        vt[:count].T,
        columns=names,
        index=usable,
    )

    variance = singular**2
    ratio = variance / variance.sum() if variance.sum() else np.zeros_like(variance)
    top_loadings: dict[str, list[dict[str, float | str]]] = {}
    for name in names:
        ordered = loadings[name].abs().sort_values(ascending=False).head(8).index
        top_loadings[name] = [
            {"feature": feature, "loading": float(loadings.loc[feature, name])}
            for feature in ordered
        ]

    details = {
        "features": usable,
        "explained_variance_ratio": [float(value) for value in ratio[:count]],
        "cumulative_explained_variance": [float(value) for value in np.cumsum(ratio[:count])],
        "top_loadings": top_loadings,
    }
    return scores, loadings, details


def analyse_manifest(
    manifest: str | Path,
    output: str | Path,
    *,
    pca_components: int = 5,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()

    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. Use --overwrite to replace it."
        )
    if overwrite and output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest_path)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    if not rows:
        raise ValueError("Manifest contains no tiles")

    descriptors: list[dict[str, Any]] = []
    for row in rows:
        sample_path = _resolve(manifest_path.parent, str(row["sample_path"]))
        metadata_path = _resolve(manifest_path.parent, str(row["metadata_path"]))
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with np.load(sample_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        descriptors.append(describe_tile(row, arrays, metadata))

    frame = pd.DataFrame(descriptors)
    frame.to_csv(output_path / "tiles.csv", index=False)

    evaluation_features = [name for name in MORPHOLOGY_FEATURES if name in frame.columns]
    control_features = [name for name in MORPHOLOGY_CONTROL_FEATURES if name in frame.columns]

    summary = _feature_summary(frame, evaluation_features)

    correlation = frame[evaluation_features].apply(pd.to_numeric, errors="coerce").corr()
    correlation.to_csv(output_path / "correlation.csv")

    control = _control_frame(frame, control_features)
    control_correlation = control.corr()
    control_correlation.to_csv(output_path / "control_correlation.csv")

    scores, loadings, pca = _pca(control, control_features, pca_components)
    identity = [
        name
        for name in ("tile_id", "city_id", "area_id", "split", "spatial_group")
        if name in frame.columns
    ]
    pd.concat([frame[identity], scores], axis=1).to_csv(
        output_path / "pca_scores.csv",
        index=False,
    )
    loadings.to_csv(output_path / "pca_loadings.csv", index_label="feature")

    city_summary = (
        frame.groupby("city_id", dropna=False)[evaluation_features]
        .mean(numeric_only=True)
        .reset_index()
    )
    city_summary.to_csv(output_path / "city_means.csv", index=False)

    result = {
        "analysis_version": 2,
        "manifest": str(manifest_path),
        "output": str(output_path),
        "tiles": int(len(frame)),
        "cities": {
            str(name): int(count)
            for name, count in frame["city_id"].value_counts().sort_index().items()
        },
        # Keep the old key for compatibility with the first analysis package.
        "features": evaluation_features,
        "evaluation_features": evaluation_features,
        "control_features": control_features,
        "summary": summary,
        "pca": {
            **pca,
            "conditional_values": {
                "rail_mode_shares": "missing for rail-absent tiles before median imputation"
            },
        },
        "files": {
            "tiles": str(output_path / "tiles.csv"),
            "correlation": str(output_path / "correlation.csv"),
            "control_correlation": str(output_path / "control_correlation.csv"),
            "pca_scores": str(output_path / "pca_scores.csv"),
            "pca_loadings": str(output_path / "pca_loadings.csv"),
            "city_means": str(output_path / "city_means.csv"),
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
