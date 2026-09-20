"""Exploratory diagnostics for episode coverage, variability, and data gaps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .episodes import build_meal_episodes
from .features import build_episode_features


def _comparable_noise_floor(frame: pd.DataFrame) -> dict[str, float | int | None]:
    differences: list[float] = []
    ordered = frame.sort_values("bolus_timestamp").reset_index(drop=True)
    for index, row in ordered.iterrows():
        earlier = ordered.iloc[:index]
        comparable = earlier.loc[
            (earlier["meal_type"] == row["meal_type"])
            & ((earlier["carbs_g"] - row["carbs_g"]).abs() <= 10)
            & ((earlier["bolus_units"] - row["bolus_units"]).abs() <= 1)
            & ((earlier["pre_glucose_mg_dl"] - row["pre_glucose_mg_dl"]).abs() <= 20)
        ]
        if not comparable.empty:
            closest = comparable.iloc[-1]
            differences.append(
                abs(float(row["outcome_glucose_mg_dl"]) - float(closest["outcome_glucose_mg_dl"]))
            )
    return {
        "comparable_pairs": len(differences),
        "median_absolute_outcome_difference_mg_dl": (
            float(np.median(differences)) if differences else None
        ),
        "interpretation": "descriptive repeatability estimate, not irreducible model error",
    }


def summarize_eda(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, object], pd.DataFrame]:
    episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
    features = build_episode_features(
        episodes,
        tables["glucose"],
        tables["food"],
        tables["insulin"],
        tables.get("context"),
    )
    missingness = {
        column: float(features[column].isna().mean())
        for column in features.columns
        if features[column].isna().any()
    }
    meal_summary = (
        features.groupby("meal_type", observed=True)["delta_glucose_mg_dl"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
        .to_dict(orient="index")
    )
    summary: dict[str, object] = {
        "episode_status": {
            str(key): int(value) for key, value in episodes["status"].value_counts().items()
        },
        "clean_episode_rows": int(len(features)),
        "meal_delta_summary_mg_dl": meal_summary,
        "outcome_delay_minutes": {
            "median": float(features["elapsed_minutes"].median()),
            "minimum": float(features["elapsed_minutes"].min()),
            "maximum": float(features["elapsed_minutes"].max()),
        },
        "missing_fraction": missingness,
        "recent_hypo_episode_fraction": float((features["hypo_events_last_24h"] > 0).mean()),
        "repeatability": _comparable_noise_floor(features),
    }
    return summary, features


def _render_markdown(summary: dict[str, object]) -> str:
    repeatability = summary["repeatability"]
    assert isinstance(repeatability, dict)
    noise = repeatability["median_absolute_outcome_difference_mg_dl"]
    noise_text = f"{noise:.1f} mg/dL" if isinstance(noise, float) else "not estimable"
    lines = [
        "# Exploratory data analysis",
        "",
        "> Descriptive research output only; this report does not assess treatment safety.",
        "",
        f"Clean episodes: **{summary['clean_episode_rows']}**",
        f"Comparable historical pairs: **{repeatability['comparable_pairs']}**",
        f"Median paired outcome difference: **{noise_text}**",
        "",
        "## Meal response",
        "",
        "| Meal | Rows | Mean delta | Median delta | Standard deviation |",
        "|---|---:|---:|---:|---:|",
    ]
    meal_summary = summary["meal_delta_summary_mg_dl"]
    assert isinstance(meal_summary, dict)
    for meal, values in meal_summary.items():
        lines.append(
            f"| {meal} | {values['count']:.0f} | {values['mean']:.1f} | "
            f"{values['median']:.1f} | {values['std']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Missing values",
            "",
            "Missing optional features are imputed inside each model training fold.",
            "",
            "| Feature | Missing fraction |",
            "|---|---:|",
        ]
    )
    missing = summary["missing_fraction"]
    assert isinstance(missing, dict)
    if missing:
        for column, fraction in missing.items():
            lines.append(f"| {column} | {fraction:.1%} |")
    else:
        lines.append("| — | 0% |")
    lines.append("")
    return "\n".join(lines)


def _plot(features: pd.DataFrame, destination: Path) -> None:
    meals = sorted(features["meal_type"].dropna().unique())
    groups = [
        features.loc[features["meal_type"] == meal, "delta_glucose_mg_dl"].to_numpy()
        for meal in meals
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(groups, tick_labels=meals, showfliers=False)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Outcome − pre-meal glucose (mg/dL)")
    axes[0].set_title("Response by meal")
    axes[1].scatter(features["carbs_g"], features["delta_glucose_mg_dl"], alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Recorded carbohydrate (g)")
    axes[1].set_ylabel("Glucose delta (mg/dL)")
    axes[1].set_title("Carbohydrate-response coverage")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_eda(
    tables: dict[str, pd.DataFrame], destination: str | Path
) -> tuple[dict[str, object], pd.DataFrame]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary, features = summarize_eda(tables)
    (destination / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (destination / "eda_report.md").write_text(
        _render_markdown(summary), encoding="utf-8"
    )
    _plot(features, destination / "meal_response.png")
    return summary, features
