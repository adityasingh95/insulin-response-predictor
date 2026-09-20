"""Data-quality assessment and privacy-local report generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .episodes import build_meal_episodes
from .schemas import SCHEMAS
from .validation import ValidationIssue, validate_dataset


def _table_summary(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"rows": int(len(frame))}
    if "timestamp" not in frame.columns or frame.empty:
        return result
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dropna()
    if not parsed.empty:
        result["first_timestamp"] = parsed.min().isoformat()
        result["last_timestamp"] = parsed.max().isoformat()
    return result


def assess_dataset(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, object], list[ValidationIssue], pd.DataFrame]:
    issues = validate_dataset(tables)
    error_codes = {issue.code for issue in issues if issue.severity == "error"}
    blocking = {"missing_table", "missing_columns", "invalid_timestamp", "naive_timestamp"}
    episodes = pd.DataFrame()
    has_required_tables = all(name in tables for name in ("glucose", "food", "insulin"))
    if not (error_codes & blocking) and has_required_tables:
        episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])

    issue_counts: dict[str, int] = {}
    for issue in issues:
        key = f"{issue.severity}:{issue.code}"
        issue_counts[key] = issue_counts.get(key, 0) + 1

    episode_counts = (
        {str(key): int(value) for key, value in episodes["status"].value_counts().items()}
        if not episodes.empty
        else {}
    )
    reason_counts = (
        {str(key): int(value) for key, value in episodes["reason"].value_counts().items()}
        if not episodes.empty
        else {}
    )
    total_episodes = int(len(episodes))
    clean_count = int(episode_counts.get("clean", 0))
    summary: dict[str, object] = {
        "tables": {name: _table_summary(frame) for name, frame in tables.items()},
        "missing_optional_context": "context" not in tables,
        "issues": issue_counts,
        "episode_counts": episode_counts,
        "episode_reasons": reason_counts,
        "clean_episode_rate": clean_count / total_episodes if total_episodes else None,
        "model_ready": not any(issue.severity == "error" for issue in issues)
        and clean_count >= 60,
    }
    return summary, issues, episodes


def render_markdown(summary: dict[str, object], issues: list[ValidationIssue]) -> str:
    lines = [
        "# Data quality assessment",
        "",
        "> Generated locally. This report may contain sensitive health-data summaries and "
        "must not be committed.",
        "",
        "## Verdict",
        "",
        f"**Model ready:** {'yes' if summary['model_ready'] else 'no'}",
        "",
        "## Tables",
        "",
        "| Table | Rows | First timestamp | Last timestamp |",
        "|---|---:|---|---|",
    ]
    tables = summary["tables"]
    assert isinstance(tables, dict)
    for name in SCHEMAS:
        table = tables.get(name, {})
        lines.append(
            f"| {name} | {table.get('rows', 0)} | {table.get('first_timestamp', '—')} | "
            f"{table.get('last_timestamp', '—')} |"
        )

    lines.extend(
        [
            "",
            "## Episode eligibility",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    episode_counts = summary["episode_counts"]
    assert isinstance(episode_counts, dict)
    for status in ("clean", "contaminated", "unusable"):
        lines.append(f"| {status} | {episode_counts.get(status, 0)} |")
    rate = summary["clean_episode_rate"]
    rate_text = (
        f"Clean episode rate: {rate:.1%}"
        if isinstance(rate, float)
        else "Clean episode rate: —"
    )
    lines.extend(["", rate_text])

    lines.extend(["", "## Validation issues", ""])
    if not issues:
        lines.append("No validation issues found.")
    else:
        lines.extend(
            [
                "| Severity | Table | Code | Row | Message |",
                "|---|---|---|---:|---|",
            ]
        )
        for issue in issues:
            message = issue.message.replace("|", "\\|")
            lines.append(
                f"| {issue.severity} | {issue.table} | {issue.code} | "
                f"{issue.row if issue.row is not None else '—'} | {message} |"
            )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            "`model_ready` requires zero validation errors and at least 60 clean episodes. "
            "Passing this",
            "gate only permits forward-model experimentation; it does not validate insulin "
            "recommendations.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_timeline(tables: dict[str, pd.DataFrame], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    glucose = tables["glucose"].copy()
    food = tables["food"].copy()
    insulin = tables["insulin"].copy()
    for frame in (glucose, food, insulin):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(glucose["timestamp"], glucose["glucose_mg_dl"], marker="o", linewidth=1)
    axes[0].set_ylabel("Glucose\n(mg/dL)")
    axes[0].grid(alpha=0.25)

    axes[1].vlines(food["timestamp"], 0, food["carbs_g"], color="#d9822b", linewidth=2)
    axes[1].scatter(food["timestamp"], food["carbs_g"], color="#d9822b", s=18)
    axes[1].set_ylabel("Carbs (g)")
    axes[1].grid(alpha=0.25)

    colors = insulin["insulin_type"].map({"rapid": "#2d72d2", "long": "#8f3985"})
    axes[2].vlines(insulin["timestamp"], 0, insulin["units"], color=colors, linewidth=2)
    axes[2].scatter(insulin["timestamp"], insulin["units"], color=colors, s=18)
    axes[2].set_ylabel("Insulin (U)")
    axes[2].set_xlabel("Time (UTC)")
    axes[2].grid(alpha=0.25)
    figure.suptitle("Glucose, meals, and insulin timeline — retrospective data")
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def write_assessment(
    tables: dict[str, pd.DataFrame], destination: str | Path
) -> tuple[dict[str, object], list[ValidationIssue], pd.DataFrame]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary, issues, episodes = assess_dataset(tables)
    (destination / "data_quality.md").write_text(render_markdown(summary, issues), encoding="utf-8")
    payload = {
        "summary": summary,
        "issues": [asdict(issue) for issue in issues],
    }
    (destination / "data_quality.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    if not episodes.empty:
        episodes.to_csv(destination / "episodes.csv", index=False)
    timestamp_errors = {"invalid_timestamp", "naive_timestamp"}
    can_plot = not any(issue.code in timestamp_errors for issue in issues)
    if can_plot and all(
        name in tables and not tables[name].empty for name in ("glucose", "food", "insulin")
    ):
        plot_timeline(tables, destination / "timeline.png")
    return summary, issues, episodes
