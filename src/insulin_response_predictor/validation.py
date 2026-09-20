"""Deterministic validation for imported health-event tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schemas import (
    DOSE_REASONS,
    EVENT_TYPES,
    GI_CLASSES,
    GLUCOSE_CONTEXTS,
    INSULIN_TYPES,
    MEAL_TYPES,
    SCHEMAS,
)


@dataclass(frozen=True)
class ValidationIssue:
    table: str
    severity: str
    code: str
    message: str
    row: int | None = None


def _require_columns(name: str, frame: pd.DataFrame) -> list[ValidationIssue]:
    missing = [column for column in SCHEMAS[name].required if column not in frame.columns]
    if not missing:
        return []
    return [
        ValidationIssue(name, "error", "missing_columns", f"Missing columns: {missing}")
    ]


def _enum_issues(
    name: str, frame: pd.DataFrame, column: str, allowed: set[str]
) -> list[ValidationIssue]:
    if column not in frame.columns:
        return []
    issues: list[ValidationIssue] = []
    for index, value in frame[column].items():
        if pd.notna(value) and str(value) not in allowed:
            issues.append(
                ValidationIssue(
                    name,
                    "error",
                    "invalid_enum",
                    f"{column}={value!r}; expected one of {sorted(allowed)}",
                    int(index) if isinstance(index, int) else None,
                )
            )
    return issues


def validate_table(name: str, frame: pd.DataFrame) -> list[ValidationIssue]:
    if name not in SCHEMAS:
        raise KeyError(f"Unknown table {name!r}")

    issues = _require_columns(name, frame)
    if issues:
        return issues

    parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for index in frame.index[parsed.isna()]:
        issues.append(
            ValidationIssue(name, "error", "invalid_timestamp", "Unparseable timestamp", int(index))
        )

    duplicate_mask = frame.duplicated(subset=["subject_id", "timestamp"], keep=False)
    for index in frame.index[duplicate_mask]:
        issues.append(
            ValidationIssue(name, "warning", "duplicate_timestamp", "Duplicate subject timestamp", int(index))
        )

    if name == "glucose":
        values = pd.to_numeric(frame["glucose_mg_dl"], errors="coerce")
        bad = values.isna() | (values < 20) | (values > 600)
        for index in frame.index[bad]:
            issues.append(
                ValidationIssue(name, "error", "implausible_glucose", "Expected 20–600 mg/dL", int(index))
            )
        issues.extend(_enum_issues(name, frame, "context", GLUCOSE_CONTEXTS))

    elif name == "food":
        carbs = pd.to_numeric(frame["carbs_g"], errors="coerce")
        bad = carbs.isna() | (carbs < 0) | (carbs > 300)
        for index in frame.index[bad]:
            issues.append(
                ValidationIssue(name, "error", "implausible_carbs", "Expected 0–300 g", int(index))
            )
        issues.extend(_enum_issues(name, frame, "meal_type", MEAL_TYPES))
        issues.extend(_enum_issues(name, frame, "gi_class", GI_CLASSES))
        mismatched = frame["is_hypo_treatment"].astype(bool) != (
            frame["meal_type"] == "hypo_treatment"
        )
        for index in frame.index[mismatched]:
            issues.append(
                ValidationIssue(
                    name,
                    "error",
                    "hypo_flag_mismatch",
                    "meal_type and is_hypo_treatment disagree",
                    int(index),
                )
            )

    elif name == "insulin":
        units = pd.to_numeric(frame["units"], errors="coerce")
        bad = units.isna() | (units <= 0) | (units > 100)
        for index in frame.index[bad]:
            issues.append(
                ValidationIssue(name, "error", "implausible_insulin", "Expected >0–100 units", int(index))
            )
        issues.extend(_enum_issues(name, frame, "insulin_type", INSULIN_TYPES))
        issues.extend(_enum_issues(name, frame, "dose_reason", DOSE_REASONS))

    elif name == "context":
        issues.extend(_enum_issues(name, frame, "event_type", EVENT_TYPES))

    return issues


def validate_dataset(tables: dict[str, pd.DataFrame]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in SCHEMAS:
        if name == "context" and name not in tables:
            continue
        if name not in tables:
            issues.append(ValidationIssue(name, "error", "missing_table", "Table not supplied"))
            continue
        issues.extend(validate_table(name, tables[name]))
    return issues
