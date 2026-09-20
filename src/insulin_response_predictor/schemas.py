"""Canonical tabular schemas and enums.

The schemas intentionally avoid patient names and other direct identifiers. Each
record belongs to a pseudonymous ``subject_id``.
"""

from __future__ import annotations

from dataclasses import dataclass


GLUCOSE_CONTEXTS = {
    "fasting",
    "pre_meal",
    "post_meal",
    "bedtime",
    "symptomatic",
    "other",
}
MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack", "hypo_treatment"}
GI_CLASSES = {"low", "medium", "high", "unknown"}
INSULIN_TYPES = {"rapid", "long"}
DOSE_REASONS = {"meal_bolus", "correction", "basal", "combined"}
EVENT_TYPES = {"exercise", "illness", "stress", "poor_sleep", "alcohol", "other"}


@dataclass(frozen=True)
class TableSchema:
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


SCHEMAS: dict[str, TableSchema] = {
    "glucose": TableSchema(
        required=("subject_id", "timestamp", "glucose_mg_dl", "source", "context"),
        optional=("recorded_at", "notes"),
    ),
    "food": TableSchema(
        required=(
            "subject_id",
            "timestamp",
            "description",
            "carbs_g",
            "meal_type",
            "gi_class",
            "is_hypo_treatment",
        ),
        optional=("protein_g", "fat_g", "recorded_at", "estimate_confidence", "notes"),
    ),
    "insulin": TableSchema(
        required=(
            "subject_id",
            "timestamp",
            "units",
            "insulin_type",
            "product",
            "dose_reason",
        ),
        optional=(
            "meal_units",
            "correction_units",
            "injection_site",
            "recorded_at",
            "notes",
        ),
    ),
    "context": TableSchema(
        required=("subject_id", "timestamp", "event_type"),
        optional=("duration_minutes", "intensity", "recorded_at", "notes"),
    ),
}
