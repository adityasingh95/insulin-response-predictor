"""CSV export loading and blank template generation."""

from __future__ import annotations

import re
from numbers import Real
from pathlib import Path

import pandas as pd

from .schemas import SCHEMAS

COLUMN_ALIASES = {
    "date_time": "timestamp",
    "datetime": "timestamp",
    "blood_glucose": "glucose_mg_dl",
    "glucose": "glucose_mg_dl",
    "glucose_value": "glucose_mg_dl",
    "mg_dl": "glucose_mg_dl",
    "carbohydrates": "carbs_g",
    "carbs": "carbs_g",
    "carbs_grams": "carbs_g",
    "dose": "units",
    "insulin_units": "units",
    "hypo_treatment": "is_hypo_treatment",
}


def normalize_column_name(value: object) -> str:
    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return COLUMN_ALIASES.get(name, name)


def coerce_boolean(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    return None


def normalize_export(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    normalized = [normalize_column_name(column) for column in result.columns]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Columns collide after normalization: {duplicates}")
    result.columns = normalized
    result = result.dropna(how="all").reset_index(drop=True)
    if "is_hypo_treatment" in result.columns:
        result["is_hypo_treatment"] = result["is_hypo_treatment"].map(coerce_boolean)
    return result


def load_csv_exports(directory: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    tables: dict[str, pd.DataFrame] = {}
    for name in SCHEMAS:
        path = directory / f"{name}.csv"
        if path.exists():
            tables[name] = normalize_export(pd.read_csv(path, encoding="utf-8-sig"))
    reference_path = directory / "meal_references.csv"
    if reference_path.exists() and "food" in tables:
        references = normalize_export(pd.read_csv(reference_path, encoding="utf-8-sig"))
        tables["food"] = apply_meal_references(tables["food"], references)
    return tables


def apply_meal_references(food: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Fill blank food details from a user-maintained repeated-meal lookup."""
    required = {"meal_reference_id", "description", "carbs_g", "gi_class"}
    missing = required - set(references.columns)
    if missing:
        raise ValueError(f"meal_references.csv is missing columns: {sorted(missing)}")
    if references["meal_reference_id"].duplicated().any():
        raise ValueError("meal_reference_id values must be unique")
    if "meal_reference_id" not in food:
        return food
    details = ["description", "carbs_g", "gi_class", "protein_g", "fat_g"]
    available = [column for column in details if column in references]
    lookup = references[["meal_reference_id", *available]].rename(
        columns={column: f"{column}_reference" for column in available}
    )
    result = food.merge(lookup, on="meal_reference_id", how="left", validate="many_to_one")
    for column in available:
        reference_column = f"{column}_reference"
        if column not in result:
            result[column] = result[reference_column]
        else:
            blank = result[column].isna() | result[column].astype(str).str.strip().eq("")
            result.loc[blank, column] = result.loc[blank, reference_column]
        result = result.drop(columns=reference_column)
    return result


def write_blank_templates(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in SCHEMAS.items():
        path = directory / f"{name}.csv"
        pd.DataFrame(columns=[*schema.required, *schema.optional]).to_csv(path, index=False)
        written.append(path)
    return written
