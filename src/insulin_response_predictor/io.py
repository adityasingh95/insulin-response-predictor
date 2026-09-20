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
    return tables


def write_blank_templates(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in SCHEMAS.items():
        path = directory / f"{name}.csv"
        pd.DataFrame(columns=[*schema.required, *schema.optional]).to_csv(path, index=False)
        written.append(path)
    return written
