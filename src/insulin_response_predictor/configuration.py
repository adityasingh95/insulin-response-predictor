"""Load explicit subject configuration without silent clinical defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

from .policy import PolicyConfig


def load_policy_config(path: str | Path) -> PolicyConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Subject configuration not found: {path}. Copy config/subject.example.yaml "
            "to the ignored config/subject.yaml and review every value."
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = payload["target"]
    dosing = payload["dosing"]
    icr = dosing["usual_icr_g_per_unit"]
    return PolicyConfig(
        target_mg_dl=float(target["midpoint_mg_dl"]),
        target_low_mg_dl=float(target["low_mg_dl"]),
        target_high_mg_dl=float(target["high_mg_dl"]),
        standard_isf_mg_dl_per_unit=float(dosing["usual_isf_mg_dl_per_unit"]),
        breakfast_icr_g_per_unit=float(icr["breakfast"]),
        lunch_icr_g_per_unit=float(icr["lunch"]),
        dinner_icr_g_per_unit=float(icr["dinner"]),
        dose_increment_units=float(dosing["increment_units"]),
    )
