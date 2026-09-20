"""Generate deterministic, fictional records for development and testing."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

SYNTHETIC_SCENARIOS = {"routine", "identifiable", "noisy"}


def generate_synthetic_dataset(
    days: int = 21, seed: int = 42, scenario: str = "routine"
) -> dict[str, pd.DataFrame]:
    if days <= 0:
        raise ValueError("days must be positive")
    if scenario not in SYNTHETIC_SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(SYNTHETIC_SCENARIOS)}")
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    glucose_rows: list[dict[str, object]] = []
    food_rows: list[dict[str, object]] = []
    insulin_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    subject = "synthetic-subject"
    meal_specs = [(8, "breakfast", 45), (13, "lunch", 60), (20, "dinner", 65)]

    for day in range(days):
        date = start + timedelta(days=day)
        basal_time = date.replace(hour=22, minute=0)
        insulin_rows.append(
            {
                "subject_id": subject,
                "timestamp": basal_time.isoformat(),
                "units": 18.0,
                "insulin_type": "long",
                "product": "synthetic-basal",
                "dose_reason": "basal",
            }
        )

        for meal_number, (hour, meal_type, base_carbs) in enumerate(meal_specs):
            meal_time = date.replace(hour=hour, minute=rng.choice([0, 5, 10]))
            pre_spread = 32 if scenario == "identifiable" else 22
            carb_spread = 16 if scenario == "identifiable" else 10
            pre_glucose = max(55, min(280, 120 + rng.gauss(0, pre_spread)))
            carbs = max(15, round(base_carbs + rng.gauss(0, carb_spread)))
            icr = {"breakfast": 8, "lunch": 10, "dinner": 9}[meal_type]
            formula_dose = carbs / icr + (pre_glucose - 120) / 40
            if scenario == "identifiable":
                dose = max(0.5, round((formula_dose + rng.gauss(0, 1.5)) * 2) / 2)
                meal_effect = {"breakfast": 8.0, "lunch": 0.0, "dinner": 4.0}[meal_type]
                outcome = (
                    pre_glucose
                    + 1.25 * carbs
                    - 12.0 * dose
                    + meal_effect
                    + rng.gauss(0, 7)
                )
            else:
                dose = max(0.5, round(formula_dose * 2) / 2)
                noise = 35 if scenario == "noisy" else 18
                outcome = pre_glucose + 0.9 * carbs - 8.0 * dose + rng.gauss(0, noise)

            pre_lead_minutes = rng.randint(5, 25)
            pre_time = meal_time - timedelta(minutes=pre_lead_minutes)
            bolus_time = meal_time - timedelta(
                minutes=rng.randint(0, min(15, pre_lead_minutes - 1))
            )
            outcome_time = meal_time + timedelta(minutes=rng.randint(150, 210))

            glucose_rows.extend(
                [
                    {
                        "subject_id": subject,
                        "timestamp": pre_time.isoformat(),
                        "glucose_mg_dl": round(pre_glucose),
                        "source": "synthetic_fingerstick",
                        "context": "pre_meal",
                    },
                    {
                        "subject_id": subject,
                        "timestamp": outcome_time.isoformat(),
                        "glucose_mg_dl": round(max(35, min(400, outcome))),
                        "source": "synthetic_fingerstick",
                        "context": "post_meal",
                    },
                ]
            )
            food_rows.append(
                {
                    "subject_id": subject,
                    "timestamp": meal_time.isoformat(),
                    "description": f"synthetic {meal_type}",
                    "carbs_g": carbs,
                    "meal_type": meal_type,
                    "gi_class": "medium",
                    "is_hypo_treatment": False,
                }
            )
            insulin_rows.append(
                {
                    "subject_id": subject,
                    "timestamp": bolus_time.isoformat(),
                    "units": dose,
                    "insulin_type": "rapid",
                    "product": "synthetic-rapid",
                    "dose_reason": "combined" if pre_glucose > 150 else "meal_bolus",
                }
            )

            # Deterministically contaminate a small number of episodes.
            if (day * 3 + meal_number) % 13 == 0:
                snack_time = meal_time + timedelta(minutes=75)
                food_rows.append(
                    {
                        "subject_id": subject,
                        "timestamp": snack_time.isoformat(),
                        "description": "synthetic snack",
                        "carbs_g": 15,
                        "meal_type": "snack",
                        "gi_class": "high",
                        "is_hypo_treatment": False,
                    }
                )

        if day % 7 == 5:
            context_rows.append(
                {
                    "subject_id": subject,
                    "timestamp": date.replace(hour=17).isoformat(),
                    "event_type": "exercise",
                    "duration_minutes": 45,
                    "intensity": "moderate",
                }
            )

    return {
        "glucose": pd.DataFrame(glucose_rows),
        "food": pd.DataFrame(food_rows),
        "insulin": pd.DataFrame(insulin_rows),
        "context": pd.DataFrame(
            context_rows,
            columns=[
                "subject_id",
                "timestamp",
                "event_type",
                "duration_minutes",
                "intensity",
            ],
        ),
    }


def write_synthetic_dataset(
    destination: str | Path,
    days: int = 21,
    seed: int = 42,
    scenario: str = "routine",
) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in generate_synthetic_dataset(
        days=days, seed=seed, scenario=scenario
    ).items():
        frame.to_csv(destination / f"{name}.csv", index=False)
