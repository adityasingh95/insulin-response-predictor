"""Construct auditable meal/bolus/outcome episodes from irregular events."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .io import coerce_boolean


@dataclass(frozen=True)
class EpisodeConfig:
    pre_glucose_lookback_minutes: int = 90
    meal_bolus_tolerance_minutes: int = 30
    outcome_min_minutes: int = 120
    outcome_target_minutes: int = 180
    outcome_max_minutes: int = 240


def _as_time(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.sort_values("timestamp").reset_index(drop=True)


def build_meal_episodes(
    glucose: pd.DataFrame,
    food: pd.DataFrame,
    insulin: pd.DataFrame,
    *,
    config: EpisodeConfig | None = None,
) -> pd.DataFrame:
    """Create one episode per non-hypo meal with an associated rapid bolus.

    An episode is labelled ``clean`` only when no additional food or rapid
    insulin occurs after the anchor meal and before the selected outcome.
    """
    cfg = config or EpisodeConfig()
    glucose = _as_time(glucose)
    food = _as_time(food)
    insulin = _as_time(insulin)

    records: list[dict[str, object]] = []
    hypo_flags = food["is_hypo_treatment"].map(coerce_boolean).map(lambda value: value is True)
    meals = food.loc[
        ~hypo_flags.astype(bool)
        & food["meal_type"].isin(["breakfast", "lunch", "dinner"])
    ]

    for meal_index, meal in meals.iterrows():
        subject = meal["subject_id"]
        meal_time = meal["timestamp"]
        subject_glucose = glucose.loc[glucose["subject_id"] == subject]
        subject_insulin = insulin.loc[insulin["subject_id"] == subject]
        subject_food = food.loc[food["subject_id"] == subject]

        rapid = subject_insulin.loc[
            (subject_insulin["insulin_type"] == "rapid")
            & subject_insulin["dose_reason"].isin(["meal_bolus", "combined"])
        ].copy()
        rapid["distance"] = (rapid["timestamp"] - meal_time).abs()
        rapid = rapid.loc[
            rapid["distance"] <= pd.Timedelta(minutes=cfg.meal_bolus_tolerance_minutes)
        ]

        if rapid.empty:
            records.append(
                {
                    "subject_id": subject,
                    "meal_index": int(meal_index),
                    "meal_timestamp": meal_time,
                    "status": "unusable",
                    "reason": "missing_meal_bolus",
                }
            )
            continue

        bolus = rapid.sort_values(["distance", "timestamp"]).iloc[0]
        anchor_time = min(meal_time, bolus["timestamp"])

        prior = subject_glucose.loc[
            (subject_glucose["timestamp"] <= anchor_time)
            & (
                subject_glucose["timestamp"]
                >= anchor_time - pd.Timedelta(minutes=cfg.pre_glucose_lookback_minutes)
            )
        ]
        if prior.empty:
            records.append(
                {
                    "subject_id": subject,
                    "meal_index": int(meal_index),
                    "meal_timestamp": meal_time,
                    "status": "unusable",
                    "reason": "missing_pre_glucose",
                }
            )
            continue

        pre = prior.iloc[-1]
        outcome_start = anchor_time + pd.Timedelta(minutes=cfg.outcome_min_minutes)
        outcome_end = anchor_time + pd.Timedelta(minutes=cfg.outcome_max_minutes)
        outcomes = subject_glucose.loc[
            (subject_glucose["timestamp"] >= outcome_start)
            & (subject_glucose["timestamp"] <= outcome_end)
        ].copy()
        if outcomes.empty:
            records.append(
                {
                    "subject_id": subject,
                    "meal_index": int(meal_index),
                    "meal_timestamp": meal_time,
                    "status": "unusable",
                    "reason": "missing_outcome",
                }
            )
            continue

        target_time = anchor_time + pd.Timedelta(minutes=cfg.outcome_target_minutes)
        outcomes["distance"] = (outcomes["timestamp"] - target_time).abs()
        outcome = outcomes.sort_values(["distance", "timestamp"]).iloc[0]

        intervening_food = subject_food.loc[
            (subject_food["timestamp"] > meal_time)
            & (subject_food["timestamp"] < outcome["timestamp"])
        ]
        intervening_insulin = subject_insulin.loc[
            (subject_insulin["timestamp"] > bolus["timestamp"])
            & (subject_insulin["timestamp"] < outcome["timestamp"])
            & (subject_insulin["insulin_type"] == "rapid")
        ]
        reasons: list[str] = []
        if not intervening_food.empty:
            has_hypo_treatment = (
                intervening_food["is_hypo_treatment"]
                .map(coerce_boolean)
                .map(lambda value: value is True)
                .any()
            )
            reasons.append(
                "hypo_treatment" if has_hypo_treatment else "intervening_food"
            )
        if not intervening_insulin.empty:
            reasons.append("intervening_insulin")

        status = "clean" if not reasons else "contaminated"
        records.append(
            {
                "subject_id": subject,
                "meal_index": int(meal_index),
                "meal_timestamp": meal_time,
                "bolus_timestamp": bolus["timestamp"],
                "pre_glucose_timestamp": pre["timestamp"],
                "outcome_timestamp": outcome["timestamp"],
                "pre_glucose_mg_dl": float(pre["glucose_mg_dl"]),
                "outcome_glucose_mg_dl": float(outcome["glucose_mg_dl"]),
                "delta_glucose_mg_dl": float(outcome["glucose_mg_dl"])
                - float(pre["glucose_mg_dl"]),
                "elapsed_minutes": (
                    outcome["timestamp"] - anchor_time
                ).total_seconds()
                / 60,
                "carbs_g": float(meal["carbs_g"]),
                "meal_type": meal["meal_type"],
                "bolus_units": float(bolus["units"]),
                "status": status,
                "reason": "+".join(reasons) if reasons else "clean",
            }
        )

    return pd.DataFrame.from_records(records)
