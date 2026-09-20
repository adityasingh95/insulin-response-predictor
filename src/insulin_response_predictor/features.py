"""Leakage-resistant episode features computed strictly as of the anchor time."""

from __future__ import annotations

import math

import pandas as pd

from .physiology import carbs_on_board, insulin_on_board


def build_episode_features(
    episodes: pd.DataFrame,
    glucose: pd.DataFrame,
    food: pd.DataFrame,
    insulin: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    glucose = glucose.copy()
    food = food.copy()
    insulin = insulin.copy()
    for frame in (glucose, food, insulin):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    context = context.copy() if context is not None else pd.DataFrame()
    if not context.empty:
        context["timestamp"] = pd.to_datetime(context["timestamp"], utc=True)

    records: list[dict[str, object]] = []
    for _, episode in episodes.loc[episodes["status"] == "clean"].iterrows():
        subject = episode["subject_id"]
        anchor = pd.Timestamp(episode["bolus_timestamp"])
        prior_glucose = glucose.loc[
            (glucose["subject_id"] == subject) & (glucose["timestamp"] <= anchor)
        ].sort_values("timestamp")
        prior_rapid = insulin.loc[
            (insulin["subject_id"] == subject)
            & (insulin["timestamp"] < anchor)
            & (insulin["insulin_type"] == "rapid")
        ]
        prior_food = food.loc[
            (food["subject_id"] == subject) & (food["timestamp"] < anchor)
        ]
        prior_long = insulin.loc[
            (insulin["subject_id"] == subject)
            & (insulin["timestamp"] < anchor)
            & (insulin["insulin_type"] == "long")
        ].sort_values("timestamp")
        recent_context = context.loc[
            (context.get("subject_id") == subject)
            & (context.get("timestamp") < anchor)
            & (context.get("timestamp") >= anchor - pd.Timedelta(hours=24))
        ] if not context.empty else pd.DataFrame()

        rapid_doses = [
            (row.timestamp.to_pydatetime(), float(row.units))
            for row in prior_rapid.itertuples()
        ]
        meals = [
            (row.timestamp.to_pydatetime(), float(row.carbs_g))
            for row in prior_food.itertuples()
        ]
        hour = anchor.hour + anchor.minute / 60
        previous = prior_glucose.iloc[-2:] if len(prior_glucose) >= 2 else prior_glucose
        slope = math.nan
        if len(previous) == 2:
            gap_hours = (
                previous.iloc[-1]["timestamp"] - previous.iloc[-2]["timestamp"]
            ).total_seconds() / 3600
            if gap_hours > 0:
                slope = (
                    float(previous.iloc[-1]["glucose_mg_dl"])
                    - float(previous.iloc[-2]["glucose_mg_dl"])
                ) / gap_hours

        records.append(
            {
                **episode.to_dict(),
                "iob_rapid_units": insulin_on_board(anchor.to_pydatetime(), rapid_doses),
                "cob_prior_g": carbs_on_board(anchor.to_pydatetime(), meals),
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "is_weekend": int(anchor.dayofweek >= 5),
                "prior_glucose_slope_mg_dl_h": slope,
                "correction_units_last_4h": _recent_corrections(prior_rapid, anchor),
                "minutes_since_rapid_dose": _minutes_since(prior_rapid, anchor),
                "hours_since_basal_dose": _minutes_since(prior_long, anchor) / 60,
                "hypo_events_last_24h": _recent_hypo_count(
                    glucose, food, subject, anchor
                ),
                "exercise_minutes_last_24h": _exercise_minutes(recent_context),
            }
        )
    return pd.DataFrame.from_records(records)


def _minutes_since(frame: pd.DataFrame, anchor: pd.Timestamp) -> float:
    if frame.empty:
        return math.nan
    return float((anchor - frame["timestamp"].max()).total_seconds() / 60)


def _recent_corrections(prior_rapid: pd.DataFrame, anchor: pd.Timestamp) -> float:
    recent = prior_rapid.loc[
        prior_rapid["timestamp"] >= anchor - pd.Timedelta(hours=4)
    ]
    total = 0.0
    for row in recent.itertuples():
        reason = getattr(row, "dose_reason", "")
        correction = getattr(row, "correction_units", math.nan)
        if pd.notna(correction):
            total += float(correction)
        elif reason == "correction":
            total += float(row.units)
    return total


def _recent_hypo_count(
    glucose: pd.DataFrame,
    food: pd.DataFrame,
    subject: object,
    anchor: pd.Timestamp,
) -> int:
    start = anchor - pd.Timedelta(hours=24)
    glucose_count = len(
        glucose.loc[
            (glucose["subject_id"] == subject)
            & (glucose["timestamp"] < anchor)
            & (glucose["timestamp"] >= start)
            & (glucose["glucose_mg_dl"] < 70)
        ]
    )
    flags = food.get("is_hypo_treatment", pd.Series(False, index=food.index))
    normalized_flags = flags.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if pd.notna(value)
        else False
    )
    food_count = len(
        food.loc[
            (food["subject_id"] == subject)
            & (food["timestamp"] < anchor)
            & (food["timestamp"] >= start)
            & normalized_flags
        ]
    )
    return int(glucose_count + food_count)


def _exercise_minutes(context: pd.DataFrame) -> float:
    if context.empty or "event_type" not in context:
        return 0.0
    exercise = context.loc[context["event_type"] == "exercise"]
    if exercise.empty or "duration_minutes" not in exercise:
        return 0.0
    return float(pd.to_numeric(exercise["duration_minutes"], errors="coerce").fillna(0).sum())
