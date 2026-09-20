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
) -> pd.DataFrame:
    glucose = glucose.copy()
    food = food.copy()
    insulin = insulin.copy()
    for frame in (glucose, food, insulin):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

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
            }
        )
    return pd.DataFrame.from_records(records)
