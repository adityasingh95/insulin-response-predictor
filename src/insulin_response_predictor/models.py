"""Forward glucose models and non-learned baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "outcome_glucose_mg_dl"
NUMERIC_FEATURES = [
    "pre_glucose_mg_dl",
    "carbs_g",
    "bolus_units",
    "elapsed_minutes",
    "iob_rapid_units",
    "cob_prior_g",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "prior_glucose_slope_mg_dl_h",
    "protein_g",
    "fat_g",
    "correction_units_last_4h",
    "minutes_since_rapid_dose",
    "hours_since_basal_dose",
    "hypo_events_last_24h",
    "exercise_minutes_last_24h",
]
CATEGORICAL_FEATURES = ["meal_type", "gi_class"]
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def persistence_prediction(frame: pd.DataFrame) -> np.ndarray:
    return frame["pre_glucose_mg_dl"].to_numpy(dtype=float)


def linear_extrapolation_prediction(frame: pd.DataFrame) -> np.ndarray:
    current = persistence_prediction(frame)
    slope = frame["prior_glucose_slope_mg_dl_h"].fillna(0).to_numpy(dtype=float)
    horizon_hours = frame["elapsed_minutes"].to_numpy(dtype=float) / 60
    return current + slope * horizon_hours


def _preprocessor(*, scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "meal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("features", _preprocessor(scale=True)),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("features", _preprocessor(scale=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=150,
                        max_leaf_nodes=8,
                        min_samples_leaf=12,
                        l2_regularization=2.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("features", _preprocessor(scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=5,
                        max_features=0.8,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
