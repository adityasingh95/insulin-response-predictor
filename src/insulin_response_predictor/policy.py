"""Gated retrospective policy experiments.

This module produces candidate doses for historical evaluation only. It is not a
treatment calculator and must never be called unless the forward-model gate passes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .episodes import build_meal_episodes
from .features import build_episode_features
from .models import MODEL_FEATURES, TARGET_COLUMN, build_models
from .splits import chronological_split


@dataclass(frozen=True)
class PolicyConfig:
    target_mg_dl: float = 120
    target_low_mg_dl: float = 80
    target_high_mg_dl: float = 180
    standard_isf_mg_dl_per_unit: float = 40
    breakfast_icr_g_per_unit: float = 8
    lunch_icr_g_per_unit: float = 10
    dinner_icr_g_per_unit: float = 9
    dose_increment_units: float = 0.5
    low_loss_multiplier: float = 8

    def __post_init__(self) -> None:
        if not (
            self.target_low_mg_dl < self.target_mg_dl < self.target_high_mg_dl
        ):
            raise ValueError("Target midpoint must sit inside the target band")
        positive = (
            self.standard_isf_mg_dl_per_unit,
            self.breakfast_icr_g_per_unit,
            self.lunch_icr_g_per_unit,
            self.dinner_icr_g_per_unit,
            self.dose_increment_units,
            self.low_loss_multiplier,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Policy configuration values must be positive")

    def icr_for(self, meal_type: str) -> float:
        return {
            "breakfast": self.breakfast_icr_g_per_unit,
            "lunch": self.lunch_icr_g_per_unit,
            "dinner": self.dinner_icr_g_per_unit,
        }[meal_type]


def _quantize(value: float, increment: float) -> float:
    return round(value / increment) * increment


def fit_formula_parameters(
    train: pd.DataFrame, config: PolicyConfig
) -> dict[str, float | int]:
    good = train[TARGET_COLUMN].between(
        config.target_low_mg_dl, config.target_high_mg_dl
    )
    fitting = train.loc[good].copy()
    if len(fitting) < 20:
        raise ValueError("Need at least 20 in-range training episodes to fit ICR/ISF")
    design = np.column_stack(
        [
            fitting["carbs_g"].to_numpy(dtype=float),
            (fitting["pre_glucose_mg_dl"] - config.target_mg_dl).to_numpy(dtype=float),
        ]
    )
    response = (
        fitting["bolus_units"].to_numpy(dtype=float)
        + fitting["iob_rapid_units"].to_numpy(dtype=float)
    )
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(design, response)
    carb_coefficient, correction_coefficient = model.coef_
    icr = 1 / carb_coefficient if carb_coefficient > 1e-6 else 10.0
    isf = 1 / correction_coefficient if correction_coefficient > 1e-6 else 40.0
    return {
        "rows": int(len(fitting)),
        "icr_g_per_unit": float(np.clip(icr, 3, 30)),
        "isf_mg_dl_per_unit": float(np.clip(isf, 10, 150)),
    }


def _meal_support(train: pd.DataFrame, meal_type: str) -> dict[str, float]:
    comparable = train.loc[train["meal_type"] == meal_type]
    return {
        "dose_min": float(comparable["bolus_units"].min()),
        "dose_max": float(comparable["bolus_units"].max()),
        "glucose_min": float(comparable["pre_glucose_mg_dl"].min()),
        "glucose_max": float(comparable["pre_glucose_mg_dl"].max()),
        "carbs_min": float(comparable["carbs_g"].min()),
        "carbs_max": float(comparable["carbs_g"].max()),
        "iob_min": float(comparable["iob_rapid_units"].min()),
        "iob_max": float(comparable["iob_rapid_units"].max()),
    }


def _abstention_reason(
    row: pd.Series, support: dict[str, float], config: PolicyConfig
) -> str | None:
    if float(row["pre_glucose_mg_dl"]) < config.target_low_mg_dl:
        return "current_glucose_below_floor"
    checks = {
        "glucose": float(row["pre_glucose_mg_dl"]),
        "carbs": float(row["carbs_g"]),
        "iob": float(row["iob_rapid_units"]),
    }
    for name, value in checks.items():
        if value < support[f"{name}_min"] or value > support[f"{name}_max"]:
            return f"outside_{name}_support"
    return None


def _constrain_dose(value: float, support: dict[str, float], increment: float) -> float:
    bounded = float(np.clip(value, support["dose_min"], support["dose_max"]))
    return float(_quantize(bounded, increment))


def _formula_candidate(
    row: pd.Series,
    *,
    icr: float,
    isf: float,
    support: dict[str, float],
    config: PolicyConfig,
) -> float:
    raw = (
        float(row["carbs_g"]) / icr
        + (float(row["pre_glucose_mg_dl"]) - config.target_mg_dl) / isf
        - float(row["iob_rapid_units"])
    )
    return _constrain_dose(max(0.0, raw), support, config.dose_increment_units)


def _predict_candidate(model: object, row: pd.Series, dose: float) -> float:
    candidate = row[MODEL_FEATURES].to_frame().T.copy()
    candidate["bolus_units"] = dose
    return float(model.predict(candidate)[0])


def _imitation_candidate(
    imitation_model: object,
    row: pd.Series,
    support: dict[str, float],
    config: PolicyConfig,
) -> float:
    candidate = row[MODEL_FEATURES].to_frame().T.copy()
    candidate["bolus_units"] = 0.0
    historical_style_dose = float(imitation_model.predict(candidate)[0])
    return _constrain_dose(
        max(0.0, historical_style_dose), support, config.dose_increment_units
    )


def _inverted_candidate(
    model: object,
    row: pd.Series,
    support: dict[str, float],
    config: PolicyConfig,
) -> tuple[float, float]:
    start = math.ceil(support["dose_min"] / config.dose_increment_units)
    stop = math.floor(support["dose_max"] / config.dose_increment_units)
    doses = np.arange(start, stop + 1, dtype=float) * config.dose_increment_units
    candidates = pd.concat([row[MODEL_FEATURES].to_frame().T] * len(doses), ignore_index=True)
    candidates["bolus_units"] = doses
    predicted = np.asarray(model.predict(candidates), dtype=float)
    squared_error = (predicted - config.target_mg_dl) ** 2
    weights = np.where(predicted < config.target_low_mg_dl, config.low_loss_multiplier, 1.0)
    best = int(np.argmin(squared_error * weights))
    return float(doses[best]), float(predicted[best])


def _policy_metrics(
    frame: pd.DataFrame, policy_name: str, config: PolicyConfig
) -> dict[str, float | int | None]:
    dose_column = f"{policy_name}_candidate_units"
    predicted_column = f"{policy_name}_predicted_glucose_mg_dl"
    valid = frame[dose_column].notna()
    evaluated = frame.loc[valid]
    abstentions = int((~valid).sum())
    if evaluated.empty:
        return {
            "evaluated_rows": 0,
            "abstentions": abstentions,
            "good_outcome_dose_mae_units": None,
            "bad_outcome_direction_accuracy": None,
            "optimistic_simulated_time_in_range": None,
            "predicted_low_count": 0,
            "outside_historical_support_count": 0,
        }

    good = evaluated[TARGET_COLUMN].between(
        config.target_low_mg_dl, config.target_high_mg_dl
    )
    good_mae = (
        float(
            np.mean(
                np.abs(
                    evaluated.loc[good, dose_column]
                    - evaluated.loc[good, "bolus_units"]
                )
            )
        )
        if good.any()
        else None
    )
    high = evaluated[TARGET_COLUMN] > config.target_high_mg_dl
    low = evaluated[TARGET_COLUMN] < config.target_low_mg_dl
    bad = high | low
    correct_direction = (high & (evaluated[dose_column] > evaluated["bolus_units"])) | (
        low & (evaluated[dose_column] < evaluated["bolus_units"])
    )
    direction_accuracy = float(correct_direction[bad].mean()) if bad.any() else None
    predicted = evaluated[predicted_column]
    outside_support = (evaluated[dose_column] < evaluated["historical_dose_min_units"]) | (
        evaluated[dose_column] > evaluated["historical_dose_max_units"]
    )
    return {
        "evaluated_rows": int(len(evaluated)),
        "abstentions": abstentions,
        "good_outcome_dose_mae_units": good_mae,
        "bad_outcome_direction_accuracy": direction_accuracy,
        "optimistic_simulated_time_in_range": float(
            predicted.between(config.target_low_mg_dl, config.target_high_mg_dl).mean()
        ),
        "predicted_low_count": int((predicted < config.target_low_mg_dl).sum()),
        "outside_historical_support_count": int(outside_support.sum()),
    }


def evaluate_retrospective_policies(
    tables: dict[str, pd.DataFrame],
    forward_result: dict[str, object],
    *,
    config: PolicyConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    config = config or PolicyConfig()
    gate = forward_result["gate"]
    assert isinstance(gate, dict)
    if not gate["any_model_passes"]:
        raise ValueError("Forward-model gate did not pass")

    episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
    features = build_episode_features(
        episodes,
        tables["glucose"],
        tables["food"],
        tables["insulin"],
        tables.get("context"),
    )
    train, test = chronological_split(features)
    gate_by_model = gate["by_model"]
    metrics = forward_result["metrics"]
    assert isinstance(gate_by_model, dict)
    assert isinstance(metrics, dict)
    passing = [name for name, passed in gate_by_model.items() if passed]
    selected_name = min(passing, key=lambda name: metrics[name]["rmse_mg_dl"])
    selected_model = build_models()[selected_name]
    selected_model.fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
    imitation_model = build_models()["ridge"]
    imitation_features = train[MODEL_FEATURES].copy()
    imitation_features["bolus_units"] = 0.0
    imitation_model.fit(imitation_features, train["bolus_units"])
    fitted = fit_formula_parameters(train, config)

    evaluated = test.copy()
    evaluated["historical_dose_min_units"] = np.nan
    evaluated["historical_dose_max_units"] = np.nan
    policy_names = [
        "standard_formula",
        "fitted_formula",
        "historical_imitation",
        "model_inversion",
    ]
    for name in policy_names:
        evaluated[f"{name}_candidate_units"] = np.nan
        evaluated[f"{name}_predicted_glucose_mg_dl"] = np.nan
        evaluated[f"{name}_abstention_reason"] = None

    for index, row in evaluated.iterrows():
        support = _meal_support(train, str(row["meal_type"]))
        evaluated.at[index, "historical_dose_min_units"] = support["dose_min"]
        evaluated.at[index, "historical_dose_max_units"] = support["dose_max"]
        reason = _abstention_reason(row, support, config)
        if reason:
            for name in policy_names:
                evaluated.at[index, f"{name}_abstention_reason"] = reason
            continue

        standard = _formula_candidate(
            row,
            icr=config.icr_for(str(row["meal_type"])),
            isf=config.standard_isf_mg_dl_per_unit,
            support=support,
            config=config,
        )
        fitted_formula = _formula_candidate(
            row,
            icr=float(fitted["icr_g_per_unit"]),
            isf=float(fitted["isf_mg_dl_per_unit"]),
            support=support,
            config=config,
        )
        inverted, inverted_prediction = _inverted_candidate(
            selected_model, row, support, config
        )
        imitation = _imitation_candidate(imitation_model, row, support, config)
        for name, dose in (
            ("standard_formula", standard),
            ("fitted_formula", fitted_formula),
            ("historical_imitation", imitation),
            ("model_inversion", inverted),
        ):
            evaluated.at[index, f"{name}_candidate_units"] = dose
            evaluated.at[index, f"{name}_predicted_glucose_mg_dl"] = (
                inverted_prediction
                if name == "model_inversion"
                else _predict_candidate(selected_model, row, dose)
            )

    result: dict[str, object] = {
        "selected_forward_model": selected_name,
        "policy_config": asdict(config),
        "fitted_formula_parameters": fitted,
        "policies": {
            name: _policy_metrics(evaluated, name, config) for name in policy_names
        },
        "limitations": [
            "All policy results are retrospective and counterfactual.",
            "Simulated time in range is optimistic because the selected forward model "
            "scores its own policy.",
            "Candidate doses are clamped to meal-specific historical support.",
            "The imitation policy reproduces recorded dosing patterns; it does not infer an "
            "optimal or clinically appropriate dose.",
            "This output is not a treatment instruction.",
        ],
    }
    return result, evaluated


def _render_policy_markdown(result: dict[str, object]) -> str:
    policies = result["policies"]
    assert isinstance(policies, dict)
    lines = [
        "# Retrospective policy experiment",
        "",
        "> Research only. Candidate doses in this report must not be used for treatment.",
        "",
        f"Selected forward model: **{result['selected_forward_model']}**",
        "",
        "| Policy | Evaluated | Abstained | Good-outcome dose MAE | Bad-outcome direction | "
        "Optimistic TIR | Predicted lows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in policies.items():
        good_mae = values["good_outcome_dose_mae_units"]
        direction = values["bad_outcome_direction_accuracy"]
        tir = values["optimistic_simulated_time_in_range"]
        good_mae_text = f"{good_mae:.2f}" if isinstance(good_mae, float) else "—"
        direction_text = f"{direction:.1%}" if isinstance(direction, float) else "—"
        tir_text = f"{tir:.1%}" if isinstance(tir, float) else "—"
        lines.append(
            f"| {name} | {values['evaluated_rows']} | {values['abstentions']} | "
            f"{good_mae_text} | {direction_text} | {tir_text} | "
            f"{values['predicted_low_count']} |"
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_policy_evaluation(
    tables: dict[str, pd.DataFrame],
    forward_result: dict[str, object],
    destination: str | Path,
    *,
    config: PolicyConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result, evaluated = evaluate_retrospective_policies(
        tables, forward_result, config=config
    )
    (destination / "policy_metrics.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (destination / "policy_report.md").write_text(
        _render_policy_markdown(result), encoding="utf-8"
    )
    columns = [
        "subject_id",
        "bolus_timestamp",
        "meal_type",
        "pre_glucose_mg_dl",
        TARGET_COLUMN,
        "carbs_g",
        "bolus_units",
        "historical_dose_min_units",
        "historical_dose_max_units",
    ] + [
        column
        for column in evaluated.columns
        if "candidate_units" in column
        or "predicted_glucose_mg_dl" in column
        or "abstention_reason" in column
    ]
    evaluated[columns].to_csv(destination / "policy_candidates.csv", index=False)
    return result, evaluated
