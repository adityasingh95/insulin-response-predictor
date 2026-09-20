"""Chronological evaluation and safety gates for forward glucose models."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .episodes import build_meal_episodes
from .features import build_episode_features
from .models import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_models,
    linear_extrapolation_prediction,
    persistence_prediction,
)
from .splits import chronological_split


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    current_glucose: np.ndarray,
    persistence_rmse: float,
) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    actual_direction = np.sign(actual - current_glucose)
    predicted_direction = np.sign(predicted - current_glucose)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction))
    skill = 1 - rmse / persistence_rmse if persistence_rmse > 0 else float("nan")
    return {
        "rmse_mg_dl": float(rmse),
        "mae_mg_dl": float(mae),
        "skill_vs_persistence": float(skill),
        "directional_accuracy": directional_accuracy,
    }


def dose_sensitivity(
    fitted_model: object,
    frame: pd.DataFrame,
    *,
    support_min: float,
    support_max: float,
    step_units: float = 0.5,
) -> dict[str, float | int | None]:
    supported = frame.loc[
        (frame["bolus_units"] >= support_min + step_units)
        & (frame["bolus_units"] <= support_max - step_units)
    ].copy()
    if supported.empty:
        return {
            "rows": 0,
            "median_mg_dl_per_unit": None,
            "mean_mg_dl_per_unit": None,
            "negative_fraction": None,
        }
    lower = supported[MODEL_FEATURES].copy()
    upper = supported[MODEL_FEATURES].copy()
    lower["bolus_units"] -= step_units
    upper["bolus_units"] += step_units
    lower_prediction = np.asarray(fitted_model.predict(lower), dtype=float)
    upper_prediction = np.asarray(fitted_model.predict(upper), dtype=float)
    sensitivity = (upper_prediction - lower_prediction) / (2 * step_units)
    return {
        "rows": int(len(supported)),
        "median_mg_dl_per_unit": float(np.median(sensitivity)),
        "mean_mg_dl_per_unit": float(np.mean(sensitivity)),
        "negative_fraction": float(np.mean(sensitivity < 0)),
    }


def _passes_gate(metrics: dict[str, float], sensitivity: dict[str, object]) -> bool:
    median = sensitivity["median_mg_dl_per_unit"]
    negative_fraction = sensitivity["negative_fraction"]
    return bool(
        metrics["skill_vs_persistence"] > 0.20
        and isinstance(median, float)
        and median < 0
        and isinstance(negative_fraction, float)
        and negative_fraction >= 0.80
    )


def evaluate_forward_models(
    tables: dict[str, pd.DataFrame],
    *,
    train_fraction: float = 0.70,
    random_state: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
    features = build_episode_features(
        episodes, tables["glucose"], tables["food"], tables["insulin"]
    )
    train, test = chronological_split(features, train_fraction=train_fraction)
    actual = test[TARGET_COLUMN].to_numpy(dtype=float)
    current = test["pre_glucose_mg_dl"].to_numpy(dtype=float)
    persistence = persistence_prediction(test)
    persistence_rmse = math.sqrt(mean_squared_error(actual, persistence))

    predictions = test[
        [
            "subject_id",
            "bolus_timestamp",
            "meal_type",
            "pre_glucose_mg_dl",
            "outcome_glucose_mg_dl",
            "bolus_units",
            "carbs_g",
        ]
    ].copy()
    predictions["persistence"] = persistence
    predictions["linear_extrapolation"] = linear_extrapolation_prediction(test)

    metrics: dict[str, dict[str, float]] = {}
    metrics["persistence"] = regression_metrics(
        actual,
        predictions["persistence"].to_numpy(),
        current_glucose=current,
        persistence_rmse=persistence_rmse,
    )
    metrics["linear_extrapolation"] = regression_metrics(
        actual,
        predictions["linear_extrapolation"].to_numpy(),
        current_glucose=current,
        persistence_rmse=persistence_rmse,
    )

    support_min = float(train["bolus_units"].min())
    support_max = float(train["bolus_units"].max())
    sensitivities: dict[str, dict[str, object]] = {}
    gates: dict[str, bool] = {}
    fitted_models = build_models(random_state=random_state)
    for name, model in fitted_models.items():
        model.fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
        prediction = np.asarray(model.predict(test[MODEL_FEATURES]), dtype=float)
        predictions[name] = prediction
        metrics[name] = regression_metrics(
            actual,
            prediction,
            current_glucose=current,
            persistence_rmse=persistence_rmse,
        )
        sensitivities[name] = dose_sensitivity(
            model,
            test,
            support_min=support_min,
            support_max=support_max,
        )
        gates[name] = _passes_gate(metrics[name], sensitivities[name])

    learned_names = list(fitted_models)
    best_model = min(learned_names, key=lambda name: metrics[name]["rmse_mg_dl"])
    result: dict[str, object] = {
        "dataset": {
            "clean_episodes": int(len(features)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": pd.Timestamp(train["bolus_timestamp"].min()).isoformat(),
            "train_end": pd.Timestamp(train["bolus_timestamp"].max()).isoformat(),
            "test_start": pd.Timestamp(test["bolus_timestamp"].min()).isoformat(),
            "test_end": pd.Timestamp(test["bolus_timestamp"].max()).isoformat(),
            "observed_bolus_support_units": [support_min, support_max],
        },
        "metrics": metrics,
        "dose_sensitivity": sensitivities,
        "gate": {
            "criteria": {
                "minimum_skill_vs_persistence": 0.20,
                "median_dose_sensitivity_must_be_negative": True,
                "minimum_negative_sensitivity_fraction": 0.80,
            },
            "by_model": gates,
            "best_model_by_rmse": best_model,
            "best_model_passes": gates[best_model],
            "any_model_passes": any(gates.values()),
        },
    }
    return result, predictions


def _render_markdown(result: dict[str, object]) -> str:
    dataset = result["dataset"]
    metrics = result["metrics"]
    sensitivity = result["dose_sensitivity"]
    gate = result["gate"]
    assert isinstance(dataset, dict)
    assert isinstance(metrics, dict)
    assert isinstance(sensitivity, dict)
    assert isinstance(gate, dict)
    lines = [
        "# Forward-model evaluation",
        "",
        "> Retrospective research only. Passing this gate does not validate insulin "
        "recommendations.",
        "",
        "## Dataset",
        "",
        f"- Clean episodes: {dataset['clean_episodes']}",
        f"- Training rows: {dataset['train_rows']}",
        f"- Test rows: {dataset['test_rows']}",
        f"- Train period: {dataset['train_start']} to {dataset['train_end']}",
        f"- Test period: {dataset['test_start']} to {dataset['test_end']}",
        "",
        "## Results",
        "",
        "| Model | RMSE | MAE | Skill | Direction | Median dose sensitivity | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    gate_by_model = gate["by_model"]
    assert isinstance(gate_by_model, dict)
    for name, model_metrics in metrics.items():
        model_sensitivity = sensitivity.get(name, {})
        median = model_sensitivity.get("median_mg_dl_per_unit")
        median_text = f"{median:.2f}" if isinstance(median, float) else "—"
        lines.append(
            f"| {name} | {model_metrics['rmse_mg_dl']:.2f} | "
            f"{model_metrics['mae_mg_dl']:.2f} | "
            f"{model_metrics['skill_vs_persistence']:.3f} | "
            f"{model_metrics['directional_accuracy']:.1%} | {median_text} | "
            f"{'pass' if gate_by_model.get(name, False) else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Best learned model by RMSE: **{gate['best_model_by_rmse']}**.",
            f"Forward-model gate: **{'PASS' if gate['any_model_passes'] else 'STOP'}**.",
            "",
            "A pass only permits retrospective policy experimentation. A stop means the data "
            "does not",
            "support a dosing layer; it is a valid project result.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_predictions(predictions: pd.DataFrame, destination: Path) -> None:
    model_columns = [
        column
        for column in predictions.columns
        if column
        not in {
            "subject_id",
            "bolus_timestamp",
            "meal_type",
            "pre_glucose_mg_dl",
            "outcome_glucose_mg_dl",
            "bolus_units",
            "carbs_g",
        }
    ]
    figure, axes = plt.subplots(1, len(model_columns), figsize=(4 * len(model_columns), 4))
    axes_array = np.atleast_1d(axes)
    actual = predictions["outcome_glucose_mg_dl"]
    low = float(min(actual.min(), predictions[model_columns].min().min()))
    high = float(max(actual.max(), predictions[model_columns].max().max()))
    for axis, name in zip(axes_array, model_columns, strict=True):
        axis.scatter(actual, predictions[name], alpha=0.7, s=22)
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Actual mg/dL")
        axis.set_ylabel("Predicted mg/dL")
        axis.grid(alpha=0.2)
    figure.suptitle("Chronological test-set predictions")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_forward_evaluation(
    tables: dict[str, pd.DataFrame], destination: str | Path
) -> tuple[dict[str, object], pd.DataFrame]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result, predictions = evaluate_forward_models(tables)
    (destination / "forward_metrics.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (destination / "forward_report.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    predictions.to_csv(destination / "forward_predictions.csv", index=False)
    _plot_predictions(predictions, destination / "predicted_vs_actual.png")
    return result, predictions
