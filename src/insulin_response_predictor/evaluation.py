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
from .splits import chronological_split, rolling_origin_splits


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


def bootstrap_metric_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    current_glucose: np.ndarray,
    persistence_rmse: float,
    samples: int = 300,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Return deterministic episode-level 95% bootstrap intervals."""
    rng = np.random.default_rng(random_state)
    collected: dict[str, list[float]] = {
        "rmse_mg_dl": [],
        "mae_mg_dl": [],
        "skill_vs_persistence": [],
        "directional_accuracy": [],
    }
    for _ in range(samples):
        indices = rng.integers(0, len(actual), len(actual))
        values = regression_metrics(
            actual[indices],
            predicted[indices],
            current_glucose=current_glucose[indices],
            persistence_rmse=persistence_rmse,
        )
        for name in collected:
            collected[name].append(values[name])
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in collected.items()
    }


def _rolling_origin_evaluation(
    features: pd.DataFrame, *, random_state: int
) -> dict[str, object]:
    initial = max(20, int(len(features) * 0.50))
    test_rows = max(5, int(len(features) * 0.15))
    splits = rolling_origin_splits(
        features, initial_train_rows=initial, test_rows=test_rows
    )
    fold_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits, start=1):
        actual = test[TARGET_COLUMN].to_numpy(dtype=float)
        current = test["pre_glucose_mg_dl"].to_numpy(dtype=float)
        persistence = persistence_prediction(test)
        baseline_rmse = math.sqrt(mean_squared_error(actual, persistence))
        fold_rows.append(
            {
                "fold": fold,
                "model": "persistence",
                **regression_metrics(
                    actual,
                    persistence,
                    current_glucose=current,
                    persistence_rmse=baseline_rmse,
                ),
            }
        )
        for name, model in build_models(random_state=random_state + fold).items():
            model.fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
            prediction = np.asarray(model.predict(test[MODEL_FEATURES]), dtype=float)
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    **regression_metrics(
                        actual,
                        prediction,
                        current_glucose=current,
                        persistence_rmse=baseline_rmse,
                    ),
                }
            )
    fold_frame = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {}
    for name, group in fold_frame.groupby("model"):
        summary[str(name)] = {
            "folds": int(len(group)),
            "mean_rmse_mg_dl": float(group["rmse_mg_dl"].mean()),
            "std_rmse_mg_dl": float(group["rmse_mg_dl"].std(ddof=0)),
            "mean_skill_vs_persistence": float(group["skill_vs_persistence"].mean()),
        }
    return {"fold_count": len(splits), "models": summary, "folds": fold_rows}


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
        episodes,
        tables["glucose"],
        tables["food"],
        tables["insulin"],
        tables.get("context"),
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
    intervals: dict[str, dict[str, list[float]]] = {}
    metrics["persistence"] = regression_metrics(
        actual,
        predictions["persistence"].to_numpy(),
        current_glucose=current,
        persistence_rmse=persistence_rmse,
    )
    intervals["persistence"] = bootstrap_metric_intervals(
        actual,
        predictions["persistence"].to_numpy(),
        current_glucose=current,
        persistence_rmse=persistence_rmse,
        random_state=random_state,
    )
    metrics["linear_extrapolation"] = regression_metrics(
        actual,
        predictions["linear_extrapolation"].to_numpy(),
        current_glucose=current,
        persistence_rmse=persistence_rmse,
    )
    intervals["linear_extrapolation"] = bootstrap_metric_intervals(
        actual,
        predictions["linear_extrapolation"].to_numpy(),
        current_glucose=current,
        persistence_rmse=persistence_rmse,
        random_state=random_state + 1,
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
        intervals[name] = bootstrap_metric_intervals(
            actual,
            prediction,
            current_glucose=current,
            persistence_rmse=persistence_rmse,
            random_state=random_state + len(intervals),
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
        "metric_95_percent_intervals": intervals,
        "rolling_origin": _rolling_origin_evaluation(
            features, random_state=random_state
        ),
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
    intervals = result["metric_95_percent_intervals"]
    rolling = result["rolling_origin"]
    assert isinstance(dataset, dict)
    assert isinstance(metrics, dict)
    assert isinstance(sensitivity, dict)
    assert isinstance(gate, dict)
    assert isinstance(intervals, dict)
    assert isinstance(rolling, dict)
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
        "| Model | RMSE (95% CI) | MAE | Skill | Direction | Median dose sensitivity | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    gate_by_model = gate["by_model"]
    assert isinstance(gate_by_model, dict)
    for name, model_metrics in metrics.items():
        model_sensitivity = sensitivity.get(name, {})
        median = model_sensitivity.get("median_mg_dl_per_unit")
        median_text = f"{median:.2f}" if isinstance(median, float) else "—"
        rmse_interval = intervals[name]["rmse_mg_dl"]
        lines.append(
            f"| {name} | {model_metrics['rmse_mg_dl']:.2f} "
            f"({rmse_interval[0]:.2f}–{rmse_interval[1]:.2f}) | "
            f"{model_metrics['mae_mg_dl']:.2f} | "
            f"{model_metrics['skill_vs_persistence']:.3f} | "
            f"{model_metrics['directional_accuracy']:.1%} | {median_text} | "
            f"{'pass' if gate_by_model.get(name, False) else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Rolling-origin robustness",
            "",
            f"Expanding-window folds: **{rolling['fold_count']}**.",
            "",
            "| Model | Mean RMSE | RMSE SD | Mean skill |",
            "|---|---:|---:|---:|",
        ]
    )
    rolling_models = rolling["models"]
    assert isinstance(rolling_models, dict)
    for name, values in rolling_models.items():
        lines.append(
            f"| {name} | {values['mean_rmse_mg_dl']:.2f} | "
            f"{values['std_rmse_mg_dl']:.2f} | "
            f"{values['mean_skill_vs_persistence']:.3f} |"
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


def _plot_residual_diagnostics(
    predictions: pd.DataFrame, best_model: str, destination: Path
) -> None:
    residual = predictions["outcome_glucose_mg_dl"] - predictions[best_model]
    timestamps = pd.to_datetime(predictions["bolus_timestamp"], utc=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(timestamps.dt.hour, residual, alpha=0.65)
    axes[0].set_xlabel("Bolus hour")
    axes[0].set_ylabel("Actual − predicted (mg/dL)")
    axes[0].set_title("Residual by time of day")
    axes[1].scatter(predictions["carbs_g"], residual, alpha=0.65)
    axes[1].set_xlabel("Carbohydrate (g)")
    axes[1].set_ylabel("Actual − predicted (mg/dL)")
    axes[1].set_title("Residual by meal size")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.grid(alpha=0.2)
    figure.suptitle(best_model.replace("_", " "))
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
    gate = result["gate"]
    assert isinstance(gate, dict)
    _plot_residual_diagnostics(
        predictions,
        str(gate["best_model_by_rmse"]),
        destination / "residual_diagnostics.png",
    )
    return result, predictions
