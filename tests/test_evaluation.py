from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from insulin_response_predictor.evaluation import (
    dose_sensitivity,
    evaluate_forward_models,
    regression_metrics,
    write_forward_evaluation,
)
from insulin_response_predictor.models import MODEL_FEATURES
from insulin_response_predictor.synthetic import generate_synthetic_dataset


class _KnownDoseEffectModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return 180 - 10 * frame["bolus_units"].to_numpy(dtype=float)


class EvaluationTests(unittest.TestCase):
    def test_perfect_prediction_has_full_skill(self) -> None:
        actual = np.array([100.0, 130.0, 160.0])
        current = np.array([110.0, 110.0, 110.0])
        metrics = regression_metrics(
            actual,
            actual,
            current_glucose=current,
            persistence_rmse=20.0,
        )
        self.assertEqual(metrics["rmse_mg_dl"], 0.0)
        self.assertEqual(metrics["skill_vs_persistence"], 1.0)
        self.assertEqual(metrics["directional_accuracy"], 1.0)

    def test_dose_sensitivity_recovers_known_sign_and_magnitude(self) -> None:
        frame = pd.DataFrame({feature: [0.0, 0.0] for feature in MODEL_FEATURES})
        frame["meal_type"] = "lunch"
        frame["bolus_units"] = [5.0, 7.0]
        sensitivity = dose_sensitivity(
            _KnownDoseEffectModel(), frame, support_min=0, support_max=12
        )
        self.assertAlmostEqual(sensitivity["median_mg_dl_per_unit"], -10.0)
        self.assertEqual(sensitivity["negative_fraction"], 1.0)

    def test_end_to_end_forward_evaluation(self) -> None:
        tables = generate_synthetic_dataset(days=45)
        result, predictions = evaluate_forward_models(tables)
        self.assertGreater(result["dataset"]["train_rows"], result["dataset"]["test_rows"])
        self.assertIn("ridge", result["metrics"])
        self.assertIn("any_model_passes", result["gate"])
        self.assertGreater(len(predictions), 5)

    def test_forward_report_bundle_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_forward_evaluation(generate_synthetic_dataset(days=45), destination)
            self.assertTrue((destination / "forward_metrics.json").exists())
            self.assertTrue((destination / "forward_report.md").exists())
            self.assertTrue((destination / "forward_predictions.csv").exists())
            self.assertTrue((destination / "predicted_vs_actual.png").exists())


if __name__ == "__main__":
    unittest.main()
