from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from insulin_response_predictor.pipeline import run_demo


class PipelineTests(unittest.TestCase):
    def test_demo_exercises_complete_and_stop_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_demo(root, days=45)
            statuses = {
                name: manifest["status"]
                for name, manifest in result["scenarios"].items()
            }
            self.assertEqual(
                statuses,
                {
                    "identifiable": "complete",
                    "noisy": "stopped_at_forward_gate",
                },
            )
            self.assertTrue((root / "demo_manifest.json").exists())
            self.assertTrue(
                (
                    root
                    / "identifiable"
                    / "output"
                    / "04_policy_experiment"
                    / "policy_metrics.json"
                ).exists()
            )
            self.assertFalse(
                (root / "noisy" / "output" / "04_policy_experiment").exists()
            )
            policy_root = root / "identifiable" / "output" / "04_policy_experiment"
            metrics = json.loads((policy_root / "policy_metrics.json").read_text())
            self.assertIn("historical_imitation", metrics["policies"])
            self.assertTrue(
                (
                    root
                    / "identifiable"
                    / "output"
                    / "02_exploratory_analysis"
                    / "eda_report.md"
                ).exists()
            )
            for values in metrics["policies"].values():
                self.assertEqual(values["outside_historical_support_count"], 0)
            candidates = pd.read_csv(policy_root / "policy_candidates.csv")
            dose_columns = [
                column for column in candidates if column.endswith("candidate_units")
            ]
            for column in dose_columns:
                doubled = candidates[column].dropna() * 2
                self.assertTrue((doubled.round() == doubled).all())


if __name__ == "__main__":
    unittest.main()
