from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from insulin_response_predictor.eda import summarize_eda, write_eda
from insulin_response_predictor.synthetic import generate_synthetic_dataset


class EdaTests(unittest.TestCase):
    def test_summary_and_artifacts_are_created(self) -> None:
        tables = generate_synthetic_dataset(days=20)
        summary, features = summarize_eda(tables)
        self.assertEqual(summary["clean_episode_rows"], len(features))
        self.assertIn("repeatability", summary)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_eda(tables, destination)
            self.assertTrue((destination / "eda_report.md").exists())
            self.assertTrue((destination / "eda_summary.json").exists())
            self.assertTrue((destination / "meal_response.png").exists())


if __name__ == "__main__":
    unittest.main()
