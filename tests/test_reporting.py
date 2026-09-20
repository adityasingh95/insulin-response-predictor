from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from insulin_response_predictor.reporting import assess_dataset, write_assessment
from insulin_response_predictor.synthetic import generate_synthetic_dataset


class ReportingTests(unittest.TestCase):
    def test_assessment_counts_episodes(self) -> None:
        summary, issues, episodes = assess_dataset(generate_synthetic_dataset(days=7))
        self.assertFalse([issue for issue in issues if issue.severity == "error"])
        self.assertEqual(len(episodes), 21)
        self.assertGreater(summary["episode_counts"]["clean"], 0)

    def test_report_bundle_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_assessment(generate_synthetic_dataset(days=3), destination)
            self.assertTrue((destination / "data_quality.md").exists())
            self.assertTrue((destination / "data_quality.json").exists())
            self.assertTrue((destination / "episodes.csv").exists())
            self.assertTrue((destination / "timeline.png").exists())
            payload = json.loads((destination / "data_quality.json").read_text())
            self.assertIn("summary", payload)

    def test_naive_timestamps_block_episode_building(self) -> None:
        tables = generate_synthetic_dataset(days=1)
        tables["glucose"]["timestamp"] = tables["glucose"]["timestamp"].str.replace(
            "+00:00", "", regex=False
        )
        summary, issues, episodes = assess_dataset(tables)
        self.assertIn("naive_timestamp", {issue.code for issue in issues})
        self.assertTrue(episodes.empty)
        self.assertFalse(summary["model_ready"])


if __name__ == "__main__":
    unittest.main()
