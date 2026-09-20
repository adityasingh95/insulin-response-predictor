from __future__ import annotations

import unittest

from insulin_response_predictor.episodes import build_meal_episodes
from insulin_response_predictor.features import build_episode_features
from insulin_response_predictor.synthetic import generate_synthetic_dataset


class FeatureTests(unittest.TestCase):
    def test_future_glucose_cannot_change_features(self) -> None:
        tables = generate_synthetic_dataset(days=4)
        episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
        baseline = build_episode_features(
            episodes, tables["glucose"], tables["food"], tables["insulin"]
        )
        tables["glucose"].loc[
            tables["glucose"]["timestamp"] == tables["glucose"]["timestamp"].max(),
            "glucose_mg_dl",
        ] = 599
        changed = build_episode_features(
            episodes, tables["glucose"], tables["food"], tables["insulin"]
        )
        columns = ["iob_rapid_units", "cob_prior_g", "hour_sin", "hour_cos", "is_weekend"]
        self.assertTrue(baseline[columns].equals(changed[columns]))


if __name__ == "__main__":
    unittest.main()
