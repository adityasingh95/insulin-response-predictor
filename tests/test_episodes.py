from __future__ import annotations

import unittest

from insulin_response_predictor.episodes import build_meal_episodes
from insulin_response_predictor.synthetic import generate_synthetic_dataset


class EpisodeTests(unittest.TestCase):
    def test_builds_clean_and_contaminated_episodes(self) -> None:
        tables = generate_synthetic_dataset(days=7)
        episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
        self.assertGreater(len(episodes), 0)
        self.assertIn("clean", set(episodes["status"]))
        self.assertIn("contaminated", set(episodes["status"]))

    def test_selected_outcomes_stay_in_window(self) -> None:
        tables = generate_synthetic_dataset(days=3)
        episodes = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
        eligible = episodes[episodes["status"].isin(["clean", "contaminated"])]
        self.assertTrue(eligible["elapsed_minutes"].between(120, 240).all())


if __name__ == "__main__":
    unittest.main()
