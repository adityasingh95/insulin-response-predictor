from __future__ import annotations

import unittest

import pandas as pd

from insulin_response_predictor.splits import chronological_split, rolling_origin_splits


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bolus_timestamp": pd.date_range(
                "2026-01-01", periods=rows, freq="6h", tz="UTC"
            ),
            "value": range(rows),
        }
    )


class SplitTests(unittest.TestCase):
    def test_chronological_split_never_overlaps(self) -> None:
        train, test = chronological_split(_frame(), train_fraction=0.70)
        self.assertLess(train["bolus_timestamp"].max(), test["bolus_timestamp"].min())
        self.assertEqual(len(train) + len(test), 40)

    def test_rolling_origin_expands_training_window(self) -> None:
        splits = rolling_origin_splits(_frame(), initial_train_rows=20, test_rows=5)
        self.assertEqual([len(train) for train, _ in splits], [20, 25, 30, 35])
        self.assertTrue(all(len(test) == 5 for _, test in splits))

    def test_too_little_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chronological_split(_frame(12))


if __name__ == "__main__":
    unittest.main()
