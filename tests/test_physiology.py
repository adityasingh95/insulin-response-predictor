from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from insulin_response_predictor.physiology import (
    carbs_on_board,
    carbs_remaining_fraction,
    insulin_on_board,
    rapid_iob_fraction,
)


class PhysiologyTests(unittest.TestCase):
    def test_iob_starts_full_and_finishes_zero(self) -> None:
        self.assertAlmostEqual(rapid_iob_fraction(0), 1.0)
        self.assertEqual(rapid_iob_fraction(300), 0.0)

    def test_iob_is_monotonically_decreasing(self) -> None:
        values = [rapid_iob_fraction(age) for age in range(0, 301, 15)]
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:], strict=True)))

    def test_future_events_are_ignored(self) -> None:
        now = datetime(2026, 1, 1, 12, tzinfo=UTC)
        future = now + timedelta(minutes=10)
        self.assertEqual(insulin_on_board(now, [(future, 5.0)]), 0.0)
        self.assertEqual(carbs_on_board(now, [(future, 40.0)]), 0.0)

    def test_carbs_curve_boundaries(self) -> None:
        self.assertEqual(carbs_remaining_fraction(0), 1.0)
        self.assertEqual(carbs_remaining_fraction(240), 0.0)


if __name__ == "__main__":
    unittest.main()
