from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from insulin_response_predictor.io import (
    apply_meal_references,
    coerce_boolean,
    load_csv_exports,
    normalize_export,
    write_blank_templates,
)


class IoTests(unittest.TestCase):
    def test_false_string_is_not_treated_as_true(self) -> None:
        self.assertFalse(coerce_boolean("FALSE"))
        self.assertTrue(coerce_boolean("yes"))
        self.assertIsNone(coerce_boolean("sometimes"))

    def test_google_style_headers_are_normalized(self) -> None:
        frame = pd.DataFrame(
            {
                "Subject ID": ["s1"],
                "Date Time": ["2026-01-01T08:00:00+05:30"],
                "Blood Glucose": [120],
            }
        )
        result = normalize_export(frame)
        self.assertEqual(list(result.columns), ["subject_id", "timestamp", "glucose_mg_dl"])

    def test_templates_round_trip_through_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_blank_templates(directory)
            self.assertEqual(len(paths), 4)
            tables = load_csv_exports(directory)
            self.assertEqual(set(tables), {"glucose", "food", "insulin", "context"})
            self.assertTrue(all(frame.empty for frame in tables.values()))

    def test_utf8_bom_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glucose.csv"
            path.write_text(
                "\ufeffSubject ID,Timestamp,Glucose,Source,Context\n"
                "s1,2026-01-01T08:00:00+05:30,120,fingerstick,pre_meal\n",
                encoding="utf-8",
            )
            tables = load_csv_exports(directory)
            self.assertEqual(float(tables["glucose"].iloc[0]["glucose_mg_dl"]), 120)

    def test_meal_reference_fills_only_blank_details(self) -> None:
        food = pd.DataFrame(
            {
                "meal_reference_id": ["usual_lunch"],
                "description": [""],
                "carbs_g": [pd.NA],
                "gi_class": ["medium"],
            }
        )
        references = pd.DataFrame(
            {
                "meal_reference_id": ["usual_lunch"],
                "description": ["reference lunch"],
                "carbs_g": [55],
                "gi_class": ["low"],
            }
        )
        result = apply_meal_references(food, references)
        self.assertEqual(result.loc[0, "description"], "reference lunch")
        self.assertEqual(result.loc[0, "carbs_g"], 55)
        self.assertEqual(result.loc[0, "gi_class"], "medium")


if __name__ == "__main__":
    unittest.main()
