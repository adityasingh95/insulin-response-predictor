from __future__ import annotations

import unittest

from insulin_response_predictor.synthetic import generate_synthetic_dataset
from insulin_response_predictor.validation import validate_dataset


class ValidationTests(unittest.TestCase):
    def test_synthetic_dataset_has_no_errors(self) -> None:
        issues = validate_dataset(generate_synthetic_dataset(days=2))
        errors = [issue for issue in issues if issue.severity == "error"]
        self.assertEqual(errors, [])

    def test_impossible_glucose_is_rejected(self) -> None:
        tables = generate_synthetic_dataset(days=1)
        tables["glucose"].loc[0, "glucose_mg_dl"] = 999
        issues = validate_dataset(tables)
        self.assertIn("implausible_glucose", {issue.code for issue in issues})


if __name__ == "__main__":
    unittest.main()
