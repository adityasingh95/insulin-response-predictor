from __future__ import annotations

import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def _load_notebook(self) -> dict:
        path = Path(__file__).parents[1] / "notebooks" / "end_to_end.ipynb"
        return json.loads(path.read_text(encoding="utf-8"))

    def _source(self, notebook: dict) -> str:
        return "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    def test_guided_notebook_is_valid_and_defaults_to_fictional_data(self) -> None:
        path = Path(__file__).parents[1] / "notebooks" / "end_to_end.ipynb"
        notebook = self._load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        source = self._source(notebook)
        self.assertIn("MODE = 'fictional'", source)
        self.assertIn("run_pipeline", source)
        self.assertIn("Research only", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(
                line for line in cell["source"] if not line.lstrip().startswith("%")
            )
            compile(code, str(path), "exec")

    def test_notebook_explains_scope_for_non_technical_readers(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("What this notebook does", source)
        self.assertIn("What this notebook does not do", source)
        self.assertIn("does not provide a dose for the next meal", source)

    def test_notebook_explains_settings_and_input_data(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("What most users should choose", source)
        self.assertIn("meal_reference_id", source)
        self.assertIn("Mandatory?", source)

    def test_notebook_explains_validation_results(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("How to read this result", source)
        self.assertIn("missing timezone", source)
        self.assertIn("fix the source spreadsheet", source)

    def test_notebook_explains_episodes_and_exploratory_analysis(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("Clean", source)
        self.assertIn("Contaminated", source)
        self.assertIn("Unusable", source)
        self.assertIn("repeatability estimate", source)

    def test_notebook_explains_models_and_metrics(self) -> None:
        source = self._source(self._load_notebook())
        for model_name in ("Persistence", "Linear extrapolation", "Ridge", "Random Forest"):
            self.assertIn(model_name, source)
        for term in ("RMSE", "MAE", "Directional accuracy", "Rolling-origin"):
            self.assertIn(term, source)

    def test_notebook_explains_pass_stop_gate(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("PASS or STOP", source)
        self.assertIn("not approval for real-world dosing", source)
        self.assertIn("STOP is a valid", source)

    def test_notebook_explains_diagnostic_charts(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("near the diagonal", source)
        self.assertIn("not sufficient validation", source)

    def test_notebook_explains_policy_comparison(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("insulin-to-carbohydrate ratio", source)
        self.assertIn("insulin sensitivity factor", source)
        self.assertIn("not \"optimal.\"", source)
        self.assertIn("abstention", source)

    def test_notebook_has_summary_and_troubleshooting(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("Summary and next steps", source)
        self.assertIn("Suggested next action", source)
        self.assertIn("Troubleshooting", source)
        self.assertIn("Python kernel not selected", source)

    def test_notebook_uses_progressive_disclosure_for_technical_detail(self) -> None:
        source = self._source(self._load_notebook())
        self.assertIn("<details>", source)
        self.assertIn("Technical note", source)


if __name__ == "__main__":
    unittest.main()
