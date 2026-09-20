from __future__ import annotations

import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_guided_notebook_is_valid_and_defaults_to_fictional_data(self) -> None:
        path = Path(__file__).parents[1] / "notebooks" / "end_to_end.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
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


if __name__ == "__main__":
    unittest.main()
