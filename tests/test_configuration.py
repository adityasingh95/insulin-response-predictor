from __future__ import annotations

import unittest
from pathlib import Path

from insulin_response_predictor.configuration import load_policy_config
from insulin_response_predictor.policy import PolicyConfig


class ConfigurationTests(unittest.TestCase):
    def test_example_subject_config_loads(self) -> None:
        config = load_policy_config(Path("config/subject.example.yaml"))
        self.assertEqual(config.target_mg_dl, 120)
        self.assertEqual(config.lunch_icr_g_per_unit, 10)
        self.assertEqual(config.dose_increment_units, 0.5)

    def test_invalid_target_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolicyConfig(target_low_mg_dl=180, target_mg_dl=120, target_high_mg_dl=80)


if __name__ == "__main__":
    unittest.main()
