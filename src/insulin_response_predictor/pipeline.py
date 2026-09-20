"""One-command orchestration for the full retrospective research pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import write_forward_evaluation
from .io import load_csv_exports
from .policy import PolicyConfig, write_policy_evaluation
from .reporting import write_assessment
from .synthetic import write_synthetic_dataset
from .validation import validate_dataset


def _write_manifest(destination: Path, manifest: dict[str, object]) -> None:
    (destination / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def run_pipeline(
    tables: dict[str, pd.DataFrame],
    destination: str | Path,
    *,
    policy_config: PolicyConfig | None = None,
) -> dict[str, object]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "started",
        "research_only": True,
        "stages": {},
        "artifacts": [],
    }
    stages = manifest["stages"]
    artifacts = manifest["artifacts"]
    assert isinstance(stages, dict)
    assert isinstance(artifacts, list)

    issues = validate_dataset(tables)
    validation_errors = [issue for issue in issues if issue.severity == "error"]
    write_assessment(tables, destination / "01_data_quality")
    artifacts.extend(
        [
            "01_data_quality/data_quality.md",
            "01_data_quality/data_quality.json",
            "01_data_quality/timeline.png",
        ]
    )
    stages["data_quality"] = {
        "status": "pass" if not validation_errors else "fail",
        "validation_errors": len(validation_errors),
    }
    if validation_errors:
        manifest["status"] = "validation_failed"
        stages["forward_model"] = {"status": "not_run"}
        stages["policy_experiment"] = {"status": "not_run"}
        _write_manifest(destination, manifest)
        return manifest

    forward_result, _ = write_forward_evaluation(tables, destination / "02_forward_model")
    artifacts.extend(
        [
            "02_forward_model/forward_report.md",
            "02_forward_model/forward_metrics.json",
            "02_forward_model/forward_predictions.csv",
            "02_forward_model/predicted_vs_actual.png",
        ]
    )
    gate = forward_result["gate"]
    assert isinstance(gate, dict)
    forward_passed = bool(gate["any_model_passes"])
    stages["forward_model"] = {
        "status": "pass" if forward_passed else "stop",
        "best_model": gate["best_model_by_rmse"],
        "any_model_passes": forward_passed,
    }
    if not forward_passed:
        manifest["status"] = "stopped_at_forward_gate"
        stages["policy_experiment"] = {
            "status": "not_run",
            "reason": "forward_model_gate_failed",
        }
        _write_manifest(destination, manifest)
        return manifest

    policy_result, _ = write_policy_evaluation(
        tables,
        forward_result,
        destination / "03_policy_experiment",
        config=policy_config,
    )
    artifacts.extend(
        [
            "03_policy_experiment/policy_report.md",
            "03_policy_experiment/policy_metrics.json",
            "03_policy_experiment/policy_candidates.csv",
        ]
    )
    stages["policy_experiment"] = {
        "status": "complete",
        "selected_forward_model": policy_result["selected_forward_model"],
    }
    manifest["status"] = "complete"
    _write_manifest(destination, manifest)
    return manifest


def run_demo(destination: str | Path, *, days: int = 90, seed: int = 42) -> dict[str, object]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scenarios: dict[str, object] = {}
    for scenario in ("identifiable", "noisy"):
        scenario_root = destination / scenario
        input_directory = scenario_root / "input"
        output_directory = scenario_root / "output"
        write_synthetic_dataset(
            input_directory, days=days, seed=seed, scenario=scenario
        )
        tables = load_csv_exports(input_directory)
        scenarios[scenario] = run_pipeline(tables, output_directory)

    result = {
        "research_only": True,
        "days_per_scenario": days,
        "scenarios": scenarios,
        "expected": {
            "identifiable": "complete",
            "noisy": "stopped_at_forward_gate",
        },
    }
    (destination / "demo_manifest.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result
