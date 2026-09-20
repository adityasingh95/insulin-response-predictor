"""Command-line interface for safe, reproducible pipeline steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .configuration import load_policy_config
from .episodes import build_meal_episodes
from .evaluation import write_forward_evaluation
from .io import load_csv_exports, write_blank_templates
from .pipeline import run_demo, run_pipeline
from .reporting import write_assessment
from .synthetic import write_synthetic_dataset
from .validation import validate_dataset


def _load_tables(directory: Path):
    return load_csv_exports(directory)


def _templates(args: argparse.Namespace) -> int:
    written = write_blank_templates(args.output)
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    write_synthetic_dataset(
        args.output, days=args.days, seed=args.seed, scenario=args.scenario
    )
    print(f"Wrote fictional development data to {args.output}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    issues = validate_dataset(_load_tables(Path(args.input)))
    payload = [issue.__dict__ for issue in issues]
    print(json.dumps(payload, indent=2))
    return 1 if any(issue.severity == "error" for issue in issues) else 0


def _episodes(args: argparse.Namespace) -> int:
    tables = _load_tables(Path(args.input))
    result = build_meal_episodes(tables["glucose"], tables["food"], tables["insulin"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    counts = result["status"].value_counts(dropna=False).to_dict()
    print(json.dumps({"output": str(output), "episode_counts": counts}, indent=2))
    return 0


def _assess(args: argparse.Namespace) -> int:
    summary, issues, _ = write_assessment(_load_tables(Path(args.input)), args.output)
    print(json.dumps({"output": args.output, "summary": summary}, indent=2, default=str))
    return 1 if any(issue.severity == "error" for issue in issues) else 0


def _evaluate_forward(args: argparse.Namespace) -> int:
    tables = _load_tables(Path(args.input))
    issues = validate_dataset(tables)
    errors = [issue.__dict__ for issue in issues if issue.severity == "error"]
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return 1
    result, _ = write_forward_evaluation(tables, args.output)
    print(json.dumps({"output": args.output, "result": result}, indent=2, default=str))
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    policy_config = load_policy_config(args.subject_config)
    manifest = run_pipeline(
        _load_tables(Path(args.input)),
        args.output,
        policy_config=policy_config,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 1 if manifest["status"] == "validation_failed" else 0


def _run_demo(args: argparse.Namespace) -> int:
    result = run_demo(args.output, days=args.days, seed=args.seed)
    print(json.dumps(result, indent=2, default=str))
    statuses = {
        name: manifest["status"] for name, manifest in result["scenarios"].items()
    }
    expected = result["expected"]
    return 0 if statuses == expected else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("generate-synthetic")
    synthetic.add_argument("--output", default="data/synthetic")
    synthetic.add_argument("--days", type=int, default=21)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument(
        "--scenario", choices=["routine", "identifiable", "noisy"], default="routine"
    )
    synthetic.set_defaults(func=_synthetic)

    templates = subparsers.add_parser("create-templates")
    templates.add_argument("--output", default="templates")
    templates.set_defaults(func=_templates)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.set_defaults(func=_validate)

    episodes = subparsers.add_parser("build-episodes")
    episodes.add_argument("--input", required=True)
    episodes.add_argument("--output", default="data/interim/episodes.csv")
    episodes.set_defaults(func=_episodes)

    assess = subparsers.add_parser("assess")
    assess.add_argument("--input", required=True)
    assess.add_argument("--output", default="reports/generated")
    assess.set_defaults(func=_assess)

    forward = subparsers.add_parser("evaluate-forward")
    forward.add_argument("--input", required=True)
    forward.add_argument("--output", default="reports/generated/forward")
    forward.set_defaults(func=_evaluate_forward)

    pipeline = subparsers.add_parser("run-pipeline")
    pipeline.add_argument("--input", required=True)
    pipeline.add_argument("--output", default="reports/generated/pipeline")
    pipeline.add_argument("--subject-config", required=True)
    pipeline.set_defaults(func=_run_pipeline)

    demo = subparsers.add_parser("run-demo")
    demo.add_argument("--output", default="demo-output")
    demo.add_argument("--days", type=int, default=90)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=_run_demo)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
