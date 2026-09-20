"""Command-line interface for safe, reproducible pipeline steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .episodes import build_meal_episodes
from .synthetic import write_synthetic_dataset
from .validation import validate_dataset


def _load_tables(directory: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in ("glucose", "food", "insulin", "context"):
        path = directory / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def _synthetic(args: argparse.Namespace) -> int:
    write_synthetic_dataset(args.output, days=args.days, seed=args.seed)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("generate-synthetic")
    synthetic.add_argument("--output", default="data/synthetic")
    synthetic.add_argument("--days", type=int, default=21)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.set_defaults(func=_synthetic)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.set_defaults(func=_validate)

    episodes = subparsers.add_parser("build-episodes")
    episodes.add_argument("--input", required=True)
    episodes.add_argument("--output", default="data/interim/episodes.csv")
    episodes.set_defaults(func=_episodes)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
