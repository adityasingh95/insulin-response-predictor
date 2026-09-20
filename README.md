# Insulin Response Predictor

An offline research pipeline for studying how glucose changes after meals and rapid-acting
insulin. The first milestone validates event data, constructs auditable meal episodes, and
creates leakage-resistant physiology features.

> **Research only — not for treatment decisions.** This repository does not provide live
> insulin recommendations and is not a medical device. Any subject-specific targets or
> regimen parameters must come from the subject's clinical care team.

## Current scope

- Canonical schemas for glucose, food, insulin, and context events
- Deterministic validation with row-level issues
- Synthetic data for development and CI
- Meal/bolus/pre-glucose/outcome pairing
- Explicit detection of intervening food, hypo treatment, and rapid insulin
- Configurable rapid insulin-on-board and carbohydrate-on-board curves
- Leakage-resistant episode and recent-context features
- Google Sheets/CSV export templates and normalized column loading
- Optional repeated-meal lookup to reduce duplicate food entry
- Local Markdown/JSON quality reports and a three-panel event timeline
- Chronological and rolling-origin benchmarks with bootstrap intervals
- Gated retrospective policy experiments and safety auditing
- Guided Jupyter notebook for running and reviewing the full workflow
- Complete synthetic PASS and STOP demonstrations

No personal data is required to run the complete fictional demonstration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
```

Generate fictional development data:

```bash
irp generate-synthetic --output data/synthetic --days 21
irp validate --input data/synthetic
irp build-episodes --input data/synthetic --output data/interim/episodes.csv
irp assess --input data/synthetic --output reports/generated/synthetic
irp evaluate-forward --input data/synthetic --output reports/generated/forward
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Run without the command line

Open `notebooks/end_to_end.ipynb` in Jupyter or VS Code and run the cells from top to bottom.
It defaults to a 90-day fictional PASS scenario, presents status as tables, and renders each
report and chart inline. One settings cell switches to local Google Sheets/CSV exports.

## Personal-data boundary

The repository may be public, but real health data must not be. The `.gitignore` excludes:

- `data/raw/`, `data/interim/`, and `data/processed/`
- `config/subject.yaml`
- model artifacts and generated reports
- SQLite, Parquet, pickle, and joblib files

Only fictional files under `data/synthetic/` are eligible for version control. Use a
pseudonymous `subject_id`; never place a name, email address, phone number, medical-record
number, or date of birth in project data.

## Expected input files

Place local CSVs in one directory using these names:

- `glucose.csv`
- `food.csv`
- `insulin.csv`
- `context.csv` (optional)
- `meal_references.csv` (optional reusable lookup)

Field definitions live in `src/insulin_response_predictor/schemas.py`. Copy
`config/subject.example.yaml` to the ignored `config/subject.yaml` for local configuration.
Blank importable templates live in `templates/`; they can also be regenerated with:

```bash
irp create-templates --output templates
```

See `docs/data-entry-guide.md` for Google Sheets tab names, allowed values, timestamp rules,
and the export workflow. The entry mechanism can change later without changing this CSV
contract.

Food details are entered in the `food` tab: description, carbohydrate grams, meal type,
glycemic-index class, and optional protein/fat. For repeated meals, place a copy of
`templates/meal_references.csv` beside the exports and use `meal_reference_id`; blank food
details are filled from that local lookup. There is no food database, barcode scanner, or
image-derived nutrition estimate in this version.

## Episode status

Each non-hypo meal is classified as:

- `clean`: bolus, pre-meal reading, and 2–4 hour outcome exist with no intervening food or
  rapid insulin.
- `contaminated`: a usable outcome exists, but an intervening event prevents clean causal
  attribution.
- `unusable`: a required bolus or glucose observation is missing.

Only clean episodes should enter the primary forward-model dataset.

## Forward-model evaluation

`irp evaluate-forward` compares persistence, linear extrapolation, Ridge, histogram gradient
boosting, and Random Forest on the same chronological test period. It reports RMSE, MAE,
skill versus persistence, directional accuracy, bootstrap confidence intervals,
rolling-origin robustness, and finite-difference dose sensitivity.

The gate requires skill above 0.20, negative median dose sensitivity, and correctly signed
sensitivity on at least 80% of supported test rows. Passing only permits further retrospective
research; it does not validate dose recommendations. See `docs/forward-model-guide.md`.

## Complete end-to-end demonstration

Run the entire setup before introducing real data:

```bash
irp run-demo --output demo-output --days 90
```

The `identifiable` fictional scenario must complete validation, forward modelling, gated
policy experiments, and safety reporting. The `noisy` fictional scenario must stop at the
forward gate and never run a policy. Both behaviors are tested in CI.

The same demonstration is available through Docker:

```bash
docker compose run --rm demo
```

See `docs/end-to-end-demo.md` for the artifact tree and expected manifests.

## Running a later real export

First copy `config/subject.example.yaml` to the ignored `config/subject.yaml` and replace every
example with clinician-approved subject values. Then run:

```bash
irp run-pipeline \
  --input data/raw/latest \
  --output reports/generated/latest \
  --subject-config config/subject.yaml
```

The policy stage is unreachable unless a forward model passes the gate. Its outputs remain
retrospective candidate-dose experiments, not instructions.

The policy comparison includes the configured ICR/ISF formula, a formula fitted to historical
in-range episodes, a historical-dose imitation baseline, and constrained inversion of a
passing forward model. Candidates are retrospective, quantized, bounded by observed
meal-specific dose support, and subject to abstention rules. None is a prescription.
