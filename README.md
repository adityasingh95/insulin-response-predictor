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
- Leakage-resistant episode features
- Google Sheets/CSV export templates and normalized column loading
- Local Markdown/JSON quality reports and a three-panel event timeline

Forward models and retrospective dose-policy experiments will be added only after the data
foundation is verified. No personal data is required to run this milestone.

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
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

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

Field definitions live in `src/insulin_response_predictor/schemas.py`. Copy
`config/subject.example.yaml` to the ignored `config/subject.yaml` for local configuration.
Blank importable templates live in `templates/`; they can also be regenerated with:

```bash
irp create-templates --output templates
```

See `docs/data-entry-guide.md` for Google Sheets tab names, allowed values, timestamp rules,
and the export workflow. The entry mechanism can change later without changing this CSV
contract.

## Episode status

Each non-hypo meal is classified as:

- `clean`: bolus, pre-meal reading, and 2–4 hour outcome exist with no intervening food or
  rapid insulin.
- `contaminated`: a usable outcome exists, but an intervening event prevents clean causal
  attribution.
- `unusable`: a required bolus or glucose observation is missing.

Only clean episodes should enter the primary forward-model dataset.
