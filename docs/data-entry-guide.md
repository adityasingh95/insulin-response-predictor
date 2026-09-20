# CSV and Google Sheets export guide

Use four tabs named `glucose`, `food`, `insulin`, and `context`. Export each tab as CSV and
place the files together using those exact names. The `context` tab is optional; the other
three are required.

## Timestamp rule

Use ISO 8601 timestamps with a UTC offset:

```text
2026-09-20T08:15:00+05:30
```

The offset is mandatory. A value such as `2026-09-20 08:15` is rejected because its timezone
is unknowable after export. `recorded_at` is optional and records when the row was entered;
`timestamp` records when the health event actually occurred.

## Glucose

Required columns:

- `subject_id`: pseudonymous identifier; never use a name, email, or medical-record number
- `timestamp`
- `glucose_mg_dl`
- `source`: for example `fingerstick` or `cgm`
- `context`: `fasting`, `pre_meal`, `post_meal`, `bedtime`, `symptomatic`, or `other`

## Food

Required columns:

- `subject_id`, `timestamp`, `description`, and `carbs_g`
- `meal_type`: `breakfast`, `lunch`, `dinner`, `snack`, or `hypo_treatment`
- `gi_class`: `low`, `medium`, `high`, or `unknown`
- `is_hypo_treatment`: `true` or `false`

Every hypo treatment must use both `meal_type=hypo_treatment` and
`is_hypo_treatment=true`. Regular food must use `false`.

Optional fields are `meal_reference_id`, `protein_g`, `fat_g`, `estimate_confidence`, and
`notes`. To reuse meals, copy `templates/meal_references.csv` beside the exports and add one
unique row per repeated meal. A matching food row inherits only blank details; explicit
values in the food row always win. Keep the lookup local because descriptions can be
sensitive.

## Insulin

Required columns:

- `subject_id`, `timestamp`, `units`, and `product`
- `insulin_type`: `rapid` or `long`
- `dose_reason`: `meal_bolus`, `correction`, `basal`, or `combined`

When `dose_reason=combined`, populate `meal_units` and `correction_units` when the split is
known. The two fields should sum to `units`.

## Context

Use the optional context tab for `exercise`, `illness`, `stress`, `poor_sleep`, `alcohol`, or
`other`. Exercise should include `duration_minutes` and a consistent intensity description.

## Export workflow

1. Export each Google Sheets tab as CSV.
2. Rename the files to `glucose.csv`, `food.csv`, `insulin.csv`, and `context.csv`.
3. Place them in a local ignored directory such as `data/raw/latest/`.
4. Open `notebooks/end_to_end.ipynb`, select `MODE = 'csv'`, and run all cells. The equivalent
   CLI entry point is `irp run-pipeline`.
5. Review data quality, exploratory analysis, forward evaluation, and—only after a gate
   pass—the retrospective policy experiment.

Do not commit exported files or generated reports. They may contain personal health data.
