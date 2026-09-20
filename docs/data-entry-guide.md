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

## Quick-start example

A coherent fictional day of data across all four tabs, ready to copy in as a formatting
reference. Replace every value before using it for real logging.

`glucose.csv`:

```csv
subject_id,timestamp,glucose_mg_dl,source,context,recorded_at,notes
SUBJ-001,2026-09-20T07:45:00+05:30,128,cgm,pre_meal,,
SUBJ-001,2026-09-20T09:53:00+05:30,142,cgm,post_meal,,
SUBJ-001,2026-09-20T13:00:00+05:30,131,cgm,pre_meal,,
SUBJ-001,2026-09-20T15:10:00+05:30,135,cgm,post_meal,,
SUBJ-001,2026-09-20T16:00:00+05:30,62,fingerstick,symptomatic,,felt shaky
```

`food.csv`:

```csv
subject_id,timestamp,description,carbs_g,meal_type,gi_class,is_hypo_treatment,meal_reference_id,protein_g,fat_g,recorded_at,estimate_confidence,notes
SUBJ-001,2026-09-20T07:50:00+05:30,Oatmeal with banana,45,breakfast,medium,false,breakfast-oatmeal,8,5,,high,
SUBJ-001,2026-09-20T13:02:00+05:30,Chicken salad wrap,38,lunch,low,false,,25,12,,medium,
SUBJ-001,2026-09-20T16:02:00+05:30,Glucose tablets x4,15,hypo_treatment,high,true,,0,0,,high,treated low
```

`insulin.csv`:

```csv
subject_id,timestamp,units,insulin_type,product,dose_reason,meal_units,correction_units,injection_site,recorded_at,notes
SUBJ-001,2026-09-20T07:52:00+05:30,4.5,rapid,Humalog,meal_bolus,,,abdomen,,
SUBJ-001,2026-09-20T13:05:00+05:30,4.0,rapid,Humalog,meal_bolus,,,thigh,,
SUBJ-001,2026-09-20T20:00:00+05:30,12,long,Lantus,basal,,,abdomen,,evening basal
```

`context.csv` (optional):

```csv
subject_id,timestamp,event_type,duration_minutes,intensity,recorded_at,notes
SUBJ-001,2026-09-20T06:30:00+05:30,exercise,30,moderate,,morning jog
```

`meal_references.csv` (optional, local only):

```csv
meal_reference_id,description,carbs_g,gi_class,protein_g,fat_g
breakfast-oatmeal,Oatmeal with banana,45,medium,8,5
```

## Data-collection checklist

For whoever is logging events day-to-day:

- Log glucose, meals, and doses as they happen - one row per event, not batched at the end
  of the day.
- Always include the UTC offset on every timestamp (see the timestamp rule above).
- Use `mg/dL` consistently for glucose; do not mix in `mmol/L` values.
- Log the bolus separately from the meal, even if they happen almost together.
- Give every hypo treatment both `meal_type=hypo_treatment` and `is_hypo_treatment=true`;
  give every regular meal `is_hypo_treatment=false`.
- When a single injection covers a meal and a correction, use `dose_reason=combined` and
  fill `meal_units`/`correction_units` so they sum to `units`; otherwise use `meal_bolus`
  or `correction` alone.
- Log context events (exercise, illness, stress, poor sleep, alcohol) the same day so the
  pipeline can account for them.
- Keep one `subject_id` for the whole dataset and never put a real name, email, or
  medical-record number anywhere in the sheet.
- Export and run `irp validate --input data/raw/latest` regularly rather than discovering
  problems after months of logging; fix mistakes in the source spreadsheet, never in the
  generated CSV or report.

## Export workflow

1. Export each Google Sheets tab as CSV.
2. Rename the files to `glucose.csv`, `food.csv`, `insulin.csv`, and `context.csv`.
3. Place them in a local ignored directory such as `data/raw/latest/`.
4. Open `notebooks/end_to_end.ipynb`, select `MODE = 'csv'`, and run all cells. The equivalent
   CLI entry point is `irp run-pipeline`.
5. Review data quality, exploratory analysis, forward evaluation, and—only after a gate
   pass—the retrospective policy experiment.

Do not commit exported files or generated reports. They may contain personal health data.
