# End-to-end synthetic demonstration

The demonstration proves the complete research workflow before personal health data is used.
It runs two entirely fictional datasets through the same production pipeline.

## Run locally

```bash
python -m pip install -e '.[dev]'
irp run-demo --output demo-output --days 90
```

Or with Docker:

```bash
docker compose run --rm demo
```

## Expected outcomes

### Identifiable scenario

This scenario contains a deliberately observable dose effect and moderate noise. It must:

1. Pass schema validation and episode construction.
2. Pass at least one forward model through the skill and dose-sensitivity gate.
3. Run all four retrospective policy experiments.
4. Produce a policy safety audit and a `complete` manifest.

It exists only to exercise the software. Its metrics have no clinical meaning.

### Noisy scenario

This scenario contains a weak, confounded signal and high outcome noise. It must:

1. Pass schema validation.
2. Fail the forward-model gate.
3. Produce a STOP report.
4. Never execute the policy stage.

This proves that the pipeline does not force every dataset toward a dosing result.

## Output layout

```text
demo-output/
├── demo_manifest.json
├── identifiable/
│   ├── input/
│   └── output/
│       ├── pipeline_manifest.json
│       ├── 01_data_quality/
│       ├── 02_exploratory_analysis/
│       ├── 03_forward_model/
│       └── 04_policy_experiment/
└── noisy/
    ├── input/
    └── output/
        ├── pipeline_manifest.json
        ├── 01_data_quality/
        ├── 02_exploratory_analysis/
        └── 03_forward_model/
```

The noisy output intentionally contains no `04_policy_experiment` directory.

## Running a later CSV export

```bash
irp run-pipeline \
  --input data/raw/latest \
  --output reports/generated/latest \
  --subject-config config/subject.yaml
```

The command exits successfully when the forward gate returns STOP because STOP is a valid
research result. It returns an error only for invalid input data or an unexpected pipeline
failure.
