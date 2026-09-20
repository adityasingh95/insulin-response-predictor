# Forward-model evaluation guide

The forward model estimates glucose observed two to four hours after a meal bolus. It is the
first analytical gate in the project. No policy or dose recommendation should be built when
this layer fails.

## Models

- **Persistence:** predicts that glucose remains at its pre-meal value.
- **Linear extrapolation:** projects the last observable glucose slope across the outcome
  horizon. Missing slopes fall back to persistence.
- **Ridge:** an interpretable, strongly regularized linear model.
- **Histogram gradient boosting:** a conservative nonlinear tabular benchmark.
- **Random Forest:** a second nonlinear sanity check with restricted depth and leaf size.

All learned models use the same feature table and the same chronological split. No random
train/test split is available.

## Metrics

- RMSE and MAE in mg/dL
- Skill versus persistence: `1 - RMSE(model) / RMSE(persistence)`
- Directional accuracy relative to the pre-meal glucose
- Finite-difference dose sensitivity in mg/dL per unit

## Dose-sensitivity gate

For each supported test row, the evaluator changes the recorded rapid dose by ±0.5 units and
measures how the fitted model's prediction changes. Rows near the edge of the historically
observed training range are excluded rather than extrapolated.

A model passes only when:

1. Skill versus persistence is greater than 0.20.
2. Median dose sensitivity is negative.
3. At least 80% of evaluated sensitivities are negative.

This check detects models that predict well for the wrong reasons but have learned a flat or
physiologically reversed relationship with observed dose.

## Interpretation limitation

Historical doses were chosen in response to meal size and glucose. They were not randomized.
Therefore, dose sensitivity remains an observational association affected by confounding by
indication; it is not a causal insulin-sensitivity estimate.

A gate pass permits investigation of retrospective policies. It does not justify real-world
use. A gate failure stops the dosing layer and is a valid project result.
