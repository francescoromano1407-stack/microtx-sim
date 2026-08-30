# Exploratory synthetic campaign contract

`configs/policy_exploratory_synthetic.toml` is a configuration for a
non-empirical computational experiment. It is not a production-campaign
variant and it does not relax, replace, or inherit authority from
`configs/policy_campaign.toml`. The production configuration remains
fail-closed.

The exploratory configuration writes only to the reserved namespace
`artifacts/policy_exploratory_synthetic/` if a later, explicitly approved run
is launched. Its SQLite ledger is persistent and located inside that same
namespace. Validation does not create the directory, ledger, cohort,
realizations, manifest, or results.

## Interpretation boundary

Every exploratory output must carry this exact wording:

> This is an exploratory computational simulation using an illustrative,
> non-empirical projected population. Results are conditional on model
> assumptions; they are not empirical estimates, are not
> population-representative claims, and are not causal evidence about
> real-world players. Monetary outputs are model-equivalent amounts, not
> observed real-world spending. No real-world generalisation is permitted.

The machine-readable contract fixes `campaign_ready=false`, prohibits
empirical, population-inference, real-world causal, and generalisation claims,
and labels unweighted tables as diagnostics only.

## Dual plan binding

The configuration uses two explicit plan identities:

- `inputs/prospective-analysis-plan-amendment-v3.json` remains the unchanged
  scientific parent understood by the existing exact weighted-estimand
  executor. It retains its v2 parent identity and does not authorize a
  campaign.
- `inputs/exploratory-synthetic-analysis-plan-v1.json` is the separate,
  content-addressed exploratory sidecar. It records the exploratory purpose,
  the complete ordered seven-scenario catalogue, the unchanged primary
  estimand and directed contrast, harm weights, population weighting,
  monetary interpretation, fixed seeds, uncertainty design, convergence
  rule, and non-empirical limitations.

The sidecar is unregistered, is not described as an externally preregistered
protocol, and cannot be promoted to campaign readiness.

## Population and paired execution

The selected projected population is illustrative and non-empirical. Its
design, runtime mapping, adapter, exact rational weights, and execution-input
identity are fixed. A later run must use one common pre-treatment cohort and
the same population assignment and weights for every scenario. Weights are
applied within each seed before cross-seed aggregation. Exact balance is a
design property, not evidence of population representativeness.

## Monetary interpretation

`simulation_cents` remain internal synthetic model units. Root tables that
contain them are diagnostic and are not real-currency or cross-country
monetary results. Raw internal units cannot be pooled across countries.

The monetary contract retains the official ECB point-rate evidence, exact
rational conversion basis, EUR target, periods, quote and scale conventions,
missing-date policy, and the single post-aggregation rounding boundary. The
FX observation is official; the bridge from internal model units to local
nominal anchors remains illustrative and is not empirically calibrated.
Converted values, if produced under an applicable planned monetary estimand,
must be called model-equivalent amounts, never observed spending or real
monetary effects.

## Statistical contract

The fixed set contains 150 seeds. Scenario order is canonical; common random
numbers, common cohorts, seed identities, and within-seed population weights
are mandatory. OAT sensitivity remains diagnostic. The joint parameter design
contains illustrative design points without a probability interpretation.
Population and monetary-rate uncertainty remain unquantified, so combined
uncertainty is unavailable and `campaign_ready` remains false.

Convergence is evaluated only from completed realizations in deterministic
blocks of 50. Declaring 150 seeds does not establish convergence. Until a run
exists, Monte Carlo diagnostics, sensitivity stability, and convergence all
have status `NOT_EXECUTED` or `NOT_EVALUATED_NOT_EXECUTED`. A later result may
still be non-converged, insufficient, or unstable.

## Validation-only command

The deterministic preflight is:

```text
microtx-sim exploratory-validate configs/policy_exploratory_synthetic.toml
```

It loads and hashes the configuration, both plans, projected-population
inputs, profiles, and monetary evidence. It does not call the batch,
sensitivity, reproduction, cohort-construction, export, manifest, or ledger
paths.

The eventual launch command proposed for review is:

```text
microtx-sim policy-batch configs/policy_exploratory_synthetic.toml
```

Do not run it without explicit approval. In this pre-execution revision the
command is also held closed in code after successful validation, before profile
loading, cohort construction, scenarios, sensitivity, export, or ledger work.
Accordingly the expected output set at this review checkpoint is empty.
`policy-sensitivity` and `reproduce` are rejected for this configuration as
unreviewed execution paths.

Opening the hold after approval requires a separate reviewed change that binds
the exploratory executor to the declared blockwise convergence and Monte Carlo
diagnostics, produces the dedicated uncertainty/convergence artifacts, and
prevents the legacy raw `simulation_cents` aggregates from becoming
cross-country monetary results. The generic policy exporter is not treated as
that approval or implementation.

## Current readiness

The configuration is structurally valid but not executed. It is not a
production campaign, and `campaign_ready` is permanently false for this
exploratory purpose. The proposed launch command remains intentionally disabled
pending explicit review and the dedicated output-path work described above.
Remaining scientific limitations include the
illustrative, non-empirical population; the illustrative internal-to-money
bridge; a missing source-bundle signature; unquantified monetary-rate and
population uncertainty; and uncalibrated parameter ranges.
