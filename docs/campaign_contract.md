# Full campaign contract and readiness

`configs/policy_campaign.toml` is the strict configuration surface for a full
policy campaign. It is **not** authorization to run one. The configuration,
schema-v3 successor analysis plan, uncertainty design, convergence rules,
execution receipt, and output profile are pre-campaign infrastructure. Every
gate fails closed, and `campaign_ready` remains `false` until all independent
scientific and technical requirements pass.

No full scientific or production campaign was run while this contract was
implemented. Structural validation, unit and integration tests, dry runs, and
bounded convergence pilots are the only permitted pre-readiness executions.

## Authoritative artifacts

The campaign candidate is defined by these versioned artifacts:

- [`policy_campaign.toml`](../configs/policy_campaign.toml) binds the intended
  horizon and player count, fixed seeds, projected population, monetary basis,
  uncertainty and convergence declarations, output profile, persistent SQLite
  ledger, and execution-receipt policy. It sets `allow_synthetic = false`,
  `fail_closed = true`, and `campaign_ready = false`.
- [`prospective-analysis-plan-amendment-v3.json`](../inputs/prospective-analysis-plan-amendment-v3.json)
  is a successor to, not a replacement for,
  [`prospective-analysis-plan.json`](../inputs/prospective-analysis-plan.json).
  It records the parent file and semantic identities, preserves the original
  primary estimand and scenario direction, and binds changed population,
  monetary, uncertainty, convergence, flow, and attestation contracts.
- [`parameter-uncertainty-design-v1.json`](../inputs/parameter-uncertainty-design-v1.json)
  declares the reproducible joint parameter design.
- [`execution-receipt.schema.json`](../schemas/execution-receipt.schema.json)
  defines the technical execution receipt. The manifest must reference the
  receipt and post-execution verification when execution eventually becomes
  admissible under a future contract.

All file hashes and semantic hashes must be recomputed from the files present
at preflight. A hash written in a configuration, plan, report, or earlier
message is only an expected identity; it is not evidence until re-attested.

The non-executing preflight writes `pre-campaign-validation-report.json`. The
report separates passed and failed checks, unresolved blockers, uncertainty
availability, convergence, and the execution-receipt attempt. Its schema fixes
`full_campaign_intentionally_not_run = true`, zero realized seeds,
`NON_CONVERGED`, and `campaign_ready = false`; a structural pass cannot promote
those states.

Run that preflight only from the repository root and a clean committed tree:

```text
python -m microtx_sim campaign-preflight configs/policy_campaign.toml
```

The command never dispatches the batch or sensitivity executors. The campaign
batch path independently requires a pre-execution receipt and rejects schema-v1
readiness blockers before the first simulation realization.

## Execution flow and layer boundary

The declared primary estimand uses the welfare-policy layer only:

```text
configuration
  -> amended plan and input attestation
  -> projected population execution
  -> common pre-treatment cohort
  -> paired policy scenarios
  -> policy_orchestrator outcomes
  -> analysis binding
  -> exact monetary conversion when declared
  -> population weighting within each seed
  -> seed and joint uncertainty summaries
  -> blockwise convergence checks
  -> output profile, manifest, and receipt verification
```

The richer strategic market implemented by `microtx_sim.core.world` is a
separate simulation layer. It contains company strategy, competition, audit,
collusion, and regulator interactions that the policy prototype does not
automatically reproduce. It is **not composed** with
`microtx_sim.simulation.policy_orchestrator` for the current estimand. Selecting
the strategic layer or both layers is rejected because no typed scientific
adapter binds their player IDs, jurisdictions, scenarios, cohorts, outcomes,
units, population weights, monetary bases, and lineage. Outputs from the two
layers must never be silently pooled.

Within the policy flow, reference and comparison branches must retain the same
seed, pre-treatment cohort, player and jurisdiction identities, and population
weights. Money-valued outcomes must be converted exactly to the declared target
basis before population weighting and cross-jurisdiction aggregation. Raw
cross-currency pooling is rejected. A passing flow attestation establishes
composition and identity only; it does not establish calibration or readiness.

## Uncertainty contract

The result surface separates five sources instead of collapsing them into one
interval:

1. **Seed uncertainty.** For fixed parameters, population, and monetary inputs,
   scenarios use common random numbers and paired pre-treatment cohorts.
   Population weights are applied within each seed before equal-weight
   cross-seed aggregation. The summary reports the point estimate, sample
   standard deviation, Monte Carlo standard error, and normal 95% Monte Carlo
   interval. The fixed seed set requires at least 100 retained seeds, and there
   is no outcome-dependent seed-exclusion path.
2. **Parameter uncertainty.** A deterministic seeded Latin-hypercube design
   records every parameter draw and its hash, and can carry a justified,
   provenance-bearing correlation matrix. The current parameter bounds are
   **illustrative ranges, not calibrated probability distributions**. The
   contract rejects relabelling an illustrative range as a distribution.
   Therefore these draws can exercise joint sensitivity and implementation
   structure but do not yet quantify empirical parameter uncertainty. OAT
   sensitivity remains diagnostic only.
3. **Monetary-rate uncertainty.** The production contract preserves the exact
   rational ECB point-rate path, target currency, quote and scale conventions,
   periods, missing-date rules, source identities, and final rounding boundary.
   An official point-rate observation is not a rate distribution. No admissible
   empirical or design-based rate-uncertainty model is currently declared, so
   monetary-rate uncertainty is `UNQUANTIFIED`. The source-bundle signature is
   missing and the bridge from `simulation_cents` to local nominal money is
   illustrative. Converted values are target-currency-equivalent **model
   amounts**, never observed real-world spending.
4. **Population uncertainty.** Exact apportionment, assignment, rational
   weighting, and balance attest deterministic design-weight application. They
   are not empirical validation of the target population and do not themselves
   create a sampling distribution. No admissible alternative-design,
   resampling, or design-based uncertainty method is currently declared, so
   population uncertainty is `UNQUANTIFIED`.
5. **Combined uncertainty.** Each realization carries seed, parameter-draw,
   population-design or replicate, rate-draw or basis, scenario, estimand,
   cohort, and weight identities. The implemented finite-design decomposition
   reports seed, parameter, population, rate, residual/interaction, and total
   components only for a balanced complete Cartesian design. A one-level or
   unidentifiable component is reported as unavailable, never as zero. Because
   required population and rate components are currently unquantified, the
   combined scientific uncertainty judgment remains insufficient.

`QUANTIFIED`, `UNQUANTIFIED`, and `UNAVAILABLE` are distinct states.
`UNQUANTIFIED` or `UNAVAILABLE` cannot be converted to a zero variance.

## Blockwise convergence

Realizations are consumed as the declared ascending seed prefix and evaluated
after deterministic blocks. Every checkpoint records the retained-seed count,
cumulative estimate, Monte Carlo standard error, interval width, absolute and
relative change from the preceding checkpoint, invalid/rejected/excluded
counts, sensitivity-instability flag, status, and blockers.

The numerical thresholds and block size are part of the amended plan and
campaign configuration. At minimum, convergence requires:

- at least 100 retained seeds;
- both the Monte Carlo standard-error and interval-width targets;
- both block-to-block change targets;
- an acceptable invalid-run rate and no excluded seeds;
- no material sensitivity instability;
- the required number of consecutive passing checkpoints; and
- every required uncertainty component to be available.

The possible checkpoint states are `CONVERGED`, `NON_CONVERGED`,
`INSUFFICIENT_PRECISION`, and `UNSTABLE`. Missing required uncertainty prevents
`CONVERGED`, even when the seed estimate appears numerically stable. Because no
full campaign was run, no campaign convergence claim exists.

## Execution receipt

A receipt is a deterministic identity attestation. Its canonical JSON payload
uses sorted keys, UTF-8, normalized POSIX-style paths, and raw file-byte hashes,
and excludes wall-clock timestamps. It binds the clean Git branch and commit,
tracked source tree, Python implementation and executable, operating system and
architecture, installed distributions, `pyproject.toml`, `uv.lock`, campaign
configuration, amended plan, population and monetary artifacts and semantic
identities, output contracts, expected artifacts, ledger, command, execution
mode, and model and receipt schema versions.

Receipt creation requires a clean Git working tree. Before execution, every
identity is recomputed and compared with the receipt. After execution, the same
identity is recomputed; any source, plan, configuration, input, interpreter,
dependency, or environment drift rejects or invalidates the run. An unavailable
required identity remains an explicit blocker instead of receiving a fabricated
digest.

Execution-receipt schema v1 deliberately fixes both
`campaign_execution_admissible = false` and `campaign_ready = false`. A verified
receipt proves that identities match; it does not authenticate publishers,
calibrate the model or money bridge, validate the population, quantify missing
uncertainty, register the analysis, or establish scientific validity.

## Readiness is a conjunction, not a hash

These statuses remain independent:

| Status | Question answered |
| --- | --- |
| Identity verification | Are the exact files, source tree, runtime, and declared semantic objects unchanged? |
| Provenance/authentication | Are external publishers and source artifacts authenticated under the evidence policy? |
| Calibration | Are model parameters and the simulation-money bridge empirically supported? |
| Population validity | Does the design support inference to the declared target population? |
| Registration | Was the prospective plan registered before outcome access? |
| Uncertainty sufficiency | Are every required uncertainty source and their joint propagation validly quantified? |
| Convergence | Did the declared fixed design meet every precision, stability, and invalid-run rule? |
| `campaign_ready` | Did **all** of the above and every environment/execution gate pass together? |

Content-addressed lineage can pass while scientific readiness fails. Structural
population balance can pass while population validity fails. An official point
FX rate can pass while rate uncertainty and source authentication fail.

The current campaign remains blocked: the plan is unregistered and lacks an
execution-calendar anchor; projected-population empirical validation and
population uncertainty are missing; parameter distributions are uncalibrated;
the monetary source signature is missing, its simulation bridge is unvalidated,
and rate uncertainty is unquantified; and execution attestation has not opened
the gate. Preflight must report these blockers without weakening or overriding
them, and the full campaign must not run.
