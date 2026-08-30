# Prospective analysis-plan composition

The policy prototype supports an optional, file-backed prospective analysis
plan. This is a composition and reproducibility boundary: it freezes the
declared scenario direction, exactly one `PRIMARY` estimand specification, any
secondary estimand specifications, harm weights, fixed seeds, output contracts,
and executable pre-treatment inclusion rules before treatment outcomes are
generated. `PRIMARY` is a reporting role for one specification. Schema v1
continues to produce only separate fixed-seed realizations. Schema v2 adds an
explicit, outcome-blind plan-level aggregation contract for that one primary.
Schema v3 retains that estimand and aggregate while adding a canonical
successor amendment for campaign-shaped population, monetary, uncertainty,
convergence, flow, output, and execution-attestation inputs.

It is not an external preregistration system. All three schema versions require
`registration_status = "UNREGISTERED"`, `preregistered = false`, and
`campaign_ready = false`.

The checked-in development fixture is
`inputs/prospective-analysis-plan.json`, with plan ID
`illustrative.prospective.composite-harm.baseline-vs-safe.v2`. It is selected
only by `configs/policy_prospective.toml`. The separate
`inputs/prospective-analysis-plan-amendment-v3.json` preserves that file as its
parent and is selected only by `configs/policy_campaign.toml`. It expands the
fixed stopping rule to 150 seeds without changing the primary estimand. The
ordinary `configs/policy_prototype.toml` and smoke configuration remain
unplanned. Both plans and their population inputs are illustrative and are not
campaign-ready scientific evidence.

## Configuration

An analysis plan is available only with projected-population execution because
the binding requires the adapter's exact target weights, joint-cell sidecars,
and per-seed balance lineage:

```toml
[population]
mode = "projected_v1"
design_bundle_path = "../inputs/population-design.toml"
runtime_mapping_bundle_path = "../inputs/population-runtime-mapping.json"
adapter_id = "reviewed.population.adapter.v1"
evidence_bundle_path = "../inputs/population-bundle.toml"
source_registry_path = "../inputs/population-sources.toml"

[analysis_plan]
plan_path = "../inputs/prospective-analysis-plan.json"
```

The checked-in opt-in configuration points these fields at
`inputs/illustrative_population/`, whose source labels, missing signature, and
test-only status remain visible in the files and manifest. The policy output
must also keep `include_player_rows = true`. Planned outcomes are selected from
`player_outcomes.csv` contracts, so a plan cannot be composed with a run that
suppresses those rows.

Relative paths are interpreted from the policy TOML without resolving away a
symbolic-link leaf. The plan loader then applies its own bounded, no-symlink,
regular-file checks and reopens the file whenever it is re-attested.

## Versioned contract

The strict JSON object contains exactly:

- `schema_version` and `plan_id`;
- expected SHA-256 identities for the causal design, batch specification,
  complete model inputs, harm weights, projected-population input, exact
  profile-input lineage, output metric-contract registry, and target-population
  output profile;
- a fixed-seed stopping rule;
- an ordered set of estimand specifications with exactly one `PRIMARY` role;
- `registration_status`, `preregistered`, `campaign_ready`, and the fixed
  campaign blockers;
- `plan_sha256`, computed over the semantic payload without its own digest.

Schema v2 preserves those fields and adds the literal six-component harm-weight
vector plus `primary_aggregate_rule`. The weight vector must hash to the
existing harm-weight identity. The aggregate rule states that the primary is a
single directed scenario contrast, so its `scenario_weights` array is empty;
secondary scenarios are not averaged into it. It separately states that exact
population-analysis weights are applied within each seed and that independent
Monte Carlo seeds receive equal weight.

Schema v3 preserves the complete schema-v2 estimand specification and aggregate
and adds one canonical `amendment`. The amendment binds the parent plan's file
and semantic identities, proves that the primary specification is unchanged,
lists changed inputs and their readiness consequences, and fixes the projected
population, monetary, uncertainty, convergence, execution-attestation, and
policy-only flow contracts. The current v3 schema requires at least 100 fixed
seeds and carries fixed fail-closed blockers for missing empirical population
validation, population and rate uncertainty, calibrated parameter
distributions, monetary authentication and bridge validation, external
registration, calendar anchoring, and execution attestation. See
[Full campaign contract and readiness](campaign_contract.md).

The v2 valid-realization criteria require a fixed-rule seed, paired reference
and comparison observations, the predeclared pre-treatment population
predicate, and one exact finite population-weighted primary realization. There
are no seed-level exclusion criteria. A missing pair, invalid realization, or
attempt to retain an outcome-favourable subset fails closed. Early stopping and
treatment-result interim looks are forbidden by the fixed-seed rule.

Each estimand names an explicit reference scenario and comparison scenario and
uses the fixed comparison-minus-reference paired weighted-mean interpretation.
It selects one whitelisted player outcome and its exact output metric contract,
declares period and currency semantics where applicable, and carries one
canonical executable inclusion predicate. The predicate can use only
pre-treatment jurisdiction, age/minor status, and the projected income,
household, gaming-state, and payer-history fields. Post-treatment variables and
unknown operators or fields are rejected. Every explicitly selected
categorical level must occur in the exact population-adapter domain; an empty
categorical tuple continues to mean all declared levels.

The checked-in primary compares `baseline_f2p` (comparison) with
`safe_fixed_price_subscription` (reference) on `composite_harm`. A positive
value means the baseline has greater population-weighted composite simulated
harm; a negative value means it has lower simulated harm. The analysis includes
all four declared illustrative jurisdictions, ages 10 through 69, both minor
statuses, gaming states, and payer-history states in the projected population.
The categorical selections and age interval are inclusion criteria. No
additional player or seed exclusion criterion is declared.

Every estimand's inclusive declared period length must equal the executed
`PolicyBatchSpec.days` horizon. A zero-day structural snapshot uses one declared
calendar day because an inclusive date interval cannot have zero length. Only
the duration is executed and checked: `period_start` and `period_end` remain
declarations, are not anchored to a simulation calendar, and cannot establish
that outcomes occurred on those dates.

Analysis-plan schema version 1 accepts only outcomes with a direct compatible
`player_outcomes.csv` metric contract. Run-binding schema version 2 adds the
separate raw-versus-effective metric identities needed for monetary execution;
money-valued estimands additionally require an explicit prospective
money-execution selection; omission remains a pre-execution error. That opt-in
layer binds the plan to one complete, internally coherent set of dated
conversion and price-period declarations and materializes
**target-currency-equivalent model amounts**. The wording is deliberate: a
target currency code describes the requested output basis, but it does not turn
an uncalibrated simulated outcome into observed EUR, GBP, or any other
real-world amount.

For each retained player observation, the money path combines that player's
jurisdiction-specific simulation-money scale and local-to-target rate into one
exact rational composite conversion. It applies that composite directly to the
simulation-cent outcome without rounding. The implementation does not first
reconstruct or round a nominal local-currency amount. Reference and comparison
outcomes are converted before their paired contrast is formed, the same
target-population weights are applied to every scenario, and only converted
values enter the cross-jurisdiction aggregate. The declared equal-seed primary
estimate is rounded exactly once when the final production value is serialized,
using nearest minor unit with half ties away from zero. This fixed order prevents
a raw-unit pool, intermediate local-currency value, or scenario-specific weight
from silently changing the result.

Run-binding version 2 is an explicit metadata migration for every prospective
plan, including score-only plans. Non-money specifications and exact results
retain the same algorithms and values, while binding snapshots and binding
digests change because they now name both the source and effective metric
contracts and carry the possibly empty monetary-basis collection. This does
not change the legacy root output-v3 CSV schema or values.

The exact key set and all declared array ordering are semantic; serialized JSON
object-key order is retained only by the file-byte identity. Unknown, missing,
duplicated, non-canonical, oversized, linked, replaced, or digest-inconsistent
inputs fail closed. `build_prospective_analysis_plan(...)` is the safe
programmatic builder;
`load_prospective_analysis_plan(...)` and
`verify_loaded_prospective_analysis_plan(...)` provide file attestation.

## Execution and output binding

`policy-validate`, `policy-batch`, `policy-sensitivity`, and `reproduce` first
resolve the population adapter and the effective model inputs. When a plan is
selected, all eight expected identities and the fixed seed list are validated
before any policy scenario executes. The profile identity is the exact
`ProfileInputLineage.fingerprint_sha256`, so changes to any `CountryProfile`
value or to its retained bundle/file provenance fail preflight even when the
projected-population adapter is unchanged.

After a policy batch finishes, the binding resolver re-attests the plan and
batch, executes the inclusion predicate against the retained pre-treatment
arrays and projected sidecar, filters the exact per-seed weights, and computes
each declared comparison directly from the retained paired player outcomes. It
records selected-player, weight, population-execution, balance, metric,
profile-input, specification, and exact-result identities in a content-addressed
run binding. Post-run re-attestation compares the batch's retained profile
fingerprint with the prospective plan again.

For schema v2 and v3, the binding's exact per-seed primary realizations feed the
declared plan-level aggregate. Its point estimate is their arithmetic mean. The
between-seed standard deviation uses sample standard deviation (`ddof=1`), the
Monte Carlo standard error is `sd / sqrt(retained_seed_count)`, and the normal
95% Monte Carlo interval is `mean +/- 1.96 * MCSE`. One seed produces zero SD,
zero MCSE, and a zero-width interval. Population weights have already been
applied inside each seed; seeds are then equally weighted. No secondary
estimand or scenario enters this calculation.

The ordinary 13-file output-v3 surface and its unweighted CSV semantics remain
unchanged. An opted-in batch additionally writes the separate
`prospective_analysis/target_population_estimands.csv` and
`prospective_analysis/target_population_estimand_metadata.json`. Schema-v2 and
schema-v3 plans additionally write `prospective_analysis/primary_aggregate.csv` and
`prospective_analysis/primary_aggregate_metadata.json`. The latter explicitly
labels the interval as simulator Monte Carlo variability, not a real-world
population confidence interval. The main manifest contains distinct
`analysis_plan`, `analysis_binding`, and `analysis_output_profile` blocks and
links every prospective file by path, byte length, and SHA-256. Omission of
`[analysis_plan]` preserves the legacy configuration, execution digests,
manifest shape, filenames, and CLI workflow; schema-v1 plans retain their
two-file prospective profile.

When prospective money execution is also selected, only the separate
prospective profile receives the target-currency-equivalent model values and
their exact conversion-execution lineage. Root output-v3 tables, plots, and
their metric contracts continue to report the same illustrative simulation
cents as before; the opt-in layer neither rewrites them nor relabels their unit.

The optional profile has a fresh-target publication contract. Any pre-existing
`prospective_analysis` entry makes either a planned or no-plan export fail
before root artifacts are changed; reruns must use a fresh output location or
remove that exact entry after review. Planned files are prepared in a private
sibling directory, published only after all ordinary root artifacts succeed,
and removed again if the final linking manifest cannot be written. Immediately
before publication, the exporter rechecks the staged file identities, reopens
the selected plan and source evidence, and recomputes the run binding; a late
mutation therefore fails without publishing the optional profile.

Even a structurally valid money binding is synthetic and unregistered. Exact
arithmetic and content identities do not authenticate the publisher or rate
source, calibrate the simulated amounts, establish a representative target
population or genuine held-out validation, complete external preregistration,
or authorize a scientific campaign. The source-bundle signature, calibration,
population, and preregistration gates therefore remain independently blocking.
The schema-v2 plan resolves cross-seed aggregation and Monte Carlo uncertainty.
The schema-v3 successor additionally binds formal uncertainty, convergence,
flow, and environment-attestation contracts, but binding a contract is not
satisfying it. Population and monetary-rate uncertainty remain unquantified,
the parameter ranges remain illustrative, and the population, monetary,
registration, calendar, convergence-evidence, and clean execution-receipt gates
remain blocking. No full campaign was run.
