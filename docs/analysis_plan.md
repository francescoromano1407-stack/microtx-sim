# Prospective analysis-plan composition

The policy prototype supports an optional, file-backed prospective analysis
plan. This is a composition and reproducibility boundary: it freezes the
declared scenario direction, exactly one `PRIMARY` estimand specification, any
secondary estimand specifications, harm weights, fixed seeds, output contracts,
and executable pre-treatment inclusion rules before treatment outcomes are
generated. `PRIMARY` is a reporting role for one specification; schema v1
produces a separate realization for each fixed seed, not one run-level primary
estimate.

It is not an external preregistration system. Schema version 1 requires
`registration_status = "UNREGISTERED"`, `preregistered = false`, and
`campaign_ready = false`. No plan is checked in and none of the supplied
configurations selects one.

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

[analysis_plan]
plan_path = "../inputs/prospective-analysis-plan.json"
```

The policy output must also keep `include_player_rows = true`. Schema-v1
planned outcomes are selected from `player_outcomes.csv` contracts, so a plan
cannot be composed with a run that suppresses those rows.

Relative paths are interpreted from the policy TOML without resolving away a
symbolic-link leaf. The plan loader then applies its own bounded, no-symlink,
regular-file checks and reopens the file whenever it is re-attested.

## Version 1 contract

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
simulation-cent outcome and performs exactly one signed nearest-minor-unit
rounding, with half ties away from zero. The implementation does not first
reconstruct or round a nominal local-currency amount. Reference and comparison
outcomes are converted per observation before their paired contrast is formed
and before target-population weights are applied. This fixed order prevents a
pooled aggregate, a contrast, or an intermediate local-currency reconstruction
from silently choosing a different rounding result.

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

The binding retains those exact per-seed realizations only. It does not define
or calculate a cross-seed point estimator, Monte Carlo interval, convergence
criterion, or other run-level uncertainty summary. Summaries emitted elsewhere
by the synthetic policy runner are not promoted into results of the prospective
plan.

The ordinary 13-file output-v3 surface and its unweighted CSV semantics remain
unchanged. An opted-in batch additionally writes the separate
`prospective_analysis/target_population_estimands.csv` and
`prospective_analysis/target_population_estimand_metadata.json` profile. The
main manifest contains distinct `analysis_plan`, `analysis_binding`, and
`analysis_output_profile` blocks linking those files and their hashes. Omission
of `[analysis_plan]` preserves the legacy configuration, execution digests,
manifest shape, filenames, and CLI workflow.

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
The plan's fixed machine-readable blockers also continue to record that the
execution calendar anchor is unbound, cross-seed aggregation and uncertainty
are unresolved, and the model implementation/environment identity is unbound.
The input-value digests do not attest the source tree, interpreter, dependency
lock, or build environment; binding those identities remains prospective P2
work.
