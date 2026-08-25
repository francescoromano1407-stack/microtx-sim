# Evidence and provenance policy

## Purpose

Evidence metadata and simulation assumptions are kept separate so that an
official citation cannot silently turn an illustrative equation into a
calibrated estimate. The authoritative machine-readable register is
`data/provenance/sources.toml`. Jurisdiction contracts are declared in
`configs/jurisdictions.toml`, while run-level assumptions are selected by a
scenario such as `configs/smoke.toml`.

The detailed inventory, current runtime use, units, denominators, and dormant
contracts are documented in [Data sources](data_sources.md). Interpretation and
reproducibility gaps are documented in [Limitations](limitations.md).

## Provenance statuses

| Status | Required interpretation |
| --- | --- |
| `CALIBRATED` | Extracted, transformed, and validated for a declared estimand. Compatible unit, period, population, and conditioning are still required. |
| `ANCHORED` | Connected to a named source but missing part of the reproducible calibration and validation chain. |
| `ILLUSTRATIVE` | A transparent modelling or scenario assumption, suitable for structural exploration but not an empirical estimate. |
| `SYNTHETIC` | Artificial input used for scaffolding, tests, or deliberately synthetic scenarios. |

A source record and a parameter can have different statuses. For example, an
official income publication can be anchored while the transformation that maps
it to the model's common internal median remains illustrative.

## Evidence contract

A calibration-ready metric should record:

- a stable source identifier and publisher;
- publication date, retrieval date, and immutable source snapshot;
- table, cell, page, API query, or other exact extraction location;
- original value, currency/unit, period, denominator, condition, geography, and
  target population;
- transformation code, rounding rule, derived unit, and uncertainty;
- the model field or equation that consumes it;
- provenance status and reviewer;
- checksum, source revision, and licence or access restrictions.

The current register supplies only part of this contract. It has source IDs,
publishers, URLs, retrieval dates, status, and support scopes, but it does not
yet store immutable snapshots, checksums, extraction scripts, or complete
table/cell lineage. `ANCHORED` must therefore not be read as reproducibly
calibrated.

The profile loader validates the catalogue-wide retrieval date in canonical
`YYYY-MM-DD` form and retains it on every `SourceProvenance` record. Policy runs
content-address the exact `CountryProfile` values used; a registered bundle also
adds its metric contracts, money scales, jurisdiction-file hash, and source-
register hash to that fingerprinted snapshot. These are software-lineage
controls, not evidence promotion: all current illustrative and synthetic
statuses remain unchanged.

## Output transformation registry

Versioned policy outputs have a separate column-level registry in
`microtx_sim.outputs.metric_contracts`. It covers all 220 columns across the
six CSV tables: 15 identifiers, 9 run-design fields, and 196 derived metrics.
Each contract records a structured unit, period, population base, eligibility
condition, storage and missing-value semantics, versioned transformation
recipe, implementation location, ordered inputs, upstream lineage identifiers,
source version, retrieval-date field, and provenance status. The canonical
registry snapshot and its SHA-256 digest are embedded in every policy manifest.

This is a map of software transformations, not empirical validation. Every
current output contract is `SYNTHETIC` and its empirical source retrieval date
is intentionally unset. The manifest separately links the registry to the exact
resolved execution-input digest, profile-input fingerprint, and any source-
register retrieval date used by that run. A campaign gate fails until derived
contracts and their dependencies are
`CALIBRATED`, retrieval dates are present, and monetary outputs have a dated
cross-country FX or purchasing-power contract.

The registry also makes important reduction conventions explicit. Player-level
harm variance uses population variance (`ddof=0`), while repeated-seed variance
uses sample variance (`ddof=1`). Normal intervals are Monte Carlo intervals for
the simulated mean, not empirical outcome intervals, and collapse to zero width
for one seed. Empty high-risk subsets and empty player arrays are encoded as
zero. Activity-level monetary opportunity proxies remain blank because the
model only computes an aggregate proxy.

## Runtime lineage categories

Every documented input belongs to one of three categories:

1. **Runtime-consumed:** it currently changes initial state or an equation.
2. **Contract-only:** it is parsed and validated but does not yet affect outcomes.
3. **Context-only:** it supports design or interpretation but is not a profile
contract consumed by the runtime. The policy prototype is configured separately
in `configs/policy_prototype.toml`; every added behavioural, harm, producer, and
EPGC value in that file remains synthetic or illustrative.

This classification must be reviewed whenever a value is connected to a new
equation. Documentation should state the effective precedence when the same
construct appears in both a profile and a run configuration.

The present model consumes illustrative age/income shapes, many synthetic agent
defaults, generic rule switches, scenario-level behaviour, information, audit,
and funding parameters. Many official metrics—gaming reach, payer rates,
spending distributions, deprivation, consumption propensity, and programme
rates/caps—remain contract-only. Their presence does not calibrate current
consumption, harm, company, audit, or subsidy outcomes.

`behavior.household_peer_influence` is runtime-consumed but has no empirical
source contract. It is a synthetic sensitivity-only coefficient over a
structurally observed, lagged household co-player signal. It must not be
promoted to a calibrated network effect without external identification and
validation.

## Units

Source currencies remain nominal local units with explicit exponents. GBP and
EUR use two decimal places; JPY and KRW use zero. Annual and monthly bases,
household and individual denominators, and conditional and unconditional
populations must not be mixed.

The current profiles map each jurisdiction's monthly anchor to a fixed `180000`
simulation-cent reference. This is an illustrative within-model scale. It is
neither an exchange rate nor a purchasing-power-parity conversion and removes
empirical cross-country income-level differences. Simulation cents must not be
reported as GBP, KRW, JPY, EUR, or comparable real purchasing power.

## Legal and clinical evidence

Regulatory sources support only the scope explicitly declared in their source
record. A source about price transparency cannot automatically justify a random
reward restriction; a product-specific classification cannot be promoted to an
unconditional national ban. Rules also require an effective date, territorial
scope, product definition, enforcement practice, and dated review before they
can support a real policy scenario.

Clinical or public-health definitions provide construct guidance, not a direct
mapping from a simulated proxy to diagnosis. The model stores functioning
impairment and loss-of-control proxies but does not diagnose gaming disorder or
addiction.

Case reports and enforcement actions establish that rare unauthorised-payment
events or dark patterns can occur. They do not identify a population hazard.
The current rare-card hazard remains an illustrative sensitivity parameter.

## Campaign gate

`SimulationConfig.validate(campaign=True)` requires the scenario itself to be
`CALIBRATED` and forbids synthetic inputs. The profile bundle then checks every
dependent contract and source, including money-scale sources. Because current
outcomes pool money across jurisdictions, campaign validation also requires a
cross-country-comparable monetary contract. The present local-currency anchor
scales cannot make that declaration: a dated FX or purchasing-power contract
must be implemented first. A campaign fails if a required dependency is
synthetic, illustrative, merely anchored, or not monetarily comparable.

Policy-output campaign validation is independently fail-closed. The output
registry currently reports all 196 derived measures as non-calibrated and
missing metric-level empirical retrieval dates, so the policy prototype cannot
be promoted merely because its files, configuration, and Git revision are
reproducible.

Causal-design validation is also independently fail-closed. The current
seven-scenario matrix and its 49 directed contrasts were registered
retrospectively inside a synthetic structural model, not preregistered as an
empirical design. The manifest therefore records explicit retrospective,
preregistration, and empirical-calibration blockers even when the canonical
matrix is used. Custom factor values remain descriptively exportable, but add a
canonical-matrix mismatch blocker and can never inherit the canonical claim.

Passing this software gate will be necessary but not sufficient. The policy
prototype already supplies repeated-seed Monte Carlo intervals, OAT sensitivity,
and a versioned run manifest, but those are structural diagnostics. A scientific
campaign still requires frozen inputs, independent extraction review,
calibrated and held-out validation targets, prespecified estimands, joint
uncertainty, power analysis, and governance for any empirical data.

## Updating evidence

When adding or replacing a metric:

1. preserve the source's original population, condition, denominator, time
   period, and unit;
2. add a source record with the narrowest defensible `supports` scope;
3. add a typed contract naming the exact target metric;
4. implement the transformation in a reproducible script or loader function;
5. add validation for units, ranges, source compatibility, and scenario
   precedence;
6. test that the intended runtime field changes and unrelated fields do not;
7. classify the metric honestly—do not promote it because the publisher is
   official;
8. update [Data sources](data_sources.md) and the relevant limitation or model
   section in the same Git milestone.

No source should be downloaded automatically during an ordinary simulation run.
Campaign inputs should be prepared, reviewed, versioned, and frozen before
execution.
