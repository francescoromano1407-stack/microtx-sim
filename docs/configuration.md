# Configuration reference

## Files and loading

The repository has two explicit TOML schemas:

- market-world scenarios are loaded by `microtx_sim.config.load_config()`;
- the seven-scenario synthetic policy prototype is loaded by
  `microtx_sim.policy_config.load_policy_config()`.

Both loaders construct immutable dataclasses and reject missing or unknown
fields. The policy loader additionally performs strict runtime type/range checks
and accepts only `provenance_status = "synthetic"`.

Three run configurations are supplied:

| File | Purpose | Scale | Status |
| --- | --- | --- | --- |
| `configs/smoke.toml` | Short connectivity check | 384 players, 3 companies, 4 games, 3 one-day cycles; full history and memory ledger | `SYNTHETIC` and executable only as a non-campaign run |
| `configs/policy_prototype.toml` | Reproducible seven-scenario policy prototype | 1,000 players, 14 days, 3 seeds, 7 scenarios | Strictly `synthetic`; tested but not empirically calibrated |
| `configs/base.toml` | Future architecture baseline | 50,000 players, 5 companies, 8 games, 365 one-day cycles; final-only history and SQLite ledger | `ILLUSTRATIVE` and deliberately blocked from current execution/campaign use |

Jurisdiction profiles and evidence contracts are stored separately in
`configs/jurisdictions.toml`. Source records are in
`data/provenance/sources.toml`.

The checked-in `jurisdictions.toml` uses profile schema version 2. It optionally accepts one
`[[monetary_conversion]]` table per jurisdiction with `jurisdiction_code`,
`source_currency`, `target_currency`, `method` (`FX` or `PPP`), positive exact
`rate_numerator` and `rate_denominator`, canonical ISO
`rate_period_start`/`rate_period_end` and
`target_price_period_start`/`target_price_period_end`, `estimand`,
`population_base`, `comparison_group`, provenance `status`, non-empty
`source_ids`, and canonical `retrieved_on`. The rounding contract also requires
the fixed `nearest_minor_unit_half_away_from_zero` method, a `rounding_scope` of
`PER_OBSERVATION` or `AFTER_AGGREGATION`, and a named `aggregation_unit`.
Jurisdiction rows may also declare `simulation_monthly_anchor_cents` and
`currency_scale_status`.

The loader supplies no conversion defaults. The checked-in file has no
conversion tables, so its four local money profiles remain deliberately
non-comparable and campaign-blocking. A future pooled campaign must provide
complete calibrated coverage with one common comparison signature and exact
coherence between conversion rates and internal scales. The target price period
must equal the rate period until a separate price-adjustment/deflator contract
exists. Rate sources must use the method-specific `foreign_exchange_rate` or
`purchasing_power_parity` support scope and the same canonical date interval.
The retrieval date cannot predate the rate-period end. Unknown
conversion-table keys are rejected.

Profile schema version 3 adds required `conversion_id` and `rate_binding_id`
fields. Each binding must resolve through a separately versioned
`source_bundle.toml`, whose own digest binds the exact source catalogue. The
bundle verifier re-reads a regular non-link CSV artifact, checks its declared
byte length and SHA-256, executes one whitelisted canonical exact-rational CSV
recipe, and requires the extracted currencies, direction, method, dates,
source, jurisdiction, numerator, and denominator to match the conversion.
Unknown fields, path traversal, links/reparse points, mutation, ambiguous rows,
floats, scientific notation, and unreduced rationals fail closed.

Those are structural and reproducible-extraction checks, not a substantive
comparability attestation. Source extraction, source-bundle signature,
output/design, population, and external preregistration are independent gates.
The checked-in source bundle is empty, `ILLUSTRATIVE`, and has signature status
`MISSING`; the checked-in jurisdiction file remains version 2 with zero
conversions. Public comparability and campaign validation therefore remain
false even if a test fixture clears the source-extraction subgate.

Legacy profile schema versions 1 and 2 remain readable on their exact historical
surfaces. Version 1 rejects monetary fields; version 2 rejects version-3 binding
fields. The profile-lineage snapshot has likewise advanced to version 3, with
frozen minimal v1/v2 readability fixtures, locked registered projection
fingerprints, and explicit downgrade rejection.

Campaign validation also requires the complete bundle to match its hashed
jurisdiction and source-registry files. Registered lineage is reloaded and
re-attested when a manifest is built; an in-memory or changed bundle cannot
self-promote by assigning calibrated status labels.

## Synthetic policy prototype schema

`configs/policy_prototype.toml` is self-contained for the policy batch. It does
not silently inherit the legacy `base.toml` or `smoke.toml` behavioural values.
The seven intervention vectors are named, versioned source-code definitions in
`microtx_sim.causal.scenarios`; the TOML selects cohort, decision, harm,
financing, and export assumptions shared by those scenarios.

The required scenarios are:

1. `baseline_f2p`;
2. `transparent_direct_price`;
3. `no_random_rewards`;
4. `no_time_limited_pressure`;
5. `spending_cap_cooling_off`;
6. `safe_fixed_price_subscription`;
7. `epgc`.

Within each seed, all seven scenarios start from the same synthetic cohort and
semantic random coordinates. Schema-2 fields named `*_effect_vs_safe` require
`safe_fixed_price_subscription` as their fixed reference; a future generic
contrast table will need a separately versioned contract. Scenario names do not assert that the
underlying parameters describe any observed game or jurisdiction.

### Policy `[meta]`

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | non-empty string | Run name stored in CLI output and `manifest.json`. |
| `provenance_status` | string | Must be exactly `synthetic`. |
| `notes` | string | Interpretation warning retained in the manifest. |

### `[policy_run]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `seeds` | non-empty list of unique integers in `[0, 2^64 - 1]` | Independent replications. Seeds are canonicalised into ascending order; a cohort is reused across scenarios within each seed. |
| `days` | non-negative integer days | Evaluation horizon; zero is available for structural edge tests. |
| `player_count` | non-negative integer | Synthetic cohort size; zero-player structural runs are supported. |

The supplied file uses seeds `101`, `202`, and `303`, 14 days, and 1,000
players. Changing these values changes the synthetic experiment and therefore
the effective-config and run-input hashes recorded in the manifest. Duplicate
seeds are rejected; input order is not experiment semantics and cannot change
aggregate results.
Manifests retain both JSON integer seeds and canonical decimal strings. Consumers
whose JSON number implementation cannot exactly represent integers above
`2^53 - 1` must use `batch.seed_decimal_strings`.

### `[decision]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `step_minutes` | positive divisor of 1,440 | Duration of one within-day action slot. |
| `temperature` | finite float in `(0, 5]` | Scale of stochastic utility shocks. |
| `habit_persistence` | float in `(0, 1]` | Retention of the previous habit state. |
| `habit_learning_rate` | float in `(0, 1]` | Update rate for habit formation. |
| `reinforcement_learning_rate` | float in `(0, 1]` | Update rate for reward-prediction state. |

The action set contains play, purchase, stop, sleep, study/work, socialise,
exercise, and another outside activity. These parameters are research variables,
not settings recommended for commercial optimisation.

### `[harm]` and `[harm_weights]`

`[harm]` controls the synthetic mapping from spending/exposure and displaced
activities to harm components:

| Field | Type | Meaning |
| --- | --- | --- |
| `affordable_spending_share` | fraction in `[0, 1]` | Spending share treated as affordable before other harm modifiers. |
| `opaque_spending_weight` | fraction in `[0, 1]` | Contribution of price opacity to spending harm. |
| `random_reward_spending_weight` | fraction in `[0, 1]` | Contribution of paid random-reward exposure. |
| `time_pressure_spending_weight` | fraction in `[0, 1]` | Contribution of limited-time pressure. |
| `sleep_debt_weight` | fraction in `[0, 1]` | Contribution of accumulated sleep debt. |

`[harm_weights]` contains non-negative weights for `monetary`,
`opportunity_cost`, `sleep`, `education_work`, `family_social`, and `wellbeing`.
At least one weight must be positive. Component results remain available even
when a weighted composite is reported.

### `[opportunity_valuation]`

This section contains non-negative integer simulation-cent proxies per displaced
hour. Adult fields cover sleep, work/study, social activity, and physical
activity. Youth fields cover sleep, education, family/social activity, and
physical activity. Youth valuation does not use wages as its primary measure.

These values are transparent synthetic assumptions. Simulation cents are not a
currency, market wage, willingness-to-pay estimate, or empirical welfare value.

### `[producer]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `development_cost_cents` | non-negative simulation cents | Fixed development cost. |
| `maintenance_cost_cents_per_day` | non-negative simulation cents/day | Maintenance cost multiplied by the horizon. |
| `institutional_license_count` | non-negative integer | Assumed external licence demand. |
| `institutional_license_price_cents` | non-negative simulation cents | Non-EPGC price per institutional licence. |
| `non_targeted_sponsorship_revenue_cents` | non-negative simulation cents | Revenue not conditioned on player behaviour. |
| `accessibility_eligible` | boolean | Eligibility for the accessibility bonus. |
| `multilingual_support_eligible` | boolean | Eligibility for the multilingual bonus. |
| `cultural_value_eligible` | boolean | Eligibility for the cultural-value bonus. |
| `safety_certified` | boolean | Eligibility for the safety-certification bonus. |

### `[epgc]`

All money fields use non-negative integer simulation cents. The section defines
payments per eligible access and institutional licence, an availability
payment, four auditable bonuses, a prohibited-mechanic penalty, a clawback in
basis points, and a maximum public budget. The evaluator reports the exact
minimum public contribution needed for non-negative safe profit and whether it
is feasible under the cap.

No EPGC field rewards playtime, retention intensity, conversion, or player
spending. This is a stylised policy calculation, not a representation of an
existing programme, legal entitlement, or subsidy application.

### `[output]`

| Field | Type | Meaning |
| --- | --- | --- |
| `output_dir` | non-empty path | Default artifact directory, resolved relative to the repository root. |
| `histogram_bins` | positive integer | Shared bin count for distribution SVGs. |
| `include_player_rows` | boolean | Enables synthetic player-level `player_outcomes.csv`. |
| `run_sensitivity` | boolean | Enables the OAT grid for `policy-batch`; `reproduce` always includes it. |

A complete `reproduce` run writes the 13 artifacts documented in
[Usage](usage.md). `--output` changes only the destination; it does not change
the model or execution-input hash.

The manifest preserves the released `config_sha256` meaning: it is the digest
of the file bytes observed when export builds the manifest, not proof of the
bytes parsed earlier. `effective_config_sha256` hashes the normalized typed
configuration object, and `run_input_sha256` hashes the exact batch design,
resolved model inputs, and profile fingerprint that produced the result. Export
fails before touching its destination if the supplied configuration differs
from the retained execution inputs.

### Policy validation and execution

```text
python -m microtx_sim policy-validate configs/policy_prototype.toml
python -m microtx_sim policy-batch configs/policy_prototype.toml
python -m microtx_sim policy-sensitivity configs/policy_prototype.toml
python -m microtx_sim reproduce configs/policy_prototype.toml
```

See [Synthetic policy prototype](policy_prototype.md) for the implemented
equations, scenario differences, and interpretation boundary.

## Legacy market-world scenario schema

### `[meta]`

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Scenario identifier reported by the CLI and run metadata. |
| `provenance_status` | enum | One of `CALIBRATED`, `ANCHORED`, `ILLUSTRATIVE`, or `SYNTHETIC`. |
| `notes` | string | Human-readable scope and interpretation warning. |

Campaign mode requires `provenance_status = "CALIBRATED"`,
`run.allow_synthetic = false`, and `run.ledger_backend = "sqlite"`. World
construction and orchestration additionally require an explicit non-temporary
persistent ledger; the configuration value alone does not create a campaign
artifact. A synthetic non-campaign scenario instead requires
`run.allow_synthetic = true`.

### `[run]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `seed` | integer in `[0, 2^64 - 1]` | Root coordinate for deterministic counter-based randomness. Negative, boolean, floating-point, and out-of-range values are rejected rather than wrapped. |
| `cycles` | positive integer | Number of ticks requested by the scenario. |
| `tick_days` | positive integer days | Calendar days advanced by one tick. |
| `player_count` | positive integer | Number of consumer rows. |
| `chunk_size` | positive integer | Consumer rows processed per dense choice block; changes memory, not alternatives. |
| `allow_synthetic` | boolean | Allows synthetic dependencies in structural, non-campaign runs. |
| `step_history_retention` | `full` or `final_only` | Retains every completed `WorldStep`, or only the latest successfully completed step. Omission defaults to `full` for compatibility. |
| `ledger_backend` | `memory` or `sqlite` | Uses an in-memory SQLite database or a file-backed SQLite database for the authoritative append-only ledger. Omission defaults to `memory` for compatibility. |

The non-campaign orchestrator additionally rejects more than 32 cycles or more
than 5,000 players. This prevents an accidental large run. Campaign mode is a
separate explicit API choice and has stricter evidence gates.

`step_history_retention = "final_only"` removes the `O(T·P)` in-memory
`WorldStep` history term without changing any simulation phase, returned step,
final outcome, or paired estimand. It replaces the retained step after each
successful tick, including across repeated `step()` or `run()` calls; a fresh
world therefore has empty history. `world.audit_count` remains cumulative over
all successfully completed steps. The setting does not bound aggregate recorder
summaries, popularity history, caller-held results, or temporary arrays. Future
run manifests and checkpoints must record the normalized effective retention
mode.

`ledger_backend = "sqlite"` bounds Python-retained ledger history by streaming
entries to a database file; the file itself remains `O(E)` in the number of
transfers. If no path is supplied to `World.create`, a non-campaign world owns a
temporary database and deletes it on `close()`. `World.create(ledger_path=...)`
creates a fresh persistent database and refuses an existing database, seal, or
SQLite journal/WAL sidecar.
Alternatively, a caller can inject a fresh `Ledger.create(...)`; caller-owned
ledgers remain open when the world closes so the caller can seal them. Storage
paths are runtime infrastructure, not causal parameters, and paired branches
must use distinct physical stores.

### `[market]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `company_count` | positive integer | Number of strategic company agents. |
| `game_count` | positive integer | Number of represented games; must be at least the company count. |
| `stat_dimensions` | integer 2–12 | Dimensions in the competitive content frontier. |
| `ranking_interval` | positive integer days | Time between public ranking events. |
| `firm_decision_interval` | positive integer days | Time between company decision events. |

Content search enumerates subsets of `stat_dimensions`, so its declared finite
candidate set grows exponentially. The upper bound of 12 is a structural guard.

### `[information]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `public_signal_noise` | float in [0, 1] | Noise scale applied to published game scores. |
| `public_signal_delay` | non-negative integer days | Minimum age of truth data eligible for publication. |
| `research_report_cost_cents` | positive integer simulation cents | Reference cost of a paid research action; the firm factory scales it by operating size. |
| `research_noise` | float in [0, 1] | Residual uncertainty of purchased research; lower means more precise. |

The public board is a released signal, not latent popularity. Research cost is
charged through company accounting; it does not grant access to `World`.

### `[behavior]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `game_choice_temperature` | float in (0, 1] | Scale of consumer choice sensitivity. |
| `switching_cost` | float in [0, 1] | Utility penalty for leaving the current game. |
| `household_peer_influence` | float in [0, 1] | Synthetic sensitivity coefficient for lagged, leave-one-out household-peer discovery and game-choice utility; it is not an estimated network effect. |
| `base_purchase_logit` | finite float | Baseline purchase propensity before heterogeneous covariates. |
| `unauthorised_card_hazard_per_exposed_minor_day` | probability in [0, 1] | Daily hazard conditional on all minor exposure conditions, not population prevalence. |
| `essential_spend_share` | fraction in [0, 1] | Share removed when converting adult monthly disposable income to periodic liquid inflow. |
| `harm_decay` | float in [0, 1] | Persistence/decay factor for dynamic harm state. |
| `daily_credit_interest_rate` | fraction in [0, 1] | Daily interest applied to used player credit. |

The unauthorised-card hazard and essential-spend share also have shared evidence
contracts. `World.create()` requires exact equality between the profile contract
and run-level value, then uses the run-level value. This makes divergence
explicit. `household_peer_influence` has no empirical contract: the configured
value is runtime-consumed, synthetic, and sensitivity-only. A value of zero
recovers the pre-peer choice path exactly.

### `[regulation]`

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `audit_interval` | positive integer days | Recurrence of government audit review. |
| `subsidy_interval` | positive integer days | Recurrence of subsidy review. |
| `maximum_fine_cents` | positive integer simulation cents | Maximum fine passed into compliance truth. |
| `audit_sensitivity` | probability in [0, 1] | Baseline probability that audit evidence detects a real breach before evasion effects. |
| `audit_specificity` | probability in [0, 1] | Probability of correctly returning no breach when none exists. |
| `random_audit_fraction` | fraction in [0, 1] | Audit capacity reserved for non-risk-ranked/random targets. |

Run-level sensitivity, specificity, and random fraction overwrite the loaded
StateAgent values for the scenario. Audit capacity, inspection cost, treasuries,
and initial subsidy budgets come from the jurisdiction profile and are currently
synthetic.

### `[causal]`

| Field | Type | Meaning |
| --- | --- | --- |
| `common_random_numbers` | boolean | Must be true for `run_paired_worlds()`. |
| `estimand` | string | Name attached to the reported `RegimeEffect`. |
| `record_individual_outcomes` | boolean | Retains the latest individual snapshot in addition to aggregate summaries. |

This section declares execution behaviour; it does not create an intervention.
Treated and control interventions are explicit Python objects.
`record_individual_outcomes` controls only `OutcomeRecorder.latest`; it is
independent of `[run].step_history_retention`, and every returned `WorldStep`
still contains its immutable outcome.

### Calendar validation

The following intervals must be exactly divisible by `tick_days`:

- public ranking;
- company decision;
- audit review;
- subsidy review;
- public-signal delay;
- the fixed 30-day income renewal.

The model rejects misalignment rather than approximating event times. Intervention
changes to audit or subsidy intervals are subject to the same rule.

### Input precedence

Effective inputs enter through several layers:

1. source records describe publishers and narrow support scopes;
2. jurisdiction contracts describe country/profile values and status;
3. `load_profile_bundle()` constructs country and StateAgent profiles;
4. the selected run scenario supplies behaviour, information, market, and
   regulation parameters;
5. `World.create()` applies validated scenario overrides to state audit accuracy;
6. explicit causal interventions modify mechanism caps, audit regimes, or
   subsidy regimes after world construction.

No layer silently promotes provenance. An intervention changes a simulated
regime; it does not make the underlying parameter calibrated.

### Validation commands

```text
python -m microtx_sim policy-validate configs/policy_prototype.toml
python -m microtx_sim validate configs/smoke.toml
python -m microtx_sim smoke configs/smoke.toml
```

`policy-validate` checks the strict synthetic policy schema without running a
cohort. `validate` loads the legacy scenario and profile contracts without
running a world. `smoke` creates a legacy market world and executes only its
short guarded run. Errors are returned on standard error with exit status 2.
