# Synthetic policy prototype: technical reference

## Scope

This document specifies the reproducible seven-scenario policy prototype. It is
technical documentation for source code, configuration, tests, and generated
artifacts. It is not a paper, policy recommendation, legal analysis, subsidy
application, clinical instrument, or empirical validation report.

The prototype uses synthetic players and illustrative coefficients to exercise
the causal workflow end to end:

```text
strict TOML configuration
    -> seeded pre-treatment cohort
    -> seven branch-local scenario runs
    -> welfare and producer accounting
    -> effects against the safe reference
    -> repeated-seed summaries and sensitivity diagnostics
    -> CSV, JSON, Markdown, and SVG artifacts
```

The richer strategic market simulator remains available separately. The policy
prototype currently prioritises controlled welfare and financing comparisons;
it does not reproduce all company, competition, audit, collusion, or regulator
feedback from `microtx_sim.core.world`.

## Module map

| Module | Responsibility | Main public symbols or outputs |
| --- | --- | --- |
| `microtx_sim.policy_config` | Strictly parses the synthetic prototype TOML and rejects unknown or ambiguous fields. | `PolicyPrototypeConfig`, `PolicyOutputConfig`, `load_policy_config()` |
| `microtx_sim.domain.monetisation` | Defines the explicit intervention vector, transparency/pressure summaries, cap, and cooling-off constraints. | `MonetisationVector` |
| `microtx_sim.causal.scenarios` | Declares exactly seven stable scenarios and the safe reference-compatible catalogue. | `ScenarioId`, `ScenarioSpec`, `required_scenarios()`, `scenario_by_id()` |
| `microtx_sim.consumers.population` | Creates the seeded heterogeneous demographic and financial cohort. | `initialize_player_table()` |
| `microtx_sim.consumers.welfare` | Creates aligned pre-treatment commitments, obligations, enjoyment, vulnerability, habit, reinforcement, sleep, and wellbeing state. | `PlayerLifeTable`, `initialize_player_life()` |
| `microtx_sim.consumers.decision` | Evaluates the complete life-action set with interpretable utilities and counter-based Gumbel shocks. | `LifeAction`, `DecisionParameters`, `choose_life_action()` |
| `microtx_sim.simulation.policy_day` | Advances branch-local time, resources, safeguards, activity, purchases, revenue source, habit, reinforcement, and wellbeing. | `PolicyState`, `advance_policy_day()` |
| `microtx_sim.metrics.harm` | Computes multidimensional harm, harmful/unplanned spending, displaced activities, and adult/youth opportunity-cost proxies. | `HarmComponent`, `WelfareHarmResult`, `compute_welfare_harm()` |
| `microtx_sim.funding.epgc` | Evaluates the pure integer-cent EPGC payment and safe-profit equations. | `EPGCPolicy`, `EPGCFirmInputs`, `EPGCResult`, `evaluate_epgc()` |
| `microtx_sim.simulation.policy_orchestrator` | Clones one branch, runs its days, computes welfare, revenue composition, producer viability, high-risk flags, and EPGC output. | `ProducerAssumptions`, `PolicyScenarioResult`, `run_policy_scenario()` |
| `microtx_sim.causal.batch` | Runs all seven scenarios for every seed on the same cohort, pairs each with the safe reference, aggregates uncertainty, and retains the exact population profile tuple and its content fingerprint. | `PolicyBatchSpec`, `PolicyBatchResult`, `run_policy_batch()` |
| `microtx_sim.analysis.sensitivity` | Runs one-at-a-time grids with common cohorts, expected-direction checks, and instability flags. | `SensitivityCase`, `SensitivityResult`, `run_sensitivity_analysis()` |
| `microtx_sim.outputs.schema` | Fixes schema version, exhaustive CSV column contracts, and the complete artifact set. | `OUTPUT_SCHEMA_VERSION`, column tuples, `POLICY_ARTIFACT_FILENAMES` |
| `microtx_sim.outputs.writers` | Writes deterministic UTF-8 CSV/JSON/text files through same-directory atomic replacement. | `write_csv_atomic()`, `write_json_atomic()`, `write_batch_artifacts()` |
| `microtx_sim.outputs.manifest` | Captures configuration, profile-input and source digests, profile contracts, Git state, environment, seeds, cohort digests, scenarios, equations, and scope limits. | `build_run_manifest()` |
| `microtx_sim.outputs.plots` | Produces dependency-free accessible SVG charts from exported values. | Harm/spending histograms, frontier, decomposition, and EPGC chart writers |
| `microtx_sim.outputs.export` | Coordinates tables, manifest, human summary, charts, hashes, and final artifact-set verification. | `export_policy_batch()`, `render_human_summary()` |
| `microtx_sim.cli` | Exposes validation, batch, sensitivity-only, and complete reproduction commands. | `policy-validate`, `policy-batch`, `policy-sensitivity`, `reproduce` |

## Configuration and execution

The supplied configuration is `configs/policy_prototype.toml`. It declares
synthetic provenance, seeds, horizon, cohort size, decision parameters, harm
parameters and weights, adult/youth opportunity valuations, producer costs and
safe revenue, EPGC payment rules, and output options.

Policy CLI commands load one `ProfileBundle` and pass that same immutable bundle
to the scenario batch and sensitivity analysis. The result records a canonical
JSON snapshot and SHA-256 fingerprint of every `CountryProfile` field plus the
bundle's metric and money-scale contracts. The manifest also records the exact
jurisdiction-file and source-register hashes and the validated global retrieval
date. Programmatic callers may still supply a custom `country_profiles` tuple;
its exact contents are fingerprinted but its lineage is explicitly labelled
`unregistered_custom_profiles`, with no default-file provenance claimed.

Validate without running a batch:

```text
python -m microtx_sim policy-validate configs/policy_prototype.toml
```

Run the configured seven-scenario batch and export its artifacts:

```text
python -m microtx_sim policy-batch configs/policy_prototype.toml
```

Run only the one-at-a-time sensitivity grid:

```text
python -m microtx_sim policy-sensitivity configs/policy_prototype.toml
```

Reproduce the batch, sensitivity analysis, tables, metadata, summary, and
charts in one command:

```text
python -m microtx_sim reproduce configs/policy_prototype.toml
```

Use `--output PATH` to select another artifact directory. These commands run a
bounded synthetic prototype, not the uncalibrated large market campaign.

## Scenario specification

The catalogue contains exactly:

1. `baseline_f2p`;
2. `transparent_direct_price`;
3. `no_random_rewards`;
4. `no_time_limited_pressure`;
5. `spending_cap_cooling_off`;
6. `safe_fixed_price_subscription`;
7. `epgc`.

The three single-mechanism alternatives start from the declared baseline and
change a narrow set of coordinates. The spending-cap scenario adds multiple
related safeguards. The final two scenarios change the business model as well
as the mechanism vector. `safe_fixed_price_subscription` is the default
reference for `Delta H`; `epgc` combines the safe vector with public financing.
Personalised offers are disabled throughout.

The complete vector and current illustrative values are generated by
`required_scenarios()` and copied into `manifest.json`. The manifest, rather
than this prose summary, is authoritative for any particular run.

## Player decision equation

Each time step evaluates play, purchase, stop, sleep, study/work, socialise,
exercise, and another-activity alternative. If `V_ia` is the deterministic
utility for player `i` and action `a`, the selected action is:

```text
a_i* = argmax_a [ V_ia + temperature * epsilon_ia ]
epsilon_ia ~ standard Gumbel field
```

The Gumbel field is produced by `CounterRNG` at stable semantic coordinates,
which yields the multinomial-logit interpretation without a mutable random
cursor. Deterministic utilities combine baseline enjoyment, obligations,
sleep/activity urgency, habit, reinforcement, affordability, transparency,
purchase pressure, personal susceptibility, and pre-commitment safeguards.

The purchase alternative is infeasible when price exceeds available budget,
guardian consent is absent where required, the spending cap would be exceeded,
or cooling-off is active. Purchase charges are integer cents and may not exceed
available resources. Utility coefficients are declared model assumptions in
source/configuration, not behavioural estimates or optimisation advice.

## Harm and opportunity-cost equations

The composite reporting view is:

```text
H_i = w_M M_i + w_OC OC_i + w_S S_i + w_E E_i + w_F F_i + w_W W_i
```

All six component scores remain available separately. Monetary harm does not
equal total spending. It depends on the union of:

- spending beyond the player's pre-committed limit;
- opacity, paid randomness, and time pressure;
- financial strain relative to disposable resources.

Opportunity cost is zero unless play exceeds the planned gaming/leisure
allocation and plausibly displaces another activity. The model records
displaced sleep, work/study, family/social, and physical-activity minutes.

Adults and youth-like synthetic profiles have separate monetary-proxy rates:

```text
OpportunityProxy_i = sum_k DisplacedHours_ik * ValueRate_group(i),k
```

Youth rates are education/welfare resource proxies; they are not wage values,
and the non-monetary component scores remain the primary youth outcomes. All
rates and weights require empirical calibration before substantive use.

## Counterfactual pairing and uncertainty

For each seed, the population and pre-treatment life state are initialised once
and reused as immutable inputs to all seven branches. Mutable state is deep-
copied inside each scenario. The same `(seed, player_id, tick, stream,
draw_index)` random coordinates are queried across branches.

The default contrast is:

```text
Delta H_i(s, r) = H_i(s, r) - H_i(safe_fixed_price_subscription, r)
```

Repeated seeds change the synthetic cohort and random field. Scenario summaries
report mean, sample variance, standard deviation, and:

```text
normal 95% Monte Carlo interval = mean +/- 1.96 * sd / sqrt(seed_count)
```

These intervals describe finite-run simulator variation. They do not include
parameter uncertainty, empirical sampling uncertainty, model uncertainty, or
real-population inference.

## Revenue and producer accounting

Revenue is reconciled into:

- direct purchases;
- opaque virtual currency;
- paid random rewards;
- fixed-price access;
- subscriptions;
- public-contract revenue;
- institutional licensing;
- non-targeted sponsorship.

For the simplified producer in the policy runner:

```text
ProducerProfit = TotalRevenue - DevelopmentCost - MaintenanceCost
```

The strategic company accounting in the market `World` is separate and richer.
The simplified producer result must not be read as a full company balance sheet,
tax account, or market-equilibrium viability estimate.

## EPGC financing

The EPGC is a pure policy-simulation module. Its API deliberately excludes
playtime, retention, conversion, and player spending. Public payment therefore
does not mechanically increase when simulated behavioural intensity increases.

The gross eligible public contract is:

```text
GrossEligibleContract =
    AccessRate * EligibleAccessCount
    + InstitutionalRate * EligibleInstitutionalLicenseCount
    + AvailabilityRate * AvailabilityPeriodCount
    + AccessibilityBonus
    + MultilingualBonus
    + CulturalValueBonus
    + SafetyCertificationBonus
```

Each bonus is included only when its corresponding eligibility/certification
flag is true. The pre-sanction award is:

```text
BudgetLimitedContract = min(GrossEligibleContract, MaximumBudget)
```

If prohibited mechanics are enabled, the configured clawback and fixed penalty
reduce recognised public-contract revenue, never below zero:

```text
PublicContractRevenue =
    BudgetLimitedContract - Clawback - AppliedPenalty
```

The safe-profit identity is exactly:

```text
Profit_safe =
    PublicContractRevenue
    + FixedPriceRevenue
    + InstitutionalLicensingRevenue
    + NonTargetedSponsorshipRevenue
    - DevelopmentCost
    - MaintenanceCost
```

The minimum net contribution needed to break even is:

```text
MinimumPublicContribution = max(
    0,
    DevelopmentCost + MaintenanceCost
    - FixedPriceRevenue
    - InstitutionalLicensingRevenue
    - NonTargetedSponsorshipRevenue
)
```

`feasible_under_budget_cap` compares this residual with the budget cap.
`sustainable_under_policy` evaluates the actual eligible, capped, and sanctioned
contract. All monetary calculations use checked integer cents; zero-cost cases
are valid and products or sums outside signed `int64` are rejected.

This calculation is not a legal interpretation of an EU or national funding
instrument and is not evidence that a particular payment would be necessary,
proportionate, permissible, or effective.

## Sensitivity analysis

The implemented analysis is one-at-a-time. It currently supports paid random
rewards, time-limited offers, opaque currency, affordable-spending share, and
decision temperature. Every level reuses the same cohort and common random
field within each seed.

For each level it reports harm, revenue, opportunity cost, EPGC contribution,
between-seed dispersion, normal Monte Carlo intervals, expected monotonic
direction, observed monotonicity, and an instability flag. A parameter is
flagged when its expected direction fails or its between-seed coefficient of
variation exceeds the configured threshold.

This is a face-validity diagnostic. It does not cover parameter interactions,
joint uncertainty, alternative structural equations, or empirically estimated
parameter distributions.

## Artifact contract

`export_policy_batch()` verifies the complete artifact set before returning.

| Artifact | Contents |
| --- | --- |
| `seed_results.csv` | One aggregate row per seed and scenario, including revenue composition, harm, opportunity cost, enjoyment, high-risk profile, EPGC value, and safe-reference effects. |
| `scenario_summary.csv` | Repeated-seed means, variance/SD fields, and normal 95% Monte Carlo intervals. |
| `player_outcomes.csv` | Optional synthetic player-level distribution rows for harm, spending, burdens, enjoyment, and high-risk flag. |
| `opportunity_cost_decomposition.csv` | Displaced-activity minutes, burden, and monetary proxies by scenario. |
| `epgc_financing.csv` | Public revenue, minimum contribution, cap, safe profit, feasibility, sustainability, clawback, and penalty by seed. |
| `sensitivity.csv` | Parameter levels, uncertainty, monotonicity, and instability diagnostics. |
| `manifest.json` | Schema/config/source and profile-input digests, exact profile snapshot, metric/money contract summaries, source retrieval date, actual profile codes, Git state, environment, command, seeds, cohort digests, scenario vectors, equations, assumptions, and scope limits. |
| `summary.md` | Concise human-readable synthetic scenario table and interpretation warning. |
| `harm_distribution.svg` | Baseline F2P composite-harm histogram. |
| `spending_distribution.svg` | Baseline F2P spending histogram. |
| `harm_revenue_frontier.svg` | Scenario mean harm versus producer revenue. |
| `opportunity_cost_decomposition.svg` | Baseline displaced-activity decomposition. |
| `epgc_subsidy_requirement.svg` | Minimum public contribution across EPGC seed runs. |

Each versioned CSV has an exhaustive ordered column contract. Policy exporters
reject undeclared row keys instead of silently extending that contract. Writers
also reject non-finite numbers and replace files atomically. The final manifest
records file sizes and SHA-256 digests for every non-manifest artifact.

Schema `2.0` preserves the released v1 prefix and complete non-empty header
order. Its breaking change is deliberate: empty seed, scenario-summary, and
sensitivity files now expose the same exhaustive columns as populated files.

## Assumptions requiring calibration

The following inputs are model assumptions, not empirical findings:

| Area | Calibration need |
| --- | --- |
| Population | Age, income, household resources, financial literacy, supervision, vulnerability, motives, and country weights. |
| Life state | Intended leisure and spending, sleep need/debt, obligations, enjoyment, impatience, FOMO, social susceptibility, habit, reinforcement, and wellbeing distributions. |
| Decision process | Every utility coefficient, temperature, time-step size, habit persistence, and reinforcement-learning rate. |
| Monetisation | Mechanism intensity scales, scenario values, purchase price, cap period and amount, cooling-off duration, and revenue-source mapping. |
| Harm | Affordability threshold, opacity/pressure/randomness mapping, component equations, high-risk thresholds, composite weights, and decay or persistence assumptions. |
| Opportunity cost | Displacement allocation and separate adult/youth proxy rates; youth non-monetary outcomes need dedicated validation. |
| Producer | Development and maintenance cost, access adoption, fixed/subscription mix, licensing quantity/value, sponsorship, and revenue attribution. |
| EPGC | Eligibility definitions, access/licence/availability rates, bonus criteria and amounts, cap, prohibited-mechanic definition, penalty, and clawback. |
| Uncertainty | Number and selection of seeds, convergence rule, interval method, sensitivity ranges, and interaction design. |

Calibration targets must be separated from held-out validation targets. Source
artifacts, transformations, population bases, price periods, currencies, and
sampling uncertainty must be versioned before any parameter is promoted from
synthetic or illustrative status.

## Current limitations

- The policy runner uses a simplified producer and does not yet integrate the
  full strategic market, company adaptation, rankings, audits, enforcement,
  collusion, or endogenous subsidy applications.
- The same-cohort design establishes internal branch comparability but cannot
  correct constructs or mechanisms omitted from the simulation.
- Behavioural and welfare equations have face validity only; no participant or
  observational data have been fitted or collected.
- The high-risk flag is an operational simulation threshold, not a diagnosis,
  prevalence estimate, or individual prediction.
- Opportunity-cost money values are reporting proxies and are not social-welfare
  estimates. Youth outcomes must not be converted primarily through wages.
- Normal intervals over a small seed set may be unreliable for skewed or
  heavy-tailed outcomes and exclude parameter and structural uncertainty.
- One-at-a-time sensitivity misses interactions and correlated uncertainty.
- EPGC accounting omits tax incidence, crowd-out, additionality, administration,
  fraud, appeals, certification timing, general-equilibrium effects, and legal
  eligibility.
- Revenue and cost values are simulation cents and have no declared real
  currency or price-year interpretation.
- SVG charts are deterministic summaries of generated values, not evidence of
  empirical validation.

## Technical next steps

1. Calibrate and externally validate the player, decision, harm, producer, and
   financing parameters using versioned sources and transformations.
2. Integrate the seven policy regimes with the strategic market `World` while
   preserving branch isolation and information boundaries.
3. Add convergence diagnostics and justify the seed count and interval method;
   retain exact paired effects at player and seed level.
4. Add global and interaction sensitivity designs, including uncertainty in
   harm weights, high-risk thresholds, opportunity values, and EPGC inputs.
5. Test subgroup estimands prospectively by jurisdiction, age, income, baseline
   vulnerability, and household constraints without turning the tool into a
   real-user profiler.
6. Replace illustrative producer and EPGC inputs with auditable cost,
   eligibility, certification, and payment contracts after legal and empirical
   review.
7. Benchmark runtime, memory, output size, checkpointing, and recovery before
   increasing cohort size or horizon.
8. Preserve the current synthetic-only runner as a regression fixture after a
   calibrated model is introduced.

## Interpretation rule

Every artifact from this prototype must be described as a conditional result of
synthetic assumptions. It may demonstrate software behavior, internal causal
pairing, accounting reconciliation, or sensitivity to declared parameters. It
must not be reported as real-world harm, a clinical risk estimate, a legal
finding, a recommended payment, or proof that the EPGC design would work in
practice.
