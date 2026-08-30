# Causal design

## Purpose and interpretation boundary

The prototype asks how much additional simulated harm is caused by a declared
mobile-game monetisation regime relative to a safer alternative, holding the
synthetic population and exogenous random field fixed. It also asks whether a
safe producer can remain financially viable under conventional revenue and a
capped European Public-Value Game Contract (EPGC).

All causal language in this document refers to interventions inside the
specified simulation. The population, behavioural coefficients, harm mapping,
producer accounts, and public-payment rules are synthetic assumptions. The
prototype does not estimate a real-world treatment effect, diagnose any person,
classify a real game, or establish an optimal or lawful funding policy.

## Two causal execution layers

The repository retains two complementary execution layers:

| Layer | Entry point | Purpose |
| --- | --- | --- |
| Strategic market pair | `microtx_sim.causal.paired_worlds.run_paired_worlds()` | Compares one explicit treated/control pair inside the heterogeneous market with companies, rankings, regulation, audits, and subsidies. |
| Seven-scenario policy batch | `microtx_sim.causal.batch.run_policy_batch()` | Compares the seven required monetisation and financing regimes over repeated seeds using the welfare-oriented player decision process. |

The policy batch is the current source of the scenario comparison tables,
distributional welfare outputs, EPGC calculation, sensitivity results, and
charts. It does not yet embed every strategic-company and regulator interaction
from the richer market `World`. Results from the two layers must therefore not
be silently combined.

## Primary estimand

Let `s` denote one scenario and let `safe` denote the declared safer reference,
`safe_fixed_price_subscription`. For player `i` under replication seed `r`, the
individual contrast is:

```text
Delta H_i(s, r) = H_i(do(s), r) - H_i(do(safe), r)
```

The repeated-seed batch estimand reported for scenario `s` is:

```text
Delta H(s) = (1 / R) * sum_r [ (1 / N) * sum_i Delta H_i(s, r) ]
```

The `do(...)` notation means that the scenario vector is set by the researcher
before the branch runs. It does not imply that the simulation by itself
identifies an effect in observed people. Spending, harmful spending, producer
revenue, and other outcomes can be contrasted in the same paired way.

The welfare-oriented composite stored by the policy prototype is:

```text
H_i = w_M M_i + w_OC OC_i + w_S S_i + w_E E_i + w_F F_i + w_W W_i
```

where the components are monetary harm, opportunity-cost burden, sleep burden,
education/work burden, family/social burden, and wellbeing loss. The complete
component vector is retained. The composite is an explicit reporting view, and
changing its weights is a model change that must appear in the run manifest.

The strategic market layer separately retains its established seven-column
harm state: financial stress, essential-spend displacement, debt, unauthorised
spending, loss of control, functioning impairment, and regret. Its
`compare_outcomes()` function preserves the entire player-by-component
difference matrix.

## Required scenario catalogue

`microtx_sim.causal.scenarios.required_scenarios()` returns exactly these seven
stable scenario identifiers. The numeric values are illustrative research
coordinates, not estimates or commercial recommendations.

| ID | Intervention relative to the catalogue baseline | Revenue interpretation |
| --- | --- | --- |
| `baseline_f2p` | High-pressure illustrative F2P vector with opaque currency, paid random rewards, progression pressure, time-limited offers, streak pressure, pay-to-progress, pay-to-win, social pressure, low purchase friction, no cap, and no cooling-off period. | Player purchases are decomposed into direct, opaque-currency, and paid-random-reward revenue. |
| `transparent_direct_price` | Removes opaque virtual currency, displays prices in real currency, and increases purchase friction while retaining the other baseline pressures. | Purchases remain transactional but their presentation is more transparent. |
| `no_random_rewards` | Sets paid random rewards to zero and otherwise retains the baseline vector. | No purchase revenue is attributed to paid random rewards. |
| `no_time_limited_pressure` | Sets time-limited offers to zero and otherwise retains the baseline vector. | Purchase deadlines are removed in isolation. |
| `spending_cap_cooling_off` | Applies a rolling spending cap, a 24-hour cooling-off period, real-currency display, and greater purchase friction. | Purchases are rejected when the cap or cooling rule binds. |
| `safe_fixed_price_subscription` | Uses the low-pressure safe vector with transparent fixed-price and subscription access. It is the fixed reference for schema-2 `*_effect_vs_safe` fields. | Revenue is assigned to fixed-price and subscription access rather than pressure-linked purchases. |
| `epgc` | Uses the low-pressure safe vector, a small transparent access price, and enables the EPGC calculation. | Safe conventional revenue may be supplemented by capped public-contract revenue. |

Personalised offers are disabled in every catalogue scenario. Scenario IDs,
the safe reference, mechanics, prices, horizon, and seed list are stored in the
manifest so a label alone never determines treatment.

The causal-design registry freezes the catalogue as a 7-by-17 atomic-factor
matrix and records all 49 directed pairwise contrasts. Each contrast carries
its exact factor differences, an identity/single-factor/bundle classification,
and descriptive roles for the catalogue checks and reported
`*_effect_vs_safe` outputs. These roles are retrospective diagnostics, not
planned estimands: the registry status is `RETROSPECTIVE_SYNTHETIC`, it is not
preregistered, and it always fails the campaign gate pending preregistration
and empirical calibration.

An optional prospective analysis-plan file can now freeze one primary directed
contrast plus secondary contrasts against the exact design digest, fixed seeds,
harm weights, projected-population input, exact `CountryProfile` input-lineage
fingerprint, and output contracts before scenario execution.
It does not alter the retrospective registry status: schema v1 is explicitly
`UNREGISTERED`, external timing is not proven, and both plan and resolved run
binding remain `campaign_ready=false`. See [Prospective analysis-plan
composition](analysis_plan.md).

Every manifest embeds the observed matrix and contrast snapshots, their
SHA-256 digests, the overall design digest, and the exact `run_input_sha256`
used for execution. A batch whose named scenarios contain custom factor values
may still be exported for descriptive work, but it is marked
`canonical_match=false`, records the exact factor mismatches, and adds the
`scenario_factor_matrix_not_canonical` campaign blocker. It is never silently
presented as the canonical design.

Only two catalogue comparisons are genuinely single-factor:
`baseline_f2p` to `no_random_rewards` changes paid randomness, and
`baseline_f2p` to `no_time_limited_pressure` changes time-limited offers. The
transparent-price comparison changes three mechanics, while the cap/cooling
comparison changes four. Relative to the safe reference, baseline differs on
15 factors and EPGC differs on three financing/access factors (fixed access
price, subscription price, and EPGC enablement). The EPGC result therefore
cannot be interpreted as an isolated EPGC-toggle effect.

## Same cohort, same initial state, and common random numbers

Within each seed, `run_policy_batch()` creates the synthetic `PlayerTable` and
pre-treatment `PlayerLifeTable` exactly once. Every scenario receives the same
player identifiers, demographics, budgets, immutable vulnerability, intended
spending and play commitments, obligations, initial sleep debt, wellbeing,
habit state, and other pre-treatment columns.

`run_policy_scenario()` deep-copies mutable life state before advancing a
branch. Post-treatment spending, time allocation, progression, habit,
reinforcement, and wellbeing therefore cannot leak into the next scenario. A
SHA-256 cohort digest is recorded for each seed and repeated on every
seed-scenario record.

`CounterRNG` addresses each stochastic draw by:

```text
(seed, entity_id, tick, stream, draw_index)
```

All branches for one seed query the same semantic random coordinates. A choice
or purchase that occurs only in one branch cannot consume a mutable generator
cursor and shift later shocks in another branch. Stream names and draw-index
meanings are part of the model version. Scenario order must not alter results.
Scenarios deliberately receive the same root seed; scenario-specific sub-seeds
would break this common-random-number pairing and are not derived.

The strategic paired-world runner applies the same principle independently: it
constructs two worlds, checks equality of causally relevant player, game, firm,
state, and jurisdiction state before treatment, and then applies explicit
treated and control interventions. Null-versus-null equality is a tested
software invariant.

Population-evidence schema version 1 is not itself a cohort assignment or
weighting mechanism. The policy batch can optionally use one exact projection
adapter: for each seed all seven scenarios share its assignment, and canonical
per-seed lineage binds the projection, ordered players, cohort, exact weights,
and a pre-treatment population balance artifact. This checks declared joint-
cell and runtime-membership conformance; it is distinct from the paired-world
branch-state balance report and is not empirical covariate balance. The market
runner supports the same optional adapter through `World.create`; omission
preserves legacy marginal initialization. Frozen output-v2-compatible CSVs
remain equal-player aggregates; weighted target estimands require the separate
standalone profile.

That legacy fallback is unavailable when `[analysis_plan]` is selected. A
planned policy configuration must resolve its declared projected adapter and
population/profile lineage before treatment; omission or mismatch is a
preflight error.

## Repeated seeds and uncertainty summaries

`PolicyBatchSpec` requires unique strict Python integer seeds in the unsigned
64-bit range `[0, 2^64 - 1]` and all seven scenarios. It rejects duplicates and
canonicalises accepted seeds into ascending order before execution, so caller
ordering cannot change floating-point aggregation order or serialized batch-table
bytes. Scenario results and cohort-digest metadata revalidate the same seed
domain rather than coercing foreign key types. Result containers require the
exact seed-by-scenario cross-product and canonicalise records by ascending seed
then declared scenario order. Each seed creates a new independent synthetic
cohort; within that seed, all scenarios remain paired. For a scalar outcome with
replication values `x_r`, the batch reports:

```text
mean = sum_r x_r / R
sample variance = sum_r (x_r - mean)^2 / (R - 1)
standard deviation = sqrt(sample variance)
normal 95% interval = mean +/- 1.96 * standard deviation / sqrt(R)
```

With a single seed, variance and interval width are reported as zero. These are
Monte Carlo diagnostics for the configured simulator, not confidence intervals
for a real population. A small seed count, non-normal tails, or unstable model
parameters can make the normal interval inadequate; convergence and alternative
interval procedures remain calibration-stage work.

Analysis-plan schema v2 applies the same declared normal convention to the one
plan-level primary, using the exact paired, population-weighted per-seed
realizations rather than the legacy scenario summaries. Population weights are
applied within seed; the fixed independent seeds are equally weighted. The
checked-in primary is `baseline_f2p - safe_fixed_price_subscription` for
composite simulated harm, so positive means more harm under the baseline and
negative means less. No secondary scenario is averaged into this contrast and
no outcome-dependent seed exclusion is permitted. Its separate metadata calls
the bounds a Monte Carlo interval for simulator output, never a confidence
interval for a real-world population.

## Reported outcomes

The batch retains one row per seed and scenario, repeated-seed summaries, and
optional synthetic player rows. Outputs include:

- total producer revenue, cost, and profit;
- revenue composition: direct purchase, opaque virtual currency, paid random
  rewards, fixed price, subscription, public contract, institutional licensing,
  and non-targeted sponsorship;
- total, unplanned, and harmful spending;
- mean, variance, median, upper-tail, component-level, and composite harm;
- adult and youth opportunity-cost proxies plus displaced sleep, work/study,
  family/social, and physical-activity time;
- sleep, education/work, family/social, and wellbeing burdens;
- enjoyment;
- the count and share of operationally high-risk simulated outcomes;
- EPGC revenue, minimum public contribution, cap feasibility, safe profit,
  penalty, and clawback;
- effects against the safe reference and repeated-seed uncertainty summaries.

A policy-run outcome is flagged `high_risk` when its configured operational rule
finds high composite harm, a large harmful-spending share, or a high sleep
burden. The exported profile includes age, minor share, budget, and baseline
vulnerability for the flagged synthetic group. This is a simulation-tail label,
not a clinical category or predicted diagnosis.

## EPGC causal role

The EPGC scenario changes financing without rewarding behavioural intensity.
Its safe-profit identity is:

```text
Profit_safe =
    PublicContractRevenue
    + FixedPriceRevenue
    + InstitutionalLicensingRevenue
    + NonTargetedSponsorshipRevenue
    - DevelopmentCost
    - MaintenanceCost
```

Public-contract eligibility is based on access, institutional licences,
availability, accessibility, multilingual support, cultural value, and safety
certification. The EPGC API has no playtime, retention, conversion, or player
spending input. A maximum budget caps the gross eligible contract. If prohibited
mechanics are enabled, a clawback and penalty reduce recognised public-contract
revenue.

The exact minimum net public contribution required for non-negative safe profit
is:

```text
MinimumPublicContribution = max(
    0,
    DevelopmentCost + MaintenanceCost
    - FixedPriceRevenue
    - InstitutionalLicensingRevenue
    - NonTargetedSponsorshipRevenue
)
```

`feasible_under_budget_cap` tests whether this residual fits within the declared
maximum budget. `sustainable_under_policy` separately tests whether the actual
eligible, capped, and sanctioned payment makes `Profit_safe >= 0`. The two flags
can differ when eligibility is insufficient or sanctions apply.

## Market interference

Individual no-interference assumptions do not hold. A monetisation intervention
can change player choices, time allocation, spending, revenue composition, and
welfare simultaneously. In the strategic market layer it can additionally
change popularity, switching, company decisions, rankings, audit signals,
subsidy applications, collaboration, and collusion.

The main contrast is therefore a market- or policy-regime effect conditional on
the selected runner. It is not a direct individual treatment effect isolated
from equilibrium spillovers. Separating direct and spillover effects would
require an explicit exposure mapping and a different assignment design.

## Identification assumptions inside the simulator

A policy contrast is internally interpretable only if:

1. every branch for a seed starts from the same pre-treatment cohort and state;
2. scenario construction is the only intentional branch difference;
3. random coordinates retain the same semantic meaning across branches;
4. branch-local mutable state is never reused to initialise another branch;
5. outcome definitions, weights, horizon, and accounting rules are identical;
6. the safe reference is declared before effects are inspected;
7. interference is included in the regime estimand rather than ignored;
8. revenue components and integer-cent accounts reconcile;
9. configuration, code revision, source-registry digest, seeds, and cohort
   digests accompany the output.

These conditions support causal comparisons inside the program. They do not
solve external validity, empirical calibration, structural misspecification,
measurement error, omitted constructs, or uncertainty about real institutions.

## Synthetic structural falsification checks

The strategic paired-world runner assesses the complete reachable initial
state of both independently created worlds before either intervention is
applied. The retained typed balance report records every checked path and any
exact value, type, shape, or mutable-alias mismatch. Shared mutable branch state
is forbidden even when the two references occur at different graph paths.
Mutable alias topology must also match within each branch, including memory
overlap between distinct NumPy views. Immutable configuration and profile
metadata may be shared, but mutable `ProfileBundle.state_agents` templates may
not; the paired runner clones those templates separately when an explicit
profile bundle is supplied. A mismatch fails the run before treatment or
simulation. Exact equality, rather than a statistical standardized-difference
threshold, is the right software contract because these branches are intended
to be independently mutable copies of one synthetic market.

Paired outcomes must have the same tick, expected array types and ranks, and
exactly equal ordered player, firm, and jurisdiction identifiers before
subtraction. Integer differences are checked before conversion to `int64`, so a
contrast cannot wrap silently. Player income is retained as an explicit
pre-treatment/exogenous negative-control difference. It must remain zero in
valid runs because none of the declared interventions changes the synthetic
income input; a nonzero difference raises a typed validation error.

The policy batch separately requires every same-seed branch to retain identical
player IDs, minor status, age, jurisdiction, baseline vulnerability, and
disposable budget with their canonical shapes and dtypes, as well as the exact
declared scenario, seed, horizon, and player count. It retains independently
owned read-only copies of every result and nested harm array, recomputes all four
reported effects from each seed's safe reference, and rejects invalid identity,
age, jurisdiction, vulnerability, budget, spending, activity, or harm domains.
The shared player-and-life cohort digest is checked after every branch so a
branch cannot contaminate later counterfactuals. Tests exercise independently
executed null branches, deliberately injected value, alias, and shared-state
imbalance, a non-null intervention with a zero income negative control,
rejection of a nonzero control, and exact recovery of planted synthetic outcome
shifts.

These checks can falsify branch isolation, alignment, subtraction, and estimator
code. Passing them does not demonstrate empirical covariate balance, validate a
negative-control assumption in observed data, or identify a real-world effect.

## Sensitivity design

The implemented one-at-a-time sensitivity runner varies declared mechanism,
affordability, and decision parameters while reusing the same cohorts and
random coordinates within each seed. It records harm, revenue, opportunity
cost, EPGC contribution, Monte Carlo dispersion, expected monotonic direction,
observed monotonicity, and an instability flag.

Each sensitivity result retains its exact batch specification, ordered cases,
normalized levels, instability threshold, fixed numerical tolerances, resolved
model inputs, and profile fingerprint. Rows are copied, validated against those
inputs, required to match the exhaustive output columns, checked for internal
variance/standard-deviation/interval/CV identities, canonically ordered, and
made immutable. Mechanic and affordability levels must lie in `[0, 1]`;
decision temperature must lie in `(0, 5]`. Combined export rejects a sensitivity
result produced from a different batch design or resolved input bundle before
writing any artifact.

One-at-a-time analysis does not identify interactions or provide a posterior
distribution. The configured grids and monotonic expectations are face-validity
checks over synthetic assumptions. Global, joint, and empirically informed
sensitivity analysis remains future work.

## Synthetic-only interpretation

No output from this prototype is an empirical prevalence, national spending
estimate, clinical finding, legal conclusion, or recommended public payment.
The manifest sets `synthetic_only=true` and
`empirical_validation_claimed=false`. The charts and tables demonstrate how the
declared model behaves and support reproducible software evaluation only.

See [Policy prototype](policy_prototype.md), [Model specification](model_spec.md),
[Data sources](data_sources.md), and [Limitations](limitations.md).
