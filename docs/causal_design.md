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

The strategic paired-world runner applies the same principle independently: it
constructs two worlds, checks equality of causally relevant player, game, firm,
state, and jurisdiction state before treatment, and then applies explicit
treated and control interventions. Null-versus-null equality is a tested
software invariant.

## Repeated seeds and uncertainty summaries

`PolicyBatchSpec` requires unique integer seeds and all seven scenarios. Each
seed creates a new independent synthetic cohort; within that seed, all
scenarios remain paired. For a scalar outcome with replication values `x_r`,
the batch reports:

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

## Sensitivity design

The implemented one-at-a-time sensitivity runner varies declared mechanism,
affordability, and decision parameters while reusing the same cohorts and
random coordinates within each seed. It records harm, revenue, opportunity
cost, EPGC contribution, Monte Carlo dispersion, expected monotonic direction,
observed monotonicity, and an instability flag.

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
