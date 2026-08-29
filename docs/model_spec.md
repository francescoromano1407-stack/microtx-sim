# Model specification

## Scope

The model represents a competitive mobile-game market, not a playable game.
Games expose abstract quality, competitive integrity, novelty, prices,
monetisation mechanisms, and a multidimensional content frontier. This is enough
to model player choice, live-content updates, competitive pressure, spending,
firm strategy, regulation, and public funding without encoding characters,
combat, or any commercial title.

The central research objective is to separate harm caused by a monetisation
regime from the vulnerability players already had before exposure, while also
measuring whether firms can remain viable under safer design, enforcement, and
public support. The repository now contains both the original competitive-market
kernel and an aligned player-welfare policy runner. Both remain synthetic
research prototypes: their equations and country profiles are not calibrated for
real-world policy estimates.

## Information model

`World` contains the latent state required by the research kernel. No consumer,
company, or government policy receives a reference to `World`. Policies receive
immutable observations constructed by specialised systems. A value becomes
available only through an explicit public release, personal experience,
telemetry process, paid report, complaint, disclosure, or audit.

The intended chain is:

```text
latent truth -> sensor/report -> observation -> private belief -> intent
                    |              |
                    + cost/noise/delay
```

Generic signals record their source, observation time, availability time,
precision, and value. Domain-specific views provide the same boundary with
stronger typing. The main restrictions are:

- consumers cannot observe true popularity, other players' latent vulnerability,
  or company and government private state;
- companies cannot observe player-level latent harm, current true popularity,
  competitors' private state, or future audit selection;
- governments cannot observe the researcher's latent harm mean or unsafe-revenue
  classification and see compliance truth only through imperfect audit evidence;
- the public sees only the released score and rank, which can be delayed, noisy,
  and affected by promotion;
- only the research kernel can resolve hidden truth, and it must not pass that
  truth wholesale into an agent decision.

Information is therefore not free. Better company estimates require research
spending; delayed boards create stale market knowledge; complaint formation and
audit capacity limit government knowledge.

## Agents and heterogeneity

### Consumers and households

Consumers are stored in a columnar `PlayerTable` so population-scale operations
can be vectorised. It contains:

- age, jurisdiction, household membership, and jurisdiction-specific minor status;
- monthly disposable income, liquid funds, allowance, credit, and household funds;
- stored-payment access, guardian consent, and supervision;
- impulsivity, reward sensitivity, social susceptibility, loss aversion,
  financial literacy, and self-control;
- overlapping competition, collection, social, exploration, and relaxation
  motives;
- immutable baseline vulnerability and the market kernel's legacy dynamic harm
  diagnostics;
- awareness and current game.

The welfare policy layer adds an exactly aligned `PlayerLifeTable`, keyed by the
same `player_id`. Its pre-treatment columns are copied and write-protected:

- planned leisure, intended play, sleep need, work/study obligations, social
  obligations, and physical-activity need, all in minutes;
- baseline game enjoyment, financial sensitivity, delay discounting, social
  pressure susceptibility, scarcity/FOMO susceptibility, and baseline
  vulnerability;
- an intended spending limit in integer simulation cents.

Its branch-local dynamic columns are sleep debt, current-game progression, habit
strength, a signed reinforcement state, cumulative historical spending,
cumulative actual play, and wellbeing. Dynamic arrays are deep-copied before a
counterfactual branch runs; no post-treatment state is shared between branches.

`initialize_player_life` uses named counter-RNG streams and bounded illustrative
priors. Time allocations are rounded clipped normals conditional on age and
work/study load. Enjoyment, financial sensitivity, delay discounting, FOMO,
habit, and wellbeing are bounded affine functions of existing continuous traits,
resources, vulnerability, and independent normal draws. Intended spending is a
bounded fraction of disposable budget. These distributions are synthetic model
assumptions, support a zero-player cohort, and are not empirical or clinical
estimates.

The separately registered population-evidence schema can attest exact joint
age × household-income-band × household-type × gaming × pre-treatment
payer-history cells with rational mass. A further static population-design
schema can bind verified evidence to complete domains, declared source-unit
partitions and target counts, then produce deterministic exact-rational Hamilton
cell counts and analysis/expansion weights. Its partition identities remain
declarations, not proof of publisher authenticity or independent holdout.

Both checked-in population defaults are empty and `ILLUSTRATIVE`. The optional
`initialize_projected_player_table` helper can construct a table from already-
resolved projection cells, but no config, world, batch, sensitivity, or CLI path
selects it. The helper does not consume or revalidate a static
`PopulationApportionmentPlan`; it derives a separate runtime projection and uses
its own exact-mass Hamilton allocation with `cell_id` tie-breaking. Static source
household-income categories and runtime personal monthly disposable-income
intervals/modeled household sizes therefore require an explicit future adapter.

Gamer and payer-history labels stay in an attested population sidecar; they do
not set current game, payment access, or spending history. Consumers recompute
the nested assignment attestation and reject stale or mutated indices. The
sidecar assignment is included in the cohort digest when present, while ordinary
tables retain the legacy digest.

Exact weighted-mean, paired-difference, and deterministic weighted-quantile
algorithms now exist as an isolated estimand primitive. The specification binds
the supplied design weights and records evidence, projection, balance, metric-
contract, and dedicated output-profile digest declarations. It does not resolve
or reverify those declared artifacts. No existing reducer or writer invokes it,
and there is no registered target-population output profile. Configured runs
therefore still use the legacy marginal generator; output schema v3 preserves
the frozen v2-compatible CSV columns and unweighted synthetic-player semantics.
No population readiness gate is cleared and no full campaign has been run.

Traits and motives are continuous, heterogeneous, and partly correlated.
Categories such as casual, competitive, and collector are overlapping
descriptions rather than exclusive scripts. Spending segments—non-payer,
minnow, dolphin, and whale—are retrospective classifications based on realised
spending and spending burden, not an intrinsic behaviour flag.

Age and income change resources and decisions continuously. A minor's ordinary
household-card purchase requires consent. The rare unauthorised-card path is
possible only when the player is a minor, lacks consent, has stored-payment
access, has sufficiently low supervision, encounters the stochastic hazard, and
has household resources available. The amount is resource-capped and feeds a
separate unauthorised-spending outcome.

### Games

The competitive-market `GameTable` stores:

- company ownership;
- quality, competitive integrity, and novelty;
- a legacy six-coordinate mechanism matrix used by strategic company dynamics:
  power sales, random rewards, artificial scarcity, social pressure, price
  obfuscation, and payment-friction removal;
- a multidimensional competitive-stat frontier;
- an integer-cent price;
- active players, exact revenue, and latent popularity;
- a delayed/noisy public score and complete public rank.

Policy scenarios use a separate immutable `MonetisationVector` with fourteen
explicit research coordinates:

1. `direct_price_cents`;
2. `opaque_virtual_currency`;
3. `paid_random_rewards`;
4. `progression_gates`;
5. `time_limited_offers`;
6. `daily_streak_pressure`;
7. `pay_to_progress`;
8. `pay_to_win`;
9. `social_guild_pressure`;
10. `purchase_friction`;
11. `spending_cap_cents`;
12. `cooling_off_hours`;
13. `real_currency_price_display`;
14. `personalized_offers`.

Risk-oriented intensities are in `[0, 1]`; higher purchase friction is the one
continuous coordinate whose larger value is safer. Caps retain integer-cent
units and cooling-off periods retain hours. Personalised offers are disabled by
default. Derived `price_transparency`, `purchase_pressure`, and `risk_exposure`
scores are reporting coordinates, while the cap and cooling-off methods enforce
hard transaction constraints. The vector is an intervention schema, not a set
of commercial optimisation recommendations.

A content release searches the full configured finite set of boost rates and
proper non-empty subsets of statistic dimensions. A valid candidate improves at
least one dimension and weakens at least one other dimension, so no release
dominates the old frontier in every statistic. The selected release maximises
the acting company's perceived NPV, not an omniscient objective.

### Companies

Companies differ in cash, costs, risk aversion, compliance culture, ethical
weight, analytics ability, discounting, exploration, reputation sensitivity,
collusive trust, and private beliefs. Period telemetry contains flows since the
previous decision rather than cumulative lifetime revenue.

Available intents include:

- hold;
- release competitive content;
- adjust monetisation;
- buy research;
- invest in compliance;
- acquire users;
- propose collaboration;
- propose collusion;
- evade controls;
- apply for a subsidy.

Every company's intent is collected before any same-tick intent is resolved.
This prevents iteration order from revealing competitors' unresolved decisions.
Actions are filtered by affordability and then resolved by the kernel. Bilateral
agreements require exact reciprocal compatible proposals. Promotion changes the
public signal process, while research changes the firm's private information.
Persistent experimental mechanism caps are re-applied after later company
actions, including monetisation changes and collusive effects.

### Governments and regulators

Each jurisdiction has dated rule abstractions, a treasury, audit capacity,
inspection cost, an audit appropriation, a subsidy budget, policy priorities,
finite sensitivity and specificity, a random-audit floor, and private beliefs.

Audit targeting uses observable complaints, reported minor harm, reported spend
anomalies, and past public detections. The state chooses targets before the
kernel constructs audit truth. Evidence can miss a real breach and can create a
false positive; evasion reduces detection probability but does not erase the
underlying breach.

Subsidy applications carry a submission tick, eligible jurisdiction, requested
amount, quality, design-safety and accessibility proxies, job estimates, and
evidence age. Applications must predate the review. The current prototype assigns
each firm one synthetic home jurisdiction. Awards are constrained by both the
subsidy budget and treasury cash. The regulator scores observable application
proxies; it does not receive the researcher's latent unsafe-revenue share.

## Consumer dynamics

### Competitive-market choice

For each tick, each consumer evaluates every represented game currently known to
them plus an outside option. Computation is split into blocks to bound temporary
memory, but no game is sampled or pruned.

Choice utility combines public quality/rank, price burden, novelty, competitive
integrity, monetisation pressure, motives, literacy, switching cost, awareness,
personal experience, a lagged household-peer signal, and a stable idiosyncratic
shock. The peer signal is calculated from other simulated players in the same
household and their pre-tick current games. It excludes the focal player, never
crosses household boundaries, and affects both a separate peer-discovery draw
and utility in proportion to social susceptibility. The configured coefficient
is a synthetic sensitivity parameter; zero recovers the prior choice path. The
selected game then produces activity, abstract matches, noisy personal quality,
competitive rank, purchase consideration, package demand, resource allocation,
and harm diagnostics. This path remains the behavioural component of the
strategic market simulation.

Purchases cannot exceed available permitted liquidity and credit. Financial
mutations are preflighted before commit. Revenue is aggregated by game and firm
with integer arithmetic. Its seven legacy diagnostics—financial stress,
essential-spend displacement, debt, unauthorised spending, loss of control,
functioning impairment, and regret—remain available for compatibility and market
feedback. They are not the six-component welfare estimand defined below.

### Welfare activity choice

The policy runner divides each day into equal decision steps; the default is 30
minutes and `step_minutes` must divide 1,440 exactly. At every step every player
evaluates all eight actions: play, purchase, stop, sleep, study/work, socialise,
exercise, and other activity. No feasible action is sampled away.

For player `i`, action `a`, deterministic utility `V_ia`, temperature `tau`, and
an independently addressed standard-Gumbel shock `epsilon_ia`, the selected
action is

```text
a_i* = argmax_a [V_ia + tau epsilon_ia]
Pr(a_i = a) = exp(V_ia / tau) / sum_b exp(V_ib / tau)
```

The implementation uses the argmax form so common random coordinates remain
stable across counterfactuals. Play utility includes enjoyment, habit,
reinforcement, progression/streak/social mechanic exposure, sleep and work
urgency, and leisure overrun. Purchase utility includes purchase pressure,
delay discounting, vulnerability, habit, paid randomness, pay-to-progress,
pay-to-win, friction, price burden, and financial sensitivity. The other
utilities use obligation urgency and time-of-day windows.

Hard feasibility is separate from utility. A purchase receives utility
`-infinity` when its exact price exceeds available budget or the policy spending
cap, violates the cooling-off rule, or fails minor consent/payment-access rules.
A stochastic shock therefore cannot override those safeguards. The player's
intended limit remains a pre-treatment commitment used to identify unplanned
spending; fixed-price/subscription adoption additionally enforces it as a hard
eligibility rule.

Each action consumes exactly one step. For every player and completed day,

```text
sum_a action_minutes_ia = 1,440.
```

Sleep, work/study, social, and exercise choices reduce their corresponding
remaining obligations. The day processor asserts time conservation and exact
reconciliation of purchase charges with revenue-source columns.

### Habit, reinforcement, and progression

Let `engaged` denote play or purchase and let `stop` denote the explicit stop
action. Habit evolves as

```text
h_(t+1) = clip(rho_h h_t + eta_h I(engaged)
               - 0.60 eta_h I(stop), 0, 1).
```

The reinforcement state is a bounded reward-prediction signal, not an addiction
measure. With observed synthetic reward `R_t`, prediction
`P_t = 0.50 + 0.35 r_t`, and `delta_t = clip(R_t - P_t, -1, 1)`,

```text
r_(t+1) = clip(0.98 r_t + eta_r delta_t I(engaged), -1, 1).
```

Observed reward combines baseline enjoyment with stochastic streak exposure and
a pay-to-win purchase term. Progression grows through play, is slowed by
progression gates, and can receive an explicit pay-to-progress increment. All
coefficients are illustrative and exposed through `DecisionParameters` where
applicable.

## Welfare harm and opportunity cost

The policy outcome stores the six requested components separately:

```text
H_i = w_M M_i + w_OC OC_i + w_S S_i + w_E E_i + w_F F_i + w_W W_i.
```

Weights are non-negative reporting assumptions; the component matrix is never
discarded. The components are:

- `M`: harmful spending burden;
- `OC`: opportunity-cost burden;
- `S`: sleep burden;
- `E`: education/work burden;
- `F`: family/social burden;
- `W`: wellbeing loss.

Spending is not harmful merely because it occurred. Let `u_i` be the share of
current spending beyond the remaining intended limit, `o_i`, `r_i`, and `t_i`
the opacity, paid-random-reward, and time-pressure exposures, and `s_i` the
share under financial strain. The current implementation forms

```text
q_i = 1 - (1 - alpha_o o_i)(1 - alpha_r r_i)(1 - alpha_t t_i)
m_i = 1 - (1 - u_i)(1 - q_i)(1 - s_i).
```

Harmful spending is current spending times `m_i`, rounded to cents, bounded by
current spending, and never below the unplanned amount. Planned, transparent,
non-random, non-pressured spending within the affordability threshold therefore
has zero `M` incidence.

Opportunity cost also does not classify ordinary leisure gaming as harm. Excess
play is

```text
x_i = max(actual_play_i - planned_game_leisure_i, 0).
```

Observed shortfalls in sleep, work/study, social life, and physical activity are
allocated proportionally and their total attributed displacement is capped by
`x_i`. Thus a deficit with no excess play produces no gaming opportunity cost.
Adult and youth profiles use different non-monetary component weights. Monetary
proxies use adult work/study and other time values, while youth uses explicitly
non-wage educational and welfare-resource proxies; non-monetary burden remains
the primary youth outcome.

`S` combines attributed sleep displacement with a bounded share of accumulated
sleep debt. `E` and `F` are attributed shortfall ratios. `W` is the positive
decline from pre-run to post-run wellbeing. These are synthetic operational
proxies, not validated clinical scales, diagnoses, or empirical causal claims.

## Popularity and competition

Latent popularity uses realised player assignments, revenue, quality,
competitive integrity, novelty, and momentum. It is stored in timestamped truth
snapshots. A public ranking event can publish only a snapshot old enough to meet
the configured delay. It adds game-level noise and promotion pressure, then
produces a stable complete rank with game-ID tie-breaking.

Companies receive only published boards, and only after publication. If a
company decision and ranking publication share a tick, the company acts first
and cannot use the later same-tick release. Firms can compete for users through
content, monetisation, and acquisition; they can also form reciprocal
collaborative or collusive proposals.

## Accounting

All operational money uses signed 64-bit integer simulation cents. This unit is
internal and is not a foreign-exchange or purchasing-power conversion of GBP,
KRW, JPY, or EUR source values.

The accounting layer:

- posts income, purchases, company costs, audit costs, fines, and subsidies to an
  append-only balanced ledger in atomic batches with binary-unique references;
- uses exact integer aggregation for game and firm revenue;
- checks cumulative arrays and material balances before overflow;
- records assessed fines separately from collected fines;
- accrues player interest from used credit with one explicit rounded cent
  conversion after a bounded rate calculation;
- builds immutable outcome snapshots from latent state.

Each ledger entry is restricted to built-in signed-`int64` ticks and positive
signed-`int64` cents. Aggregate reconciliation streams into unbounded Python
integers. The `memory` and file-backed `sqlite` backends have identical ordered
entry semantics and canonical logical hashes. One outer SQLite transaction
covers a competitive-market day. Persistent finalization seals contain both a
backend-independent logical digest and a raw database-file digest; they verify
ledger content against a trusted manifest, not run completion, provenance, or a
world checkpoint.

Outstanding assessed fines reduce reported operating margin even when they
cannot be collected immediately. Subsidies increase cash but are removed when
computing operating margin, allowing viability without public transfers to
remain visible.

## Daily order

### Competitive-market day

`simulation/day.py` is the sole owner of temporal ordering:

1. pop all due events;
2. renew income and resolve company decisions due before consumption;
3. run consumer choice, activity, purchase, rare-event, and legacy harm-diagnostic logic;
4. aggregate spend and revenue and accrue interest;
5. publish a ranking if due;
6. select and resolve audits and collect fines if due;
7. review mature subsidy applications if due;
8. decay novelty;
9. assert the per-row ledger balance invariant and build an outcome;
10. commit the outer ledger transaction;
11. record the outcome, retain the completed `WorldStep` under the configured
    full or final-only policy, and advance the calendar.

Recurring events are rescheduled by the day processor. Domain phases implement
what an event does, not when it next occurs. All configured intervals and the
30-day income renewal must be exact multiples of `tick_days`.

Any phase, ledger, commit, or recording failure poisons the mutable world and
prohibits retry. Uncommitted ledger rows roll back, but the rest of world state
is not restored byte-for-byte; this is a fail-closed execution rule, not a
checkpoint/restart contract.

### Welfare policy day

`simulation/policy_day.py` owns the separate 1,440-minute welfare allocation:

1. renew monthly available budget and cap accounting when due;
2. initialise daily sleep, work/study, social, and physical obligations;
3. evaluate the full eight-action set for every player at every time step;
4. update time allocations, exact charges, cap use, and revenue source;
5. update habit, reinforcement, and progression after each action;
6. compute daily shortfalls, enjoyment, sleep debt, and wellbeing;
7. reconcile 1,440 minutes and all purchase revenue sources.

`simulation/policy_orchestrator.py` clones `PlayerLifeTable` for one scenario,
runs these days, computes welfare harm, and then constructs producer and EPGC
accounts. It does not expose the researcher's component scores to player choice.

## Outcomes

The competitive-market `OutcomeSnapshot` contains:

- player-level cumulative spend, income, debt, and seven harm dimensions;
- firm cash, operating margin, and safe-revenue share;
- state subsidy outlay;
- the tick at which the snapshot was constructed.

A summary reports total spending, number of players with debt, mean and 99th
percentile composite harm, solvent firms, mean safe-revenue share, and total
subsidy outlay. Paired-world comparisons preserve all component-level
differences before constructing a regime summary.

The welfare `PolicyScenarioResult` is separate and contains player identifiers
and cohort descriptors, cumulative spending, the six-component
`WelfareHarmResult`, an explicit weighted composite, enjoyment, high-risk
operational flags, minutes by action, reconciled producer revenue composition,
cost and profit, and an optional EPGC result. It reports harmful spending,
unplanned spending, monetary harm proxies, adult/youth opportunity-cost proxies,
and attributed displaced minutes without overwriting their component scores.

## Randomness and paired worlds

`CounterRNG` is a counter-based deterministic random field. A draw is addressed
by `(seed, entity_id, tick, stream, draw_index)` rather than consumed from a
mutable cursor. Stable named streams prevent Python hash randomisation from
changing results.

This design makes consumer blocking and entity iteration order reproducible and
allows paired branches to share semantically identical exogenous shocks.
Interventions remain explicit. Pre-treatment player/game tables, firms, states,
and jurisdiction metadata are checked for equality before a market paired run.
Policy scenarios receive the same initial `PlayerTable` and `PlayerLifeTable`;
each branch deep-copies only dynamic life state and reuses stable semantic random
coordinates.

Because rankings, switching, company strategy, collaboration, collusion, and
regulation create spillovers, the natural result is a market-regime effect with
interference, not a naive individual no-interference effect.

## Computational complexity

Let `P` be players, `G` games, `F` firms, `S` jurisdictions, `B` consumer block
size, `D` content-stat dimensions, and `K` boost rates.

| Operation | Time | Temporary memory |
| --- | ---: | ---: |
| Household peer indexing and exact consumer game evaluation | `O(P log P + P·G·log K_peer)`, with `K_peer <= P` observed household/game pairs | `O(B·G + P)` |
| Welfare activity allocation | `O(P·A·T)` | `O(P·A)` |
| Welfare harm decomposition | `O(P)` | `O(P)` |
| Popularity and public ranking | `O(P + G log G)` | `O(P + G)` |
| Current regulator scans | `O(S·P·F + S·F·G)` | `O(P + F)` per jurisdiction |
| Audit target sorting | `O(S·F log F)` | `O(F)` per jurisdiction |
| Content candidate enumeration | `O(K·D·2^D)` per search | same order while materialised |
| Accounting and current-tick ledger append | `O(P + G + F + N_tick log E)` | `O(P + F + N_tick)` Python state |
| Full ledger verification/sealing | `O(E)` | bounded row buffers, plus `O(accounts)` for requested net reconciliation |

Here `A = 8` welfare actions and `T = 1440 / step_minutes` decisions per day.
Blocking changes memory consumption, not the declared market alternative set.
Content search is exact only within the declared finite grid and is exponential
in `D`; configuration restricts `D` to 2–12. For `R` simulated ticks and `E`
retained financial transfers, and `N_tick` new transfers in the current tick,
full step history adds `O(R·P)` retained state. Final-only retention removes that
term while preserving each returned step and the final estimand. File-backed
SQLite keeps historical ledger rows out of Python while retaining `O(E)` disk;
smaller `O(R)` summaries and market histories remain. Campaign-scale resource
measurement, disclosure controls, and empirical readiness are therefore still
future work.

## Status of assumptions

Exact computation does not make the behavioural model empirically exact.
Population profiles, behavioural coefficients, company utilities, observation
processes, enforcement parameters, subsidy rules, and harm mappings contain
illustrative or synthetic assumptions. See [Data sources](data_sources.md) for
runtime input lineage and [Limitations](limitations.md) for interpretation
boundaries.
