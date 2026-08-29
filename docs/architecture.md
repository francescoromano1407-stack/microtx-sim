# Architecture

## Purpose and scope

`microtx-sim` is an agent-based causal simulation of a competitive mobile-game
market. Its central research question is:

> How much additional harm is causally attributable to mobile-game monetisation
> mechanisms after accounting for a player's pre-existing vulnerability, and
> which combination of regulation and public funding can sustain an economically
> viable game without dependence on compulsive spending?

The model represents heterogeneous consumers and households, abstract games,
strategic companies, jurisdictions, public rankings, audits, sanctions,
subsidies, and research outcomes. It does not implement a playable game,
characters, combat, or a complete power system. Competitive content is a vector
of abstract statistics so that strategic releases can be studied without
hard-coding any commercial title.

The current repository is a stable, executable synthetic research prototype.
Its architecture and invariants are tested, but its illustrative profiles
cannot support substantive policy estimates.

## Design principles

### No omniscient agent

`World` is the research kernel's latent state, not an observation available to
an agent. Company and regulator policies receive narrow immutable views. Players
choose from public information, personal resources, and noisy experience. True
popularity, compliance breaches, baseline vulnerability, researcher-defined
unsafe revenue, and the regulator's private state are not universal knowledge.

The intended information path is:

```text
latent state -> sensor/report -> delayed, noisy or purchased observation
             -> private belief -> decision intent -> kernel resolution
```

Information therefore has a source, an observation time, an availability time,
precision, and sometimes a monetary cost. `core/observations.py` provides the
generic `Signal`, `ObservationView`, `Belief`, and `BeliefBook` contracts.
Domain-specific immutable views are used where a typed interface is clearer.

### Heterogeneity instead of fixed scripts

Consumer labels such as casual, competitive, collector, and whale are not
mutually exclusive intrinsic types. Each player has continuous, correlated
traits and overlapping motive weights. A whale is an ex-post description of
observed spending that is both in the payer distribution's upper tail and large
relative to that player's disposable income.

Companies differ in risk aversion, compliance culture, ethical weight, analytics
capability, discounting, exploration, costs, cash, and private beliefs. States
differ in rules, budgets, audit capacity, priorities, accuracy, and subsidy
weights. Actions emerge from these differences and fallible observations rather
than a scenario script.

The aligned `PlayerLifeTable` adds continuous time commitments, enjoyment,
financial sensitivity, delay discounting, social/FOMO susceptibility, intended
limits, habit, reinforcement, progression, sleep debt, and wellbeing. Baseline
columns are immutable; only branch-local dynamic columns change after treatment.

### Welfare components are not market diagnostics

The competitive-market engine retains seven operational harm diagnostics for
market feedback and backwards compatibility. The welfare policy runner computes
the research estimand separately as `M/OC/S/E/F/W`: monetary harm, opportunity
cost, sleep, education/work, family/social, and wellbeing. Keeping these types in
different modules prevents a legacy diagnostic column from silently being
relabelled as a welfare construct.

### Exact finite choices and exact cents

Every known game is evaluated for every consumer. Population blocks bound peak
memory but do not sample or prune alternatives. Competitive content search
enumerates the complete declared finite candidate set. Financial state uses
integer simulation cents, checked accumulation, and an append-only balanced
transfer ledger.

“Exact” describes computation within the declared model. Behavioural equations,
finite action menus, signal processes, and calibration targets remain scientific
assumptions and approximations to the real world.

### Explicit counterfactuals

Paired worlds are constructed with identical pre-treatment populations, games,
firms, states, and counter-based random coordinates. An intervention is applied
explicitly to one or both branches. Branch-specific actions do not shift future
random draws in the other branch. Because rankings and company reactions create
interference, the primary result is a market-regime effect rather than a naive
individual treatment effect.

## Layered design

The dependency direction is from experiment control toward the kernel, then
toward narrow domain systems. Agent policies do not depend back on the world.

```text
CLI / causal experiment
          |
          +--> market orchestrator
          |          |
          |          v
          |    World <-> market day <-> company / market / state phases
          |          |                         |
          |          +----- typed observations-+
          |
          +--> policy batch / scenario orchestrator
                     |
                     v
               policy day -> full action choice -> life transitions
                     |                                  |
                     +----------> welfare harm <--------+
                                        |
                                  producer / EPGC result

shared foundations: players, RNG, monetisation domain, metrics, evidence profiles

optional population path:
static evidence/design -> exact runtime mapping -> projection adapter
                       -> per-seed assignment/balance -> execution lineage
```

`World` owns strategic-market data and intervention hooks. Its day processor owns
event order, and phase modules translate between latent tables and bounded domain
interfaces. The policy runner instead owns a cloned `PlayerLifeTable` and exact
daily time accounts; it has no reference to company or regulator private state.
Both orchestrators own validation, timing, and repeated steps. This division
prevents temporal workflow, agent policy, welfare measurement, and state storage
from accumulating in one monolithic class.

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed TOML configuration and structural/campaign validation. |
| `rng.py` | Stateless counter-based deterministic random field and stable named streams. |
| `types.py` | Shared enums for motives, spending segments, mechanisms, harm, actions, events, and provenance. |
| `agents/players.py` | Columnar player/household state, retrospective spending classification, and optional attested runtime-projection sidecar. |
| `agents/companies.py` | Company observation, private state, action intent, heterogeneous firm policy. |
| `agents/jurisdictions.py` | Regulation rules, risk signals, audit/subsidy intents, and private regulator state. |
| `consumers/population.py` | Legacy marginal player construction plus exact-count projected-table initialization used by the adapter. |
| `data/population_evidence.py` | Exact-byte schema-v1 joint-population evidence and fail-closed readiness assessment; no runtime cohort projection. |
| `data/population_design.py` | Static exact domains, declared source partitions, calibration target construction, and Hamilton counts/weights; always fail-closed for campaign use. |
| `data/population_projection.py` | File-backed source-to-runtime mapping, exact static-plan adapter, content-addressed runtime projection, and assignment execution. |
| `data/population_execution.py` | Opt-in configuration resolution and immutable per-seed population execution lineage. |
| `consumers/logic.py` | Exact game choice, activity, abstract competition, purchases, rare card events, and harm transitions. |
| `consumers/welfare.py` | Aligned immutable-baseline/dynamic `PlayerLifeTable` and deterministic synthetic priors. |
| `consumers/decision.py` | Full eight-action Gumbel/logit choice with hard budget, consent, cap, and cooling constraints. |
| `companies/logic.py` | Company telemetry, bounded observations, simultaneous intent collection, content and strategy resolution. |
| `states/logic.py` | Observable risk construction, audit selection, imperfect evidence, and enforcement resolution. |
| `market/popularity.py` | Exact latent popularity snapshots and delayed, noisy public rankings. |
| `domain/games.py` | Game table and exact non-dominating content-candidate search. |
| `domain/monetisation.py` | Explicit fourteen-coordinate policy vector, derived risk views, and transaction safeguards. |
| `core/world.py` | Latent mutable state, construction, event queue, intervention state, and compatibility entry points. |
| `core/events.py` | Deterministic priority queue with stable order, immutable payloads, cancellation, and rescheduling. |
| `core/observations.py` | Generic immutable signals, observations, and signal-only belief updates. |
| `core/ledger.py` | Exact append-only SQLite ledger facade, batching, streaming, and finalization seals. |
| `simulation/accounting.py` | Income, exact revenue aggregation, interest, overflow checks, and outcome construction. |
| `simulation/company_phase.py` | Kernel coordination of one periodic company decision. |
| `simulation/market_phase.py` | Kernel coordination of latent popularity and public publication. |
| `simulation/government_phase.py` | Kernel coordination of complaints, compliance truth, audits, fines, and subsidy review. |
| `simulation/day.py` | One complete simulated day/tick and all event rescheduling. |
| `simulation/orchestrator.py` | Validated multi-cycle execution and run summaries. |
| `simulation/policy_day.py` | Exact 1,440-minute action allocation and welfare-state transitions. |
| `simulation/policy_orchestrator.py` | One cloned policy branch, welfare calculation, revenue composition, and producer/EPGC result. |
| `metrics/outcomes.py` | Legacy market diagnostics, firm viability, state outlay, summaries, and recording. |
| `metrics/harm.py` | Pure six-component welfare harm, adult/youth opportunity valuation, and reporting weights. |
| `metrics/population_balance.py` | Fail-closed pre-treatment joint-cell count/mass comparison and separate runtime membership attestation. |
| `metrics/population_estimands.py` | Exact weighted mean, paired-difference, and weighted-quantile specification/result contract. |
| `causal/interventions.py` | Persistent mechanism caps and composable audit/subsidy regimes. |
| `causal/paired_worlds.py` | Structurally paired worlds, common random numbers, outcome differences, and regime effects. |
| `causal/scenarios.py` | Seven named monetisation/public-value policy regimes. |
| `causal/batch.py` | Same-cohort, repeated-seed scenario comparison and uncertainty summaries. |
| `data/profiles.py` | Source registry loading, country/state profiles, unit contracts, and provenance gates. |
| `outputs/population.py` | Standalone two-file `target_population_estimands` writer, separate from output-v3. |

The legacy `systems/` namespace contains compatibility imports. New code should
use the domain packages above so ownership remains visible in import paths.

## State ownership and observation boundaries

| Actor or layer | Directly owns/knows | Receives imperfectly | Must not receive |
| --- | --- | --- | --- |
| Consumer/household | Own age, motives, traits, resources, consent/supervision, current game, time obligations, commitments, habit, and wellbeing | Public rank/score, disclosed mechanics and price, discovery, lagged game use by simulated co-players in the same household, noisy personal quality | True popularity, other players' vulnerability/welfare, firm or regulator private state |
| Company | Cash, own portfolio, investments, costs, collusive trust | Own telemetry estimates, released rankings, demand estimates, expected audits/fines/subsidies, purchased research | Player-level latent vulnerability/harm, true popularity, actual audit selection, competitors' private state |
| State/regulator | Treasury, budgets, capacity, rules, policy priorities, accumulated audit beliefs | Complaints, reported minor harm, reported spending anomalies, public detections, audit evidence, verified subsidy dossier | Researcher's latent harm mean, latent unsafe-revenue share, undetected compliance truth before audit |
| Public market | Published score and rank | Delayed source data plus noise and promotion pressure | The current latent ranking snapshot |
| Research kernel | Complete state needed to resolve mechanisms and measure outcomes, including latent welfare components | Not applicable | Must not pass this state wholesale to an agent policy |

The kernel necessarily computes hidden compliance truth to resolve an audit. The
state first selects firms from public signals; only then does the resolution
system compare those selected firms with truth using finite sensitivity and
specificity. Evasion reduces detection, not the underlying breach.

## Agent representations

### Consumers and households

Players use a structure-of-arrays `PlayerTable` for population-scale demographic,
financial, and market operations. Columns cover:

- age, jurisdiction, household, and jurisdiction-specific minor status;
- disposable income, liquid resources, allowance, credit, and household funds;
- stored-payment access, guardian supervision, and guardian consent;
- impulsivity, reward sensitivity, social susceptibility, loss aversion,
  financial literacy, and self-control;
- competition, collection, social, exploration, and relaxation motives;
- immutable pre-treatment vulnerability and legacy market diagnostics;
- current game and awareness.

An aligned structure-of-arrays `PlayerLifeTable` adds the welfare state needed by
policy scenarios. Its write-protected baseline contains planned leisure,
intended play, sleep need, work/study, social and physical obligations, baseline
enjoyment, financial sensitivity, delay discounting, social/FOMO
susceptibilities, vulnerability, and an intended spending limit. Branch-local
dynamic arrays contain sleep debt, progression, habit, reinforcement, historical
spending, actual play, and wellbeing.

The initialiser uses counter-addressed, bounded illustrative priors: rounded
clipped normals for time variables; trait-, resource-, age-, and
vulnerability-conditioned affine priors for continuous states; and a bounded
fraction of disposable resources for the intended spending limit. These are
reproducible synthetic distributions, not empirical population estimates. Empty
cohorts are valid, and alignment is checked by exact player ID.

Population evidence is a separate provenance boundary. Its current schema can
retain exact joint cells and their hashes in profile-input lineage v4. A further
static population-design layer can bind exact evidence results to complete
domains, declared calibration/validation source partitions, target counts, and
deterministic exact-rational Hamilton counts and weights. Both checked-in
defaults are empty and `ILLUSTRATIVE`. Partition record and cluster hashes are
declarations only; they do not prove source authenticity or a genuine holdout,
and design schema v1 is never campaign-ready.

An optional `[population]` configuration selects one exact projection adapter.
Its strict file-backed mapping is the single contract that connects static
source household-income/type semantics to runtime personal monthly disposable-
income intervals and modeled household sizes. The adapter reopens and re-attests
the evidence-linked design, its `PopulationApportionmentPlan`, and the mapping;
it consumes the plan's existing per-cell counts and rational weights without
performing another allocation. Gamer and payer-history labels stay in the
sidecar rather than being mapped to game choice, payment access, or spending
history.

The projection execution binds the adapter, runtime projection, ordered player
IDs, and per-player cell assignment. Before treatment, the balance layer
compares target and realized mass/count discrepancies for every complete joint
cell and independently attests runtime jurisdiction, age/minor threshold,
income, and household membership. The policy batch and sensitivity analysis
retain a detached record for every seed and require the same population input;
the manifest includes this lineage only when the opt-in path was executed.
`World.create` supports the same configured adapter for the market path. With no
`[population]` section, the ordinary legacy marginal initializer and historical
cohort digest remain unchanged.

An exact-rational estimand layer implements weighted means, paired mean
differences, and deterministic weighted quantiles. The separate
`target_population_estimands` writer re-attests specification/result pairs and
writes one CSV plus metadata file, but it copies rather than independently
resolves the declared evidence, projection, balance, metric-contract, and weight
identities. It is not part of the automatic output-v3 bundle, whose frozen
v2-compatible CSV surface remains unweighted. These exact structural contracts
do not establish publisher authenticity, calibration, empirical holdout
performance, public comparability, or campaign readiness; no full campaign has
been run.

Traits are sampled with correlations and motives overlap. Age and income affect
resources and behaviour continuously. The unauthorised-card event is possible
only for an exposed minor lacking consent, with stored-payment access and low
supervision; the event then remains stochastic and resource-capped.

The strategic `World` also constructs a pre-tick, leave-one-out household-peer
game-use signal. Sparse household/game counts are indexed independently of raw
identifier values, and only block-local player-by-game shares are materialised.
A separate counter-RNG field governs peer discovery; social susceptibility and
the explicit synthetic `household_peer_influence` coefficient scale both that
channel and peer utility. This is a bounded household-network prototype, not a
calibrated social graph or household communication model.

### Welfare decision and transition boundary

`consumers/decision.py` is a stateless policy over the narrow inputs it receives:
the player's own tables, the disclosed `MonetisationVector`, current remaining
obligations and budget, and the counter RNG. It never receives `World`, company
beliefs, other players' welfare, or future outcomes. For the complete feasible
action set,

```text
a_i* = argmax_a [V_ia + tau epsilon_ia],  epsilon_ia ~ Gumbel(0, 1),
Pr(a_i=a) = exp(V_ia/tau) / sum_b exp(V_ib/tau).
```

The eight mutually exclusive actions are play, purchase, stop, sleep,
study/work, socialise, exercise, and other. Infeasible purchase utility is
`-infinity`, so affordability, consent, policy spending caps, and cooling-off
cannot be bypassed by random utility. The intended spending limit is retained as
the pre-treatment reference for unplanned-spending harm; access-plan adoption
also enforces it directly.

`simulation/policy_day.py` owns mutation and time accounting. With default
30-minute steps it makes 48 full-set choices and asserts
`sum_a minutes_ia = 1,440` for each completed player-day. Habit and reinforcement
evolve as

```text
h' = clip(rho h + eta I(play or purchase) - 0.60 eta I(stop), 0, 1)
delta = clip(observed_reward - (0.50 + 0.35 r), -1, 1)
r' = clip(0.98 r + eta_r delta I(play or purchase), -1, 1).
```

Progression changes separately through play, gates, and pay-to-progress. The
decision module computes intent; the policy day applies state transitions and
reconciles time and money. The harm module receives realised period aggregates
afterward and therefore cannot alter the choices whose welfare it measures.

### Games, monetisation, and companies

A strategic-market game stores quality, competitive integrity, novelty, a legacy
six-intensity monetisation matrix, a multidimensional statistic frontier, price,
activity, revenue, latent popularity, and its public board. That matrix remains
the interface for existing company adaptation.

Policy scenarios use the distinct immutable `MonetisationVector`:
`direct_price_cents`, `opaque_virtual_currency`, `paid_random_rewards`,
`progression_gates`, `time_limited_offers`, `daily_streak_pressure`,
`pay_to_progress`, `pay_to_win`, `social_guild_pressure`, `purchase_friction`,
`spending_cap_cents`, `cooling_off_hours`, `real_currency_price_display`, and
`personalized_offers`. Personalisation defaults to disabled. Normalised risk
coordinates, integer-cent caps, and hour-based cooling remain semantically
distinct; central properties derive transparency, pressure, and aggregate risk
without exposing player-level vulnerability to a company.

For a content release, `ContentPlanner` evaluates every proper non-empty subset
of statistic dimensions and every configured boost rate. At least one dimension
improves and at least one weakens; no candidate dominates the entire old
frontier. Selection maximises the company's perceived NPV, which incorporates
its own estimated demand, costs, audit risk, and reputation sensitivity.

Company intents include holding, releasing content, changing monetisation,
buying research, investing in compliance, acquiring users, seeking public
support, evading controls, collaborating, and colluding. Intent collection is
completed for all firms before effects resolve, preventing decision order from
revealing another firm's same-tick action. Bilateral agreements require exactly
reciprocal compatible proposals.

### Jurisdictions

State agents hold jurisdiction-specific rules and finite budgets. Audit capacity
is divided between risk-ranked targets and a random floor. Risk combines recent,
imperfect complaints, minor-harm reports, spending anomalies, and public
detections. Audit evidence can miss breaches and can produce false positives.

Subsidies are ranked using verified quality, design-safety and accessibility
proxies, employment, evidence age, and state priorities. A firm is assigned one
synthetic home jurisdiction in the current prototype. Applications must predate
the review and remain constrained by both the subsidy appropriation and treasury
cash.

## Randomness and reproducibility

`CounterRNG` maps `(seed, entity_id, tick, stream, draw_index)` to a deterministic
64-bit word using domain-separated SplitMix64 mixing. Named streams use a stable
FNV-1a identifier rather than Python's process-randomised `hash`. Uniform,
normal, and Bernoulli draws are vectorised views of this random field.

Consequences:

- changing player iteration order does not change an entity's draw;
- changing consumer block size does not change choices or outcomes;
- one subsystem consuming more draws cannot move another subsystem's cursor;
- paired branches share exogenous shocks at equal semantic coordinates;
- independent worlds or seed replications can be parallelised without splitting
  a mutable generator.

Stream and draw-index semantics are part of the model version. Renaming or
re-indexing them can change results even when equations are unchanged.

## Accounting and outcomes

The ledger records positive transfers between named source and destination
accounts with unique references. External income still names an external
counter-account, so the sum of all account changes is zero. Operational balances
remain in the world; the ledger is an independently recomputable audit trail.
Each entry requires built-in signed-64-bit tick and cent values plus non-empty
binary-collated text fields. Entry amounts are bounded to signed `int64`, while
streamed account nets and total flows accumulate in Python integers and can
therefore exceed `int64` without wrapping.

One public `Ledger` facade uses SQLite for both backends. `memory` selects an
in-memory database for small compatibility runs; `sqlite` selects a file-backed
database. Kernel consumers construct complete batches and rely on the database
unique index rather than retaining every historical reference in Python. One
outer transaction covers a simulated tick and nested batches use savepoints.
The database commits before recorder history or the calendar advances. A failed
tick permanently poisons its `World`: uncommitted ledger rows roll back, but
mutable NumPy, agent, and event state is not advertised as a restart checkpoint.

A caller can seal a non-temporary persistent ledger when it chooses to finalize
that accounting artifact. Sealing performs full integrity, schema, and
logical-history checks, closes the database, and writes a sidecar containing a
canonical logical SHA-256 and the final raw database SHA-256. The logical digest
identifies ordered entries across backends; the raw digest checks the file
against a trusted manifest. The unsigned pair does not authenticate run
provenance or prove that every intended simulation tick ran. An unsealed
database is an incomplete ledger artifact, not a resumable checkpoint.

Revenue aggregation uses integer scatter-add. Interest is calculated from
outstanding principal, bounded to the exactly representable floating range,
rounded once to cents, and accumulated with overflow checks. Assessed fines and
collected fines remain separate. Firm operating margin subtracts subsidies and
unpaid assessed fines, while cash changes only for amounts actually paid.

The market `OutcomeSnapshot` retains seven legacy operational diagnostics:
financial stress, essential-spend displacement, debt, unauthorised spending,
loss of control, functioning impairment, and regret. Firm outputs include cash,
operating margin, and safe-revenue share; state outputs include subsidy outlay.

The policy runner instead produces the welfare equation

```text
H_i = w_M M_i + w_OC OC_i + w_S S_i + w_E E_i + w_F F_i + w_W W_i.
```

`M` is caused only by the union of spending beyond the remaining commitment,
opacity/randomness/time pressure, and financial strain; total spending is not
itself a harm label. For `OC`, excess play is
`max(actual_play - planned_game_leisure, 0)`. Observed sleep, work/study, social,
and physical shortfalls are allocated proportionally and total attributed
displacement cannot exceed that excess. No excess play means zero gaming
opportunity cost even when an unrelated deficit exists.

`S` adds attributed sleep displacement to a bounded sleep-debt term; `E` and `F`
are attributed obligation-shortfall ratios; `W` is positive wellbeing decline.
Adult opportunity-cost proxies use adult time values. Youth profiles use lower,
explicitly non-wage educational and welfare-resource proxies, while their
non-monetary score is primary. Monetary proxies and component scores remain
separate and reconcile in `WelfareHarmResult`. A weighted composite is only a
reporting view; no component is discarded and none is a clinical diagnosis.

## Complexity

Let `P` be players, `G` games, `F` firms, `S` jurisdictions, `D` content-stat
dimensions, `K` boost rates, `B` the consumer block size, `A = 8` welfare
actions, and `Q = 1440 / step_minutes` welfare decisions per day.

| Operation | Time | Temporary memory | Notes |
| --- | ---: | ---: | --- |
| Household peer indexing and exact consumer game choice | `O(P log P + P·G·log K_peer)`, with `K_peer <= P` observed household/game pairs | `O(B·G + P)` | Sparse household/game pairs are indexed once per tick and queried by binary search; every known alternative is evaluated. |
| Activity, purchase, harm, aggregation | `O(P·G + Σ n_g log n_g)` as currently grouped | `O(P + G)` | Repeated per-game masks are exact but leave room for exact grouped optimisation. |
| Welfare daily action process | `O(P·A·Q)` | `O(P·A)` | Every action is evaluated at every time step. |
| Welfare harm decomposition | `O(P)` | `O(P)` | Six scores plus displaced-time and monetary-proxy columns. |
| Popularity/public ranking | `O(P + G log G)` | `O(P + G)` | Publication uses a delayed truth snapshot. |
| Regulatory observation and truth | `O(S·P·F + S·F·G)` as currently scanned | `O(P + F)` per state | Complaints and breaches remain exact; grouping can reduce repeated masks. |
| Audit target selection | `O(S·F log F)` | `O(F)` per state | Risk and random-floor targets are sorted. |
| Content candidate search | `O(K·D·2^D)` per search | `O(K·D·2^D)` currently materialised | Exact within the finite grid; `D` is limited to 12. |
| Accounting aggregation and current-tick ledger append | `O(P + G + F + N_tick log E)` | `O(P + F + N_tick)` Python state | `N_tick` new transfers use an indexed uniqueness check and one transaction; file storage remains `O(E)`. |
| Full ledger verification or sealing | `O(E)` | bounded streaming buffers plus account map when requested | Not performed on every tick. |

The block size changes memory use, not the mathematical choice set. Persistent
population state is linear in `P`; game state is linear in `G·D`. With
`step_history_retention = "full"`, the research prototype retains every
`WorldStep`, and long-run Python retention grows approximately as `O(T·P)` for
`T` ticks. `final_only` retains the latest successfully completed step and
therefore removes that term without changing the returned steps or final
estimand. With `ledger_backend = "sqlite"`, transfers use `O(E)` disk while
Python-retained ledger history stays bounded apart from the current batch and
explicit compatibility snapshots. Aggregate summaries and small game-level
histories still grow with `T`.

The retained NumPy payload in one full step is at least
`162P + 16G + 32F + 16S` bytes. At the future baseline of 50,000 players, eight
games, five firms, four jurisdictions, and 365 ticks, full history alone is
about 2.754 GiB per world or 5.507 GiB for paired worlds, before Python objects,
ledger storage, live state, and working arrays. The baseline now selects
`final_only` and file-backed SQLite, but campaign-scale execution still requires
a measured resource benchmark, storage-capacity plan, and all empirical gates.

## Structural safeguards

- Scheduled day intervals must be exact multiples of `tick_days`.
- Mechanism caps are re-applied after subsequent company actions.
- Player resources cannot become negative and minor-owned credit is prohibited.
- Baseline vulnerability is copied and write-protected before treatment.
- `PlayerLifeTable` baseline commitments are write-protected and dynamic state is
  deep-copied for every policy branch.
- Every completed welfare day conserves exactly 1,440 minutes per player.
- Purchase feasibility enforces available budget, consent, policy spending caps,
  and cooling-off before the stochastic argmax; access-plan adoption also checks
  the intended spending limit.
- Money arrays are non-negative `int64` where appropriate and checked before
  cumulative addition.
- Public ranks are a complete permutation with stable game-ID tie-breaking.
- Event order is stable by tick, priority, insertion sequence, and event ID.
- Ledger references are binary-unique, each tick commits as one ledger
  transaction, and a failed tick cannot be retried on the same world.
- Paired worlds use physically distinct ledger stores.
- A scientific campaign rejects every non-calibrated evidence dependency.
- A non-campaign execution is limited to 32 cycles and 5,000 players.
- Null paired worlds are expected to be exactly identical.

These safeguards make invalid assumptions visible early; they do not substitute
for empirical calibration or external validation.
