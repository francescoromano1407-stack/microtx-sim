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

The current repository is an executable research skeleton. Its architecture and
invariants can be tested, but its illustrative profiles cannot support
substantive policy estimates.

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
          v
simulation orchestrator
          |
          v
World (latent state) <-> day processor <-> phase coordinators
          |                                   |
          |                         accounting / company / market / state
          v                                   |
consumer, company, state and popularity logic |
          |                                   |
          +---------- typed observations -----+
          |
agents, game domain, metrics, RNG, events, ledger, evidence profiles
```

`World` owns data and intervention hooks. The day processor owns temporal order.
Phase modules translate between latent tables and bounded domain interfaces. The
run orchestrator owns validation, campaign guards, timing, and repeated steps.
This division prevents temporal workflow, policy behaviour, and state storage
from accumulating in one monolithic class.

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed TOML configuration and structural/campaign validation. |
| `rng.py` | Stateless counter-based deterministic random field and stable named streams. |
| `types.py` | Shared enums for motives, spending segments, mechanisms, harm, actions, events, and provenance. |
| `agents/players.py` | Columnar player and household state plus retrospective spending classification. |
| `agents/companies.py` | Company observation, private state, action intent, heterogeneous firm policy. |
| `agents/jurisdictions.py` | Regulation rules, risk signals, audit/subsidy intents, and private regulator state. |
| `consumers/population.py` | Jurisdiction-aware construction of heterogeneous players. |
| `consumers/logic.py` | Exact game choice, activity, abstract competition, purchases, rare card events, and harm transitions. |
| `companies/logic.py` | Company telemetry, bounded observations, simultaneous intent collection, content and strategy resolution. |
| `states/logic.py` | Observable risk construction, audit selection, imperfect evidence, and enforcement resolution. |
| `market/popularity.py` | Exact latent popularity snapshots and delayed, noisy public rankings. |
| `domain/games.py` | Game table and exact non-dominating content-candidate search. |
| `core/world.py` | Latent mutable state, construction, event queue, intervention state, and compatibility entry points. |
| `core/events.py` | Deterministic priority queue with stable order, immutable payloads, cancellation, and rescheduling. |
| `core/observations.py` | Generic immutable signals, observations, and signal-only belief updates. |
| `core/ledger.py` | Append-only balanced integer-cent transfer log. |
| `simulation/accounting.py` | Income, exact revenue aggregation, interest, overflow checks, and outcome construction. |
| `simulation/company_phase.py` | Kernel coordination of one periodic company decision. |
| `simulation/market_phase.py` | Kernel coordination of latent popularity and public publication. |
| `simulation/government_phase.py` | Kernel coordination of complaints, compliance truth, audits, fines, and subsidy review. |
| `simulation/day.py` | One complete simulated day/tick and all event rescheduling. |
| `simulation/orchestrator.py` | Validated multi-cycle execution and run summaries. |
| `metrics/outcomes.py` | Seven-dimensional harm, firm viability, state outlay, summaries, and recording. |
| `causal/interventions.py` | Persistent mechanism caps and composable audit/subsidy regimes. |
| `causal/paired_worlds.py` | Structurally paired worlds, common random numbers, outcome differences, and regime effects. |
| `data/profiles.py` | Source registry loading, country/state profiles, unit contracts, and provenance gates. |

The legacy `systems/` namespace contains compatibility imports. New code should
use the domain packages above so ownership remains visible in import paths.

## State ownership and observation boundaries

| Actor or layer | Directly owns/knows | Receives imperfectly | Must not receive |
| --- | --- | --- | --- |
| Consumer/household | Age, motives, traits, available liquidity/credit, consent and supervision state, current game | Public rank/score, discovery, noisy personal quality | True popularity, other players' vulnerability, firm or regulator private state |
| Company | Cash, own portfolio, investments, costs, collusive trust | Own telemetry estimates, released rankings, demand estimates, expected audits/fines/subsidies, purchased research | Player-level latent vulnerability/harm, true popularity, actual audit selection, competitors' private state |
| State/regulator | Treasury, budgets, capacity, rules, policy priorities, accumulated audit beliefs | Complaints, reported minor harm, reported spending anomalies, public detections, audit evidence, verified subsidy dossier | Researcher's latent harm mean, latent unsafe-revenue share, undetected compliance truth before audit |
| Public market | Published score and rank | Delayed source data plus noise and promotion pressure | The current latent ranking snapshot |
| Research kernel | Complete state needed to resolve mechanisms and measure outcomes | Not applicable | Must not pass this state wholesale to a policy |

The kernel necessarily computes hidden compliance truth to resolve an audit. The
state first selects firms from public signals; only then does the resolution
system compare those selected firms with truth using finite sensitivity and
specificity. Evasion reduces detection, not the underlying breach.

## Agent representations

### Consumers and households

Players use a structure-of-arrays `PlayerTable` for population-scale operations.
Columns cover:

- age, jurisdiction, household, and jurisdiction-specific minor status;
- disposable income, liquid resources, allowance, credit, and household funds;
- stored-payment access, guardian supervision, and guardian consent;
- impulsivity, reward sensitivity, social susceptibility, loss aversion,
  financial literacy, and self-control;
- competition, collection, social, exploration, and relaxation motives;
- immutable pre-treatment vulnerability and seven dynamic harm dimensions;
- current game and awareness.

Traits are sampled with correlations and motives overlap. Age and income affect
resources and behaviour continuously. The unauthorised-card event is possible
only for an exposed minor lacking consent, with stored-payment access and low
supervision; the event then remains stochastic and resource-capped.

### Games and companies

A game stores quality, competitive integrity, novelty, six monetisation
intensities, a multidimensional statistic frontier, price, activity, revenue,
latent popularity, and its public board. The six current mechanisms are power
sales, random rewards, artificial scarcity, social pressure, price obfuscation,
and payment-friction removal.

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
synthetic home jurisdiction in the current skeleton. Applications must predate
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

Revenue aggregation uses integer scatter-add. Interest is calculated from
outstanding principal, bounded to the exactly representable floating range,
rounded once to cents, and accumulated with overflow checks. Assessed fines and
collected fines remain separate. Firm operating margin subtracts subsidies and
unpaid assessed fines, while cash changes only for amounts actually paid.

Player harm is stored in seven dimensions:

1. financial stress;
2. displacement of essential spending;
3. debt;
4. unauthorised spending;
5. loss of control;
6. functioning impairment;
7. regret.

A weighted composite is a reporting view; the underlying dimensions are never
discarded. Firm outputs include cash, operating margin, and safe-revenue share.
State outputs include subsidy outlay.

## Complexity

Let `P` be players, `G` games, `F` firms, `S` jurisdictions, `D` content-stat
dimensions, `K` boost rates, and `B` the consumer block size.

| Operation | Time | Temporary memory | Notes |
| --- | ---: | ---: | --- |
| Exact consumer game choice | `O(P·G)` | `O(B·G + P)` | Every known alternative is evaluated. |
| Activity, purchase, harm, aggregation | `O(P·G + Σ n_g log n_g)` as currently grouped | `O(P + G)` | Repeated per-game masks are exact but leave room for exact grouped optimisation. |
| Popularity/public ranking | `O(P + G log G)` | `O(P + G)` | Publication uses a delayed truth snapshot. |
| Regulatory observation and truth | `O(S·P·F + S·F·G)` as currently scanned | `O(P + F)` per state | Complaints and breaches remain exact; grouping can reduce repeated masks. |
| Audit target selection | `O(S·F log F)` | `O(F)` per state | Risk and random-floor targets are sorted. |
| Content candidate search | `O(K·D·2^D)` per search | `O(K·D·2^D)` currently materialised | Exact within the finite grid; `D` is limited to 12. |
| Accounting | `O(P + G + F + entries)` | `O(P + F)` | All monetary aggregation preserves integer cents. |

The block size changes memory use, not the mathematical choice set. Persistent
population state is linear in `P`; game state is linear in `G·D`. The current
research skeleton retains `WorldStep` results in `step_history`, and its ledger
is append-only, so a long run can also grow approximately as `O(T·P + E)` with
`T` ticks and `E` financial transfers. Campaign output must stream or thin this
history and define a ledger-retention policy rather than relying on the
skeleton's in-memory retention.

## Structural safeguards

- Scheduled day intervals must be exact multiples of `tick_days`.
- Mechanism caps are re-applied after subsequent company actions.
- Player resources cannot become negative and minor-owned credit is prohibited.
- Baseline vulnerability is copied and write-protected before treatment.
- Money arrays are non-negative `int64` where appropriate and checked before
  cumulative addition.
- Public ranks are a complete permutation with stable game-ID tie-breaking.
- Event order is stable by tick, priority, insertion sequence, and event ID.
- A scientific campaign rejects every non-calibrated evidence dependency.
- A non-campaign execution is limited to 32 cycles and 5,000 players.
- Null paired worlds are expected to be exactly identical.

These safeguards make invalid assumptions visible early; they do not substitute
for empirical calibration or external validation.
