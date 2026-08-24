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
public support. The present implementation is a structural research skeleton;
its equations and country profiles are not yet calibrated for real-world policy
estimates.

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
can be vectorised. Each player has:

- age, jurisdiction, household membership, and jurisdiction-specific minor status;
- monthly disposable income, liquid funds, allowance, credit, and household funds;
- stored-payment access, guardian consent, and supervision;
- impulsivity, reward sensitivity, social susceptibility, loss aversion,
  financial literacy, and self-control;
- overlapping competition, collection, social, exploration, and relaxation
  motives;
- immutable baseline vulnerability and seven dynamic harm dimensions;
- awareness, current game, activity, and spending history.

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

Each game stores:

- company ownership;
- quality, competitive integrity, and novelty;
- six mechanism intensities: power sales, random rewards, artificial scarcity,
  social pressure, price obfuscation, and payment-friction removal;
- a multidimensional competitive-stat frontier;
- an integer-cent price;
- active players, exact revenue, and latent popularity;
- a delayed/noisy public score and complete public rank.

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
evidence age. Applications must predate the review. The current skeleton assigns
each firm one synthetic home jurisdiction. Awards are constrained by both the
subsidy budget and treasury cash. The regulator scores observable application
proxies; it does not receive the researcher's latent unsafe-revenue share.

## Consumer dynamics

For each tick, each consumer evaluates every represented game currently known to
them plus an outside option. Computation is split into blocks to bound temporary
memory, but no game is sampled or pruned.

Choice utility combines public quality/rank, price burden, novelty, competitive
integrity, monetisation pressure, motives, literacy, switching cost, awareness,
personal experience, and a stable idiosyncratic shock. The selected game then
produces activity, abstract matches, noisy personal quality, competitive rank,
purchase consideration, package demand, resource allocation, and harm
transitions.

Purchases cannot exceed available permitted liquidity and credit. Financial
mutations are preflighted before commit. Revenue is aggregated by game and firm
with integer arithmetic. The seven harm dimensions remain separate:

1. financial stress;
2. displacement of essential spending;
3. debt;
4. unauthorised spending;
5. loss of control;
6. functioning impairment;
7. regret.

A composite harm index is only an explicitly weighted reporting view. The model
does not diagnose addiction or gaming disorder.

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
  append-only balanced ledger;
- uses exact integer aggregation for game and firm revenue;
- checks cumulative arrays and material balances before overflow;
- records assessed fines separately from collected fines;
- accrues player interest from used credit with one explicit rounded cent
  conversion after a bounded rate calculation;
- builds immutable outcome snapshots from latent state.

Outstanding assessed fines reduce reported operating margin even when they
cannot be collected immediately. Subsidies increase cash but are removed when
computing operating margin, allowing viability without public transfers to
remain visible.

## Daily order

`simulation/day.py` is the sole owner of temporal ordering:

1. pop all due events;
2. renew income and resolve company decisions due before consumption;
3. run consumer choice, activity, purchase, rare-event, and harm logic;
4. aggregate spend and revenue and accrue interest;
5. publish a ranking if due;
6. select and resolve audits and collect fines if due;
7. review mature subsidy applications if due;
8. decay novelty;
9. assert ledger balance;
10. build and record an outcome;
11. append a `WorldStep` and advance the calendar.

Recurring events are rescheduled by the day processor. Domain phases implement
what an event does, not when it next occurs. All configured intervals and the
30-day income renewal must be exact multiples of `tick_days`.

## Outcomes

An `OutcomeSnapshot` contains:

- player-level cumulative spend, income, debt, and seven harm dimensions;
- firm cash, operating margin, and safe-revenue share;
- state subsidy outlay;
- the tick at which the snapshot was constructed.

A summary reports total spending, number of players with debt, mean and 99th
percentile composite harm, solvent firms, mean safe-revenue share, and total
subsidy outlay. Paired-world comparisons preserve all component-level
differences before constructing a regime summary.

## Randomness and paired worlds

`CounterRNG` is a counter-based deterministic random field. A draw is addressed
by `(seed, entity_id, tick, stream, draw_index)` rather than consumed from a
mutable cursor. Stable named streams prevent Python hash randomisation from
changing results.

This design makes consumer blocking and entity iteration order reproducible and
allows paired branches to share semantically identical exogenous shocks.
Interventions remain explicit. Pre-treatment player/game tables, firms, states,
and jurisdiction metadata are checked for equality before a paired run.

Because rankings, switching, company strategy, collaboration, collusion, and
regulation create spillovers, the natural result is a market-regime effect with
interference, not a naive individual no-interference effect.

## Computational complexity

Let `P` be players, `G` games, `F` firms, `S` jurisdictions, `B` consumer block
size, `D` content-stat dimensions, and `K` boost rates.

| Operation | Time | Temporary memory |
| --- | ---: | ---: |
| Exact consumer game evaluation | `O(P·G)` | `O(B·G + P)` |
| Popularity and public ranking | `O(P + G log G)` | `O(P + G)` |
| Current regulator scans | `O(S·P·F + S·F·G)` | `O(P + F)` per jurisdiction |
| Audit target sorting | `O(S·F log F)` | `O(F)` per jurisdiction |
| Content candidate enumeration | `O(K·D·2^D)` per search | same order while materialised |
| Accounting | `O(P + G + F + ledger entries)` | `O(P + F)` |

Blocking changes memory consumption, not the declared consumer alternative set.
Content search is exact only within the declared finite grid and is exponential
in `D`; configuration restricts `D` to 2–12. Long runs also retain an append-only
ledger and in-memory step history, so campaign-scale persistence is future work.

## Status of assumptions

Exact computation does not make the behavioural model empirically exact.
Population profiles, behavioural coefficients, company utilities, observation
processes, enforcement parameters, subsidy rules, and harm mappings contain
illustrative or synthetic assumptions. See [Data sources](data_sources.md) for
runtime input lineage and [Limitations](limitations.md) for interpretation
boundaries.
