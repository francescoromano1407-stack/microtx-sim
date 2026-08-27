# Simulation lifecycle

## Calendar model

One cycle advances the world by `run.tick_days`. The current scenarios use a
one-day tick. Every recurring calendar interval—company decisions, rankings,
audits, subsidy reviews, public-signal delay, and the 30-day income renewal—must
be exactly divisible by `tick_days`. Misaligned configurations are rejected
instead of silently shifting an event.

Slow processes use `EventQueue`, a deterministic priority queue ordered by:

```text
(tick, priority, insertion sequence, event id)
```

Payloads are defensively frozen. Cancellation is lazy but cannot resurrect an
old event; rescheduling creates a new immutable event. At world construction the
queue contains:

| Event | First tick | Priority | Recurrence |
| --- | ---: | ---: | ---: |
| Company decision | 0 | 0 | `market.firm_decision_interval` |
| Public ranking | 0 | 20 | `market.ranking_interval` |
| Audit review | 0 | 30 | active audit interval |
| Subsidy review | 0 | 40 | active subsidy interval |
| Income renewal | 30 | -10 | 30 days |

Priority is part of model semantics. Events due on a tick are popped once, then
handled in two passes around the consumer phase. This allows company decisions
and income to affect current-day behaviour while rankings, audits, and subsidies
use that day's realised consumer result.

## One simulated day

`simulation/day.py` owns the complete order. Domain phase modules implement the
mechanism but do not choose when it occurs.

```text
pop due events
      |
      v
pre-consumer events: income renewal -> company decision
      |
      v
consumer choice -> activity/competition -> purchase -> harm transition
      |
      v
exact accounting: cumulative spend -> revenue -> interest
      |
      v
post-consumer events: ranking -> audits/fines -> subsidy review
      |
      v
novelty decay -> ledger check -> outcome -> COMMIT -> history -> calendar
```

### 1. Pop events due at the current tick

The processor reads all live events with `event.tick <= world.tick`. Validated
scenario alignment normally makes them equal to the current tick. Local result
slots start empty: no company resolution, no published ranking, no audits, and
zero subsidy payment unless the corresponding event occurs.

### 2. Process pre-consumer events

Income renewal runs first when due because its priority is `-10`. Adults receive
the configured disposable portion of monthly income; minors receive allowance.
Each positive inflow is posted from an external jurisdiction account to the
player's liquid account, and the event is rescheduled 30 days later.

For a company decision, `simulation/company_phase.py`:

1. captures own period telemetry using revenue accumulated since the previous
   company decision and current activity estimates;
2. builds fallible company observations from own telemetry, released rankings,
   market estimates, and private beliefs;
3. collects every firm's intent before resolving any intent;
4. resolves feasible costs, content, monetisation, research, compliance,
   acquisition, evasion, subsidy applications, and bilateral agreements;
5. resets period revenue, re-applies persistent intervention caps, updates
   promotion pressure, and stores new subsidy applications.

After the phase returns, the day processor reschedules the next company
decision.

The tick-zero company decision therefore has no eligible published-ranking
history: it receives the explicit unknown-rank fallback and no realised
simulation revenue. The bootstrap board is available to consumers but is not
silently inserted into company memory. Same-tick purchases occur only afterward.
Simultaneous intent collection prevents one firm's decision order from exposing
another firm's unresolved action.

### 3. Advance all consumers

The consumer system receives player and game tables, the counter RNG, and the
ledger, but not `World`. It performs one transactional calculation:

1. **Discovery and exact game choice.** Each player discovers games
   probabilistically and evaluates every game currently known to them plus an
   outside option. Utility uses public score/rank, price burden, novelty,
   monetisation, motives, literacy, awareness, inertia, and stable random taste.
2. **Activity and abstract competition.** Personal quality experience is noisy.
   Activity time and matches depend on motivation and functioning. Skill is a
   stable heterogeneous construction; performance varies by tick. Players are
   ranked within the chosen game with stable player-ID tie-breaking.
3. **Purchase planning.** Consideration and conversion depend on engagement,
   mechanism pressure, vulnerability, age, motives, personal quality experience,
   price burden, available resources, literacy, impulsivity, and self-control.
   A heavy-tailed package count is capped by a configured per-tick maximum.
4. **Resource allocation.** Ordinary purchases use permitted liquidity and
   credit. A minor without guardian consent is blocked from ordinary household
   card use. The rare unauthorised-card path requires all exposure conditions
   and cannot exceed household/card resources.
5. **Harm transition.** Seven dimensions evolve separately from realised spend,
   unsafe exposure, debt, time, and the rare event, with configured persistence.
6. **Preflight and commit.** Shapes, resources, and overflow are checked first.
   The complete purchase-ledger batch then passes indexed uniqueness checks and
   appends atomically before game choice, balances, harm, activity, and revenue
   mutate. The enclosing day transaction remains the durable boundary.

Dense `P × G` utility work is split into player blocks. The block changes peak
memory only: stable player/game random coordinates and evaluation of all known
alternatives make results block-size independent.

### 4. Aggregate exact financial consequences

`simulation/accounting.py` adds player spend, unsafe spend, unauthorised spend,
and period game revenue to cumulative arrays after explicit non-negative `int64`
overflow checks. Game revenue is scattered to the owning firms using integer
aggregation, and firm cash is credited by the exact amount.

Interest accrues on used credit for the tick. The computation is bounded to the
exact-cent range before rounding and accumulating. Assessed liabilities and
actual transfers remain separate.

### 5. Publish popularity when due

`simulation/market_phase.py` first records a latent `TruthRankingSnapshot` from
current assignments, revenue, quality, competitive integrity, novelty, and
momentum. `PopularitySystem.publish()` then selects the newest truth snapshot old
enough to satisfy the configured delay.

If no snapshot is old enough, the bootstrap public board remains unchanged and
no publication is returned. Otherwise independent game-level noise and current
promotion pressure are added, a stable rank permutation is produced, and only
that released board is recorded for later company decisions. Current latent
popularity never backfills a delayed signal.

The ranking event is then rescheduled. Because publication follows the company
decision on a shared tick, firms can use it only at a later decision.

### 6. Audit and enforce when due

`simulation/government_phase.py` constructs a separate view for each jurisdiction:

1. players or households generate complaint reports stochastically from regret
   and observed unauthorised spending;
2. only reports reveal spending burdens and minor-harm signals to the regulator;
3. each state's budget and capacity bound its risk-ranked and random-floor audit
   selections;
4. after selection, the kernel constructs compliance truth from jurisdiction
   rules, game mechanisms, the latest compliance effectiveness, and evasion;
5. the audit system resolves evidence using finite sensitivity and specificity;
   evasion reduces effective sensitivity but never removes the actual breach;
6. inspection costs are paid, fines are assessed, collectable cash is transferred,
   unpaid assessments remain liabilities, and detected breaches become public
   history;
7. control returns to the day processor, which schedules the next audit event
   using the active regime interval.

This ordering is the critical information barrier: the state selects targets
without seeing hidden truth, then learns only the imperfect evidence produced by
auditing those targets.

### 7. Review subsidies when due

Applications submitted during the current tick are not eligible immediately;
only applications with `submitted_tick < review_tick` mature. If a firm has
reapplied, its latest mature dossier is used once. The current prototype assigns
each firm one synthetic home jurisdiction, so the relevant state sees only
eligible applications.

The state scores verified quality, design safety, accessibility, estimated jobs,
and evidence recency using its own priorities. Awards are constrained by both
the state subsidy budget and treasury cash. Payments move from the state treasury
to firm cash through the ledger and update cumulative firm/state totals. The
day processor then reschedules the review event.

The verified design-safety score is an observable application proxy; it is not
the researcher's latent unsafe-revenue classification.

### 8. Close the day

The day processor begins one outer ledger transaction before popping events.
Phase-level batches use savepoints, so a duplicate reference or native storage
failure cannot prefix-commit only part of a batch. A step rejects any
caller-owned outer ledger transaction; otherwise history and the calendar could
advance before that caller commits. After event phases:

1. game novelty decays smoothly according to elapsed tick days;
2. the ledger schema invariant is checked without rescanning prior transfers;
3. an immutable `OutcomeSnapshot` is built at the current tick;
4. the outer ledger transaction commits;
5. aggregate summary history and, when configured, the latest individual
   snapshot are recorded;
6. a `WorldStep` containing phase results is retained in full history or
   replaces the previously retained step under `final_only`; the cumulative
   audit counter is updated in either mode;
7. `world.tick` advances by `tick_days`.

The outcome includes cumulative player spend, income, debt including accrued
interest, the seven harm columns, firm cash, operating margin, safe-revenue
share, and state subsidy outlay. Outstanding assessed fines reduce reported firm
margin even when insufficient cash prevented collection.

Only a successfully committed step changes retained history, `audit_count`, or
the calendar. If any phase, storage operation, commit, or later recording step
fails, the world is marked poisoned and every later `step()` or `run()` call is
rejected. The ledger transaction rolls back uncommitted rows, but NumPy arrays,
agent state, and the event queue are not rolled back byte-for-byte. Poisoning
therefore prevents unsafe retry; it is not whole-world transactionality and it
does not implement checkpoint/restart.

## Multi-cycle orchestration

`simulation/orchestrator.py` owns run-level policy:

1. validate structural configuration;
2. in campaign mode, require every profile dependency to be calibrated and an
   explicit non-temporary SQLite ledger; population comparability also fails
   closed because schema-v1 evidence has no signature, sampling/synthesis,
   runtime-projection, output-estimand, or balance-validation binding;
3. resolve the cycle count and reject non-positive values;
4. for non-campaign runs, enforce the safety ceiling of 32 cycles and 5,000
   players;
5. call the day processor for the requested cycles;
6. return elapsed time, final outcome, and its summary.

`core/engine.py` remains a compatibility import. `World.step()` and
`World.run()` are thin convenience delegates to the day processor and cycle
runner; experiment-level execution should use `SimulationOrchestrator` so that
validation and execution guards cannot be skipped. The CLI `smoke` command
creates a world from the synthetic smoke configuration and uses this
orchestrator for a deliberately short structural run.

## Paired-world lifecycle

`run_paired_worlds()` builds two independently mutable worlds from the same
configuration and profile bundle. Before treatment, it checks equality of every
causally relevant player and game column plus firms, states, and jurisdiction
metadata. It then applies explicit treated and control interventions and runs
both worlds for the same number of cycles.

Each branch owns or receives a physically distinct ledger store. Pre-treatment
balance compares ordered logical ledger entries and their canonical digest while
excluding connection and path identity; shared storage is a hard imbalance.
Owned temporary stores are closed and deleted when the paired runner finishes.
Injected persistent ledgers remain caller-owned so they can be sealed separately.

Common random numbers do not mean shared mutable generator state. Both branches
query the same counter field at the same semantic coordinates. A purchase or
company action that occurs in only one branch cannot consume a cursor and shift
the other's future shocks.

Final paired differences preserve:

- all seven player harm dimensions;
- player spend and debt;
- firm operating margin and cash;
- state subsidy outlay.

The reported `RegimeEffect` adds a selected composite harm view, total financial
effects, and the share of players whose composite harm differs. Market feedback
means this is an equilibrium regime contrast with interference.

## Scheduling examples

For the supplied three-cycle smoke configuration:

- day 0: company decision, consumer phase, attempted ranking publication, audits,
  and subsidy review;
- day 1: consumer phase; a day-0 truth snapshot is now old enough for a later
  scheduled publication, but no ranking event is due yet;
- day 2: company decision still using the unknown-rank fallback, consumer phase,
  the first ranking publication from delayed data, and audits;
- day 3 would be the next subsidy review, but the three-cycle smoke run stops
  after days 0–2.

For the base architecture scenario, company decisions and audits occur every 28
days, rankings every 7 days, subsidy review every 84 days, and public rankings
use data delayed by 7 days. That configuration is illustrative and deliberately
blocked from scientific campaign execution.

## Lifecycle invariants to test after a change

- Firm intents are collected before any same-tick intent is resolved.
- Current-day public ranking cannot affect an earlier same-day company decision.
- Period telemetry is reset only after the company decision consumes it.
- Mechanism caps survive later updates, collusion, and monetisation changes.
- A subsidy application cannot be submitted and paid in the same tick.
- State audit selection cannot access kernel compliance truth.
- Evasion changes detection probability, not the truth label.
- Consumer results do not change when population block size changes.
- Every committed cent has a balanced, unique ledger transfer.
- A failed tick poisons the world, rolls back its uncommitted ledger rows, and
  cannot be retried on the same mutable state.
- Paired branches never share physical ledger storage.
- Outcome ticks increase strictly and the calendar advances exactly once.
- A null treated/control pair remains exactly identical.
