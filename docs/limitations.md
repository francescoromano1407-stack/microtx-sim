# Limitations and interpretation boundaries

## What this release can and cannot establish

This release is a stable, tested **synthetic simulation prototype**. The market
layer demonstrates information boundaries, heterogeneous agents, exact
alternative evaluation, market feedback, imperfect audits, conditional
subsidies, and paired worlds. The policy layer adds time allocation, fourteen
explicit mechanics, six welfare components, seven named counterfactuals,
repeated seeds, EPGC financing, sensitivity analysis, and versioned outputs. It
still does not estimate real-world harm or identify an empirically optimal
regulation or funding policy.

Outputs from the smoke run and policy prototype are software and structural
checks only. They must not
be reported as:

- national prevalence, spending, welfare, or market-size estimates;
- causal effects in real players or firms;
- legal conclusions about a product or jurisdiction;
- clinical diagnoses or probabilities of gaming disorder or addiction;
- foreign-exchange, purchasing-power, or cross-country income comparisons;
- forecasts of subsidy cost, tax expenditure, enforcement yield, or industry
  employment.

The campaign gate enforces part of this boundary: the current profile bundle and
source records are not fully `CALIBRATED`, so campaign execution is rejected.

## Empirical limitations

### No fitted behavioural model

Purchase, play, switching, harm, credit, firm, audit, collusion, and subsidy
equations are structural hypotheses with illustrative coefficients. They have
not been fitted to individual-level longitudinal data, transaction logs, audit
records, or firm accounts. Correlations among player traits and among firm traits
are designed to create heterogeneous behavior, not estimated population
covariances.

The model therefore supports “what follows inside these equations under this
intervention,” not “what would happen in the observed mobile-game market.”

### Narrow and artificial population frame

The four jurisdictions receive equal synthetic population weights. The player
age support is 8–69 in the UK and Belgium and 10–69 in Korea and Japan. People
outside those ranges are absent, and the age weights are illustrative. A
Belgian `65+` deprivation statistic, for example, is not equivalent to the
model's much narrower 65–69 player band.

Income observations refer to households or equivalised household income, while
the simulation assigns resources to individual player records and synthetic
households. Household composition, within-household allocation, taxes, benefits,
regional price levels, wealth, and real credit-market institutions are not
calibrated.

The repository now has population-evidence bundle schema version 1 for exact
joint age × household-income-band × household-type × gaming × pre-treatment
payer-history cells with exact rational mass. This is an input-lineage
boundary, not empirical population evidence. The checked-in bundle contains a
complete modeled joint target, but is `ILLUSTRATIVE`, unsigned, and has
`campaign_ready=false`; its validation cells are not an independent holdout.

Even a populated schema-v1 bundle could establish only exact source-cell
extraction. It does not bind complete declared income/household domains,
income-band boundaries, or a disjoint validation-sample identity, so its
calibration-target and held-out-validation gates are hard-coded false.

A separate static population-design contract can fill those declaration-level
gaps: it binds complete domains, target counts, evidence-result identities, and
deterministic calibration/validation cluster assignments, and can create an
exact-rational Hamilton plan. The checked-in design is complete but remains
`ILLUSTRATIVE`. Its record and cluster hashes are caller-supplied declarations;
without signed immutable source-unit keys, they do not prove publisher
authenticity, prevent aliases or role-specific salting, or establish genuine
held-out data. Static design schema v1 is therefore always fail-closed for
campaign readiness.

An optional `[population]` configuration can select a content-addressed runtime
mapping and adapter. The adapter re-attests the static
`PopulationApportionmentPlan`, consumes its exact counts and rational weights
without reallocating them, and binds the declared conversion from source
household-income categories to runtime personal monthly disposable-income
intervals and modeled household sizes. No checked-in configuration or runtime
mapping selects this path.

Projected gamer and payer-history labels remain attested sidecar-only and do not
initialize behaviour or payment state. Consumers recompute the nested
attestation and reject stale or mutated assignment indices. Per-seed execution
lineage also binds exact weights and a pre-treatment population balance artifact
covering all joint cells and runtime jurisdiction/age/income/household
membership. This proves conformance to the declared mapping and plan, not that
the household-to-personal-income transport is representative, calibrated, or
scientifically valid.

The exact weighted-estimand primitive has a dedicated standalone two-file
writer/profile and does not independently authenticate its declared upstream
identities. An explicit prospective-plan selection now supplies a separate
resolver which validates planned inputs before treatment, executes the
pre-treatment predicate, and binds exact results afterward; only that opt-in
path invokes the writer automatically. Output-v3 retains the frozen
v2-compatible CSV tables and unweighted synthetic-player semantics, including
for projected runs. The checked-in evidence, design, and schema-v2 income mapping
remain illustrative; no calibrated target, genuine holdout, public comparability, or
campaign readiness has been established. The P0 comparable-populations
requirement remains open, and no full campaign has been run.

Plan schemas v1 and v2, the schema-v3 successor, and run-binding schema v2 are
unregistered and campaign-ineligible. The successor's uncertainty,
convergence, and execution-attestation contracts expose missing evidence; they
do not cure it.
Money-valued estimands require an additional explicit opt-in prospective money
execution; without it they fail before treatment. When selected, the execution
applies one jurisdiction-specific composite conversion directly to each retained
simulation-cent observation, before reference/comparison contrasts, identical
scenario population weights, and cross-jurisdiction aggregation. Exact values
remain unrounded until one signed nearest-minor-unit rounding, with half ties
away from zero, at the final serialized estimate. It never reconstructs or
rounds an intermediate nominal local-currency amount. The results are target-currency-equivalent model amounts,
not observed or calibrated money, and only the separate prospective profile
receives them. Legacy root output-v3 artifacts remain unchanged in simulation
cents.

A schema-v1 `PRIMARY` label identifies one estimand specification whose outputs
remain separate per-seed realizations. The checked-in schema-v2 plan defines
an equal-seed primary mean, sample SD, Monte Carlo standard error, and normal
95% Monte Carlo interval over complete exact paired realizations. This resolves
the software aggregation contract, not empirical uncertainty or convergence.
The schema-v3 successor retains that primary and adds a deterministic joint
parameter design and blockwise rules, but its illustrative parameter ranges,
unquantified population and rate uncertainty, and absent campaign realizations
remain blocking.
Declared calendar dates are not
connected to a simulation clock: preflight checks only that their inclusive
duration equals the executed horizon, treating a zero-day structural snapshot
as one declared day. The plan also does not bind source-code, interpreter,
dependency-lock, or build-environment identity. Exact conversion execution does
not authenticate its sources, calibrate model money, establish a representative
population or genuine holdout, or provide external preregistration. The
checked-in population files are algebraic test fixtures with no authentic
signature. These gaps remain explicit campaign blockers and monetary
comparability remains open.

### Internal money is not observed purchasing power

All four monthly income anchors are mapped to the same `180000` simulation-cent
reference. This creates a coherent internal price/income unit but removes any
empirical cross-country level difference. The mapping is neither FX nor PPP.
Within-country dispersion remains illustrative, and nominal currency anchors
must not be compared through the internal scale.

Integer accounting avoids binary floating-point drift in cash flows, but
same-currency conversion into simulation cents may round to the nearest integer.
This numerical property does not make the underlying monetary assumptions
empirically exact.

The code now has a typed, dated, exact-rational FX/PPP conversion contract, a
content-addressed exact-CSV extraction contract, a strict pooled-currency
campaign gate, and opt-in prospective execution plumbing. The checked-in
`ecb-eur-fx-2024-v1` basis supplies complete official rate coverage and is used
by the prospective path. These are validation and arithmetic boundaries, not a
model-money calibration result: the source bundle has signature `MISSING`, the
internal-to-local scale is illustrative, and no root diagnostic output is
cross-country comparable.

The general parser retains historical rounding-stage declarations. The current
prospective monetary contract is stricter: only `AFTER_AGGREGATION` is
admissible, using exact composites and one final rounding after contrasts,
identical scenario weights, jurisdiction aggregation, and seed aggregation.
Method-specific source scope, typed
date intervals, registered-file re-attestation, and exact execution still cannot
decide whether the declared estimand or population is scientifically appropriate.
Structural coherence or a reproducible test extraction therefore does not
promote the public comparability flag. Source authenticity, calibration,
output/design binding, representative population and genuine holdout validity,
external preregistration, and campaign readiness remain separate blockers.

The same limitation applies to population evidence: a bundle or artifact hash
can reproduce the bytes and extraction but cannot authenticate the publisher,
establish calibration, select a defensible target population, or validate
transport from evidence cells into simulated and reported outcomes.

### Official metrics are mostly dormant

National gaming reach, payer incidence, spending-body and tail statistics,
regret, overspending, parental monitoring, Korean consumption propensity, and
Belgian deprivation are retained as provenance contracts but do not currently
shape the initialized population or purchase equations. Official subsidy
instruments, rates, caps, and payment schedules likewise do not determine awards.

Consequently, citing these records does not validate current player demand,
harm, or public-finance outputs. Conditional denominators also prevent direct
reuse: payer-only spending cannot be treated as population spending, Korean
zero-spender inclusion remains unresolved, and the Japanese extreme tail is an
ever-per-title observation rather than a monthly hazard.

The UK adults 2024 evidence review also found no support for treating 2--5% as
the active in-app-purchase payer rate. Ofcom's 2020 values are overlapping
ever-purchase measures among game-playing adults, the DCMS 7% value concerns
paid loot boxes only, and Ukie supplies market totals rather than individual
payer counts. A payer fraction and aggregate ARPU cannot identify a generalized
Pareto threshold, scale, and shape. UK payer incidence and the spending tail
therefore remain unidentified and unquantified in prose; the immutable v1
target table has no corresponding typed row, so a successor must encode them as
`UNQUANTIFIED`. Any Pareto/GPD implementation is a sensitivity assumption, not
an empirical calibration.

### Rare events are sensitivity assumptions

The unauthorized-card process is driven by an illustrative daily hazard
conditional on exposure. Official cases show that such events can occur but do
not identify their prevalence. The current hazard, household cap, discovery,
complaint, and harm transitions must be varied in sensitivity analysis rather
than treated as measured risk.

## Causal limitations

### Paired worlds identify a model effect, not a real-world effect

Treated and control worlds share initialized agents and counter-based random
coordinates. This removes Monte Carlo differences caused by event ordering and
supports a clean contrast inside the simulator. It does not solve confounding,
measurement error, transportability, or model misspecification in the real
world.

The primary result is a market-regime contrast conditional on the model's
structural equations. It is not an observational regression adjustment and not
an experimentally identified effect in human participants.

The paired runner now fails before treatment if independently created synthetic
worlds differ anywhere in their reachable initial state, retains that exact
balance assessment, and carries player income as a pre-treatment/exogenous
negative-control difference that invalidates a run when nonzero. Ordered
player, firm, and jurisdiction identifiers guard positional subtraction. The
balance walk rejects mutable objects or overlapping NumPy buffers shared across
branches even at different graph paths, and it compares within-branch mutable
alias topology. Shared immutable inputs remain permitted; explicit profile
bundles receive branch-local copies of their mutable jurisdiction templates.
The policy batch likewise rejects drift in its immutable pre-treatment result
fields, exact scenario attribution, or shared cohort state. Null-branch and
planted-effect recovery tests are structural software falsification checks only.
They cannot establish empirical balance, prove that an observed negative control
is valid, or rule out unmeasured confounding and misspecification.

### Interference is intrinsic

Rankings, firm responses, switching, collaboration, collusion, and enforcement
feedback cause one agent's treatment environment to affect others. Standard
individual-level no-interference assumptions do not hold. The natural estimand
is therefore a market-equilibrium regime effect. Direct and spillover effects
would need additional cluster assignments, exposure mappings, and reporting.

### Vulnerability is controlled structurally but not calibrated

Baseline vulnerability is immutable and shared between paired branches, which
prevents treatment from rewriting a pre-treatment attribute. Its distribution
and its mapping to behavior are nevertheless illustrative. Holding a synthetic
confounder fixed improves internal pairing; it does not establish that all
real-world vulnerability was measured or correctly represented.

### Outcome constructs are proxies

The market layer records financial stress, essential-spend displacement, debt,
unauthorized spend, loss of control, functioning impairment, and regret. The
policy layer separately reports monetary harm, opportunity cost, sleep,
education/work, family/social, and wellbeing. Both vectors are operational
proxies, not validated clinical scales, and they must not be conflated.
“Unsafe revenue” is a researcher-side accounting label; firms and regulators do
not observe it as latent truth.

Any composite harm score depends on explicit researcher weights. Different
weights can change aggregate conclusions and must be reported alongside the
components. Solvency, cash, operating margin, update continuity, and safer-revenue
share are model viability outcomes, not audited company accounts.

### Monte Carlo uncertainty is not empirical uncertainty

The legacy policy runner executes all seven required scenarios over repeated
seeds, reports variance and normal 95% Monte Carlo intervals, and retains a
compact one-at-a-time sensitivity grid. The campaign analysis framework adds a
deterministic joint parameter design, explicit seed/parameter/population/rate
realization identities, finite-design variance decomposition, and blockwise
convergence checks. These mechanisms quantify only uncertainty sources that
have an admissible declared design; they do not turn illustrative bounds into
probability distributions, calculate statistical power, correct for multiple
comparisons, or create population confidence or posterior intervals.

The prospective analysis binding retains exact per-seed planned estimands and a
plan-defined cross-seed primary aggregate. The current campaign successor still
cannot produce a sufficient total-uncertainty result: parameter bounds are
illustrative, and admissible population and monetary-rate uncertainty designs
are absent. Unidentifiable components are reported as unavailable, never zero.
The market runner separately exposes the lower-level treated/control pair for
equilibrium interventions and is not composed with the policy estimand.

## Legal and regulatory limitations

National rules are compressed into generic booleans and thresholds. Real law
depends on product design, consideration and prize definitions, consumer age,
contract formation, platform behavior, territorial scope, enforcement practice,
and time. Sources retrieved for one rule cannot silently support every other
rule in the same jurisdiction.

Specific current limitations include:

- UK generic rules abstract price transparency, consent, and minor-exhortation
  principles; they are not a complete current-law opinion;
- Korean odds disclosure is anchored, while several other active rule switches
  remain illustrative;
- Japan's complete-gacha restriction is product-specific and is not represented
  by the generic paid-random-reward runtime switch;
- the Belgian paid-random-reward switch remains illustrative because the cited
  classification calls for product-specific legal assessment;
- EU virtual-currency principles are simplified into generic Belgian price and
  minor-exhortation controls;
- audit breach thresholds, compliance cutoffs, maximum fines, evasion effects,
  sensitivity, and specificity are scenario parameters rather than estimates of
  actual enforcement.

Legal sources can change after retrieval. The model is not legal advice and
should not be used to classify a real game without jurisdiction-specific review.

## Public-funding limitations

The market subsidy mechanism and the policy EPGC calculation are intentionally
distinct. Market firms have a synthetic home jurisdiction and apply before a
state scores stylized quality, safety, accessibility, and jobs proxies. The EPGC
module instead uses an explicit safe-profit equation, public payments for
access/licences/availability, four public-value bonuses, a budget cap, and
penalties/clawbacks. Neither mechanism reproduces an existing programme or
validated administrative criteria.

State treasuries, inspection costs, EPGC schedules, producer costs, payment
capacity, and firm expectations are synthetic simulation-cent quantities. The
runtime does not model:

- tax-credit accounting, tax-shelter investors, or milestone certification;
- exact eligible expenditure and aid-intensity rules;
- deadweight loss, tax incidence, crowd-out, additionality, or opportunity cost;
- application selection bias, fraud, appeals, or post-award audit processes
  beyond the EPGC's mechanical prohibited-mechanic clawback;
- exchange rates, government budget constraints, or national fiscal scale;
- employment displacement and general-equilibrium effects.

The EPGC payment API intentionally has no playtime, retention, conversion, or
spending performance input. That structural safeguard does not prove that a
real contract would be lawful, incentive-compatible, or fiscally sustainable.
Neither funding layer implements VGEC, KOCCA, JLOX+, or the Belgian tax shelter.

## Information-boundary limitations

Agent methods do not receive the latent `World`. Firms use local telemetry,
public rankings, beliefs, and purchased research; regulators use complaints,
public anomalies, and imperfect audits. This architectural separation prevents
accidental omniscience in the current code.

It does not prove that the information environment is realistic. Signal noise,
delay, precision, costs, complaint formation, and learning rules are illustrative.
The only household communication slice is a synthetic, lagged, leave-one-out
signal of co-players' current games. The model has no calibrated or general
social network, advertising-auction data, app-store recommendation system,
platform-level enforcement, or press cycle. “Information has a cost” and the
household-peer channel are implemented structurally, not estimated empirically.

## Market and gameplay abstraction

There is no playable game, character roster, combat system, or observed live-ops
economy. Competitive content is a multidimensional frontier with trade-offs;
monetization is a vector of abstract mechanism intensities. Content search is
exact over the configured candidate grid, but the grid and utility function are
model choices.

Player game discovery considers public signals and the bounded household-peer
channel, then retains all known alternatives within each computation block. It
does not reproduce a global app store, organic search, broader friendship or
status networks, influencers, device constraints, regional availability, churn
surveys, or genre-specific preferences. Firms, games, and states are few
relative to the real market, and firm home jurisdictions are not empirical
ownership locations.

Collaboration, collusion, acquisition, research, compliance, and evasion are
utility-driven actions, but their feasible sets and payoffs are stylized. The
existence of emergent behavior in this system is not evidence that real firms
use the same decision process.

## Technical and computational limitations

The player step evaluates all represented alternatives. Sparse household-peer
indexing and lookup make game choice
`O(P log P + P·G·log K_peer)`, where `K_peer <= P` is the number of observed
household/game pairs. Chunking reduces temporary memory to approximately
`O(B·G + P)` but does not reduce the exact alternative set. Popularity and
accounting are roughly `O(P + G)`, while audit selection includes sorting firms.
Content candidate enumeration is exponential in the number of stat dimensions;
the configuration therefore caps that dimension at 12.

The market non-campaign runner deliberately rejects more than 32 cycles or
5,000 players. `smoke.toml` is within that guard. The policy prototype has its
own exact `O(P*A*T)` action engine and a checked-in 1,000-player, 14-day,
three-seed configuration. This is a reproducibility batch, not a scientific
campaign. `base.toml` remains a rejected future-scale market configuration.

The strategic-market `World` implementation has no whole-world
checkpoint/restart, distributed execution, campaign scheduler, general result
database, ledger schema migration, or long-run resource benchmark. The separate
exploratory policy executor now checkpoints only logically complete seed and
sensitivity units; it does not make a partially advanced strategic world
resumable. Final-only step retention removes the dominant `O(T·P)` `WorldStep`
history term, and file-backed SQLite removes `O(E)` Python-object retention for
ledger entries. Both are selected by the blocked future-scale baseline.

These changes do not make execution campaign-scale ready. The database remains
append-only at `O(E)` disk, aggregate recorder summaries grow by tick,
popularity truth history grows with scheduled ranking observations, and callers
can retain returned steps and run results. Compatibility snapshots, account-net
reconciliation, full integrity checks, and sealing scan `O(E)` entries. SQLite
is a local single-world audit store, not a distributed result system or a
privacy-controlled microdata repository. Paired worlds need two separate files.

If a tick fails, its uncommitted ledger rows roll back and the world is poisoned,
but already-mutated NumPy, event, or agent state is not restored. Retry and
resume are prohibited. A seal verifies finalized, ordered accounting content
against its trusted manifest; it neither proves that every intended simulation
tick completed nor captures enough world state to restart a run. Its hashes
detect accidental or uncoordinated changes, but the seal is not digitally signed
and does not authenticate an artifact against an adversary who can rewrite both
the database and manifest. Unsealed databases are incomplete.
Integer overflow checks exist at important accumulation boundaries, but they are
not a proof against every possible extreme custom configuration.

Exact reproducibility is conditional on the same code, configuration, source
files, interpreter, and platform-level numerical behavior. The dependency graph,
build backend, and CI installer are locked, but the operating-system image,
interpreter binary, external source publications, and source bundles are not
signed immutable artifacts. Runtime duration is not deterministic and is not
part of the causal result.

## Source and reproducibility limitations

The source loader checks the local register and contracts; it does not download
or inspect publications. The register has a retrieval date but no immutable
snapshots, hashes, extraction scripts, table/cell identifiers, revision IDs,
licenses, or stored sampling uncertainty. “Latest” and current-guidance URLs can
change. `ANCHORED` therefore means traceable starting point, not reproducible
calibration.

Before a scientific campaign, the project needs at minimum:

1. immutable source artifacts and scripted, tested extractions;
2. calibrated player, household, firm, game, information, regulator, and funding
   parameters with explicit conditions and denominators;
3. reviewed dated rates for the implemented conversion contract and a justified
   common money target, or country-specific market separation;
4. legal review with dated product and territorial scope;
5. external validation against outcomes not used for fitting;
6. empirically calibrated joint parameter and structural-uncertainty designs
   for the implemented propagation framework;
7. a preregistered empirical estimand, power analysis, and reporting plan;
8. performance, persistence, executed convergence evidence, and recovery tests
   at intended campaign scale.
9. a content-addressed model implementation and execution-environment identity,
   including reviewed source-tree, interpreter, dependency-lock, and build
   contracts bound to the prospective plan and result.
