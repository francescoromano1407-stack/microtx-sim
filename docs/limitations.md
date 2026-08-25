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

The policy runner executes all seven required scenarios over repeated seeds and
reports variance and normal 95% Monte Carlo intervals. It also runs a compact
one-at-a-time sensitivity grid with monotonicity and instability flags. These
tools quantify behaviour of the configured synthetic model only. They do not
propagate joint parameter or structural uncertainty, calculate statistical
power, correct for multiple comparisons, or create population confidence or
posterior intervals. The market runner still exposes the lower-level composable
treated/control pair for equilibrium interventions.

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

The implementation remains in-memory and has no checkpoint/restart, distributed
execution, campaign scheduler, result database, schema migration, or long-run
resource benchmark. Final-only step retention removes the dominant `O(T·P)`
`WorldStep` history term and is selected by the blocked future-scale baseline.
It does not make execution campaign-scale ready: the exact-cent ledger is still
append-only, aggregate recorder summaries grow by tick, popularity truth history
grows with scheduled ranking observations, and callers can retain returned
steps and run results. Integer overflow checks exist at important accumulation
boundaries, but they are not a proof against every possible extreme custom
configuration.

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
3. a justified common money target or country-specific market separation;
4. legal review with dated product and territorial scope;
5. external validation against outcomes not used for fitting;
6. joint parameter and structural sensitivity beyond the implemented OAT grid;
7. a preregistered empirical estimand, power analysis, and reporting plan;
8. performance, persistence, convergence, and recovery tests at intended
   campaign scale.
