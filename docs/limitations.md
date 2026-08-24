# Limitations and interpretation boundaries

## What this release can and cannot establish

This release is an executable research skeleton. It demonstrates an information
boundary, heterogeneous agents, exact alternative evaluation, market feedback,
imperfect audits, conditional subsidies, and paired counterfactual worlds. It
does not estimate the real-world harm caused by mobile-game monetization and it
does not identify an empirically optimal regulation or funding policy.

Outputs from the current smoke scenario are software checks only. They must not
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

The simulator records financial stress, essential-spend displacement, debt,
unauthorized spend, loss of control, functioning impairment, and regret as
separate dynamic indicators. These are operational states, not validated
clinical scales. “Unsafe revenue” is a model classification used for research
accounting; neither firms nor regulators observe the researcher's latent label.

Any composite harm score depends on explicit researcher weights. Different
weights can change aggregate conclusions and must be reported alongside the
components. Solvency, cash, operating margin, update continuity, and safer-revenue
share are model viability outcomes, not audited company accounts.

### No campaign-level uncertainty design yet

The current runner executes an explicit treated/control pair and supports
composable mechanism caps, audit regimes, and subsidy regimes. It does not yet
automatically construct a full factorial design, run multiple independent seeds,
estimate sampling uncertainty, propagate parameter uncertainty, calculate power,
or correct for multiple comparisons. A null-versus-null pairing is a structural
test, not empirical validation.

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

The central research question includes economic viability under public funding,
but the present funding mechanism is intentionally abstract. Firms have a
synthetic home jurisdiction, assigned mechanically from their identifier. A firm
must apply before review, and the state scores observable proxies for quality,
design safety, accessibility, and jobs. These proxies and weights are not
validated administrative criteria.

State treasuries, inspection costs, subsidy budgets, payment capacity, and firm
expectations are synthetic simulation-cent quantities. The runtime does not yet
model:

- tax-credit accounting, tax-shelter investors, or milestone certification;
- exact eligible expenditure and aid-intensity rules;
- deadweight loss, tax incidence, crowd-out, additionality, or opportunity cost;
- application selection bias, fraud, appeals, clawbacks, or post-award audits;
- exchange rates, government budget constraints, or national fiscal scale;
- employment displacement and general-equilibrium effects.

An intervention can change a uniform budget and scoring weights, but this does
not implement the official VGEC, KOCCA, JLOX+, or Belgian tax-shelter programmes.

## Information-boundary limitations

Agent methods do not receive the latent `World`. Firms use local telemetry,
public rankings, beliefs, and purchased research; regulators use complaints,
public anomalies, and imperfect audits. This architectural separation prevents
accidental omniscience in the current code.

It does not prove that the information environment is realistic. Signal noise,
delay, precision, costs, complaint formation, and learning rules are illustrative.
The model has no calibrated social network, advertising-auction data, app-store
recommendation system, platform-level enforcement, press cycle, or household
communication process. “Information has a cost” is implemented structurally,
not estimated empirically.

## Market and gameplay abstraction

There is no playable game, character roster, combat system, or observed live-ops
economy. Competitive content is a multidimensional frontier with trade-offs;
monetization is a vector of abstract mechanism intensities. Content search is
exact over the configured candidate grid, but the grid and utility function are
model choices.

Player game discovery considers the titles represented in the model and retains
all known alternatives within each computation block. It does not reproduce a
global app store, organic search, influencers, device constraints, regional
availability, churn surveys, or genre-specific preferences. Firms, games, and
states are few relative to the real market, and firm home jurisdictions are not
empirical ownership locations.

Collaboration, collusion, acquisition, research, compliance, and evasion are
utility-driven actions, but their feasible sets and payoffs are stylized. The
existence of emergent behavior in this system is not evidence that real firms
use the same decision process.

## Technical and computational limitations

The player step evaluates all represented alternatives and has time complexity
`O(P·G)`. Chunking reduces temporary memory to approximately `O(B·G + P)` but
does not reduce the exact alternative set. Popularity and accounting are roughly
`O(P + G)`, while audit selection includes sorting firms. Content candidate
enumeration is exponential in the number of stat dimensions; the configuration
therefore caps that dimension at 12.

The current non-campaign runner deliberately rejects more than 32 cycles or
5,000 players. `smoke.toml` is within that guard. `base.toml` describes a larger
future-scale scenario but is also rejected because it disallows the bundle's
current synthetic dependencies. No full campaign has been run for this release.

The implementation is in-memory and has no checkpoint/restart, distributed
execution, campaign scheduler, result database, schema migration, or long-run
resource benchmark. Integer overflow checks exist at important accumulation
boundaries, but they are not a proof against every possible extreme custom
configuration.

Exact reproducibility is conditional on the same code, configuration, source
files, Python/NumPy behavior, and platform-level numerical behavior. The project
specifies minimum dependency versions rather than a complete immutable
environment lock. Runtime duration is not deterministic and is not part of the
causal result.

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
6. sensitivity analysis for rare events, harm weights, behavioral equations,
   regulation, and funding assumptions;
7. multi-seed or otherwise justified uncertainty estimation and a preregistered
   estimand/reporting plan;
8. performance, persistence, and recovery tests at intended campaign scale.

