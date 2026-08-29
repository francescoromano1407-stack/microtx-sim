# Data sources, units, and input lineage

## Purpose and current evidence status

This document describes what the current code actually reads and uses. It is not
a claim that the model has been empirically calibrated. The machine-readable
source register is `data/provenance/sources.toml`; jurisdiction inputs are in
`configs/jurisdictions.toml`; run-level assumptions are in `configs/base.toml`
and `configs/smoke.toml`. The content-addressed rate-evidence registry is
`data/provenance/source_bundle.toml`; the separate population-evidence registry
is `data/provenance/population_bundle.toml`. The policy prototype reads
`configs/policy_prototype.toml` and reuses the jurisdiction profile loader.

At this stage:

- every record in the source register is `ANCHORED`, not `CALIBRATED`;
- the jurisdiction bundle is globally `ILLUSTRATIVE`;
- the common money scale is `ILLUSTRATIVE`;
- profile schema version 2 can retain declaration-only dated exact FX/PPP
  contracts, while version 3 requires verified rate-binding identifiers;
- the checked-in source bundle binds the exact source-catalogue digest but has
  zero artifacts, zero bindings, `ILLUSTRATIVE` status, and signature
  `MISSING`; no fallback rates are invented;
- population-evidence schema version 1 can verify exact-byte joint population
  cells, but the checked-in population bundle has zero artifacts and bindings,
  `ILLUSTRATIVE` status, signature `MISSING`, and `campaign_ready=false`;
- population allocation, audit capacity, regulator operating budgets, and many
  agent defaults are `SYNTHETIC` or otherwise uncalibrated;
- a scientific campaign is therefore rejected by construction.

The policy prototype's time allocations, fourteen mechanic intensities,
decision coefficients, harm thresholds and weights, opportunity-cost values,
producer costs, and EPGC payment schedule are all explicit `SYNTHETIC` or
`ILLUSTRATIVE` assumptions. They are not taken from the official records below.
The run manifest stores their exact values so a structural result can be
reproduced without implying that it was empirically calibrated.

The four provenance statuses have deliberately different meanings:

| Status | Meaning in this repository |
|---|---|
| `CALIBRATED` | Extracted, transformed, and validated for a declared estimand. Comparability still depends on compatible units, periods, populations, and conditioning. |
| `ANCHORED` | Connected to a named source, but not yet supported by a complete reproducible extraction and calibration pipeline. |
| `ILLUSTRATIVE` | A transparent scenario or modelling assumption. It may be useful for structural tests but is not an empirical estimate. |
| `SYNTHETIC` | Generated for an artificial scenario, operating scaffold, or structural check. It is never automatically admissible as scientific evidence. |

The loader validates status names, source identifiers, source-support scopes for
regulatory rules, conditions, denominators, periods, and declared currencies.
It does not independently verify that a publication is correct or that a value
was transcribed from the cited table without error.

## Three distinct classes of input

The presence of a value in `jurisdictions.toml` does not imply that it affects a
simulation outcome. The current implementation separates inputs as follows.

| Class | Meaning | Examples |
|---|---|---|
| Runtime-consumed | Read into an object or equation that affects the current world | age weights, income dispersion, generic regulation booleans, audit capacity, run-level purchase and audit parameters |
| Contract-only | Parsed, validated, and retained with provenance metadata, but not yet connected to an outcome-changing behavioural or policy equation | nominal national income anchors, national gaming reach and spending statistics, Belgian deprivation rates, official subsidy rates/caps/instruments |
| Context-only | Listed in the source register or cited in narrative documentation, but not referenced by a current profile contract | WHO definition, UK causal review, FTC case, Eurostat/OECD context sources |

This distinction is essential: a dormant contract is useful for future work, but
it cannot make the current simulated consumption distribution realistic by
itself.

For the policy engine, existing runtime-consumed age, income, household, trait,
and motive profiles initialize the shared cohort. `PlayerLifeTable` then adds
seeded synthetic leisure, sleep, work/study, social, physical-activity,
enjoyment, financial-sensitivity, discounting, FOMO, commitment, habit, and
wellbeing fields. No official source record currently calibrates those added
distributions.

## Population-evidence bundle

Population-evidence schema version 1 is a strict, content-addressed input
contract for future target-population work. Each binding names a target
population and jurisdiction, reference period, population base, universe, unit
of analysis, eligibility and exclusion rules, age scope, household-income and
household definitions, gaming and payer definitions, zero-spender treatment,
source IDs, retrieval date, and either a `CALIBRATION` or `VALIDATION` role.
The one whitelisted interpreter reads strict UTF-8 CSV with the exact declared
header and selects rows for one binding.

Every selected row is one joint
age × household-income-band × household-type × gaming × pre-treatment
payer-history cell. Gaming is explicitly `GAMER` or `NON_GAMER`; payer history
is explicitly `EVER_PAYER` or `NEVER_PAYER`. Cell mass is a non-negative
reduced integer numerator over a positive integer denominator. Cell identifiers
must be sorted and unique, joint semantic cells may not repeat, the masses must
sum exactly to `1/1`, every age in the binding's declared scope must be covered,
and age intervals cannot overlap within a fixed income-band, household-type,
gaming, and payer-history stratum. This is a joint distribution contract rather
than a collection of independent marginal anchors.

Clearing the profile structural subgate is stricter than parsing one extract:
every observed income-band × household-type × `GAMER`/`NON_GAMER` ×
`EVER_PAYER`/`NEVER_PAYER` stratum must cover the full declared age scope exactly
once. A true zero therefore needs an explicit zero-mass cell; an omitted state
cannot masquerade as a complete target. Each role must use harmonised joint-cell
support across jurisdictions.

This is not enough to certify a calibration target. Schema version 1 infers the
income and household categories from observed rows rather than binding complete
declared domains or income-band bounds. It also has no typed identity for a
sample partition or independent validation dataset. Consequently the typed
profile assessment keeps both `calibration_targets_bound` and
`heldout_validation_targets_bound` false for every schema-v1 bundle; the role
labels are retained as declarations for a future schema.

The bundle binds the exact source-register SHA-256, bundle bytes, regular CSV
artifact paths, byte lengths and SHA-256 digests, canonical extraction recipes,
and extracted-cell results. Registered profile lineage reopens and re-attests
those inputs when it is constructed and again when a manifest is emitted.
Hashes establish which bytes and transformation were used; they do not prove
publisher authenticity, scientific calibration, or that the selected
population is appropriate.

Profile integration accepts a population source only when its complete declared
geography equals the binding's country and its period is either the exact date
interval or matching full calendar year. Qualified or broader regional scope
needs a future typed contract rather than inference from free-form text. File,
ID, hash, or URL differences alone are deliberately not accepted as proof of a
held-out sample.

Schema version 1 deliberately cannot be campaign-ready. It supports only a
missing signature and leaves sampling/synthesis, runtime projection,
output-estimand binding, and balance-validation gates false. The checked-in
bundle is empty, `ILLUSTRATIVE`, unsigned, and has `campaign_ready=false`, so it
does not choose any target population or supply calibration or held-out
validation cells. This milestone is provenance infrastructure only.

### Static population design and exact apportionment

The separate population-design schema version 1 can bind exact verified
population-evidence results to complete categorical domains. The domain includes
explicit age intervals; jurisdiction-specific source household-income category
bounds, definitions, currency, and period; household-type definitions; and the
full gamer/non-gamer and ever/never-payer states. Runtime personal monthly
disposable-income intervals and modeled household sizes are separate projection
inputs, not static source-domain claims. Jurisdiction declarations bind
calibration and validation evidence identities to exact target-population counts.

The same static contract declares source-record and source-cluster identities
and assigns each cluster deterministically to one role. It requires the declared
partition to cover the bound evidence cells without crossing roles. Given a
complete calibration target, `exact_rational_hamilton/1` produces deterministic
integer cell counts plus exact rational analysis and expansion weights; the
weights reconstruct unit target mass and the target-population count.

These are structural and reproducibility properties, not authenticity or
holdout findings. Record and cluster hashes can be declared with aliases or
role-specific salting unless signed immutable source-unit keys are independently
verified. A partition declaration therefore does not prove publisher identity,
sample independence, or genuinely held-out validation. Static design schema v1
always reports `campaign_ready=false`.

The checked-in `population_design.toml` is empty and `ILLUSTRATIVE`: it has no
age, income, or household domain rows, jurisdictions, evidence-result hashes, or
partition records. It is an executable contract example, not a selected target
population or a completed sampling design.

## Monetary units and transformations

### Layer 1: nominal local-currency minor units

Values whose names end in `_minor_units` are stored as integer minor units of
their declared local currency. They remain jurisdiction-specific and must not be
added, ranked, or compared across countries.

| Jurisdiction | Currency | Reported income input | Monthly nominal anchor used by the loader | Status |
|---|---:|---:|---:|---|
| UK | GBP, pence | 3,666,300 pence annually (£36,663) | 305,525 pence | `ANCHORED` |
| KR | KRW, exponent zero | quintiles 1,037,000; 2,467,000; 3,515,000; 5,104,000; 8,912,000 monthly | 3,515,000 KRW, the central reported quintile anchor | `ANCHORED` |
| JP | JPY, exponent zero | 300,000 JPY monthly | 300,000 JPY | `ILLUSTRATIVE` |
| BE | EUR, cents | 3,129,900 cents annually (€31,299) | 260,825 cents | `ANCHORED` |

UK and Belgian annual values are divided by 12 using integer arithmetic and
rounded to the nearest declared minor unit. With the current values both results
are exact. The Korean central quintile is an anchor from a household table; it
is not asserted to be an individual-income median. The Japanese value has no
income source and is explicitly illustrative.

### Cross-country conversion and source-evidence contracts

Profile schema version 2 accepts optional `[[monetary_conversion]]` records as
immutable `MonetaryConversionContract` values. Each record names one
jurisdiction and its source and target ISO-style currency codes; selects `FX`
or `PPP`;
stores an exact positive rational rate in target minor units per source minor
unit; declares typed start/end dates for the rate and target-price periods,
comparison group, status, estimand, population base, source IDs, retrieval date,
rounding stage, and aggregation unit. Signed integer conversion rounds half ties
away from zero, either per observation or after the declared aggregate is
formed. There is no default method, target, rate, period, rounding stage, or
scientific status.

Profile schema version 3 keeps those terms and additionally requires stable
`conversion_id` and `rate_binding_id` values. The referenced binding lives in a
separate source-bundle schema. That bundle binds the exact SHA-256 of
`sources.toml`, each regular CSV artifact's normalized relative path, media
type, byte length, and digest, and a canonical JSON extraction recipe. The only
version-1 interpreter selects exactly one row by source, jurisdiction, quote
direction, method, rate dates, and retrieval date, then parses a positive
reduced integer numerator and denominator without a floating-point conversion.
The artifact and bundle are re-read when registered lineage is built and again
when a manifest is emitted. Content hashes establish reproducibility, not
publisher authenticity or scientific correctness. Repository attributes pin
the hashed TOML/JSON inputs to LF bytes and disable text conversion for rate
artifacts, so the same Git blobs retain the same digests across platforms.

For pooled campaign outputs, the gate requires exactly one contract per local
money profile. Every contract and referenced source must be `CALIBRATED`; source
records must declare a compatible FX/PPP support scope and the same retrieval
date, and at least one compatible source must match the rate period. All
jurisdictions must share one target currency, method, rate period, target price
period, estimand, population base, comparison group, rounding stage, and
aggregation unit. FX evidence must explicitly support `foreign_exchange_rate`;
PPP evidence must support `purchasing_power_parity`. A generic conversion label
cannot justify switching methods, and retrieval cannot predate the declared
rate-period end. Until a separate deflator contract exists,
the target-price and rate intervals must be identical. The gate also checks exact algebraic
coherence between each local-to-target rate and the corresponding
local-to-simulation scale. Matching labels alone cannot make outputs comparable.

These gates are deliberately not one public substantive-comparability claim.
The typed assessment reports separate source-extraction, source-signature,
output/design, population, and preregistration results. A valid test artifact
can clear only source extraction. Schema-v1 source bundles have signature
`MISSING`, and the prototype has no run-specific output conversion binding,
calibrated target-population specification, or external preregistration, so
public comparability remains false.

The checked-in `jurisdictions.toml` deliberately remains schema version 2 with
zero conversion records. The checked-in source bundle deliberately has zero
artifacts and bindings. Those absences, the catalogue digest, and the missing
signature are fingerprinted and summarized in exported manifests. The software
can reproduce a reviewed rate extraction later, but P0 monetary calibration,
scientific rate selection, output use, population binding, and preregistration
are not complete.

The complete profile bundle must also match the jurisdiction and source files
whose hashes it claims. Registered lineage reloads those files both when the
lineage object is constructed and when a manifest is emitted. Programmatic or
changed bundles remain unregistered and fail full campaign validation even when
their test-only statuses are labelled `CALIBRATED`.

### Layer 2: internal simulation cents

The player, game, firm, regulator, and ledger systems require one internally
consistent integer money unit. The loader therefore maps every jurisdiction's
monthly nominal anchor to exactly `180000` internal *simulation cents*. The
conversion ratio is represented as an exact rational number, and conversions of
other same-currency integer amounts round to the nearest simulation cent.

This operation is a normalization to a common reference-income index. It is:

- not a foreign-exchange conversion;
- not an official purchasing-power-parity estimate;
- not evidence that national disposable incomes are equal;
- not suitable for cross-country welfare or price-level comparisons.

The internal field names retain the `_cents` suffix because the ledger operates
on integers. They do not denote GBP pence, euro cents, won, or yen. National
currency amounts survive only in `MoneyScaleContract` provenance records.
The fixed `180000` destination, rather than the level of the national anchor,
enters `CountryProfile`; changing a nominal anchor currently changes its recorded
scale ratio but not the initialized reference-income median.

### Layer 3: run-level monetary assumptions

Game prices, firm cash, fines, research costs, audit costs, subsidy budgets,
allowances, liquidity, debt, and outcomes are all denominated in simulation
cents. Current regulator treasury, inspection cost, subsidy budget, preference
weights, and initial audit accuracy are synthetic scaffold values. Audit
accuracy and random-audit share are then overwritten by the selected scenario
configuration. Official subsidy rates, caps, payment shares, and tax instruments
are not used to convert the synthetic operating budget.

## Inputs currently consumed by the runtime

### Population and player initialization

The loader passes the following jurisdiction fields into `CountryProfile`:

- jurisdiction code;
- equal allocation weight of `0.25` per country;
- configured age-band edges and weights;
- the common internal monthly income anchor of `180000` simulation cents;
- `income_log_sigma`, or Korea's `income_within_quintile_log_sigma`;
- source identifiers for traceability.

The equal country weights are a synthetic experimental allocation, not national
population shares. The configured age range is 8–69 in the UK and Belgium and
10–69 in Korea and Japan. Age weights and income-shape parameters are
illustrative.

The configured runtime paths do not consume population-evidence or population-
design cells. They still sample jurisdiction and age from the legacy marginal
profiles, derive income and household state from synthetic defaults, and give
each generated player one implicit analysis unit.

An opt-in `initialize_projected_player_table` library helper can instead accept
already-resolved cells, apply deterministic Hamilton allocation, and attach a
content-addressed population assignment to `PlayerTable`. No checked-in config,
CLI, `World.create`, policy batch, or sensitivity path selects this helper.
It does not consume or revalidate a `PopulationApportionmentPlan`. Instead it
sorts its separate runtime cells by `cell_id`, derives a content-addressed
runtime projection, and performs its own exact-mass Hamilton allocation with
`cell_id` tie-breaking. A future adapter must bind the static plan counts and
the source-household-category to runtime-personal-income/household conversion;
matching labels do not provide that bridge.

Gamer and payer-history values in projected cells remain attested sidecar
metadata only: they do not initialize `current_game`, payment access, or
historical spending. When this sidecar is present, its assignment digest is
recomputed and included in the cohort digest; stale or mutated indices are
rejected. When absent, the legacy cohort digest is preserved.

An exact-rational estimand primitive can consume the sidecar's design weights
for a weighted mean, paired treated-minus-control mean difference, or lower
inverse-CDF weighted quantile. It binds the target, design, projection, balance,
metric contract, and a dedicated output-profile identity as digest declarations;
only the supplied design weights are re-attested by this primitive. It does not
resolve the other digests to verified artifacts. No batch reducer, world summary,
CSV writer, or registered output profile calls it yet. Existing output schema v3
artifacts therefore retain the frozen v2-compatible columns and unweighted
synthetic-player aggregation semantics.

The following `CountryProfile` inputs are inherited from scaffold defaults and
are not country estimates: adult age, the age-income curve, minor allowance,
personal and household liquidity, credit access and limits, stored-payment
access, household size, guardian consent and supervision, awareness, trait
means/scales/correlation, and motive logits. They affect simulated players, but
they require explicit provenance and calibration before scientific use.

### Behavioural and information configuration

The selected `base.toml` or `smoke.toml` supplies the effective values for game
choice temperature, switching cost, base purchase logit, unauthorized-card
hazard, essential-spend share, harm decay, credit interest, public-signal delay
and noise, and research cost/noise.

The unauthorized-card hazard and essential-spend share also appear as shared
profile contracts. `World.create` requires exact equality between each shared
contract and its run-level value, then uses the run-level value. This prevents
the documented prior from silently diverging from the effective equation.

The unauthorized-card value is a daily hazard conditional on an exposed minor,
not a prevalence estimate. Exposure itself requires the model's payment-access,
opportunity, and household conditions.

### Regulation

Five generic booleans are represented by `RegulationRules` and affect the
kernel's compliance truth: paid-random-reward restriction, odds disclosure,
real-money price display, parental authorization, and direct exhortation to
minors. They are operational abstractions, not complete statements of national
law.

Each rule has its own status and, when `ANCHORED`, its own compatible source.
Illustrative rules can be consumed without a source in structural runs. The
current anchored rule links are:

- UK: parental authorization/express consent, real-money price transparency,
  and direct exhortation to minors, linked to [`CMA_GAME_PRINCIPLES`](https://www.gov.uk/government/publications/buying-features-in-online-games-advice-for-parents-and-carers);
- Korea: odds disclosure, linked to [`MCST_ODDS_DISCLOSURE_2024`](https://www.mcst.go.kr/english/policy/pressView.jsp?pSeq=407);
- Japan: complete-gacha scope, linked to [`JP_COMPLETE_GACHA_FAQ`](https://www.caa.go.jp/policies/policy/representation/fair_labeling/faq/card), but this
  product subtype is contract-only because the generic runtime rule type does
  not encode it;
- Belgium: real-money price and minor-exhortation principles, linked to
  [`EU_CPC_VIRTUAL_CURRENCY_2025`](https://commission.europa.eu/document/download/8af13e88-6540-436c-b137-9853e7fe866a_en?filename=Key+principles+on+in-game+virtual+currencies.pdf).

The Belgian paid-random-reward switch is active in the runtime but remains
`ILLUSTRATIVE`; the Gaming Commission source is attached as a classification and
product-specific scope anchor, not as evidence for an unconditional universal
ban.

Audit interval, sensitivity, specificity, random-audit share, and maximum fine
come from the selected run configuration. Audit capacity comes from the shared
jurisdiction profile and is synthetic. Audit budgets and inspection cost are
also synthetic. The audit kernel uses finite sensitivity and specificity and
can be affected by firm evasion.

### Public funding

The runtime supports subsidy applications, domestic-jurisdiction assignment,
conditional scoring, a treasury, budget depletion, and double-entry cash flows.
Every firm is assigned a synthetic home jurisdiction from its identifier. The
operating budget and scoring weights are synthetic simulation-cent values and
can be changed by an explicit `SubsidyRegime` intervention.

Official programme labels, rates, caps, payment shares, eligibility language,
and tax-shelter structures are contract-only. The current award equation does
not reproduce the legal or accounting mechanics of VGEC, KOCCA production
support, JLOX+, or the Belgian gaming tax shelter.

## Contract-only national metrics

The following values are parsed and validated but do not currently initialize
players or enter purchasing, harm, audit, or subsidy equations:

| Jurisdiction | Dormant contracts |
|---|---|
| UK | minor gaming reach; payer probability conditional on recent gaming; regret and overspending conditional on recent purchase; parental monitoring; VGEC instrument and rate |
| KR | consumption propensity by income quintile; gaming reach at ages 10–69; mobile share conditional on being a gamer; monthly mobile and IAP spending; KOCCA instrument, cap, and payment shares |
| JP | smartphone-game reach; nonpayer mass; payer spending body; ever-per-title extreme-spend tail; complete-gacha subtype; JLOX+ instrument, rate, and cap |
| BE | material/social deprivation by age; gaming tax-shelter instrument |

Conditioning must be preserved. In particular, a payer-only statistic cannot be
applied to all players, Korean mobile spending has unresolved zero-spender
inclusion, and the Japanese extreme-spend observation is an ever-per-title tail,
not a monthly event probability.

## Source-register inventory

The register contains 26 official or institutional records. A runtime rule link
below affects the current world; a contract-only anchor is parsed into
provenance but does not change outcomes at the current fixed reference median.

| Source ID | Geography/topic | Current role |
|---|---|---|
| [`EUROSTAT_ILC_DI03`](https://ec.europa.eu/eurostat/databrowser/view/ilc_di03/default/table?lang=en) | European income by age/sex | Context-only; no current numeric extraction |
| [`EUROSTAT_DEMO_PJANIND`](https://ec.europa.eu/eurostat/databrowser/view/demo_pjanind/default/table?lang=en) | European population structure | Context-only; current age weights are illustrative |
| [`OECD_HOUSEHOLD_DISPOSABLE_INCOME`](https://www.oecd.org/en/data/indicators/household-disposable-income.html) | OECD disposable income | Context-only; no reviewed PPP or FX rate is loaded into the implemented conversion contract |
| [`ONS_HDI_FYE2024`](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/bulletins/householddisposableincomeandinequality/latest) | UK disposable income | Contract-only nominal anchor and scale metadata; the runtime median is fixed |
| [`OFCOM_CHILD_SPENDING_2025`](https://www.ofcom.org.uk/media-use-and-attitudes/media-habits-children/top-trends-from-our-latest-look-at-uk-childrens-online-lives) | UK children's online lives | Context-only; the profile uses the detailed PDF record instead |
| [`UK_LOOT_BOX_RESPONSE_2022`](https://www.gov.uk/government/calls-for-evidence/loot-boxes-in-video-games-call-for-evidence/outcome/government-response-to-the-call-for-evidence-on-loot-boxes-in-video-games) | UK loot-box evidence review | Context-only causal and harm background |
| [`WHO_GAMING_DISORDER`](https://www.who.int/standards/classifications/frequently-asked-questions/gaming-disorder) | Functional-impairment definition | Context-only; the model does not diagnose gaming disorder |
| [`FTC_EPIC_ORDER_2023`](https://www.ftc.gov/news-events/news/press-releases/2023/03/ftc-finalizes-order-requiring-fortnite-maker-epic-games-pay-245-million-tricking-users-making) | Dark patterns and unwanted charges | Context-only case evidence, not a prevalence estimate |
| [`AGCM_MOBILE_GAMES_2026`](https://www.agcm.it/media/comunicati-stampa/2026/1/PS13020-PS13039) | Italian investigations | Context-only; an investigation is not a final finding |
| [`USK_CRITERIA_2023`](https://usk.de/usk-pressemitteilung-umsetzung-neues-jugendschutzgesetz/) | German age-rating criteria | Context-only |
| [`CMA_GAME_PRINCIPLES`](https://www.gov.uk/government/publications/buying-features-in-online-games-advice-for-parents-and-carers) | UK price/consent/minor principles | Consumed by selected UK generic rule switches |
| [`EU_SILC_USER_GUIDE`](https://ec.europa.eu/eurostat/documents/203647/16195750/2021_Doc65_EUSILC_User_Guide.pdf) | Income equivalisation and microdata | Context-only methodology |
| [`OFCOM_CHILD_SPENDING_PDF_2025`](https://www.ofcom.org.uk/siteassets/resources/documents/online-safety/research-statistics-and-data/online-services-research/childrens-online-spending-and-potential-financial-harm-quantitative-research.pdf?v=399305) | UK child spending and harm | Contract-only national metrics |
| [`HMRC_VGEC`](https://www.gov.uk/hmrc-internal-manuals/creative-industries-expenditure-credit-manual/crec061300) | UK public funding | Contract-only programme instrument and rate |
| [`KOCCA_GAME_USER_2024`](https://welcon.kocca.kr/mobile/en/support/resources/377) | Korean game use and conditional spend | Contract-only national metrics |
| [`KOSTAT_HOUSEHOLD_Q4_2024`](https://www.kostat.go.kr/boardDownload.es?bid=11736&list_no=436048&seq=2) | Korean household income/expenditure | Contract-only central-quintile anchor and propensity values |
| [`MCST_ODDS_DISCLOSURE_2024`](https://www.mcst.go.kr/english/policy/pressView.jsp?pSeq=407) | Korean probability disclosure | Consumed by the odds-disclosure rule |
| [`KOCCA_PRODUCTION_SUPPORT_2024`](https://welcon.kocca.kr/ko/info/business/1953703) | Korean production support | Contract-only programme mechanics |
| [`JP_SMARTPHONE_GAMES_2016`](https://www.cao.go.jp/consumer/iinkaikouhyou/2016/0920_iken.html) | Japanese smartphone-game survey | Contract-only; underlying observations date to 2015 |
| [`JP_COMPLETE_GACHA_FAQ`](https://www.caa.go.jp/policies/policy/representation/fair_labeling/faq/card) | Japanese complete-gacha scope | Contract-only subtype; not represented by the generic runtime restriction |
| [`METI_JLOX_PLUS_2024`](https://www.hkd.meti.go.jp/hokch/20240403/index.htm) | Japanese content support | Contract-only programme mechanics |
| [`STATBEL_SILC_2025`](https://statbel.fgov.be/en/themes/households/poverty-and-living-conditions/faq) | Belgian disposable income | Contract-only nominal anchor and scale metadata; the runtime median is fixed |
| [`STATBEL_DEPRIVATION_2025`](https://statbel.fgov.be/en/themes/households/poverty-and-living-conditions/material-and-social-deprivation) | Belgian material/social deprivation | Contract-only age anchors; not a direct liquidity equation |
| [`BE_GAMING_COMMISSION_LOOT_BOX_2018`](https://www.gamingcommission.be/sites/default/files/2021-08/onderzoeksrapport-loot-boxen-Engels-publicatie.pdf) | Belgian loot-box classification | Scope anchor for an illustrative runtime switch |
| [`EU_CPC_VIRTUAL_CURRENCY_2025`](https://commission.europa.eu/document/download/8af13e88-6540-436c-b137-9853e7fe866a_en?filename=Key+principles+on+in-game+virtual+currencies.pdf) | EU virtual-currency principles | Consumed by selected Belgian generic rule switches |
| [`BE_GAMING_TAX_SHELTER`](https://finance.belgium.be/en/node/16499) | Belgian public funding | Contract-only programme instrument |

## Validation and precedence

Validation occurs at several boundaries:

1. `load_config` parses one run scenario and validates types, ranges, positive
   intervals, and calendar alignment with `tick_days`.
2. `load_profile_bundle` parses the registered TOML inputs, validates source
   integrity and canonical ISO retrieval dates, hashes the exact files, builds
   metric, local-money, optional FX/PPP conversion contracts, and the separately
   registered population-evidence assessment, and creates country and state
   agents. Population evidence is retained in lineage but not projected into
   those agents; the separate population-design file is not selected here.
3. `World.create` applies `allow_synthetic`, checks the duplicated shared
   behavioural values for exact equality, and applies the run-level audit
   parameters to each state.
4. Campaign mode separately requires the scenario, every used profile contract,
   every used source, every nominal anchor, and the money scale to be
   `CALIBRATED`. Pooled money additionally requires complete, calibrated,
   common-basis conversion coverage and exact internal-scale coherence.
   Population comparability is independently fail-closed. Neither evidence-
   bundle schema v1 nor static population-design schema v1 can establish the
   missing signature, authentic held-out source units, configured runtime use,
   dedicated output integration, or balance result.

`smoke.toml` permits synthetic dependencies and is the intended executable
structural check. `base.toml` sets `allow_synthetic=false`; with the current
synthetic profile dependencies it is intentionally rejected at world creation.
It is a future-scale configuration, not an authorized campaign.

The policy runner retains the exact `CountryProfile` tuple it used. A canonical
snapshot fingerprints every profile field and, for a loaded `ProfileBundle`,
the metric contracts, money scales, conversion contracts, population bundle,
verified population results, and typed population-readiness assessment.
Registered profile-input lineage is now schema version 4; versions 1–3 remain
readable on their historical surfaces. Exported manifests identify the actual
profile codes, jurisdiction and source-register hashes, global source retrieval
date, population-bundle digest and blockers, and compact contract-status
summaries. A caller-supplied bare profile tuple is fingerprinted but marked
`unregistered_custom_profiles`; it is never attributed to the repository's
default evidence files.

For the opt-in projected table only, policy cohort identity additionally hashes
the projected sidecar assignment, which itself binds the content-derived runtime
projection, cell semantics, ordered player IDs, and cell indices. Consumers
recompute the nested attestation and reject stale or mutated indices. This
protects paired-cohort identity in an explicit projected run; it does not make
that run configured, calibrated, or export a target-population estimand.

Output transformations have their own exhaustive registry. This keeps an input
contract such as an age or income anchor distinct from a derived table measure
such as a repeated-seed variance or confidence bound. The registry snapshot is
content-addressed in the run manifest and links back to the exact profile-input
fingerprint. Its current `SYNTHETIC` statuses and absent metric-level empirical
retrieval dates are explicit campaign blockers; a source-register date alone
does not promote an output transformation.

## Reproducibility gaps

The source register records publisher, title, URL, period, geography, declared
support scope, status, and a global retrieval date. The loader retains that date
on every source record, and run lineage hashes the registry file. This does not
turn the linked publications into immutable source snapshots. The register does
not yet contain:

- immutable snapshots or archived URLs;
- raw downloaded tables or extracts;
- table, cell, row, page, or variable identifiers;
- transcription and transformation scripts;
- hashes or checksums for underlying downloaded source artifacts;
- sampling weights, uncertainty estimates, or revision identifiers;
- licenses and redistribution constraints.

Some URLs point to “latest” or current guidance and may change after retrieval.
The loader performs no network request and verifies no page content. Reproducing
a future calibrated estimate will require immutable raw artifacts, scripted
extraction, exact environment capture, and transformation-level tests in
addition to the current register.

The population-evidence schema supplies a place to register exact local CSV
bytes and a deterministic extraction, but the default bundle contains no such
artifact and no publisher signature. The static design schema supplies exact
domains, partition declarations, target counts, and Hamilton weights, but its
default is also empty and illustrative. Even populated files would establish
declared reproducibility only. Publisher authentication, independently verified
held-out source units, reviewed calibration, configuration/call-site selection,
an adapter binding static counts and source-to-runtime conversion, a dedicated
weighted-output writer/profile, and held-out balance checks remain separate
work. No population readiness gate has been cleared and no full campaign has
been run.
