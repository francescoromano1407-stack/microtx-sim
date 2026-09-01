# UK adults 2024 calibration evidence and campaign readiness

## Decision summary

The public evidence cache supports a **partial UK calibration**, not a complete
behavioural calibration. It is strong enough to define an age-by-sex population
margin, several time-use targets, rounded household-income margins, and explicit
wage and expenditure proxies. It does not identify individual microtransaction
spending, unplanned spending, the simulator's global decision temperature, an
adult UK high-risk threshold, the composite-harm weights, or the conversion from
`simulation_cents` to observed money.

A new exploratory campaign should **not** be run yet. The new evidence is not
connected to the outcome-changing runtime fields, several key outputs remain
unquantified or normative, and no independent validation gate has been passed.
Once the integration and validation conditions in this document are satisfied,
a new successor campaign would be preferable to interpreting the previous
synthetic campaign under changed weights. That future campaign must be separately
authorized; it is not executed as part of this calibration checkpoint.

This document treats the instructions in
`data/public_calibration_sources_uk_adults_2024/CALIBRATION_EVIDENCE_INDEX.md`
as evidence notes, not as authority to change the model or run a campaign.

### Committed artefacts and status

The machine-readable package is `status=PARTIAL` and
`campaign_ready=false`. It comprises `calibration_bundle.json`, `targets.csv`,
`population_weights.csv`, and `source_manifest.json` under
`inputs/calibration/uk-adults-2024-v1`. Source retrieval timestamps were not
recorded with exact times in the supplied cache, so `retrieved_at` remains
`null`. Known date-only metadata is preserved separately; for example, the
already-bound ECB bundle records `retrieved_on=2026-08-30`. The manifest-level
`verified_at=2026-09-01` records local byte/hash verification only; it is not a
source access or publication date.

## Estimand and evidence roles

The proposed baseline population is UK usual residents aged 18--64 at mid-2024.
This age range avoids silently extrapolating adult survey evidence to minors and
uses five constructed bands aggregated from ONS single-year counts (the
18--24 band spans seven years). Evidence with a different
population, period, denominator, or construct is retained but is not silently
transported to that baseline.

The following roles are deliberately separate.

| Role | Permitted use in this calibration |
| --- | --- |
| `CALIBRATION` | May enter a declared fit objective or synthetic-population margin when population, period, unit, and conditioning match. |
| `VALIDATION` | Held out from fitting and used only against acceptance thresholds fixed before inspection of fitted results. |
| `DIAGNOSTIC` | Checks sign, order, scale, or plausibility; does not identify a UK population parameter. |
| `PROXY` | Represents a related construct under an explicit scenario assumption. It is not the construct itself. |
| `UNQUANTIFIED` | No defensible point target is available from the collected public evidence. |
| `NORMATIVE` | Requires a value judgement and sensitivity analysis, rather than empirical point estimation. |

`CALIBRATION` is not a property of a publisher alone. It applies only to a
specific source cell, extraction, transformation, and estimand. For example,
the ONS all-adult gaming mean is official evidence, but its 18-and-over
denominator does not make it an exact 18--64 target.

## Exact targets extracted from the cache

### Population structure

The values below were extracted from the ONS mid-2024 workbook, UK row
`K02000001`, sheets `MYE2 - Persons`, `MYE2 - Females`, and `MYE2 - Males`,
single-year ages 18--64 in exact range `W9:BQ9`. Counts reconcile exactly within each
band. Integer counts are the exact source transcriptions and remain the
authoritative basis for allocation. `population_weights.csv` stores their
15-decimal rounded ratios for convenience (sum `1.000000000000001` within the
declared tolerance); consumers should derive exact allocation from counts when
rounding matters.

The population reference date is mid-2024, but ONS released this edition on 26
September 2025. The reference year must therefore not be misreported as its
publication year.

| Age | Persons | Female | Male | Derived person weight |
| --- | ---: | ---: | ---: | ---: |
| 18--24 | 5,791,521 | 2,821,237 | 2,970,284 | 0.138352599 |
| 25--34 | 9,345,597 | 4,754,911 | 4,590,686 | 0.223255278 |
| 35--44 | 9,304,892 | 4,805,400 | 4,499,492 | 0.222282884 |
| 45--54 | 8,513,477 | 4,343,516 | 4,169,961 | 0.203376914 |
| 55--64 | 8,905,100 | 4,554,117 | 4,350,983 | 0.212732325 |
| **18--64** | **41,860,587** | **21,279,181** | **20,581,406** | **1.000000000** |

These are defensible population margins. They do not identify gaming status,
payer history, household income, or any joint correlations with those fields.
The runtime now has an optional source-recorded `PlayerTable.sex` field and a
point-zero-only binding for these ten cells. It deterministically assigns the
source categories `FEMALE` and `MALE` to UK residents aged 18--64 and leaves the
field empty outside that scope. Coupled Hamilton rounding preserves each ONS
age-band sex total and SHA-256 ranking makes the within-cell synthetic
allocation reproducible. The assignment is aggregate-informed synthetic data,
not observed individual sex and not an inference about gender identity. It does
not change the age sampling margin, and all treatment entry points reject it.
Exact arithmetic reconciliation does not remove uncertainty from the official
population estimates.

Source: [ONS, mid-year population estimates, mid-2024 edition](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland).

### Time use and sleep

The March 2024 all-adult values come from ONS Time Use worksheet 24, table 24a.
They are UK averages for all adults aged 18 and over, not estimates restricted
to ages 18--64. Confidence limits are those published in the workbook.

| Activity | Mean minutes/day | Published 95% interval | Role for an 18--64 model |
| --- | ---: | ---: | --- |
| Sleeping | 518.5 | 513.4--523.7 | `DIAGNOSTIC`; calibratable only after age-scope reconciliation |
| Playing games or computer gaming | 16.8 | 13.5--20.0 | `DIAGNOSTIC`; 18+ denominator and realised diary time |
| Working | 133.2 | 125.2--141.2 | `DIAGNOSTIC`; age and employment scopes differ |
| Working from home | 49.2 | 43.4--55.1 | `DIAGNOSTIC`; it is a subset of working time |
| Socialising | 30.8 | 27.0--34.5 | `DIAGNOSTIC`; activity definitions must first match |

The age-specific `Sleep and rest` series in worksheet 7 is compatible with the
five proposed age bands, but is a broader construct than `Sleeping`.

| Age | Mean minutes/day | Published 95% interval |
| --- | ---: | ---: |
| 18--24 | 577.9 | 545.9--609.8 |
| 25--34 | 564.2 | 545.9--582.5 |
| 35--44 | 531.9 | 520.0--543.8 |
| 45--54 | 521.8 | 509.2--534.3 |
| 55--64 | 520.4 | 509.3--531.4 |

The separate March 2024 weekday/weekend workbook reports `Sleep and rest` of
524.0 minutes on weekdays, 578.4 minutes at weekends, and 539.5 minutes on an
average day. These values may constrain calendar allocation only if the runtime
implements the same day type and activity definition.

The 23 September--1 October 2023 all-adult values are reserved as temporal
validation: sleeping 522.8 minutes (517.3--528.2) and gaming 14.8 minutes
(11.7--17.9). They must not also enter the calibration loss. ONS describes these
time-use estimates as official statistics in development, which is an additional
reason to propagate their uncertainty rather than treating the means as exact
truth.

The holdout workbook contains a publisher-side metadata inconsistency:
worksheet 20 and the workbook cover identify September 2023, while the table's
column header says March 2023. The bundle records the cover/sheet period and
preserves this discrepancy as a limitation; it does not silently repair the
source bytes.

Source: [ONS, Time use in the UK](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/datasets/timeuseintheuk).

### Income, expenditure, and opportunity-value proxies

The DWP Family Resources Survey (FRS) 2023/24 workbook, sheet `2_6`, UK row in
`A9:N10`, gives the following rounded distribution of gross weekly household
income. Its unweighted sample size is 16,754.

| Gross weekly household income | Published share (%) |
| --- | ---: |
| Under GBP 200 | 5 |
| GBP 200 to under GBP 400 | 14 |
| GBP 400 to under GBP 600 | 18 |
| GBP 600 to under GBP 800 | 14 |
| GBP 800 to under GBP 1,000 | 11 |
| GBP 1,000 to under GBP 1,200 | 9 |
| GBP 1,200 to under GBP 1,400 | 7 |
| GBP 1,400 to under GBP 1,600 | 5 |
| GBP 1,600 to under GBP 1,800 | 4 |
| GBP 1,800 to under GBP 2,000 | 3 |
| GBP 2,000 or more | 11 |

The published rounded shares sum to 101%, which is plausible table rounding,
not a licence to silently rescale them to 100%. A fitting procedure should
model the rounding interval or document an explicit constrained reconciliation.
FRS household gross income is not the runtime's individual monthly disposable
income. Transport between those constructs requires household composition,
tax/benefit, period, and equivalisation assumptions that are not identified by
this table.

The corrected ONS Living Costs and Food workbook, sheet `3.1`, reports weekly
`Recreation & culture` expenditure by income decile 1--10 and all households as
GBP 31.1, 31.0, 38.7, 53.3, 62.6, 69.2, 73.3, 92.3, 99.1, 177.8, and 72.8.
These are aggregate budget diagnostics, not observed microtransaction budgets.
The corresponding `Computer software and games` row is `[0.40]`, `..`,
`[0.20]`, `[0.40]`, `[0.80]`, 1.50, 1.10, 0.50, 0.80, 1.30, and 0.70.
Brackets and `..` carry reliability or availability warnings in the source and
must remain visible. This row must not be converted into
`affordable_spending_share` or in-game spending.

ONS released the corrected FYE 2024 workbook on 11 June 2026 after identifying
a processing error in percentage standard errors in Table A1. The expenditure
means used here come from the content-addressed corrected workbook; this later
correction date is part of the source identity and does not turn the evidence
into a 2026 behavioural baseline.

The revised ONS Annual Survey of Hours and Earnings 2024 table for all employee
jobs in the UK reports 28,210 thousand jobs and gross hourly pay of GBP 11.64 at
the 10th percentile, GBP 17.09 at the median, GBP 21.65 at the mean, and GBP
34.75 at the 90th percentile. These values support explicit wage-proxy
opportunity-cost scenarios. A market wage is not automatically the welfare
value of leisure, sleep, or unpaid time.

Sources: [DWP, Family Resources Survey 2023/24](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2023-to-2024),
[ONS, Family spending workbook 1](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/datasets/familyspendingworkbook1detailedexpenditureandtrends),
and [ONS, earnings and hours by region and age](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/earningsandhoursworkedukregionbyagegroup).

### Open Play: conditional trace evidence, not a UK population target

Open Play v1.2.5 is a valuable longitudinal trace dataset, but its own manuscript
states that the final sample is unlikely to represent either the general
population or adults who play games. Recruitment combined partial screening
quotas and convenience sampling, participation required willingness to share
trace data and complete an intensive protocol, and the actual age scope is
18--40 in the UK and US. PlayStation and non-Steam PC play are not observed;
platform coverage and observation units differ. The local ZIP is attested to the
version-specific DOI [10.5281/zenodo.21986572](https://doi.org/10.5281/zenodo.21986572),
release date 17 August 2026, and Git commit
`74f6bbac98cde5118152fa751d72e06287659dc2`. The concept DOI is not used as the
exact-version locator.

A reproducible local filter (`qualified = TRUE`, `country = UK`, age 18--40)
was independently rerun against the content-addressed cache and selects 932
participants. Their mean age is 27.15 years, median 26, with 10th and 90th
percentiles of 21 and 35. Of these participants, 654 (70.17%) report the
category `Man`; those survey categories are not collapsed into the binary ONS
sex margin. The following aggregates are consequently
`DIAGNOSTIC`, or at most conditional calibration evidence for a separately
declared young-adult-gamer submodel.

| Clean measure | Coverage | Row-level minutes: median [p10, p90] | Interpretation |
| --- | ---: | ---: | --- |
| Nintendo (2022-05-01 to 2025-09-26) | 261 participants; 82,186 records | 27.375 [not bundled, 100.325] | Cleaned/merged platform sessions; first-party games only |
| Steam (2025-01-27 to 2025-10-06) | 831 participants; 296,707 records | 39.0 [not bundled, 62.0] | Inferred per-poll segments from cumulative hourly API deltas; not observed behavioural sessions |
| Xbox | 0 UK participants; 0 records | Not available | No UK target |

Intake dates span 26 September 2024 to 9 June 2025. To match the manuscript's
strict `<50 hours/week` analysis, 19 responses exactly equal to 3,000 minutes
are excluded. The remaining 830 selected participants have mean self-reported
weekly play of 1,004.27 minutes and median 900. Ofcom/Ampere reports 417 minutes
per week for UK gamers aged 16--64 in Q2 2024, so the Open Play mean is 2.408
times the external gamer-only mean. The denominators, dates, and age ranges
still differ, so this is not a formal validation residual, but the magnitude is
strong evidence against transporting Open Play as a population play-level
target.

Open Play also partly addresses same-sample gaming-risk measurement, but only
conditionally. The first complete four-item Gaming Disorder Test response among
694 selected participants has a mean sum of 8.716 on the 4--20 scale. The first
complete PSQI sleep-duration response among 518 selected participants averages
429.83 minutes per day. The PSQI value is one self-reported sleep-duration item,
not a full PSQI score or diary measure; the GDT value is a severity measure, not
prevalence or diagnosis. Neither is an adult UK prevalence estimate and neither
observes microtransaction spending. First-complete responses avoid pooling
repeated waves as if they were independent people.
First-complete GDT dates span 1 October 2024 to 15 July 2025; first-complete
PSQI-duration dates span 26 February to 27 August 2025. These periods must not be
silently relabelled as a single 2024 baseline.

The published Open Play design says daily surveys and time-use diaries were
US-only. The local clean file nevertheless contains 43 time-use rows belonging
to three qualified UK IDs. That inconsistency is a fail-closed data-quality
flag: those rows are excluded from calibration and validation until their
cohort and processing provenance are resolved. It does not justify creating a
UK time-use target.

Ofcom Online Nation 2024 should remain an aggregate validation source rather
than being used to tune the same gaming quantities. The reported play figures
are third-party Ampere self-reports for gamers aged 16--64, not Ofcom-estimated
population telemetry, and no sampling uncertainty is published for the cited
means. Population, age groups, platform measures, and denominator must be
matched explicitly before comparison with Open Play or the simulator.

Source: [Ofcom, Online Nation 2024](https://www.ofcom.org.uk/siteassets/resources/documents/research-and-data/online-research/online-nation/2024/online-nation-2024-report.pdf?v=386238).

### Microtransaction incidence and spending tails remain unidentified

The proposed claim that only 2--5% of active UK players make in-app purchases
is not supported by the external open-source cross-checks and must not be used
as a calibration target. These checks postdate the immutable v1 source
manifest and are not part of its 15 attested source records. Ofcom's 2020 survey
instead reports that 31% of UK adult
game players had **ever** bought additional content in a free-to-play game and
25% had done so in a purchased game. Those overlapping, historical
ever-purchase measures are neither a 2024 active-payer rate nor a measure of
monthly microtransaction incidence.

The DCMS response cites an Ipsos MORI survey commissioned by an industry trade
association: 7% of video-game players reportedly paid for a loot box in the
preceding 12 months to December 2020, while 45% spent money on video games
generally. The cited underlying report is unpublished, so those estimates
cannot be independently audited. Separately, the DCMS call-for-evidence player
survey was self-selected and explicitly non-representative. Loot boxes are only
one microtransaction subtype, and the government response states that robust
market data are lacking. These quantities therefore cannot be combined into a
2--5% all-microtransaction payer target. Ukie's 2024 valuation provides
aggregate market-category totals, not a count or sampling frame of individual
active payers. The local Open Play release contains play telemetry and
questionnaires but no transaction price, purchase, refund, exposure, or
choice-set field.

Consequently, both active-payer incidence and the conditional spending
distribution remain unidentified and unquantified in this review; a successor
bundle must encode them explicitly as `UNQUANTIFIED`. The immutable v1
`targets.csv` has no payer-incidence or spending-tail row. A generalized Pareto distribution is not
identified by a payer fraction and one aggregate ARPU: its threshold, scale,
and shape require additional independent tail information, and the aggregation
must first reconcile payer-only spending with population ARPU. Pareto/GPD
families may be declared in sensitivity analysis with externally chosen ranges;
they must not be labelled empirically calibrated from the current bundle.

Sources: [Ofcom, Online Nation 2020](https://www.ofcom.org.uk/siteassets/resources/documents/research-and-data/online-research/online-nation/2020/online-nation-2020-report.pdf?v=324898),
[DCMS, government response on loot boxes](https://www.gov.uk/government/calls-for-evidence/loot-boxes-in-video-games-call-for-evidence/outcome/government-response-to-the-call-for-evidence-on-loot-boxes-in-video-games),
[InGAME rapid evidence assessment publication](https://www.gov.uk/government/calls-for-evidence/loot-boxes-in-video-games-call-for-evidence),
and [Ukie 2024 consumer-market valuation release](https://ukie.org.uk/publications/time-to-press-start-on-growth).
These are external cross-check sources only; they are not retroactively added to
the content-addressed v1 source manifest.

### Monetary series and normative outcomes

The cache also contains ONS CPIH version 53, official ECB 2024 nominal FX
records, World Bank 2024 FX and PPP observations, and the Eurostat--OECD PPP
manual. They support separate inflation, nominal-conversion, and purchasing-
power contracts. They do not identify a conversion from `simulation_cents` to
observed GBP or a monetary value for simulated harm.

The separately recorded UK values are: ECB annual-average 2024 GBP/EUR
0.8466166015625 (GBP per EUR); CPIH overall index 134.6 in November 2024 versus
130.0 in November 2023, a 3.538% change; World Bank private-consumption PPP
0.682739 GBP per international dollar; GDP PPP 0.664153; and official FX
0.782414580986262 GBP per USD. Quote orientation and construct remain explicit:
GBP-to-EUR conversion requires inverting the ECB quote, household-consumption
PPP is not GDP PPP, and none of these values is a simulator-unit bridge.

Composite-harm weights remain `NORMATIVE`. The component outcomes should be
reported before any composite, double counting should be checked, and plausible
weight sets should be included in sensitivity analysis. A dominance of the
composite by one component is a scale/normalisation warning, not an empirical
finding that the component is intrinsically more important.

## Raw-cache, privacy, and licence handling

The source directory is a local evidence cache of 214 files and 511,833,861
bytes. It should remain outside Git. The repository should retain only small,
reviewable derived aggregates; exact source identifiers and extraction
locations; and byte lengths and SHA-256 digests where licence permits. Before
any fitted campaign, it should also retain privacy-preserving code that
reproduces the aggregate extraction. Workbooks, PDFs, archives, pseudonymised
participant rows, and any participant identifier must not be committed merely
to make the calibration portable.

Open Play's local `zenodo.json` says CC0, while the bundled `LICENSE` modifies
CC0 with an explicit no-reidentification clause. Repository handling follows the
stricter bundled terms and records this metadata conflict. No attempt may be
made to identify participants or link their rows to external data. Outputs
should suppress participant IDs and avoid disclosing rare cells that could
facilitate singling out. The aggregate values above are sufficient for this
checkpoint. Official UK files retain their publisher and Open Government
Licence notices; third-party materials retain their own terms.

Any future subject-access data-donation study requires ethics approval, informed
consent, data minimisation, a lawful basis, secure transfer and storage,
retention/deletion rules, participant withdrawal procedures, and disclosure-risk
review. The right of access gives an individual a copy of personal information;
it does not guarantee that a platform will expose a standardized research table
with price, currency, SKU, offer exposure, and rejection fields. See the
[EDPB right-of-access guidelines](https://www.edpb.europa.eu/documents/guideline/guidelines-012022-on-data-subject-rights-right-of-access_en)
and the [ICO subject-access guide](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/subject-access-requests/a-guide-to-subject-access/).
Data donation can also have substantial consent and successful-upload attrition
(Keusch et al., 2024, [doi:10.1177/1525822X231225907](https://doi.org/10.1177/1525822X231225907)),
so donor selection and zero-spender coverage must be measured rather than
assumed away.

## Assessment of the four proposed gap solutions

### 1. Transaction donation plus storefront reconstruction

**Decision: promising as a future pilot; not a current calibration source.**

Participant-donated platform exports can recover some retrospective trace data
and are preferable to invented transaction distributions. Before a full study,
small platform-specific pilots must inventory the fields actually returned,
time coverage, refunds, regional prices, virtual-currency purchases, bundles,
subscriptions, and missing or off-platform payments. The export schema and
successful-donation rate should be treated as measured study outcomes.

Catalog history can enrich a transaction record, but catalog availability is
not proof that a participant saw an offer and cannot reconstruct rejected
alternatives. Collection must use official APIs, store pages, or licensed
archives under their applicable terms. SteamDB explicitly says that it has no
public API and disallows automated crawling without permission; see its
[FAQ](https://steamdb.info/faq/). Skinport describes a secondary-market sales
[history API](https://docs.skinport.com/sales/history), not a complete
first-party choice set. These sources therefore cannot be treated as a
universal catalog reconstruction layer.

**Hawkes fallback: rejected for gap filling.** A Hawkes process is useful for
describing self-exciting event timing after sufficiently informative timestamps
exist (Hawkes, 1971,
[doi:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83)). It cannot
recover missing amounts, non-donors, zero-spenders, unobserved offers, or rejected
choices, and a fitted burst process does not make a small convenience sample
representative.

### 2. Discrete-choice experiment for `decision.temperature`

**Decision: promising only after the model parameter is redesigned.**

A preregistered DCE with randomized prices and mechanics, realistic choice
cards, repeated tasks, and a genuine opt-out can estimate relative preferences
and heterogeneity. It should record the exact choice set and use panel-aware
standard errors or a mixed-logit design where justified.

It does not directly identify the current absolute `decision.temperature`.
Random-utility models identify utility coefficients relative to an error-scale
normalization: multiplying systematic utility and the disturbance by the same
positive constant leaves choices unchanged. The proposed
`tau = sqrt(6) * sigma_epsilon / pi` is therefore a chosen extreme-value
normalization unless an external utility scale is identified; it is not an
independently observed temperature. See Train (2009),
[Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html),
and Swait and Louviere (1993),
[doi:10.1177/002224379303000303](https://doi.org/10.1177/002224379303000303).

The runtime uses one temperature across eight mutually exclusive life actions
(`PLAY`, `PURCHASE`, `STOP`, `SLEEP`, `STUDY_WORK`, `SOCIALIZE`, `EXERCISE`, and
`OTHER`). A skin/loot-box/opt-out DCE identifies a purchase-choice scale under
its own task design, not this global life-allocation scale. A separate purchase
choice layer and an explicit bridge between DCE attributes and simulator utility
are prerequisites.

### 3. EMA for planned versus unplanned spending

**Decision: the strongest proposed primary-data design; not replaceable by the
suggested algebraic fallback.**

A 14--30 day ecological momentary assessment can ask, before a session, whether
the participant plans to spend and the maximum intended amount, then record
actual spending and triggering context immediately after the session. With
consent, donated transaction records can reconcile amounts and timestamps.
Repeated real-time sampling reduces retrospective reconstruction and measures
the intended/actual distinction close to the event (Shiffman, Stone, and
Hufford, 2008,
[doi:10.1146/annurev.clinpsy.3.022806.091415](https://doi.org/10.1146/annurev.clinpsy.3.022806.091415)).
The design still requires explicit missingness, compliance, reactivity, and
participant-burden analyses.

The proposed threshold equations do not identify intention: defining planned
spending as a capped fraction of a discretionary budget simply embeds the
answer in an assumption. They can fail to preserve the stated accounting
identity under ambiguous triggers and do not measure counterfactual intent.
They may be retained only as clearly labelled sensitivity scenarios, never as
empirical calibration.

### 4. Same-sample UK risk measure or statistical matching

**Decision: a same-person UK micro-study is promising; statistical matching is
diagnostic only.**

A UK adult gamer study should collect spending, play, sleep, work interference,
financial strain, and impairment in the same people. The PGSI may be retained
for gambling-related behaviour, but it was designed as a gambling screen and
must not be relabelled as gaming disorder or a clinical diagnosis. The UK
Gambling Commission documents its scope and cautions in
[Problem gambling screens](https://www.gamblingcommission.gov.uk/statistics-and-research/publication/problem-gambling-screens).
For gaming impairment, a WHO/ICD-11-aligned measure such as the Gaming Disorder
Test is conceptually closer; see the [WHO gaming-disorder definition](https://www.who.int/standards/classifications/frequently-asked-questions/gaming-disorder)
and Pontes et al. (2021),
[doi:10.1007/s11469-019-00088-z](https://doi.org/10.1007/s11469-019-00088-z).
Screening scores remain screening scores, and cross-sectional associations do
not identify causal effects of microtransactions.

Prolific offers UK representative samples, but its published representativeness is by a
limited set of demographic strata rather than by gaming, spending, or
impairment. Its terms verified on 1 September 2026 require at least GBP 6/hour,
recommend GBP 9/hour, and add a 33.3% qualified academic/non-profit platform
fee for an eligible account registered with an institutional address; the
otherwise-published fee is 42.8%. At the qualified rate, a 20-minute one-off
study therefore costs about GBP 4,000 for 1,000 participants or GBP 6,000 for
1,500 before VAT, excluding pilots, bonuses, longitudinal attrition, and
transaction-donation work. The suggested GBP
1,000--2,000 budget is not credible for the proposed sample and protocol under
those terms. Pricing and sample limits are dynamic and must be rechecked at
launch. The current UK standard representative-sample range is 300--5,000, but
the published age strata end at `55+`, rather than 55--64, and custom gamer
screening changes the population represented. See Prolific's
[pricing](https://researcher-help.prolific.com/en/articles/445239-what-is-your-pricing)
and [representative-sample specification](https://researcher-help.prolific.com/en/articles/445161-what-are-representative-samples-on-prolific).

Matching separate behavioural and risk surveys on age, sex, gaming hours, or
income can create a synthetic file, but it cannot reveal the unobserved
within-person dependence between spending and impairment without an identifying
assumption. Mahalanobis, Gower, or nearest-neighbour distance does not solve that
fundamental data-fusion problem. The result may support sensitivity analyses
under explicit conditional-independence assumptions, not a calibrated
`P(harm | spending, sleep)`. See Eurostat's methodological review,
[Statistical matching: a model based approach for data integration](https://ec.europa.eu/eurostat/web/products-statistical-working-papers/-/ks-ra-13-020).

The proposed official questionnaires do not contain the shared matching vector
as stated. The Gambling Survey for Great Britain covers Great Britain rather
than the whole UK and does not jointly measure microtransactions, gaming hours,
and sleep. The Adult Psychiatric Morbidity Survey is England-only and contains
sleep and mental-health measures but no video-game or microtransaction module.
See the [GSGB 2024 questionnaire](https://www.gamblingcommission.gov.uk/report/gambling-survey-for-great-britain-annual-report-2024-official-statistics/gsgb-annual-report-2024-appendix-a-online-questionnaire)
and [APMS phase-one questionnaire](https://digital.nhs.uk/binaries/content/assets/website-assets/publications/apms-2023-4/appendix-c---phase-one-questionnaire.pdf).

## Runtime integration limitations

The evidence cannot change simulation outcomes merely by being present in a
source folder.

- `PlayerTable` now has an optional, immutable source-recorded sex field. A
  schema-v3 projection attestation binds its source, scope, bundle digest,
  weights digest, derivation-input digest, assignment method, and complete
  vector for point-zero only. A source-specific verifier recomputes the coupled
  Hamilton allocation. World, policy-day, player-dynamics, multi-cycle, and
  campaign entry points all reject it before treatment until that contract is
  extended.
- The point-zero bridge consumes only the ten ONS age-by-sex rows. It does not
  resample the age margin. All 75 separate target rows are heterogeneous
  calibration, validation, diagnostic, proxy, unquantified, or normative
  records rather than sampling weights; they are additional to the ten
  `population_weights.csv` rows.
- The profile loader currently requires exactly UK, Korea, Japan, and Belgium.
  A UK-only estimand requires an explicit loader/runtime change or a declared UK
  subpopulation within the four-country cohort.
- Projected gamer and payer-history labels are sidecar metadata and do not set
  gameplay, payment access, or spending history.
- Work/study, leisure, sleep, intended play, intended spending, and related
  welfare state are initialized from hard-coded illustrative equations in
  `consumers/welfare.py`, not from these targets.
- `decision.temperature` is one global coefficient over eight actions. It is
  not identified by Open Play telemetry or aggregate reports.
- The high-risk rule is hard-coded as composite harm at least 0.35, harmful
  spending share at least 0.10, or sleep score at least 0.50. No adult UK
  prevalence or external threshold validates that classification.
- Household-income margins cannot be inserted into the runtime personal-income
  field without a documented transport model. ASHE wages are proxies, not a
  welfare valuation of displaced time.
- `simulation_cents` has no observed transaction bridge. FX, CPIH, and PPP
  evidence cannot supply that missing scale.

Accordingly, this checkpoint should produce a typed, content-addressed target
bundle and diagnostics only. It must not claim that a runtime configuration is
calibrated until the execution lineage proves which target changed which
outcome-changing field.

The resulting bundle is `inputs/calibration/uk-adults-2024-v1`. It contains 75
typed target rows, ten ONS age-by-sex cells, and 15 content-addressed source
records. Its original source bytes can be re-attested, without executing a
scenario, with:

```text
python tools/validate_uk_adults_2024_calibration.py
```

That command verifies the declared source and companion bytes, typed contracts,
cross-file population/FRS reconciliation, and static compatibility diagnostics. It does
not rebuild all 75 rows from source worksheets and participant files. The
aggregate extraction was locally rerun, but a complete checked-in extraction
recipe is not yet repository-reproducible; this remains a campaign-readiness
blocker, especially for Open Play filters, first-response selection, and
quantiles. Holdout separation also relies on declared free-text source locators:
the loader rejects identical records and byte-identical source aliases, but it
does not prove that two different locators select disjoint rows. A future fitted
bundle must bind canonical selectors or row-set/partition digests.

The initializer-only runtime audit is separate and intentionally returns a
non-zero status while any structural gate fails:

```text
python tools/run_point_zero_audit.py --pretty
```

It requires the ignored, hash-attested local source cache, initializes only the
population and player-life tables for seed 101, emits JSON, and returns `0` for
pass or `1` for fail/error. It never initializes a scenario, executes a policy
day, or dispatches a campaign.

## Point-zero runtime audit (no campaign executed)

A read-only seed-101 initializer audit used the existing 50,000-player
exploratory configuration. It selected 10,024 UK records aged 18--64; no policy
day, scenario contrast, sensitivity run, or exploratory campaign was executed.
The checked-in `tools/run_point_zero_audit.py` command reproduces this bounded
initialization from the strictly re-attested local cache and exits fail-closed.
The current report has six passing and six failing gates. The additional
fail-closed identification gate records that schema v1 does not identify
`P(gamer | ONS age-by-sex cell)`; successful behavioural wiring alone cannot
turn the illustrative `baseline_gamer` sidecar into empirical evidence.
The current weighted age distribution has total-variation distance 3.31% from
the ONS margin: differences are +1.104, -1.179, +0.162, +2.040, and -2.126
percentage points for ages 18--24 through 55--64, respectively. The existing
55--69 source band and uniform within-band age sampling cannot exactly reproduce
the five new bands.

The runtime bridge assigns 5,093 selected records to the source category
`FEMALE` and 4,931 to `MALE`; all 39,976 out-of-scope rows remain empty. Coupled
Hamilton remainders preserve the exact rounded ONS sex total within every age
band while deterministic SHA-256 ranking allocates those totals across existing
runtime cells. The complete vector, derivation inputs, and calibration-bundle
identity enter the assignment and execution digests. Because the bridge applies
ONS conditional sex proportions but does not resample age, the joint
age-by-sex total-variation distance is also 3.31% and fails the predeclared 2%
gate. A passing lineage gate therefore proves consumption, not population fit.

The initialized means are 480.22 minutes/day for sleep need, 419.99 for
work/study obligation, 78.60 for social obligation, and 82.83 for intended
play. The ONS diagnostic means are 518.5 sleeping, 133.2 working, 30.8
socialising, and 16.8 realised gaming minutes/day. These are not legitimate
fit residuals because needs, obligations, and intentions differ from realised
primary-activity diary time. Their scale nevertheless rules out simply
relabeling the current fields as calibrated outputs: intended play is 4.93
times the ONS all-adult gaming mean, and the initializer has no observed adult
age gradient in sleep need.

The most serious structural check is gamer binding. `baseline_gamer` remains
sidecar metadata, `current_game=-1` for every initialized player, and 99.743% of
sidecar non-gamers still receive positive intended play. Gamer and non-gamer
means are nearly identical (83.07 versus 82.55 minutes/day). Sidecar gamers
average 581.48 intended minutes/week, 39.4% above the Ofcom/Ampere gamer-only
mean of 417. For ages 18--40, the corresponding simulator mean is 584.66, 41.8%
below the manuscript-aligned selected Open Play mean of 1,004.27; the latter remains diagnostic
because Open Play is nonrepresentative. A 60-minute decision step also cannot
be compared directly with 39-minute inferred Steam per-poll records or 27.375-
minute Nintendo records without a declared measurement operator.

No source value itself is too implausible to retain: the FRS 101% total is
published rounding, and the anomalous UK-linked Open Play diary rows remain
excluded. The runtime/data mismatch is, however, large enough to stop the
workflow before a new campaign. `campaign_ready=false` is therefore a
scientific fail-closed decision, not merely an implementation status.

## Disposition of the seven proposed runtime changes

| Proposed change | Current disposition | Scientific reason |
| --- | --- | --- |
| 1. Two-stage gamer hurdle | **Blocked, measured by the audit** | The bundle does not identify `P(gamer | ONS age-by-sex cell)`. The existing `baseline_gamer` labels are illustrative sidecar metadata, so forcing their non-gamers to zero would turn an unsupported label into behavior. The audit instead fails on 99.743% positive play intention, 100% positive spending limits, and the absent purchase-probability hurdle. |
| 2. Random-utility/logit choice | **Already present; no rewrite** | The decision layers already use Gumbel/logit utility with play obligations, disposable income, liquidity/credit, player traits, price burden, and an outside/no-purchase option. The unresolved issue is that the projected gamer/payer labels do not enter those equations; global temperature remains sensitivity-only. |
| 3. Ofcom versus Open Play | **Separated by evidence role** | Ofcom/Ampere gamer means are retained as external diagnostics/possible validation, while Open Play is a non-representative conditional trace diagnostic. Neither is used as the other's scale correction or as an ONS participation probability. |
| 4. Runtime sex field | **Implemented for point zero** | Optional source-recorded `FEMALE`/`MALE` values, exact scope/vector, derivation inputs, and canonical coupled-Hamilton method now enter assignment and execution lineage. They are synthetic aggregate allocations, not individual observations or gender identity, and every treatment entry point rejects them. |
| 5. Calibration sidecar consumption | **Partially implemented, fail-closed** | The typed point-zero bridge consumes the ten ONS age-by-sex rows. It does not treat all 75 heterogeneous target rows as weights, does not resample the age distribution, and is rejected before all treatment execution. |
| 6. Macro-to-micro transaction layer | **Blocked as unidentified** | No attested v1 source or later external cross-check supports the 2--5% payer premise or identifies a GPD/Pareto threshold, scale, and shape. Payer incidence and tail parameters must be typed as `UNQUANTIFIED` in a successor; only labelled sensitivity ranges would be defensible now. |
| 7. Canonical point-zero/holdout infrastructure | **Point-zero implemented; extraction remains open** | The new CLI is deterministic and binary fail-closed. Existing population-design partitions use canonical SHA-256 thresholding; Python's process-randomized built-in `hash()` must not be used for participant splits. A privacy-reviewed Open Play extractor and row/partition digests are still required before fitting. |

This is intentionally not represented as seven completed calibrations. The
runtime mismatch and unidentified transaction/gamer parameters trigger the
requested stop condition: structural diagnostics may be committed, but no new
exploratory campaign or empirically labelled behavioural fit may proceed.

## Plan recognition and lineage decision

No immutable analysis-plan content was changed by this checkpoint. The complete
chain remains the exploratory sidecar schema v1
(`exploratory-synthetic-analysis-plan-v1.json`) to the prospective amendment
schema v3 (`prospective-analysis-plan-amendment-v3.json`) to its scientific
parent schema v2 (`prospective-analysis-plan.json`). The current configuration
binds each exact identity. This evidence audit neither changes the primary
estimand nor supplies runtime-consumed calibrated inputs, so rewriting those
identities would create false lineage.

The verifier did require an infrastructure correction: the first portable
profile fingerprint normalized top-level file-lineage paths but retained
absolute source- and population-evidence `bundle_path` values. Plans produced
on Windows consequently failed verification on GitHub's Linux runners even
when every input byte was identical. New runtime lineage normalizes only the
declared repository-path fields. Historical raw and portable-v1 fingerprints
remain readable, and plan binding accepts only two checked-in, directional
legacy-to-canonical digest pairs reproduced for the illustrative and production
snapshots. No general digest aliasing is allowed, the plan JSON and hashes are
unchanged, and an unregistered pair still fails closed.

If the population mapping, behavioural initializer, money basis, target weights,
or estimand later changes, the existing files must not be overwritten. Create a
new successor plan and artifact namespace, preserve the v3 parent and current
exploratory sidecar as historical inputs, and update the policy-config constants,
CLI validator, plan builder, expected digests, and a new versioned exploratory-
plan sidecar and configuration together.
The successor must describe the changed inputs and the continuing blockers.
Local content addressing is not external registration: registration status must
remain `UNREGISTERED` unless an external registry actually supplies a stable
record.

## Fail-closed campaign recommendation

**Recommendation: do not launch a new exploratory campaign now.** The evidence
bundle is scientifically useful, but the campaign would still execute the
illustrative welfare and choice system while appearing to carry calibrated UK
weights. That mismatch would be more misleading than retaining the current
explicitly synthetic results.

A successor exploratory campaign becomes appropriate only after these universal
gates pass:

1. the exact target bundle, source hashes, extraction coordinates,
   transformations, licences, and an archived extraction recipe validate
   reproducibly;
2. the declared UK 18--64 population mapping is consumed by runtime state, with
   sex either represented or explicitly outside the estimand;
3. every fitted target is mapped to a semantically compatible runtime quantity,
   validation partitions and numerical thresholds are fixed before fitting, and
   the chosen holdouts pass those checks;
4. outputs depending on unplanned spending, adult high risk,
   `decision.temperature`, composite weights, or real-money conversion remain
   disabled, sensitivity-only, or explicitly labelled
   `UNQUANTIFIED`/`NORMATIVE`;
5. a successor analysis plan and new versioned exploratory-plan sidecar bind the
   changed runtime inputs, estimand, uncertainty design, fixed seeds,
   code/environment identity, and fail-closed readiness state; and
6. preflight and a separately authorized bounded smoke run pass without data,
   arithmetic, plausibility, or lineage errors.

Additional gates apply only when the associated evidence family enters the
future estimand or fit. FRS requires a documented gross-household-to-personal-
disposable-income bridge. Open Play requires a source-compatible measurement
operator and participant-clustered uncertainty. Its conflicting UK-linked diary
rows may remain explicitly excluded rather than being forcibly resolved. ONS
September 2023 and Ofcom must remain untouched holdouts only if their target
families are selected for validation.

Until then, the defensible result is a partial calibration evidence package and
a recorded **no-go** decision for a new campaign, not a new set of experimental
claims.

## Reference cross-check record

Links and mutable metadata in this section were last checked on 1 September
2026. Stable DOIs are used where available; publisher pages are used for
official-statistics release identity. Inline methodological references above
retain their DOI or publisher link.

- Office for National Statistics. *Population estimates for the UK, England,
  Wales, Scotland and Northern Ireland: mid-2024 edition*. Released 26 September
  2025. [Dataset](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland).
- Office for National Statistics. *Time use in the UK* workbooks, March 2024
  and 23 September--1 October 2023; plus ad hoc dataset 2097, released 24 May
  2024. [Dataset series](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/datasets/timeuseintheuk)
  and [weekday/weekend release](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/adhocs/2097timeuseintheukbyweekdayandweekenddaymarch2024).
- Department for Work and Pensions. *Family Resources Survey, financial year
  2023 to 2024*. [Release](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2023-to-2024).
- Office for National Statistics. *Family spending workbook 1: detailed
  expenditure and trends*, corrected FYE 2024 workbook, released/corrected 11
  June 2026. [Dataset](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/datasets/familyspendingworkbook1detailedexpenditureandtrends).
- Office for National Statistics. *Earnings and hours worked, UK region by age
  group*, 2024 revised edition on the release updated 23 October 2025.
  [Dataset](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/earningsandhoursworkedukregionbyagegroup).
- Ofcom. *Online Nation 2024*, 28 November 2024. Page-67 gaming means are
  attributed to Ampere Games Consumer and have no published uncertainty in the
  report. [Report](https://www.ofcom.org.uk/siteassets/resources/documents/research-and-data/online-research/online-nation/2024/online-nation-2024-report.pdf?v=386238)
  and [Ofcom re-use terms](https://www.ofcom.org.uk/about-ofcom/our-website/copyright).
- Ofcom. *Online Nation 2020*. The microtransaction cross-check uses the
  overlapping Q14/Q16 ever-purchase measures among UK game-playing adults
  aged 18+, base 1,374; it is not an active-payer estimate.
  [Report](https://www.ofcom.org.uk/siteassets/resources/documents/research-and-data/online-research/online-nation/2020/online-nation-2020-report.pdf?v=324898).
- Department for Digital, Culture, Media & Sport. *Government response to the
  call for evidence on loot boxes in video games*, 17 July 2022. The cited 7%
  and 45% estimates come from an unpublished industry-commissioned Ipsos MORI
  report; the separate call-for-evidence player survey was self-selected and
  non-representative. [Response](https://www.gov.uk/government/calls-for-evidence/loot-boxes-in-video-games-call-for-evidence/outcome/government-response-to-the-call-for-evidence-on-loot-boxes-in-video-games).
- InGAME. *Loot boxes in video games: rapid evidence assessment*, published by
  DCMS on 17 July 2022. [Publication page](https://www.gov.uk/government/calls-for-evidence/loot-boxes-in-video-games-call-for-evidence).
- Ukie. *Time to press start on growth*, consumer-market valuation released in
  May 2025 for the 2024 market. It reports aggregate categories, not
  participant-level payer incidence. [Release](https://ukie.org.uk/publications/time-to-press-start-on-growth).
- Ballou, Vuorre, Földes, Hakman, and Magnusson. *digital-wellbeing/open-play:
  Open Play v1.2.5*, 17 August 2026. Version DOI
  [10.5281/zenodo.21986572](https://doi.org/10.5281/zenodo.21986572) and
  [release repository](https://github.com/digital-wellbeing/open-play/releases/tag/v1.2.5).
- European Central Bank. *EXR.A.GBP.EUR.SP00.A*, annual 2024 observation.
  [ECB Data Portal](https://data.ecb.europa.eu/data/datasets/EXR/EXR.A.GBP.EUR.SP00.A).
- Prolific. *Pricing*, *What are representative samples*, and *Using
  representative samples--FAQs*. [Pricing](https://researcher-help.prolific.com/en/articles/445239-what-is-your-pricing),
  [sample construction](https://researcher-help.prolific.com/en/articles/445161-what-are-representative-samples-on-prolific),
  and [dynamic sample limits](https://researcher-help.prolific.com/en/articles/445162-using-representative-samples-on-prolific-faqs).
- Gambling Commission. *Gambling Survey for Great Britain 2024 questionnaire*;
  NHS England. *Adult Psychiatric Morbidity Survey phase-one questionnaire*.
  [GSGB](https://www.gamblingcommission.gov.uk/report/gambling-survey-for-great-britain-annual-report-2024-official-statistics/gsgb-annual-report-2024-appendix-a-online-questionnaire)
  and [APMS](https://digital.nhs.uk/binaries/content/assets/website-assets/publications/apms-2023-4/appendix-c---phase-one-questionnaire.pdf).
- SteamDB and Skinport. Current service documentation was checked only to assess
  the proposed catalogue fallback; neither supplies participant-level exposure
  or rejected choices. [SteamDB FAQ](https://steamdb.info/faq/) and
  [Skinport sales-history API](https://docs.skinport.com/sales/history).
