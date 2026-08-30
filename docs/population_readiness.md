# Target population, evidence, and campaign readiness

## Status and scope

This document records the selected population target and the evidence work still
required before it can support a scientific campaign. It does not promote any
checked-in input to `CALIBRATED`. At the documentation review date, 2026-08-30,
the checked-in population-evidence bundle contains 1,728 complete modeled joint
rows and the design contains 864 calibration cells, but both remain
`ILLUSTRATIVE`, unsigned, and campaign-ineligible. The registered runtime mapping
is likewise an explicit assumed runtime model, not evidence of representativeness.

The repository source catalogue records a catalogue-wide retrieval date of
2026-08-24. The official pages and documents below were independently rechecked
on 2026-08-30 for this documentation review. Those dates do not replace the
artifact-level retrieval date, byte length, SHA-256 digest, publication date,
revision, and exact extraction location required by a future population bundle.

## Selected analytic population and unit

The selected analytic unit is **one modeled person** who is a usual resident of
the United Kingdom (`UK`), South Korea (`KR`), Japan (`JP`), or Belgium (`BE`)
and is aged 10 through 69 inclusive at the pre-treatment reference date. The
10--69 frame is a design decision that aligns the common age support with the
verified South Korean game-user survey and available Japanese demographic
frames. It is not an observed participation estimate and does not assert that
people outside this range are irrelevant.

The target includes gamers and non-gamers, and payers and non-payers. Restricting
the target to observed gamers or payers after treatment would change the
estimand and permit post-treatment selection. Eligibility, age, jurisdiction,
cell membership, and weights must therefore be fixed before treatment.

### Equal-country standardization and pooled-person estimates differ

The project's cross-jurisdiction primary population target is an
**equal-country standardized mean or proportion**. If `theta_j` is the
within-country estimate for jurisdiction `j`, the cross-jurisdiction target is

```text
theta_equal_country = (theta_UK + theta_KR + theta_JP + theta_BE) / 4
```

The one-quarter jurisdiction factors are researcher-defined standardization
weights. They are not empirical national population shares. Within each
jurisdiction, the declared joint-cell masses must sum exactly to one. For a
sampled cell `c` with target conditional mass `m_jc` and modeled count `n_jc`,
the cross-jurisdiction analysis weight per modeled person is
`(1/4) * m_jc / n_jc`. These exact rational weights must sum to one over the
four-country analytic cohort. The runtime must not silently normalize,
redistribute, or replace them.

The checked-in design instantiates this target with exactly 10,000
design-person equivalents per jurisdiction (40,000 total expansion units). Its
six age bands are 10--17, 18--24, 25--34, 35--44, 45--54, and 55--69; it crosses
three local-currency income bands, three household types, two gaming states, and
two payer-history states. That is 216 cells per jurisdiction and 864 cells per
role. The evidence CSV contains both an illustrative calibration role and a
separately perturbed illustrative validation role, for 1,728 rows. The latter is
a deterministic sensitivity declaration, not an independent empirical holdout.

### Exact illustrative joint-generation recipe

The following assumptions generate the checked cells. They are repeated in
content-addressed binding metadata in `population_bundle.toml`; this table is a
human-readable rendering, not empirical calibration.

| Jurisdiction | Age shares for 10--17 / 18--24 / 25--34 / 35--44 / 45--54 / 55--69 (%) | Current-gaming shares in the same order (%) | Income shift `s` |
| --- | --- | --- | --- |
| BE | 11 / 12 / 17 / 18 / 18 / 24 | 84 / 72 / 59 / 48 / 38 / 27 | 0 |
| JP | 8 / 10 / 16 / 19 / 22 / 25 | 85 / 73 / 63 / 52 / 41 / 29 | 2 |
| KR | 9 / 12 / 18 / 20 / 22 / 19 | 82 / 78 / 71 / 62 / 50 / 35 | 4 |
| UK | 12 / 12 / 17 / 18 / 18 / 23 | 90 / 79 / 67 / 54 / 43 / 31 | 0 |

For canonical age ordinal `a = 0..5`, household percentages are shown in
one-person / multi-person-without-minor / with-minor order:

```text
a=0:   0 /  0 / 100
a=1:  45 / 45 /  10
a=2:  35 / 50 /  15
a=3:  25 / 55 /  20
a=4:  27 / 65 /   8
a=5:  38 / 60 /   2
```

For canonical household ordinal `h = 0..2`, conditional income-band
percentages are

```text
low(a,h,j)    = 42 - 4a - 5h - s_j
high(a,h,j)   = 18 + 3a + 5h + s_j
middle(a,h,j) = 100 - low(a,h,j) - high(a,h,j).
```

Calibration gaming percentages use the table above. Validation subtracts four
percentage points with a floor of one. For canonical income ordinal `i = 0..2`,
the `EVER_PAYER` percentage is

```text
gamer:     24 + 7i + 2*max(0, 3-a)
non-gamer:  3 + 2i +   max(0, 2-a).
```

Validation subtracts three points with a floor of one for gamers and adds one
with a cap of 99 for non-gamers. The exact factorization is

```text
P(age)
* P(household | age)
* P(income | age, household, jurisdiction)
* P(gaming | age, jurisdiction, role)
* P(payer history | age, income, gaming, role).
```

This factorization is the complete declared conditional model; it does not
assert that the variables are empirically independent. Each jurisdiction-role
distribution is multiplied by 10,000, floored, then the remaining units are
assigned by descending exact rational remainder with source ordinal as the
last tie-break. Thus zero cells and all rounding are deterministic. Evidence
household source order is canonical ordinals `2 / 0 / 1`; the design maps each
semantic key and evidence cell ID into canonical order `0 / 1 / 2`, so source
order is never treated as canonical order by position.

Runtime households use capacities one, two, and three respectively. Every
`household.with-minor` household is deterministically seeded with one
pre-treatment minor before the remaining members are assigned. The declared
shares above make that constraint feasible in every jurisdiction-income group.
The balance validator rejects a with-minor household without a minor and also
rejects a minor in either no-minor household category.

A **pooled-person mean** answers a different question. It weights country means
by official target-population counts, `N_j / sum(N_j)`, and will generally differ
from the equal-country mean. A population total is different again and uses
expansion weights `N_j * m_jc / n_jc`. Expansion weights reconstruct the
jurisdiction target counts; they must not be substituted for equal-country
analysis weights or presented as a common-currency welfare total without the
separate monetary-comparability contract. Every output must identify which of
these estimands it implements.

## Required joint-cell design

Every declared population cell must be a joint, pre-treatment combination of:

- jurisdiction;
- non-overlapping age interval and its deterministic minor/adult status under a
  declared jurisdiction-specific threshold;
- source household-income band, including its exact bounds, currency, period,
  income concept, and equivalisation rule;
- source household type and definition;
- gaming state, exactly `GAMER` or `NON_GAMER`;
- payer-history state, exactly `EVER_PAYER` or `NEVER_PAYER` under a declared
  pre-treatment lookback and zero-spender rule;
- the runtime mapping to personal monthly disposable-income interval and
  modeled players per household required by the projected-population adapter.

Cell identity, source order, canonical order, exact target mass, target count,
analysis weight, expansion weight, evidence source, and transformation or
assumption must remain addressable separately. A zero cell must be explicit;
an omitted cell is not evidence of zero mass. Age, income, household, gaming,
and payer states must not be sampled independently unless the exact
factorization assumption is declared, justified, and tested for sensitivity.

## Evidence classifications

Population documentation and artifacts must preserve three different claims:

| Classification | Meaning for population design |
| --- | --- |
| Observed | A value or microdata field directly reported by the issuing institution for its stated population, period, unit, and denominator. |
| Derived | A reproducible transformation of observed inputs, such as regrouping ages, applying survey weights, forming a conditional joint cell, currency-period conversion, or fitting distribution parameters. The recipe and uncertainty remain part of the evidence. |
| Assumed | A researcher choice not identified by the source, including equal-country standardization, independence or data-fusion assumptions, income-distribution shape, truncation, or source-to-runtime transport. |

`ANCHORED` means that a named source is linked; it does not make a reported or
derived value reproducibly calibrated. Company accounts, legal reports, and
case reports cannot be relabeled as demographic observations.

## Official demographic and gaming sources

The source locations below are candidates for immutable ingestion. No digest is
reported here because no raw source artifact was added to the population bundle.

### United Kingdom

1. **Population by single year of age.** Office for National Statistics,
   [Population estimates for the UK, England, Wales, Scotland and Northern
   Ireland: mid-2024](https://www.ons.gov.uk/releases/populationestimatesfortheukenglandwalesscotlandandnorthernirelandmid2024),
   released 2025-09-26. Use the exact `Mid-2024 edition` workbook from the
   [dataset page](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland),
   table/tab `MYE2 -- Persons`, usual-resident persons by sex and single year of
   age. Regrouping ages 10--69 is derived; the published counts are observed.
2. **Joint household and income base.** Department for Work and Pensions,
   [Family Resources Survey, financial year 2023 to
   2024](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2023-to-2024),
   published 2025-03-27 and updated 2026-01-15. The research-data citation is
   UK Data Service Study Number 9367, second edition, May 2026,
   [DOI 10.5255/UKDA-SN-9367-2](https://doi.org/10.5255/UKDA-SN-9367-2).
   Household, benefit-unit, person, income, and survey-weight records can support
   a person-weighted age x household type x household-income base. Public-use
   anonymisation and top-coding can prevent exact reproduction of published
   estimates and must be recorded.
3. **Income location anchor.** Office for National Statistics,
   [Household disposable income and inequality, financial year ending
   2024](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/bulletins/householddisposableincomeandinequality/latest),
   released 2025-05-02, Figure 1, equivalised disposable household income of
   individuals. The prose headline is rounded. The configured exact value of
   GBP 36,663 must be tied to an archived downloadable table cell before use;
   a mutable `latest` page is not exact-byte evidence.
4. **Child gaming and recent spending.** Ofcom, [Children's online spending and
   potential financial harm: Quantitative
   research](https://www.ofcom.org.uk/siteassets/resources/documents/online-safety/research-statistics-and-data/online-services-research/childrens-online-spending-and-potential-financial-harm-quantitative-research.pdf?v=400633),
   June 2025. Page 4 describes 2,205 children aged 8--17 and their parents and a
   March 2025 survey; page 6 reports 97% gaming reach and 53% recent spending
   among children who played; page 10 describes weighting and 95% significance;
   page 30 reports the 53% estimate with unweighted base 2,105 and weighted base
   2,084; page 31 gives age/gender breaks; page 54 gives the sample profile.
   These are observed survey estimates. The payer denominator is recent gamers
   and the period is the previous month, so 53% is not an `EVER_PAYER` mass.
   The report's 8--17 scope also does not exactly identify the selected target's
   10--17 minor segment; exact trimming requires suitable microdata or an
   explicitly modeled and validated age-harmonization step.

The Family Resources Survey and Ofcom samples do not constitute one observed
joint distribution. Multiplying their marginals would be a derived data-fusion
model and requires a declared recipe, assumptions, uncertainty, sensitivity,
and independent validation.

### South Korea

1. **Population by age.** Statistics Korea/KOSIS table `DT_204C110_10`,
   [Korean population by
   age](https://kosis.kr/statisticsList/mass/mass_list_e.jsp?list_id=204_20401&org_id=204&process=statHtml&tbl_id=DT_204C110_10&vw_cd=MT_ZTITLE),
   annual 2024 CSV. The exact table query, downloaded bytes, reference date,
   status of foreign residents, and age regrouping must be retained.
2. **Population definitions.** Statistics Korea, [2024 Register-based Population
   and Housing Census](https://kostat.go.kr/boardDownload.es?bid=11747&list_no=439064&seq=1),
   released 2025-07-29, population observed at 2024-11-01. Use this for the
   universe and coverage definitions and KOSIS for exact age cells.
3. **Joint household and income base.** Statistics Korea's Household Income and
   Expenditure Survey microdata are distributed through
   [MDIS](https://mdis.kostat.go.kr/). The official survey handbook, section
   3-2, identifies nationwide general households and member relationship, sex,
   age, household income, and expenditure. A versioned 2024 extract, codebook,
   survey weights, exclusions, and disclosure rules remain to be obtained.
4. **Income aggregates.** Statistics Korea, [Household Income and Expenditure
   Trends in the Fourth Quarter of
   2024](https://mods.go.kr/boardDownload.es?bid=11736&list_no=436048&seq=2),
   Table 5, original income unit `1,000 won/month` and propensities in percent.
   Observed disposable-income quintile means are 1,037; 2,467; 3,515; 5,104;
   and 8,912 thousand KRW. Observed average propensities to consume are 133.6%,
   79.9%, 78.5%, 68.5%, and 55.0%. Multiplication by 1,000 and division by 100
   are derived transformations. These values are not quintile boundaries,
   person-equivalised medians, or within-band lognormal parameters.
5. **Gaming and mobile spending.** Korea Creative Content Agency, [2024 Game
   User Survey, report code
   KOCCA24-26](https://welcon.kocca.kr/ko/info/report/1954596), registered
   2025-01-03, and the [official English
   summary](https://welcon.kocca.kr/mobile/en/support/resources/377). The summary
   covers ages 10--69 and a base of approximately 10,000. Page 1 reports an
   observed overall game-use rate of 59.9%. Page 2 reports, for 7,402 mobile
   gamers, monthly paid-download spending of KRW 2,874, in-app purchases of KRW
   17,238, subscriptions of KRW 1,762, and total spending of KRW 21,875. The
   original Korean questionnaire must establish whether these means include
   zero spenders. They do not identify payer-history x income x household cells.

### Japan

1. **Population by age.** Statistics Bureau of Japan, [Population Estimates
   2024](https://www.stat.go.jp/data/jinsui/2024np/index.html), published
   2025-04-14. Use e-Stat [Table
   1](https://www.e-stat.go.jp/en/stat-search/files?cycle=7&layout=datalist&month=0&page=1&result_back=1&stat_infid=000040268910&tclass1=000001011679&tclass2val=0&toukei=00200524&tstat=000000090001&year=20240),
   `Population by Age (Single Years), Sex and Sex ratio -- Total population,
   Japanese population, October 1, 2024`, published 2025-04-14 at 14:00. The
   design must choose and preserve `total` versus `Japanese` population.
2. **Joint household and income base.** Statistics Bureau of Japan/e-Stat,
   [2024 National Survey of Family Income, Consumption and Wealth, table
   7-202-1-1](https://www.e-stat.go.jp/en/dbview?sid=0004056234), published
   2026-08-28. The table distributes household members by sex, age group, family
   composition, and equivalised yearly disposable-income group under the new
   OECD scale. Regrouping, use of survey weights, and mapping to the common age
   and household domains are derived steps that require an exact downloaded
   extract and metadata.
3. **Youth gaming.** Children and Families Agency, [FY2024 Survey on Internet
   Usage Environment among Young
   People](https://www.cfa.go.jp/assets/contents/node/basic_page/field_ref_resources/9a55b57d-cd9d-4cf6-8ed4-3da8efa12d63/0a26134c/20250328_policies_youth-kankyou_internet_research_results-etc_16.pdf),
   file dated 2025-03-28. Page 12, summary section 7, reports 84.9% playing games
   among young people who use the internet. This is conditional on internet use
   and is not automatically all-person or mobile-game prevalence. The exact
   chart CSV should be archived from the [official survey
   hub](https://www.cfa.go.jp/policies/youth-kankyou/internet_research/results-etc/).
4. **Historical smartphone-game paying evidence.** Cabinet Office Consumer
   Commission, [Opinion on consumer issues surrounding smartphone
   games](https://www.cao.go.jp/consumer/iinkaikouhyou/2016/0920_iken.html),
   published 2016 using 2015/February 2016 evidence, Figures 4--7. It reports
   approximately 70% never-payers among smartphone-game users, more than 60% of
   payers below JPY 1,500 in average monthly spending over the previous six
   months, and fewer than 2% of payers ever spending at least JPY 100,000 on one
   title. These conditioning sets and periods must remain intact. The ever-title
   tail is not a monthly hazard, and the old source cannot establish current
   joint population mass.

The configured JPY 300,000 monthly income location and log-sigma 0.58 are
illustrative, not findings from these sources.

### Belgium

1. **Population by age.** Statbel, [Population by place of residence,
   nationality, marital status, age and sex --
   2025](https://statbel.fgov.be/en/open-data/population-place-residence-nationality-marital-status-age-and-sex-16),
   downloadable TXT ZIP/XLSX. The dynamic page does not replace an archived
   edition with its exact publication timestamp, reference date, universe, and
   bytes.
2. **Household structure.** Statbel, [Census 2021 population by sex, age, and
   position in the
   household](https://statbel.fgov.be/nl/open-data/census-2021-bevolking-totaal-belgie-naar-geslacht-leeftijd-h-en-positie-het-huishouden-h),
   period 2021, downloadable XLSX/TXT. Its older census period must not be
   silently presented as contemporaneous with 2024 income or 2025 population.
3. **Income.** Statbel, [EU-SILC poverty and living-conditions
   FAQ](https://statbel.fgov.be/en/themes/households/poverty-and-living-conditions/faq),
   SILC 2025 with 2024 income. It reports observed median equivalised disposable
   income of EUR 31,299 and defines the modified OECD scale as 1.0 for the first
   adult, 0.5 for each additional person aged at least 14, and 0.3 for each
   child under 14. The dynamic FAQ lacks immutable table-level lineage.
   Person/household microdata must be requested through Statbel's [research
   microdata service](https://statbel.fgov.be/en/about-statbel/what-we-do/microdata-research).
4. **Gaming evidence gap.** No national official source was verified for Belgian
   minor gaming prevalence or payer history. [Apenstaartjaren
   2024](https://www.mediawijs.be/nl/onderzoek/apenstaartjaren) covers Flemish
   children and young people aged 6--18 and may inform sensitivity analysis,
   but it is not a Belgian national target frame. Statbel's household ICT survey
   samples ages 16--74 and cannot fill the younger-child or payer-history gap.
   The Belgian Gaming Commission loot-box report is legal classification
   evidence, not prevalence evidence.

The missing Belgian national gaming/payer evidence is an explicit campaign
blocker. It must not be replaced by an undocumented prevalence value.

## Income synthesis requirements

The legacy jurisdiction-profile income-shape parameters remain illustrative:
UK 0.62, KR 0.28, JP 0.58, and BE 0.60, with no profile-level truncation. The
separate checked-in runtime mapping uses schema v2 and declares a bounded,
censored log-normal model for every income-band × household-type mapping. The
median, reduced rational log-sigma, local currency/month, inclusive bounds,
source assumption ID, transformation, censoring, and half-to-even rounding rule
are all content-addressed. Household type affects the joint-cell mass and
modeled household size, not the within-band income parameters. There is
deliberately **no pointwise transform** from a source household's annual income
to a simulated person's monthly income: the source annual band is only a joint
stratum label, while each runtime median, dispersion, and bound is a standalone
monthly personal-income assumption. No division by 12, equivalisation, or
within-household allocation identity is claimed. This absence of an empirical
transport equation is itself recorded in every schema-v2 mapping entry.

| Jurisdiction | Band | Bounds, median (runtime minor units/month) | Log-sigma | Status |
| --- | --- | --- | --- | --- |
| BE | low | 50,000--199,999; 120,000 | 11/20 | assumed |
| BE | middle | 200,000--399,999; 285,000 | 9/20 | assumed |
| BE | high | 400,000--999,999; 550,000 | 11/20 | assumed |
| JP | low | 50,000--249,999; 160,000 | 29/50 | assumed |
| JP | middle | 250,000--499,999; 340,000 | 12/25 | assumed |
| JP | high | 500,000--1,499,999; 650,000 | 29/50 | assumed |
| KR | low | 500,000--2,499,999; 1,500,000 | 9/20 | assumed |
| KR | middle | 2,500,000--4,999,999; 3,500,000 | 7/20 | assumed |
| KR | high | 5,000,000--14,999,999; 7,000,000 | 9/20 | assumed |
| UK | low | 50,000--199,999; 130,000 | 11/20 | assumed |
| UK | middle | 200,000--399,999; 280,000 | 9/20 | assumed |
| UK | high | 400,000--1,199,999; 550,000 | 11/20 | assumed |

These parameters are not identified by the official sources. Values outside the
declared interval are censored to its nearest inclusive bound, then rounded to
integer minor units. There is no rejection sampling, redistribution, or runtime
normalization.

If a cell uses `X ~ LogNormal(mu, sigma^2)`, its artifact must state:

- whether `X` is source household disposable income, equivalised household
  disposable income attributed to a person, or runtime personal monthly
  disposable income;
- the source currency, price/reference period, periodicity, equivalisation, and
  conversion into the runtime concept;
- the observed location and dispersion targets and their exact source cells;
- the fitting equation, such as `mu = log(median)` and a declared quantile-based
  derivation of `sigma`;
- any lower/upper bounds and whether the distribution is truncated, censored,
  top-coded, or winsorized;
- which results are observed, which parameters are derived, and which choices
  remain assumptions;
- validation against source quantiles and sensitivity to dispersion and bounds.

A median determines `mu` under a lognormal assumption but does not determine
`sigma`. Korean quintile means are neither cut points nor medians and cannot be
inserted as lognormal parameters without another identified model. The mapping
from household or equivalised income to a modeled person's monthly disposable
resources is also a transport assumption, not an observed identity.

### Minor gaming and household income

No verified source supports a causal claim that gaming changes a minor's
household income. If gaming participation is synthesized conditionally, it must
be described as selection, for example
`P(GAMER | jurisdiction, age, household income, household type)`, and must record
the observed prevalence, its population and denominator, the exact adjustment
or data-fusion equation, necessary assumptions, and sensitivity results.
Combining an age-specific gaming marginal with a separate household survey is
derived/model-based even when both inputs are official. It must not be narrated
as a gaming-caused income difference.

The checked-in mapping therefore declares `minor_gaming_adjustment = "NONE"`
and `minor_gaming_adjustment_reason = "INSUFFICIENT_VERIFIED_EVIDENCE"`. The
joint construction makes gaming conditional on jurisdiction and age and makes
payer history conditional on gaming, age, and income band; it does not change
the income model after observing minor gaming and makes no causal claim.

## Company accounts are plausibility evidence only

Company filings can bound monetization-channel or cost assumptions; they cannot
create demographic cells or validate simulated harms.

- **Playtika Holding Corp., FY2024.** [Form
  10-K](https://www.sec.gov/Archives/edgar/data/1828016/000182801625000011/playtika-20241231.htm),
  SEC accession `0001828016-25-000011`, filed 2025-02-27. Page 72 reports 8.1
  million average daily active users, 312 thousand average daily paying users,
  3.8% average daily payer conversion, and USD 0.86 ARPDAU. Page 73 reports USD
  1,855.1 million third-party-platform revenue and USD 694.2 million
  direct-to-consumer revenue out of USD 2,549.3 million total; the derived DTC
  share is 27.23%. Pages 70--71 report typical payment-processing costs of 3--4%
  for DTC purchases versus 30% for third-party platforms.
- **DoubleDown Interactive Co., Ltd., FY2024.** [Form
  20-F](https://www.sec.gov/Archives/edgar/data/1799567/000162828025018568/ddi-20241231.htm),
  SEC accession `0001628280-25-018568`, filed 2025-04-21. Pages 53--54 report
  6.7% social-casino payer conversion, USD 1.30 ARPDAU, USD 283 average monthly
  revenue per payer, and 65.1% mobile penetration. Page 54 warns that a person
  using multiple games or devices may be counted more than once. The cost table
  reports platform cost equal to 26.1% of revenue.

These are observed company-wide or portfolio metrics with company-specific
account/device/game denominators. Playtika can inform broad channel and payer
plausibility. DoubleDown's selected social-casino audience is better treated as
a high-spend sensitivity bound. Neither filing identifies UK/KR/JP/BE person
weights, minor prevalence, population payer conversion, household income, or
harm. Any transformation into a simulator parameter must remain `DERIVED` or
`ASSUMED`, name the reporting period and unit, and disclose non-comparability.

## Campaign-readiness blockers

The population campaign gate must remain closed until all of the following are
resolved together:

- immutable official source files, exact table/cell queries, publication and
  retrieval dates, byte lengths, SHA-256 digests, licences, and reproducible
  extraction recipes;
- a common target-period decision and compatible usual-resident definitions;
- complete observed or explicitly modeled joint-cell support, including
  explicit zero cells and compatible gaming/payer definitions;
- national Belgian gaming and payer-history evidence, or a prospectively
  approved and sensitivity-tested model that is clearly not empirical mass;
- calibrated income dispersion, bounds, equivalisation, and a reviewed
  household-income-to-runtime-personal-income mapping;
- survey weights, sampling uncertainty, exclusions, top-coding/disclosure
  treatment, and non-overlapping calibration and genuine held-out source units;
- exact declared analysis and expansion weights with no runtime normalization;
- a verified projected-population adapter and runtime mapping covering every
  jurisdiction and joint cell;
- one content-addressed cohort assignment per seed, reused unchanged across all
  scenarios, plus pre-treatment balance evidence;
- explicitly weighted primary population outputs retaining selected/excluded
  counts, total weight, cell/weight identity, and all upstream lineage;
- independent monetary comparability, prospective analysis, model/build
  identity, external preregistration, and empirical outcome-validation gates.

Hash agreement proves which bytes and transformation were used. It does not
prove publisher authenticity, representativeness, validity of data fusion,
calibrated transport, a genuine holdout, or validity of simulated harm. Passing
population software checks is necessary but not sufficient for a scientific
campaign. The repository must continue to reject campaign execution while any
of these independent blockers remains unresolved.
