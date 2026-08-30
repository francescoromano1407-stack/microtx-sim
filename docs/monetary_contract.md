# Monetary production contract

## Primary basis

The sole primary monetary basis is `ecb-eur-fx-2024-v1`:

- target currency: EUR;
- target unit: euro cent (`10^-2` EUR);
- reference period: calendar 2024 (`2024-01-01` through `2024-12-31`);
- method: observed market FX, not PPP;
- timing: the ECB-published annual-average reference-rate observation;
- ECB quote: source-currency units per EUR;
- runtime quote: target minor units per source minor unit;
- missing-date rule: use the official annual observation without local daily,
  weekend, holiday, or missing-date imputation;
- rounding: signed nearest target minor unit, half ties away from zero, exactly
  once after scenario contrast, population weighting, cross-jurisdiction
  aggregation, and equal-seed primary aggregation.

No PPP value participates in this basis. A future PPP estimand would require a
different named bundle, comparison group, source contract, and output basis.

## Official observations and transformations

The source institution is the European Central Bank. The official title for
each non-identity series is its ECB reference exchange-rate title. The source
API is the [ECB Data Portal EXR endpoint](https://data-api.ecb.europa.eu/service/data/EXR/A.GBP+JPY+KRW.EUR.SP00.A?startPeriod=2024&endPeriod=2024&format=csvdata),
and the stable series identifiers below are retained as permanent identifiers.
The data were retrieved on 2026-08-30. The endpoint does not provide an
observation-specific publication date, so the bundle records
`NOT_PROVIDED_BY_ENDPOINT` instead of inventing one.

| Jurisdiction | Local currency | Series / identity | Official 2024 observation | Exact target-minor/source-minor factor |
| --- | --- | --- | ---: | ---: |
| UK | GBP (pence, exponent 2) | `EXR.A.GBP.EUR.SP00.A` | `0.8466166015625` GBP/EUR | `5120000/4334677` EUR-cent/GBP-penny |
| South Korea | KRW (won, exponent 0) | `EXR.A.KRW.EUR.SP00.A` | `1475.4041015624998` KRW/EUR | `500000000000000/7377020507812499` EUR-cent/KRW |
| Japan | JPY (yen, exponent 0) | `EXR.A.JPY.EUR.SP00.A` | `163.8519140625` JPY/EUR | `2560000/4194609` EUR-cent/JPY |
| Belgium | EUR (cent, exponent 2) | explicit EUR/EUR identity | `1` EUR/EUR | `1/1` EUR-cent/EUR-cent |

For a source observation `q` quoted as source-currency units per EUR, source
minor exponent `s`, and target exponent 2, the exact execution factor is
`(10^2 / 10^s) / q`. Decimal observation strings are parsed exactly and reduced
to the rational shown above; no binary floating-point or rounded display rate
enters execution. Belgium is not sent through a foreign-exchange lookup because
its local and target currencies are both EUR; its declared identity record is
still required and attested.

The bundle checks in the unmodified 967-byte ECB API response as well as the
readable conversion table. It preserves the institution, title, series
identifier, source URL, observation period, unit, quote direction,
transformation, and retrieval metadata for every row. The bundle records both
exact byte lengths and SHA-256 values, binds the source-catalogue SHA-256, and
provides a deterministic official-row selection and decimal-to-rational
extraction recipe.

## Meaning of `simulation_cents`

`simulation_cents` are integer internal purchasing-power/model units used by
the player, game, firm, regulator, and ledger systems. They are not pence, won,
yen, euro cents, official FX values, official PPP values, or observed spending.
The loader normalizes each jurisdiction's declared local nominal monthly anchor
to `180000` internal units. That produces an exact internal-to-local scale, but
the bridge is `ILLUSTRATIVE`, not empirically calibrated. In particular, the
Japanese local anchor is itself illustrative, and the Korean household-table
anchor is not an individual-income median.

Consequently, the production monetary artifact is labelled an **EUR-equivalent
model amount in 2024 EUR cents**. It must not be described as observed national
spending or as an empirically calibrated real-money effect. The official FX
rates are valid source inputs; the model-unit bridge remains the empirical
limitation.

## Runtime and aggregation order

For every retained observation and jurisdiction, runtime performs the following
exact-rational operations:

1. retain the raw integer `simulation_cents` diagnostic;
2. apply the jurisdiction's declared internal-to-local minor-unit scale;
3. apply that jurisdiction's checked-in local-minor-to-EUR-cent FX factor;
4. form the reference/comparison contrast in the declared direction;
5. apply the checked-in population design weight, using the same weight for
   every scenario;
6. sum the converted weighted values across all jurisdictions;
7. take the declared equal-seed primary aggregate;
8. round once at serialization of the final estimate.

All intermediate monetary values are exact `Fraction` values. Direct
cross-jurisdiction summation of raw `simulation_cents`, scenario-specific
weights, implicit currencies, missing conversions, and intermediate rounding
are rejected. Omitting the monetary bundle exposes no runtime conversions, so a
money-valued prospective estimand fails closed rather than falling back to raw
internal units.

## Outputs and lineage

The legacy root output tables remain diagnostic simulator-unit artifacts. A
money-valued prospective execution additionally writes two separate files under
`prospective_analysis/`:

- `production_monetary_estimates.csv`, containing the declared estimand,
  scenario contrast, target/reference basis, population and weighting rule,
  final rounded estimate and its exact rational value, uncertainty fields,
  rounding rule, bundle/source identities, and plan/run bindings;
- `production_monetary_metadata.json`, containing the complete conversion
  bases, source hashes and byte lengths, exact applied weights per seed,
  conversion/aggregation/rounding order, model-amount interpretation, execution
  lineage, and campaign blockers.

The campaign manifest embeds the same monetary-lineage payload, plus the Git
commit and the population design, population hash, and run bindings. Output
validation rejects raw `simulation_cents` as a final monetary estimand and
keeps the production artifacts separate from diagnostic outputs.

## Authentication and readiness

The bundle signature status is `MISSING`, with algorithm `NONE`. SHA-256 is an
integrity fingerprint, not an authentic publisher or trusted-authority
signature. No signature is fabricated. Campaign readiness therefore remains
false. Independent blockers also include the illustrative internal-to-local
scale, illustrative population evidence/design and lack of a genuine holdout,
unregistered prospective plan/run bindings, and unresolved substantive model
calibration. Arithmetic and lineage completeness do not establish empirical
validity.
