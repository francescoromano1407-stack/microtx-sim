# ECB EUR FX monetary input bundle, calendar 2024

This bundle declares EUR as the single target currency and calendar 2024 as
the reference period, matching the checked-in projected-population design.
The basis is observed market FX, not PPP. The ECB annual series identifiers
are `EXR.A.GBP.EUR.SP00.A`, `EXR.A.JPY.EUR.SP00.A`, and
`EXR.A.KRW.EUR.SP00.A`. ECB quotes are source-currency units per EUR; runtime
factors are reduced exact rationals in target minor units per source minor
unit. Belgium uses an explicit EUR/EUR identity factor.

The unmodified ECB endpoint response bytes are retained in
`artifacts/ecb_exr_annual_2024.csv`; the independently readable declared
conversion table is `artifacts/conversion_rates.csv`. Runtime extracts the
non-identity rational factors directly from the official response's exact
`OBS_VALUE` strings. The period is the official annual 2024 observation. No
missing date, weekend, or holiday is filled locally; the project uses the
ECB-published annual observation without imputation. The endpoint does not
expose an observation-specific publication date, and that unavailability is
recorded explicitly rather than guessed.

Each jurisdiction first uses its declared local nominal monthly anchor to map
raw internal `simulation_cents` onto an exact local-currency scale, and then
uses the declared FX factor to obtain an exact EUR-cent equivalent. These
unrounded rational values are population-weighted and aggregated before a
single signed half-away-from-zero rounding at the production output boundary.
The local anchor bridge remains illustrative and does not establish observed
real-world spending.

The bundle is unsigned. Its SHA-256 values are integrity fingerprints, not a
publisher signature. An authentic trusted signature remains unavailable, so
the campaign gate must remain closed.
