# UK adults 2024 partial calibration bundle

This directory contains compact, reviewable derivatives of the user-supplied
public-evidence cache. It does not contain participant-level data, workbooks,
PDFs, or archives, and it does not authorize a simulation campaign.

## Contents

- `calibration_bundle.json` binds the three machine-readable companion files,
  retains the published 101% FRS rounding total, lists scientific blockers, and
  declares `campaign_ready=false`.
- `targets.csv` separates calibration, held-out validation, and diagnostic
  records. Unidentified and normative concepts are explicit rows rather than
  defaults.
- `population_weights.csv` contains the ten ONS age-band-by-sex cells for UK
  usual residents aged 18--64 at mid-2024. The current simulator cannot consume
  the sex margin without a state-schema change.
- `source_manifest.json` records publisher metadata, source locations, byte
  lengths, SHA-256 digests, roles, licences, and known limitations.

The source cache is expected at
`data/public_calibration_sources_uk_adults_2024/` and is intentionally ignored by
Git. Retrieval timestamps are `null` because the supplied cache did not record
exact times; known date-only metadata (including the ECB retrieval date) is
retained in source versions or notes. `verified_at` records only the local
hash-verification date.

Validate the bundle and print non-executing diagnostics from the repository
root with:

```text
python tools/validate_uk_adults_2024_calibration.py
```

Passing this command means that the declarations and original local bytes are
internally consistent. It does not mean that the evidence is representative,
that targets are connected to runtime equations, or that a campaign is ready.
The verifier is designed for a trusted local cache that is not being mutated
concurrently; it is not a defence against an adversary racing filesystem swaps
during verification. Large sources are hashed in bounded chunks and are never
loaded wholesale into memory.
The deliberately failing campaign gate can be checked with:

```text
python tools/validate_uk_adults_2024_calibration.py --campaign-gate
```

The methodological decisions, cross-checks, and conditions for a future
successor campaign are documented in
`docs/uk_adults_2024_calibration.md`.
