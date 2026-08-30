# Usage

## Install

Python 3.11 or later is required. From the repository root:

```text
python -m pip install -e ".[dev]"
```

The only runtime dependency is NumPy. The development extra installs pytest,
although the suite also runs with the standard-library unittest runner.

## Supported workflows

The repository exposes two intentionally separate workflows:

- the synthetic policy prototype in `configs/policy_prototype.toml`, which runs
  the seven required counterfactual scenarios and exports reproducible research
  artifacts;
- the older market-model smoke configuration in `configs/smoke.toml`, which is
  retained as a short connectivity and information-boundary check.

Neither workflow is empirically calibrated. A successful command means that
the software and declared synthetic assumptions were processed consistently; it
does not validate a real-world effect.

## Validate the policy prototype

```text
python -m microtx_sim policy-validate configs/policy_prototype.toml
```

This command strictly parses the policy TOML, validates all ranges and exact
section keys, confirms that all seven named scenarios are present, and prints a
JSON scope summary. It does not create a cohort, advance the simulation, or
write artifacts. The expected mode is `synthetic_policy_prototype`, and
`empirical_validation_claimed` is always false.

### Opt into projected-population execution

A custom market or policy configuration may add the four projection keys
below. A self-contained illustrative profile may also bind the optional paired
`evidence_bundle_path` and `source_registry_path` locators.

```toml
[population]
mode = "projected_v1"
design_bundle_path = "../data/provenance/population_design.toml"
runtime_mapping_bundle_path = "../data/provenance/population_runtime_mapping.json"
adapter_id = "reviewed.population.adapter.v1"
# Optional, but required together when the run does not use registered defaults:
evidence_bundle_path = "../inputs/population-bundle.toml"
source_registry_path = "../inputs/population-sources.toml"
```

Paths are relative to the configuration file unless absolute. Validation then
loads and re-attests the design, its bound profile evidence, and the mapping
before any cohort is created. The mapping must explicitly translate every
static household-income/type key to runtime personal monthly disposable-income
and modeled-household semantics. The adapter uses the design plan's exact
counts and weights; it does not make a second allocation. The ordinary
prototype and smoke configs omit this section and use the legacy marginal
population initializer. The separate `configs/policy_prospective.toml` selects
checked-in illustrative/test population inputs and the concrete plan; use
`policy-validate` on that config for non-executing preflight. Its test
provenance is not campaign-ready.

## Run the seven-scenario batch

```text
python -m microtx_sim policy-batch configs/policy_prototype.toml
```

For each configured seed, the batch reuses the same initial synthetic cohort
across all seven scenarios. The default output directory is
`artifacts/policy_prototype/`. Override it without changing the configuration:

```text
python -m microtx_sim policy-batch configs/policy_prototype.toml --output artifacts/my_batch
```

The supplied configuration enables its one-at-a-time sensitivity grid. Use the
following only when a faster batch without that optional grid is explicitly
intended:

```text
python -m microtx_sim policy-batch configs/policy_prototype.toml --skip-sensitivity
```

Skipping sensitivity leaves the scenario batch intact but cannot be presented
as the complete reproducibility workflow.

## Run sensitivity analysis only

```text
python -m microtx_sim policy-sensitivity configs/policy_prototype.toml
```

This writes `sensitivity.csv` and `sensitivity_metadata.json` to the configured
output directory, or to the directory supplied with `--output`. It is useful
for focused monotonicity and stability checks; it does not write the full batch
artifact set. Its metadata resolves both the sensitivity-design digest and the
canonical batch/model/profile `run_input_sha256` cited by the CSV contracts,
and records the output-schema version, row count, and exact CSV digest.

## Reproduce the complete synthetic result set

```text
python -m microtx_sim reproduce configs/policy_prototype.toml
```

`reproduce` always runs the configured seven-scenario batch and sensitivity
analysis, then writes these 13 files:

| Artifact | Purpose |
| --- | --- |
| `seed_results.csv` | One aggregate record per seed and scenario. |
| `scenario_summary.csv` | Across-seed means, dispersion, intervals, and scenario contrasts. |
| `player_outcomes.csv` | Optional synthetic player-level outcomes controlled by `include_player_rows`. |
| `opportunity_cost_decomposition.csv` | Displaced-activity minutes, burden scores, and monetary proxies. |
| `epgc_financing.csv` | Auditable public-contract components, costs, minimum contribution, and safe profit. |
| `sensitivity.csv` | One-at-a-time sensitivity results and expected-direction checks. |
| `manifest.json` | Export-time config-file observation, effective typed-config and exact execution-input snapshots/hashes, sensitivity design snapshot/hash, source/profile lineage including population-evidence assessment, Git state, seeds, cohort digests, equations, and scope limits. |
| `summary.md` | Human-readable synthetic scenario comparison. |
| `harm_distribution.svg` | Player harm distribution. |
| `spending_distribution.svg` | Player spending distribution. |
| `harm_revenue_frontier.svg` | Scenario harm-versus-revenue relationship. |
| `opportunity_cost_decomposition.svg` | Opportunity-cost components. |
| `epgc_subsidy_requirement.svg` | Minimum EPGC contribution by scenario/seed. |

Use `--output PATH` to select another destination. CSV, JSON, Markdown, and SVG
files are written through atomic replacements. CSV column order and SVG geometry
are deterministic for identical inputs; versioned policy CSVs reject undeclared
row keys. `manifest.json` intentionally records
the run timestamp, absolute paths, environment, Git revision, and dirty state,
so the manifest itself is an audit record rather than a promise of identical
bytes across machines or invocation times.

Before any file is created or replaced, export requires the configuration,
batch, and optional sensitivity result to agree on the batch specification,
five resolved model-input groups, and profile fingerprint. Preflight mismatches
leave a new destination absent and an existing destination unchanged.

The CLI uses one validated profile bundle for both the policy batch and its
sensitivity analysis. Python callers can instead pass `country_profiles=...`;
those values are still fingerprinted exactly, but the manifest labels their
evidence lineage `unregistered_custom_profiles` and leaves repository input-file
hashes unset.

Registered profile-input lineage is schema version 4 and includes the
population-evidence bundle snapshot, verified cell results, and fail-closed
readiness assessment; historical lineage versions 1–3 remain readable. The
checked-in population bundle is empty, `ILLUSTRATIVE`, unsigned, and reports
`campaign_ready=false`. Its hashes attest repeatable bytes and extraction, not
publisher authenticity or calibration.

When `[population]` is selected, the batch and sensitivity analysis use the same
content-addressed adapter. For each seed, all seven scenarios share the same
projected assignment, and the manifest records the adapter, runtime projection,
ordered-player-ID and assignment digests, cohort digest, exact weights, and
pre-treatment population balance. That balance reports exact target-versus-
realized count and mass discrepancies for every full joint cell and separately
attests runtime jurisdiction, age, income, and household membership. Without
the section, this conditional lineage is absent and legacy initialization is
unchanged.

The current full output-bundle schema is `3.0`. Its CSV filenames and columns
and their unweighted synthetic-player semantics are unchanged from the frozen
`2.0` exhaustive-column contract; version 3 adds
an independently versioned and fingerprinted manifest envelope. Released
schemas `1.0` and `2.0` remain documented legacy forms without that manifest
version. Standalone sensitivity output uses its own two-file
`standalone_sensitivity` profile at schema `1.0`, rather than claiming to be a
13-artifact full bundle.

The dedicated `target_population_estimands` profile is also separate from the
13-file bundle. Programmatic callers may pass re-attested exact estimand
specification/result pairs to
`microtx_sim.outputs.write_target_population_estimands(...)`; it writes exactly
`target_population_estimands.csv` and
`target_population_estimand_metadata.json`. It orders records deterministically,
preserves exact rational values as decimal integers, and rejects conflicting or
malformed pairs before writing. Its upstream evidence, weights, projection,
balance, and metric-contract hashes are copied declarations rather than
independently resolved artifacts, and the profile fixes `campaign_ready=false`
and `full_output_bundle=false`.

The automatic output-v3 CSV summaries remain unweighted even when projected
execution is selected; they are not silently reinterpreted as target-population
estimates. Producing the standalone weighted files is either an explicit
additional library step or the result of an explicit `[analysis_plan]` opt-in.
In the latter case, the CLI validates planned identities before execution,
resolves exact results afterward, and writes the two files under
`prospective_analysis/` while keeping the 13 root artifacts unchanged. See
[Prospective analysis-plan composition](analysis_plan.md). Neither path supplies population calibration, empirical
validation, public comparability, or campaign readiness, and no full campaign
has been run.

Use a new or empty destination when the exact directory inventory matters. The
exporter atomically replaces its own 13 filenames but does not delete unrelated
files left by an earlier command.

Generated artifacts are ignored by Git. For an archival run, first commit the
code and configuration, confirm that `git status --short` is empty, and then run
`reproduce`; the manifest will identify that clean revision.

See [Synthetic policy prototype](policy_prototype.md) for scenario definitions,
equations, and interpretation, and [Limitations](limitations.md) before using
any output.

## Run the legacy structural check

Validate the market-model scenario and its evidence contracts without advancing
the world:

```text
python -m microtx_sim validate configs/smoke.toml
```

Then run the intentionally short connectivity check:

```text
python -m microtx_sim smoke configs/smoke.toml
```

The supplied smoke scenario runs 384 players for three one-day cycles. The
command prints JSON containing timing, the effective ledger backend, an outcome
summary, audit count, and ledger-entry count. These numbers are software
diagnostics, not empirical results.

There is intentionally no unrestricted scientific-campaign command.
Non-campaign market orchestration rejects more than 32 cycles or 5,000 players;
campaign mode must be requested through the Python API and passes stricter
provenance checks. The policy commands remain synthetic even when they run more
than the three-tick smoke check.

## Python API: one world

```python
from microtx_sim.config import load_config
from microtx_sim.core.world import World
from microtx_sim.simulation import SimulationOrchestrator

config = load_config("configs/smoke.toml", campaign=False)
with World.create(config, campaign=False) as world:
    run = SimulationOrchestrator.run(world, campaign=False)
    print(run.summary)
    print(world.step_history)
    print(world.audit_count)
    print(world.ledger.entry_count())
    for entry in world.ledger.iter_entries():
        print(entry)
```

`World.step()` advances one tick and returns a `WorldStep`.
`World.run(cycles)` is a low-level convenience loop; prefer
`SimulationOrchestrator.run()` for experiment-level use because it enforces
validation and non-campaign size guards.

With `run.step_history_retention = "full"`, `step_history` contains every
successfully completed step. With `"final_only"`, it is empty before the first
step and thereafter contains exactly the latest successfully completed step,
including across repeated calls. The returned `WorldStep` is complete in both
modes. `world.audit_count` always covers the whole run; it must not be
reconstructed from compact history. Ledger retention is controlled separately by
`run.ledger_backend`: `memory` keeps SQLite pages in memory, while `sqlite`
streams them to a file. The compatibility `world.ledger.entries` property still
materialises the complete history and should not be used in campaign-scale code.

For a small persistent structural run, select the SQLite backend and provide a
fresh path explicitly:

```python
from dataclasses import replace

from microtx_sim.config import load_config
from microtx_sim.core.ledger import Ledger
from microtx_sim.core.world import World
from microtx_sim.simulation import SimulationOrchestrator
from microtx_sim.types import LedgerBackend

config = load_config("configs/smoke.toml", campaign=False)
config = replace(
    config,
    run=replace(config.run, ledger_backend=LedgerBackend.SQLITE),
)
world = World.create(
    config,
    campaign=False,
    ledger_path="smoke-ledger.sqlite3",
)
SimulationOrchestrator.run(world, campaign=False)
seal = world.ledger.seal(metadata={"scope": "synthetic smoke only"})
world.close()
verified = Ledger.verify(seal.database_path)
print(verified.logical_sha256)
```

`Ledger.create` and `World.create(ledger_path=...)` refuse existing database,
seal, or SQLite sidecar artifacts. Sealing performs full verification, closes
the database, and writes `<database>.seal.json`. The logical digest identifies the ordered ledger
across backends; the raw file digest checks the SQLite bytes against a trusted
manifest. The unsigned pair does not prove run completion or authenticate
provenance. A database without both its internal finalization row and sidecar is
incomplete. This mechanism is an audit-artifact contract, not restart support.

## Python API: paired counterfactuals

```python
from microtx_sim.causal import MechanismCap, NullIntervention, run_paired_worlds
from microtx_sim.config import load_config
from microtx_sim.types import MonetisationMechanism

config = load_config("configs/smoke.toml", campaign=False)
paired = run_paired_worlds(
    config,
    treated=MechanismCap(
        mechanism=MonetisationMechanism.RANDOM_REWARD,
        maximum=0.10,
    ),
    control=NullIntervention(),
    cycles=3,
    campaign=False,
)

print(paired.effect)
```

The pair shares pre-treatment state and semantic random coordinates but owns
separate mutable objects. Current inputs remain synthetic/illustrative, so this
example demonstrates the causal machinery only.

When `ledger_backend` is `sqlite`, the default non-campaign pair uses two owned
temporary databases and deletes them after producing immutable outcomes. To
retain ledger artifacts, create two distinct persistent `Ledger` objects and
pass them as `treated_ledger=` and `control_ledger=`. The paired runner leaves
injected ledgers open for the caller to seal; supplying only one ledger or
sharing one store between branches is rejected.

## Compose policy regimes

```python
from microtx_sim.causal import (
    AuditRegime,
    CompositeIntervention,
    MechanismCap,
    SubsidyRegime,
)
from microtx_sim.types import MonetisationMechanism

policy = CompositeIntervention(
    (
        MechanismCap(
            mechanism=MonetisationMechanism.PRICE_OBFUSCATION,
            maximum=0.05,
        ),
        AuditRegime(
            interval_days=2,
            sensitivity=0.90,
            random_fraction=0.40,
        ),
        SubsidyRegime(
            budget_cents_per_state=2_000_000,
            interval_days=3,
            design_safety_weight=0.60,
        ),
    )
)
```

Only supplied fields are changed. Intervals must align with `tick_days`, and all
fractions must lie in [0, 1].

## Inspect outcomes

`OutcomeSnapshot` keeps component data:

- `player_harm`: player × seven harm dimensions;
- `player_spend_cents`, `player_income_cents`, and `player_debt_cents`;
- `firm_cash_cents`, `firm_operating_margin_cents`, and
  `firm_safe_revenue_share`;
- `state_subsidy_outlay_cents`.

`snapshot.composite_harm(weights)` creates a weighted view. Always retain and
report the component outcomes because a composite depends on researcher-chosen
weights.

## Run tests

```text
python -m unittest discover -s tests -v
python -m pytest
```

The standard-library command is authoritative and requires no optional test
runner. Before treating the prototype as reproducible, also run:

```text
python -m microtx_sim policy-validate configs/policy_prototype.toml
python -m microtx_sim reproduce configs/policy_prototype.toml
```

Routine development should use focused tests before the full suite. Do not
launch the 365-cycle base scenario or an empirical campaign merely to test a
code change.

## Legacy imports

The implementation moved from `microtx_sim.systems` into the canonical domain
packages `microtx_sim.consumers`, `microtx_sim.companies`,
`microtx_sim.states`, and `microtx_sim.market`. Compatibility imports remain,
but new integrations should use the canonical paths.
