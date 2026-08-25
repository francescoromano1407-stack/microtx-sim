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
artifact set.

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
| `manifest.json` | Configuration/source/profile-input hashes, exact profile snapshot, source retrieval date, contract summaries, actual profile codes, Git state, environment, seeds, cohort digests, equations, and scope limits. |
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

The CLI uses one validated profile bundle for both the policy batch and its
sensitivity analysis. Python callers can instead pass `country_profiles=...`;
those values are still fingerprinted exactly, but the manifest labels their
evidence lineage `unregistered_custom_profiles` and leaves repository input-file
hashes unset.

The current output schema is `2.0`. It preserves v1 populated-table header order
but expands empty seed, scenario-summary, and sensitivity headers to the full
declared contracts.

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
command prints JSON containing timing, an outcome summary, audit count, and
ledger-entry count. These numbers are software diagnostics, not empirical
results.

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
world = World.create(config, campaign=False)
run = SimulationOrchestrator.run(world, campaign=False)

print(run.summary)
print(world.step_history)
print(world.ledger.entries)
```

`World.step()` advances one tick and returns a `WorldStep`.
`World.run(cycles)` is a low-level convenience loop; prefer
`SimulationOrchestrator.run()` for experiment-level use because it enforces
validation and non-campaign size guards.

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
