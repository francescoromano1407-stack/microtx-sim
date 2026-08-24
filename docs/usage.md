# Usage

## Install

Python 3.11 or later is required. From the repository root:

```text
python -m pip install -e ".[dev]"
```

The only runtime dependency is NumPy. The development extra installs pytest,
although the suite also runs with the standard-library unittest runner.

## Validate inputs

```text
python -m microtx_sim validate configs/smoke.toml
```

This parses the scenario, loads the local evidence registry and jurisdiction
contracts, and reports provenance caveats. It does not advance the simulation.

## Run the short structural check

```text
python -m microtx_sim smoke configs/smoke.toml
```

The supplied smoke scenario runs 384 players for three one-day cycles. The
command prints JSON containing timing, an outcome summary, audit count, and
ledger-entry count. These numbers are software diagnostics, not empirical
results.

There is intentionally no unrestricted campaign command. Non-campaign
orchestration rejects more than 32 cycles or 5,000 players; campaign mode must
be requested through the Python API and passes stricter provenance checks.

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

Routine verification should use focused tests and the tiny smoke scenario. Do
not launch the 365-cycle base scenario or a full campaign merely to test a code
change.

## Legacy imports

The implementation moved from `microtx_sim.systems` into the canonical domain
packages `microtx_sim.consumers`, `microtx_sim.companies`,
`microtx_sim.states`, and `microtx_sim.market`. Compatibility imports remain,
but new integrations should use the canonical paths.
