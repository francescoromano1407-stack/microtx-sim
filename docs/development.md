# Development guide

This guide covers local setup, module boundaries, deterministic implementation,
testing, and the repository workflow. It is written for the stable, tested
synthetic prototype; empirical and large scientific campaigns remain
deliberately gated.

## Requirements and setup

- Python 3.11 or later
- NumPy 2.0 or later
- pytest 8 or later for the optional development environment

Create an isolated environment using the tool of your choice, then install the
package in editable mode:

```text
python -m pip install -e ".[dev]"
```

For the fully locked project environment, install uv 0.11.6 and synchronize the
committed lock file without changing it:

```text
uv sync --locked --extra dev
```

`uv.lock` is the reproducible runtime and test dependency boundary used by
continuous integration; the isolated setuptools build backend is pinned
separately in `pyproject.toml`. Run `uv lock` only as a deliberate
dependency-update operation, review both files, and rerun the complete test
matrix. The `[tool.uv]` requirement prevents a different resolver version from
silently rewriting the lock.

The package exposes both a module entry point and the `microtx-sim` console
command. Validate both supported entry points before changing model logic:

```text
python -m microtx_sim policy-validate configs/policy_prototype.toml
python -m microtx_sim validate configs/smoke.toml
python -m microtx_sim smoke configs/smoke.toml
```

The smoke command executes only three one-day cycles with 384 players. It is a
connectivity check, not an empirical result. The non-campaign runner rejects
more than 32 cycles or more than 5,000 players. Campaign mode additionally
requires calibrated configuration and provenance.

The policy configuration is a separate synthetic-only prototype. Its complete
reproducibility workflow is:

```text
python -m microtx_sim policy-batch configs/policy_prototype.toml
python -m microtx_sim policy-sensitivity configs/policy_prototype.toml
python -m microtx_sim reproduce configs/policy_prototype.toml
```

`reproduce` creates 13 ignored artifacts under
`artifacts/policy_prototype/`. Passing tests or producing those files validates
software contracts only, not external validity or calibration.

## Source-tree responsibilities

Keep data, agent state, domain rules, daily execution, and experiment control
separate:

```text
src/microtx_sim/
├── analysis/     one-at-a-time sensitivity analysis and stability checks
├── agents/       agent records, immutable observations, and private state
├── causal/       interventions, paired worlds, seven scenarios, and batch contrasts
├── companies/    company decision and resolution logic
├── consumers/    population, life-state, action choice, and consumer dynamics
├── core/         generic events, observations, ledger, and world state
├── data/         evidence profiles and provenance validation
├── domain/       abstract game, content, and monetisation intervention vectors
├── funding/      EPGC public-value financing equation and audit trail
├── market/       popularity truth and delayed public rankings
├── metrics/      market outcomes plus policy harm and opportunity-cost measures
├── outputs/      atomic tables, run manifests, summaries, and deterministic SVGs
├── states/       audit and public-funding logic
└── simulation/   market and policy day processors and run orchestrators
```

`policy_config.py` owns the strict synthetic policy TOML boundary. Keep it
separate from the legacy `SimulationConfig`: accepting a new policy field must
be an explicit schema and documentation change.

Place a change in the module that owns its mechanism. For example, an acquisition
policy belongs in company logic; a complaint-to-audit sensor belongs in state
logic; scheduling their order belongs in the day processor. The world owns
latent mutable state but should not become a second implementation of domain
policies.

## Information-boundary rules

The research kernel may know latent truth; agents may not. Preserve these rules
when adding any mechanism:

1. Never pass `World` or mutable kernel tables to an agent policy.
2. Build an immutable, detached observation containing only information that the
   agent could legitimately possess at that time.
3. Attach an `as_of` or data tick, signal age, precision, and source where they
   affect interpretation.
4. Implement information costs, delays, noise, and imperfect detection before
   the decision, not as annotations added after the decision.
5. Update private beliefs only from received signals or audit evidence.
6. Keep compliance truth, latent vulnerability, true popularity, and researcher
   harm classifications out of company and regulator observations.
7. Test both immutability and the absence of future information.

A convenient review question is: “Could this policy reconstruct a value that it
did not pay for, observe, or previously learn?” If yes, the interface leaks
information.

## Deterministic randomness

All stochastic mechanisms must use `CounterRNG`. A draw is a pure function of:

```text
(seed, entity_id, tick, stream, draw_index)
```

Use `stable_stream_id()` for a named stream. Do not use Python's salted `hash`,
the global NumPy random state, a mutable generator cursor, or iteration position
as an implicit random coordinate. Stream names and draw-index meanings are part
of the reproducibility contract and should be changed only deliberately.

When adding an action menu, assign shocks to stable action coordinates even when
an action is infeasible in one counterfactual branch. When vectorising or
changing block size, write a test that permutes entity order or compares chunk
sizes. Identical semantic coordinates must produce identical draws.

The policy batch must additionally preserve one cohort digest per seed across
all seven scenarios. A scenario-specific branch may make an action infeasible,
but it must not shift later semantic random coordinates in another branch.

## Exact computation and money

The implementation's “no approximation” goal applies to finite computational
choice sets and accounting, not to the truth of behavioural equations.

- Evaluate every known game in consumer choice; blocking may change memory use,
  but must not sample or discard alternatives.
- Enumerate the complete declared content-candidate set. If the set is changed,
  document that as a model change.
- Store money in signed 64-bit integer simulation cents and check overflow before
  cumulative mutation.
- Use integer aggregation for financial flows and keep ledger references unique.
- Do not pass money through floating point except for an explicitly bounded rate
  calculation followed by documented rounding.
- Do not aggregate nominal source currencies until a conversion contract has
  been specified.

Any optimisation should first have a reference test demonstrating equality of
decisions, cents, rankings, and outcomes.

## Adding a mechanism

Use this sequence for a material feature:

1. Define the scientific construct and whether it is latent, observed, believed,
   or reported.
2. Add its provenance or mark it explicitly as synthetic/illustrative.
3. Add typed state and validate its shape, range, unit, and mutability.
4. Define the observation interface before implementing the policy.
5. Allocate stable random streams and document each coordinate.
6. Implement the mechanism in its owning domain module.
7. Connect it to the day processor with an explicit event priority or phase.
8. Reconcile all cash flows and liabilities.
9. Add unit, information-boundary, deterministic, and integration tests.
10. Update the model and lifecycle documentation in the same milestone.

## Tests

Run the full suite from the repository root with the standard-library runner:

```text
uv run --no-sync python -m unittest discover -s tests -v
```

Without uv, the equivalent authoritative command is
`python -m unittest discover -s tests -v`. The optional development dependency
also supports the shorter `python -m pytest` command.

The suite is organised by contract:

- `test_players.py`: population invariants, correlated heterogeneity, immutable
  baseline vulnerability, and ex-post spending segments;
- `test_player_dynamics.py`: exact choice, resources, rare card use, harm columns,
  monetary reconciliation, and chunk independence;
- `test_firm_strategy.py`: fallible observations, heterogeneous firms, content
  trade-offs, strategic actions, and reciprocal agreements;
- `test_regulation.py`: signal-based selection, hidden compliance truth, imperfect
  evidence, and evasion;
- `test_popularity.py`: separation of true and delayed/noisy public rankings;
- `test_rng_events.py`: deterministic streams, event ordering, immutability, and
  belief timing;
- `test_module_boundaries.py`: canonical domain imports, legacy compatibility,
  and separation of daily/orchestration logic from `World`;
- `test_profiles.py`: source references, units, profile contracts, and campaign
  provenance gates;
- `test_causal.py`: paired differences and composable interventions;
- `test_world_integration.py`: scheduling and end-to-end system connections;
- `test_config_domain.py`: configuration safeguards, ledger balance, and exact
  content search;
- `test_player_life.py` and `test_policy_decision.py`: heterogeneous life state,
  action feasibility, seeded choice, spending limits, and zero-player paths;
- `test_monetisation.py`: every intervention-vector field and safety control;
- `test_welfare_harm.py`: six-component harm, adult/youth opportunity cost,
  reconciliation, and extreme inputs;
- `test_epgc.py`: safe-profit equation, bonuses, budget cap, penalties,
  clawbacks, zero cost, and overflow boundaries;
- `test_policy_batch.py`: all seven scenarios, shared cohorts, paired contrasts,
  deterministic seeds, and complete-batch integration;
- `test_sensitivity.py`: OAT grid, expected-direction checks, and unstable-field
  reporting;
- `test_outputs.py` and `test_policy_export.py`: schema, atomic writers,
  escaping, empty rows, deterministic SVGs, hashes, and the 13-file contract;
- `test_policy_config.py` and `test_policy_cli.py`: strict TOML parsing, command
  dispatch, output overrides, deterministic smoke execution, and error
  behaviour;
- `test_documentation.py`: local Markdown links across the README and reference
  documentation.

For a change local to one module, run its focused test first and the full suite
before a milestone commit. For example:

```text
python -m unittest tests.test_outputs -v
python -m unittest tests.test_policy_batch -v
python -m unittest discover -s tests -v
```

A simulation test should use the smallest population and horizon that exercise
the contract. Do not launch the 50,000-player base scenario or an empirical
campaign as routine verification.

## Reproducible artifact contract

`microtx_sim.outputs` owns the exhaustive versioned column sets, fixed filenames,
atomic UTF-8 CSV/JSON/Markdown writes, and dependency-free SVG renderers. Keep
these properties when changing outputs:

1. preserve canonical column order and reject undeclared keys in versioned
   tables; add fields explicitly to the schema before exporting them;
2. write through a same-directory temporary file and atomic replacement;
3. reject `NaN`, infinity, inconsistent reconciliations, and conflicting schema
   metadata;
4. keep empty and zero-player results valid and machine-readable;
5. update `OUTPUT_SCHEMA_VERSION` for an incompatible contract;
6. include configuration/source hashes, seeds, cohort digests, Git state,
   environment, equations, and scope limits in `manifest.json`;
7. never imply that deterministic synthetic output is empirically validated.

Output schema `2.0` is the first exhaustive-column contract. It retains the
released v1 prefix and non-empty header order, but empty seed, scenario-summary,
and sensitivity tables now include every declared extension column. The named
`*_V1_PREFIX_COLUMNS` tuples are the migration boundary for compatibility tests;
an exact schema fingerprint forces any later column or artifact change through
an explicit versioned migration.

The six CSV files, manifest, Markdown summary, and five SVG charts form the
13-file `reproduce` contract listed in [Usage](usage.md).

## Code conventions

- Prefer typed dataclasses and narrow interfaces over unstructured dictionaries.
- Validate arrays at construction boundaries, including dtype and shape.
- Keep public observations immutable and detach mutable arrays before exposing
  them.
- Use structure-of-arrays NumPy tables for population-scale data and ordinary
  objects for the small number of strategic agents.
- Document units and population bases in names or contracts.
- Treat event priority as model semantics and cover ordering changes with an
  integration test.
- Keep outcome dimensions separate in storage. A weighted composite is a
  reporting view, not a destructive replacement.

## Git workflow

Use small, meaningful commits whenever model logic or architecture changes
materially. A practical sequence is:

1. inspect the worktree and preserve unrelated changes;
2. implement one coherent milestone;
3. review the diff for information leaks, unit changes, accidental generated
   files, and undocumented random-stream changes;
4. run focused tests, then the full suite;
5. run `policy-validate` when the policy schema or assumptions changed;
6. commit code, tests, configuration, and documentation as one coherent
   milestone;
7. confirm `git status --short` is empty;
8. run `reproduce` from that clean revision when an auditable artifact set is
   required; the generated directory remains ignored by Git;
9. begin the next logical change from a clean or clearly understood worktree.

Use English, outcome-oriented commit subjects such as:

```text
refactor: separate daily execution from world state
docs: document evidence status and campaign limits
feat(policy): add seeded seven-scenario prototype batch
test(outputs): verify deterministic artifact contract
```

Do not commit secrets, local environments, caches, large generated runs, or raw
restricted microdata. Do not commit `artifacts/`; the manifest records the exact
commit identifier, dirty state, configuration hash, source-registry hash, and
runtime environment. A dirty manifest is useful for diagnosis but is not a
clean archival reproduction.

## Pre-campaign checklist

Before authorising a substantive run:

- configuration status is `CALIBRATED`;
- every dependency in the profile bundle passes campaign validation;
- source and transformation versions are frozen;
- estimands and interventions are registered;
- output storage and privacy controls are configured;
- null paired worlds are exactly identical;
- replicated small runs are deterministic;
- ledger balance, overflow, and disk-space checks pass;
- runtime and memory have been benchmarked at a representative scale;
- the code revision and environment are recorded in a run manifest.
