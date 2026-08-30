# Microtransactions — causal agent-based simulation

This repository contains a stable, tested synthetic simulation prototype for
the question:

> How much additional harm is causally attributable to mobile-game monetisation
> mechanisms after accounting for a player's pre-existing vulnerability, and
> which combination of regulation and public funding can sustain an
> economically viable game without depending on compulsive spending?

The project does not implement a playable game. It models a competitive market
containing heterogeneous consumers and households, abstract games, strategic
mobile-game companies, public rankings, governments, audits, sanctions, and
conditional subsidies.

## Current status

The code is a tested research prototype, not a calibrated policy model. It now
supports a reproducible seven-scenario synthetic policy batch, common seeded
cohorts, multidimensional welfare-harm and opportunity-cost outputs, an EPGC
financing calculation, one-at-a-time sensitivity analysis, and a versioned
13-artifact export. These capabilities establish software behaviour inside the
declared model only. They must not be used to report real-world causal effects,
national spending estimates, clinical conclusions, legal conclusions, or an
optimal funding policy.

- `configs/smoke.toml` runs 384 players for three cycles and is explicitly
  synthetic; it remains the smallest connectivity check for the market model.
- `configs/policy_prototype.toml` runs the seven required policy scenarios over
  identical seeded synthetic cohorts and is the supported reproducible
  prototype workflow.
- `configs/policy_prospective.toml` is an explicitly opt-in, unregistered
  16-player algebraic validation fixture binding the checked-in plan and
  designed population. It is not a campaign configuration in the scientific
  readiness sense and must not be promoted as one.
- `configs/base.toml` describes a future 50,000-player, five-company, eight-game
  scale and selects final-step-only history plus a file-backed SQLite ledger,
  but campaign validation deliberately rejects its uncalibrated inputs. That
  future scale has not been run or benchmarked.
- No empirical or scientific campaign has been run or authorised for this
  release.
- Profile schema version 2 retains declaration-only dated FX/PPP contracts;
  version 3 additionally requires each conversion to reference a verified,
  content-addressed rate extraction. The checked-in source bundle binds the
  source-catalogue digest but is empty, illustrative, and explicitly has no
  signature. No conversion rates are checked in. Source extraction, signature,
  output/design, population, and preregistration gates therefore remain
  independently false, so present outputs are non-comparable across
  jurisdictions.
- A separate population-evidence bundle schema version 1 can attest exact CSV
  bytes and strict joint age × household-income-band × household-type × gaming
  × pre-treatment payer-history cells whose reduced rational masses sum exactly
  to one. The checked-in population bundle is empty, `ILLUSTRATIVE`, unsigned,
  and always reports `campaign_ready=false`. Schema v1 does not declare complete
  income/household domains or a disjoint sample partition, so even a populated
  bundle cannot clear calibration-target or held-out-validation subgates.
- A separate static population-design schema can bind those exact evidence
  results to complete age, income, household, gaming, and payer-history domains,
  declared calibration/validation source-unit partitions, target counts, and a
  deterministic exact-rational Hamilton apportionment plan. The checked-in
  `population_design.toml` is also empty and `ILLUSTRATIVE`. Partition record and
  cluster hashes are declarations, not publisher-authenticity or independently
  held-out proof, and population-design schema v1 remains fail-closed for
  campaign use.
- Registered profile-input lineage is now version 4 and retains the population
  bundle, extracted cells, hashes, and readiness assessment; lineage versions
  1–3 remain readable. These hashes establish reproducibility, not publisher
  authenticity or calibration.
- Market and policy configurations may opt into `population.mode =
  "projected_v1"` with exact design, runtime-mapping, and adapter identifiers.
  The file-backed adapter re-attests the static `PopulationApportionmentPlan`,
  consumes its exact cell counts and rational weights without reallocating them,
  and binds the explicit conversion from source household-income/type semantics
  to runtime personal monthly disposable-income intervals and modeled household
  sizes. No checked-in configuration selects this path and no runtime mapping
  bundle is checked in, so the supplied runs retain the legacy marginal
  initializer.
- A selected projection is content-addressed through adapter, runtime projection,
  ordered player IDs, and assignment identities. Batch and sensitivity runs
  record the same per-seed cohort, exact weights, and pre-treatment balance
  lineage; the balance artifact reports planned-versus-realized full joint-cell
  counts and masses and separately attests runtime jurisdiction, age, income,
  and household membership. Baseline gamer and payer labels remain sidecar-only
  metadata and do not set gameplay, payment access, or spending history.
- An isolated exact-rational estimand primitive implements weighted means,
  paired mean differences, and deterministic weighted quantiles. It re-attests
  supplied design weights and records caller-supplied projection, balance,
  metric-contract, evidence, and output-profile digest declarations; it does not
  resolve those declarations to verified artifacts. A dedicated standalone
  `target_population_estimands` profile can write exact results to
  `target_population_estimands.csv` and
  `target_population_estimand_metadata.json`; it is not part of the 13-file
  output-v3 bundle. It remains a standalone programmatic profile by default;
  an optional prospective analysis plan can now make the policy CLI resolve and
  write it under `prospective_analysis/` after validating all planned input
  identities before execution. Output schema v3 retains
  the frozen v2-compatible CSV columns and their unweighted synthetic-player
  semantics. The adapter, balance, lineage, and writer are reproducibility
  infrastructure, not population calibration, external provenance, empirical
  validation, public comparability, or campaign readiness. Comparable
  populations remain an open P0 item and no full campaign has been run.
- A strict content-addressed prospective-plan schema freezes exactly one primary
  estimand, explicit scenario direction, fixed seeds, harm weights, metric and
  output identities, the exact profile-input-lineage fingerprint, and an
  executable pre-treatment population predicate. It
  is opt-in, requires projected-population execution, and binds exact per-seed
  results back to population execution and balance lineage. The checked-in
  schema-v2 illustrative plan adds an outcome-blind equal-seed primary mean,
  sample SD, Monte Carlo standard error, and normal 95% Monte Carlo interval in
  a separate prospective profile. It remains `UNREGISTERED`,
  `preregistered=false`, and `campaign_ready=false` and is selected only by
  `configs/policy_prospective.toml`.

See [Data sources](docs/data_sources.md) for the distinction between inputs that
currently affect equations and evidence retained only for future calibration.
See [Limitations](docs/limitations.md) before interpreting any output.

## Model principles

- **Local knowledge:** latent truth is private to the research kernel. Agents
  receive noisy, delayed, reported, or purchased observations and update
  fallible private beliefs.
- **Explicit counterfactuals:** paired worlds share their pre-treatment
  population and semantic random coordinates while an explicit intervention
  changes monetisation, audits, or subsidies.
- **Heterogeneous agents:** age, income, motives, vulnerability, skills,
  supervision, financial literacy, credit access, firm culture, and regulator
  priorities are continuous and correlated where appropriate.
- **Emergent strategy:** content releases, monetisation changes, research,
  compliance, evasion, acquisition, collaboration, and collusion are selected
  from perceived utility or NPV rather than a scripted scenario.
- **Exact declared choices:** each consumer evaluates every represented game,
  and content planning enumerates the complete configured finite candidate set.
  Blocking limits memory without sampling alternatives.
- **Exact-cent accounting:** financial state uses integer simulation cents,
  overflow checks, exact aggregation, and atomic append-only ledger batches.
  The ledger can stream to SQLite without retaining every transfer as a Python
  object.
- **Provenance gates:** scientific campaign mode rejects synthetic,
  illustrative, or merely anchored dependencies.

## Modular structure

| Area | Canonical module |
| --- | --- |
| Consumer population and behaviour | `microtx_sim.consumers` |
| Company observations and strategy | `microtx_sim.companies` |
| Government audit and enforcement policy | `microtx_sim.states` |
| Popularity and delayed public rankings | `microtx_sim.market` |
| Latent world state and intervention façade | `microtx_sim.core.world` |
| Accounting and phase coordination | `microtx_sim.simulation` |
| One simulated tick/day | `microtx_sim.simulation.day` |
| Multi-cycle execution | `microtx_sim.simulation.orchestrator` |
| Paired-world causal contrasts | `microtx_sim.causal` |
| Prospective plan and exact run binding | `microtx_sim.causal.analysis_plan`, `microtx_sim.causal.analysis_binding` |
| Seven-scenario synthetic policy batch | `microtx_sim.causal.batch` |
| Daily life-action policy process | `microtx_sim.simulation.policy_day` |
| Welfare harm and opportunity cost | `microtx_sim.metrics.harm` |
| EPGC financing | `microtx_sim.funding` |
| Sensitivity analysis | `microtx_sim.analysis` |
| Versioned tables, manifests, summaries, and SVG charts | `microtx_sim.outputs` |
| Profiles and provenance validation | `microtx_sim.data` |

The old `microtx_sim.systems` imports remain as compatibility shims. New code
should use the canonical domain packages above.

## Quick start

Python 3.11 or later is required.

```text
python -m pip install -e ".[dev]"
python -m microtx_sim policy-validate configs/policy_prototype.toml
python -m microtx_sim policy-validate configs/policy_prospective.toml
python -m microtx_sim policy-batch configs/policy_prototype.toml
python -m microtx_sim policy-sensitivity configs/policy_prototype.toml
python -m microtx_sim reproduce configs/policy_prototype.toml
python -m unittest discover -s tests -v
```

`reproduce` writes the complete 13-artifact synthetic result set to
`artifacts/policy_prototype/` unless `--output` overrides that location. The
directory is ignored by Git. The command does not authorise or start an
empirical campaign.

For the smaller legacy connectivity check, run:

```text
python -m microtx_sim validate configs/smoke.toml
python -m microtx_sim smoke configs/smoke.toml
```

## Documentation

Start with the [documentation index](docs/index.md). The main references are:

- [Architecture](docs/architecture.md)
- [Model specification](docs/model_spec.md)
- [Simulated lifecycle](docs/simulation_lifecycle.md)
- [Causal design](docs/causal_design.md)
- [Prospective analysis plan](docs/analysis_plan.md)
- [Synthetic policy prototype](docs/policy_prototype.md)
- [Configuration reference](docs/configuration.md)
- [Data sources and lineage](docs/data_sources.md)
- [Limitations](docs/limitations.md)
- [Development guide](docs/development.md)
- [Roadmap](docs/roadmap.md)
