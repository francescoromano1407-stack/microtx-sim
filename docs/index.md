# Documentation index

This documentation describes the stable, tested synthetic prototype as it
exists in the repository. It separates implemented software behaviour from
calibration plans and should be read before interpreting any simulation output.

## Start here

| Document | Use it for |
| --- | --- |
| [README](../README.md) | Project status, principles, module map, and quick start. |
| [Architecture](architecture.md) | System boundaries, agent knowledge, ownership, randomness, accounting, and complexity. |
| [Model specification](model_spec.md) | Agents, mechanisms, dynamics, outcomes, and mathematical/computational scope. |
| [Synthetic policy prototype](policy_prototype.md) | Seven scenarios, daily life actions, welfare harm, EPGC financing, sensitivity analysis, and synthetic interpretation. |
| [Simulation lifecycle](simulation_lifecycle.md) | Exact ordering of one tick, events, phase coordination, and paired runs. |
| [Causal design](causal_design.md) | Estimand, paired worlds, common random numbers, interventions, assumptions, and future campaign design. |

## Running and extending the project

| Document | Use it for |
| --- | --- |
| [Usage](usage.md) | Exact validation, batch, sensitivity, reproduction, smoke, and test commands plus the 13 artifacts. |
| [Configuration](configuration.md) | Market and policy TOML schemas, units, validation, precedence, and supplied scenarios. |
| [Development guide](development.md) | Setup, module ownership, reproducibility contract, tests, and Git workflow. |

## Evidence and interpretation

| Document | Use it for |
| --- | --- |
| [Data sources](data_sources.md) | Full source inventory, effective runtime inputs, dormant contracts, units, and lineage. |
| [Evidence policy](evidence.md) | Provenance statuses, evidence contracts, campaign gates, and update workflow. |
| [Limitations](limitations.md) | Empirical, causal, legal, funding, information, market, and technical limits. |
| [Roadmap](roadmap.md) | Calibration blockers, causal-design work, institutional fidelity, validation, and exact performance improvements. |

## Current release boundary

`configs/policy_prototype.toml` is the supported synthetic seven-scenario
prototype. It can reproducibly generate six CSV files, one manifest, one
Markdown summary, and five SVG charts. `configs/smoke.toml` remains a three-cycle
structural check. The larger `configs/base.toml` is a future-scale architecture
scenario and is deliberately blocked by provenance validation.

Population evidence/design, an optional exact runtime mapping/adapter, per-seed
projection and pre-treatment balance lineage, and a separate two-file target-
population estimand writer are implemented as reproducibility infrastructure.
The supplied configurations omit projected execution, the checked-in evidence
and design are empty and illustrative, and no runtime mapping is checked in.
Existing output-v2-compatible CSVs remain unweighted. These contracts do not
provide calibration, public comparability, or campaign readiness, so comparable
population profiles remain an open P0 task.

No empirical validation is claimed and no full scientific campaign has been
run. The exported comparisons are conditional results of synthetic structural
assumptions, not estimates of observed people, markets, or policies.

The documentation is written in English. Official source titles may retain
proper names from the publishing institution, but all model descriptions,
interfaces, assumptions, and user-facing messages use English.
