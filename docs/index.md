# Documentation index

This documentation describes the executable research skeleton as it exists in
the repository. It separates implemented mechanisms from calibration plans and
should be read before interpreting any simulation output.

## Start here

| Document | Use it for |
| --- | --- |
| [README](../README.md) | Project status, principles, module map, and quick start. |
| [Architecture](architecture.md) | System boundaries, agent knowledge, ownership, randomness, accounting, and complexity. |
| [Model specification](model_spec.md) | Agents, mechanisms, dynamics, outcomes, and mathematical/computational scope. |
| [Simulation lifecycle](simulation_lifecycle.md) | Exact ordering of one tick, events, phase coordination, and paired runs. |
| [Causal design](causal_design.md) | Estimand, paired worlds, common random numbers, interventions, assumptions, and future campaign design. |

## Running and extending the project

| Document | Use it for |
| --- | --- |
| [Usage](usage.md) | CLI commands and minimal Python examples. |
| [Configuration](configuration.md) | Every scenario field, units, validation, precedence, and supplied scenarios. |
| [Development guide](development.md) | Setup, module ownership, information boundaries, tests, and Git workflow. |

## Evidence and interpretation

| Document | Use it for |
| --- | --- |
| [Data sources](data_sources.md) | Full source inventory, effective runtime inputs, dormant contracts, units, and lineage. |
| [Evidence policy](evidence.md) | Provenance statuses, evidence contracts, campaign gates, and update workflow. |
| [Limitations](limitations.md) | Empirical, causal, legal, funding, information, market, and technical limits. |
| [Roadmap](roadmap.md) | Calibration blockers, causal-design work, institutional fidelity, validation, and exact performance improvements. |

## Current release boundary

`configs/smoke.toml` is a three-cycle synthetic structural check. The larger
`configs/base.toml` is a future-scale architecture scenario and is deliberately
blocked by provenance validation. No full scientific campaign has been run.

The documentation is written in English. Official source titles may retain
proper names from the publishing institution, but all model descriptions,
interfaces, assumptions, and user-facing messages use English.
