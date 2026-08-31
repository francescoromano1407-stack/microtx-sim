# Documentation index

This documentation describes the stable, tested synthetic prototype and its
fail-closed pre-campaign infrastructure as they exist in the repository. It
separates implemented software behaviour from calibration plans and readiness
claims and should be read before interpreting any simulation output.

## Start here

| Document | Use it for |
| --- | --- |
| [README](../README.md) | Project status, principles, module map, and quick start. |
| [Architecture](architecture.md) | System boundaries, agent knowledge, ownership, randomness, accounting, and complexity. |
| [Model specification](model_spec.md) | Agents, mechanisms, dynamics, outcomes, and mathematical/computational scope. |
| [Synthetic policy prototype](policy_prototype.md) | Seven scenarios, daily life actions, welfare harm, EPGC financing, sensitivity analysis, and synthetic interpretation. |
| [Simulation lifecycle](simulation_lifecycle.md) | Exact ordering of one tick, events, phase coordination, and paired runs. |
| [Causal design](causal_design.md) | Estimand, paired worlds, common random numbers, interventions, assumptions, and future campaign design. |
| [Prospective analysis plan](analysis_plan.md) | Concrete opt-in plan, pre-run validation, exact run binding, plan-level Monte Carlo aggregate, and fail-closed campaign scope. |
| [Full campaign contract](campaign_contract.md) | Successor-plan binding, uncertainty and convergence, execution attestation, layer boundaries, and fail-closed readiness states. |
| [Exploratory synthetic campaign](exploratory_synthetic_campaign.md) | Separate non-empirical run purpose, dual plan binding, interpretation limits, and validation-only review gate. |
| [Optimized exploratory execution engine](execution_engine.md) | Explicit CPU/GPU backend contract, bounded host scheduling, exact checkpoint/resume/progress semantics, parity rules, and bounded benchmark evidence. |

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
| [Target population and readiness](population_readiness.md) | Selected analytic population, joint cells, official demographic and gaming evidence, weight semantics, income modeling, and unresolved campaign blockers. |
| [Evidence policy](evidence.md) | Provenance statuses, evidence contracts, campaign gates, and update workflow. |
| [Limitations](limitations.md) | Empirical, causal, legal, funding, information, market, and technical limits. |
| [Roadmap](roadmap.md) | Calibration blockers, causal-design work, institutional fidelity, validation, and exact performance improvements. |

## Current release boundary

`configs/policy_prototype.toml` is the supported synthetic seven-scenario
prototype. It can reproducibly generate six CSV files, one manifest, one
Markdown summary, and five SVG charts. `configs/smoke.toml` remains a three-cycle
structural check. The larger `configs/base.toml` is a future-scale architecture
scenario and is deliberately blocked by provenance validation.

`configs/policy_campaign.toml` is a separate, complete campaign-shaped
configuration bound to a schema-v3 successor plan, formal uncertainty and
convergence contracts, projected population and monetary inputs, a persistent
ledger, and execution attestation. It remains deliberately blocked and was not
executed. Its hashes establish identity only, and its unavailable scientific
inputs cannot be replaced by zero-variance assumptions or readiness labels.

`configs/policy_exploratory_synthetic.toml` is a third, explicitly exploratory
configuration. It preserves the scientific parent estimand while binding a
separate non-empirical sidecar plan and isolated output namespace. It does not
alter the production configuration or inherit production authority. It has
not been executed and remains `campaign_ready=false`. Its reviewed command is
technically launchable. The optimized v2 executor writes complete main seeds and
individual sensitivity units to content-addressed atomic checkpoints. A launch
with the exact same v2 run identity resumes verified committed work and requeues
only uncommitted in-flight units. The earlier v1 interrupted attempt is
preserved as lineage and is not treated as a compatible resume point.

Population evidence/design, an optional exact runtime mapping/adapter, per-seed
projection and pre-treatment balance lineage, and a separate two-file target-
population estimand writer are implemented as reproducibility infrastructure.
Ordinary development configurations omit projected execution. The checked-in
evidence, design, and schema-v2 runtime mapping are complete but illustrative;
the campaign candidate selects them only to exercise the fail-closed preflight.
Existing output-v2-compatible CSVs remain unweighted. These contracts do not
provide calibration, public comparability, or campaign readiness, so comparable
population profiles remain an open P0 task.

No empirical validation is claimed and no full scientific campaign has been
run. The exported comparisons are conditional results of synthetic structural
assumptions, not estimates of observed people, markets, or policies.

The documentation is written in English. Official source titles may retain
proper names from the publishing institution, but all model descriptions,
interfaces, assumptions, and user-facing messages use English.
