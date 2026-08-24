# Causal design

## Research question

The project asks:

> How much additional harm is causally attributable to mobile-game monetisation
> mechanisms after accounting for a player's pre-existing vulnerability, and
> which combination of regulation and public funding can sustain an economically
> viable game without depending on compulsive spending?

The implemented design estimates contrasts inside the declared agent-based
model. It does not yet identify a real-world treatment effect because the
behavioural equations, exposure process, country profiles, and intervention
parameters are not empirically calibrated.

## Primary estimand

For player `i`, define incremental harm under monetisation regime `M` relative
to a declared comparison regime `M0` as:

```text
tau_i = H_i(M, R, S; U_i) - H_i(M0, R0, S0; U_i)
```

where:

- `H_i` is the seven-component harm outcome at the evaluation horizon;
- `R` and `R0` are treated and control regulatory regimes;
- `S` and `S0` are treated and control public-funding regimes;
- `U_i` contains the player's pre-treatment characteristics and semantic
  exogenous random coordinates.

The default configuration names the aggregate target
`market_regime_total_effect`. The comparison regime is never inferred from a
label: treated and control interventions are supplied explicitly.

The stored harm components are financial stress, essential-spend displacement,
debt, unauthorised spending, loss of control, functioning impairment, and
regret. `compare_outcomes()` preserves the full player-by-component difference
matrix. The mean composite effect is a reporting view constructed with explicit
weights; it does not replace the components.

## Structural control of pre-existing vulnerability

Baseline vulnerability is initialised before treatment, copied into both worlds,
and write-protected. The two worlds are created independently but checked for
exact equality across causally relevant player and game columns, firms, states,
and jurisdiction metadata before any intervention is applied.

This is preferable to subtracting vulnerability with a post-hoc regression
inside the simulator: each player acts as their own structural counterfactual at
the same pre-treatment state. It does not prove that the vulnerability construct
or its distribution is empirically valid, and it does not control omitted
real-world vulnerabilities that the model does not represent.

## Common random numbers

`CounterRNG` addresses a draw by:

```text
(seed, entity_id, tick, stream, draw_index)
```

The treated and control worlds therefore query the same random field at the same
semantic coordinates. There is no shared mutable generator cursor. A purchase
or action that happens only in one branch cannot consume an extra draw and shift
later exogenous shocks in the other branch.

Random coordinates must remain semantically stable. Firm action shocks are
allocated by action identity before financial feasibility is applied; consumer
draws do not depend on block position or iteration order. Changing a stream name
or draw-index meaning is a model-version change.

A null-versus-null pair is tested for exact equality. Exact null identity is a
software and pairing invariant, not evidence that a non-null intervention is
realistically specified.

## Implemented interventions

The current intervention protocol supports:

| Intervention | Effect |
| --- | --- |
| `NullIntervention` | Makes no change and provides an explicit neutral branch. |
| `MechanismCap` | Applies a persistent maximum to one monetisation mechanism, globally or for selected games. |
| `AuditRegime` | Changes audit interval, sensitivity, specificity, or random-target fraction. |
| `SubsidyRegime` | Changes per-state budget, review interval, and quality/design-safety/accessibility weights. |
| `CompositeIntervention` | Applies an ordered tuple of interventions to the same world. |

Mechanism caps remain active after later company decisions. Audit and subsidy
intervals must remain aligned with `tick_days`. Subsidy applications must predate
the review and are visible only in their eligible synthetic home jurisdiction.

The current code runs one explicit treated/control pair. It does not yet build
the full factorial matrix of monetisation × regulation × funding, run multiple
independent seeds, estimate Monte Carlo intervals, or propagate parameter
uncertainty.

## Market interference

Individual no-interference assumptions do not hold. A mechanism cap can change
spending and popularity, which changes the public board, switching, company
content and monetisation choices, audit signals, subsidy applications, and later
outcomes for players in both directly and indirectly affected games.
Collaboration and collusion create additional spillovers.

The primary output should therefore be interpreted as an equilibrium
market-regime contrast conditional on this model. Estimating separate direct and
spillover effects will require an additional design, such as game- or
jurisdiction-level assignment, explicit exposure mappings, network structure,
and prespecified cluster-level estimands.

## Paired outputs

`run_paired_worlds()` returns:

- run metadata and final outcomes for treated and control worlds;
- player-level differences in all seven harm dimensions;
- player spend and debt differences;
- firm operating-margin and cash differences;
- state subsidy-outlay differences;
- a `RegimeEffect` containing mean composite harm, total spend, total debt, total
  operating margin, total subsidy effect, and affected-player share.

Firm viability is currently represented by cash, operating margin, and
safe-revenue share. These are model outcomes, not audited accounts. A future
campaign must prospectively define the evaluation horizon, solvency threshold,
required content cadence, safe-revenue threshold, treatment of unpaid fines, and
social cost of public funds.

## Identification assumptions inside the simulator

A paired contrast has a clear model interpretation only if:

1. both branches begin from exactly the same pre-treatment state;
2. the intervention is the only intentional branch difference;
3. common random coordinates retain the same semantic meaning;
4. no treated value is used to initialise a post-treatment control variable;
5. outcome definitions are identical across branches;
6. the selected horizon is sufficient for the market feedback under study;
7. interference is included in the regime estimand rather than ignored;
8. the model equations and parameter statuses are reported with the result.

These conditions support internal simulation identification. They do not solve
external validity, calibration, measurement error, structural misspecification,
or real-world confounding.

## Planned campaign design

Before scientific execution, the project should add a preregistered factorial
or response-surface design covering:

- a declared neutral and several mechanism-specific monetisation regimes;
- audit intensity, accuracy, targeting, and enforcement regimes;
- funding budget, eligibility, scoring, and payment regimes;
- independent seed replications;
- jurisdiction-specific and vulnerable-subgroup contrasts;
- component-level, tail, and viability outcomes;
- sensitivity ranges for behavioural, information, rare-event, harm-weight, and
  institutional assumptions;
- uncertainty intervals and multiplicity rules.

Calibration targets and validation targets must be separated. Primary outcomes,
weights, stopping rules, exclusions, and the role of public funds in welfare
must be fixed before treatment results are inspected.

## Current interpretation boundary

No full campaign is authorised or run in this release. The smoke scenario only
checks software connections. Any effect produced with current inputs is a
conditional result of an illustrative model, not an empirical estimate.

See [Model specification](model_spec.md), [Data sources](data_sources.md), and
[Limitations](limitations.md) for the assumptions that currently block
scientific interpretation.
