# Roadmap

The repository now provides a stable, tested synthetic prototype: the market
ABM, policy time-allocation engine, seven-scenario repeated-seed batch, EPGC
financing, sensitivity grid, and versioned outputs are executable. It is
**not calibrated for substantive estimates**. The roadmap therefore treats
calibration and empirical identification as prerequisites, not as polish.

## Completed for the synthetic prototype

- heterogeneous seeded player and welfare state;
- explicit fourteen-coordinate monetisation intervention vector;
- full eight-action daily decision process with exact time conservation;
- six-component harm and adult/youth opportunity-cost proxies;
- all seven named counterfactual scenarios with common cohorts and random fields;
- repeated-seed variance and normal Monte Carlo intervals;
- EPGC safe-profit equation, capped payments, bonuses, penalties, clawbacks,
  and minimum contribution;
- one-at-a-time sensitivity, monotonic checks, and instability flags;
- CSV/JSON/Markdown/SVG outputs with configuration, seed, cohort, source, and
  environment metadata;
- unit and integration coverage for zero-player, zero-cost, extreme, seed,
  cap, cooling-off, EPGC, batch, output, and CLI cases.

## Completion criteria for a research campaign

A campaign should not start until all of the following are true:

- every parameter that can materially change an estimand is either calibrated,
  estimated, or declared as a sensitivity parameter with a defensible range;
- population weights reproduce the intended age, income, household, gaming, and
  payer populations in each jurisdiction;
- currency, price-period, and population-base contracts are coherent across
  inputs and outputs;
- observational targets are held out for validation rather than reused only for
  calibration;
- the intervention matrix, primary estimand, harm weights, stopping rule, and
  exclusion rules are specified before inspecting treatment results;
- repeated-seed uncertainty, sensitivity analysis, and convergence diagnostics
  are implemented;
- campaign provenance validation passes without synthetic or illustrative
  dependencies.

## P0 — Evidence, units, and calibration

These tasks block scientific interpretation.

1. **Build comparable population profiles.** Replace equal synthetic country
   weights and illustrative age distributions with explicit target-population
   weights. Add joint age–income–household distributions rather than assuming
   that marginal anchors are sufficient.

   **Infrastructure progress only:** population-evidence bundle schema version
   1 now attests exact CSV bytes and strict joint age × household-income-band ×
   household-type × gaming × pre-treatment payer-history cells with exact
   reduced rational mass. Registered profile-input lineage version 4 retains
   the bundle, verified extraction, and independent readiness blockers while
   versions 1–3 remain readable. The default bundle is empty, `ILLUSTRATIVE`,
   unsigned, and `campaign_ready=false`; hashes prove reproducibility, not
   publisher authenticity or calibration.

   A separate static population-design schema now binds exact evidence-result
   identities to complete domains, declared calibration/validation source-unit
   partitions and target counts, and implements deterministic exact-rational
   Hamilton allocation with analysis and expansion weights. The checked-in
   design is complete but `ILLUSTRATIVE`. Its record and cluster identities are
   declarations, not authenticity or independently held-out proof, so static
   schema v1 remains `campaign_ready=false` even when declarations are complete.

   The executable plumbing is now implemented as an opt-in path. A strict,
   content-addressed mapping declares the source-household-income/type to
   runtime-personal-income/household conversion. Its adapter re-attests the
   static `PopulationApportionmentPlan`, consumes its exact counts and rational
   weights without a second allocation, and binds the resulting projection and
   ordered player assignment. Market and policy configurations can select it;
   omission preserves the legacy marginal generator. The blocked campaign
   candidate selects the checked-in schema-v2 mapping; development defaults do not.

   Policy batch and sensitivity execution now retain exact per-seed projection,
   cohort, weight, and pre-treatment population-balance lineage. The balance
   report covers planned versus realized counts and mass for every full joint
   cell and separately attests runtime jurisdiction, age, income, and household
   membership. Gamer and payer-history fields remain sidecar-only. A dedicated
   standalone `target_population_estimands` writer/profile can serialize exact
   weighted means, paired differences, and weighted quantiles; it does not alter
   or join the automatic output-v3 bundle.

   An additional opt-in prospective-plan composition layer now freezes exactly
   one `PRIMARY` estimand specification, explicit scenario direction, fixed
   seeds, harm weights, output/metric identities, and an executable
   pre-treatment population predicate. It validates expected inputs before
   scenario execution and binds exact per-seed realizations afterward to
   population weights and balance lineage in a separate output profile. It does
   not define a run-level primary aggregate or Monte Carlo uncertainty summary.
   Declared period dates are not executed calendar anchors; only their inclusive
   duration is checked against the batch horizon, with one declared day for a
   zero-day structural snapshot. No plan is checked in; schema v1 is necessarily
   unregistered and remains `campaign_ready=false`. A money-valued planned
   estimand must additionally select the opt-in prospective money execution and
   otherwise fails before treatment. That execution creates target-currency-
   equivalent model amounts by applying one jurisdiction-specific composite to
   each retained observation before contrasts and weights, then rounding once to
   the nearest signed minor unit with half ties away from zero. It does not
   reconstruct nominal local currency, and only the separate prospective profile
   receives the converted values and lineage. Legacy root output-v3 remains in
   unchanged simulation cents. Calendar anchoring, cross-seed aggregation and
   uncertainty, and model implementation/environment identity remain explicit
   campaign blockers.

   This P0 item remains open because exact execution does not provide an
   authenticated and calibrated target, an independently verified holdout, a
   scientifically reviewed source-to-runtime conversion, preregistration, or a
   campaign-ready output contract. Output-v3 CSV columns retain their unweighted
   semantics, `public_population_comparability` and `campaign_ready` remain
   false, and this milestone neither authorises nor constitutes a full campaign.
2. **Resolve monetary comparability.** Keep source currencies during ingestion,
   then define dated exchange-rate or purchasing-power contracts for any pooled
   estimand. Never treat nominal GBP, KRW, JPY, and EUR minor units as directly
   comparable simulation cents.

   Profile schema version 2 implements the typed declaration-only exact-rate contract,
   method-specific source and typed-date validation, explicit aggregation-stage
   rounding, registered-file re-attestation, lossless decimal-string rate
   mirrors, lineage/manifest retention, complete-coverage gate, common-basis
   check, and exact internal-scale coherence check. No rates are checked in and
   the present bundle fails with a missing contract for every jurisdiction, so
   this P0 item remains open pending reviewed calibrated evidence, a declared
   target estimand, and binding to the preregistered campaign population. The
   public comparability claim and campaign gate remain false with an explicit
   `source_rate_binding=missing` blocker even when test-only algebra clears the
   structural checks. Version 3 now additionally requires stable conversion and
   rate-binding IDs and verifies an exact rational from content-addressed CSV
   bytes through one canonical declarative recipe. Registered profile-input
   lineage version 4 re-attests the monetary source catalogue, bundle,
   artifact, recipe, and extraction and also carries the independent population
   assessment; lineage versions 1–3 remain readable and exercise the migration
   boundary. The checked-in
   source bundle is empty, illustrative, and missing a signature. Even a valid
   test extraction clears only the source subgate: signature, output/design,
   population, and external-preregistration blockers remain independent.

   **Infrastructure progress only:** an opt-in prospective execution can now bind
   a complete dated target-currency basis and exact jurisdiction-specific
   composites to a money-valued planned estimand. It converts each retained
   simulation-cent observation before the paired contrast and target-population
   weighting, using exactly one signed half-away-from-zero minor-unit rounding
   and no intermediate nominal local-currency reconstruction. These outputs are
   target-currency-equivalent **model** amounts and remain confined to the
   separate prospective profile; legacy root outputs remain unchanged simulation
   cents. No supplied plan or checked-in conversion selects this path. Source
   authentication, calibrated money scales and behavioral spending inputs,
   representative population and independently held-out validation, external
   preregistration, and campaign readiness all remain blocking, so P0.2 remains
   open and no campaign is authorised.
3. **Calibrate participation and spending.** Estimate gaming participation,
   payer conversion including zero spenders, conditional spend distributions,
   price sensitivity, switching, churn, and time allocation by age and income.
   Validate both the body and upper tail of spending.
4. **Calibrate vulnerability and harm transitions.** Link pre-treatment traits,
   baseline vulnerability, and the seven dynamic harm dimensions to measured
   constructs. Treat every psychological equation as a model hypothesis, not as
   a clinical diagnosis.
5. **Identify rare-event hazards.** Case reports establish possibility but not
   prevalence. Estimate stored-card exposure, guardian consent and supervision,
   unauthorised-use incidence, transaction size, detection, and reporting as
   separate quantities.
6. **Calibrate company responses.** Obtain targets for update cadence, content
   cost, acquisition cost, monetisation changes, compliance spending, expected
   penalties, research value, and bilateral agreements. Validate market shares,
   revenue, retention, and release timing out of sample.
7. **Operationalise regulation and funding.** Encode rule effective dates,
   coverage, enforcement capacity, fine collection, subsidy eligibility,
   certification, and payment timing. Distinguish law on the books from observed
   enforcement.
8. **Promote provenance contracts.** Assign every derived metric a transformation
   recipe, source version, retrieval date, population base, unit, period, and
   status. Only then promote an input from `ILLUSTRATIVE` to `CALIBRATED`.

## P1 — Empirical causal design and uncertainty

1. Extend the implemented seven-regime policy batch into a preregistered
   empirical factorial design spanning market audit and financing interventions.
2. Add calibrated between-jurisdiction heterogeneity, convergence diagnostics,
   joint parameter uncertainty, and externally justified interval semantics.
3. Extend the implemented OAT grid to rare-event priors, harm weights,
   observation noise, behavioural persistence, company expectations, structural
   alternatives, and unobserved-confounding assumptions.
4. Separate direct exposure effects from market-equilibrium spillovers through
   game- or jurisdiction-level assignment, recorded exposure mappings, or other
   designs that acknowledge interference.
5. Add mechanism-specific contrasts and mediation analyses without conditioning
   on post-treatment variables by accident.
6. Define economic viability prospectively: solvency horizon, operating margin,
   safe-revenue share, content cadence, player reach, and the treatment of unpaid
   fines and public support.
7. Add pre-treatment balance checks, null-intervention identity checks, negative
   controls, and recovery tests on synthetic data with known effects.

## P1 — Market and institutional fidelity

1. Allow players to use multiple games, retain longitudinal histories, and form
   social or household networks. Discovery, imitation, and competitive status
   can then generate network spillovers instead of relying only on aggregate
   rankings.

   A first synthetic slice is now wired into the strategic `World`: lagged
   leave-one-out game use by other simulated players in the same household can
   affect discovery and utility through an explicit sensitivity-only
   coefficient. This does not yet provide simultaneous multi-game portfolios,
   a general social graph, calibrated peer effects, or competitive-status
   transmission, so the roadmap item remains open.
2. Model refunds, chargebacks, complaints, parental disputes, account suspension,
   and remediation after unauthorised spending.
3. Add advertising, subscriptions, battle passes, direct cosmetics, app-store
   fees, and platform policies so viable non-compulsive business models have
   realistic alternatives.
4. Add firm entry, exit, insolvency, portfolio reallocation, and heterogeneous
   access to finance. Public funding should affect real production constraints,
   not only terminal cash.
5. Expand collaboration and collusion into persistent agreements with monitoring,
   defection, detection, and enforcement. Keep agreement formation endogenous
   and require reciprocal compatible proposals.
6. Represent regulator learning, rule changes, appeal, litigation, collection
   delays, cross-border activity, and coordination between authorities.
7. Model a priced information market: research quality, vendor incentives,
   selective disclosure, signal decay, and asymmetric access.

## P2 — Validation, outputs, and reproducibility

1. Evolve the implemented output schema and manifest through explicit migration
   tests; add signed immutable source bundles and bind a canonical model
   implementation, interpreter, dependency lock, and build environment into
   prospective plans and results. The existing lock file alone does not supply
   the missing plan/result execution-identity attestation.
2. Add streaming or columnar storage for campaign-scale outputs. The current
   aggregate and optional synthetic individual CSV writers are sufficient for
   the prototype, but real microdata would require disclosure-risk controls.
3. Add empirical posterior-predictive or holdout checks for spending, retention,
   ranking mobility, firm actions, audit findings, and subsidy awards.
4. Extend the existing unit/integration coverage with property tests, broader
   overflow boundaries, schema migration tests, and empirical campaign fixtures.

   Policy revenue composition now uses Python-integer aggregation before the
   signed-64-bit storage check, with exact-boundary and prior wrap-to-zero
   regression tests across purchase, fixed-price, and subscription channels.
   Property generation, migration fixtures, and empirical fixtures remain open.
5. Add continuous integration for supported Python versions, documentation links,
   unit tests, and a very small deterministic smoke run.

## P2 — Performance without changing the estimand

An explicit full/final-only `WorldStep` retention contract now removes the
`O(T·P)` history term for the future-scale baseline with byte-exact outcome and
paired-effect equality tests. An exact SQLite ledger now also batches transfers,
uses indexed binary-unique references, bounds Python-retained history, and
finalizes persistent ledger artifacts with logical and raw-file digests. These are
prerequisites, not the planned benchmark: disk growth, smaller tick histories,
and the complete world still require representative measurement.

1. Benchmark the planned 50,000-player, 8-game scenario before optimisation and
   record peak memory, time per phase, and ledger growth.
2. Preserve exact all-game choice evaluation. Tune block size, reuse temporary
   arrays, and remove avoidable allocations rather than sampling alternatives.
3. Replace repeated player-by-game and player-by-firm masks with exact grouped
   indices or scatter operations, with equality tests against the reference
   implementation.
4. **Implemented for the synthetic prototype:** batch ledger construction while
   retaining integer-cent reconciliation and unique references. Keep backend
   equivalence, failure rollback, seal verification, and memory-retention tests
   as regression gates.
5. Parallelise independent worlds or replications, not stateful fragments of one
   world. Counter-based streams make this safe when coordinate contracts remain
   unchanged.
6. Add checkpoints and restart manifests only after their byte-for-byte
   reproducibility contract is tested.

## Deferred questions

Several choices should remain explicit research decisions rather than silent
implementation defaults:

- whether the primary estimand targets all mobile-game users, payers, minors, or
  the general population;
- how outcomes denominated in different currencies enter a pooled welfare
  function;
- which harm weights, if any, are appropriate for a primary composite outcome;
- whether public funding is evaluated as a budget-neutral reallocation, an
  external transfer, or a social-cost term;
- how long a company must remain solvent and how much safe content it must
  produce to count as economically viable;
- which forms of monetisation constitute the neutral counterfactual while still
  allowing a functioning competitive game.
