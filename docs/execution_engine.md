# Optimized exploratory execution engine

## Scope and status

This document describes the execution-only refactor for the exploratory
synthetic policy run. It does not change the model, the prospective or
exploratory analysis plans, the seed set, scenarios, estimands, population,
weights, money, sensitivity design, or stopping rules. The production campaign
configuration and its fail-closed gates remain separate.

The complete exploratory campaign has **not** been started. The measurements in
this document are bounded deterministic fixtures, not campaign results. They
say nothing empirical about real players or populations, and they do not make
`campaign_ready` true.

The machine-readable observations are in
[`benchmarks/execution-engine-bounded-2026-09-01.json`](benchmarks/execution-engine-bounded-2026-09-01.json).

## Execution boundary

The optimized path retains the CPU model as the reference implementation:

```text
configured execution identity
  -> explicit backend resolution and native-thread attestation
  -> immutable work plan
  -> bounded host scheduler
  -> CPU population, RNG, decisions, policy, money, and state transitions
  -> optional bounded GPU composite-harm reporting reduction
  -> coordinator-only atomic checkpoint commit
  -> deterministic reconstruction in declared seed/scenario/design order
  -> uncertainty, convergence, exploratory output, and final checksums
```

Independent main work is scheduled by **complete seed**. A main worker creates
one projected pre-treatment cohort and executes all seven scenarios in the
declared order. This preserves the common-cohort and common-random-number
contract. A sensitivity work item is exactly one `(parameter, level, seed)`
combination. Workers do not write checkpoints; the coordinator is the only
checkpoint writer.

## Backend selection and fail-closed behaviour

`ExecutionBackendConfig` recognizes three explicit modes:

| Requested mode | Resolution |
| --- | --- |
| `cpu` | NumPy CPU reference. It may not resolve to GPU. |
| `gpu` | CuPy/CUDA. Missing CuPy, CUDA, the requested device, or device metadata is an error; CPU is not substituted. |
| `auto` | An explicitly requested selection policy. It records whether GPU or CPU was selected and records the reason when no compatible GPU exists. |

The checked-in exploratory configuration currently requests `cpu`. Backend
identity records requested and resolved modes, implementation and version,
device index and name, compute capability, CUDA driver/runtime when applicable,
precision, batch limits, worker count, scheduling policy, native-thread
attestation, and the no-fallback policy. A change to this identity makes an
existing checkpoint incompatible.

The precision contract is `FLOAT64_STRICT_INTEGER_EXACT`. Mixed precision and
TF32 are not accepted. GPU batches are bounded simultaneously by the configured
item count, maximum batch bytes, and fraction of total device memory.

### Current GPU placement

Profiling identified the inner policy choice/RNG path as the real numerical
bottleneck. That path remains on CPU because moving the counter RNG, Gumbel
arithmetic, or `argmax` could change random-stream or categorical semantics.
The current GPU-capable kernel is therefore deliberately narrower: the
post-simulation six-component `composite_harm` reporting reduction. It uses
batched CuPy float64 matrix-vector operations. It does not receive RNG state,
money, action choices, or mutable model state. High-risk classification remains
on the CPU reference result.

This placement is scientifically conservative but means that a large whole-run
GPU speedup has not been established. A future device-resident transition
kernel would require a separate, much stronger categorical parity corpus and is
not part of this implementation.

## Determinism and numerical parity

The contracts are deliberately asymmetric:

- CPU reference and explicit CPU backend require bitwise equality.
- RNG words and streams, integer-cent values, scenario order, actions,
  treatment assignment, and categorical results remain exact CPU operations.
- A GPU composite may differ only as a continuous float64 reduction, with
  `absolute_tolerance = relative_tolerance = 5e-13`.
- Boolean classification at the declared composite threshold `0.35` must be
  exactly equal to the CPU reference; tolerance never masks a category change.
- The aggregate direction checked by the parity fixture must be unchanged.
- Every policy result is recomputed against the CPU composite contract. A GPU
  value outside tolerance or with a changed threshold classification is
  rejected.
- Every sensitivity work item stores both its selected-backend mean harm and
  its CPU-reference mean harm. The final monotonicity, coefficient-of-variation,
  and instability conclusions must match the CPU reference.
- The primary weighted estimand is reconstructed a second time from the CPU
  reference values. Convergence statuses and blockers, estimand direction, and
  the final sufficiency judgment must all match before publication.

Before dispatch, the selected backend runs a deterministic `257 x 6` parity
fixture. CPU must be bitwise equal. GPU must pass continuous tolerance,
threshold, and direction checks. The parity report and backend identity are
retained with the checkpoint-backed result.

There is no compatible GPU on the current host, so real GPU parity—including
end-to-end estimand and convergence-decision parity—has not been demonstrated
here. An explicit GPU request therefore fails before work starts on this host.
The conditional GPU test is skipped only because no device/runtime exists; a
separate test verifies the fail-closed error.

## Bounded host concurrency

The current exploratory contract declares:

| Control | Value |
| --- | ---: |
| Host executor | `BOUNDED_PROCESS_POOL_SPAWN` |
| Declared workers | 2 |
| Maximum in-flight units | 2 |
| Declared execution memory limit | 4,096 MiB |
| Estimated memory per worker | 1,024 MiB |
| Native numerical threads per worker | 1 |

The effective CPU worker count is the minimum of the declared worker count and
the memory-derived bound. The in-flight count cannot exceed that effective
count. CPU workers use an explicit `spawn` process context, including on
Windows. The coordinator transfers the pickle-safe static model context once
per worker; each submitted job contains only its seed or sensitivity-unit ID.
`ProfileBundle` and the resolved backend are deliberately not transferred
because their immutable proxy/module members are not pickle-safe. Workers
receive the already validated country-profile tuple and reconstruct the
backend from its configuration, then reject any backend or native-thread
identity mismatch. GPU mode forces one host worker to prevent competing CUDA
contexts.

NumPy on the current host links to OpenBLAS, which initially reported 24 native
threads. Environment variables alone are not accepted after NumPy has loaded.
Before the execution identity and checkpoint are opened, the engine locates the
single loaded NumPy OpenBLAS DLL, hashes it, calls its thread setter directly,
and verifies through its getter that the count is exactly one. The library
path/hash, getter/setter symbols, and enforced value enter the backend/checkpoint
identity. An unknown, ambiguous, or non-controllable native numerical runtime
fails closed instead of running an oversubscribed pool.

Futures can finish in wall-clock order, but completed payloads are held in a
bounded coordinator buffer and committed in immutable work-plan order. Workers
return only primitive, losslessly encoded checkpoint payloads and never receive
the checkpoint store or output paths. Completed units cannot be executed or
committed twice. Main results are decoded one seed at a time, so all 150 seeds'
player arrays need not coexist in memory.

## Checkpoint and resume contract

The resumable store uses schema `microtx_sim.resumable_checkpoint.v2`. Each
attempt has its own immutable directory and retains at least:

```text
progress/
  .attempt-NNNNNN.lock (coordinator lease while the attempt is open)
  attempt-NNNNNN/
    execution_identity.json
    work_plan.json
    execution_lineage.json
    checkpoint.json
    progress.json
    finalization-staging/
    finalization_attestation.json
    units/
      main_batch/
      sensitivity/
```

The execution identity binds the run and attempt IDs, implementation, source
tree, Git commit and branch, configuration, exploratory plan, seed set, work
plan, resolved backend/scheduler/native-thread runtime, Python runtime, and
dependency identity. Resume requires the exact expected identity, work plan,
and lineage. A changed source tree, configuration, plan, seed set, backend,
device, native thread library, interpreter, dependency lock, worker contract,
or attempt ID is rejected; it requires a new run/attempt identity.

### Atomicity and crash handling

- A complete main seed—all seven scenario payloads—is one atomic block.
- One `(parameter, level, seed)` sensitivity payload is one atomic block.
- Payload blocks are immutable and content-addressed.
- JSON is written to a temporary file in the destination directory, flushed
  and file-synchronized, then installed with atomic replacement. Directory
  synchronization is used where the operating system exposes it; on Windows,
  file synchronization precedes replacement.
- The checkpoint index becomes authoritative only after the payload block is
  durable. A crash may leave an unreferenced block, but an unreferenced or
  partially written block is never accepted as completed work.
- A nonblocking operating-system byte-range lease permits exactly one
  coordinator to mutate an attempt. The operating system releases the lease
  after a process crash.
- Loading verifies checkpoint identity, immutable files, payload hashes,
  declared descriptors, absence of duplicate units, and all-or-none main-seed
  blocks.
- Final artifacts are first written into the attempt staging directory and
  bound by a content-addressed finalization attestation. Publication can resume
  byte-for-byte after interruption; unattested, altered, symlinked, or partial
  bundles are rejected.

On resume, verified completed units remain complete. Units that were merely
in-flight are cleared and requeued. A completed unit cannot be begun or
committed again. `COMPLETE` is available only after every declared work unit is
committed and final output checksums are supplied.

Checkpoint result payloads use
`microtx_sim.policy_and_sensitivity_checkpoint_payload.v2`. Arrays are encoded
losslessly with explicit little-endian dtype, shape, uncompressed byte count,
compression and checksum metadata. Internal `simulation_cents` values are
retained only inside verified checkpoints and are labelled
`INTERNAL_MODEL_UNIT_NOT_REAL_MONEY`; raw cross-country pooling is prohibited.
The configured partial-result profile therefore describes attested internal
model state, not merely the old unweighted diagnostic CSV, and explicitly
prohibits treating checkpoint contents as monetary or estimand outputs.

The old `attempt-000001` used the earlier non-resumable checkpoint shape. It is
preserved as an interrupted execution lineage artifact. Its v1 progress file
records all 150 main seeds/1,050 seed-scenario diagnostic rows, but records no
sensitivity completion and no final outputs; its own `resume_supported` value
is false. It is not silently upgraded or treated as a valid v2 resume point.

## Exact progress semantics

Progress is based only on committed declared work units. Elapsed time and
memory use never enter the numerator.

For the checked-in exploratory design:

| Phase | Exact denominator |
| --- | ---: |
| Main batch | `150 seeds x 7 scenarios = 1,050` units |
| Sensitivity | `5 parameters x 3 levels x 150 seeds = 2,250` units |
| Overall | `1,050 + 2,250 = 3,300` units |

A main seed is committed atomically, so successful completion of one seed adds
seven main units. During execution of that seed the completed numerator remains
at the preceding committed value, while its unit IDs are listed as in progress.
A sensitivity commit adds one unit.

`progress.json` reports overall, main, and sensitivity counts; remaining units;
a floating display rounded to six decimal places; and an exact reduced rational
percentage `{numerator, denominator, unit="percent"}`. It also records current
phase, parameter, level, seed, scenario, completed seed IDs, remaining units,
checkpoint/payload hashes, identity hashes, timestamps, status, error, resume
count, and output paths. Console progress formats the same completed/total
counts. ETA is not used as a completion measure.

## Bounded profiling and benchmarks

The baseline profile was collected before the execution refactor on commit
`cc2cfc755124d2ac036e479c6b4f9b4b8fa9668e`. No CLI campaign command was used.

| Deterministic fixture | Observation |
| --- | --- |
| 256 players, 1 day, 120-minute steps, 1 seed, all 7 scenarios | 0.128 s total; `run_policy_scenario` 0.106 s; `advance_policy_day` 0.100 s; choice 0.078 s; uniform RNG 0.074 s; counter words 0.066 s; SplitMix64 0.043 s. |
| 2,000 players, 14 days, 60-minute steps, baseline scenario | 0.617 s; choice 73%, uniform RNG 66%, counter words 57%, SplitMix64 40%. |
| 20,000 players, 2 days, baseline scenario | 0.391 s; choice 66%, RNG 48%, SplitMix64 34%, `argmax` 3.6%, exact revenue summation 2.3%. |

One-day/30-minute-step scaling observations were 0.0533 s for 1,000 players,
0.1233 s for 5,000, and 0.3341 s for 20,000. These corresponded to about 0.90,
1.95, and 2.87 million player-steps per second on this host. They are local
microbenchmarks, not campaign runtime forecasts.

After OpenBLAS was directly restricted from 24 threads to one, a 2,000-player,
2-day, 60-minute-step baseline fixture was repeated three times. Median CPU
reference time was 0.0711724 s and median explicit-CPU-backend time was
0.0711069 s. Composite harm, high-risk flags, spending, and action minutes were
bitwise equal in every repetition. The nearly equal timings demonstrate that
the explicit CPU abstraction did not add material overhead on this tiny
fixture; they are not evidence of a statistically established speedup.

A separate Windows-`spawn` fixture executed four complete seeds, all seven
scenarios, 10,000 players, four days, and 60-minute steps. It included lossless
checkpoint-payload encoding and return to the coordinator. Serial execution
took 11.8993002 s; a cold two-process pool took 7.8696755 s, a 1.5120x local
speedup. The serial and process-pool payloads were bitwise identical and shared
SHA-256 `bf043b9367368e1b7134a5a9db68471d534b0b19c43f391885f26169fcdee2c6`.
The timing fixture used the single-UK synthetic initializer, not the projected
campaign population. This bounded observation includes process startup and IPC
but is not a campaign runtime forecast.

The memory probe extrapolated approximately 12.0 MiB for a 50,000-player cohort
plus life table, 11.8 MiB for mutable state, and 12.3 MiB per scenario result.
Seven retained scenario results are therefore roughly 86 MiB per seed, and
retaining all 150 seed payloads in memory would be roughly 12.6 GiB before
Python overhead. This is why the optimized runner checkpoints and later decodes
one seed at a time.

## Remaining limitations

- The current host has no detected NVIDIA/CUDA or AMD ROCm tooling, and no
  CuPy, Numba, JAX, or Torch installation. GPU execution and speed are therefore
  unmeasured here.
- The only GPU-eligible operation is a post-simulation reporting reduction, not
  the profiled policy-choice/RNG bottleneck. A whole-run GPU speedup must not be
  claimed from this implementation.
- Process-pool speedup has not been benchmarked at the full 50,000-player,
  150-seed design. The bounded two-process fixture and checkpoint semantics are
  tested, but full-design performance and capacity still require an explicitly
  approved run.
- Memory figures are bounded-fixture extrapolations, not peak resident-set
  measurements of the complete campaign.
- Numerical parity infrastructure does not supply empirical validity,
  calibration, population representativeness, monetary calibration, or
  campaign readiness.
- The configured persistent SQLite ledger remains a declared configuration
  contract and is recorded in final execution metadata. The resumable policy
  executor's authoritative intermediate state is the content-addressed
  checkpoint store; it does not reinterpret checkpointed model cents as ledger
  transactions.

No full exploratory or production campaign was run while producing this
documentation.
