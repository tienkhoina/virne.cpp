# Online RL environment and solver registry

Status: **COMPLETE for the documented CPU cluster**, 2026-09-10.
Authoritative report: `results/online_cluster_final_2026-09-10.json`.

## Scope and public API

`rl_core/online_environment.{h,cpp}` owns a `SolutionStepEnvironment` event
lifecycle and one active `InstanceEnvironment`. Step modes include placement,
joint placement/routing, physical-major node pairs, full node slots and detached
solutions. Each instance uses a private physical snapshot. At request completion
its solution journal is submitted to the lifecycle, which owns resource commit,
rollback, recorder state and departure release. A terminal observation has no
active instance/event; `request_done` is separate from whole-simulation `done`.

`solver_registry.{h,cpp}` registers 34 sequential source names in the existing
typed `SolverRegistry`. Names resolve once to SolverId/policy/algorithm and
environment options. `LearningSolver` implements the native Solver seam with
detached journals and exposes learn, evaluation, update and checkpoint methods.
Controller/Counter selections are validated instead of being silently ignored.

`online_learning.{h,cpp}` runs a complete prepared epoch using one registered
solver, commits each request and optionally flushes the pending rollout. The
caller controls reset/renew. Capture is optional so callers need not retain
per-request tensors or solutions when collecting aggregate metrics.

The graph-facing wrappers use the existing core ABI; the tensor-free session
interface keeps LibTorch linkage private. No global ABI switch, Python runtime,
new image or duplicate build directory is required.

## Source boundaries

The concrete Python online base classes leave observation/reward methods
abstract. The oracle supplies those hooks using the actual source feature and
reward calculators. Actual step, event lifecycle, Recorder, Counter and
Controller methods remain the references. The simulator is prepared on both
sides; numerical generation and result-file IO are outside this cluster.

Original `ppo_att` initializes PGSolver despite its name, so this registration
retains PG; `ppo_att+` selects PPO. Distributed A3C, safe-RL, recurrent registrations
and the legacy `a3n_gcn_mlp` registration are outside the 34-name catalog. They
are not registered as working placeholders. Existing core runtime/CLI training
configuration is a separate integration boundary. The epoch runner uses the
verified instance-agent learning semantics. OnlineAgent's separate admission
training loop across requests is also still on the full RL backlog.

## Grouped gate

`validate_online_cluster.py` runs the new online and registry harnesses together.
It collects failures per case before any behavior repair. The seven online
cases use three requests and six events, including two overlapping requests,
two departures before the last arrival, failure rollback and final release.
The registry gate checks every catalog descriptor and exercises four source
solver configurations plus a full epoch through the registered PPO solver.

The final gate passes **13 groups and 1,361 byte records**, with all 71 original
learning source files matching the inventory hashes. The seven environment
groups contribute 1,149 records; registry descriptors and five integration
groups contribute 212 records. The catalog covers 34 names. Additional native
assertions check external instance completion without counting rewards twice.

| Evidence | Result |
| --- | --- |
| `results/online_environment_complete_2026-09-10.json` | 7/7 environment cases PASS; earlier passing cases explicitly reused |
| `results/online_cluster_repaired_2026-09-10_solver_registry.json` | 6/6 registry groups PASS; descriptor and four solver groups reused from the initial gate |
| `results/online_cluster_final_2026-09-10.json` | Aggregate acceptance, source integrity and benchmark hashes |

The first grouped run found an incorrect native reset call: the wrapper called
`drain_leaves` on an arrival event. Later still-open cases exposed mismatched
node-slots/solution mask shapes and missing solution-step reward calculation.
The external-completion assertion caught double counting of rewards already
reported by `step`. These native issues were repaired only after their failing
cases; retries selected only unresolved cases and reused passing groups.

The original online Python node-pair path raises from a property without a
setter and also has invalid action decoding/candidate setup. Node-slots only
assigned a mapping without allocating/journaling node resources, causing a
KeyError during departure release. Repeated placement with `reusable=False`
reached an unsupported self-route. The oracle applies explicit in-memory glue
corrections for those paths, using the original Controller operations. Detached
solution submission calls the original `Controller.deploy` before the source
online transition. No original Python file was edited. The solution fixture
also stops its instance loop when `done`, respecting the native terminal guard.

## Frozen paired benchmarks

Both use CPU affinity `[2]`, one numeric/Torch thread, three warmups and five
samples. Each runtime retains one process for its warmups and measurements.
Output hashes match; these timings cover only the prepared three-request fixture.

| Workload | C++ median / p95 | Python median / p95 | Python/C++ |
| --- | ---: | ---: | ---: |
| Online reset + joint transitions + departure release | 0.872739 / 0.919962 ms | 42.594040 / 47.307972 ms | **48.805015x** |
| Registered PPO epoch + departures + final rollout flush | 10.439762 / 10.717543 ms | 107.996127 / 188.136213 ms | **10.344692x** |

Environment measurements are frozen in
`results/online_cluster_complete_2026-09-10_environment_benchmark.json`, with
output SHA256 `d570c5b29db934184ad68c827aadad05a61bc0af2ade644711e89a90711b620b`.
Registry measurements are frozen in
`results/online_cluster_final_2026-09-10_registry_benchmark.json`, with output
SHA256 `f2144a680c79d4ba4e5e2a31558439d05f7ecb47d6712028cfcea7e7517acd99`.

Initial registry/model/lifecycle construction, process startup and serialization
are excluded. A greedy solve constructs the lazy native model before timing on
both sides. Per-request environment construction remains part of the epoch.
Sequential order preserves shared RNG and resource mutations; no parallel
throughput gain is claimed. New MSVC/CUDA verification is outside this gate.

Two preliminary registry timing attempts had unequal setup/warmup scope and are
invalid measurements; they are retained as diagnostics. See
`results/online_registry_benchmark_scope_correction_2026-09-10.md`. Only that
invalid timing was corrected. The valid environment timing, old tests and all
other accepted benchmarks were not rerun.

## Build and reproduction provenance

The following documents the accepted workflow, not a request to rerun it:

```text
docker exec virne-rl-build cmake --build /work/build-ml-linux --target vne_online_environment_harness vne_learning_registry_harness -j 2
docker exec --env LD_LIBRARY_PATH= virne-rl-oracle python /workspace/virne.cpp/porting/validate_online_cluster.py --binary-dir /workspace/virne.cpp/build-ml-linux/porting --output /workspace/virne.cpp/porting/results/online_cluster_2026-09-10.json
```

Use `benchmark_online_cluster.py` only for a new or explicitly reopened workload;
the final benchmark rows are now frozen. It reuses passing component reports
and checks original sources, native outputs, and artifact hashes. Docker image
and mount details remain in `learning_session.md` and `oracle/README.md`.
