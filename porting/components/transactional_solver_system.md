# API: transactional solver/system integration

State: **IMPLEMENTED** on 2026-07-30. This is an additive integration seam;
the frozen `solve(const SolverInstance&)` behavior and component benchmarks
remain unchanged.

## API

```cpp
enum class SolverMutationState : std::uint8_t { detached, committed };

struct MutableSolverInstance {
    const network::VirtualNetwork& virtual_network;
    network::PhysicalNetwork& physical_network;
    core::controller::PreparedControllerMutation& mutation;
};

struct MutableSolverResult {
    core::Solution solution;
    SolverMutationState mutation_state;
};

virtual MutableSolverResult Solver::solve_mutable(
    const MutableSolverInstance&);

PreparedControllerMutation Controller::prepare_mutation(
    const VirtualNetwork&, PhysicalNetwork&) const;

EnvironmentSolverTransaction BaseEnvironment::solver_transaction();

EnvironmentStepResult SolutionStepEnvironment::step(
    Solution&, EnvironmentSolutionState = EnvironmentSolutionState::detached);
```

`OnlineSystem` and `TimeWindowSystem` obtain one typed transaction from the
current Environment and call `solve_mutable`. `OfflineSystem` keeps one working
physical clone per run, starts the same transaction for each independent
request, then always restores the immutable snapshot. The base implementation
is a backward-compatible adapter over const `solve` and returns `detached`.
`BaseNodeRankSolver`, `RandomRankSolver`, and `CandidateRankSolver` (PL/NEA)
override it: they rank before mutation, map directly on the Environment p-net,
and return `committed` only after complete placement and routing.

## Mutation and rollback contract

`PreparedControllerMutation` binds resource names once, retains compact
resource IDs, and prepares only `ResourceUpdator`; it does not construct a
constraint checker, NodeMapper, LinkMapper, or TopologyAnalyzer. Environment
stores one such view per request instead of a full `PreparedController`.

`begin_transaction()` snapshots only the selected physical resource slots by
their already-bound `AttrId`. Rollback assigns the original `AttrValue` back;
it never adds demand to a subtracted `double`, so restoration is bit-exact and
also covers an exception between physical mutation and insertion into a
Solution table. The fixed Solution tables remain the observable partial-output
journal and are not modified by rollback.

Environment rolls the checkpoint back on solver exceptions, admission or
mapping-validation rejection, detached-deploy failure, and Recorder/add-record
failure. It commits the checkpoint only after the arrival record and release
index exist. On success it skips deployment for `committed`, while a
`detached` fallback is deployed exactly once. Leave processing uses the
lightweight view to release resources.

Arrival counting calls `PreparedCounter::count_solution` once and forwards the
precomputed fixed Solution fields to `Recorder::count_precomputed_arrival`;
Recorder no longer repeats the virtual-network count.

## Performance rules

- no physical-network clone on the mutable node-rank path; the transaction
  copies selected resource values only, without graph topology/factories;
- offline mode clones the physical snapshot once per run, not once per
  request;
- no second successful deploy;
- no eager per-request mapper/topology preparation in Environment;
- no string lookup in rank, map, mutation, rollback, deploy, release, or event
  loops;
- configured worker widths and deterministic mutation/RNG order are retained.

## Verification

The default `ffd_rank`, seed-0 Main case is exact at 100 physical nodes, 528
links and 1,000 requests: Python and C++ both accept 752 requests and produce
`long_term_r2c_ratio=0.48211107669531345`. The one integration signal measured
51.994x faster C++ system execution in the GCC 11 container. Commands, raw
times, scope and focused test evidence are in
`../results/system_transaction_integration_2026-07-30.md`.
