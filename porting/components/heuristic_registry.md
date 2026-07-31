# Component API: complete non-ML heuristic registry

State: **COMPLETE / COLLECTIVE NATIVE GATE PASS** on 2026-07-30.

The registry covers every canonical decorated solver in Python's
`solver/heuristic` sources. The eight node-rank solvers retain their frozen
cross-language evidence; BFS and joint place-route reuse the completed Graph,
NodeRank, Controller, Solution and random APIs.

| Fixed ID | Name | Category |
|---:|---|---|
| 0 | `order_rank` | `node_ranking` |
| 1 | `random_rank` | `node_ranking` |
| 2 | `grc_rank` | `node_ranking` |
| 3 | `ffd_rank` | `node_ranking` |
| 4 | `nrm_rank` | `node_ranking` |
| 5 | `pl_rank` | `node_ranking` |
| 6 | `nea_rank` | `node_ranking` |
| 7 | `rw_rank` | `node_ranking` |
| 8 | `order_rank_bfs` | `heuristic` |
| 9 | `random_rank_bfs` | `heuristic` |
| 10 | `rw_rank_bfs` | `heuristic` |
| 11 | `random_joint_pr` | `heuristic` |
| 12 | `order_joint_pr` | `heuristic` |
| 13 | `ffd_joint_pr` | `heuristic` |

`ego_network.py` and `fit.py` are unfinished, undecorated prototypes without a
canonical solver name or complete `solve`; they are not compatibility registry
entries. The orphan C++ stubs `nea_bc`, `nea_par`, `nrm_bc_env` and
`nrm_par_solver` have no Python provenance. Meta-heuristics, exact solvers,
MCF/OR-Tools and ML/RL remain separate scope.

## Public API

All C++ names below are in `virne::solver::heuristic`.

```cpp
struct HeuristicSolverRegistryOptions {
    NodeRankSolverWorkers workers;
    GRCRankSolverParameters grc;
    RandomWalkRankSolverParameters random_walk;
    BfsSolverParameters bfs;
};

struct HeuristicSolverIds {
    SolverId order_rank, random_rank, grc_rank, ffd_rank;
    SolverId nrm_rank, pl_rank, nea_rank, rw_rank;
    SolverId order_rank_bfs, random_rank_bfs, rw_rank_bfs;
    SolverId random_joint_pr, order_joint_pr, ffd_joint_pr;
};

HeuristicSolverIds register_heuristic_solvers(
    SolverRegistry&,
    NumpyRandomState&,
    PyRandom&,
    HeuristicSolverRegistryOptions = {});
```

Every returned ID is a direct field. Solver names are resolved once at the
cold registry boundary; request loops retain only `SolverId`, enums, vertices,
attribute IDs and direct slots.

The BFS API exposes `BfsSolverParameters{max_visit, max_depth,
shortest_method, k_shortest}`, the three concrete solver classes and their
`register_*` functions. Random BFS borrows the caller's `NumpyRandomState`.
Order BFS preserves Python's `100/10/k=10` deploy defaults; all three preserve
the original sequential queue/attempt/undo order. Ranking and completed
constraint/path leaves receive the four caller-controlled worker fields.

The joint API exposes `JointPRStrategy`, typed `JointPRException`,
`RandomJointPRSolver`, `OrderJointPRSolver`, `FFDJointPRSolver` and their
registration functions. Random selection borrows `PyRandom`; FFD reranks the
live physical network for each placement. Candidate sets retain CPython 3.10
integer-set order. Link aggregation is evaluated with numeric resource IDs;
the Python Controller's observable bug that computes and then ignores its
link-suitable set is retained. Joint routing always uses `k=1`.

Both families support `solve_mutable`. Success commits the live mutation;
failure or exception restores the transaction while retaining the partial
`Solution` journal for diagnostics.

## Main configuration

Use any table name through `solver.solver_name`. The four existing worker
overrides apply to every family:

```text
++native.workers.rank=4
++native.workers.node_candidate=4
++native.workers.link_topology_constraint=4
++native.workers.link_candidate=4
```

Python Main constructs BFS without forwarding `config.solver` as keyword
arguments, so its effective BFS defaults are always
`50/5/bfs_shortest/10`. C++ Main matches that behavior. Optional C++-only
overrides are isolated under `native.bfs.{max_visit,max_depth,
shortest_method,k_shortest}` and do not enter the Python-compatible config
tree.

## Validation and performance

One collective CTest, `vne_heuristic_all_unit`, enumerates all 14 fixed fields,
locks ID/name/category/order, creates and solves every factory on a shared
feasible graph, verifies const input immutability, compares exact Solution
output and NumPy/Python RNG continuation at workers `1` and `4`, and checks
partial-failure rollback for one BFS and one joint solver.

The Docker GCC Release gate is **1/1 PASS**. Its single catalog timing signal
was 70.8778 ms at workers=1 and 36.7232 ms at workers=4, a 1.930x speedup.
This is a compact integration signal, not a new frozen cross-language
benchmark. The accepted Python/C++ timings for the eight node-rank solvers
remain frozen in `heuristic_node_rank_variants_*_2026-07-30.json`; they range
from 1.174x to 75.990x at workers=1 and were not rerun.

## Hot-path overhead audit (2026-07-31)

No API or registry ID changed. BFS now uses contiguous reserved FIFO storage
and one reusable incident-neighbor prefix buffer per traversal. Joint PR reuses
constraint-request/feasible-node storage across virtual nodes, maintains the
selected-node insertion order incrementally, and uses generation-stamped FFD
candidate membership instead of allocating and clearing a physical-node mask
for each placement. FFD's graph-local rank preparation is also bound lazily at
its original first-use point and reused while each subsequent rank observes
the mutated resource values. The ignored Python link filter still performs
its aggregation and shape/error checks; only the subsequently pure,
non-throwing materialization of the discarded result is omitted. Queue order,
CPython set order, RNG consumption, exact first errors, mutation order and
caller worker settings remain unchanged. Frozen benchmarks were not rerun.
