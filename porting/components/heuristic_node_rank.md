# Component API: `solver.heuristic.node_rank`

State: **COMPLETE / FROZEN ALGORITHMS** on 2026-07-30, with an additive mutable
System integration seam. Frozen solver differentials and benchmarks are
unchanged.

Python oracle: `../virne/virne/solver/heuristic/node_rank.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`44A6F63F1A1798935453C057C6981A692F5DC4F03BDE6A6535607AECBDC61389`.
The frozen scope is `BaseNodeRankSolver` plus every concrete class in the
Python file: `order_rank`, `random_rank`, `grc_rank`, `ffd_rank`, `nrm_rank`,
`pl_rank`, `nea_rank`, and `rw_rank`. BFS and joint place-route are completed
and documented separately in `heuristic_registry.md`. MCF/OR-Tools,
meta-heuristics, ML and RL remain separate leaves.

## Public API

All names are in `virne::solver::heuristic`.

```cpp
struct NodeRankSolverWorkers {
    std::size_t rank_workers = 1U;
    std::size_t node_candidate_workers = 1U;
    std::size_t link_topology_constraint_workers = 1U;
    std::size_t link_candidate_workers = 1U;
};

class BaseNodeRankSolver : public Solver {
protected:
    BaseNodeRankSolver(
        SolverDependencies,
        SolverConfig,
        rank::NodeRankMethod,
        rank::NodeRankParameters = {},
        NodeRankSolverWorkers = {});

public:
    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
    rank::NodeRankMethod node_rank_method() const noexcept;
    const NodeRankSolverWorkers& workers() const noexcept;
};

class OrderRankSolver final : public BaseNodeRankSolver {
public:
    OrderRankSolver(
        SolverDependencies,
        SolverConfig,
        NodeRankSolverWorkers = {});
};

class FFDRankSolver final : public BaseNodeRankSolver {
public:
    FFDRankSolver(SolverDependencies, SolverConfig,
                  NodeRankSolverWorkers = {});
};

class NRMRankSolver final : public BaseNodeRankSolver {
public:
    NRMRankSolver(SolverDependencies, SolverConfig,
                  NodeRankSolverWorkers = {});
};

struct GRCRankSolverParameters {
    double sigma = 1.0e-5;
    double damping = 0.85;
};

class GRCRankSolver final : public BaseNodeRankSolver {
public:
    GRCRankSolver(SolverDependencies, SolverConfig,
                  GRCRankSolverParameters = {},
                  NodeRankSolverWorkers = {});
};

struct RandomWalkRankSolverParameters {
    double sigma = 1.0e-4;
    double jump_probability = 0.15;
    double forwarding_probability = 0.85;
};

class RandomWalkRankSolver final : public BaseNodeRankSolver {
public:
    RandomWalkRankSolver(SolverDependencies, SolverConfig,
                         RandomWalkRankSolverParameters = {},
                         NodeRankSolverWorkers = {});
};

class RandomRankSolver final : public Solver {
public:
    RandomRankSolver(SolverDependencies, SolverConfig,
                     NumpyRandomState&,
                     NodeRankSolverWorkers = {});
    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
};

enum class CandidateRankStrategy : std::uint8_t {
    proximity, essentiality
};

class CandidateRankSolver : public Solver {
public:
    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
    CandidateRankStrategy strategy() const noexcept;
};

class PLRankSolver final : public CandidateRankSolver {
public:
    PLRankSolver(SolverDependencies, SolverConfig,
                 NodeRankSolverWorkers = {});
};

class NEARankSolver final : public CandidateRankSolver {
public:
    NEARankSolver(SolverDependencies, SolverConfig,
                  NodeRankSolverWorkers = {});
};

SolverId register_order_rank_solver(
    SolverRegistry&,
    NodeRankSolverWorkers = {});

// Equivalent register_* functions exist for ffd/random/grc/nrm/pl/nea/rw.
```

Standard solvers fix their typed `NodeRankMethod` and parameters. Random rank
borrows an explicit caller-owned `NumpyRandomState`; it has no hidden seed or
global RNG. PL/NEA use typed `CandidateRankStrategy` and prepared constraint
IDs. Registration captures workers/parameters by value and confines each
dynamic compatibility name to the cold registry boundary.

## Solve and performance contract

Construction reads the Controller's fixed typed selection once and retains
NodeMapper/LinkMapper configured with numeric attribute and constraint IDs.
No string map is used for fixed fields or in a hot loop.

The compatibility `solve(const SolverInstance&)` creates
`Solution::from_v_net`, ranks in Python order, clones the physical network once,
then maps nodes and live links on that clone. `solve_mutable` preserves the same
ranking/mapping/error order but maps directly on the caller's prepared physical
network. It returns `committed` only after complete success; failure and
exceptions restore the recorded journal, while the owning Environment exact
checkpoint covers mutations that precede journal insertion. Partial Solution
output and dependency exception order are retained.

Node mapping always uses `reusable=false` and `inplace=true`. Ranking receives
`rank_workers`; mapping receives the other three worker fields together with
the typed `SolverConfig` matching/shortest-path settings. Widths `0/1` select
the dependency's sequential path. Wider widths are caller configuration only:
there is no automatic tuning, and solve phases remain deterministically
ordered.

PL/NEA retain Python's candidate traversal, including CPython 3.10 integer-set
difference order for sparse physical IDs. The compatibility helper is numeric
only; no string or map lookup enters candidate, graph, rank, mapping, or
constraint hot loops. Wider independent candidate scoring uses the persistent
deterministic executor; nested worker dispatch stays sequential to prevent
oversubscription.

## Frozen acceptance evidence

- Real-stack unit coverage includes constructor/accessors, registration and
  config forwarding, success, placement and routing failure flags, retained
  partial mappings, multi-link live graph order, rank-error precedence,
  unchanged input resources, workers `0/1/2/8`, and concurrent instances.
- The original order-only differential remains frozen. The combined variant
  differential is **PASS** for ten shared cases across all eight solvers at
  native workers `0/1/2/8`, using exact AST-isolated Python target classes.
  Focused unit coverage also locks sparse-ID candidate order and RNG
  continuation.
- Production, unit, differential harness and benchmark compile under strict
  warnings. ASan, UBSan and leak checks pass; targeted CTest is **2/2 PASS**;
  the aggregate build passes.
- The direct-ID audit confirms one cold registration string, direct fixed-field
  access, numeric IDs in hot paths, and caller-controlled workers with no
  auto-tuning.

The original order benchmark and the combined variant benchmark are both
frozen. On the compact combined fixture every workers=1 solver beat Python,
from `1.174x` through `75.990x`; workers 2/8 intentionally remain explicit
caller choices because fan-out overhead dominates several small cases. Exact
artifacts are `heuristic_node_rank_*_2026-07-29` and
`heuristic_node_rank_variants_*_2026-07-30` under `porting/results/`. Do not
rerun or edit either accepted benchmark.

The mutable integration result is exact for the default 1,000-request
`ffd_rank` Main case and is recorded in
`../results/system_transaction_integration_2026-07-30.md`.

## Hot-path overhead audit (2026-07-31)

The public API and frozen algorithm/output contracts are unchanged. PL/NEA
now retain the insertion-ordered selected-node list incrementally for one
solve and reuse candidate request, feasible-node, score, and ordered-exception
buffers across virtual nodes. Physical scores still rerank after each resource
mutation, but PL skips sorting that intermediate NRM result because it
immediately reindexes every score by fixed node ID before its own stable
candidate maximum. Its graph-local rank preparation is bound lazily at the
original first-use point and reused while live resource values continue to be
observed. The dynamic `weight` name is still resolved exactly once before the
first weighted traversal. These changes remove allocator and unobservable
comparison traffic only; candidate set order, first-error order, numeric
evaluation, configured worker widths, and rollback behavior are unchanged.
Frozen benchmarks were not rerun.
