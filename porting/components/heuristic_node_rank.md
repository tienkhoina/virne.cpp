# Component API: `solver.heuristic.node_rank`

State: **COMPLETE / FROZEN** on 2026-07-29.

Python oracle: `../virne/virne/solver/heuristic/node_rank.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`44A6F63F1A1798935453C057C6981A692F5DC4F03BDE6A6535607AECBDC61389`.
The frozen scope is `BaseNodeRankSolver`, deterministic `OrderRankSolver`, and
explicit `order_rank` registration. Other heuristic ranks, system solvers,
MCF/OR-Tools, ML and RL remain out of scope.

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

SolverId register_order_rank_solver(
    SolverRegistry&,
    NodeRankSolverWorkers = {});
```

`OrderRankSolver` fixes `rank::NodeRankMethod::order` and default rank
parameters. Registration captures workers by value and registers the cold
string `"order_rank"` as `SolverCategory::node_ranking`; registry errors are
not translated.

## Solve and performance contract

Construction reads the Controller's fixed typed selection once and retains
NodeMapper/LinkMapper configured with numeric attribute and constraint IDs.
No string map is used for fixed fields or in a hot loop.

For each solve, the implementation creates `Solution::from_v_net`, ranks the
virtual then original physical network, extracts ordered numeric node IDs,
clones the physical network exactly once, maps nodes greedily, then maps live
virtual links on that same clone. The const input physical network is never
mutated. Placement failure writes `place_result=false` then `result=false`;
routing failure writes `route_result=false` then `result=false`; success writes
`result=true`. Partial mappings and dependency exception order are retained.

Node mapping always uses `reusable=false` and `inplace=true`. Ranking receives
`rank_workers`; mapping receives the other three worker fields together with
the typed `SolverConfig` matching/shortest-path settings. Widths `0/1` select
the dependency's sequential path. Wider widths are caller configuration only:
there is no automatic tuning, and solve phases remain deterministically
ordered.

## Frozen acceptance evidence

- Real-stack unit coverage includes constructor/accessors, registration and
  config forwarding, success, placement and routing failure flags, retained
  partial mappings, multi-link live graph order, rank-error precedence,
  unchanged input resources, workers `0/1/2/8`, and concurrent instances.
- The differential is **PASS** for six shared cases (five solution cases) at
  native workers `0/1/2/8`, using the exact AST-isolated Python target classes.
- Production, unit, differential harness and benchmark compile under strict
  warnings. ASan, UBSan and leak checks pass; targeted CTest is **2/2 PASS**;
  the aggregate build passes.
- The direct-ID audit confirms one cold registration string, direct fixed-field
  access, numeric IDs in hot paths, and caller-controlled workers with no
  auto-tuning.

Benchmark interpretation, exact frozen timings and artifact provenance are in
[`heuristic_node_rank_2026-07-29.md`](../results/heuristic_node_rank_2026-07-29.md).
The benchmark is frozen and must not be rerun or updated.
