# Component API: `core.controller.TopologyAnalyzer`

State: **COMPLETE / FROZEN** on 2026-07-29. Do not rerun or update the
accepted benchmark.

Python oracle: `../virne/virne/core/controller/topology_analyzer.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`665519C5C4BF50C2318E2D22D30679881DDA0EDCBC0B618C4FB2055AC5A01B28`,
8,559 bytes. Completed Graph, TopologicalMetricCalculator, utils-network,
Base/Virtual/PhysicalNetwork, LinkAttribute, and ConstraintChecker documents
were read first. The Python leaf was opened once because no component contract
or native implementation existed; completed dependency source remains frozen.

## Exact Python behavior

- All algorithms are unweighted: the local `weight` is always `None`.
  Supported public modes are first shortest, first `k` simple paths, simple
  paths while `len(path) <= k`, all shortest paths, constraint-aware BFS, and
  shortest path over currently feasible edges. The unreachable
  `available_k_shortest` branch is rejected by the earlier assertion.
- `k_shortest_length` compares `k` with path **node count**, not hop count, and
  stops at the first longer path. `max_hop` also compares against node count,
  and only the first returned path is checked; later longer paths remain.
- Path/search/checker exceptions inside `find_shortest_paths` are swallowed and
  produce an empty list. Method validation occurs before that catch. Native
  enums make ordinary invalid spellings unrepresentable; invalid enum values
  retain a typed error.
- Constraint BFS checks an edge before visited-state handling and follows
  neighbor insertion order. The Python direct helper references an unassigned
  local on no-path and has broken source-equals-target behavior; the native
  direct helper returns `nullopt` for no-path and `{source}` for identical
  endpoints, while the high-level no-path result remains the same empty list.
- Available-network views evaluate link-level constraints against live virtual
  and physical values. Pruned views snapshot the virtual link at construction,
  then apply `value *= ratio; value -= div` in selected-resource order while
  physical values remain live. Duplicate resource IDs repeat the adjustment.

## Fixed-field and ID rule

`ShortestPathMethod`, options, virtual/physical endpoints, `k`, maximum path
node count, worker widths, errors, and requests are direct fields/enums. Link
resource selection uses virtual-registry `ConstraintId` values. `prepare()`
resolves each unique dynamic resource name once against the independent
virtual/physical registries and graphs. Hot path, edge, candidate, predicate,
and worker loops retain only typed attribute pointers, graph-local `AttrId`
values, `Vertex`, stable edge IDs, direct `AttrMap` references, masks, and
pre-sized result slots. No fixed field uses a string-keyed map and no repeated
loop hashes or compares a string.

## Stable native API

```cpp
enum class ShortestPathMethod : std::uint8_t {
    first_shortest,
    k_shortest,
    k_shortest_length,
    all_shortest,
    bfs_shortest,
    available_shortest,
};

struct ShortestPathOptions {
    ShortestPathMethod method = ShortestPathMethod::k_shortest;
    std::int64_t k = 10;
    double max_path_nodes = 1.0e6;
    std::size_t constraint_workers = 1;
};
struct TopologyPathRequest {
    ConstraintLink virtual_link;
    ConstraintLink physical_pair;
    ShortestPathOptions options;
};
struct TopologyAnalyzerSelection {
    ConstraintCheckerSelection constraints;
    std::vector<ConstraintId> link_resources;
};

class TopologyAnalyzer {
public:
    explicit TopologyAnalyzer(TopologyAnalyzerSelection);
    PreparedTopologyAnalyzer prepare(
        const network::VirtualNetwork&,
        const network::PhysicalNetwork&) const;
};

class PreparedTopologyAnalyzer {
public:
    std::vector<std::vector<Vertex>> find_shortest_paths(
        const TopologyPathRequest&) const;
    std::optional<std::vector<Vertex>> find_bfs_shortest_path(
        ConstraintLink, Vertex source, Vertex target) const;
    SearchMask create_available_mask(
        ConstraintLink, std::size_t workers = 1) const;
    nx::GraphView create_available_network(ConstraintLink) const;
    SearchMask create_pruned_mask(
        ConstraintLink, double ratio = 1.0, double div = 0.0,
        std::size_t workers = 1) const;
    nx::GraphView create_pruned_network(
        ConstraintLink, double ratio = 1.0, double div = 0.0) const;
    std::vector<std::vector<std::vector<Vertex>>>
    find_shortest_paths_batch(
        const std::vector<TopologyPathRequest>&,
        std::size_t workers = 1) const;
};
```

Prepared analyzers and returned views are non-owning and must not outlive or
race structural/schema mutation of their networks. A live view materializes a
fresh direct `SearchMask` once per graph algorithm. Explicit mask methods are
snapshot extensions and can use caller-configured workers.

For `first_shortest` and `available_shortest`, the implementation uses the
completed raw-order BFS primitive and reconstructs from direct predecessor
slots. This is required for Python's `dijkstra_path(weight=None)` FIFO tie
order. The frozen convenience `nx::dijkstra_path(..., nullopt)` delegates to
point-to-point unweighted shortest path and selected a different equal-length
route on a cyclic corpus. A targeted four-node differential locks Python's
`[1,0,3]` result without modifying the frozen Graph library.

## Threading and gate

Constraint-aware BFS remains sequential because neighbor and first-path order
are observable. Available/pruned mask edges and complete batch requests are
independent; zero/one is sequential, wider caller widths use deterministic
contiguous blocks, pre-sized slots, input-order results, and the lowest failing
request. There is no automatic machine policy.

Cover all six modes, path tie order, `k<=0`, length cutoff, source equals
target, disconnected/invalid endpoints, first-path-only max filtering,
hard/soft feasibility, BFS neighbor order, prune ratio/div, snapshot versus
live values, duplicate/empty resources, independent registry ordering,
workers `0/1/2/8`, and concurrent callers. After exact differential parity,
run one compact checksum-gated batch benchmark at workers `1/2/8`, compare
runtime with Python, and freeze it permanently.

The final gate passed **24/24 exact shared Python cases** and nine native unit
groups. Strict GCC 11 production/harness/benchmark compilation with conversion,
sign-conversion, shadow, and warning errors passes. ASan, UBSan, leak detection,
targeted CTest, and frozen-foundation integrity pass. The permanently frozen
4,096-query benchmark retained checksum `10025764477037659827`; C++ was
83.768x, 147.126x, and 177.706x faster at caller workers `1/2/8`. See
`porting/results/topology_analyzer_2026-07-29.md`.
