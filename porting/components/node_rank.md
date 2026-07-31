# Component API: `solver.rank.NodeRank`

State: **COMPLETE / FROZEN** on 2026-07-29. The accepted differential,
benchmark, drivers, machine-readable results, and measurements are provenance;
do not rerun or edit them while porting later leaves.

Python oracle: `../virne/virne/solver/rank/node_rank.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`428F0DEB188E685A9F3EB6177A70DF533519F62C560284063FC4CF6EC7624FFF`,
10,425 bytes and 255 physical lines. Before implementation this leaf read the
frozen Graph, performance, random, NodeAttribute, LinkAttribute, BaseNetwork,
and LinkRank contracts. Those dependencies and every accepted benchmark remain
closed; implementation code is opened only if a missing public primitive or a
measured optimization requires it.

## Scope

This independent non-ML leaf owns the eight registered methods in source order:
`order`, `random`, `ffd`, `nrm`, `nea`, `grc`, `rw`, and `nps`. It returns an
ordered typed ranking and does not add Python's dynamic `node_ranking`,
`ranked_nodes`, or `node_ranking_values` fields to frozen `BaseNetwork`.
Solver execution/registry integration, candidate search, heuristics, MCF,
system orchestration, RL, Torch, CUDA, and learning code remain outside it.

## Stable typed API

All names are in `virne::solver::rank`.

```cpp
using NodeRankResourceId =
    network::attribute::AttributeRegistryId;

enum class NodeRankMethod : std::uint8_t {
    order, random, ffd, nrm, nea, grc, rw, nps
};

enum class NodeRankValueKind : std::uint8_t { scalar, proximity };

struct NodeRankEntry {
    Vertex node_id = 0U;
    NodeRankValueKind kind = NodeRankValueKind::scalar;
    double value = 0.0;
    // NPS weighted Dijkstra distance; zero for scalar rankings.
    double distance = 0.0;
};

using NodeRanking = std::vector<NodeRankEntry>;

struct NodeRankSelection {
    // nullopt selects every resource of that owner in registry order.
    std::optional<std::vector<NodeRankResourceId>> node_resources;
    std::optional<std::vector<NodeRankResourceId>> link_resources;
};

struct GRCNodeRankParameters {
    double sigma = 1e-5;
    double damping = 0.85;
};

struct RWNodeRankParameters {
    double sigma = 1e-4;
    double jump_probability = 0.15;
    double forwarding_probability = 0.85;
};

struct NodeRankParameters {
    GRCNodeRankParameters grc;
    RWNodeRankParameters rw;
};

struct NodeRankOptions {
    bool sort = true;
    std::size_t workers = 1U;
    // Python has no cap. nullopt retains that behavior; a caller may opt into
    // a deterministic typed failure for hostile/non-convergent parameters.
    std::optional<std::size_t> max_iterations;
};

inline constexpr std::size_t invalid_node_rank_input_index =
    std::numeric_limits<std::size_t>::max();

enum class NodeRankErrorCode : std::uint8_t {
    unsupported_method,
    random_stream_required,
    invalid_node_resource_selection,
    invalid_link_resource_selection,
    empty_node_resource_selection,
    empty_link_resource_selection,
    ragged_node_resource_matrix,
    non_numeric_node_resource_value,
    ranking_length_mismatch,
    stale_cardinality,
    invalid_matrix_shape,
    sparse_assignment_mismatch,
    iteration_limit_reached,
    invalid_prepared_state
};

enum class NodeRankOperation : std::uint8_t {
    resolve_method,
    prepare,
    validate_prepared,
    gather_nodes,
    gather_links,
    build_matrix,
    randomize,
    reduce,
    iterate,
    traverse,
    sort
};

class NodeRankException : public std::runtime_error {
public:
    NodeRankException(
        NodeRankErrorCode,
        NodeRankOperation,
        std::string message,
        std::size_t input_index = invalid_node_rank_input_index,
        std::optional<NodeRankResourceId> resource_id = std::nullopt,
        std::optional<Vertex> node_id = std::nullopt);
    NodeRankErrorCode code() const noexcept;
    NodeRankOperation operation() const noexcept;
    std::size_t input_index() const noexcept;
    const std::optional<NodeRankResourceId>& resource_id() const noexcept;
    const std::optional<Vertex>& node_id() const noexcept;
};

NodeRankMethod node_rank_method_from_string(std::string_view);
std::string_view node_rank_method_name(NodeRankMethod) noexcept;

class PreparedNodeRanker;

class NodeRanker {
public:
    explicit NodeRanker(
        NodeRankSelection = {}, NodeRankParameters = {});
    PreparedNodeRanker prepare(const network::BaseNetwork&) const;
};

class PreparedNodeRanker {
public:
    // Dispatches deterministic methods. `random` requires rank_random so the
    // mutable RNG owner can never be hidden in global state.
    NodeRanking rank(NodeRankMethod, NodeRankOptions = {}) const;
    NodeRanking rank_order(NodeRankOptions = {}) const;
    NodeRanking rank_random(
        NumpyRandomState&, NodeRankOptions = {}) const;
    NodeRanking rank_ffd(NodeRankOptions = {}) const;
    NodeRanking rank_nrm(NodeRankOptions = {}) const;
    NodeRanking rank_nea(NodeRankOptions = {}) const;
    NodeRanking rank_grc(NodeRankOptions = {}) const;
    NodeRanking rank_rw(NodeRankOptions = {}) const;
    NodeRanking rank_nps(NodeRankOptions = {}) const;

    const std::vector<NodeRankResourceId>& node_resource_ids() const noexcept;
    const std::vector<NodeRankResourceId>& link_resource_ids() const noexcept;
};
```

Fixed method, parameters, options, result shape, errors, operations, and
error metadata are direct fields/enums. `NodeRankException` carries the error
code, operation, optional input/resource/node location, and uses
`invalid_node_rank_input_index` when no input position applies. A compatibility
method string resolves once to `NodeRankMethod`.

## Exact behavior and numeric lanes

- `order` assigns `1/live_nodes`, keeps node order, ignores `sort`, and retains
  Python's empty-graph error.
- `random` shuffles `[0,N)` through the caller-owned completed
  `NumpyRandomState`; the shuffled node IDs become scores paired with original
  node positions before optional descending sort. Shuffle and RNG continuation
  are sequential.
- `ffd` reduces selected node-resource rows.
- `nrm` multiplies that node sum by the double incident-link sum over selected
  link resources. No link resources therefore produces signed zero exactly as
  Python; no node resources is an error.
- `nea` multiplies the node sum in its existing numeric lane by frozen Graph
  degree; an undirected self-loop contributes two.
- `grc` computes normalized node capacity, averages separately row-normalized
  link-resource matrices, then iterates the Python formula and L2 stop test.
- `rw` computes node/link capacity, replaces each nonzero topology adjacency
  entry by destination capacity, absolute-normalizes each flow row, constructs
  the exact jump/follow matrix, then iterates the Python formula.
- `nps` computes NRM, chooses the first dense vertex with maximum adjacency-key
  count, and uses frozen weighted `single_source_dijkstra_path_length` semantics.
  It returns only reachable vertices as `{distance, nrm}` entries; sorted output
  is stable ascending `(distance, -nrm)`.

Bool/int-only node rows reduce in an int64 modulo-`2^64` lane. The presence of
double selects binary64 and preserves resource order. NEA integral
multiplication also wraps modulo `2^64`; NRM/NPS convert node capacity to
binary64 before multiplying the double link aggregation. Missing values retain
the completed compact-row behavior: unequal rows are ragged, equal short rows
finish reduction before ranking-length validation. Nonnumeric values are typed
errors. GRC/RW retain IEEE divide-by-zero, infinity, NaN, and stop-comparison
behavior; an optional iteration limit is the only native extension.

Scalar sorting is stable descending. Finite values use a strict weak-order
fast path; NaN uses a private generic CPython 3.10.20 Timsort schedule. NPS uses
the same schedule with its composite ascending key. No unordered comparator is
passed to a Standard Library sort.

## IDs, ownership, and parallelism

Preparation validates selected owner/kind once and binds every registry ID to
one graph-local `AttrId`. Every node, edge, adjacency, resource, reduction,
matrix, BFS/Dijkstra-consumer, and worker loop uses only `Vertex`, stable edge
IDs/descriptors where needed, `AttrId`, numeric variants, matrix offsets, and
pre-sized slots. No hot loop hashes or compares a string. Repeated explicit
resource IDs retain caller order.

`PreparedNodeRanker` is a non-owning view. It caches only identities and
bindings, never scores or derived matrices, so ordinary value/edge mutations
are observed on the next call. Registry/graph replacement invalidates it;
mutation may not race a call. Concurrent read-only calls share a narrow graph
order/matrix-gather lock while independent numeric work remains parallel.

Worker `0/1` is sequential. Wider caller widths split only deterministic
contiguous node/cell/destination-column blocks after sequential validation.
Each node/resource or matrix dot product preserves Python operand order; L2
convergence reduction and final stable sort remain sequential. Random draws,
NPS root selection, and Dijkstra remain sequential. There is no host-selected
automatic width. GRC/RW iterative dot products use explicit `std::fma` only at
the NumPy/OpenBLAS-compatible multiply-add boundary; implicit compiler
contraction is disabled for this target so the accepted raw64 anchors remain
exact without changing other numeric lanes.

## Accepted gate

The focused unit gate passed all eight methods, sorted/unsorted ordering,
ties/signed zero/infinities/NaNs, RNG output and continuation, int64 wrap and
mixed resources, empty/missing/ragged/short/nonnumeric data, isolates,
self-loops, weighted/disconnected NPS, GRC/RW anchors and iteration boundaries,
prepared reuse/invalidation, workers `0/1/2/8`, and concurrent read-only
callers. The private generic Timsort probe also passed scalar reverse
merge/gallop and NPS composite ascending comparison-schedule cases containing
NaNs.

The exact AST-isolated Python differential passed `13/13` shared cases at
workers `1/2/8`. Ordered node/kind payloads and every raw binary64 score and
distance bit matched; case payload SHA-256 is
`8F373451937E9DE30D5E445AC2D5C7195A041B3F20A14FD94FC526D8604C0F3D`.
Strict GCC 11 warnings-as-errors and ASan/UBSan/leak checks passed. Targeted
CTest passed `3/3`: frozen-component integrity, `vne_node_rank_unit`, and
`vne_python310_generic_timsort_probe`.

The single frozen FFD benchmark used 131,072 nodes x 8 resources,
`sort=false`, one warm-up, and three samples. Python measured `47.637617 ms`;
C++ measured `37.317814`, `37.238257`, and `36.310914 ms` at workers `1/2/8`,
or `1.277x`, `1.279x`, and `1.312x` faster. Every route produced 131,072
entries, 2,097,152 bytes, and checksum `11449996351475094403`. Full
provenance and hashes are in `../results/node_rank_2026-07-29.md`; both
machine-readable JSON files beside it are frozen.

## Integration worker correction (2026-07-30)

The API is unchanged. Numeric blocks now use the persistent deterministic
executor documented in `deterministic_executor.md`. Cheap linear reductions
remain sequential below a fixed 4,096-item grain; quadratic matrix work uses a
fixed operation grain. Caller width `8` therefore does not create threads for
ordinary 100/500-node solver ranking calls, while large work still uses the
requested width. Nested dispatch is sequential and no host width is selected.
Focused unit/concurrent-caller gates pass; the frozen benchmark was not rerun.

## Allocation audit (2026-07-31)

The API, numeric lanes, RNG schedule and ordering are unchanged. Random ranking
now writes shuffled scores directly into the final `NodeRanking`; FFD writes
the already-reduced integral/double resource sums directly; and NRM/NEA write
their independent products into pre-sized final entries before attaching fixed
node metadata. The former intermediate full-size `double` vectors were pure
copies and are no longer allocated. Validation remains after RNG/reduction in
the same observable stage, and the frozen differential/benchmark artifacts
were not rerun or modified.
