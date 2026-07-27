# Graph/DiGraph API contract

This document is the public contract for the graph library. `Graph` keeps the
existing undirected signatures and `DiGraph` supplies a concrete overload for
every supported operation. NetworkX names keep NetworkX's public argument
order and defaults wherever those concepts have a fixed C++ representation.
Python's dynamic result shapes are represented by explicit overloads: for
example, `shortest_path(g)`, `shortest_path(g, source)`, and
`shortest_path(g, source, target)` return the all-pairs, single-source, and
single-path C++ types respectively. Internal templates are implementation
details and are not part of the public API.

The exact C++ declarations in `graph.h`, `graph_types.h`, `attribute.h`,
`distance_matrix.h`, `sparse_matrix.h`, `csr_matrix.h`, `views/*.h`,
`algorithms/*.h`, `nx/*.h`, `cache/weight_cache.h`, `generators/gml_loader.h`,
`generators/topology_generators.h`, `generators/waxman_generator.h`, and
`io/graph_saver.h` are the signature source of truth. Within this supported
subset, every `DiGraph` overload keeps
the matching `Graph` parameter order, defaults and return type (substituting
only the graph/edge type). Unsupported backend dispatch, Python callbacks and
arbitrary Python object labels are deliberately not simulated by hidden
runtime variants.

Headers below an explicit `detail/` directory and the iterator/helper
templates used only to implement a documented facade are private
implementation details. They are not additions to the frozen surface merely
because a public header must mention their types. The public facades and
result containers themselves are inventoried below.

The zero-declaration legacy placeholders `algorithms/dfs.h` and
`io/graph_loader.h` are not supported APIs and are not compiled into
`graph_lib`; `views/neighbor_view.h` is only a compatibility include marker,
while the real `NeighborRange` contract lives in `views/adjacency_view.h`.
Their presence does not promise a DFS facade or a second loader surface. The
original Virne source scan found no non-visual call requiring either one; GML
loading is owned by the explicit APIs documented below.

## Pinned compatibility baseline

- Boost: **1.85.0** (`BOOST_VERSION == 108500`), workspace-vendored in
  `libs/boost`.
- GNU toolchain layout used by the release benchmark: **GCC 11.4.0** with
  **libstdc++ 11** (`_GLIBCXX_RELEASE == 11`,
  `__GLIBCXX__ == 20230528`). This is also the tested layout for the Random
  module's direct-output-vector optimization; see the
  [Random contract](../random/README.md).
- NetworkX oracle: **3.4.2**, installed only in the repository-local `.venv`.
- yaml-cpp: **0.8.0**, workspace-vendored in `libs/yaml-cpp`.
- tabulate: **1.4.0**, workspace-vendored in `libs/tabulate`.

The C++ source trees are Git-ignored workspace payloads. The
[dependency policy](../DEPENDENCIES.md) and
[checksum manifest](../DEPENDENCIES.sha256) are the source/archive/checksum
contract for reconstructing them without using OS or Conda packages.

The graph hot path intentionally reads Boost `adjacency_list` implementation
storage. Node attributes use `m_vertices[v].m_property`, adjacency uses
`m_out_edges`/`m_in_edges`, edge endpoints use the descriptor's `m_source` /
`m_target`, and edge attributes/IDs use its internal property pointer. Graph
move operations swap `m_vertices` and `m_edges` directly because Boost 1.85's
`adjacency_list` move path otherwise copies and can reorder incidence lists.
Undirected self-loop insertion/copy also removes Boost's duplicate incidence
record once, then `degree()` restores NetworkX's contribution of two. Edge
iteration uses the global `m_edges` `listS`: after arbitrary out-of-node-order
insertion, `edges()` stably relinks that list by source so its order matches
NetworkX. Relinking preserves stored list iterators and property addresses;
monotone generator insertion needs no sort.

These are accepted ABI/layout hacks, not stable BGL APIs. `graph_types.h`
therefore rejects Boost versions other than 1.85.0 and asserts the exact vertex
bundle, raw-neighbor property, descriptor property-pointer, vertex-index, and
global `listS` types at compile time. A Boost upgrade is not a routine package
bump: update the pin and assertions, then pass the clean Release build, all
C++ tests, ASan/UBSan, the 91-value base oracle, the 34-case completion oracle,
the 91-case generator/order oracle, and every benchmark gate. The Random
module's separate libstdc++ layout optimization is likewise hard-pinned by
`static_assert` to GCC 11.4/libstdc++ 11 (`20230528`) whenever GNU libstdc++ is
selected; other standard libraries use the portable fallback. Changing that
guard requires the same explicit layout audit plus parity/sanitizer testing.

## Core graph types

Both types are simple graphs: adding an existing edge returns that edge and
does not create a parallel edge.  `Vertex` values are contiguous indices in
`[0, num_nodes())`.  Edge IDs are monotonic and are not reused after removal.

| Operation | `Graph` | `DiGraph` |
|---|---|---|
| `add_node()` | supported | supported |
| `add_edge(u, v)` | one undirected edge | one arc `u -> v`; `v -> u` is distinct |
| `add_nodes_from(...)`, `add_edges_from(...)` | plain and `AttrObject` variants | plain and `AttrObject` variants |
| `node_attrs(v)`, `edge_attrs(e)`, `graph_attrs()` | supported | supported |
| `num_nodes()`, `num_edges()` | nodes/edges | nodes/arcs |
| `number_of_nodes()`, `number_of_edges()` | NetworkX-named aliases | NetworkX-named aliases |
| `degree(v)` | undirected degree | in-degree + out-degree; a self-loop counts twice |
| `degree()` | ordered `(node, degree)` vector | ordered total-degree vector |
| `in_degree()`, `out_degree()` | n/a | ordered whole-graph vectors; scalar overloads remain |
| `neighbors(v)` | adjacent vertices | successors |
| `neighbors_fast(v)` | raw adjacent edge records | raw outgoing edge records |
| `successors`, `predecessors` | n/a | ordered vertex ranges |
| `successors_fast`, `predecessors_fast` | n/a | raw outgoing/incoming edge records |
| `out_edges`, `in_edges` and fast variants | n/a | ordered descriptors/raw records |
| `nodes()`, `edges()` | supported | supported |
| `edges(v)` | incident edges in adjacency order | outgoing arcs in successor order |
| `node_view`, `edge_view`, `adjacency_view` | mutable/const zero-copy facades | mutable/const zero-copy facades |
| `has_edge`, `edge`, `remove_edge` | unordered endpoints | ordered endpoints |
| `source(e)`, `target(e)` | stored endpoints | ordered source/target |
| `edge_id`, `edge_by_id`, `edge_endpoints`, `edge_id_capacity` | supported | supported |
| `attr_id`, `attr_name`, `attribute_registry` | supported | supported |
| conversion | `to_directed()` | n/a |
| `raw()` | `BGLGraph` | `BGLDiGraph` |

`edges()` follows NetworkX 3.4.2 iteration order exactly for wrapper-managed
graphs. `Graph` groups edges by the first node in node insertion order, uses
that node as the returned orientation, and preserves its adjacency insertion
order; with contiguous indices this is the smaller endpoint. `DiGraph` groups
arcs by source and preserves successor insertion order. Removing and re-adding
an edge moves it to the end of that adjacency group. `to_directed()` walks the
undirected adjacency list node by node, so its arc order also matches
`Graph.to_directed()` rather than merely interleaving forward/reverse arcs.
Graph copies restore each Boost incidence list by stable edge ID after BGL's
global-list copy, preserving neighbor/successor/predecessor insertion order
even if `edges()` normalized the source graph first.

`edges(v)` matches NetworkX's single-node EdgeView selection. An undirected
edge is oriented from `v` to its neighbor, adjacency insertion order is
preserved, and a self-loop is returned once. Native BGL would store two raw
undirected incidence records, but the wrapper collapses that duplication at
insertion/copy boundaries. For `DiGraph`, selection is outgoing only. The no-argument
degree methods return `std::vector<std::pair<Vertex, size_t>>` in contiguous
node order, suitable for the original `dict(network.degree())` call sites.

The dense-matrix constructors
`Graph(adjacency, weight_attr="weight")` and
`DiGraph(adjacency, weight_attr="weight")` require a square
`std::vector<std::vector<double>>`. They create all row indices as nodes, omit
zero entries, preserve nonzero diagonal entries as self-loops, and store each
nonzero under the selected weight attribute. In `Graph`, reciprocal nonzero
cells address the same simple edge; the later row-major cell updates its
weight, matching the frozen NetworkX conversion oracle.

`graph_attrs()` returns the mutable/const graph-level `AttrMap` used by GML
metadata. `edge_id_capacity()` returns the size of the monotonic edge-ID
address space, including holes left by removed edges; it is therefore the
correct extent for an edge-ID-indexed `SearchMask` or snapshot array, not the
number of live edges. `edge_by_id` and `edge_endpoints` reject a hole or an ID
outside that extent.

### Views, bulk construction, and lifetime

`node_view()`, `edge_view()`, and `adjacency_view()` return pointer-sized
facades that own no graph data. Their ordered iteration and `data()`/`items()`
ranges return direct `AttrMap` references, never copied dictionaries:

- `NodeView::at(v)`/`operator[](v)` looks up node attributes;
- `EdgeView::at(u, v)`/`operator[]({u, v})` looks up edge attributes;
- `EdgeView::incident(v)` is the undirected selection and
  `EdgeView::outgoing(v)` is the directed selection;
- `EdgeView` and its one-node selections iterate public `(u, v)` endpoint
  pairs; `descriptors()` is the explicit escape hatch for algorithms that need
  Boost descriptors;
- `AdjacencyView::at(v)` iterates neighbors for `Graph` and successors for
  `DiGraph`; its data range pairs each neighbor with the live edge attributes.

The facade inventory below is frozen; exact iterator return types remain in
[`views/node_view.h`](views/node_view.h),
[`views/edge_view.h`](views/edge_view.h), and
[`views/adjacency_view.h`](views/adjacency_view.h).

| Facade | Public operations |
|---|---|
| `NodeView<GraphType>` | `begin/end`, `size`, `empty`, `contains`, checked `at` and `operator[]`, `data/items`, `keys` |
| `EdgeView<GraphType>` | endpoint `begin/end`, `size`, `empty`, `contains`, checked `at` and `operator[]`, `data/items`, `for_node`/`operator()`, Graph-only `incident`, DiGraph-only `outgoing`, and `descriptor_begin/descriptor_end/descriptors` |
| `IncidentEdgeRange<GraphType>` | endpoint `begin/end`, `size`, `empty`, `data/items`, and `descriptors` |
| `AdjacencyView<GraphType>` | node `begin/end`, `size`, `contains`, checked `at`, unchecked `operator[]`, `data/items`, and `keys` |
| `NeighborRange<GraphType>` | neighbor `begin/end`, `size`, `empty`, `contains`, checked `at` and `operator[]`, `data/items`, and `keys` |

`NodeDataRef`, `EdgeDataRef`, `NeighborDataRef`, and `AdjacencyDataRef` expose
the indexed node/edge endpoints together with live references/ranges. A
neighbor iterator additionally exposes `edge()` and `attrs()` without a second
endpoint lookup. Iterator implementation classes themselves are not intended
as independently constructed APIs.

The NetworkX-style free-function facade in [`nx/views.h`](nx/views.h) is also
part of the public surface:

```cpp
namespace nx {
inline auto neighbors(const Graph& graph, Vertex node)
{ return graph.neighbors(node); }
inline auto neighbors(const DiGraph& graph, Vertex node)
{ return graph.neighbors(node); }
DegreeItems degree(const Graph& graph);
DegreeItems degree(const DiGraph& graph);
DegreeItems in_degree(const DiGraph& graph);
DegreeItems out_degree(const DiGraph& graph);
}
```

A view must not outlive its graph. Attribute mutation through a mutable view is
supported and immediately visible everywhere. Structural mutation invalidates
active iterators/ranges; obtain a new `begin()` after adding or removing an
edge. `neighbors_fast()` remains the preferred unchecked primitive for the
tightest profiled loop; the views add bounds/order/self-loop semantics while
still using contiguous indices and raw adjacency storage internally.

Bulk input accepts `std::vector<Vertex>`, `NodeWithAttrs`, `EdgeEndpoints`, or
`EdgeWithAttrs`; Graph/DiGraph also have edge-list constructors, with optional
explicit node count for isolates. Attributes are recursively cloned. Repeated
nodes merge supplied keys, and repeated edges keep their first position while
later keys update the same simple edge, matching NetworkX. Because the core
cannot represent arbitrary Python labels, `add_nodes_from` accepts only a
collectively contiguous extension. In contrast, `add_edge` and edge-list
construction materialize every dense index through `max(u, v)`, so an endpoint
gap becomes intervening isolate nodes. This is an explicit dense-index
translation rule and differs from Python NetworkX, where `add_edge(2, 5)`
creates only labels `2` and `5`. Callers porting sparse integer labels must
relabel them first. The explicit-node-count constructor requires every
endpoint to be within `[0, node_count)`.

`neighbors_fast()` and the directed `successors_fast`, `predecessors_fast`,
`out_edges_fast`, and `in_edges_fast` aliases are deliberately unchecked and
are intended for validated indices inside hot loops. The direct
`node_attrs(v)`, `edge_attrs(e)`, `edge_id(e)`, `source(e)`, and `target(e)`
accessors likewise require an in-range index or a live descriptor belonging to
that graph. This is the explicitly accepted price of bypassing BGL
property-map dispatch. Checked public views (`NodeView::at`, `EdgeView::at`,
`AdjacencyView::at`) remain available at boundaries.
`AdjacencyView::operator[]` deliberately follows `std::vector::operator[]` and
is unchecked for validated dense-index loops; use `at()` at a boundary.

### Mandatory hot-loop ID contract

This is a normative design rule for every current implementation and every
future consumer of the frozen graph core:

- A public boundary **MAY** accept an attribute name as `string` or
  `string_view`, preserving readable NetworkX-shaped calls.
- Before entering any loop over nodes, edges, neighbors, sources, candidates,
  batches, samples, or a predicate invoked by such a loop, code **MUST**
  resolve that name exactly once to `AttrId`.
- Inside such a loop, code **MUST** use contiguous `Vertex` indices, stable
  `uint32_t` edge IDs where an edge index is needed, `AttrId`, and
  `AttrMap::find/at/set(AttrId)` or direct indexed storage. It **MUST NOT** call
  `attr_id`, `attr_name`, `find("...")`, `at("...")`, `operator[]("...")`, or
  perform string/endpoint-map lookup per iteration.
- A public string-taking algorithm **MUST** delegate to an ID-taking internal
  implementation after one resolution. Nested algorithms, including every
  Yen spur search, **MUST** propagate the resolved ID rather than the original
  string.
- A repeated predicate/callback **MUST** capture a previously resolved
  `AttrId`. Predicate subgraph callbacks are boundary logic; algorithms
  materialize their `SearchMask` once before the traversal hot loop.
- GML/YAML/config parsing, serialization, diagnostics, and interactive query
  code are boundary exceptions because names are the input/output format.
  If boundary code hands data to a performance-critical traversal or batch,
  resolution **MUST** occur before that compute loop; parser-internal name
  handling is not itself a graph hot-loop contract.

An `AttrId` belongs to one `AttributeRegistry`; its numeric value **MUST NOT**
be transferred to an independently constructed or independently loaded graph
without resolving the name in that graph. Graph copies,
`Graph::to_directed()`, and `convert_node_labels_to_integers` deliberately
preserve the complete name-to-ID order, so an already resolved ID remains
valid across those specific conversions. `attr_name(id)` returns a
`string_view` backed by stable registry storage, so later `attr_id(name)` calls
do not invalidate it; the view still must not outlive the registry. Numeric
`AttrId` setters reject IDs outside their bound registry. `AttrMap::update`
translates by name when source and target use different registries, while maps
sharing one registry retain the raw-ID fast path. A `WeightCache` is a snapshot
and must still be rebuilt after any graph or attribute mutation.

### Attribute and matrix container inventory

The concrete support containers below are part of the public boundary. Their
exact declarations are in [`attribute.h`](attribute.h),
[`distance_matrix.h`](distance_matrix.h),
[`sparse_matrix.h`](sparse_matrix.h), and [`csr_matrix.h`](csr_matrix.h).

| Type | Frozen public surface |
|---|---|
| `AttrValue` | `int64_t`, `double`, `bool`, `string`, recursive `AttrListPtr`, or recursive `AttrObjectPtr` |
| `AttrList` / `AttrObject` | ordered `values`/`entries`; object `find(name)` and `set(name, value)`; `make_attr_list`, `make_attr_object`, `attr_list`, `attr_object`, `attr_value_equal`, `clone_attr_value`, and `attr_to_double` boundary helpers |
| `AttributeRegistry` | `intern(name)`, non-interning `find(name)`, checked `name(id)`, and `size()` |
| `AttrMap` | registry binding; name and `AttrId` `at/find/contains/get`; boundary-only string `operator[]`; `keys/items`; `update`; ID-only `set`; name/ID `erase`; `size/empty/clear`; the shared `registry` handle; and read-only `slots/attribute_ids` access |
| `nx::OrderedAttributeMap<Key, Hash>` | ordered iteration plus O(1) `emplace`, `operator[]`, `at`, `find`, `contains`, `count`, `size/empty`, `values`, and `items`; edge specialization also accepts stable edge IDs |
| `nx::OrderedVertexMap<T>` | dense O(1) `find/contains/count/at/operator[]`, ordered iteration, `values/items`, and normal reserve/insert/emplace operations |
| `DistanceMatrix` | public `n/data`, indexed `operator()(row,col)`, `rows/cols/size/empty`, mutable/const `raw_data`, and `fill` |
| `SparseMatrix` | public COO `rows/cols/row/col/value`, `reserve/add`, `nnz/empty`, mutable/const `tocoo` and `data`, `shape`, `toarray`, `nonzero`, and `to_csr` |
| `CSRMatrix` | public `rows/cols/row_ptr/col_idx/values`, `nnz/empty`, `rows_count/cols_count`, and `row_ptr_data/col_idx_data/values_data` |

Container `items()`/`values()` methods that return vectors are boundary
conveniences and may copy. Hot loops use `attribute_ids()`, ID lookup, direct
matrix arrays, or ordered-container iteration instead.

Although `raw()` has a mutable overload for the existing low-level contract,
consumers must treat it as read-only. Calling Boost mutation functions on the
returned object bypasses `AttrMap` binding, simple-edge uniqueness, stable edge
ID allocation and endpoint bookkeeping, and invalidates caches. Add/remove
nodes and edges only through `Graph`/`DiGraph`; direct Boost storage access is
reserved for profiled traversal.

## Algorithms

The declarations below are the concrete public overloads. Directed traversals
always follow outgoing arcs. Bidirectional searches use the raw incoming list
for their backward frontier.

```cpp
BFSResult bfs(const Graph& g, Vertex source);
BFSResult bfs(const DiGraph& g, Vertex source);
BFSResult bfs(const Graph& g, Vertex source, const SearchMask& mask);
BFSResult bfs(const DiGraph& g, Vertex source, const SearchMask& mask);

nx::OrderedVertexMap<size_t> bfs_nx(const Graph& g, Vertex source);
nx::OrderedVertexMap<size_t> bfs_nx(const DiGraph& g, Vertex source);

BFSStats bfs_stats(const Graph& g, Vertex source, BFSWorkspace& ws);
BFSStats bfs_stats(const DiGraph& g, Vertex source, BFSWorkspace& ws);

BidirectionalBFSResult bidirectional_bfs(const Graph& g, Vertex source,
                                         Vertex target);
BidirectionalBFSResult bidirectional_bfs(const DiGraph& g, Vertex source,
                                         Vertex target);
BidirectionalBFSResult bidirectional_bfs(const Graph& g, Vertex source,
    Vertex target, const SearchMask& mask);
BidirectionalBFSResult bidirectional_bfs(const DiGraph& g, Vertex source,
    Vertex target, const SearchMask& mask);

DijkstraResult dijkstra(
    const Graph& g, Vertex source,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra(
    const DiGraph& g, Vertex source,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra_with_cutoff(
    const Graph& g, Vertex source, std::optional<double> cutoff,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra_with_cutoff(
    const DiGraph& g, Vertex source, std::optional<double> cutoff,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra_with_cutoff(
    const Graph& g, Vertex source, const SearchMask& mask,
    std::optional<double> cutoff,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra_with_cutoff(
    const DiGraph& g, Vertex source, const SearchMask& mask,
    std::optional<double> cutoff,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const Graph& g, Vertex source, const SearchMask& mask,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra(
    const DiGraph& g, Vertex source, const SearchMask& mask,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const Graph& g, Vertex source,
    const VertexSet& banned_vertices, const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra(
    const DiGraph& g, Vertex source,
    const VertexSet& banned_vertices, const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra(
    const Graph& g, Vertex source, const SearchMask& mask,
    const VertexSet& banned_vertices, const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");
DijkstraResult dijkstra(
    const DiGraph& g, Vertex source, const SearchMask& mask,
    const VertexSet& banned_vertices, const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");

BidirectionalPathResult bidirectional_dijkstra(
    const Graph& g, Vertex source, Vertex target,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");
BidirectionalPathResult bidirectional_dijkstra(
    const DiGraph& g, Vertex source, Vertex target,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");
BidirectionalPathResult bidirectional_dijkstra(
    const Graph& g, Vertex source, Vertex target, const SearchMask& mask,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");
BidirectionalPathResult bidirectional_dijkstra(
    const DiGraph& g, Vertex source, Vertex target, const SearchMask& mask,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");

double edge_cost(const Graph& g, Vertex u, Vertex v,
                 const std::string& weight_attr = "weight");
double edge_cost(const DiGraph& g, Vertex u, Vertex v,
                 const std::string& weight_attr = "weight");

double path_cost(const Graph& g, const std::vector<Vertex>& path,
                 const std::string& weight_attr = "weight");
double path_cost(const DiGraph& g, const std::vector<Vertex>& path,
                 const std::string& weight_attr = "weight");

std::vector<double> path_prefix_costs(const Graph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr = "weight");
std::vector<double> path_prefix_costs(const DiGraph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr = "weight");

std::vector<Vertex> build_path(const DijkstraResult& result,
                               Vertex source, Vertex target);

DistanceMatrix floyd_warshall(const Graph& g,
    const std::string& weight_attr = "weight");
DistanceMatrix floyd_warshall(const DiGraph& g,
    const std::string& weight_attr = "weight");

std::vector<Vertex> join_paths(const std::vector<Vertex>& root,
                               const std::vector<Vertex>& spur);

std::vector<PathResult> yen_k_shortest_paths(const Graph& g, Vertex source,
    Vertex target, size_t k,
    const std::string& weight_attr = "weight");
std::vector<PathResult> yen_k_shortest_paths(const DiGraph& g, Vertex source,
    Vertex target, size_t k,
    const std::string& weight_attr = "weight");
std::vector<PathResult> yen_k_shortest_paths(const Graph& g, Vertex source,
    Vertex target, const SearchMask& mask, size_t k,
    const std::string& weight_attr = "weight");
std::vector<PathResult> yen_k_shortest_paths(const DiGraph& g, Vertex source,
    Vertex target, const SearchMask& mask, size_t k,
    const std::string& weight_attr = "weight");

std::vector<PathResult> generate_candidates(const Graph& g,
    const PathResult& shortest, Vertex target,
    const std::string& weight_attr = "weight");
std::vector<PathResult> generate_candidates(const DiGraph& g,
    const PathResult& shortest, Vertex target,
    const std::string& weight_attr = "weight");
```

`SearchMask` is the public, dense filtering primitive declared in
[`algorithms/search_mask.h`](algorithms/search_mask.h):

```cpp
SearchMask();
SearchMask(size_t node_count, size_t edge_id_capacity,
           bool allowed = true);
bool allows_node(Vertex v) const noexcept;
bool allows_edge(uint32_t edge_id) const noexcept;
bool allows(Vertex u, Vertex v, uint32_t edge_id) const noexcept;
void set_node(Vertex v, bool allowed);
void set_edge(uint32_t edge_id, bool allowed);
const std::vector<uint8_t>& node_flags() const noexcept;
const std::vector<uint8_t>& edge_flags() const noexcept;
bool filters_nodes() const noexcept;
bool filters_edges() const noexcept;
```

An empty node or edge side means “allow all”. A materialized side is indexed
directly by `Vertex` or stable edge ID; construct its edge side with
`edge_id_capacity()`, not `num_edges()`. Setters are checked, while `allows_*`
returns `false` for an out-of-range index on a materialized side.

Public algorithm support/result types are frozen as follows:

| Type | Public data/operations |
|---|---|
| `BFSResult` | `distance`, `predecessor`, NetworkX discovery-order `discovery_order` |
| `BFSStats` / `BFSWorkspace` | `sum_dist`, `reachable`; reusable dense `dist/queue` workspace constructed with a node count |
| `DijkstraResult` | `distance`, `predecessor`, NetworkX-compatible `settled_order` |
| `BidirectionalBFSResult` | `found`, hop `distance`, `path` |
| `BidirectionalPathResult` | `found`, weighted `cost`, `path` |
| `VertexSet` / `EdgeSet` | hash sets used by banned-vertex/edge overloads; `normalize_edge_key(u,v)` supplies the undirected key, while directed callers retain `(u,v)` order |
| `PathResult` / `CandidateCompare` | `path`, `cost`, stable `insertion_order` tie breaker and the corresponding minimum-candidate ordering |
| `ShortestSimplePathOptions` | optional `max_paths/max_hops`, `max_cost`, and boundary string `weight_attr` (`""` means unweighted) |

`ShortestSimplePathGenerator` has concrete constructors for `Graph` and
`DiGraph`, with and without a `const SearchMask&`, plus move construction/move
assignment, `std::optional<PathResult> next()`, and `size_t yielded() const`.
It is non-copyable and borrows the graph. Its exact declarations are in
[`algorithms/k_shortest_paths.h`](algorithms/k_shortest_paths.h).

No public function template replaces these concrete overloads.
Dijkstra-family algorithms require non-negative numeric weights. A missing
weight attribute has value `1.0`.

The `nx::*` layer additionally exposes the names used by Virne:
`shortest_path`, `shortest_path_length`, `all_shortest_paths`,
`single_source_shortest_path_length`, `dijkstra_path`,
`dijkstra_path_length`, `single_source_dijkstra_path_length`,
`floyd_warshall`, and eager/lazy `shortest_simple_paths`. Each has concrete
`Graph`, `DiGraph`, `GraphView`, and `DiGraphView` overloads where applicable.
Single-source and all-pairs mappings use `OrderedVertexMap<T>`: dense vertex
lookup is O(1), while iteration follows NetworkX discovery/settlement order.
`items()` and `values()` expose that order without a hash lookup in the loop.

The path signatures below preserve NetworkX's public parameter order (shown
for `Graph`; concrete `DiGraph`, `GraphView`, and `DiGraphView` overloads have
the same arguments):

```cpp
std::vector<std::vector<Vertex>> all_shortest_paths(
    const Graph& g, Vertex source, Vertex target,
    std::optional<std::string_view> weight);

SingleSourcePathLengths single_source_shortest_path_length(
    const Graph& g, Vertex source,
    std::optional<double> cutoff = std::nullopt);

std::vector<Vertex> dijkstra_path(
    const Graph& g, Vertex source, Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

double dijkstra_path_length(
    const Graph& g, Vertex source, Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

SingleSourceDijkstraPathLengths single_source_dijkstra_path_length(
    const Graph& g, Vertex source,
    std::optional<double> cutoff = std::nullopt,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

ShortestSimplePathGenerator shortest_simple_paths(
    const Graph& g, Vertex source, Vertex target,
    std::optional<std::string_view> weight);
```

`cutoff` is applied during relaxation rather than by filtering a complete
result afterward. The source remains present with distance `0`, including for
a negative cutoff, matching the NetworkX oracle.

### Result, mask, and error contract

- `BFSResult::distance` uses `std::numeric_limits<size_t>::max()` for an
  unreachable vertex; its predecessor is `Vertex(-1)`. `DijkstraResult` uses
  `std::numeric_limits<double>::max()` and a self predecessor. `build_path`
  returns an empty vector for an invalid or unreachable target.
- `has_edge(u, v)` and `remove_edge(u, v)` return `false` for out-of-range or
  absent endpoints. `edge(u, v)` throws `std::out_of_range` for an invalid
  endpoint and `std::runtime_error` when both endpoints are valid but the edge
  is absent.
- Scalar `degree`/`in_degree`/`out_degree`, `neighbors`, `successors`,
  `predecessors`, `out_edges`, `in_edges`, and the checked single-node
  view/range constructors throw `std::out_of_range` for an invalid vertex.
  Their `*_fast` counterparts remain unchecked by design.
- `edge_by_id` and `edge_endpoints` throw `std::runtime_error` for an ID outside
  `edge_id_capacity()` or for a removed-edge hole. Descriptor-taking accessors
  require a currently live descriptor owned by that graph.
- `BFSStats::reachable` excludes the source, while `sum_dist` sums distances
  to the other reachable vertices. `BFSWorkspace` must have been constructed
  for exactly `g.num_nodes()` or `bfs_stats` throws `std::invalid_argument`.
- Single-source `bfs`, `bfs_nx`, `bfs_stats`, and `dijkstra` throw
  `std::out_of_range` for an invalid source. The low-level bidirectional
  functions instead return `found == false` for invalid endpoints/no path;
  the point-to-point `nx::shortest_path`, `nx::shortest_path_length`,
  `nx::dijkstra_path`, and `nx::dijkstra_path_length` wrappers convert
  no-path to `std::runtime_error`.
- An `EdgeSet` key `(u, v)` is ordered for `DiGraph`; banning it does not ban
  `(v, u)`. For `Graph`, endpoint lookup is undirected. A masked single-source
  Dijkstra rejects a banned source; bidirectional Dijkstra reports not found
  when either endpoint is banned.
- `yen_k_shortest_paths` and the eager `nx::shortest_simple_paths(..., k)`
  overload return an empty vector when no path exists; the lazy overload is
  exhausted. This differs from NetworkX's Python `NetworkXNoPath` exception.

## NetworkX-compatible subset

The C++ subset keeps the NetworkX names, argument order, defaults and observable
iteration/tie order needed by Virne. C++ overloads replace Python's dynamic
return-type switching.

The following differences from NetworkX 3.4.2 are deliberate parts of the
frozen C++ API. Backend-dispatch parameters are omitted and callable Python
weights are not supported. `all_shortest_paths`, `dijkstra_path`,
`dijkstra_path_length`, and `single_source_dijkstra_path_length` use
`std::optional<std::string_view>{"weight"}` so `std::nullopt` is the exact C++
spelling of `weight=None`. The NetworkX-shaped lazy `shortest_simple_paths`
overload also accepts `optional<string_view>`; its internal options/eager
convenience overloads and the centrality surface use an empty string for
`weight=None`.

| C++ API | Frozen difference from the Python API |
|---|---|
| `shortest_path`, `shortest_path_length` | explicit 0/1/2-endpoint overloads model all-pairs/single-source/point-to-point results; the supported calls are unweighted and omit Python's `weight`/`method` dispatch |
| `all_shortest_paths` | three-argument unweighted overload plus a fourth `optional<string_view>` weight; `nullopt` is `weight=None`, while callable weights and `method` are omitted |
| `single_source_shortest_path_length` | keeps NetworkX's `(G, source, cutoff=nullopt)` order; Python backend dispatch is omitted |
| `dijkstra_path`, `dijkstra_path_length` | keep NetworkX's `(G, source, target, weight="weight")` order using `optional<string_view>`; `nullopt` selects unweighted traversal |
| `single_source_dijkstra_path_length` | keeps NetworkX's `(G, source, cutoff=nullopt, weight="weight")` order; a three-argument `string_view` compatibility overload spells the old C++ call |
| `shortest_simple_paths` | lazy overload keeps NetworkX's `(G, source, target, weight=nullopt)` order; an options overload adds C++ stopping controls, while the Virne convenience overload adds `k` and materializes the first `k` paths |
| node/edge attribute APIs | add `AttrId` overloads; getters omit Python's `default` and setters require an attribute name/ID. Edge-map iteration is keyed by `(u, v)` endpoints like NetworkX; stable `uint32_t` ID lookup/setter overloads remain compatibility accelerators |
| node labels / `add_edge` | `Vertex` is a dense index; an endpoint gap materializes `0..max(u,v)`, unlike NetworkX's arbitrary/sparse Python labels |
| `subgraph`, `subgraph_view` | return a read-only structural view backed by a dense `SearchMask`; predicate variants rebuild the mask on every `mask()` call so current attributes and edge additions/removals are visible |
| `convert_node_labels_to_integers` | supports NetworkX's four ordering strings and optional old-label attribute, but `first_label` must be zero; after a degree-based permutation, public node/edge iteration is normalized to numeric core order rather than retaining NetworkX's nonnumeric insertion order |
| `adjacency_matrix` | returns Virne `SparseMatrix` and does not expose `nodelist`/`dtype` |
| `to_scipy_sparse_matrix` | returns the local `CSRMatrix`; accepts the original Virne `format="csr"` case and rejects other formats instead of depending on SciPy |
| `floyd_warshall` | returns a dense `DistanceMatrix` rather than a Python mapping of mappings |
| all centrality APIs | return `NodeScores` indexed by contiguous `Vertex`, rather than dictionaries keyed by Python node objects |
| `eigenvector_centrality` | returns `NodeScores`, defaults `max_iter` to NetworkX's `100`, and omits `nstart`/weighted mode |
| `closeness_centrality` | returns all scores in `NodeScores` and omits single-node, distance-attribute, and `wf_improved` options |
| `betweenness_centrality` | is always normalized, defaults to unweighted (`""`), accepts an explicit attribute-name string, and omits sampling/endpoints/seed options |
| `read_gml` | supports filesystem paths and `label="id"`; explicit directed/auto names replace Python's dynamic graph result; arbitrary string node labels, compressed streams, destringizer callbacks, and multigraphs are outside the dense-index core |
| `write_gml` | supports filesystem paths and omits Python's `stringizer` callback |
| `path_graph`, `star_graph`, `grid_2d_graph` | integer forms only; grid tuple labels are converted to row-major contiguous indices |

### Public NetworkX overload inventory

Unless a row says otherwise, every family below has concrete `Graph` and
`DiGraph` overloads with the same argument order and defaults. Exact return
aliases and map types remain defined in the linked header.

| Header/family | Frozen overload surface |
|---|---|
| [`nx/attributes.h`](nx/attributes.h) node get/set | `get_node_attributes(g, AttrId)` and `(g, name)`; `set_node_attributes(g, values, AttrId)` and `(g, values, name)` |
| [`nx/attributes.h`](nx/attributes.h) edge get | `get_edge_attributes(g, AttrId/name)` plus compatibility `get_edge_attributes_by_id(g, AttrId)`; all iterate endpoint keys and retain the stable-ID side index |
| [`nx/attributes.h`](nx/attributes.h) edge set | endpoint-map and `uint32_t`-ID maps with either `AttrId` or `name`; explicit `set_edge_attributes_by_id` spellings remain supported |
| [`nx/sparse.h`](nx/sparse.h) adjacency | `adjacency_matrix(g, weight_attr="weight")`; `to_scipy_sparse_matrix(g, weight_attr="weight", format="csr")` |
| [`nx/sparse.h`](nx/sparse.h) attributes | `attr_sparse_matrix(g, AttrId/name, normalized=false, rc_order={})` |
| [`nx/centrality.h`](nx/centrality.h) | `degree_centrality(g)`; `eigenvector_centrality(g, max_iter=100, tol=1e-6)`; `closeness_centrality(g)`; `betweenness_centrality(g, weight_attr="")` |
| [`nx/connectivity.h`](nx/connectivity.h) | `is_connected(g)`; the `DiGraph` overload is weak connectivity |
| [`nx/relabel.h`](nx/relabel.h) | `convert_node_labels_to_integers(g, first_label=0, ordering="default", label_attribute=nullopt)` |
| [`nx/subgraph.h`](nx/subgraph.h) | endpoint-predicate, edge-ID-predicate and fixed-`SearchMask` `subgraph_view` construction, plus `subgraph(g, nodes)` |

These rows are an inventory, not permission to perform their string-taking
forms repeatedly in a hot loop; the mandatory ID contract above still
applies.

### Endpoint-keyed edge attributes

Both name and `AttrId` getter overloads iterate an `EdgeAttributeMap` in graph
edge order with NetworkX-shaped endpoint keys. `EdgeAttributeKey::first` and
`second` are the endpoints; its stable ID and implicit `uint32_t` conversion
exist only for old indexed callers. `at`, `find`, `contains`, and `count`
accept either endpoint keys or edge IDs. Map iteration is const so mutating a
key cannot invalidate its lookup index.

Endpoint setter overloads accept
`std::unordered_map<EdgeEndpoints, AttrValue, nx::EdgeEndpointHash>` and ignore
keys for absent edges as NetworkX does. Explicit `uint32_t` setter overloads
and `set_edge_attributes_by_id` remain for already-resolved hot paths; a stale
or absent ID is an error. Public code may query by strings, but a hot loop must
resolve the name once to `AttrId` and operate on indices/direct attribute
storage thereafter. `set_node_attributes` likewise ignores entries for node
indices not present in the graph instead of indexing dense storage out of
bounds.

Sparse results use SciPy-compatible row/column ordering. `adjacency_matrix`
throws on an empty graph and defaults a missing weight to `1.0`.
`attr_sparse_matrix` likewise defaults missing values only for the attribute
named `"weight"`; a missing custom requested attribute is an error. Normalized
rows first drop explicit zero entries, sort by row/column, sum in column order,
compute one reciprocal per row, and multiply each entry exactly as the
NetworkX/SciPy oracle does. This preserves its floating-point rounding and its
IEEE non-finite results when nonzero emitted entries cancel to a zero row sum.

| API | Directed behavior |
|---|---|
| `nx::shortest_path_length` | outgoing unweighted paths |
| `nx::shortest_path` | outgoing unweighted paths |
| `nx::single_source_shortest_path_length` | reachable successors only |
| `nx::dijkstra_path` | outgoing weighted paths |
| `nx::dijkstra_path_length` | outgoing weighted paths |
| `nx::single_source_dijkstra_path_length` | reachable successors only |
| `nx::floyd_warshall` | ordered all-pairs distances |
| `nx::shortest_simple_paths` | first `k` directed Yen paths |
| `nx::get_node_attributes`, `nx::set_node_attributes` | identical to `Graph` |
| `nx::get_edge_attributes`, `nx::set_edge_attributes` | endpoint keys are ordered `(source, target)` pairs; ID overloads identify arcs without changing endpoint-keyed iteration |
| `nx::adjacency_matrix` | one coordinate per arc |
| `nx::attr_sparse_matrix` (Virne extension) | one coordinate per attributed arc |
| `nx::degree_centrality` | total degree divided by `n - 1` |
| `nx::eigenvector_centrality` | NetworkX left eigenvector (predecessor influence) |
| `nx::closeness_centrality` | NetworkX inward distance convention |
| `nx::betweenness_centrality` | ordered directed pairs; Brandes scaling |
| `nx::is_connected` | weak connectivity for `DiGraph` |

### Original Virne graph-call coverage

A source scan of the Python Virne tree found the following non-visual graph
surface, all implemented and covered by a compile/unit/differential test:

- `Graph`, `DiGraph`, node/edge/adjacency/degree views, bulk construction and
  `neighbors`;
- shortest path/length, all shortest paths, Dijkstra, Floyd-Warshall and
  shortest simple paths;
- node/edge get/set attributes, induced/predicate subgraphs and integer
  relabelling;
- adjacency/attribute/CSR sparse matrices and connectivity;
- degree, eigenvector, closeness and betweenness centrality;
- path, star, grid, Erdős-Rényi, connected Erdős-Rényi and Waxman generators;
- GML read/write, including the directed flag and recursive graph/node/edge
  metadata used by saved datasets.

`nx.draw`, `nx.draw_networkx_nodes`, and `nx.spring_layout` occur only in the
Python Matplotlib visualization helper. They are deliberately not graph-core
APIs and are not ported. This avoids adding a rendering/layout dependency that
the simulator, solver, dataset generation and future LibTorch adapter do not
use.

Seeded generator parity compares node order, edge orientation/order, graph
kind and generated attributes against NetworkX 3.4.2. The suite includes
shared `PyRandom` streams across consecutive calls, seeds above 32 bits, empty
and complete boundary cases, and directed graphs; it currently passes 91/91
exact cases.

NetworkX itself exposes `is_weakly_connected` rather than accepting a
`DiGraph` in `is_connected`.  The C++ library has one pre-existing
connectivity entry point, so its documented directed mapping is weak
connectivity.

The subset also preserves its existing C++ edge-case behavior: `is_connected`
returns `true` for an empty graph; `eigenvector_centrality` returns an empty
vector for an empty graph and the last iterate after `max_iter`. NetworkX's
Python functions raise in those cases. These differences are frozen rather
than hidden behind new exception types.

For an undirected self-loop, sparse matrices emit one diagonal coordinate.
For a directed graph, reverse coordinates are never synthesized.

Predicate `subgraph_view` objects retain the graph by non-owning pointer and
retain copied `std::function` predicates. Calling `mask()` costs `O(V + E)` and
re-evaluates those predicates; algorithms materialize it once before their hot
loop. The explicit `subgraph_view(graph, SearchMask)` overload is intentionally
a fixed snapshot. `nx::subgraph(graph, nodes)` fixes membership to requested
nodes that existed at construction, like NetworkX's induced node set, while
new or removed edges among those nodes are reflected on the next `mask()`.

`FilteredGraphView<GraphType>` publicly exposes `graph()`, `mask()`,
`contains_node(v)`, and `contains_edge(u,v)`; `GraphView` and `DiGraphView` are
its supported aliases. Predicate construction has both endpoint and stable-ID
forms for each graph type:

```cpp
GraphView nx::subgraph_view(
    const Graph& graph, NodeFilter filter_node = {},
    EdgeFilter filter_edge = {});
DiGraphView nx::subgraph_view(
    const DiGraph& graph, NodeFilter filter_node = {},
    EdgeFilter filter_edge = {});
GraphView nx::subgraph_view_by_id(
    const Graph& graph, NodeFilter filter_node,
    EdgeIdFilter filter_edge);
DiGraphView nx::subgraph_view_by_id(
    const DiGraph& graph, NodeFilter filter_node,
    EdgeIdFilter filter_edge);
GraphView nx::subgraph_view(const Graph& graph, SearchMask mask);
DiGraphView nx::subgraph_view(const DiGraph& graph, SearchMask mask);
```

`EdgeIdFilter` receives `(u, v, edge_id)`. It is the required predicate form
when an edge attribute participates in mask construction: capture its
`AttrId`, read the descriptor/property by ID, and do not resolve a string in
the callback. Neither view owns its graph; a view must not outlive it.

## Generator inventory

The exact declarations are in
[`generators/topology_generators.h`](generators/topology_generators.h) and
[`generators/waxman_generator.h`](generators/waxman_generator.h).

| Family | Supported public overloads and defaults |
|---|---|
| deterministic topology | `path_graph(num_nodes)`, `star_graph(outer_nodes)`, `grid_2d_graph(rows, columns, periodic=false)`; all return `Graph` |
| Erdős–Rényi | `erdos_renyi_graph(n,p)` plus `uint64_t seed` and `PyRandom&` overloads; matching directed calls use `erdos_renyi_digraph` and return `DiGraph` |
| retry | `connected_retry(function<Graph(size_t)>, max_attempts=10000)` |
| connected Erdős–Rényi | `(n,p,uint64_t seed,max_attempts=10000)` and `(n,p,PyRandom&,max_attempts=10000)`; Graph-only |
| Waxman | `WaxmanConfig{num_nodes=100,beta=0.4,alpha=0.1,seed=42}`; `WaxmanGenerator::generate(config)` and `(config,PyRandom&)`; `nx::waxman_graph(n,beta=0.4,alpha=0.1)` and `(n,beta,alpha,PyRandom&)` |
| connected Waxman | `connected_waxman_graph(config,max_attempts=10000)` and `(config,PyRandom&,max_attempts=10000)`; Graph-only |

The two-argument Erdős–Rényi and three-argument Waxman convenience calls use
the process-global `PyRandom` stream for `seed=None` behavior. The explicit
`PyRandom&` overloads preserve caller-owned stream continuity; integer-seed
overloads/configuration construct the frozen Python-compatible stream.

## Cache and I/O

```cpp
class WeightCache {
public:
    explicit WeightCache(const Graph& g);
    AttrId attribute_id(const std::string& name) const;
    double value(const RawNeighbor& edge, AttrId attr_id) const;
    double value(Vertex u, Vertex v, AttrId attr_id) const;
};

class DiWeightCache {
public:
    explicit DiWeightCache(const DiGraph& g);
    AttrId attribute_id(const std::string& name) const;
    double value(const DiRawNeighbor& edge, AttrId attr_id) const;
    double value(Vertex u, Vertex v, AttrId attr_id) const;
};

namespace GraphSaver {
void save_gml(const Graph& graph, const std::string& path);
void save_gml(const DiGraph& graph, const std::string& path);
}

namespace nx {
void write_gml(const Graph& graph, const std::string& path);
void write_gml(const DiGraph& graph, const std::string& path);
Graph read_gml(const std::string& path,
               const std::string& label = "id");
DiGraph read_gml_directed(const std::string& path,
                          const std::string& label = "id");
LoadedGraph read_gml_auto(const std::string& path,
                          const std::string& label = "id");
}

class GmlLoader {
public:
    static Graph load(const std::string& path,
                      const std::string& label = "id");
    static DiGraph load_directed(const std::string& path,
                                 const std::string& label = "id");
    static LoadedGraph load_auto(const std::string& path,
                                 const std::string& label = "id");
};
```

Weight caches are immutable snapshots. Resolve `AttrId` before a loop and
rebuild the cache after any graph or attribute mutation, including changing
an existing attribute value. An attribute absent on an edge occupies `0.0` in
the cache; this differs intentionally from the shortest-path algorithms'
missing-weight default of `1.0`.
The raw-neighbor `value(edge, attr_id)` overload is intentionally unchecked;
the endpoint overload retains snapshot validation.  This mirrors the accepted
`neighbors_fast` risk contract.
The explicit `load_directed` name is required because C++ cannot overload
`load(path)` solely by return type. `load_auto`/`read_gml_auto` return
`std::variant<Graph, DiGraph>` according to GML's `directed` flag. Recursive
list/object attributes are deep-copied, nesting is limited to 256, cyclic
metadata is rejected, and `label="id"` is required by the contiguous-index
core.

`WaxmanGenerator::generate` and `nx::connected_waxman_graph` remain Graph-only.
`nx::erdos_renyi_digraph` is the directed generator and, like the undirected
overload, its two-argument form consumes the process-global `PyRandom` stream
for NetworkX `seed=None`. The original
Python topology call passes positional Waxman parameters as
`waxman_graph(n, wm_alpha, wm_beta)`; preserving an old dataset stream means
preserving that original positional call exactly, including NetworkX's
`beta`-before-`alpha` signature. New `WaxmanConfig` code uses named
`alpha`/`beta` fields and has no ambiguity. Because the supported overload
always infers NetworkX's `L`, fewer than two nodes raise after consuming only
their position draws. A zero probability denominator raises after consuming
the first edge draw, preserving Python's left-to-right evaluation and the
following RNG state.

## Future PyG/LibTorch adapter

The core is intentionally ready for a separate, optional LibTorch adapter:

- contiguous `Vertex` values map directly to tensor row indices;
- `DiGraph::edge_view()` iteration already yields one ordered
  `(source, target)` column per arc for a PyG-style `edge_index` tensor of
  shape `[2, E]`; descriptor-oriented code can instead use
  `edge_view().descriptors()`;
- a `Graph` adapter can materialize both directions, matching PyG's usual
  undirected representation; an undirected self-loop still emits only one
  diagonal column;
- node and edge feature columns can be resolved once by `AttrId`, then copied
  linearly without string lookup in the tensor-building loop; numeric and
  boolean `AttrValue`s can be converted directly, while strings require an
  encoder owned by the adapter;
- edge IDs are stable but may contain holes after removal, so an adapter must
  build `edge_index` and its matching `edge_attr` rows together in the same
  edge iteration rather than using edge ID as a dense tensor row;
- the normal tensor-building path should use endpoint indices and resolved
  `AttrId`s; read-only `raw()` access and stable edge IDs remain an explicitly
  risky escape hatch only if profiling proves that public indexed iteration is
  insufficient. LibTorch itself can stay out of the graph core and be
  vendored/linked manually under `libs` when that adapter is added.

No LibTorch or PyG API is added in this change; this preserves the requested
surface while keeping the memory/index contract suitable for it.

## Stability boundary

Treat `graph/` as a frozen foundation after this release.  New consumers such
as LibTorch/PyG belong in a separate adapter target and must use the public
surface above (`num_nodes`, `num_edges`, `nodes`, `edges`, `source`, `target`,
`node_attrs`, `edge_attrs`, `attr_id`, `attr_name`, `attribute_registry`,
`edge_id`, and `AttrId`; only when profiling proves it necessary, the const
`raw()` overload). They must not add Torch headers, tensor ownership, Python
objects, or framework-specific state to `Graph` or `DiGraph`.

Before implementing any new graph consumer, maintainer or automated agent
**MUST** read this contract and [`../API_MUST_BUILD.md`](../API_MUST_BUILD.md),
identify the boundary where names become IDs, and keep all traversal hot loops
on the indexed surface. Development may add an adapter or consumer without
reopening this foundation. A proposed core change is blocked until its need
cannot be met by the documented facade, an adapter, `SearchMask`, or a resolved
`AttrId`/edge-ID path.

A future adapter may depend on a manually vendored LibTorch tree under
`libs/`, but the graph target itself must remain dependency-free beyond the
pinned Boost headers.  Any unavoidable change to this frozen contract requires
all of the following: an explicit API version decision, Graph and DiGraph
regression tests, the 91-value NetworkX oracle, and the complete benchmark
table, including the 34 completion and 91 generator cases. Adding a consumer
alone is not a reason to modify the graph core.

## Verification

See the [benchmark guide](../benchmarks/README.md) and
[recorded results](../benchmarks/RESULTS.md). The pinned suite checks
91 base values, 34 completion/order workloads, 91 generator/order cases, and
a recursive GML fixture against NetworkX 3.4.2. It benchmarks 32 base and 34
completion rows; every Graph row must be strictly faster than NetworkX.

`test_digraph` exercises the directed core and runs a deterministic 12-graph
differential: single-source BFS/Dijkstra/Floyd are cross-checked against both
bidirectional implementations for every ordered source/target pair, including
masked weighted searches. It is not, by itself, an exhaustive compile test for
every overload in this document. The aggregate CTest suite supplies the
remaining foundation, edge-order, mutable/const view, bulk/subgraph, GML and
sparse coverage; the NetworkX scripts supply value/order differentials and
strict benchmark gates. New overloads must receive an explicit compile/link
call and a behavioral test in the relevant aggregate target rather than
relying on incidental linkage through `test_digraph`.

The release gate additionally requires a clean out-of-tree Release build,
11/11 CTest cases, exact Random differential sequences, and a second clean
11/11 CTest plus Random differential run under ASan/UBSan. LeakSanitizer is
disabled only because it cannot operate
reliably under the runner's ptrace environment; AddressSanitizer and
UndefinedBehaviorSanitizer remain fail-fast.
