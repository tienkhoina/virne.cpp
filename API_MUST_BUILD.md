# Required public graph surface

## Canonical documents

Read these before changing this surface:

1. [`DEPENDENCIES.md`](DEPENDENCIES.md) for local-only dependencies and pinned
   memory layouts;
2. [`graph/API.md`](graph/API.md) for the supported Graph/DiGraph semantics and
   signature contract;
3. [`random/README.md`](random/README.md) for the independent canonical Random
   surface and state-consumption contract;
4. [`benchmarks/README.md`](benchmarks/README.md) and
   [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for release gates and the
   recorded baseline;
5. this file for the mandatory build/audit checklist.

The exact C++ declarations in the public headers named by `graph/API.md` are
the signature source of truth.

## Normative indexed-hot-path checklist

- String-shaped public APIs are boundary conveniences. Every repeated node,
  edge, neighbor, path, source, candidate, batch or sample loop **MUST** resolve
  each attribute name to `AttrId` once before entering the loop and **MUST** use
  `at(AttrId)`, `find(AttrId)`, `set(AttrId, ...)` or indexed/direct storage
  inside it.
- A predicate, callback or lambda evaluated during `O(V)`, `O(E)` or repeated
  traversal **MUST** capture its already-resolved `AttrId`; it **MUST NOT** call
  `attr_id`, `at("...")`, `find("...")`, `contains("...")` or another
  string/hash lookup on each invocation.
- An outer algorithm **MUST** pass resolved IDs through internal helpers and
  nested algorithms. It **MUST NOT** re-resolve an attribute for every source,
  spur, candidate, path or retry. NetworkX-shaped public signatures remain
  unchanged; an internal ID-taking implementation is the required pattern.
- `AttrId` is owned by one graph's `AttributeRegistry`. A copied graph and the
  library's documented conversions preserve the registry mapping, but code
  **MUST** resolve the name separately for unrelated graphs and **MUST NOT**
  assume equal numeric IDs imply equal attributes across registries.
- Config/YAML/GML parsing, serialization and a single public API entry are
  boundary-I/O exceptions where names and hashes are allowed. They **MUST** be
  converted to IDs before compute/traversal code; this exception does not cover
  solver loops or repeatedly invoked graph predicates.
- `Vertex` remains a dense index. Stable edge IDs can contain holes after
  removal. Code **MUST NOT** turn either into a string-keyed lookup in a hot
  path.

Violation of this checklist is a design regression even when output tests
still pass.

## Required paired surface

Both `Graph` and `DiGraph` overloads must compile and link for:

- core node/edge/attribute/iterator/ID/raw access, edge-list and dense-matrix
  construction, `number_of_nodes`/`number_of_edges`, and attributed add/bulk
  operations;
- mutable/const node, endpoint-edge and adjacency views, including checked
  boundary access and explicit descriptor ranges;
- BFS, `bfs_nx`, BFS stats, bidirectional BFS;
- Dijkstra, masked Dijkstra, bidirectional Dijkstra, `edge_cost`, `path_cost`
  and `path_prefix_costs` for both graph types; graph-independent `build_path`
  and `join_paths`;
- Floyd–Warshall, Yen paths and candidate generation;
- all functions in `graph/nx/attributes.h`;
- all functions in `graph/nx/shortest_paths.h`;
- all functions in `graph/nx/centrality.h`;
- all functions in `graph/nx/sparse.h`;
- all functions in `graph/nx/subgraph.h` and `graph/nx/relabel.h`;
- `nx::is_connected` (weak connectivity for `DiGraph`);
- `GraphSaver::save_gml`, `nx::write_gml`, and the explicit
  undirected/directed/auto GML readers;
- `WeightCache` and `DiWeightCache`.

The supported topology surface is `nx::path_graph`, `nx::star_graph`,
`nx::grid_2d_graph` (including `periodic`), Erdős–Rényi Graph/DiGraph
overloads, connected retry helpers and Waxman Graph overloads. Seeded and
shared-stream output order is part of the contract and is checked by the
91-case generator oracle.

`WaxmanGenerator` remains intentionally `Graph`-only. `GmlLoader::load` keeps
its original `Graph` return type; directed GML uses `load_directed` because
C++ cannot overload by return type.

The only supported simple-path wrapper is `nx::shortest_simple_paths` from
`graph/nx/shortest_paths.h`. There is no separate global
`shortest_simple_paths` API.

The completed public-header audit found no missing `Graph`/`DiGraph` overload
pair in the supported surface above. Graph-only Waxman APIs and the explicit
undirected/directed/auto GML reader names are documented exceptions, not
missing pairs. Future changes **MUST** repeat this declaration audit and update
`graph/API.md` before extending the supported surface.

## Verification commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cmake --build build --target benchmark_random \
  random_differential_harness -j2
ctest --test-dir build --output-on-failure
.venv/bin/python benchmarks/compare_nx.py
.venv/bin/python benchmarks/compare_graph_completion.py
.venv/bin/python benchmarks/compare_generators.py
.venv/bin/python benchmarks/compare_gml.py --binary build/gml_harness
.venv/bin/python random/differential_test.py \
  --cpp build/random/random_differential_harness
.venv/bin/python random/benchmark_compare.py \
  --cpp build/random/benchmark_random
.venv/bin/python benchmarks/check_view_overhead.py
```

`ctest`, the differential oracles and every strict benchmark gate above are
all required; a successful compile alone does not close the checklist.
