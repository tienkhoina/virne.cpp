# NetworkX parity and benchmark harnesses

NetworkX 3.4.2 is the frozen Python oracle. It lives only in the repository
`.venv`; production C++ neither imports nor links Python, NumPy, or SciPy.

## Release interpretation

The comparison gates have deliberately different meanings:

- every measured Graph/DiGraph row in `compare_nx.py` and
  `compare_graph_completion.py`, both GML rows, and all seven measured Random
  rows **MUST** satisfy strict `C++ < Python` timing after parity passes;
- `compare_generators.py` is an exact value/type/order/seed-state oracle, not a
  claim that all 91 generator cases are individually benchmarked;
- public view versus indexed/raw Boost traversal is a zero-overhead gate, not
  a strict speedup gate. Raw traversal is the implementation floor, so node
  attributes, edge attributes, and adjacency are expected to be statistically
  equivalent and may cross `1.0x` in either direction. The robust gate is the
  median of nine runs at or below `1.05x` for each row;
- an unmeasured public overload is correctness-covered but **MUST NOT** be
  described as faster than Python until it receives a paired benchmark row.

Consequently, “all benchmarked Graph/Random rows are faster than Python” is a
valid release statement. “Every public API is faster” and “adjacency view is
strictly faster than raw storage” are not valid statements.

```bash
cd virne.cpp
python3.10 -m venv .venv
.venv/bin/python -m pip install -r benchmarks/requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

All production dependencies still come only from `libs/`. The virtual
environment is a test-oracle exception and is pinned by
`benchmarks/requirements.txt`.

## Base graph oracle and benchmark

```bash
.venv/bin/python benchmarks/compare_nx.py --parity-only
.venv/bin/python benchmarks/compare_nx.py
```

The parity fixture checks 91 deterministic values: Graph/DiGraph structure,
reciprocal arcs, exact path/tie order, weighted and unweighted searches,
centrality and sparse coordinates. Discrete results are exact; floating values
use `rel_tol=5e-6, abs_tol=2e-6` only where NetworkX and C++ operation order can
differ harmlessly.

The benchmark contains 32 rows spanning raw/cache traversal, BFS,
bidirectional BFS, Dijkstra and masks, path helpers, Floyd-Warshall, Yen,
sparse conversion, connectivity and all supported centralities. Graphs,
attribute IDs, views and reusable workspaces are built before timing. Each row
uses three warm-ups and the median of fifteen samples; tiny helpers batch 1024
calls. The script fails unless every C++ row is strictly faster than NetworkX.

## Completion, order and view oracle

```bash
.venv/bin/python benchmarks/compare_graph_completion.py
```

This adds 34 exact workloads for all-pairs return shapes, ordered
`OrderedVertexMap` results, Graph/DiGraph/filtered-view tie order, weighted all
shortest paths, lazy and unweighted Yen behavior, normalized attribute sparse
matrices, dense-matrix construction, endpoint-keyed edge attributes,
`weight=None`, weighted and unweighted single-source cutoffs, `to_directed`, raw in/out traversal,
weighted/unweighted Brandes and seeded topology generators. Every row reports
C++ and NetworkX time and must remain faster on the release machine.

## Seeded generator and GML oracles

```bash
.venv/bin/python benchmarks/compare_generators.py
.venv/bin/python benchmarks/compare_gml.py --binary build/gml_harness
```

The generator suite checks 91/91 exact cases. Equality includes graph kind,
node insertion order, edge orientation/order and generated attributes, not
just an unordered edge set. It covers boundary probabilities, 64-bit Python
seeds and consecutive Graph/DiGraph generator calls sharing one explicit or
process-global (`seed=None`) `PyRandom` stream.

The GML harness round-trips recursive graph/node/edge metadata and checks a
100-node, 500-edge, 9-graph-attribute fixture before timing `read_gml` and
`write_gml` against NetworkX.

## Random, view, progress and CSV microbenchmarks

```bash
cmake --build build --target random_differential benchmark_random \
  graph_view_overhead_benchmark progress_benchmark csv_benchmark -j2
.venv/bin/python random/differential_test.py \
  --cpp build/random/random_differential_harness
.venv/bin/python random/benchmark_compare.py \
  --cpp build/random/benchmark_random
build/graph/graph_view_overhead_benchmark
.venv/bin/python benchmarks/check_view_overhead.py --cpu 8
build/progress/progress_benchmark
build/csv/csv_benchmark
.venv/bin/python csv/benchmark_csv_python.py
```

Random differential testing is bit-exact: 1,260 CPython values across nine
seeds, 2,368 NumPy legacy `RandomState` values across eight seeds, and a
262,145-value large-`randint` output/next-state case. Its seven timing rows use
scalar CPython or one vectorized legacy NumPy call as appropriate and are a
strict `C++ < baseline` gate. The view
microbenchmark compares zero-copy public views to the accepted indexed/raw
Boost path. `check_view_overhead.py` repeats it nine times and enforces the
documented median ceiling; omit `--cpu` when CPU 8 is unavailable. Progress
measures non-rendering update throughput. CSV compares identical 200,000-row
RFC 4180 files with the Python standard library.

Current machine metadata and all recorded tables are in
[`RESULTS.md`](RESULTS.md). Timing values are host-dependent; exact parity,
sanitizer health and the faster-than-NetworkX gates are the durable release
criteria.
