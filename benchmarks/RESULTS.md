# Release parity and benchmark results

Measured during **2026-07-25 03:26–03:50 UTC** from a clean Release build (`-O3 -DNDEBUG`)
on an Intel Xeon E5-2697 v4 host (64 visible logical CPUs), GCC 11.4.0 and
Linux 5.15.0-185. The repository-local oracle was CPython 3.10.12,
NetworkX 3.4.2, NumPy 1.26.4 and SciPy 1.15.3.

The durable result is:

- base graph parity: **91/91 values exact/within the documented float tolerance**;
- completion and order parity: **34/34 workloads exact**;
- seeded generator parity: **91/91 cases exact**, including edge order;
- GML parity: **100 nodes, 500 edges and 9 graph attributes exact**;
- Graph benchmark gate: **66/66 rows faster than NetworkX**;
- Random differential: **1,260 CPython + 2,368 NumPy values bit-exact**, plus
  a **262,145-value** large-`randint` continuation case;
- Random benchmark gate: **7/7 rows faster than CPython/NumPy**;
- clean out-of-tree Release CTest: **11/11 PASS**;
- full clean ASan/UBSan CTest: **11/11 PASS**, plus the complete Random
  differential oracle.

## Closure re-audit — 2026-07-27 UTC

The finalized source was rebuilt and every Graph/Random oracle and strict
timing gate was rerun. Detailed per-row tables below remain the stable
2026-07-25 baseline; this table records the independent closure run so future
work can distinguish correctness/performance regression from ordinary host
timing variance.

| Gate | Correctness | Measured result | Status |
|---|---:|---:|:---:|
| base Graph/DiGraph | 91/91 values | 32/32 faster; 4.51x–116.26x | PASS |
| completion/order | 34/34 workloads exact | 34/34 faster; 3.60x–542.50x | PASS |
| seeded generators | 91/91 exact, including edge order | parity-only oracle | PASS |
| GML read/write | 100 nodes, 500 edges, 9 graph attrs exact | 4.16x–14.63x | PASS |
| Random differential | 1,260 CPython + 2,368 NumPy + 262,145 large-randint values | bit-exact continuation state | PASS |
| Random measured rows | parity precondition passed | 7/7 faster; 1.20x–53.24x | PASS |
| CTest | 11/11 | Release build | PASS |

The robust view/raw check used nine Release runs pinned to CPU 8. Its gate is
median `view/raw <= 1.05x`, not strict `view < raw`:

| Traversal | Median view/raw | Observed range | Status |
|---|---:|---:|:---:|
| node attributes | 1.001x | 0.943x–1.011x | PASS |
| edge attributes | 0.999x | 0.984x–1.011x | PASS |
| adjacency | 1.000x | 0.927x–1.018x | PASS |

All three public views are statistically at the indexed/raw floor. The two
attribute rows are therefore the accepted parity cases requested by design,
but adjacency must also be described as zero-overhead parity rather than a
stable speedup: its samples cross `1.0x` in both directions.

## Base Graph/DiGraph algorithms

Deterministic directed graphs are built before timing. Attribute names and
reusable workspaces/views are prepared before timing. Each row uses three
warm-ups followed by the median of fifteen samples. Small helpers use 1,024
calls per sample and report milliseconds per call. Every checksum is consumed.
All 32 rows passed `C++ < NetworkX`; speedup ranged from **3.76x to
89.96x**.

| Algorithm / API | Nodes | Edges | Calls/sample | C++ ms/call | NetworkX ms/call | Speedup | Status |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `raw_neighbor_scan` | 5000 | 25000 | 1 | 0.292340 | 9.523073 | 32.58x | PASS |
| `weight_cache_scan` | 5000 | 25000 | 1 | 0.246583 | 16.429092 | 66.63x | PASS |
| `bfs` | 5000 | 25000 | 1 | 0.580283 | 6.963061 | 12.00x | PASS |
| `bfs_nx` | 5000 | 25000 | 1 | 0.823810 | 6.731187 | 8.17x | PASS |
| `bfs_stats` | 5000 | 25000 | 1 | 0.607163 | 6.621087 | 10.90x | PASS |
| `bidirectional_bfs` | 5000 | 25000 | 1 | 0.239697 | 2.998268 | 12.51x | PASS |
| `nx.single_source_shortest_path_length` | 5000 | 25000 | 1 | 0.750309 | 5.325120 | 7.10x | PASS |
| `nx.shortest_path_length` | 5000 | 25000 | 1 | 0.260963 | 2.676225 | 10.26x | PASS |
| `nx.shortest_path` | 5000 | 25000 | 1 | 0.240700 | 2.699782 | 11.22x | PASS |
| `dijkstra` | 5000 | 25000 | 1 | 3.574502 | 32.615483 | 9.12x | PASS |
| `dijkstra_masked` | 5000 | 25000 | 1 | 5.640685 | 87.682938 | 15.54x | PASS |
| `bidirectional_dijkstra` | 5000 | 25000 | 1 | 4.246597 | 19.469164 | 4.58x | PASS |
| `edge_cost` | 5000 | 25000 | 1024 | 0.000182 | 0.000820 | 4.51x | PASS |
| `path_cost` | 5000 | 25000 | 1024 | 0.001034 | 0.016474 | 15.94x | PASS |
| `path_prefix_costs` | 5000 | 25000 | 1024 | 0.000684 | 0.045558 | 66.64x | PASS |
| `build_path` | 5000 | 25000 | 1024 | 0.000512 | 0.007491 | 14.62x | PASS |
| `join_paths` | 5000 | 25000 | 1024 | 0.000201 | 0.001070 | 5.32x | PASS |
| `nx.single_source_dijkstra_path_length` | 5000 | 25000 | 1 | 3.600208 | 72.301636 | 20.08x | PASS |
| `nx.dijkstra_path_length` | 5000 | 25000 | 1 | 2.832681 | 64.225385 | 22.67x | PASS |
| `nx.dijkstra_path` | 5000 | 25000 | 1 | 2.769406 | 82.538801 | 29.80x | PASS |
| `floyd_warshall` | 72 | 360 | 1 | 1.421436 | 127.869136 | 89.96x | PASS |
| `nx.floyd_warshall` | 72 | 360 | 1 | 1.888566 | 108.117582 | 57.25x | PASS |
| `yen_k_shortest_paths` | 120 | 480 | 1 | 1.186546 | 10.752566 | 9.06x | PASS |
| `generate_candidates` | 120 | 480 | 1 | 0.164713 | 2.890019 | 17.55x | PASS |
| `nx.shortest_simple_paths` | 120 | 480 | 1 | 1.203022 | 6.642250 | 5.52x | PASS |
| `nx.adjacency_matrix` | 5000 | 25000 | 1 | 5.968546 | 64.202831 | 10.76x | PASS |
| `nx.attr_sparse_matrix` | 5000 | 25000 | 1 | 6.186940 | 23.274746 | 3.76x | PASS |
| `nx.is_connected` | 5000 | 25000 | 1 | 0.724912 | 15.677415 | 21.63x | PASS |
| `nx.degree_centrality` | 5000 | 25000 | 1 | 0.052127 | 3.323980 | 63.77x | PASS |
| `nx.eigenvector_centrality` | 1500 | 7500 | 1 | 0.613424 | 15.063660 | 24.56x | PASS |
| `nx.closeness_centrality` | 140 | 700 | 1 | 0.941750 | 22.864965 | 24.28x | PASS |
| `nx.betweenness_centrality` | 140 | 700 | 1 | 11.919683 | 212.737779 | 17.85x | PASS |

## Completion, tie order, views and generators

This independent harness checks exact ordered output before reporting time.
All 34 rows were exact and faster than NetworkX; speedup ranged from **9.86x
to 1218.45x**.

| API | Parity | C++ ms | NetworkX ms | Speedup |
|---|:---:|---:|---:|---:|
| `all_pairs_length` | exact | 3.465 | 124.581 | 35.96x |
| `all_pairs_path` | exact | 12.873 | 365.446 | 28.39x |
| `ordered_unweighted_graph` | exact | 0.167 | 6.266 | 37.49x |
| `ordered_unweighted_digraph` | exact | 0.161 | 6.036 | 37.45x |
| `ordered_unweighted_graph_view` | exact | 0.254 | 16.493 | 64.93x |
| `ordered_unweighted_digraph_view` | exact | 0.265 | 14.807 | 55.87x |
| `ordered_weighted_graph` | exact | 0.480 | 13.477 | 28.08x |
| `ordered_weighted_digraph` | exact | 0.371 | 11.201 | 30.19x |
| `ordered_weighted_graph_view` | exact | 0.441 | 31.748 | 71.97x |
| `ordered_weighted_digraph_view` | exact | 0.450 | 25.796 | 57.37x |
| `ordered_all_pairs_graph_view` | exact | 0.258 | 19.452 | 75.34x |
| `ordered_all_pairs_digraph_view` | exact | 0.191 | 11.992 | 62.80x |
| `all_shortest_paths` | exact | 2.990 | 132.226 | 44.22x |
| `all_shortest_weighted_graph_order` | exact | 0.340 | 5.555 | 16.33x |
| `all_shortest_weighted_digraph_order` | exact | 0.277 | 5.000 | 18.07x |
| `all_shortest_weighted_view_order` | exact | 0.196 | 8.741 | 44.53x |
| `filtered_dijkstra` | exact | 28.737 | 1589.919 | 55.33x |
| `lazy_yen_20` | exact | 94.145 | 2670.688 | 28.37x |
| `yen_unweighted_graph_order` | exact | 5.253 | 51.812 | 9.86x |
| `yen_unweighted_digraph_order` | exact | 2.152 | 24.398 | 11.34x |
| `yen_unweighted_view_order` | exact | 1.923 | 34.174 | 17.77x |
| `attr_sparse_normalized` | exact | 17.791 | 3926.193 | 220.69x |
| `dense_matrix_constructor` | exact | 3.984 | 95.865 | 24.06x |
| `edge_attributes_endpoint` | exact | 56.334 | 759.332 | 13.48x |
| `dijkstra_weight_none` | exact | 4.333 | 936.846 | 216.19x |
| `single_source_dijkstra_cutoff` | exact | 1.089 | 42.477 | 39.02x |
| `single_source_shortest_cutoff` | exact | 0.904 | 37.199 | 41.17x |
| `to_directed` | exact | 40.860 | 716.021 | 17.52x |
| `digraph_in_out_fast` | exact | 5.297 | 6453.736 | 1218.45x |
| `betweenness_unweighted` | exact | 10.664 | 216.899 | 20.34x |
| `betweenness_weighted` | exact | 13.412 | 397.754 | 29.66x |
| `erdos_renyi` | exact | 21.074 | 236.663 | 11.23x |
| `connected_erdos_renyi` | exact | 4.878 | 50.637 | 10.38x |
| `waxman` | exact | 21.078 | 260.121 | 12.34x |

## GML

The recursive fixture passed before timing.

| API | C++ ms | NetworkX ms | Speedup | Status |
|---|---:|---:|---:|:---:|
| `read_gml` | 4.256573 | 74.130087 | 17.42x | PASS |
| `write_gml` | 2.261218 | 9.620194 | 4.25x | PASS |

## Boost-backed public view cost

This is a C++-only microbenchmark on a deterministic 2,048-node,
16,384-edge Graph. Each sample alternates ten short raw/view blocks in both
orders, and each invocation performs 400 checksummed traversals. The table is
the median ratio and range across nine CPU-affined Release invocations. A
value above one is overhead, not a NetworkX speedup.

| Traversal | Median view/raw | Observed range |
|---|---:|---:|
| node attributes | 0.991x | 0.949x–1.023x |
| edge attributes | 1.009x | 1.000x–1.051x |
| adjacency | 0.997x | 0.972x–1.078x |

Node data iteration reads `m_vertices[].m_property` directly. Edge data reads
the descriptor property pointer directly. Adjacency uses a pointer-sized raw
neighbor iterator and increments the Boost incidence address directly; the
old ~1.99x adjacency overhead is gone. The three medians are within 1% of raw
and their ranges cross or touch one, consistent with measurement noise.
Extreme hot loops still use validated indices and pre-resolved `AttrId` values.

## Random

Correctness is bit-exact. CPython rows are scalar loops; NumPy rows use one
vectorized legacy `RandomState` call, matching the corresponding C++ vector
overload. Every row is a strict `C++ < CPython/NumPy` release gate; the table
uses seven paired, alternating-order repetitions after one warm-up.

| API | Samples | C++ ns/value | CPython/NumPy ns/value | C++ speedup |
|---|---:|---:|---:|---:|
| `PyRandom.random` | 5000000 | 24.89 | 189.60 | 7.62x |
| `PyRandom.randint(-1000,1000)` | 5000000 | 27.05 | 1139.12 | 42.11x |
| `NumpyRandomState.random` | 5000000 | 21.86 | 25.06 | 1.15x |
| `NumpyRandomState.randint(-1000,1001)` | 5000000 | 12.32 | 14.82 | 1.20x |
| `NumpyRandomState.normal(0,1)` | 1000000 | 61.37 | 79.04 | 1.29x |
| `NumpyRandomState.exponential(1)` | 1000000 | 26.68 | 49.19 | 1.84x |
| `NumpyRandomState.poisson(20)` | 1000000 | 136.75 | 224.55 | 1.64x |

## Progress and CSV

Progress reports the median of five samples from one final Release run.

| Hot-path operation | Calls | Throughput |
|---|---:|---:|
| `Progress::update(absolute)` | 2000000 | 28.7201 Mops/s |
| `Progress::advance(delta)` | 2000000 | 21.6616 Mops/s |
| `TqdmProgress::update(delta)` | 2000000 | 22.4049 Mops/s |
| changed postfix + update | 100000 | 2.80762 Mops/s |

Both CSV implementations produced an identical 14,833,394-byte RFC 4180
file. Each time is a five-sample median.

| Operation | C++ | Python `csv` | C++ speedup |
|---|---:|---:|---:|
| write 200,000 rows | 244.78 ms | 475.64 ms | 1.94x |
| read 200,000 rows | 879.46 ms | 924.96 ms | 1.05x |

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
.venv/bin/python benchmarks/compare_nx.py
.venv/bin/python benchmarks/compare_graph_completion.py
.venv/bin/python benchmarks/compare_generators.py
.venv/bin/python benchmarks/compare_gml.py --binary build/gml_harness
cmake --build build --target random_differential benchmark_random -j2
.venv/bin/python random/benchmark_compare.py \
  --cpp build/random/benchmark_random
```

Timing changes with host load. Exact output/order, sanitizer health, version
pins, and the strict Graph/Random speed gates are the release contract.
