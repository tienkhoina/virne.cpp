# Component: `network.topology.topological_metric_calculator`

State: **COMPLETE** on 2026-07-27.

## Source and target

- Python source: sibling
  `../virne/virne/network/topology/topological_metric_calculator.py`, commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Exact source SHA-256:
  `4bc760a97ab9b7a46ea115db981747a82fdbfed32096e4b327257270625bcf8d`.
- C++ implementation:
  `virne/network/topology/topological_metric_calculator.h` and `.cpp`.
- Isolated target: `vne_topological_metric_calculator`; it links only frozen
  `graph_lib` and `Threads::Threads`.
- Unit/CTest target: `vne_topological_metric_calculator_unit`.
- Differential/timing driver:
  `vne_topological_metric_calculator_harness` plus
  `porting/compare_topological_metric_calculator.py`.
- Worker sweep: `porting/sweep_topological_metric_workers.py`.

## Ported contract

`TopologicalMetrics` retains four independent optional columns: degree,
closeness, eigenvector, and betweenness centrality. `MetricColumn` is a
contiguous float array with the fixed logical shape `(N, 1)`. An enabled
metric on an empty graph is therefore a present empty column, which is
distinct from a disabled metric (`std::nullopt`).

`TopologicalMetricOptions` uses direct typed fields. Static `calculate()`
defaults to all four metrics, min/max normalization, and the automatic worker
policy. `TopologicalMetricOptions::degree_only()` preserves the different
Python constructor default. Calculation and exception order remains degree,
closeness, eigenvector, then betweenness.

The original implementation calls the unweighted/default NetworkX variants:

- degree centrality with self-loops counted twice;
- unweighted closeness with disconnected-component correction;
- unweighted eigenvector power iteration, 100 iterations and tolerance
  `1e-6`;
- exact, normalized, unweighted Brandes betweenness.

The wrapper reproduces NetworkX's null-graph and non-convergence eigenvector
errors. The frozen centrality primitive cannot be called blindly here because
it returns an empty/final iterate in those cases and uses different
floating-point operation order.

## Float32 and exact-output rules

Python converts each NetworkX `double` to `numpy.float32` before finding the
minimum and maximum. C++ does the same conversion first, then performs
subtract/divide in `float`; constant columns become positive zero. This order
is observable. For example, the normalized branch-graph degree midpoint is
bit pattern `1056964607`, one ULP below literal `0.5F`.

Eigenvector iteration follows node and adjacency insertion order and mirrors
CPython 3.10's scaled compensated variadic `math.hypot`. Brandes uses
NetworkX's coefficient operation order and one final normalization multiply.
The gate compares raw `uint32` payloads for every float; there is no tolerance,
rounding, sorting, or approximate checksum.

## Fixed fields, IDs, and hot loops

This component follows `porting/PERFORMANCE_CONTRACT.md`.

- `TopologicalMetricKind` is the compact fixed discriminant used by worker
  policy; no metric-name string enters an algorithm.
- Metric selection, normalization, worker count, result columns, benchmark
  requests, and corpus requests are direct typed fields.
- The calculator is unweighted because the Python component is unweighted, so
  no edge-attribute lookup occurs at all. A future weighted extension must
  resolve its boundary string to `AttrId` once before entering Brandes.
- The only dynamic production string is the cache key. Each cache API call
  performs one boundary hash-table operation; the key never reaches a node,
  edge, source, worker, or normalization loop.

## Deterministic parallel algorithms

Closeness sources are independent. Workers reuse private distance/queue
buffers, claim source IDs atomically, and write directly to the fixed output
slot for that source.

Betweenness sources are also computed independently, but arbitrary reduction
would change floating-point bits. Workers therefore write source contribution
rows into a bounded block buffer. After each block, C++ reduces rows in exact
source order and node order. The buffer is capped at 64 MiB; no lock, string,
or hash lookup occurs in the Brandes hot loop. This was materially faster than
the first exact design that committed each source through a condition
variable.

A persistent executor removes thread-creation cost and serializes concurrent
submissions safely. Results and errors are deterministic. Degree remains a
single contiguous pass, and eigenvector stays sequential because its ordered
power-iteration reductions are observable.

The final five-round/31-sample sweep on the eight-CPU reference cpuset selected
eight workers when closeness or betweenness runs alone and seven when both are
enabled in one calculation. Automatic mode uses those limits, bounded by Linux
CPU affinity and source count; small graphs stay sequential. Explicit worker
counts remain available for retuning.

## Cache identity and synchronization

The process-wide cache stores `shared_ptr<TopologicalMetrics>`, preserving the
same mutable object identity across add/get and overwrite. A shared mutex
protects the map, and clear mutates the existing cache. Concurrent add/get/
clear is tested. As with a Python object, callers must synchronize concurrent
mutation of the returned metrics payload itself. A null pointer is rejected by
the strongly typed C++ boundary.

## Differential coverage

The gate passes **92 exact cases**:

- 28 named cases cover every optional-field state, empty/singleton graphs,
  disconnected components, self-loops, complete/symmetric columns, insertion
  order, constructor defaults, all-flags-disabled, raw/normalized values,
  worker equality, null-graph failure, and 100-iteration non-convergence;
- 64 generated connected graphs use a language-independent uint64 LCG and
  Fisher-Yates edge order, varied size/density/self-loops, alternating raw and
  normalized output, and workers `1/2/4/8`;
- every present flag, logical shape, exception kind, and float32 payload is
  compared exactly against the directly loaded original source;
- timed rows additionally require identical FNV checksums before reporting a
  speedup.

The C++ unit also exercises all 16 metric masks across normalize off/on and
workers `1/2/4/8/auto`, concurrent calculator callers, and cache concurrency.

## Representation boundary

Python `BaseNetwork` derives from undirected `networkx.Graph`; this component
therefore exposes the frozen C++ `Graph`, whose node labels are contiguous
numeric `Vertex` IDs. Arbitrary Python hashable labels and `MultiGraph` are not
representable in the frozen foundation. Edge weight attributes are
intentionally ignored, exactly as in the original calculator calls.

## Verification

- Release build: PASS.
- C++ unit and concurrency matrix: PASS.
- Differential: PASS, 92/92 exact cases.
- Canonical timing/checksum: PASS, five warm-ups and 31 samples.
- Worker sweep: PASS, workers 1 through 8 over three interleaved rounds, then
  five 31-sample finalist rounds.
- Automatic-policy validation: PASS.
- Full repository CTest: PASS, 15/15; unit stress: PASS, 100 fresh
  processes.
- `-Wall -Wextra -Wpedantic -Werror`: PASS.
- Frozen graph/CSV/config/yaml-cpp integrity: PASS; those directories were not
  edited.

See `porting/results/topological_metric_calculator_2026-07-27.md` for recorded
measurements and `porting/README.md` for commands.
