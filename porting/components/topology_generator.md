# Component: `network.topology.topology_generator`

State: **COMPLETE** on 2026-07-27.

## Source and target

- Python source: sibling
  `../virne/virne/network/topology/topology_generator.py`, commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Exact source SHA-256:
  `f69d3c45f288b6194db2945cfa33d2cb4a32032af8130ce4e7a53ab2dd697722`.
- The original repository has 19 direct unit tests for this class.
- C++ implementation:
  `virne/network/topology/topology_generator.h` and `.cpp`.
- Isolated target: `vne_network_topology_generator`, linked only to frozen
  `graph_lib`, `random_lib`, and `Threads::Threads`.
- Unit/CTest target: `vne_topology_generator_unit`.
- Differential/timing driver: `vne_topology_generator_harness` plus
  `porting/compare_topology_generator.py`.
- Worker sweep: `porting/sweep_topology_generator_workers.py`.

## Contract ported

| Python type | C++ behavior |
|---|---|
| `path` | `num_nodes` contiguous vertices and path edges |
| `star` | one hub plus `num_nodes - 1` leaves; `num_nodes=1` is valid |
| `grid_2d` | requires `m` and `n`, ignores `num_nodes` after validation, never periodic |
| `random` | connected undirected Erdős-Rényi, default `random_prob=0.5` |
| `waxman` | connected Waxman, defaults `wm_alpha=0.5`, `wm_beta=0.2` |

The original Waxman call is positional:
`nx.waxman_graph(num_nodes, wm_alpha, wm_beta)`. NetworkX binds those values as
`beta=wm_alpha` and `alpha=wm_beta`; the C++ port intentionally preserves this
quirk. It must not be silently corrected when physical-network code is ported.

Default stochastic generation retries without a limit and continues the same
Python-compatible RNG stream across failed candidates and subsequent calls.
`TopologyOptions::max_attempts` is an optional, explicit C++ safety extension;
an unset value retains the Python contract. Retry attempts are never run in
parallel because that would change the accepted graph and post-call RNG state.

## Fixed fields, dynamic names, and hot loops

This component follows `porting/PERFORMANCE_CONTRACT.md`.

- `TopologyOptions` stores fixed configuration as direct typed fields.
- `TopologyRequest::type` is `TopologyType`, not a string.
- String overloads exist for the Python/config boundary, validate
  `num_nodes`, resolve the string once with `topology_type_from_string`, then
  dispatch through `switch(TopologyType)`.
- Batch loops access enum/number/option fields directly; no type string is
  hashed or compared in a request loop.
- Waxman's frozen primitive resolves `pos` once to `AttrId` before its node
  loop. New unit/harness graph loops likewise resolve the optional `pos` ID
  once and use `AttrMap::find(AttrId)`.

## Grid order and representation boundary

NetworkX returns tuple labels `(row, column)`. The frozen C++ graph foundation
uses contiguous numeric vertices, so the oracle explicitly verifies the
mapping `vertex = row * columns + column`. Literal tuple labels are not
representable by `Graph`; this is the established frozen representation
boundary.

The frozen primitive had matching public `EdgeView` order but a different
neighbor insertion order. That difference can change BFS and tie-breaking.
The wrapper therefore builds all vertical edges first and all horizontal edges
second, matching NetworkX 3.4 exactly. Differential records include every
neighbor list; examples for a `3x4` grid are node 1 `[5,0,2]` and node 5
`[1,9,4,6]`.

Zero grid dimensions return an empty graph. Negative dimensions raise instead
of being converted to zero.

## Deterministic concurrency

`generate_batch` is a C++ extension for independent requests. Every request
owns a fixed `TopologyType`, options, and seed. Workers claim request indices
dynamically for load balance, write into pre-sized result/error slots, preserve
input result order, and rethrow the lowest-index failure after all workers
join. No RNG state is shared.

On the eight-CPU reference cpuset, the final post-enum sweep selected five
workers for homogeneous path/grid batches and six for random/star/Waxman.
Mixed batches use six. Automatic width is additionally limited by request
count and Linux process CPU affinity. Explicit worker counts remain available
for retuning on another machine.

The global-RNG overload is deliberately sequential and documented as requiring
serialized caller access.

## Differential coverage

The 33 exact top-level cases cover:

- validation order/messages for zero/negative nodes, unsupported/case-sensitive
  type, missing grid dimensions, and negative grid dimensions;
- path/star single and normal sizes;
- grids `1x1`, `1x4`, `3x4`, `0x4`, ignored `num_nodes`, row-major mapping,
  exact EdgeView and neighbor order;
- random defaults, `p=0`/`p=1`, known multi-attempt seeds, two-call sequences;
- Waxman defaults, positional mapping, known multi-attempt seeds, two-call
  sequences, exact IEEE-754 position bits, and the one-node error path;
- explicit and process-global RNG continuation;
- eight subsequent `getrandbits(32)` values after every relevant call/error;
- seeded batch equality at one and eight workers.

Benchmark checksums include node/edge order, every neighbor sequence, attribute
counts, and raw Waxman position bits. Nothing is sorted to hide an order drift.

## Known frozen-foundation constraint

The frozen `Graph::add_edge` duplicate check scans the hub adjacency for a
large star, so a single giant star scales worse than path/grid construction.
The single-star timing uses 4,096 nodes (still far above normal Virne topology
sizes); path uses 65,536. The graph foundation was intentionally not modified.
The 64-graph star batch at 2,048 nodes demonstrates the parallel production
path and remains substantially faster than Python.

## Verification

- Release GCC 11 build: PASS.
- C++ unit: PASS; 100 fresh-process stress repetitions: PASS.
- Differential: PASS, 33/33 exact cases.
- Canonical timing/checksum: PASS, 5 warm-ups and 31 measured samples.
- Final worker sweep: PASS for worker finalists 5 through 8, three interleaved
  rounds; earlier full sweep also exercised 1 through 8.
- Automatic-policy timing smoke: PASS with exact checksums.
- Full repository CTest: PASS, 14/14.
- `-Wall -Wextra -Wpedantic -Werror`: PASS.
- Frozen graph/CSV/config/yaml-cpp integrity: PASS.

See `porting/results/topology_generator_2026-07-27.md` for measurements and
`porting/README.md` for commands.
