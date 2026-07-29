# Component: `virne.utils.network`

State: **COMPLETE** on 2026-07-27.

## Source and target

- Python oracle source: sibling
  `../virne/virne/utils/network.py` at commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- C++ implementation: `virne/utils/network.h` and
  `virne/utils/network.cpp`.
- Isolated target: `vne_utils_network` (links only frozen `graph_lib` and
  `Threads::Threads`).
- Unit target/CTest: `vne_utils_network_unit`.
- Cross-language driver: `vne_utils_network_harness` plus
  `porting/compare_utils_network.py`.
- Worker sweep: `porting/sweep_utils_network_workers.py`.

## Ported contracts

| Python function | C++ API | Notes |
|---|---|---|
| `path_to_links` | `path_to_links` | stable indexed output; auto stays sequential |
| `get_bfs_tree_level` | overloads for `Graph` and `DiGraph` | exact discovery/level order from frozen graph API |
| — | `get_bfs_tree_levels` | deterministic C++ batch extension, index `i` always belongs to `sources[i]` |
| `flatten_recurrent_dict` | `flatten_recurrent_dict` | insertion-order recursive flatten |
| `flatten_dict_list_for_gml` | same name | Python scalar/string/list/dict repr and key collision behavior |
| `sanitize_attr_setting` | same name | mutates and returns the same object; preserves partial mutation before an error |

`DynamicValue`/`DynamicKey` retain insertion order and the scalar surface used by
Virne settings. GML conversion covers Python quote selection, ASCII control
escaping, float formatting thresholds, NaN/Inf/negative zero, numeric keys and
collisions after `str(key)`.

## Determinism and concurrency

The persistent executor is lazy and grows only to the requested width, up to
eight reusable workers. Automatic width respects Linux process CPU affinity to
avoid oversubscribing a CPU-limited container. Calls are serialized at the shared executor boundary;
each call splits into contiguous index blocks, stores results in pre-sized
slots, and rethrows failures by lowest worker/block order. Concurrent callers
were exercised by the unit test.

- Single-source BFS remains sequential because its traversal order is
  observable.
- Batch BFS and GML dict-list conversion auto-use up to eight workers.
- Path conversion defaults to sequential: a three-round 1..8 sweep found no
  reliable parallel win for the 262,145-node contiguous workload.
- Recursive flatten and sanitize remain sequential because traversal and first
  failure/partial mutation are observable.

## Differential coverage

The 15 top-level parity groups include:

- normal/short paths;
- undirected, directed, isolated and invalid-source BFS;
- worker-count equality and concurrent callers;
- nested/empty/null recursive values;
- GML booleans, `None`, list/dict repr, both quote styles, control bytes,
  float thresholds, subnormal/min/max, NaN/Inf/-0.0, scalar keys and collisions;
- a deterministic 512-bit-pattern floating-point corpus compared against
  Python `str(float)`;
- sanitize strings with signs/whitespace/underscores, floats, booleans,
  invalid values, int64 overflow guard, identity and partial mutation.

The benchmark gate also compares exact checksums on every timed workload.

## Representation boundary

The production C++ graph uses numeric contiguous `Vertex` IDs, so
`path_to_links` accepts `std::vector<Vertex>` rather than arbitrary hashable
Python labels. Sanitized integers are constrained to signed 64-bit values;
Python integers outside that range raise a documented C++ range error. Unicode
numeric spellings are outside the Virne GML/config contract. Exception classes
are language-appropriate (`AssertionError`/NetworkX errors versus standard C++
exceptions), while success/error behavior and mutation order are preserved.

## Verification result

- Release build: PASS (GCC 11 toolchain container).
- C++ unit: PASS.
- Differential: PASS, 15/15 groups.
- Performance/checksum gate: PASS for all five functions.
- Full worker sweep: PASS, workers 1 through 8, 3 rounds and 31 samples per
  worker.
- Full repository CTest with the original dataset mounted at `/virne`: PASS,
  13/13 tests.
- `-Wall -Wextra -Wpedantic -Werror`: PASS for implementation and harness.
- Concurrent unit stress: PASS, 100 fresh-process repetitions. A TSan binary
  was built, but the Docker runtime cannot execute TSan (`unexpected memory
  mapping`), so TSan is not counted as a passed gate.

See `porting/results/utils_network_2026-07-27.md` for exact measurements and
`porting/README.md` for commands.
