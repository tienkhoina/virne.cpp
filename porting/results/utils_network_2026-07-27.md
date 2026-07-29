# `utils.network` verification results — 2026-07-27

## Environment

- Host Docker allocation: 8 CPUs; canonical run pinned to CPUs `0-7`.
- C++: Release build in `virne-cpp-toolchain:gcc11.4`, image ID
  `sha256:4a4c20b8a900f81d8cbc8aa7f16150fcd07eb549dd95f256b973a9a4d9bb77d8`.
- Python oracle image ID:
  `sha256:5f9aab417e4750995319dea872241b4c494bb6c63f0132d0c15a34fda726a05e`.
- CPython 3.10.20, NetworkX 3.4.2, NumPy 2.2.6.
- Python source commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- BFS fixture: `datasets/topology/Waxman500.gml`, 500 nodes and 13,573
  edges; parsing is outside timed regions.
- Each canonical row: 5 warm-ups, 31 measured samples, median reported.
  Checksums and serialization are outside the timed region.

Differential result: **PASS (15 groups, including 512 generated float bit
patterns)**. Performance/checksum result:
**PASS**. Full Release CTest result with the sibling dataset mounted read-only
at `/virne`: **PASS (13/13)**. Warning-clean GCC build and 100 repetitions of
the concurrent unit also passed. ThreadSanitizer could not start under this
Docker kernel because of its known runtime memory-map limitation; no TSan pass
is claimed.

## Canonical selected-policy run

| Operation | Python median ms | C++ median ms | Python / C++ | Selected policy |
|---|---:|---:|---:|---|
| `path_to_links` | 28.472 | 0.833 | 34.17x | auto sequential |
| BFS, 50 fixed sources | 12.544 | 0.581 | 21.60x | auto, up to 8 workers |
| BFS, all 500 sources | 124.961 | 5.063 | 24.68x | auto, up to 8 workers |
| `flatten_recurrent_dict` | 4.668 | 0.536 | 8.71x | sequential |
| `flatten_dict_list_for_gml` | 6.579 | 1.102 | 5.97x | auto, up to 8 workers |
| `sanitize_attr_setting` | 5.243 | 0.818 | 6.41x | sequential |

For reference, explicit eight-worker BFS medians were 0.578 ms (50 sources)
and 4.202 ms (500 sources); normal run-to-run scheduling noise explains the
small difference from the equivalent automatic rows.

## Stable worker sweep

Protocol: workers 1 through 8, 3 interleaved rounds, 5 warm-ups and 31 samples
per round: 93 measured C++ samples per worker. Every worker count produced the
same Python checksum.

| Workers | Path ms | BFS-50 ms | BFS-500 ms | GML flatten ms |
|---:|---:|---:|---:|---:|
| 1 | 0.691 | 1.429 | 15.142 | 1.982 |
| 2 | 0.821 | 0.917 | 9.025 | 1.618 |
| 3 | 0.861 | 0.747 | 7.506 | 1.599 |
| 4 | 0.874 | 0.714 | 6.220 | 1.405 |
| 5 | 0.993 | 0.673 | 5.025 | 1.238 |
| 6 | 1.054 | 0.642 | 4.794 | 1.205 |
| 7 | 0.874 | 0.640 | 3.948 | 1.084 |
| 8 | 0.953 | 0.577 | 3.738 | 1.046 |

Aggregated C++ sequential baselines were 0.783 ms for path, 1.418 ms for
BFS-50, 15.299 ms for BFS-500 and 2.068 ms for GML flattening. Therefore the
automatic policy is sequential for path and up to eight workers for BFS/GML.
At eight workers, BFS-500 was 4.09x faster than C++ sequential and 34.55x
faster than Python in the sweep.

Measurements are performance evidence for this machine/container allocation,
not universal constants. Re-run the sweep when CPU topology or compiler changes.
