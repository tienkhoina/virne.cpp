# `topology_generator` verification results - 2026-07-27

## Environment and protocol

- C++ Release build: `virne-cpp-toolchain:gcc11.4`.
- Python oracle image:
  `virne-python-oracle:py310-nonml`, CPython 3.10.20, NetworkX 3.4.2.
- Oracle source commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Oracle source SHA-256:
  `f69d3c45f288b6194db2945cfa33d2cb4a32032af8130ce4e7a53ab2dd697722`.
- Eight visible CPUs for recorded container runs.
- Canonical rows: five warm-ups, 31 measured samples, median reported.
- RNG preparation/seeding, checksums, serialization, result destruction, and
  process/container startup are outside timed regions.
- Every C++ row must match the Python checksum before a speedup is calculated.

Differential result: **PASS (33 exact cases)**. Release timing/checksum gate:
**PASS for every row**.

## Canonical post-enum/AttrId run

The canonical batch comparison used explicit seven-worker execution so every
batch shared one controlled worker count. Production auto policy was tuned
separately below.

| Operation | Python median ms | C++ median ms | Python / C++ | C++ policy |
|---|---:|---:|---:|---|
| path, 65,536 nodes | 191.156 | 13.485 | 14.18x | sequential |
| star, 4,096 nodes | 6.138 | 3.226 | 1.90x | sequential |
| grid, `256x256` | 497.341 | 30.465 | 16.32x | sequential |
| random, `n=500,p=.03` | 17.835 | 1.605 | 11.11x | sequential RNG-exact |
| Waxman, `n=500`, defaults | 98.489 | 5.734 | 17.18x | sequential RNG-exact |
| batch path, 64 x 2,048 | 412.941 | 9.038 | 45.69x | 7 workers |
| batch star, 64 x 2,048 | 416.163 | 29.894 | 13.92x | 7 workers |
| batch grid, 32 x `64x64` | 832.754 | 14.824 | 56.18x | 7 workers |
| batch random, 32 x `n=250,p=.04` | 178.259 | 10.008 | 17.81x | 7 workers |
| batch Waxman, 16 x `n=250` | 433.312 | 12.846 | 33.73x | 7 workers |

The corresponding C++ sequential batch medians were 20.123, 77.311, 57.292,
20.854, and 28.189 ms for path, star, grid, random, and Waxman. Thus seven-way
batch execution was about 2.2x to 3.9x faster than the measured C++ sequential
batch path in the same canonical run.

## Final worker sweep after enum/ID conversion

Protocol: workers 5 through 8, three rotated/reversed rounds, two warm-ups and
five samples per round (15 measured samples per worker). All checksums matched.
Workers 1 through 8 had already been exercised by the preceding full sweep;
the finalist sweep was repeated after `TopologyRequest` changed from string to
`TopologyType` and attribute reads changed to pre-resolved `AttrId`.

| Workers | Path batch ms | Star batch ms | Grid batch ms | Random batch ms | Waxman batch ms |
|---:|---:|---:|---:|---:|---:|
| 5 | 7.566 | 29.270 | **15.598** | 6.316 | 8.107 |
| 6 | 15.916 | **18.096** | 16.108 | **5.688** | **7.169** |
| 7 | 8.806 | 18.998 | 16.655 | 7.964 | 7.356 |
| 8 | 8.350 | 23.991 | 17.625 | 5.717 | 7.727 |

Every finalist beat C++ sequential in all three rounds. The reference machine
showed substantial scheduler noise (notably path at six workers), so production
uses the measured family policy rather than claiming these numbers are
universal: five workers for homogeneous path/grid, six for homogeneous
star/random/Waxman or mixed batches, bounded by CPU affinity.

## Automatic-policy validation

A post-policy 2-warm-up/5-sample run passed all differential/checksum gates.
Automatic C++ medians were 6.873 ms (path batch), 11.571 ms (star), 13.177 ms
(grid), 4.451 ms (random), and 5.934 ms (Waxman), respectively 39.60x, 25.80x,
40.17x, 25.13x, and 49.46x faster than Python in that validation run.

## Other verification

- Full Release build: PASS.
- Full CTest: PASS, 14/14.
- Unit stress: PASS, 100 fresh processes.
- Warning-clean C++17 syntax build: PASS with
  `-Wall -Wextra -Wpedantic -Werror`.
- Frozen component integrity test: PASS; no changes under `graph/`, `csv/`,
  `config/`, or `libs/yaml-cpp/`.

Measurements are evidence for this compiler/container/CPU allocation. Re-run
the worker sweep whenever allocator, compiler, graph foundation, or CPU
topology changes.
