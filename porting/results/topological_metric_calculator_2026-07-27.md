# `topological_metric_calculator` verification results - 2026-07-27

## Environment and protocol

- C++ Release build: `virne-cpp-toolchain:gcc11.4`.
- Python oracle image: `virne-python-oracle:py310-nonml`.
- Runtime: CPython 3.10.20, NetworkX 3.4.2, NumPy 2.2.6.
- Original source commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Source SHA-256:
  `4bc760a97ab9b7a46ea115db981747a82fdbfed32096e4b327257270625bcf8d`.
- Eight CPUs visible in the recorded containers.
- Canonical rows use five warm-ups and 31 measured samples. Median, MAD, and
  p95 are calculated from per-call samples.
- Graph construction, process/container startup, serialization, and checksum
  calculation are outside timed regions.
- Every output float, optional field, shape, exception, and timed checksum must
  match before performance is considered.

Differential result: **PASS (92 exact cases)**. Canonical performance result:
**PASS for all five rows**.

## Canonical explicit-eight-worker comparison

One explicit width makes the cross-language table controlled and repeatable.
Degree and eigenvector are intentionally sequential regardless of this value;
closeness and betweenness use it.

| Operation | Python median ms | C++ median ms | Python / C++ |
|---|---:|---:|---:|
| degree, 20,000 nodes / 100,000 edges | 4.569 | 0.070 | 65.59x |
| eigenvector, 4,000 / 20,000 | 15.814 | 0.161 | 98.11x |
| closeness, 500 / 2,500 | 110.296 | 0.839 | 131.51x |
| betweenness, 240 / 1,200 | 142.844 | 0.900 | 158.68x |
| all four metrics, 240 / 1,200 | 168.071 | 1.145 | 146.75x |

The corresponding C++ MAD/p95 pairs were 0.000/0.086, 0.025/0.241,
0.042/1.042, 0.046/1.166, and 0.064/1.449 ms. All FNV checksums were identical
to Python's bit-exact float32 result.

## Worker sweep and selected policy

An initial full sweep exercised workers 1 through 8 over three rotated/reversed
rounds. Because its optimum moved under scheduler noise, the final policy used
a stricter finalist sweep: workers 1 and 4 through 8, five interleaved rounds,
five warm-ups and 31 samples per invocation. Every worker/round retained the
same exact checksum.

Compared with C++ worker 1, the reliable selected candidates were:

| Policy workload | Selected workers | C++ speedup over worker 1 |
|---|---:|---:|
| closeness-only | 8 | 3.665x |
| betweenness-only | 8 | 2.282x |
| all-metrics benchmark with one explicit width | 7 | 2.139x |

Across all five rows, worker 8 had the best finalist geometric mean at 1.777x.
Production auto mode uses eight workers when either expensive metric runs
alone and seven when closeness and betweenness are both enabled. Degree and
eigenvector remain ordered sequential operations. The limits respect process
CPU affinity and source count. Below 64 closeness sources or 32 betweenness
sources, automatic mode avoids executor overhead and stays sequential.

## Automatic-policy validation

A separate three-warm-up/11-sample run with `worker_count=0` passed all 92
differential cases and timed checksums.

| Operation | Python median ms | Auto C++ median ms | Python / C++ |
|---|---:|---:|---:|
| degree | 4.468 | 0.123 | 36.35x |
| eigenvector | 14.688 | 0.144 | 101.96x |
| closeness | 111.542 | 1.016 | 109.76x |
| betweenness | 142.524 | 0.886 | 160.87x |
| all metrics | 165.653 | 1.231 | 134.54x |

Python timing variation in this shorter validation run is why the canonical
31-sample table remains the primary cross-language record; this table validates
the production policy rather than replacing it.

## Other verification

- Full Release build: PASS.
- Full repository CTest: PASS, 15/15.
- Unit stress: PASS, 100 fresh processes.
- Unit option/worker/cache/concurrency matrix: PASS.
- Warning-clean C++17 syntax build: PASS with
  `-Wall -Wextra -Wpedantic -Werror`.
- Frozen integrity CTest: PASS; no diff under `graph/`, `csv/`, `config/`, or
  `libs/yaml-cpp/`.
- Python comparator pins the source, NetworkX, and NumPy versions and rejects
  an unexpected Torch import.

Measurements are evidence for this compiler/container/CPU allocation. Re-run
the worker sweep when the allocator, compiler, graph foundation, or CPU
topology changes.
