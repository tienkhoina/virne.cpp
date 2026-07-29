# `class_dict` verification results - 2026-07-27

## Environment and protocol

- C++ Release build: `virne-cpp-toolchain:gcc11.4`, GCC 11.4.
- Python oracle image: `virne-python-oracle:py310-nonml`, CPython 3.10.20.
- Original source commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Source SHA-256:
  `19637BBC0D4EFF9F240C5BE6B799B84AF96C7423E8C32AD97626468E3E3DE8FE`.
- Eight CPUs were visible in the recorded containers.
- Canonical rows use an explicit width of eight, five warm-ups, and 31
  measured samples. The table reports medians.
- Fixture construction, source loading, process/container startup, and
  checksum comparison are outside timed regions.
- Every differential record and benchmark checksum must match before timing
  is accepted.

Benchmark checksums guard fixture/order continuity with low overhead. The
16-case tagged differential and full-field batch unit, rather than those
lightweight timing checksums alone, establish structural and identity parity.

Differential result: **PASS (16 exact cases)**. Canonical performance result:
**PASS for all eight rows**.

## Canonical explicit-eight comparison

The first six operations are single-object operations; worker width affects
only the two batch rows. Python's integer-index and attribute operations are
compared with the corresponding resolved C++ paths, while both sides retain
the same logical value/update and checksum.

| Operation | Workload | Python median ms | C++ median ms | Python / C++ |
|---|---:|---:|---:|---:|
| string lookup | 200,000 lookups, 256 fields | 84.607 | 3.161 | 26.77x |
| compact-ID lookup | 200,000 lookups, 256 fields | 374.017 | 0.885 | 422.71x |
| resolved-reference get | 200,000 reads, 256 fields | 58.835 | 0.252 | 233.24x |
| resolved-reference set | 200,000 writes, 256 fields | 108.886 | 0.378 | 288.09x |
| `from_dict` | 256 objects, 128 fields | 5.279 | 2.491 | 2.12x |
| `to_dict` | 128 snapshots, 128 fields plus nested data | 10.569 | 0.745 | 14.18x |
| batch `from_dict` | 512 objects x 64 fields | 6.938 | 1.642 | 4.23x |
| batch `to_dict` | 512 objects x 64 fields | 18.860 | 0.633 | 29.80x |

The C++ hot-loop distinction is intentional. A dynamic string is resolved
once to `ClassFieldId`; code that repeatedly touches the same field should
retain either that ID or a typed reference. The string row measures the
boundary operation, not the recommended inner-loop representation.

## Final worker sweep and automatic policy

The final sweep used 512 objects with 64 top-level fields each, five warm-ups,
31 samples per invocation, and five rotated/reversed rounds. All row metadata
and checksums were invariant for every explicit width and automatic mode.

| Mode | Batch `from_dict` median ms | Batch `to_dict` median ms | Speedup over worker 1 |
|---|---:|---:|---:|
| worker 1 | 2.627515 | 1.338897 | baseline |
| worker 8 | 1.487162 | 0.609931 | 1.767x / 2.195x |
| automatic | 1.524392 | 0.625439 | 1.724x / 2.141x |

Eight workers were the selected single explicit width for both batch
operations. Production automatic mode stays sequential below 8,192 aggregate
top-level fields and uses up to eight workers at or above the threshold,
bounded by item count and CPU affinity. A separate threshold sweep with
64-field objects confirmed sequential automatic execution below the cutoff.
Because nested payload depth is not included in this cheap work estimate,
callers with nested-heavy batches retain an explicit worker override.

The automatic comparator run also passed all 16 exact cases, every timing row,
and every checksum. Its purpose is policy validation; the controlled explicit
eight run remains the cross-language performance table.

## Other verification

- Full Release build: PASS.
- Full repository CTest: PASS, 16/16.
- Concurrent top-level callers, deterministic lowest-input error selection,
  and reentrant custom-copy execution: PASS.
- AddressSanitizer and UndefinedBehaviorSanitizer unit runs: PASS.
- Warning-clean C++17 production/unit/harness builds: PASS with
  `-Wall -Wextra -Wpedantic -Werror`.
- Frozen component integrity: PASS; no edits under `graph/`, `csv/`, `config/`,
  or `libs/yaml-cpp/`.

Measurements describe this compiler, allocator, container, and CPU allocation.
Re-run the sweep when any of those conditions or the value representation
changes.
