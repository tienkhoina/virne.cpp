# `virne.utils.config` result — 2026-07-29

State: **COMPLETE / FROZEN**.

The leaf reuses frozen `config_lib`, `random_lib`, dataset path builders, and
`AttributeKind`. All fixed schema values are direct fields/enums; extracted
kind membership is a fixed ID-indexed five-slot array. There is no fixed-field
string map and no string lookup in a counting loop.

## Correctness gates

- Release unit: **PASS**, including injected/system run IDs, exact RNG
  continuation, mapping resolution, all summary fields, typed errors,
  worker `0/1/2/8` order, lowest-index failure, and run-directory adapters.
- AST-isolated Python differential: **10 shared + 6 native + 5 documented
  boundaries = 21 PASS**.
- GCC 11.4 strict production and unit syntax compile with
  `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror`:
  **PASS**.
- ASan + UBSan + leak detection unit: **PASS**.
- Targeted CTest: frozen integrity + utils-config unit, **2/2 PASS**.

## Frozen benchmark

Workload: derive 2,048 typed simulation summaries; one warm-up and three timed
samples. Every row retained 2,048 entries and checksum
`6644728919515556009`.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 144.391 ms | 3.973 ms | 36.342x |
| 2 | 144.391 ms | 2.731 ms | 52.867x |
| 8 | 144.391 ms | 3.339 ms | 43.241x |

Worker count remains caller configuration; the library does not embed a
host-derived automatic policy. The accepted benchmark is final and must not be
rerun or modified.

## Frozen artifacts

- Differential JSON SHA-256:
  `E3E158013377503797A10BCCD288EA3DADA040D504E8E95D6C70C8D1C5588CD6`.
- Benchmark JSON SHA-256:
  `1A6E112113FE01AF6D6C6075752C584150BB1D6471C7CE3303B4CCF429691C2F`.
- Header/source SHA-256:
  `FC2997BBDBBCD5FFEAE7C9E49E967A91FD108514BDD1E318FA18B9D1D0235DFA` /
  `82302305D8D4C8BF5AA9A40E2730F422F9BB7F7FB1D93EF577EE100E5167F975`.
