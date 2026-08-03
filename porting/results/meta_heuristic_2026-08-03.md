# Meta-heuristic focused gate — 2026-08-03

This is the single focused benchmark for the newly ported meta-heuristic
leaf. The frozen graph, CSV, YAML and previous solver benchmark rows were not
rerun.

| Item | Value |
| --- | --- |
| Compiler | MinGW-w64 GCC 14.2.0, Release host validation (CMake C++20; meta leaf is C++17-compatible) |
| Target | `vne_meta_heuristic_unit` (isolated `vne_meta_heuristic` library) |
| Fixture | 4-node physical complete graph, 2-node virtual graph, CPU/BW resources |
| Algorithms | `ga_meta`, `sa_meta`, `ts_meta`, `pso_meta`, `aco_meta` |
| Population / iterations | 4 / 3 (focused correctness gate) |
| Worker runs | 1 and 4 outer evaluation workers; controller candidate/topology width matched |
| workers=1 | 3.8750 ms |
| workers=4 | 11.2293 ms |
| Output gate | PASS: all five solution signatures identical |
| RNG gate | PASS: caller-owned `PyRandom` continuation identical |

The four-worker run is slower on this intentionally tiny fixture because the
executor startup/synchronization cost dominates. The important contract is
that the same deterministic batch path is used for larger requests; the unit
does not tune or rewrite the benchmark to hide this overhead.
