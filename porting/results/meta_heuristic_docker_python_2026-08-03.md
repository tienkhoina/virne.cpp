# Docker meta-heuristic versus Python oracle — 2026-08-03

This is one parity-config benchmark requested after the Docker engine became
available. It uses the original `virne-cpu:latest` image (Python 3.10.20) and
the current GCC 11.4 native executable. No frozen graph/CSV/YAML benchmark row
was rerun.

| Item | Value |
| --- | --- |
| Solver | `ga_meta` |
| Seed | `0` |
| Physical network | Waxman, 20 nodes, CPU/GPU/RAM 50–100, BW 50–100 |
| Virtual requests | 20; random size 2–10; CPU/GPU/RAM 0–20; BW 0–50 |
| Arrival / lifetime | Poisson `lam=0.04` / exponential `scale=500` |
| Native workers | rank/node-candidate/link-topology/link-candidate = 4 |
| Native compiler | Docker GCC 11.4 Release |
| C++ run time | `670.830502 ms` |
| C++ accepted | `3/20` (`0.15`) |
| Python run time | `7.022930861 s` |
| Python accepted | `2/20` (`0.10`) |
| Runtime ratio | C++ `10.47x` faster (`run_time` only) |

The resolved generation fields were made identical before this run. The
acceptance output is **not yet differential-equivalent**: Python's meta leaf
uses separate Python/NumPy random streams, candidate-weighted initialization,
and its own environment/evolution policy, while the native leaf uses the
coordinator-owned `PyRandom` stream and typed deterministic batches. The result
demonstrates Docker build/runtime and performance, not final Python-output
parity. The standalone native worker gate remains passing.
