# Exact solvers focused validation - 2026-07-31

Status: **focused module gates pass; canonical/full parity remains open**.

## Correctness smoke

`vne_exact_solver_unit` exercises `mip`, `d_round`, and `r_round` with two node
resources (`cpu`, `gpu`) and two link resources (`bw`, `spectrum`). It shifts
physical registry IDs, rejects requests infeasible only in the second resource
lane, checks const immutability and complete journals, fails closed for an
unmodeled non-resource hard constraint, path-level resource and an integer
coefficient above `2^53`. It also forces multi-resource split flow, checks
aggregate reusable capacity, and covers mutable deployment/release with both
caller-owned and solver-owned transactions.

```text
1/1 vne_exact_solver_unit ... Passed
Total Test time (real) = 1.43 sec
```

The 1.43 s value is CTest wall time, including executable launch/test-runner
overhead. It is not a solver-only benchmark and direct executable runtime may
differ.

## Focused Python/C++ timing

Command:

```powershell
python porting\benchmark_exact_solver.py
```

Fixture: four-node physical ring, two virtual nodes and one virtual link, one
node resource (`cpu`) plus one link resource (`bw`). Each runtime used one
warm-up and five measured MIP solves. The comparator matched this structural
signature for every side: request accepted, two node slots, one routed edge.

| Runtime | Backend/package | Median | Minimum |
|---|---|---:|---:|
| Original Python `MipSolver` loaded from pinned source | CPython 3.12.3, OR-Tools 9.12.4544 SCIP | 16.8899 ms | 16.3675 ms |
| Native `mip` | clang-cl 19.1.7 Release, local OR-Tools 9.15.6755 SCIP | 8.6353 ms | 7.7236 ms |

Focused median speedup: **1.955914x**. Native OR-Tools is installed locally at
`libs/ortools-win`; CMake imports that pinned payload rather than a global
package. Placement/path variables are anonymous and retained by numeric table
slot, so the measurement does not include per-variable diagnostic-name
construction.

## Interpretation limits

This is a deliberately small performance/correctness signal, not a canonical
or full differential:

- Python and C++ use different OR-Tools releases, and this host Python is not
  the pinned CPython 3.10 Docker oracle.
- The comparator checks structural output, not the concrete assignment chosen
  among symmetric optimal mappings.
- Only the one-resource `mip` compatibility fixture is timed; the focused unit
  separately covers multi-resource constraints and both rounding registries.
- Native Main smoke runs passed for `mip`, `d_round`, and `r_round` with the
  multi-resource settings. No Main cross-language parity, exhaustive
  differential or frozen canonical benchmark is claimed here.
