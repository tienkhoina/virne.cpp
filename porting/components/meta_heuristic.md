# Meta-heuristic solver registry

This leaf ports the five canonical non-ML entries that are registered by the
Python package:

| Native name | Algorithm |
| --- | --- |
| `ga_meta` | genetic algorithm |
| `sa_meta` | simulated annealing |
| `ts_meta` | tabu search |
| `pso_meta` | particle swarm optimization |
| `aco_meta` | ant colony optimization |

The unregistered `infeasiblity_servival_genetic_algorithm_solver.py` file is
not a public Python registry entry and is intentionally not exposed here.

## C++ API

`virne/solver/meta_heuristic/meta_heuristic.h` exposes:

- `MetaHeuristicOptions`, with the Python defaults (`population_size=8`,
  `max_iterations=12`) and typed worker widths;
- `MetaHeuristicSolver`, implementing both const `solve` and transactional
  `solve_mutable`;
- `register_meta_heuristic_solvers`, returning fixed `SolverId` fields for all
  five names.

The system resolves the configured string once at startup. Request loops use
the compact `SolverId`; no solver-name lookup or dynamic attribute map is used
in a hot loop.

## Threading and determinism

Random transitions are generated on the coordinator through the shared
`PyRandom` stream. Candidate positions are therefore deterministic and
race-free. Independent candidate evaluations run through
`deterministic_parallel_blocks` with `native.workers.rank`; this includes each
generation/round of GA, SA, Tabu, PSO and ACO. Each evaluation uses a private
physical clone and a private `Solution`. Controller mapping workers are passed
through the existing typed controller options. The executor keeps index order
and collapses nested controller batches on worker threads, so wider evaluation
does not deadlock or alter output.

`native.workers.node_candidate` and `native.workers.link_candidate` are folded
to the larger controller candidate width because the prepared node-slot API
shares one candidate window for node and link mapping. `native.workers.rank`
controls the outer candidate-evaluation batch.

## Verification

`porting/meta_heuristic_unit.cpp` registers and executes every name on a small
multi-resource fixture. It compares worker width 1 and 4 for solution
signatures and the caller-owned RNG continuation, and prints one timing row.
Run it once for the module benchmark after building:

```powershell
docker exec virne-cpp-dev cmake --build /work/build `
  --target vne_meta_heuristic_unit -j 8
docker exec virne-cpp-dev /work/build/porting/vne_meta_heuristic_unit
```

The one recorded run belongs in `porting/results/meta_heuristic_*.md`; old
frozen benchmark rows must not be rerun or rewritten.

## 2026-08-03 hot-path correction

The implementation now prepares the immutable `PreparedCounter` once per
candidate batch, reserves the known solution-map cardinalities, replaces the
quadratic neighbor/repair scans with ordered ID marking, reuses the ACO
candidate/weight buffers, and does not evaluate Tabu individuals for which no
legal move was found. These changes preserve candidate order, random-draw
order, and the worker-width signature gate.

The Docker GCC 11.4 unit was rerun once after the change:

| Worker width | Before | After |
| ---: | ---: | ---: |
| 1 | 15.0056 ms | 9.82711 ms |
| 4 | 3.80198 ms | 2.68363 ms |

Both widths retain identical five-algorithm solution signatures and caller
RNG continuation. The original focused row in
`porting/results/meta_heuristic_2026-08-03.md` remains frozen; this is an
optimization verification row, not a new Python differential claim.
