# Non-ML Python component map

Progress note (2026-08-01): the complete non-solver foundation through
`core.Environment`, LinkRank, NodeRank, BaseSolver, all 14 canonical decorated
classes in `solver.heuristic`, and the native non-ML system/main runtime are
implemented. Accepted component benchmarks remain frozen. The combined node
rank solver differential passes ten shared cases at workers `0/1/2/8`; every
workers=1 row in its frozen compact fixture is faster than Python. The
collective registry gate creates and solves all 14 names and compares exact
workers=1/4 output plus both caller-owned RNG continuations.
The default seed-0 mutable Main integration is exact at 752/248 accepted/
rejected requests and removes the per-request physical clone/double-deploy
bottleneck; its one GCC 11 signal is documented separately from frozen
component benchmarks.

The exact registry, split-capable integral MIP and compact meta-flow GLOP paths are now
implemented at `solver/exact`: `mip`, `d_round` and `r_round` are direct
registry fields, and the native capacity formulation supports every selected
node/link resource instead of Python's single `cpu`/`bw` lanes. The focused
multi-resource unit passes all three names, including forced split flow,
aggregate capacity, typed integer journals and fail-closed numeric/topology
guards. Backend parallelism is exposed as the fixed
`native.workers.exact` field. A small
one-resource MIP smoke structurally matches Python and measures a 1.955914x
median speedup; native Main smokes pass all three names. Different OR-Tools
releases, symmetric mapping choices and cross-language Main/rounding
differentials keep that signal non-canonical; this
note therefore does not mark the leaf complete.

The separate `exact_with_risk` registry entry copies the integral exact model
and changes only its objective. A bounded residual-fragmentation surrogate
breaks equal-flow-length ties without allowing risk to select a longer route;
floating-only flow uses a two-phase primary-objective lock. Its API and focused
proof are in `components/exact_with_risk.md`.

The canonical `solver.heuristic` directory is complete; read
`components/heuristic_registry.md` rather than reopening its implementation.
For exact work, read `components/exact_solvers.md`; finish its listed
validation before moving to meta-heuristics. Reuse completed dependencies and
open their source only where the API document cannot resolve parity or a
measured hot-path optimization. Torch-backed seeding, ML/RL and specialized
Controller MCF remain deferred.

Source inventory was taken at original Virne commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`. The checkout was clean.

## Dependency order

1. Leaf utilities: `utils.class_dict`, `utils.setting`, `utils.network`,
   `utils.stats`, `utils.manager`, topology generator/metrics, then the non-Torch
   part of `utils.dataset`.
2. Attribute model: attribute methods, `BaseAttribute`, node/link/graph
   attributes, factories, `AttributeBenchmarkManager`.
3. Network model: `BaseNetwork`, `PhysicalNetwork`, `VirtualNetwork`, virtual
   events/request simulator, dataset `Generator`.
4. Core/control: `Solution`, constraint checker, resource updater, topology
   analyzer, node/link mapper, controller, counter, recorder and logger.
5. Runtime: environments, base solver/registry, rankers, heuristic solvers,
   exact solvers, meta-heuristics and systems.

The whole `virne/solver/learning/**` tree is excluded for this phase.

## Inventory facts

- Python non-ML scope: 56 substantive modules, about 9,657 lines when package
  initializers are excluded.
- Existing C++ tree originally contained 48 `.cpp` and 48 `.h` stubs, all zero
  bytes, plus five module CMake files.
- 43 C++ stub paths map directly to Python modules.
- At the initial inventory, Python modules without a C++ scaffold included the
  exact solvers (`d_rounding`, `mip`, `r_rounding`), eleven meta-heuristic
  modules, and `changeable_system`/`time_window_system`. Exact now has the
  native aggregate `solver/exact/exact_solver.{h,cpp}`; the inventory fact is
  retained only as baseline provenance.
- C++ orphan stubs without a Python source counterpart are `solver/debug_obs`
  and heuristic `nea_bc`, `nea_par`, `nrm_bc_env`, `nrm_par_solver`; do not
  implement them without first locating their provenance.
- Original tests contain 122 cases, all under the network area. There are no
  original performance tests and no coverage for most utilities/core/solvers.

## Import boundaries to avoid

- `virne/__init__.py` imports the solver registry; the solver initializer imports
  the learning package.
- `virne.utils` imports `dataset.py`, which imports Torch for seeding.
- `dataset_generator.py` imports `torch.seed` even though it is unused.
- `BaseSystem` imports `RLSolver` directly.
- Logger defaults can pull TensorBoard/Torch and WandB.
- The default configuration selects a learning solver.

Use direct source loading for independent Python oracles until these boundaries
are ported. Do not copy the eager ML coupling into C++.

## Known Python behaviors requiring explicit decisions

- Several algorithms assume contiguous node IDs `0..N-1`.
- Attribute factory contains graph classes but does not register graph owner.
- Controller reads several fields from inconsistent config levels and can read
  `constraint_restrictions` from attribute types that do not define it.
- `TopologyAnalyzer` advertises a branch rejected by its own assertion.
- A custom BFS path can reference an uninitialized path when no route exists.
- `write_setting` returns a `ValueError` object instead of raising it for an
  unsupported suffix.
- `get_distribution_average`, unsafe mapping and one time-window transition
  path are incomplete.

Differential tests must first preserve observable behavior. Any bug fix must be
documented and tested as an intentional deviation, not silently folded into a
port.
