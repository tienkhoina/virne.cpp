# Non-ML Python component map

Progress note (2026-07-29): the complete non-solver/non-system non-ML inventory
through `core.Environment` and the independent non-ML `solver.rank.LinkRank`
and `solver.rank.NodeRank` leaves plus the non-ML `solver.base_solver`
foundation are complete/frozen. The next independent leaf is
`solver.heuristic.node_rank.OrderRankSolver`, implemented over a reusable typed
`BaseNodeRankSolver` engine. Read only completed component/API documents for
dependencies before implementation; open dependency code only where an API is
genuinely unclear or a measured hot-path optimization requires it.
Torch-backed seeding, ML/RL, MCF, candidate-search-dependent heuristics, and
system orchestration remain deferred.

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
- Python modules without a C++ scaffold include exact solvers
  (`d_rounding`, `mip`, `r_rounding`), eleven meta-heuristic modules, and
  `changeable_system`/`time_window_system`.
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
