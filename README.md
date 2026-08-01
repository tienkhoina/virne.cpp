# VirneCpp

VirneCpp is the C++17, non-ML port of the Virne virtual-network-embedding
runtime. It keeps the Python-compatible configuration and solver names while
using typed fields, dense graph IDs and local C++ dependencies in runtime
loops. Reinforcement learning and other ML components are intentionally out
of scope for this port.

## Current scope

- Graph and DiGraph primitives, NetworkX-compatible shortest paths, centrality,
  generators, GML and sparse helpers.
- Config/YAML loading and Python-compatible CLI overrides; deterministic random
  streams; CSV; progress and logger backends.
- Physical/virtual network construction, request simulation and online or
  offline system execution.
- Registered heuristics: `order_rank`, `random_rank`, `grc_rank`, `ffd_rank`,
  `nrm_rank`, `pl_rank`, `nea_rank`, `rw_rank`, `order_rank_bfs`,
  `random_rank_bfs`, `rw_rank_bfs`, `random_joint_pr`, `order_joint_pr` and
  `ffd_joint_pr`.
- OR-Tools exact leaves: `mip`, `d_round`, `r_round` and
  `exact_with_risk`. Exact leaves support the prepared multi-resource
  formulation. `exact_with_risk` adds a normalized residual-fragmentation
  tie-breaker without allowing a longer route to beat a shorter route.

The porting status and the non-ML component map are maintained in
[`PORTING_STATUS.md`](PORTING_STATUS.md) and
[`porting/NON_ML_COMPONENT_MAP.md`](porting/NON_ML_COMPONENT_MAP.md).

## Repository guide

| Area | Entry point |
| --- | --- |
| Dependency versions and local archives | [`DEPENDENCIES.md`](DEPENDENCIES.md) |
| Graph API and hot-path contract | [`graph/API.md`](graph/API.md) |
| Config and override semantics | [`config/README.md`](config/README.md) |
| Random stream contract | [`random/README.md`](random/README.md) |
| CSV API | [`csv/README.md`](csv/README.md) |
| Progress/logger behavior | [`progress/README.md`](progress/README.md) |
| Required public surface | [`API_MUST_BUILD.md`](API_MUST_BUILD.md) |
| Port workflow and frozen foundations | [`porting/README.md`](porting/README.md), [`porting/FROZEN_COMPONENTS.md`](porting/FROZEN_COMPONENTS.md) |
| Port performance contract | [`porting/PERFORMANCE_CONTRACT.md`](porting/PERFORMANCE_CONTRACT.md) |
| Python parity and frozen benchmarks | [`benchmarks/README.md`](benchmarks/README.md), [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) |
| Exact solver API | [`porting/components/exact_solvers.md`](porting/components/exact_solvers.md) |
| Fragmentation-aware exact API | [`porting/components/exact_with_risk.md`](porting/components/exact_with_risk.md) |

When a subsystem document and this README differ, the subsystem document and
the public header are authoritative.

Before changing a subsystem, read its dependency and API documents first.
Inspect implementation code only when the documented API is ambiguous or a
measured hot path requires a lower-level optimization; then update the API,
parity and benchmark notes with the change.

## Dependency policy

Production libraries are resolved only from the workspace-local `libs/`
directory. CMake does not use `find_package` or silently fall back to an OS,
Conda or global prefix.

| Dependency | Version | Required local path |
| --- | ---: | --- |
| Boost | 1.85.0 | `libs/boost` |
| yaml-cpp | 0.8.0 | `libs/yaml-cpp` |
| tabulate | 1.4.0 | `libs/tabulate` |
| OR-Tools C++ | 9.15.6755 | `libs/ortools` (Linux) or `libs/ortools-win` (Windows) |

`libs/` is intentionally ignored by Git. A fresh checkout must restore the
pinned archives and verify their SHA-256 values using
[`DEPENDENCIES.sha256`](DEPENDENCIES.sha256); the complete extraction commands
are in [`DEPENDENCIES.md`](DEPENDENCIES.md). The accepted release toolchain
is GCC 11.4/libstdc++ 11 on Ubuntu 22.04. The Python environment is a
test-only oracle and is not linked to production binaries.

## Build in Docker (recommended)

Docker Desktop is the reproducible build environment. From the repository
root:

```powershell
docker compose build
docker compose up -d

docker exec virne-cpp-dev cmake -S /work -B /work/build `
  -DCMAKE_BUILD_TYPE=Release
docker exec virne-cpp-dev cmake --build /work/build -j 8
```

The compose service bind-mounts the checkout at `/work`; build artifacts stay
in the checkout and dependencies stay under `libs/`.

For a host build, use CMake 3.20+, a supported compiler and the same local
dependency paths. CMake selects C++17 globally and C++20 only for the exact
target when building with MSVC:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 8
```

## Tests

Run the focused exact gate after building:

```powershell
docker exec virne-cpp-dev cmake --build /work/build `
  --target vne_exact_solver_unit -j 8
docker exec virne-cpp-dev ctest --test-dir /work/build `
  -R '^vne_exact_solver_unit$' --output-on-failure
```

Run the configured CTest suite when the checkout contains all required
fixtures:

```powershell
docker exec virne-cpp-dev ctest --test-dir /work/build --output-on-failure
```

Python differential tests and the frozen timing gates are documented in
[`benchmarks/README.md`](benchmarks/README.md). Do not compare timings from
different compilers, fixtures or dependency builds.

## Running the native system

The executable accepts the same `key=value` overrides used by the Python
entrypoint. Config-group overrides accept both `group=option` and the
Python-compatible `+group=option` spelling. Native-only controls live below
`native.*` and are resolved once before the request loop.

Example heuristic run with multi-resource settings and eight workers:

```powershell
docker exec -it virne-cpp-dev /work/build/main `
  --config /work/setting/main.yaml `
  solver.solver_name=ffd_rank `
  '+p_net_setting=p_net_setting_multi_resource' `
  '+v_sim_setting=v_sim_setting_multi_resource' `
  p_net_setting.topology.num_nodes=12 `
  v_sim_setting.num_v_nets=3 `
  v_sim_setting.v_net_size.low=2 `
  v_sim_setting.v_net_size.high=3 `
  experiment.seed=0 `
  experiment.save_root_dir=/work/results `
  experiment.run_id=ffd_rank_demo `
  experiment.if_save_config=true `
  logger.level=INFO `
  'logger.backends=[console,file]' `
  '++native.output.report=summary' `
  '++native.progress.enabled=true' `
  '++native.workers.rank=8' `
  '++native.workers.node_candidate=8' `
  '++native.workers.link_topology_constraint=8' `
  '++native.workers.link_candidate=8'
```

Use `++native.output.report=none` for a quiet run. The default logger prints
the resolved config once, and the progress bar emits one flushed in-place
frame per update when attached to a TTY.

## Exact solver with fragmentation-aware tie breaking

`exact_with_risk` is an independent registry name. Its primary objective is
the original route flow; a bounded secondary term scores residual scarcity,
resource imbalance and bridge criticality. The bound is strict for integral
flow, and floating-only flow uses a second locked objective phase. Both modes
require an `OPTIMAL` OR-Tools result.

```powershell
docker exec -it virne-cpp-dev /work/build/main `
  --config /work/setting/main.yaml `
  solver.solver_name=exact_with_risk `
  '+p_net_setting=p_net_setting_multi_resource' `
  '+v_sim_setting=v_sim_setting_multi_resource' `
  p_net_setting.topology.num_nodes=12 `
  v_sim_setting.num_v_nets=3 `
  v_sim_setting.v_net_size.low=2 `
  v_sim_setting.v_net_size.high=3 `
  experiment.seed=0 `
  experiment.save_root_dir=/work/results `
  experiment.run_id=exact_with_risk_demo `
  experiment.if_save_config=true `
  experiment.if_save_p_net=false `
  experiment.if_save_v_nets=false `
  logger.level=INFO `
  'logger.backends=[console,file]' `
  '++native.output.report=summary' `
  '++native.progress.enabled=true' `
  '++native.workers.exact=8' `
  '++native.exact.time_limit_ms=10000' `
  '++native.exact.search_node_limit=5000000' `
  '++native.exact_risk.scarcity_weight=1' `
  '++native.exact_risk.balance_weight=1' `
  '++native.exact_risk.criticality_weight=1'
```

The exact solver API, multi-resource constraints, objective proof and focused
fixture are specified in
[`porting/components/exact_with_risk.md`](porting/components/exact_with_risk.md).

## Design rules for new code

1. Resolve dynamic names to graph-local IDs once at the configuration/model
   boundary. Hot loops use `AttrId`, dense indices and direct fields; they do
   not perform string, YAML or hash-map lookups.
2. Represent fixed schema with typed fields. Use string maps only at public
   configuration or serialization boundaries.
3. Keep solver output and failure behavior compatible with the Python oracle;
   add a compact unit/differential check when a leaf is ported.
4. Preserve the existing benchmark artifacts. Add one focused benchmark only
   after correctness passes, then freeze its result.
