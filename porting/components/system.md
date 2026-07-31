# Component API: non-ML system orchestration

State: **IMPLEMENTED / MAIN RUNTIME GATE PASS** on 2026-07-30.

The Python checkout contains a working `BaseSystem`/online path but its import
boundary eagerly loads RL. Its standalone offline, changeable and time-window
modules are empty or incomplete. This native component therefore preserves the
shared non-ML online lifecycle and defines typed deterministic semantics for
the three TODO modes. It never imports or links Torch, CUDA, RL or learning
solvers.

## API

All declarations are in `virne::system`.

```cpp
enum class SystemMode : std::uint8_t {
    online, offline, changeable, time_window
};

struct SystemRunConfig {
    std::size_t num_simulations = 1;
    std::optional<std::uint32_t> seed;
    bool renew_virtual_networks = false;
    bool renew_event_schedule = false;
    network::VirtualSimulationWorkers simulator_workers;
    bool capture_solutions = true;
    SystemProgressSink* progress = nullptr;
    std::size_t progress_epoch_offset = 0;
};

struct SystemProgressUpdate {
    std::size_t epoch_index;
    std::size_t completed;
    std::size_t total;
    std::int64_t success_count;
    std::int64_t inservice_count;
    double long_term_r2c_ratio;
};

class SystemProgressSink {
public:
    virtual void begin_epoch(std::size_t epoch, std::size_t total) = 0;
    virtual void update(const SystemProgressUpdate&) = 0;
    virtual void end_epoch(const SystemProgressUpdate&) = 0;
};

struct SystemStepTrace {
    std::size_t epoch_index;
    network::VirtualRequestId request_id;
    core::Solution solution;
    bool accepted;
    core::EnvironmentFailureReason failure_reason;
    std::size_t auto_released_events;
    double event_time;
    std::size_t stage_index;
    std::size_t window_index;
};

struct SystemEpochSummary {
    std::size_t epoch_index;
    std::size_t arrival_steps;
    std::size_t accepted;
    std::size_t rejected;
    std::size_t auto_released_events;
    core::CounterSummary summary;
};

struct SystemRunResult {
    std::vector<SystemStepTrace> steps;
    std::vector<SystemEpochSummary> epochs;
};

class OnlineSystem : public BaseSystem {
public:
    SystemRunResult run(RandomContext&, const SystemRunConfig& = {});
};

struct OfflineRunConfig {
    std::size_t num_simulations = 1;
    bool capture_solutions = true;
    std::size_t counter_workers = 1;
    SystemProgressSink* progress = nullptr;
};

class OfflineSystem {
public:
    OfflineSystem(const network::PhysicalNetwork&,
                  const network::VirtualNetworkRequestSimulator&,
                  solver::Solver&);
    SystemRunResult run(const OfflineRunConfig& = {});
};

struct ChangeableStage {
    ChangeableStage(core::SolutionStepEnvironment&,
                    SystemRunConfig = {});
    std::reference_wrapper<core::SolutionStepEnvironment> environment;
    SystemRunConfig config;
};

class ChangeableSystem {
public:
    ChangeableSystem(std::vector<ChangeableStage>, solver::Solver&);
    ChangeableRunResult run(RandomContext&);
};

struct TimeWindowRunConfig {
    SystemRunConfig system;
    double window_size = 100.0;
    double window_origin = 0.0;
};

class TimeWindowSystem : public BaseSystem {
public:
    TimeWindowRunResult run(RandomContext&,
                            const TimeWindowRunConfig& = {});
};

struct RuntimeSelections {
    core::controller::ControllerSelection controller;
    core::CounterSelection counter;
};

RuntimeSelections runtime_selections_from_virtual_config(
    const network::VirtualNetworkSimulationConfig&, bool reusable);

struct MainConfig {
    // Other typed runtime fields omitted here.
    core::LoggerConfig logger_config;
    std::string resolved_config_yaml;
    std::string resolved_native_config_yaml;
};

MainConfig main_config_from_hydra(Config&, PyRandom& run_id_random);
YAML::Node python_compatible_config_root(const Config&);
MainRunReport run_main_config(const MainConfig&);
```

`main_config_from_hydra` is a cold mutating adapter, matching Python main: when
`experiment.run_id` is `auto`, it writes the generated ID into the supplied
composed `Config` before any other field or interpolation is resolved. The
same effective tree can then be saved without another clone or patch.
The adapter also applies Python `add_simulation_into_config` once: dataset
paths, all `simulation.*` counts, `simulation.p_net_num_nodes`, and the five
`rl.feature_constructor` summary fields are written into the composed tree
from typed attribute kinds. These fields are cold metadata only; no YAML or
string lookup is introduced into a request loop.

`SystemException` reports typed `SystemErrorCode` and `SystemOperation` for an
empty changeable stage list, an invalid window definition, invalid event time,
or a window-index overflow.

## Semantics and performance rules

- Online resets or explicitly renews the completed Environment, then performs
  one ordered `solve -> step` transition per arrival. Environment owns deploy,
  release, rollback-safe rejection, Recorder and final Counter summary.
- Offline evaluates every request independently against one immutable physical
  snapshot. It clones that snapshot once per run, prepares lightweight numeric
  mutation views once, and restores an exact checkpoint after every request;
  mutable solvers therefore avoid a graph clone per request. The solver remains
  serialized so RNG consumption and solver state are stable.
- Changeable owns no environments. It receives typed stage references and
  configs, runs them in caller order, then flattens results with direct global
  epoch, stage and step indexes.
- Time-window mode retains Environment's ordered mutations and groups adjacent
  arrivals by `floor((event_time - origin) / size)`. It never hashes a time or
  request string and it does not reorder requests merely to gain concurrency.
- Network generation, rank/candidate calculation, mapping and environment leaf
  workers remain the caller-configured worker fields of those completed
  components. Mutation order, shared RNG and mutable solver state are not
  parallelized.

All fixed state is represented by fields/enums. Dynamic solver/config names
resolve once in the completed cold main-config adapter to a `SolverId`; no
system or progress loop performs string lookup. A null progress pointer adds
only one predictable branch per request. Enabled progress consumes the direct
typed Recorder state after each ordered environment step. Offline computes its
Python-compatible active-request count with a departure min-heap only when a
progress sink is present.

`SystemProgressSink*` is a borrowed, synchronous, single-producer callback. It
must outlive `run()` and is called only by the coordinator thread; worker tasks
must never write to it directly. `progress_epoch_offset` changes progress
labels only, letting changeable stages emit monotonically increasing epoch
IDs without changing local result indexes.

Resource IDs are registry-local. `RuntimeSelections::controller` stores the
virtual registry IDs in definition order (`cpu/gpu/ram`, for example); the
prepared Controller resolves each dynamic name once and retains independent
virtual/physical `AttrId` bindings. `RuntimeSelections::counter` deliberately
leaves both optional filters empty, so each virtual or physical registry
selects all of its own `AttributeKind::resource` IDs during preparation. Never
copy a virtual registry ID into the physical Counter: physical extrema entries
may be interleaved with resources and therefore have a different numeric
layout. Prepared count, check, mutation, mapping and rollback loops use IDs
only.

## Main CLI output

At INFO level, native main uses the completed Logger backend to emit the same
startup shape as Python: `Use <name> Solver ...`, followed by `Config:\n` and
the complete Python application tree before network setup or solver execution.
`python_compatible_config_root` removes Hydra internals and the C++-only
`native` control plane; native worker/output controls are emitted immediately
afterward as `Native config`. `MainConfig::resolved_config_yaml` and
`resolved_native_config_yaml` own both fixed cold snapshots. YAML is traversed
and serialized once for logging, never from a request loop. An `auto` run ID is
replaced with the concrete generated ID. Saved `config.yaml` is the
Python-compatible tree; `native_config.yaml` preserves the separate C++
extension for reproducibility.
`logger.backends=[]` or a level above INFO suppresses these messages through
Logger itself and leaves the snapshot empty, avoiding an otherwise unused
whole-tree dump. Main does not bypass the backend with `cout`/`cerr`.
`setup_time_ns` starts before these startup logs, matching the Python setup
comparison; `run_time_ns` starts only at system execution. As with Python's
`resolve=True` dump, resolved environment values are not redacted.

The default main output is intentionally compact and Python-like: one live bar
per epoch on an interactive terminal, one final line per epoch when stderr is
redirected, and one summary JSON object on stdout. It no longer serializes
every node slot and link path by default. Docker needs `exec -t` (or `-it`) to
show in-place updates; without a TTY only the final line is emitted.

```text
++native.output.report=summary   # default: compact JSON
++native.output.report=full      # differential/debug payload
++native.output.report=none      # no stdout report
++native.progress.enabled=true   # default
++native.progress.width=30
++native.progress.minimum_interval_ms=100
```

`full` implicitly captures solutions unless `native.capture_solutions` is
overridden. `summary` and `none` default to no trace capture. The progress bar
shows the same core fields as Python tqdm: acceptance (`ac`), long-term
revenue/cost (`r2c`) and current in-service count. Formatting is rate-limited
outside the solver and mapping hot loops.

The native CLI disables C/C++ stdio synchronization before its first I/O and
detaches `cin`, `cerr` and `clog` ties. Each progress frame is assembled in one
reused buffer, written once to buffered stderr (`clog`) and explicitly flushed
once. The sink reads the current TTY width only on rate-limited refreshes,
reserves the last terminal column, and switches from full to compact/essential
format while shrinking the bar. This prevents Docker/PowerShell auto-wrap, so
TTY frames use `\r` on one physical row and only the final frame adds a newline.
If neither the TTY nor `COLUMNS` exposes a width, live redraw is suppressed and
only one compact final frame is emitted.
Redirected stderr receives only the full newline-terminated final frame. The
summary is flushed once after serialization. Production output does not mix
`printf`/`scanf` with iostreams.

Native `p_net_setting=p_net_setting_multi_resource` replaces the selected group.
Python-compatible `+p_net_setting=p_net_setting_multi_resource` recursively
merges the selected group over the composed default, exactly like Hydra; the
same applies to `v_sim_setting`. Only these two fixed selectors receive this
cold-boundary compatibility path, so `++native.*` and ordinary create-only
overrides retain their existing semantics.

## Current gate

`base_system.cpp`, `online_system.cpp`, `offline_system.cpp`,
`changeable_system.cpp`, `time_window_system.cpp`, and `system_unit.cpp` pass
the host strict syntax gate with conversion/sign/shadow warnings as errors and
the GCC 11 Docker link/run gate. Units cover typed progress for online,
offline, changeable and time-window paths, including global stage epoch labels
and overlapping offline in-service lifetimes. Offline unit coverage also proves
that committed mutable mutations occur, every request/epoch sees the original
snapshot, and the caller p-net remains unchanged. The pinned `order_rank` main
differential is exact. The post-logging one-case gate recorded `19.532x` faster
setup and `23.198x` faster system execution in
`porting/results/system_main_e2e_differential_2026-07-30.json`; it is a
differential timing row, not a retuned benchmark. The earlier representative
run measured about 6.9x faster setup, 15.6x faster system execution and 8.3x
faster total execution than Python. Completed benchmark artifacts were not
rerun or edited for this logging change.

The later default-setting verification intentionally omits explicit physical
node and virtual-request counts. At seed 0, Python and C++ both produce
100/528/1,000 nodes/links/requests, 752 accepted, 248 rejected, acceptance
`0.752`, and `long_term_r2c_ratio=0.48211107669531345`. One GCC 11 container
signal measured 24,024.558 ms Python versus 462.063 ms C++ system time
(`51.994x`). See
`../results/system_transaction_integration_2026-07-30.md`; this did not rerun
or replace any frozen component benchmark.

The multi-resource main gate uses the shipped CPU/GPU/RAM virtual setting and
the physical setting whose extrema definitions are interleaved with those
resources. The `ffd_rank` 20-node/8-request seed-0 fixture passed the full
canonical Python/C++ report comparison: 5 accepted, 3 rejected, acceptance
`0.625`, and long-term r2c `0.6683087027914613`. One non-canonical timing signal
was 54.476 ms Python versus 4.115 ms C++; it is not a frozen benchmark. CTest
`vne_main_multi_resource_smoke` locks the Python-compatible `+group` CLI path.
`vne_main_config_log_smoke` locks the INFO/backend route for the complete
resolved config snapshot.

The progress redraw gate uses 60-column PTYs on Windows and Docker/Linux. The
Docker run produced 41 live frames, exactly one final newline, and a maximum
visible width of 59 columns. MinGW GCC and native clang-cl also compile the
sink; clang-cl links and runs the full CLI.
