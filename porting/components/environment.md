# Component API: `core.environment` (non-RL lifecycle)

State: **COMPLETE / FROZEN (NON-RL)** on 2026-07-29, with the additive typed
transaction seam documented below on 2026-07-30. The accepted component
benchmark remains frozen.

Python oracle: `../virne/virne/core/environment.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`6004CFF2114E504E30C5490232763F71CB9E7799216C4F6CE8BADC27A0E42B34`,
21,138 bytes and 481 physical lines.

## Scope

This port owns the typed event/reset/admission lifecycle shared by
`BaseEnvironment` and `SolutionStepEnvironment`: physical-network reset,
simulator event binding, prepared Counter/lightweight mutation-view reuse,
caller-supplied Solution admission, deploy/release, Recorder commit, automatic
consecutive leave processing, and a typed final summary.

It deliberately does not own a solver, candidate search, MCF, observation,
reward, action mask, feature construction, ranking, Torch, RL, training,
logging backends, or system orchestration. `JointPRStepEnvironment` remains
deferred. A solver normally inspects const views and submits a detached
Solution. Native mutable solvers receive only an explicit transaction object;
Environment owns its commit/bit-exact rollback lifecycle.

## Fixed schema and prepared-ID rule

Environment phase, operation, errors, failure reason, event state, admission
policy, workers, reset policy, transition result, and summary are direct
fields/enums. Request IDs are genuinely dynamic numeric values. Reset resolves
request ID to request index once and writes one dense request-index slot per
event. The event loop performs no string or hash lookup.

Counter and `PreparedControllerMutation` prepare once per virtual request after
every physical clone or simulator renewal. The mutation view prepares only the
ResourceUpdator and pre-resolved resource IDs; it does not eagerly construct
mappers or topology analyzers. Recorder prepares its physical Counter once
during reset. Prepared objects never survive a network replacement. Event
mutation, deploy/release, Recorder commit, history append, and first-error order
remain sequential. Caller worker widths are forwarded only to completed
components; no host-dependent worker policy is introduced.

## Stable native API

All names below are in `virne::core`.

```cpp
enum class EnvironmentPhase : std::uint8_t {
    unready, active, finished
};

enum class EnvironmentFailureReason : std::uint8_t {
    none, early_rejection, placement, routing, unknown
};

struct EnvironmentWorkers {
    std::size_t counter_workers = 1;
    std::size_t recorder_workers = 1;
    std::size_t mutation_workers = 1;
};

struct EnvironmentAdmissionPolicy {
    double r2c_ratio_threshold = 0.0;
    std::size_t virtual_network_size_threshold = 10000;
};

struct EnvironmentConfig {
    controller::ControllerSelection controller;
    CounterSelection counter;
    RecorderConfig recorder;
    EnvironmentWorkers workers;
    EnvironmentAdmissionPolicy admission;
};

struct EnvironmentResetOptions {
    std::optional<std::uint32_t> seed;
    bool renew_virtual_networks = true;
    bool renew_event_schedule = true;
    network::VirtualSimulationWorkers simulator_workers;
};

struct EnvironmentEventState {
    std::size_t schedule_index;
    network::VirtualEventId event_id;
    network::VirtualEventType type;
    network::VirtualRequestId request_id;
    std::size_t request_index;
    double time;
};

struct EnvironmentState {
    EnvironmentPhase phase = EnvironmentPhase::unready;
    std::size_t num_processed_events = 0;
    std::optional<EnvironmentEventState> current_event;
};

struct EnvironmentStepResult {
    RecorderRecord record;
    bool done;
    bool accepted;
    EnvironmentFailureReason failure_reason;
    std::size_t auto_released_events;
    std::optional<CounterSummary> summary;
};

struct EnvironmentDrainResult {
    bool done;
    std::size_t released_events;
    std::optional<CounterSummary> summary;
};

class BaseEnvironment {
public:
    BaseEnvironment(
        network::PhysicalNetwork,
        network::VirtualNetworkRequestSimulator,
        EnvironmentConfig);

    void reset();
    void renew_and_reset(RandomContext&, EnvironmentResetOptions = {});
    EnvironmentDrainResult drain_leaves();

    const EnvironmentState& state() const noexcept;
    const EnvironmentEventState& current_event() const;
    const network::VirtualNetwork& current_virtual_network() const;
    const Solution& current_solution() const;
    Solution make_solution() const;
    const network::PhysicalNetwork& physical_network() const noexcept;
    const network::VirtualNetworkRequestSimulator& simulator() const noexcept;
    const Recorder& recorder() const noexcept;
    CounterSummary summary() const;

    BaseEnvironment(const BaseEnvironment&) = delete;
    BaseEnvironment& operator=(const BaseEnvironment&) = delete;
    BaseEnvironment(BaseEnvironment&&) = delete;
    BaseEnvironment& operator=(BaseEnvironment&&) = delete;
};

class SolutionStepEnvironment final : public BaseEnvironment {
public:
    using BaseEnvironment::BaseEnvironment;
    EnvironmentStepResult step(
        Solution& solution,
        EnvironmentSolutionState state = EnvironmentSolutionState::detached);
};

EnvironmentSolverTransaction BaseEnvironment::solver_transaction();
```

`EnvironmentSolverTransaction` is a non-owning pair of references to the
mutable physical network and prepared numeric-ID mutation view. It has no
RAII destructor; the system explicitly commits or rolls back the mutation at
the arrival boundary. The current virtual network remains available through
`current_virtual_network()`. Full API and ownership rules are in
`transactional_solver_system.md`.

Typed `EnvironmentException` carries `EnvironmentErrorCode`,
`EnvironmentOperation`, and optional schedule/request IDs. Empty schedules,
missing/duplicate request IDs, unknown event requests, event-ID overflow,
non-dense event IDs, wrong phase/type, solution/request mismatch, and incomplete
successful mappings fail without string dispatch.

## Observable lifecycle

Reset first destroys every prepared/current view. The first reset reuses the
owned pristine physical network; later resets move-assign a clone of the
immutable snapshot into the stable physical member. It then resets and binds
Recorder's initial physical baseline, optionally renews the simulator through
the completed `RandomContext`, builds the dense event plan, prepares every
request, and readies schedule slot zero. An empty or invalid schedule fails
after the physical/Recorder reset, matching the useful Python partial order.

Arrival preserves Python order: apply hard-violation and admission gates; run
the prepared virtual Counter once; on success validate complete mappings, set
`Success`, and deploy only a detached Solution; on rejection classify flags in
early/place/route/unknown priority and set the Python description. Recorder
consumes the already-counted fixed fields, appends the arrival, and transition
automatically consumes consecutive leaves. A committed mutable Solution is not
deployed a second time.

A leave creates a fresh Solution, fetches the stored arrival Solution before
mutation through a dense request-index-to-history-index slot, releases with it,
writes `Leave Event` only to the fresh leave
Solution, then Recorder counts and appends that leave. Controller boolean
returns are ignored like Python. Dependency exceptions are not caught: partial
resource/Recorder mutation and the current cursor remain observable. If an
automatic leave throws, `drain_leaves()` exposes the same retry point instead
of forcing a full reset.

When the final event is consumed, Environment computes and retains the typed
Recorder summary but performs no implicit wall-clock formatting, logging, or
filesystem I/O. Explicit frozen Recorder APIs remain available to the caller.

## Deliberate native corrections/boundaries

Python accidentally treats event IDs and request IDs as vector indexes. Native
code advances by schedule cursor and supports sparse/reordered numeric request
IDs after one cold resolution. Event IDs remain validated as dense schedule
indexes because frozen Recorder intentionally uses them as ordered history
indexes; dense shared cases remain identical.

Python replaces the complete physical object on rejection. Native code keeps
the physical object stable and restores selected resource slots from an exact
`AttrValue` checkpoint. This covers mapping failure, solver/deploy/Recorder
exceptions and admission rejection without topology clones or inverse floating
point arithmetic. Fixed targets are encoded by enum, graph ID and `AttrId`; no
string lookup enters transaction, rollback or event loops.

Python's wall-clock/local-time summary, mutable default dict leakage, global
`'p_net'` feature caches, arbitrary DictConfig mutation, debug formatting,
NumPy observation/action objects, assertions used as runtime validation, and
dynamic monkey patches remain outside this typed non-RL core.

## Accepted gate and performance

The exact SHA-locked AST differential passed 9/9 shared lifecycle cases at
workers `1/2/8`. Unit coverage additionally locks workers `0/1/2/8`, sparse and
reordered request IDs, dense event validation, accepted deploy/auto-release,
every rejection priority, hard admission, reset/reprepare, typed schedule and
mapping errors, failed-leave recovery, and independent concurrent environments.
Strict GCC 11 with conversion/sign/shadow warnings, ASan/UBSan/leak checks, and
targeted frozen-integrity CTest pass.

The single accepted benchmark uses 96 eight-node requests and 192 overlapping
arrival/leave events over a 4,096-node/4,095-link physical path; 64 requests are
accepted. It gates 192 ordered records, 72,481 output bytes, full checksum
`17358322786803582063`, and final raw64 physical checksum
`5251282115348753471`. With one warm-up and three samples, Python took
9,675.094 ms; C++ took 65.883 / 254.199 / 993.396 ms at workers `1/2/8`, for
146.854x / 38.061x / 9.739x speedups. Reset/preparation, fixture/input copies,
process startup and checksum work are outside the timer. The accepted benchmark
is now frozen and must not be rerun or edited.

Frozen pre-transaction baseline hashes: `environment.h`
`CE821B41DAF17C76496220B5C17FBA8580123F9FE2D42A71E8AB2FD47488E21D`;
`environment.cpp`
`2EE1C2D9B29A8AA97E3578B781983FBC1E34BE8686E2E6CE710B5C8883C29FF8`.
Compact evidence is recorded in
`../results/environment_2026-07-29.md`; all Environment benchmark and
differential artifacts are frozen with it. The additive transaction integration
is verified separately in
`../results/system_transaction_integration_2026-07-30.md`.
