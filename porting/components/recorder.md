# Component API: `core.Recorder`

State: **COMPLETE / FROZEN (NON-ML)** on 2026-07-29.

Python oracle: `../virne/virne/core/recorder.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`41E66702F29644E6F8A374EF909A1C41F87982558863BE17B6FDF72CEB971F4A`,
15,618 bytes and 338 physical lines.

Recorder reuses the completed Counter, Solution, PhysicalNetwork,
VirtualNetwork, `utils.config`, ClassDict, and frozen CSV APIs. The Python
imports of NumPy, SymPy, the solver registry, and `OrderedDict` are unused.
OmegaConf is only constructor decoding and is replaced by direct config
fields. RL feature/reward/training summary naming remains deferred with ML.

## Fixed schema and hot-path rule

Every recorder state value, initial physical total, event kind, solution
snapshot, path, option, operation, and error is a direct field or enum. Fixed
record columns are never stored in a string-keyed map. Truly dynamic extra
record data uses the completed ClassDict boundary: a name resolves once to
`ClassFieldId`; repeated access is by ID. Event/count/resource/node loops use
only direct fields, numeric node/request IDs, prepared attribute IDs, and
dense slots.

Physical-node service membership is stored by direct physical node ID. The
virtual-request-to-arrival-event index is numeric-keyed because request IDs are
genuinely dynamic. No hot path performs string lookup.

## Stable non-ML C++ API

```cpp
struct RecorderConfig {
    std::filesystem::path save_root_dir;
    std::string solver_name;
    std::string run_id;
    std::string record_dir_name = "records";
    bool temporary_records = true;
};

struct RecorderOptions { std::size_t workers = 1U; };

struct RecorderEvent {
    std::int64_t event_id = 0;
    network::VirtualEventType type = network::VirtualEventType::leave;
};

struct RecorderInitialPhysicalState {
    double available_resource = 0.0;
    double node_available_resource = 0.0;
    double link_available_resource = 0.0;
};

struct RecorderState {
    std::optional<RecorderEvent> event;
    std::int64_t virtual_network_count = 0;
    std::int64_t success_count = 0;
    std::int64_t inservice_count = 0;
    double total_revenue = 0.0;
    double total_cost = 0.0;
    double total_time_revenue = 0.0;
    double total_time_cost = 0.0;
    double long_term_r2c_ratio = 0.0;
    double long_term_time_r2c_ratio = 0.0;
    std::size_t running_physical_node_count = 0U;
    double physical_available_resource = 0.0;
    double physical_node_available_resource = 0.0;
    double physical_link_available_resource = 0.0;
    double physical_node_resource_utilization = 0.0;
    double physical_link_resource_utilization = 0.0;
};

struct RecorderRecord {
    RecorderRecord(RecorderState, Solution, utils::ClassDict = {});
    RecorderState state;
    Solution solution;
    utils::ClassDict extra;
};

enum class RecorderErrorCode : std::uint8_t;
enum class RecorderOperation : std::uint8_t;
class RecorderException : public std::runtime_error;

struct RecorderSummaryColumn { std::string name; std::string value; };
class RecorderSummaryExtension {
public:
    virtual ~RecorderSummaryExtension() = default;
    virtual void append_columns(
        const Recorder&, const CounterSummary&,
        std::vector<RecorderSummaryColumn>&) const = 0;
};

class Recorder {
public:
    Recorder(Counter counter, RecorderConfig config);

    void reset();
    void set_event(RecorderEvent event) noexcept;
    void count_initial_physical_network(
        const network::PhysicalNetwork&, RecorderOptions = {});

    RecorderRecord count(
        const network::VirtualNetwork&,
        const network::PhysicalNetwork&,
        Solution&,
        RecorderOptions = {});
    RecorderRecord count_prepared(
        const PreparedCounter& virtual_counter,
        const network::PhysicalNetwork&,
        Solution&,
        RecorderOptions = {});

    const RecorderRecord& add_record(RecorderRecord record);
    const RecorderRecord& record_by_event(std::int64_t event_id) const;
    const RecorderRecord& record_by_virtual_network(
        SolutionNodeId virtual_network_id) const;
    std::vector<SolutionNodeId> running_physical_nodes() const;

    void temporary_save_record(const RecorderRecord&);
    std::filesystem::path save_records(
        std::string_view filename, RecorderOptions = {}) const;
    CounterSummary summary_records() const;
    std::filesystem::path append_summary(
        const CounterSummary&, std::string_view filename = "summary.csv") const;
    void set_summary_extension(
        std::shared_ptr<const RecorderSummaryExtension>) noexcept;

    const RecorderState& state() const noexcept;
    const std::optional<RecorderInitialPhysicalState>&
        initial_physical_state() const noexcept;
    const std::vector<RecorderRecord>& memory() const noexcept;
    const std::filesystem::path& summary_dir() const noexcept;
    const std::filesystem::path& record_dir() const noexcept;
};
```

The convenience `count` prepares the virtual Counter once for that call.
Performance-sensitive orchestration prepares at its cold boundary and calls
`count_prepared`; every timed reduction then uses only IDs. The physical
Counter is prepared and retained by `count_initial_physical_network`, so later
resource counts observe live values without rebinding names.

`RecorderSummaryExtension` is intentionally a cold, optional serialization
hook. It receives the already-computed typed `CounterSummary` and may append
serialized columns only during `append_summary`; it is never consulted by
`count`, `count_prepared`, membership updates, or history snapshots. No RL,
reward calculation, feature construction, training, solver registry, or ML
dependency is implemented here.

## Observable behavior

- `reset` clears history, event/request indexes, memberships and all running
  totals while retaining a previously counted initial physical baseline, as
  Python does. If temporary saving is enabled it selects the first absent
  `temp-N.csv`.
- Physical available/node/link totals are evaluated in that order. Utilization
  is node then link; a zero initial denominator raises at the same stage after
  earlier direct fields have been written.
- Leave reads the stored arrival result first. A successful service decrements
  `inservice_count`, removes one matching request ID for every current node
  slot, and updates the running physical-node count.
- Arrival records request-to-event ID, increments request count, then on
  success updates success/in-service counters, totals and exact-zero-guarded
  ratios in Python order. A time ratio above one (or NaN) fails before node
  membership mutation, preserving the prior partial state.
- `count` mutates the Solution through Counter only for arrival events. It
  returns a value snapshot but does not append history; `add_record` performs
  the deep value snapshot and optional temporary append.
- Event lookup supports Python-style negative memory indexes. Virtual-network
  lookup uses the arrival event ID stored for that numeric request.
- Frozen CSV append emits a header once and rejects ordered-schema drift before
  changing the file. Record serialization uses a fixed native schema and
  canonical typed Solution encodings; arbitrary Python object repr/Pandas
  extension dtypes are outside the typed domain.
- Filename arguments are a single safe component; absolute paths, separators,
  and `..` are rejected before any destination is opened.

## Parallel and performance contract

Recorder event transitions are sequential because state, first error and
partial mutation are observable. `RecorderOptions.workers` is forwarded only
to completed Counter reductions. Independent CSV row serialization may use
caller-configured workers, with pre-sized output slots and input-order errors;
the file append itself stays ordered and sequential.

The accepted compact benchmark exercised prepared arrival counting plus a deep
history snapshot at workers `1/2/8`, one warm-up and three samples. Fixture
creation, preparation, filesystem I/O, process startup and checksum work stayed
outside the timer. It gated all state/Solution fields, ordered mappings,
history, membership state, serialized byte count, and checksum. C++ was
24.907x, 17.061x, and 7.404x faster than the exact isolated Python leaf. The
protocol and raw timings are frozen in
`porting/results/recorder_benchmark_2026-07-29.json`; do not rerun or edit it.

## Deferred boundary

RL reward-calculator names, feature flags, training epochs, solver-registry
imports, TensorBoard/WandB integration, arbitrary DictConfig mutation,
Pandas object/extension dtypes, arbitrary Python objects and reflection remain
out of scope with ML. The non-ML summary API does not silently synthesize those
fields.

## Frozen handoff

The exact Recorder source hash is
`41E66702F29644E6F8A374EF909A1C41F87982558863BE17B6FDF72CEB971F4A`.
The AST-isolated non-ML differential passed 6/6 shared cases, including
workers `1/2/8`; strict GCC 11, ASan/UBSan/leak checks, focused unit tests, and
targeted frozen-integrity CTest passed. See
`porting/results/recorder_2026-07-29.md` for the compact API/performance
handoff.
