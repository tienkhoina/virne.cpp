# Component API: `solver.base_solver`

State: **COMPLETE / FROZEN** on 2026-07-29. The accepted differential,
benchmark, drivers, machine-readable results, and measurements are provenance;
do not rerun or edit them while porting later leaves.

Python oracle: `../virne/virne/solver/base_solver.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`196573E631654A6D14888685B93977601412F110A9B5620398850CCE121A2806`,
4,861 bytes and 116 physical lines. The frozen Solution, Controller, Counter,
Recorder, Logger, Environment, configuration, random, LinkRank, and NodeRank
API documents were read before defining this leaf; their implementations and
accepted benchmarks remain closed.

## Scope

This leaf owns the non-ML solver holder and registry foundation only. It
snapshots a typed solver configuration, retains non-owning references to the
four completed runtime collaborators, exposes the default no-op `ready()` and
base `solve()` failure, and registers factories behind compact IDs.

Concrete search/mapping algorithms, candidate generation, system lifecycle,
MCF/OR-Tools, observations, rewards, training, RL, Torch, CUDA, and learning
imports remain outside the target. Reinforcement and unsupervised categories
are reserved enum values only, so later optional modules need not change the
registry ABI.

## Stable typed API

All names are in `virne::solver` unless qualified.

```cpp
struct SolverConfigInput {
    std::optional<std::uint32_t> seed;
    int verbose = 1;
    utils::RunDirectoryInput run_directory;
    bool reusable = false;
    rank::NodeRankMethod node_ranking_method = rank::NodeRankMethod::order;
    rank::LinkRankMethod link_ranking_method = rank::LinkRankMethod::order;
    core::controller::NodeMatchingMethod matching_method =
        core::controller::NodeMatchingMethod::greedy;
    core::controller::ShortestPathMethod shortest_method =
        core::controller::ShortestPathMethod::k_shortest;
    std::int64_t k_shortest = 10;
    bool allow_rejection = false;
    bool allow_revocable = false;
};

struct SolverConfig {
    std::optional<std::uint32_t> seed;
    int verbose = 1;
    std::filesystem::path save_dir;
    bool reusable = false;
    rank::NodeRankMethod node_ranking_method = rank::NodeRankMethod::order;
    rank::LinkRankMethod link_ranking_method = rank::LinkRankMethod::order;
    core::controller::NodeMatchingMethod matching_method =
        core::controller::NodeMatchingMethod::greedy;
    core::controller::ShortestPathMethod shortest_method =
        core::controller::ShortestPathMethod::k_shortest;
    std::int64_t k_shortest = 10;
    bool allow_rejection = false;
    bool allow_revocable = false;
};

SolverConfig make_solver_config(const SolverConfigInput&);

struct SolverDependencies {
    std::reference_wrapper<const core::controller::Controller> controller;
    std::reference_wrapper<core::Recorder> recorder;
    std::reference_wrapper<const core::Counter> counter;
    std::reference_wrapper<core::Logger> logger;
};

struct SolverInstance {
    const network::VirtualNetwork& virtual_network;
    const network::PhysicalNetwork& physical_network;
};

class Solver {
public:
    Solver(SolverDependencies, SolverConfig);
    virtual ~Solver() = default;

    virtual void ready();
    virtual core::Solution solve(const SolverInstance&);

    const SolverConfig& config() const noexcept;
    const core::controller::Controller& controller() const noexcept;
    core::Recorder& recorder() const noexcept;
    const core::Counter& counter() const noexcept;
    core::Logger& logger() const noexcept;
    std::uint64_t num_arrived_virtual_networks() const noexcept;
    void set_num_arrived_virtual_networks(std::uint64_t) noexcept;
    void increment_num_arrived_virtual_networks();
};

struct SolverId { std::uint32_t value = 0U; };

enum class SolverCategory : std::uint8_t {
    unknown,
    rounding,
    exact,
    heuristic,
    node_ranking,
    meta_heuristic,
    reinforcement_learning,
    unsupervised_learning,
};

enum class SolverErrorCode : std::uint8_t {
    unsupported_category,
    empty_solver_name,
    duplicate_solver_name,
    empty_factory,
    registry_frozen,
    registry_not_frozen,
    unknown_solver,
    invalid_solver_id,
    factory_returned_null,
    arrived_count_overflow,
    solve_not_implemented,
};

enum class SolverOperation : std::uint8_t {
    parse_category,
    construct_solver,
    ready,
    solve,
    update_arrived_count,
    register_solver,
    freeze_registry,
    resolve_solver,
    lookup_solver,
    list_registered,
    create_solver,
};

class SolverException : public std::runtime_error {
public:
    SolverException(
        SolverErrorCode,
        SolverOperation,
        std::string message,
        std::optional<SolverId> solver_id = std::nullopt);
    SolverErrorCode code() const noexcept;
    SolverOperation operation() const noexcept;
    const std::optional<SolverId>& solver_id() const noexcept;
};

using SolverFactory = std::function<std::unique_ptr<Solver>(
    SolverDependencies, SolverConfig)>;

struct SolverDescriptor {
    SolverId id;
    std::string name;
    SolverCategory category = SolverCategory::unknown;
};

class SolverRegistry {
public:
    SolverId register_solver(
        std::string name, SolverCategory, SolverFactory);
    void freeze();
    bool frozen() const noexcept;
    std::size_t size() const noexcept;

    // Compatibility strings are cold: resolve once, then retain SolverId.
    SolverId resolve(std::string_view name) const;
    const SolverDescriptor& descriptor(SolverId) const;
    std::vector<SolverDescriptor> list_registered() const;
    std::unique_ptr<Solver> create(
        SolverId, SolverDependencies, SolverConfig) const;
};

SolverCategory solver_category_from_string(std::string_view);
std::string_view solver_category_name(SolverCategory) noexcept;
```

`SolverException` carries its concrete error code, operation, and optional
compact solver ID. The native field is correctly named `matching_method`; only
differential and compatibility serialization emit Python's observable
`matching_mathod` typo.

## Behavior, IDs, and ownership

`make_solver_config` copies every input field and delegates the POSIX-compatible
three-component path rule to frozen `utils::get_run_id_dir`; it does not rebuild
path logic. A `Solver` copies this fixed snapshot, initializes the arrived count
to zero, and keeps exactly the collaborator identities supplied by the caller.
Collaborators and networks are non-owning and must outlive their call surface.
`ready()` is a no-op. Base `solve()` raises the typed native equivalent of
Python's `NotImplementedError`; derived solvers override it.

`SolverInstance` deliberately exposes both networks as const views: this base
seam models a solver that inspects an Environment snapshot and returns a typed
`Solution`. A future interactive solver that mutates a private physical clone
or rebinds `PreparedController` needs a separate explicit adapter; it must not
cast away constness or mutate Environment-owned state through this API.

Solver names are genuinely dynamic startup data. Registration owns each name
once, inserts one contiguous descriptor/factory slot, and returns its stable
`SolverId`. A name string is used only by `register_solver` or one cold
`resolve`; all repeated descriptor/factory access is direct by ID. Fixed
category/config/result/error data never use string maps. Slot insertion relies
on vector geometric growth; it no longer reserves `size()+1` on every
registration and therefore avoids quadratic whole-registry relocation.

Duplicate or empty names and empty factories are rejected without partial
registration. `register_solver` and `freeze` linearize under the same exclusive
lock: either the new ID commits before freeze, or freeze wins and registration
returns `registry_frozen` without a partial name/slot. `freeze()` permanently
closes mutation. Before freeze, registry operations are synchronized; after
freeze, concurrent `resolve`, direct-ID lookup, independent list snapshots,
and factory creation are read-only. Descriptor order is registration order. A
factory registered for concurrent creation must itself support concurrent const
invocation; the frozen compact-ID path deliberately does not copy or lock its
`std::function`. Each created mutable solver belongs to one caller/worker; the
registry does not make solver instances internally thread-safe. `SolverId` is
registry-local, exactly like completed attribute registry IDs; passing an
in-range ID from another registry is a caller-domain error. Descriptor
references remain valid only while their frozen registry lives.

Factories must construct holders without mutating Recorder, Logger, or other
external state before returning. Factory exceptions propagate unchanged and
leave registry structure intact; only a successful call returning null becomes
`factory_returned_null`. External factory side effects are not transactional.
Multiple solver instances may share the internally synchronized Logger, but
calls that can mutate one shared Recorder must be serialized or use separate
Recorder instances. Mutable collaborator access through a const Solver
expresses non-owning service access, not solver thread safety.

Both `Solver` and `SolverRegistry` delete copy and move operations. The direct
enum defaults in `SolverConfig` match `SolverConfigInput`: node/link order,
greedy node matching, and k-shortest path selection.

Python's unused `env_cls`, permissive `None`/unhashable keys, arbitrary class
registration, generic Config identity, unknown kwargs, partial constructor
mutation, and eager import-side-effect registration are dynamic-language
boundaries. Native modules explicitly register typed factories during startup;
the core registry never imports or links learning code.

## Accepted gate

The focused unit gate passed every config field, verbose override, POSIX
absolute/empty path cases, exact collaborator identity and constness,
initial/counted/overflow arrivals, no-op ready, base solve failure, every
category, registration order, duplicate/missing lookup, stable direct IDs,
snapshot copies, ignored Python `env_cls`, freeze, the register-vs-freeze
transition, invalid IDs/factories/null results, unchanged factory-exception
passthrough, and concurrent frozen resolve/list/descriptor/create readers at
workers `1/2/8`.

The exact Python differential passed `13/13` shared cases at native workers
`1/2/8`. Strict GCC 11 warnings-as-errors builds passed independently for the
production library, unit, and harness; ASan/UBSan/leak checks passed; targeted
CTest passed `2/2`. The hot-ID audit confirmed that dynamic solver names occur
only at registration and one cold resolution; descriptor/factory/create loops
use compact `SolverId` slots directly.

The single frozen cold-start benchmark pre-registered 32 descriptors and
constructed 4,096 holders from typed configs, with one warm-up and three
samples. Python measured `17.813469 ms`; C++ measured `1.594245`, `1.798216`,
and `1.712720 ms` at workers `1/2/8`, or `11.174x`, `9.906x`, and `10.401x`
faster. Every route produced 4,096 entries, 419,498 bytes, and checksum
`13751587758314786690`. Full provenance and hashes are in
`../results/base_solver_2026-07-29.md`; both machine-readable result JSON files
beside it are frozen.
