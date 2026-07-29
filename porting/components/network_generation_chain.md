# Component inventory: network generation chain

State: **COMPLETE / FROZEN** on 2026-07-29.

This document records the completed non-ML orchestration leaves:

- `network.virtual_network_request_simulator`, Python SHA-256
  `970E63F9DAC59F60E2ED1786606DC87D3271AF062BA1D8D6C67AEF8D3C7478E1`,
  12,399 bytes;
- `network.dataset_generator`, Python SHA-256
  `43D5DBE625FCD15F273067700B3C9D0B69CF931E064F6542C65802B0A4BA4E5C`,
  8,503 bytes.

The inventory was taken from the original Virne checkout at the already
recorded commit `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`. Existing C++ component
documents, rather than completed implementation sources, were used for all
dependencies. No solver, system, learning code, or frozen dependency library
is part of this contract.

## Completion record

- Simulator API and guarantees:
  [`virtual_network_request_simulator.md`](virtual_network_request_simulator.md).
  Accepted artifacts: core
  [differential](../results/virtual_network_request_simulator_core_differential_2026-07-29.json),
  I/O [differential](../results/virtual_network_request_simulator_io_differential_2026-07-29.json),
  and frozen [benchmark](../results/virtual_network_request_simulator_benchmark_2026-07-29.json).
- Generator API and guarantees:
  [`dataset_generator.md`](dataset_generator.md). Accepted provenance is
  summarized in
  [`dataset_generator_2026-07-29.md`](../results/dataset_generator_2026-07-29.md),
  with frozen [differential](../results/dataset_generator_differential_2026-07-29.json)
  and [benchmark](../results/dataset_generator_benchmark_2026-07-29.json)
  artifacts.

All accepted timing artifacts are frozen provenance. They must not be rerun or
updated while porting downstream modules.

## Dependency boundary and completed port order

The dependency chain is:

```text
random_lib + utils.dataset RNG + setting + topology + attributes
    -> BaseNetwork -> VirtualNetwork
    -> PhysicalNetwork
    -> VirtualNetworkEvent
    -> VirtualNetworkRequestSimulator
    -> dataset Generator
```

The leaves were completed in this order after `PhysicalNetwork` and
`VirtualNetwork`:

1. `VirtualNetworkEvent`, its validation, ordering key, and compact event
   index. This is an independent typed leaf.
2. Typed simulation-config decoding and `arrange_v_nets`; reuse the completed
   `DistributionRequest`, `generate_data_with_distribution`, and
   `RandomContext` APIs.
3. Sequential request generation through completed `VirtualNetwork`, topology,
   and attribute APIs; then event renewal and the request-to-event index.
4. Save/load plus a synchronized deep-copy cache. Reuse completed GML and
   YAML/JSON APIs; do not reopen or duplicate them.
5. Thin `Generator` orchestration, followed by its four-stage changeable
   workload variant.

`BaseNetwork::from_gml` is already documented; the simulator loader may wrap
its result with `VirtualNetwork(BaseNetwork&&)`. No new graph loader is needed.
The completed frozen `VirtualDatasetSetting` remains the path-builder input;
the richer live simulation config below is a separate owner and converts to it
once at the save-path boundary.

## Python API inventory

| API | Observable result |
|---|---|
| `VirtualNetworkEvent(id, type, v_net_id, time)` | validated mutable event; `type` is 0/1, request ID and time are non-negative |
| simulator constructor | stores the supplied network/event sequences directly, deep-copies settings, builds `v2event_dict` |
| `from_setting(setting, seed=None)` | optional global reseed, resolved `DictConfig` conversion, empty simulator |
| `renew(v_nets=True, events=True, seed=None)` | optional reseed, networks first, events second, returns both sequences |
| `arrange_v_nets()` | generates size, lifetime, arrival intervals/cumulative times, then optional max latency |
| `_renew_v_nets()` | constructs every `VirtualNetwork`, topology first and attributes second |
| `_renew_events()` | builds arrivals then leaves, stable-sorts by time, assigns dense event IDs |
| `save_dataset(dir)` / `save_setting(path)` | GML requests first, event setting second, simulator setting last |
| `load_dataset(dir)` | setting, events, sorted GML files, cardinality check, deep-copy cache |
| `Generator.generate_dataset(...)` | physical first, virtual second; disabled side returns `None` |
| `generate_p_net_dataset_from_config(...)` | seed, construct physical network, optional save |
| `generate_v_nets_dataset_from_config(...)` | seed, construct simulator, renew, optional save |
| `generate_changeable_v_nets_dataset_from_config(...)` | four full generation stages, quarter slices, one final event renewal |

Python's dynamic `__getitem__`/`__setitem__` event reflection, mutable default
arguments, arbitrary mapping/object protocols, and `DictConfig` runtime type are
cold-oracle boundaries. They must not become string maps or unchecked mutable
fields in the native core. Python does not validate event `id`, accepts booleans
as 0/1 event types, and permits `__setitem__` to bypass all post-init checks;
native construction deliberately validates its fixed typed representation.

## Original native API proposal

This sketch records the design input used during the port. The completed
component documents linked above are authoritative for the public API.

Fixed config values are direct fields. Exact spelling can be finalized with the
owning headers without changing these semantics.

```cpp
using VirtualRequestId = std::int64_t;
using VirtualEventId = std::size_t;

enum class VirtualEventType : std::uint8_t { leave = 0, arrival = 1 };

struct VirtualNetworkEvent {
    VirtualEventId id;
    VirtualEventType type;
    VirtualRequestId v_net_id;
    double time;
};

struct SimulationDistribution {
    DatasetValueKind value_kind;
    DistributionSpec distribution;
};

struct VirtualSimulationOutput {
    std::string events_file_name;
    std::string setting_file_name;
};

struct VirtualNetworkSimulationConfig {
    std::size_t num_virtual_networks;
    SimulationDistribution virtual_network_size;
    SimulationDistribution lifetime;
    SimulationDistribution arrival_rate;
    std::optional<SimulationDistribution> max_latency;
    topology::TopologyType topology_type;
    topology::TopologyOptions topology_options;
    std::vector<attribute::AttributeFactorySpec> node_attribute_specs;
    std::vector<attribute::AttributeFactorySpec> link_attribute_specs;
    VirtualSimulationOutput output;
    std::optional<virne::utils::SettingDocument> source_setting;
};

struct VirtualSimulationWorkers {
    std::size_t attribute_workers = 1;
    std::size_t event_workers = 1;
    std::size_t io_workers = 1;
};

class VirtualNetworkDatasetCache;

class VirtualNetworkRequestSimulator {
public:
    static VirtualNetworkRequestSimulator from_setting(
        VirtualNetworkSimulationConfig);

    std::size_t num_v_nets() const noexcept;
    std::size_t num_events() const noexcept;
    const std::vector<VirtualNetwork>& v_nets() const noexcept;
    const std::vector<VirtualNetworkEvent>& events() const noexcept;

    void renew(RandomContext&, bool renew_v_nets = true,
               bool renew_events = true,
               const VirtualSimulationWorkers& = {});
    void arrange_v_nets(NumpyRandomState&);
    void renew_events(std::size_t workers = 1);

    void save_dataset(const std::filesystem::path&,
                      std::size_t workers = 1) const;
    static VirtualNetworkRequestSimulator load_dataset(
        const std::filesystem::path&, VirtualNetworkDatasetCache&,
        std::size_t workers = 1);
    VirtualNetworkRequestSimulator clone() const;
};

class Generator {
public:
    static GeneratedDataset generate_dataset(
        const GeneratorConfig&, RandomContext&,
        const GeneratorSelection&, const VirtualSimulationWorkers& = {});
    static PhysicalNetwork generate_p_net_dataset_from_config(...);
    static VirtualNetworkRequestSimulator
        generate_v_nets_dataset_from_config(...);
    static VirtualNetworkRequestSimulator
        generate_changeable_v_nets_dataset_from_config(...);
};
```

The public generator receives caller-owned `RandomContext&`; an optional seed
resets both CPython- and NumPy-compatible streams once at the same points as
Python. Zero and one orchestration workers mean sequential. Wider values are
caller configuration, never machine auto-tuning in this component.

## Exact order, RNG, and partial mutation

One `RandomContext` owns two independent but ordered streams. Never look up a
global context inside a loop and never draw from either stream concurrently.

`arrange_v_nets` consumes the NumPy stream in this exact order:

1. all virtual-network sizes;
2. all lifetimes;
3. all arrival intervals;
4. cumulative sum of arrival intervals, with no RNG use;
5. all maximum latencies only when that fixed option is present.

Request creation then visits IDs `0..num_virtual_networks-1`. For each request
it constructs fixed metadata, optionally assigns `max_latency`, generates the
topology with `RandomContext::python()`, and only then generates node followed
by link attributes with `RandomContext::numpy()`. Topology retry counts and
attribute distributions are data-dependent, so the outer request loop cannot
be parallelized without changing accepted graphs, values, or both continuation
states. Attribute-internal workers may only use the already documented
deterministic worker path.

`renew` reseeds first when a seed is explicitly supplied, renews networks
before events, and retains all earlier mutations and RNG consumption if a later
step fails. The typed native flags are booleans; Python oddities such as
`1 == True` are boundary tests, not a reason to accept arbitrary scalars.

`Generator.generate_dataset` executes physical generation before virtual
generation. Each selected sub-generator calls `set_seed(config.seed)`. Thus,
with a present seed and both sides enabled, the streams are reset twice and
physical generation cannot perturb virtual output. With an absent seed both
calls are no-ops, so virtual generation continues after physical consumption.
If both sides are disabled there is no validation, seed call, or side effect.

The changeable generator must retain these quirks:

- it generates all `N` requests independently in each of four stages and keeps
  only one quarter from each; the streams continue across stages;
- stage 1/2 multiply every present node/link attribute `high` by 1.5/2 and
  truncate with Python `int`; stage 3/4 similarly scale only
  `v_net_size.high`;
- each selected request retains its original stage-local ID and generated
  arrival/lifetime, so arrival times may decrease at a stage boundary;
- after merging, events alone are renewed and sorted;
- the optional save directory is derived from the final stage's size-scaled
  temporary setting, while the saved simulator setting remains the original.

The four temporary event lists are unobserved. Native code may skip their
construction and move selected networks instead of deep-copying them, but it
must still generate and discard all non-selected networks to preserve stream
consumption and continuation.

## Event order and compact indexing

For `N` networks, construct a contiguous `2N` event buffer as all arrivals in
network order followed by all leaves in network order. Sort stably by `time`.
Consequently, equal-time arrivals precede every equal-time leave, and indices
within each phase remain network order. Assign dense event IDs only after the
sort. Python's request ID, arrival, and lifetime fallbacks apply only when an
object attribute is absent; an explicitly present invalid/`None` value fails at
its `int`/`float` conversion instead of taking the fallback.

The request-to-event mapping applies last-write-wins for duplicate
`(v_net_id,type)` pairs. Generated dense request IDs use a direct `2N` slot
table. Loaded sparse IDs may use one numeric pair hash lookup while building the
index; consumers then carry `VirtualRequestId`/`VirtualEventId`, never strings.
Python permits `NaN` because `NaN < 0` is false, but its comparison sort is not
a valid C++ strict-weak ordering. Treat NaN event times and post-construction
mutation into invalid fields as explicit Python-only boundaries; ordinary
native events require non-negative, non-NaN time. Positive infinity may retain
normal numeric ordering if accepted by the final typed validator.

## Save, load, errors, and cache

Save order is observable: create the root, create `v_nets`, write GML files in
current network order using `v_net-{id:05d}.gml`, write events, then write the
simulator setting. A failure leaves the completed prefix on disk. Resolve each
dynamic file name once. Parallel GML serialization is safe on the success path
only after proving output names are unique; duplicate IDs intentionally
overwrite sequentially in Python. Exact error-side-effect mode should stage
disjoint temporary outputs and commit in index order, or remain sequential.

Load order is also fixed:

1. check raw cache eligibility;
2. assert root directory, `v_nets`, hard-coded `events.yaml`, then hard-coded
   `v_sim_setting.yaml` in that order;
3. read the setting, then events and validate them in file order;
4. lexically sort every directory entry and load each as a GML request;
5. require `2 * v_nets.size() == events.size()`;
6. construct/index, cache a deep clone, and return a deep clone.

The hard-coded load names deliberately do not follow configurable save names.
All directory entries, not only `.gml` files, are attempted. Preserve both
behaviors in the compatibility API; a corrected native loader must have a
different explicit name.

Python stores every successful load in a class-wide cache, but reads a cached
entry only when the raw directory spelling contains `seed_`. It does not
canonicalize paths and has no invalidation, so cached disk contents can be
stale. The C++ cache must preserve raw-key/deep-clone behavior while owning a
mutex; returned simulators share no mutable graph, registry, event, or setting
storage. Independent GML files may be parsed into pre-sized slots with caller
workers, input-order results, and lowest-index error selection after the
setting/event phases. The process-global compatibility cache is not a license
for unsynchronized mutable global state; prefer an explicit cache owner.

Map Python assertions, missing keys, invalid events, distribution/topology/
attribute failures, GML/setting failures, and event-count mismatch to typed
error code plus operation stage. Exact Python exception text and disabled
`assert` behavior belong to oracle boundaries. Never roll back already consumed
RNG state, in-memory mutations, or documented filesystem prefix effects.

## Hot-loop and threading rules

- Config keys are fixed direct fields. Resolve topology/distribution/value
  strings once to enums during decode.
- Resolve dynamic attribute names once to factory registry IDs and graph
  `AttrId`s. The request-generation, graph, attribute, and aggregation loops use
  direct slots/IDs only.
- Arrangement buffers, request storage, events, and dense event indices are
  contiguous. Cumulative arrival time is one numeric-index loop.
- Outer stochastic request generation and both shared RNG streams are
  sequential. No lock, thread-local copy, stage splitting, or speculative retry
  may change stream order.
- Event enter/leave filling and independent loaded-file parsing are safe worker
  candidates. Preserve output order and choose the lowest input-index error.
- Save workers require disjoint targets and ordered commit semantics. Cache
  synchronization is separate from I/O workers.
- Do not introduce a global executor or an embedded worker policy. Worker
  counts come from the caller/config; only retain a wider path if the later
  compact benchmark shows a gain.

## Explicit non-ML boundary

`dataset_generator.py` imports `torch.seed` but never uses it; several other
imports there are also dead. Do not copy this eager dependency into C++.
`RandomContext` intentionally covers only the CPython and NumPy streams used by
this chain. Torch/CUDA/cuDNN/environment seeding remains deferred with the ML
adapter and must not be claimed by the native generator.

OmegaConf is a cold config adapter only. Solvers, placement, controller,
recorder, environments, systems, and all `solver/learning/**` consumers are
downstream and excluded. This port ends with owned physical/virtual datasets,
typed requests, and an ordered event schedule; it performs no embedding,
resource update, reward, training, or solver selection.

The implementation leaves passed their unit/differential gates before one
compact timing acceptance. Those timing artifacts and all older accepted
dependency benchmarks are now frozen and remain untouched.
