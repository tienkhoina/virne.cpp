# Component API: `network.virtual_network_request_simulator`

State: **COMPLETE / FROZEN** on 2026-07-29.

Python source SHA-256:
`970E63F9DAC59F60E2ED1786606DC87D3271AF062BA1D8D6C67AEF8D3C7478E1`.
Use `network_generation_chain.md` for the complete Python inventory, RNG/I/O
order and cache quirks. Completed dependency APIs are consumed only from their
component documents. The source is reopened only if a simulator API remains
ambiguous or an exact differential fails.

## Stable core API

All names are in `virne::network`.

```cpp
struct SimulationDistribution {
    virne::utils::DatasetValueKind value_kind;
    virne::utils::DistributionSpec distribution;
};

struct VirtualSimulationOutput {
    std::optional<std::string> save_dir;
    std::string events_file_name = "events.yaml";
    std::string setting_file_name = "v_sim_setting.yaml";
};

struct VirtualNetworkSimulationConfig {
    std::size_t num_virtual_networks;
    SimulationDistribution virtual_network_size;
    SimulationDistribution lifetime;
    SimulationDistribution arrival_rate;
    std::optional<SimulationDistribution> max_latency;
    TopologyType topology_type;
    TopologyOptions topology_options;
    std::vector<attribute::AttributeFactorySpec> node_attribute_specs;
    std::vector<attribute::AttributeFactorySpec> link_attribute_specs;
    std::optional<AttrValue> topology_metadata;
    std::optional<AttrValue> output_metadata;
    VirtualSimulationOutput output;
    std::optional<virne::utils::SettingDocument> source_setting;
};

struct VirtualSimulationWorkers {
    std::size_t factory_workers = 1;
    std::size_t arrangement_workers = 1;
    std::size_t attribute_workers = 1;
    std::size_t event_workers = 1;
    std::size_t io_workers = 1;
};

class VirtualNetworkRequestSimulator {
public:
    static VirtualNetworkRequestSimulator from_setting(
        VirtualNetworkSimulationConfig);
    static VirtualNetworkRequestSimulator from_state(
        VirtualNetworkSimulationConfig,
        std::vector<VirtualNetwork>,
        std::vector<VirtualNetworkEvent>);
    static VirtualNetworkRequestSimulator from_setting(
        const virne::utils::SettingDocument&,
        std::optional<std::uint32_t> seed = std::nullopt);
    static VirtualNetworkRequestSimulator from_setting(
        const virne::utils::SettingDocument&, RandomContext&,
        std::optional<std::uint32_t> seed = std::nullopt);
    void renew(RandomContext&, bool renew_v_nets = true,
               bool renew_events = true,
               std::optional<std::uint32_t> seed = std::nullopt,
               const VirtualSimulationWorkers& = {});
    void arrange_v_nets(NumpyRandomState&, std::size_t workers = 1);
    void renew_v_nets(RandomContext&, const VirtualSimulationWorkers& = {});
    void renew_events(std::size_t workers = 1);

    const std::vector<VirtualNetwork>& v_nets() const noexcept;
    const std::vector<VirtualNetworkEvent>& events() const noexcept;
    std::vector<VirtualNetwork> release_v_nets() && noexcept;
    const std::vector<std::int64_t>& arranged_sizes() const noexcept;
    const std::vector<double>& arranged_lifetimes() const noexcept;
    const std::vector<double>& arranged_arrival_times() const noexcept;
    const std::optional<std::vector<double>>& arranged_max_latencies()
        const noexcept;
    std::optional<VirtualEventId> event_id(
        VirtualRequestId, VirtualEventType) const noexcept;

    void save_setting(const std::filesystem::path&) const;
    void save_dataset(const std::filesystem::path&,
                      std::size_t workers = 1) const;
    static VirtualNetworkRequestSimulator load_dataset(
        const std::filesystem::path&, VirtualNetworkDatasetCache&,
        std::size_t workers = 1);
    static VirtualNetworkRequestSimulator load_dataset(
        const std::filesystem::path&, std::size_t workers = 1);
    VirtualNetworkRequestSimulator clone() const;
};

VirtualNetworkSimulationConfig virtual_network_simulation_config_from_setting(
    const virne::utils::SettingDocument&);

class VirtualNetworkDatasetCache {
public:
    std::optional<VirtualNetworkRequestSimulator> find(
        const std::string& raw_directory) const;
    void store(std::string, const VirtualNetworkRequestSimulator&);
    std::size_t size() const;
};
```

`from_state` is the cold load/test boundary. It preserves supplied event order
and event IDs, then rebuilds only the compact numeric lookup. The arranged
buffer views expose owned contiguous storage for differential verification;
they do not permit mutation.

## Performance and ordering

Every config field and discriminant is typed before construction. Dynamic
attribute names become factory/graph IDs once. Arrangement uses contiguous
typed buffers. It consumes the one NumPy stream in the fixed order size,
lifetime, arrival intervals, optional maximum latency; cumulative arrival is
then one ordered numeric loop. Request creation is sequential because topology
retry and attribute draws share ordered Python/NumPy streams. Caller workers
are forwarded only to completed deterministic factory/cast/attribute/event
operations and are never auto-selected.

Network renewal builds a temporary vector and commits only after every request
succeeds, matching Python assignment of `list(map(...))`; prior arranged
buffers and RNG consumption remain visible after failure. Event inputs are all
arrivals followed by all leaves. Construction may use configured workers,
stable time sort is deterministic, and dense event IDs are assigned after the
sort. Generated dense request IDs use direct two-slot indexes; sparse loaded
IDs use numeric keys only at the cold index boundary. No event hot loop uses a
string or map lookup.

The size distribution is fixed to the integer lane. Lifetime, cumulative
arrival time, and maximum latency are stored as binary64 because Python calls
`float(...)` before writing each request's graph metadata. Floating cumulative
arrival uses ordered binary64 addition; integer cumulative arrival reproduces
NumPy int64 modular arithmetic without signed-overflow UB; boolean cumulative
arrival promotes to int64. Worker zero is normalized to explicit sequential
before calling the completed dataset leaf, whose zero spelling otherwise means
an automatic policy.

`id`, `arrival_time`, and `lifetime` are emitted through the typed
`BaseNetworkConfig::graph_attributes` lane. Recursive topology/output metadata
is cloned per request. Optional `max_latency` is both a direct fixed field and
a graph value set through the request-local `AttrId` resolved once. Config,
event, generation, and cumulative loops never look up fixed string keys.

The simulator-specific `max_latency` discovered by this component is a fixed
direct optional in `VirtualNetwork`. Simulator construction also writes its
already-resolved graph metadata ID because Python explicitly calls
`set_graph_attribute`, rather than ordinary object assignment.

## Persistence and cache contract

The raw setting adapter resolves every fixed root, distribution, topology and
output key once to a `SettingKeyId`, converts discriminants immediately to
enums, and calls the completed attribute factory decoder for each definition.
It retains a deep setting snapshot only for exact persistence. The overload
with `RandomContext&` reseeds both streams before decoding, matching Python's
observable seed-before-conversion order; the compatibility overload uses the
synchronized-by-caller global random context.

Events serialize as a setting list of fixed objects with fields `id`, `type`,
`v_net_id`, and `time`. Save order is request GML in current order, events,
then the original simulator setting. Request filenames are
`v_nets/v_net-{id:05d}.gml`, with the vector index fallback. Worker zero/one
is sequential. Wider caller widths serialize to disjoint staging files and
commit them in request order, so duplicate IDs retain last-write-wins and the
lowest failure leaves only the same completed target prefix. No machine worker
policy is embedded.

The compatibility loader checks the raw-key cache first only when the raw path
contains `seed_`, then validates root, `v_nets`, hard-coded `events.yaml`, and
hard-coded `v_sim_setting.yaml` in that order. It reads setting then events,
lexically sorts every directory entry, and parses independent GML files into
pre-sized slots at caller workers with lowest-index error selection. It
requires exactly two events per loaded request. Every successful disk load is
stored, including non-`seed_` paths; cache reads and writes use a mutex and
deep clones, while clone work occurs outside the lock through an immutable
`shared_ptr`. Returned networks, registries, events, metadata, and setting
trees share no mutable storage with the cache.

Typed configs without `source_setting` can generate in memory but deliberately
fail `save_setting`/`save_dataset`: there is no reverse-engineering of factory
objects or dynamic metadata. Normal raw-setting and Generator paths always
carry the exact source snapshot.

`VirtualSimulationOutput::save_dir` is a cold typed config field consumed by
the Generator boundary; it does not enter arrangement, request construction,
event scheduling, GML, or cache loops. It was added after benchmark acceptance,
and none of the measured hot paths changed. The rvalue-only
`release_v_nets() && noexcept` transfers the completed request vector without
deep-copying it.

## Gate status

Core unit coverage and the canonical Python differential pass 18 shared cases,
two native cases, and four recorded boundaries (24 classified cases). It
covers distribution order/continuation, cumulative lanes, optional latency,
generation/failure commit, event ties/indexing, workers `0/1/2/8`,
flags/seeding, clone/move, and independent concurrent contexts.

The I/O/cache gate passes 16 shared cases, zero native cases, and three
recorded boundaries (`case_count: 19`). It covers raw decode/deep source
snapshot, semantic GML/YAML round trips at workers `0/1/2/8`, hard-coded load
names, layout/error order, event validation/cardinality, sorted lowest-index
load failure, `seed_`/non-`seed_` cache behavior, and deep cache clones.

The single accepted 65,536-request benchmark passed exact output gates:

| Hot path | Workers 1 | Workers 2 | Workers 8 |
|---|---:|---:|---:|
| Arrangement | 2.807x | 2.095x | 1.986x |
| Event schedule | 22.664x | 26.236x | 20.735x |

The benchmark and all dependency benchmarks are frozen forever and must not be
rerun or updated. Full evidence is in
`../results/virtual_network_request_simulator_2026-07-29.md`.
