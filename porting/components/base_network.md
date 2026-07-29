# Component API: `network.base_network`

State: **COMPLETE AND FROZEN** on 2026-07-28.
The accepted benchmarks of every dependency remain frozen and must not be
rerun or updated while implementing this component.

## Documentation-first record

Before opening the unported source, this audit read the non-ML component map,
frozen-foundation manifest, port-wide performance contract, mandatory graph
consumer checklist, frozen Graph/DiGraph API, raw-setting API, topology
generator API, `utils.network`, every completed node/link/graph attribute API,
and `AttributeBenchmarkManager`. This contract was subsequently reconciled
against the frozen `attribute_factory.md` API before implementation. Completed
implementation source and the factory header were not opened because the
factory note contains every BaseNetwork-facing signature. The only Python
production file opened was the BaseNetwork source listed below because no
BaseNetwork API note existed.

Future work starts from this document. Open `base_network.py` or a completed
dependency implementation only for a genuinely undocumented API, an exact
differential mismatch, or a measured low-level optimization that requires
layout knowledge. Add every newly learned public fact here in the same change.

## Source identity and scope

- Python source: `../virne/virne/network/base_network.py`.
- Original commit: `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`;
  the Python checkout was clean during the audit.
- SHA-256:
  `94CB1185B5E6F0046D9393F97AF1EE0DD1C0D688596B102A0EC3CC98314ADDCE`.
- Size: 19,895 bytes; extent: 483 physical lines.
- Scope is the undirected non-ML `BaseNetwork` model. `PhysicalNetwork`,
  `VirtualNetwork`, request simulation, dataset generation, solver, system,
  and learning code remain later components.

The Python class derives from `networkx.Graph`; it never selects `DiGraph` and
does not construct a graph-attribute factory registry. Its graph metadata is
dynamic graph-level data, while only node and link attribute definitions are
registered. The imported `GraphAttribute`, NumPy, several report-view types,
and `lru_cache` are unused. Do not add dependencies merely to mirror imports.

## Complete Python surface

| Surface | Observable result |
|---|---|
| constructor `(incoming_graph_data=None, config=None, **kwargs)` | initializes/merges graph metadata and builds ordered node/link definition dictionaries |
| `create_attrs_from_setting()` | rebuilds node/link definitions from the current stored setting lists |
| `check_attrs_existence()` | validates only the first node and first link against every definition |
| `generate_topology(num_nodes, type='path', **kwargs)` | replaces node/adjacency storage after successful delegated generation |
| `generate_attrs_data(node=True, link=True)` | generates eligible node definitions, then eligible link definitions, with ordered partial mutation |
| `num_nodes`, `num_links`, `num_edges` | cached snapshots of `number_of_nodes/edges()` |
| `links` | cached live `EdgeView` |
| `adjacency_matrix` | fresh weighted CSR adjacency matrix |
| `get_graph_attrs(names)` | live whole metadata map for `None`, otherwise an ordered selected dictionary |
| `get_node_attr_types()`, `get_link_attr_types()` | type values in definition order |
| `get_node_attrs(types=None, names=None)` and link counterpart | ordered definition selection; `types` takes precedence |
| feature-count properties | total or exact-resource definition counts |
| `init_graph_attrs()` | ensures two setting keys, then reflects graph metadata onto the Python object |
| graph/node/link data setters | ordered delegation with no rollback |
| node/link/adjacency/aggregation data getters | ordered row collection |
| `subgraph`, `subnetwork` | induced live view sharing both definition dictionaries |
| `get_subgraph_view`, `get_subnetwork_view` | predicate live view sharing both definition dictionaries |
| `__getitem__` | integer adjacency lookup; string reflection lookup; other keys return the `TypeError` class |
| `__setitem__` | object-attribute assignment only, not graph-metadata assignment |
| `__repr__` | fixed five-field summary in insertion order |
| `clone()` | `copy.deepcopy(self)` |
| `_prepare_gml_graph()` | materialized plain undirected GML-safe graph |
| `from_gml(path, label='id')` | load, optionally relabel, construct, sample-check, and restore flattened metadata |
| `save_attrs_dict(path)` | raw setting document containing graph metadata and definition snapshots |

There is no BaseNetwork cache of attribute benchmarks in this source. The
completed benchmark manager calls this class as a data-provider in Python;
the native adapter specified below prepares its rows and delegates reduction.

## Construction and ordered registry behavior

Construction order is observable and must be retained for the supported
typed domain:

1. Convert/copy `incoming_graph_data` using Graph semantics.
2. A falsey `config` becomes an empty dictionary; otherwise the OmegaConf
   boundary converts it to an ordinary dictionary, then Python asserts the
   result is a dictionary.
3. Store the resulting config object on `self.config`.
4. Append configured node settings after incoming graph node settings, and do
   the same for links.
5. Deduplicate each list by exact `name`. A later definition replaces the
   value at that name but keeps the first insertion position. Thus config
   overrides incoming metadata without moving a pre-existing name.
6. Replace graph metadata `node_attrs_setting` and `link_attrs_setting` with
   those ordered deduplicated lists, then construct definitions in that order.
7. If present, copy configured `topology` and then `output` into graph
   metadata.
8. Apply `graph_attrs_setting` entries in their order (`None` means empty),
   then apply constructor keyword entries. Later values overwrite without
   moving an existing graph key.

Missing `name`, non-addable setting containers, invalid factory fields, and
dynamic mapping failures occur at their natural stage. Earlier graph changes
remain. Python factory lookup constructs only node and link definitions here;
do not silently register the completed graph attribute classes.

Native construction decodes known schemas exactly once. Fixed config fields
are direct members and fixed discriminants are enums. Truly dynamic graph
metadata may use frozen `AttrMap`/`AttrValue`. The cold decoder may accept a
`SettingDocument`, but after decoding no generation/filter/data loop may read
a `SettingObject`, compare a type string, or hash a fixed field name.

## Stable native data model

All names below belong to `virne::network`; completed attribute types remain
in `virne::network::attribute`.

```cpp
struct NodeNetworkAttributeBinding {
    attribute::AttributeRegistryId registry_id = 0;
    AttrId value_id = 0;
    const attribute::NodeAttributeRegistry* registry_identity = nullptr;
    const Graph* graph_identity = nullptr;
};

struct LinkNetworkAttributeBinding {
    attribute::AttributeRegistryId registry_id = 0;
    AttrId value_id = 0;
    const attribute::LinkAttributeRegistry* registry_identity = nullptr;
    const Graph* graph_identity = nullptr;
};

struct GraphAttributeAssignment {
    std::string name;
    AttrValue value;
};

struct BaseNetworkConfig {
    std::vector<attribute::AttributeFactorySpec> node_attribute_specs;
    std::vector<attribute::AttributeFactorySpec> link_attribute_specs;
    std::optional<AttrValue> topology;
    std::optional<AttrValue> output;
    std::vector<GraphAttributeAssignment> graph_attributes;
    std::optional<virne::utils::SettingDocument> source_config;
    std::size_t factory_workers = 1;
};

struct BaseNetworkConstruction {
    std::optional<Graph> incoming_graph;
    BaseNetworkConfig config;
    std::vector<GraphAttributeAssignment> extra_graph_attributes;
};

struct AttributeSelection {
    std::optional<std::vector<attribute::AttributeKind>> kinds;
    std::optional<std::vector<attribute::AttributeRegistryId>> ids;
};

enum class AttributeDataLayout : std::uint8_t { sparse, dense };

struct NodeAttributeDataUpdate {
    attribute::AttributeRegistryId registry_id = 0;
    AttributeDataLayout layout = AttributeDataLayout::dense;
    std::vector<attribute::NodeAttributeAssignment> sparse_values;
    std::vector<AttrValue> dense_values;
};

struct LinkAttributeDataUpdate {
    attribute::AttributeRegistryId registry_id = 0;
    AttributeDataLayout layout = AttributeDataLayout::dense;
    std::vector<attribute::LinkAttributeAssignment> sparse_values;
    std::vector<AttrValue> dense_values;
};

BaseNetworkConstruction base_network_construction_from_setting(
    std::optional<Graph> incoming_graph,
    const virne::utils::SettingDocument* config,
    std::vector<GraphAttributeAssignment> extra_graph_attributes = {},
    std::size_t factory_workers = 1);
```

`AttributeFactorySpec`, `AttributeRegistryId`, `NodeAttributeRegistry`, and
`LinkAttributeRegistry` are owned by the frozen attribute-factory contract;
BaseNetwork does not redeclare, wrap, downcast, or substitute its own registry
types for them.
The two registries uniquely own polymorphic definitions, expose typed
node/link references, retain insertion order, and are movable but non-copyable.
The prepared config carries factory specs into construction; the live network
stores the resulting registries plus private canonical cold factory snapshots.
Those private snapshots describe the live registries and are distinct from the
public `graph["node_attrs_setting"]` / `graph["link_attrs_setting"]` metadata,
which later graph assignments may deliberately make divergent. Clone and
explicit rebuild internals reuse the private snapshots and frozen factory APIs
rather than reverse-engineering polymorphic objects.

`BaseNetworkConfig` is the prepared native boundary. The cold
`base_network_construction_from_setting` adapter is the Python-compatible raw
boundary. It extracts both incoming-graph and configured setting lists, then
performs BaseNetwork's name-only append/deduplication before decoding retained
items, because the original constructor discards an overridden item before
inspecting its other factory fields. It then calls
`attribute_factory_spec_from_setting` once for each retained raw item and
invokes `create_node_attributes_from_specs` /
`create_link_attributes_from_specs` with `factory_workers`. Owner/kind/
restriction/checking/distribution strings never survive this cold boundary.
Canonical setting snapshots are emitted from the retained typed specs; they
are not the runtime source of fixed fields.

## Stable native registry and class surface

```cpp
class BaseNetworkView;

class BaseNetwork {
public:
    BaseNetwork();
    explicit BaseNetwork(BaseNetworkConstruction construction);
    BaseNetwork(const BaseNetwork&) = delete;
    BaseNetwork& operator=(const BaseNetwork&) = delete;
    BaseNetwork(BaseNetwork&& other);
    BaseNetwork& operator=(BaseNetwork&& other);

    const Graph& graph() const noexcept;
    Graph& graph() noexcept;
    const attribute::NodeAttributeRegistry& node_attributes() const noexcept;
    const attribute::LinkAttributeRegistry& link_attributes() const noexcept;
    const std::optional<virne::utils::SettingDocument>& config_snapshot()
        const noexcept;
    void create_attrs_from_setting();

    std::optional<NodeNetworkAttributeBinding> bind_node_attribute(
        std::string_view name) const;
    std::optional<LinkNetworkAttributeBinding> bind_link_attribute(
        std::string_view name) const;
    void rebind_attribute_values();

    std::size_t num_nodes() const;
    std::size_t num_links() const;
    std::size_t num_edges() const;
    std::size_t live_num_nodes() const noexcept;
    std::size_t live_num_links() const noexcept;
    void invalidate_cached_cardinalities() noexcept;

    std::size_t num_node_features() const noexcept;
    std::size_t num_link_features() const noexcept;
    std::size_t num_node_resource_features() const noexcept;
    std::size_t num_link_resource_features() const noexcept;
    std::vector<attribute::AttributeKind> get_node_attr_types() const;
    std::vector<attribute::AttributeKind> get_link_attr_types() const;

    const AttrMap& graph_attributes() const noexcept;
    AttrMap& graph_attributes() noexcept;
    AttrId bind_graph_attribute(std::string_view name);
    const AttrValue& graph_attribute(AttrId id) const;
    void set_graph_attribute(AttrId id, AttrValue value);
    void set_graph_attrs_data(
        const std::vector<GraphAttributeAssignment>& values);
    void init_graph_attrs();

    std::vector<attribute::AttributeRegistryId> select_node_attributes(
        const AttributeSelection&) const;
    std::vector<attribute::AttributeRegistryId> select_link_attributes(
        const AttributeSelection&) const;

    void check_attrs_existence() const;
    void generate_topology(const topology::TopologyRequest& request);
    void generate_topology(const topology::TopologyRequest& request,
                           PyRandom& rng);
    void generate_attrs_data(NumpyRandomState& rng,
                             bool node = true,
                             bool link = true,
                             std::size_t workers = 1);

    void set_node_attrs_data(
        const std::vector<NodeAttributeDataUpdate>& updates,
        std::size_t workers = 1);
    void set_link_attrs_data(
        const std::vector<LinkAttributeDataUpdate>& updates,
        std::size_t workers = 1);

    CSRMatrix adjacency_matrix() const;
    BaseNetworkView subgraph(const std::vector<Vertex>& nodes) const;
    BaseNetworkView subnetwork(const std::vector<Vertex>& nodes) const;
    BaseNetworkView get_subgraph_view(
        NodeFilter filter_node = {}, EdgeFilter filter_edge = {}) const;
    BaseNetworkView get_subnetwork_view(
        NodeFilter filter_node = {}, EdgeFilter filter_edge = {}) const;

    std::string repr(std::string_view class_name = "BaseNetwork") const;
    BaseNetwork clone() const;
    Graph prepare_gml_graph() const;
    static BaseNetwork from_gml(const std::string& path,
                                std::string_view label = "id");
    void save_attrs_dict(const std::string& path) const;
};

class BaseNetworkView {
public:
    const GraphView& graph_view() const noexcept;
    const attribute::NodeAttributeRegistry& node_attributes() const noexcept;
    const attribute::LinkAttributeRegistry& link_attributes() const noexcept;
    const AttrMap& graph_attributes() const noexcept;
    std::size_t num_nodes() const;
    std::size_t num_links() const;
    const BaseNetwork& parent() const noexcept;
};
```

Exact completed namespace qualifications and topology request signatures may
be adjusted to their documented public declarations during compile
integration; the semantic surface and resolution boundaries above are frozen.
`bind_*_attribute` performs the single dynamic factory-registry name lookup and
returns the already-bound graph value `AttrId`. Bindings carry the factory
registry and graph identities, so equal numeric IDs from unrelated networks
are rejected without string work. Definitions and their value names are bound
once after the construction merge. A copied/loaded network rebuilds bindings
even though frozen Graph copies preserve graph registry order. Replacing the
complete Graph through the mutable escape hatch requires
`rebind_attribute_values()` before attribute access; ordinary node/edge
structural mutation retains the same graph registry.

`AttributeRegistryId` is local to its exact node or link registry and is never
interchanged merely because two numeric values match. It is also distinct from
the Graph value `AttrId`. Every factory rebuild resolves extrema
`originator_name` against the newly completed owning registry; an
`originator_id` from another registry instance is never carried across clone,
load, or rebuild.

`create_attrs_from_setting()` is an explicit cold compatibility operation. It
passes the current node snapshot to
`attribute::create_node_attributes_from_setting` and replaces the node
registry only after that whole call succeeds, then repeats with
`attribute::create_link_attributes_from_setting`. Raw setting decode is
sequential; typed construction uses the stored caller-supplied
`factory_workers`, pre-sized slots, lowest-index error selection, and the
factory's final sequential first-position deduplication. Thus a link factory
error can leave the newly replaced node registry beside the old link registry,
while a node factory error replaces neither. Successful replacements rebuild
every value binding and atomically replace the corresponding private canonical
factory snapshot. Constructor graph-attribute/extra assignments may overwrite
either public graph-metadata snapshot after initial registry creation, matching
Python's possible metadata/registry divergence; only this explicit method
reconciles the live registry and private snapshot to that public metadata.

`source_config` retains the supported cold input document for diagnostics and
the Python-visible `config` concept. It is never consulted by a generation,
filter, graph, benchmark, or worker loop; those paths use the decoded direct
fields. Unknown config keys may remain only in this snapshot. Mutating the
snapshot does not silently mutate the typed runtime schema: an explicit decode
and reconstruction is required.

## Cardinality and topology semantics

Python's three cardinalities are `cached_property` snapshots. Their first read
stores a value that is not invalidated by later structural mutation or by
`generate_topology`. Preserve that observable behavior in `num_nodes`,
`num_links`, and `num_edges`. `num_links` and `num_edges` have independent
caches even though both initially call `number_of_edges()`.

The native `live_num_*` and explicit invalidation functions are safe C++
extensions. They make stale-cache handling explicit without silently changing
the Python-compatible methods. Internal parity paths use the cached methods
where Python does; new performance code that requires live structure may use
the direct O(1) frozen Graph counts deliberately.

Topology generation delegates the completed topology generator. It validates
and builds a temporary graph first. Only after success does it replace node
and adjacency topology while retaining BaseNetwork graph metadata and factory
registries, then re-resolves every attribute value name in the new graph
registry before returning. It must not partly replace the network on generator
failure and must not invalidate the three caches. Shared `PyRandom` retries
remain sequential and preserve exact consumption. A caller-owned seeded batch
belongs to the completed topology component, not this single-network method.

## Attribute generation semantics

`generate_attrs_data` processes every node registry entry in order before any
link registry entry. An attribute is eligible when its direct `generative`
field is true or its kind is exactly `AttributeKind::extrema`. For each
eligible attribute:

1. Fetch its typed attribute reference from the factory registry once.
2. Generate using the completed Base/Node/Link implementation and the one
   caller-owned NumPy-compatible RNG stream.
3. Compare the result length against the corresponding cached cardinality.
4. Only after the length passes, set values through the attribute's resolved
   `AttrId` binding.

A length mismatch leaves this attribute unchanged but retains mutations and
RNG consumption from all earlier attributes. Any later error also preserves
earlier work. Extrema reads the factory-resolved originator registry ID and
therefore observes preceding generation; graph-owner extrema is irrelevant
because this registry has only node/link definitions. Node and link flags are
evaluated in that order.

The factory deliberately permits a missing extrema originator to survive
construction as its documented invalid registry ID. BaseNetwork must not add
an eager constructor check: the corresponding generation/use fails at that
entry's normal position, after all earlier generation, RNG consumption, and
mutation have occurred.

No parallelism may reorder registry entries, share or split the RNG draws, change
extrema dependencies, select a different first failure, or roll back partial
mutation. The configured worker count may be forwarded to the completed
leaf's post-draw transforms, gathers, and disjoint dense setter. Zero/one is
the canonical sequential path; wider values are caller configuration and are
capped by independent item count/CPU allowance inside the leaf. BaseNetwork
must not embed a benchmark-selected worker count or machine-specific policy.

## Selection and data adapters

Selection preserves factory-registry insertion order. With neither filter
present, all attributes are selected. A present kind filter takes precedence over a
present ID/name filter, including when that kind filter is empty. Kind tests
use `AttributeKind`; names are resolved once to `AttributeRegistryId` before the
scan. Repeated consumers retain those IDs and never hash the names again.

Native row adapters are:

```cpp
std::vector<std::vector<AttrValue>> get_node_attrs_data(
    const BaseNetwork&,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    std::size_t workers = 1);

std::vector<std::vector<AttrValue>> get_link_attrs_data(
    const BaseNetwork&,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    std::size_t workers = 1);

std::vector<DistanceMatrix> get_adjacency_attrs_data(
    const BaseNetwork&,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    bool normalized = false,
    std::size_t workers = 1);

std::vector<std::vector<double>> get_aggregation_attrs_data(
    const BaseNetwork&,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    attribute::LinkAggregation aggregation =
        attribute::LinkAggregation::sum,
    bool normalized = false,
    std::size_t workers = 1);
```

The Python node/link row getters inspect element zero before choosing the name
or object branch; empty input therefore raises `IndexError`. Preserve this as
a typed empty-selection error for these two adapters. Adjacency and
aggregation adapters return an empty outer vector for empty input. Rows use
node/edge order and omit missing values exactly as the completed leaf getters.
Do not pad ragged rows here.

Setters accept already-resolved factory registry IDs plus values and process
attributes in caller order. Earlier writes remain when a later attribute
fails. A repeated registry ID is applied repeatedly. Strings are not accepted
inside these loops; a boundary helper can bind a requested name list once.

The measured `sum`/non-normalized aggregation path consumes the frozen graph's
SciPy-ordered sparse COO directly and accumulates each column in row order.
This is bit-identical to the completed dense column reduction while avoiding
an `O(num_nodes^2)` temporary on sparse networks. Other aggregation and
normalization modes continue to delegate to the completed LinkAttribute API.

The adjacency property delegates frozen `nx::to_scipy_sparse_matrix` with
weight name `weight` resolved once and format `csr`. It returns native
`CSRMatrix`; an empty graph and missing/default-weight behavior follow the
frozen graph contract, not SciPy object identity.

## AttributeBenchmarkManager adapter

BaseNetwork owns data gathering only. It must not copy the completed manager's
float32 max, NaN/signed-zero, repetition, duplicate-key, cache, or threading
logic.

```cpp
struct BaseNetworkBenchmarkSelection {
    bool node = true;
    bool link = true;
    bool link_sum = true;
    std::optional<std::vector<attribute::AttributeKind>> node_kinds =
        std::vector<attribute::AttributeKind>{
            attribute::AttributeKind::resource,
            attribute::AttributeKind::extrema};
    std::optional<std::vector<attribute::AttributeKind>> link_kinds =
        std::vector<attribute::AttributeKind>{
            attribute::AttributeKind::resource,
            attribute::AttributeKind::extrema};
    std::size_t workers = 1;
};

attribute::AttributeBenchmarkRequest prepare_attribute_benchmark_request(
    const BaseNetwork&, const BaseNetworkBenchmarkSelection&);

attribute::AttributeBenchmarks get_attribute_benchmarks(
    const BaseNetwork&, const BaseNetworkBenchmarkSelection&);
```

For each enabled group, select attributes in factory-registry order. `nullopt`
kind filters mean all entries and derive `extrema_requested` from the selected
kinds. Copy each entry's `AttributeRegistryId` into the manager descriptor's
`AttributeDefinitionId` alias only within its node/link group; IDs from the two
registries are never interchanged. Convert supported bool/int64/double graph
values to binary32 at this
boundary with Python/NumPy-compatible rounding; reject strings/recursive
values and ragged nonempty rows with typed stage information. Node prepared
data uses ordinary rows. Direct-link prepared data uses
`column_repetitions=2` rather than allocating Python's concatenated copy.
Link-sum prepared data calls the BaseNetwork aggregation API with typed `sum`
and no normalization, which selects the measured sparse ordered specialization,
then uses repetition one.

`get_attribute_benchmarks` must call
`AttributeBenchmarkManager::get_benchmarks(prepare_...)`. It must never
reimplement reduction. Disabled groups remain `nullopt`, and group construction
order is node, link, link-sum. An enabled node or direct-link group with zero
selected definitions fails at BaseNetwork's historical element-zero access;
an enabled link-sum group with zero definitions reaches the manager and returns
an empty ordered map. The differential must lock this original integration
split rather than normalizing all three empty cases.

## Graph metadata, item access, and fixed fields

`get_graph_attrs(None)` returns the exact live Python graph dictionary. Native
code exposes live `AttrMap&`. A selected read accepts resolved `AttrId`s and
returns entries in request order; a missing ID fails at that position.
Duplicate selections overwrite by name while keeping first insertion order in
the cold ordered-result adapter.

`init_graph_attrs` ensures `node_attrs_setting` and `link_attrs_setting` exist
before reflecting every graph entry except exact `num_nodes` onto the Python
object. `set_graph_attribute` always writes the graph dictionary first;
`num_nodes` stops there, while another name may then fail during `setattr`,
leaving the graph write visible. Arbitrary reflection, descriptors, method
shadowing, and object corruption are Python-only dynamic boundaries. Native
code keeps known `topology`, `output`, config, and caches as direct fields and
all unknown metadata in `AttrMap`; it does not synthesize arbitrary C++ data
members.

Python integer `__getitem__` delegates adjacency access (and `bool` counts as
an integer), string keys use `getattr(..., None)` rather than graph metadata,
and every other key returns the `TypeError` class instead of raising.
`__setitem__` changes only a Python object attribute. Native code exposes
explicit adjacency, fixed-field, and graph-attribute APIs; Python reflection
and return-of-a-type-object are recorded boundaries, not a `std::any` indexer.

The supported native `repr` records exactly these semantic fields in order:
cached node count, cached link count, node definition names, link definition
names, and graph metadata excluding the two setting snapshots. Python class
name/reflection and arbitrary-object `repr` side effects remain dynamic
boundaries; supported `AttrValue` lanes use the frozen canonical formatting.

## Existence check

The check is deliberately a sample check, despite its docstring saying “all”:

1. No nodes raises before inspecting links or definitions.
2. No links raises next.
3. Select the first node in graph order and require every node definition name
   on that one node, in definition order.
4. Select the first link in edge order and require every link definition name
   on that one edge, in definition order.

It does not inspect any later node/link. Python uses `assert` for missing
attributes, but native behavior must be stable across build flags, so use a
typed BaseNetwork exception while retaining selection and first-failure order.
Every name is already a bound `AttrId`; the loops use direct `AttrMap::find(id)`.

## Subgraphs and views

Python `subgraph` is an induced, structurally read-only live view. The returned
view shares the exact node/link definition dictionaries and observes parent
attribute data. Predicate views likewise share definitions and re-evaluate
their filters against current structure/attributes.

Native BaseNetwork view adapters wrap the frozen `GraphView` plus non-owning
references to the parent's factory-owned node/link registries. They do not
copy registry entries or graph data and must not outlive the parent. Fixed-node
induced views use a `SearchMask`; predicate views use the frozen endpoint/
edge-ID predicate surface. Any predicate reading an attribute captures an
already-resolved `AttrId`; it may not resolve or hash a name on invocation.
View bindings remain parent bindings. Structural mutation through the view is
unsupported, while live attribute mutation follows frozen view rules.

`subnetwork` and `get_subnetwork_view` are aliases, not separate algorithms.
Materializing a standalone BaseNetwork from a view is an explicit copy API and
must rebuild definition/value bindings.

## Copy and lifetime

Direct BaseNetwork copy construction/assignment is deleted because the frozen
factory registries uniquely own their polymorphic entries. Move construction
and assignment transfer those registries and immediately rebuild every
registry/graph identity-bearing value binding; no binding pointer into the
moved-from object's members survives.

`clone()` deep-copies topology, recursive graph/node/link values, dynamic
metadata, and cached cardinality values. Because factory registries are
non-copyable, it reconstructs fresh node/link registries from the canonical
cold setting snapshots through `create_node_attributes_from_setting` and
`create_link_attributes_from_setting`; it never copies owning pointers or
downcasts entries. Alias relationships inside the copied recursive value graph
are preserved where the frozen graph copy contract preserves them, but the
clone shares no mutable attribute or graph storage with the source. Factory
registry order and graph attribute-registry order are preserved. All
identity-bearing bindings are rebound to the clone.

Copies and moves invalidate externally retained definition references,
bindings, views, `AttrValue&`, and registry-backed string views unless the
specific operation documents otherwise. A BaseNetwork view is non-owning.
The mutable `graph()` escape hatch follows the frozen graph lifetime rules and
does not silently refresh Python-compatible cardinality caches.

## GML behavior

`prepare_gml_graph` materializes a plain undirected Graph in current node and
edge order and copies their attributes. It does not start by copying graph
metadata. It then:

1. writes `node_attrs_setting`, then `link_attrs_setting`, using the completed
   `flatten_dict_list_for_gml` compatibility transformation;
2. visits original graph metadata in insertion order, skipping those two keys;
3. for a mapping value, emits one level as `key___subkey` with each subvalue
   converted to its Python-compatible string;
4. otherwise copies the value directly.

Key collisions overwrite the earlier value without moving its insertion
position. The config/GML boundary may handle strings, but its graph copy loops
must traverse direct IDs/ordered slots and resolve each output key once.

`from_gml` delegates frozen `read_gml(path, label)`. The supported native
dense core requires `label="id"`; arbitrary string labels and Python's
conditional relabel objects are a documented representation boundary. After
loading, construct BaseNetwork from the graph, run the exact sample existence
check, then restore non-setting metadata in loaded insertion order. A key
containing `___` must split into exactly two components; multiple delimiters
are an error at restore stage. Reconstructed subkeys form an ordered dynamic
object under the main graph key, and the flattened key is erased. Other keys
remain graph metadata and update recognized fixed-field mirrors only at the
cold boundary. Earlier reconstruction persists on failure.

The frozen GML loader already handles recursive metadata; this BaseNetwork
layer still preserves its historical flatten/unflatten profile for files it
produces. It must not modify the frozen GML implementation.

## Setting export

`save_attrs_dict(path)` builds a `SettingDocument` whose root keys are exactly
`graph_attrs_dict`, `node_attrs`, and `link_attrs` in that order. Graph metadata
includes the two stored definition-setting snapshots. Node/link arrays contain
attribute snapshots in factory-registry order. Conversion from direct typed
attribute fields and `AttrValue` occurs once at this serialization boundary.

Call the completed non-strict `write_setting`. Preserve its wrong-suffix
create/truncate-and-return-error behavior: BaseNetwork ignores the returned
error and completes like Python `None`. Filesystem/open/serialization
exceptions otherwise propagate at the dependency stage. Do not reopen or
duplicate the setting writer.

## Typed errors

```cpp
enum class BaseNetworkErrorCode : std::uint8_t {
    invalid_config,
    missing_config_field,
    attribute_registry_mismatch,
    graph_binding_mismatch,
    no_nodes,
    no_links,
    missing_node_attribute,
    missing_link_attribute,
    generated_length_mismatch,
    empty_attribute_selection,
    non_numeric_benchmark_value,
    ragged_benchmark_matrix,
    invalid_gml_flattened_key,
};

enum class BaseNetworkOperation : std::uint8_t {
    decode_config,
    construct,
    bind_attribute,
    check_attributes,
    generate_topology,
    generate_node_attributes,
    generate_link_attributes,
    get_attribute_data,
    set_attribute_data,
    prepare_benchmarks,
    prepare_gml,
    restore_gml,
    save_attributes,
};

class BaseNetworkException : public std::runtime_error {
public:
    BaseNetworkException(BaseNetworkErrorCode,
                         BaseNetworkOperation,
                         std::size_t input_index,
                         std::string message);
    BaseNetworkErrorCode code() const noexcept;
    BaseNetworkOperation operation() const noexcept;
    std::size_t input_index() const noexcept;
};
```

Dependency errors retain their existing typed class unless BaseNetwork adds
essential input-index context. If wrapped, preserve the original exception as
a nested cause and lock the BaseNetwork stage/index. Validation finishes
before a leaf operation only when Python also validates before side effects;
otherwise retain ordered partial mutation.

## Hot-loop and threading contract

- Attribute names resolve once to factory `AttributeRegistryId`; each
  attribute's value name resolves separately to the owning Graph's `AttrId`
  once.
- Fixed owner/kind/generative/restriction/checking/topology fields are direct
  typed members/enums. No fixed config is kept in a string map after decode.
- Node/edge/data loops receive registry identity, `AttributeRegistryId`, `AttrId`,
  contiguous `Vertex`/edge descriptors, direct attribute objects, and
  pre-sized result slots. No such loop calls `bind`, `attr_id`, registry name
  lookup, string `find/at/set`, or compares a kind/aggregation string.
- Typed-attribute/number-lane dispatch occurs outside the element loop. A
  repeated predicate captures resolved IDs. A benchmark consumer binds output
  names to `AttributeBenchmarkId` before its own normalization loop.
- Shared RNG generation, definition-order generation, existence first-failure,
  GML restoration, and multi-definition mutation remain sequential.
- Independent read-only rows may use deterministic contiguous blocks after all
  definitions/bindings are validated. Output/error slots are pre-sized and the
  lowest input-index failure wins. A definition setter may parallelize only
  disjoint node/edge slots through its completed leaf API.
- Worker count is a direct caller/config field. Zero/one selects sequential;
  wider values are bounded by independent work and CPU allowance. BaseNetwork
  contains no auto-tuned width, size threshold, or machine-specific default.
- Concurrent read-only calls on one immutable network and independent mutation
  of distinct networks are supported. Mutation may not race with reads,
  bindings, views, serialization, or another mutation of the same network.

## Deliberate Python-only boundaries

- OmegaConf object truthiness/reflection and arbitrary config mappings;
- arbitrary hashable/sparse node labels (native vertices are contiguous);
- arbitrary attribute objects, monkey-patching, descriptors, `setattr`, method
  shadowing, and `__getitem__` returning the `TypeError` class;
- Python `assert` disappearing under `-O`;
- mixed name/object lists selected solely by `isinstance(first, str)`, string
  substring membership, user-defined `__contains__`, and object identity;
- NumPy/SciPy object identity, ragged/object dtypes, unbounded integers, and
  arbitrary numeric conversion protocols;
- callable Python subgraph predicates that cannot be represented as the frozen
  typed filter surface;
- arbitrary GML labels, DictConfig values, and `str`/`repr` side effects; and
- Python `None` in arbitrary graph metadata, because frozen `AttrValue` has no
  null lane (null remains representable only in the cold `SettingDocument`).
- With an all-types manager request, Python can use a present `None` originator
  as a dictionary key for non-extrema attributes; the native string-key domain
  applies the documented attribute-name fallback.

These boundaries receive explicit oracle characterization and are never used
to justify dynamic fixed-field storage in production C++.

## Accepted implementation gate

The isolated unit/differential must cover at least:

- empty/incoming/config construction, every merge order, duplicate override
  with first position retained, `None` graph settings, and graph-setting/extra
  overwrite order, including an invalid overwritten item whose non-name fields
  are never decoded;
- all eight factory-supported node/link owner-kind pairs, factory errors,
  factory-registry and graph identity, unrelated equal IDs, non-copyable
  registry move/clone/rebind behavior, and proof that no graph-attribute
  factory registry is invented by BaseNetwork; factory construction covers
  workers `0/1/2/8`, lowest-index failure, ordered duplicate replacement, and
  node-success/link-failure atomic replacement in `create_attrs_from_setting`;
- first-read cardinality caches, stale values after graph mutation/topology,
  independent link/edge caches, live-count extension, and explicit invalidation;
- topology success/failure and shared RNG continuation;
- ordered node-before-link generation, generative/extrema selection, exact RNG
  continuation, length mismatch before set, prior partial mutation, resolved
  originators, workers `0/1/2/8`, and concurrent independent networks;
- all filters including both-present precedence and empty-present filters,
  feature counts, empty row-get bug, missing/ragged data, edge/node order,
  adjacency and every typed aggregation;
- sample-only existence checks, empty node/link precedence, later-node/link
  omissions deliberately ignored, and definition-order first failure;
- manager preparation for default/all/disabled groups, exact float32 bits,
  link virtual repetition, sum rows, nonnumeric/ragged errors, and byte-for-byte
  equality with the already-completed manager (never a second reduction);
- graph metadata set/select/order, reserved names, supported repr, clone depth,
  induced/predicate live views and lifetime;
- GML flatten collisions/order, zero/one/multiple delimiters, load check order,
  reconstructed objects, and setting export including wrong suffix behavior;
- Python-only boundaries listed above as recorded non-native cases; and
- strict warnings, ASan/UBSan/leaks, full CTest, frozen integrity, and scoped
  whitespace checks.

Only after exact differential passes, run one compact representative benchmark
for this new component. It may contain a few rows for resolved multi-attribute
data gathering/setting and the prepared manager adapter at caller-configured
workers `1/2/8`, with one warm-up and three samples. Fixture creation, config
decode, process startup, serialization, and checksum work stay outside timed
regions. Check exact value types/raw bits/order/RNG continuation and output
bytes before timing. Tune production only if output differs or C++ is slower.
Once the rows pass, freeze this benchmark source, command, result, hashes, and
timings immediately; never rerun or update it during later network modules.

The completed gate passed the isolated broad unit, 29 exact shared Python
cases, and 11 recorded Python-only boundaries (40 total). GCC 11 strict
production/unit/harness/benchmark checks, ASan/UBSan/leaks, release CTest
30/30, and frozen-foundation integrity pass.

The accepted 8,192-element benchmark is frozen. Resolved get paths are
3.245x-6.326x faster than Python, setters are 2.398x-6.495x faster, and the
prepared manager adapter is 67.875x-103.983x faster at caller-configured
workers `1/2/8`. The sparse manager path retained checksum
`7234129939705340940`. See `../results/base_network_2026-07-28.md`; do not
rerun or update its benchmark source, driver, binary, JSON, or timings.
