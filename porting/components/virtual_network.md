# Component API: `network.virtual_network`

State: **COMPLETE / FROZEN** on 2026-07-28.

Python source: `../virne/virne/network/virtual_network.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`0B73E73FEB43793559976F08FD93ED227698B810ED8741EA5BCC1534ADB3768C`,
3,352 bytes. The completed BaseNetwork/attribute/graph documents were read
before opening this previously undocumented leaf.

## Python behavior

- Construction delegates entirely to BaseNetwork. Annotated fixed fields
  `id`, `arrival_time`, and `lifetime` are not class defaults; they exist only
  after constructor metadata/reflection or later assignment.
- The downstream simulator writes the fixed request field `max_latency`
  through `set_graph_attribute`; the typed native request therefore stores it
  as a fourth direct optional while retaining the graph metadata write.
- The topology wrapper changes only the dynamic Python default type to
  `random`; native callers pass the completed typed `TopologyRequest`.
- `to_gml` writes `BaseNetwork._prepare_gml_graph()` through NetworkX GML.
- Node/link resource demand selects definitions whose kind is resource,
  gathers rows in definition and graph order, lets NumPy build one rectangular
  array, sums it, and converts to Python float. Any exception at selection,
  gathering, array construction, numeric conversion, or summation returns
  `0.0`.
- Node and link totals recompute on every read. Combined total repeats both
  gathers inside one try and is a `cached_property`: its first success or
  fallback `0.0` remains stale after later graph/resource mutation.

## Stable C++ API

```cpp
class VirtualNetwork final : public BaseNetwork {
public:
    VirtualNetwork();
    explicit VirtualNetwork(BaseNetworkConstruction);
    explicit VirtualNetwork(BaseNetwork&&);
    VirtualNetwork(VirtualNetwork&&) noexcept;
    VirtualNetwork& operator=(VirtualNetwork&&) noexcept;
    VirtualNetwork(const VirtualNetwork&) = delete;
    VirtualNetwork& operator=(const VirtualNetwork&) = delete;

    const std::optional<std::int64_t>& request_id() const noexcept;
    const std::optional<double>& arrival_time() const noexcept;
    const std::optional<double>& lifetime() const noexcept;
    const std::optional<double>& max_latency() const noexcept;
    void set_request_id(std::int64_t) noexcept;
    void set_arrival_time(double) noexcept;
    void set_lifetime(double) noexcept;
    void set_max_latency(double) noexcept;

    double total_node_resource_demand(std::size_t workers = 1) const noexcept;
    double total_link_resource_demand(std::size_t workers = 1) const noexcept;
    double total_resource_demand(std::size_t workers = 1) const noexcept;
    void invalidate_cached_total_resource_demand() noexcept;

    void to_gml(const std::string& path) const;
    VirtualNetwork clone() const;
};
```

Construction resolves the four fixed metadata names at most once and stores
their supported numeric values in direct optional fields. Explicit fixed-field
setters update those fields, not the dynamic graph metadata, matching Python
object assignment. Arbitrary reflection remains a Python-only boundary.

## ID, summation, and threading rules

- Resource kind is the enum `AttributeKind::resource`. Definition selection
  produces compact `AttributeRegistryId`s once. BaseNetwork gathers using its
  graph-local `AttrId` bindings; no demand loop accepts or compares strings.
- Rows must be rectangular to model NumPy array creation. Numeric bool/int64/
  double lanes are summed in definition-major, graph-order layout and returned
  as double. Strings, recursive values, ragged rows, selection/gather errors,
  and integer/float conversion failures are caught at the property boundary
  and produce `0.0`.
- Workers are direct caller configuration forwarded to independent completed
  row gathers. Zero/one is sequential; wider values remain output ordered and
  contain no embedded machine policy.
- The combined cached total records the first result, including fallback zero.
  The explicit invalidation method is a native extension and is never invoked
  implicitly by graph/resource mutation.

## Frozen implementation record

The completed unit and differential gate covers missing/present fixed fields,
constructor metadata, setters, move/clone, zero/one/multiple resource
definitions, node/link ordering, bool/int/double lanes, empty graphs,
missing/ragged/nonnumeric values, exception-to-zero behavior, combined
first-failure order, stale combined cache and explicit invalidation, workers
`0/1/2/8`, concurrent independent networks, and exact GML output. Arbitrary
NumPy dtype/overflow/object protocols and Python reflection remain recorded
dynamic boundaries.

Exact differential is **PASS** for 20 classified cases. The single accepted
32,768-element demand benchmark is **PASS** at caller-configured workers
`1/2/8`, with identical gated output and C++ speedups of `6.926x`, `7.594x`,
and `4.970x`. These artifacts are frozen provenance and must not be rerun or
updated. See `porting/results/virtual_network_2026-07-28.md`.
