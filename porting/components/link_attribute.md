# Component API: `network.attribute.link_attribute`

State: **COMPLETE** on 2026-07-28.

Source: `../virne/virne/network/attribute/link_attribute.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`A95CFD2B8E2B46D4DE23F70934CA942DE502E674E2A7EAA139482DF640E8646F`,
9,026 bytes. Completed NodeAttribute, BaseAttribute, attribute-method, dataset/RNG,
and frozen graph API documents were read before this leaf; completed dependency
source remains closed.

## Python behavior to retain

- `LinkAttribute` gets an endpoint-pair value, sets mappings or edge-order dense
  inputs, and returns attributed values in NetworkX edge order. A short dense
  input fails during conversion before the first graph write; extra values are
  ignored. Missing values are omitted by `get_data`.
- Adjacency data is the dense result of NetworkX `attr_sparse_matrix` with node
  order fixed to `list(network.nodes)`. Aggregation supports only sum, mean,
  max, and min on axis zero after materialization. Normalization is performed
  by the frozen exact graph sparse API.
- Abstract `update_path` always raises its fixed not-implemented error.
- Status is link/status information. Extrema requires a non-`None` originator,
  is link/extrema information, and delegates to that already-resolved link
  attribute. Resource is link/resource, defaults to hard/link checking, and
  uses the completed exact resource/check policy.
- Resource path updates validate type, method, and path length in that order,
  then update adjacent path edges sequentially. Failure may leave earlier path
  edges mutated and therefore this operation cannot be parallelized.
- Latency is link/latency, defaults to hard/path checking, sums physical path
  values in list order, then uses the completed satisfiability policy. Generic
  generation delegates to BaseAttribute. Distribution `position` instead scans
  the first node's attribute names in insertion order for the first name
  containing `pos`, resolves it once, computes Euclidean endpoint distances in
  edge order, and applies `distance * (max-min) + min`.

## Stable C++ surface for implementation

```cpp
struct LinkAttributeBinding { AttrId value_id = 0; };
struct LinkAttributePairBinding {
    LinkAttributeBinding virtual_graph;
    LinkAttributeBinding physical_graph;
};
struct LinkAttributeAssignment {
    Vertex source = 0;
    Vertex target = 0;
    AttrValue value;
};
enum class LinkAggregation : std::uint8_t { sum, mean, maximum, minimum };

class LinkAttribute : public BaseAttribute {
public:
    explicit LinkAttribute(BaseAttributeSpec);
    LinkAttributeBinding bind(const Graph&) const;
    LinkAttributeBinding bind(const DiGraph&) const;
    const AttrValue& get(const Graph&, Vertex, Vertex, LinkAttributeBinding) const;
    const AttrValue& get(const DiGraph&, Vertex, Vertex, LinkAttributeBinding) const;
    void set_data(Graph&, const std::vector<LinkAttributeAssignment>&,
                  LinkAttributeBinding) const;
    void set_data_dense(Graph&, const std::vector<AttrValue>&,
                        LinkAttributeBinding, std::size_t workers = 1) const;
    std::vector<AttrValue> get_data(
        const Graph&, LinkAttributeBinding, std::size_t workers = 1) const;
    DistanceMatrix get_adjacency_data(
        const Graph&, LinkAttributeBinding, bool normalized = false,
        std::size_t workers = 1) const;
    std::vector<double> get_aggregation_data(
        const Graph&, LinkAttributeBinding, LinkAggregation,
        bool normalized = false, std::size_t workers = 1) const;
};

class LinkStatusAttribute final : public LinkAttribute { /* typed spec */ };
class LinkExtremaAttribute final : public LinkAttribute {
public:
    std::vector<AttrValue> generate_from_resolved_originator(
        const Graph&, const LinkAttribute&, LinkAttributeBinding,
        std::size_t workers = 1) const;
};
class LinkResourceAttribute final : public LinkAttribute {
public:
    bool update_path(const AttrMap& virtual_link, AttrId virtual_id,
                     Graph& physical_graph, const std::vector<Vertex>& path,
                     LinkAttributeBinding physical_binding,
                     ResourceUpdateOperation, bool safe = true) const;
};
class LinkLatencyAttribute final : public LinkAttribute {
public:
    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_link, AttrId virtual_id,
        const std::vector<const AttrMap*>& physical_path, AttrId physical_id,
        ComparisonOperation method = ComparisonOperation::greater_equal) const;
    std::vector<double> generate_from_position(
        const Graph&, NodeAttributeBinding position_binding,
        std::size_t workers = 1) const;
};
```

Every Graph API above has a corresponding DiGraph overload. Exact typed specs,
error enums, cold string resolvers, and overload declarations are finalized in
the implementation header.

## ID and performance rules

- `bind(graph)` is the only dynamic link-name-to-`AttrId` boundary. Virtual and
  physical graph IDs are resolved separately and never copied between unrelated
  registries. Aggregation strings resolve once to `LinkAggregation`.
- Edge loops collect or traverse descriptors in frozen NetworkX order and use
  direct `edge_attrs(edge).find/set(AttrId)`. No edge/path worker hashes, resolves,
  or compares a dynamic string. Position-name substring scanning is a cold
  one-time resolver; distance loops receive only its `AttrId`.
- Sparse writes and path updates stay sequential for public order/partial-side
  effects. Dense unique-edge writes/reads, dense matrix conversion, and independent
  position distances may use deterministic contiguous worker blocks.
- Worker count is caller configuration: zero/one sequential, wider values are
  count/hardware capped. No host-specific threshold is embedded.

## Gate

Cover Graph/DiGraph, loops and reversed endpoints, edge order after non-monotonic
insertion, sparse/dense/short/extra data, missing/recursive/non-numeric values,
normalized and raw matrices, all aggregations including empty and non-finite
cases, status/extrema/resource/latency invariants, path validation and partial
mutation, latency sum order, position resolver order and distance transform,
workers `0/1/2/8`, and concurrent independent graphs.

Use only representative dense roundtrip and position-latency timings. Exact
output/checksum gates precede timing; tune only if C++ is slower or differs.

## Accepted verification and performance

The isolated unit covers `Graph`/`DiGraph`, non-monotonic edge order, loops and
reversed endpoints, short dense-input atomicity, separate registries, raw and
normalized matrices, every aggregation including NaN and signed zero,
status/extrema/resource/latency invariants, ordered partial path mutation,
position resolution, workers `0/1/2/8`, and concurrent independent graphs.
The direct-source oracle passed 35 C++/Python cases plus five explicitly
recorded Python-only protocol boundaries (40 total). Because the pinned non-ML
oracle image has no SciPy, the comparator uses a controlled dense sparse facade
only for the exact materialized behavior consumed by this leaf; the frozen
graph sparse implementation retains its own full differential gate.

The compact frozen benchmark used 50,000 edges, workers `1/2/8`, one warm-up,
and three samples. Every row passed exact checksum/output-byte gates. Dense
roundtrip was `5.722x` to `7.408x` faster than Python; position-derived latency
was `8.550x` to `72.780x` faster. Worker width remains caller configuration;
path mutation stays sequential because order and partial side effects are
public. No tuning was performed after the gate passed, and the benchmark must
not be rerun or updated during dependent-module work.

GCC 11 strict warnings for production/unit/harness/benchmark pass. ASan/UBSan,
Release CTest, and frozen-integrity evidence are recorded in
`../results/link_attribute_2026-07-28.md`.
