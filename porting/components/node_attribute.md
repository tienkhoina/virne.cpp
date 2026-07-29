# Component API: `network.attribute.node_attribute`

State: **COMPLETE** on 2026-07-28.

Source: `../virne/virne/network/attribute/node_attribute.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`E90E286320E59EBBBA9957B701F6F12ACDC1785E4821DE80CCC5A0AB0F3EE56A`,
5,782 bytes.

This contract was written after reading only the completed BaseAttribute,
attribute-method, dataset/RNG, and frozen graph API documents. Frozen graph,
CSV, config, yaml-cpp, and random source remain unchanged.

## Python behavior to retain

- `NodeAttribute.get` reads `net.nodes[id][name]`. `set_data` accepts a mapping
  or converts any other indexable input to a node-order mapping before the
  first write. `get_data` returns values for attributed nodes in node order and
  omits missing nodes. All three reject a missing dynamic name first.
- `NodeStatusAttribute` fixes owner/type to node/status and Information policy
  overwrites `is_constraint=False` after Base construction.
- `NodeExtremaAttribute` requires a non-`None` originator from config/kwargs,
  fixes node/extrema, and delegates generation to that already-resolved node
  attribute. Its information policy is non-constraint.
- `NodeResourceAttribute` fixes node/resource, resolves restriction with
  `constraint_restrictions` before `restriction`, defaults checking level to
  node, and is a constraint. A check reads virtual then physical values by the
  dynamic name, rejects either missing value, then uses the completed exact
  satisfiability policy.
- `NodePositionAttribute` fixes node/position, defaults to non-generative,
  radius bounds `0.0..1.0`, and hard restriction. Generative mode performs
  three consecutive Base generation calls on one NumPy stream, clips only the
  radius with NumPy `clip`, then zips `(x,y,r)` in node order. Non-generative
  mode deliberately checks and reads the literal attribute name `pos`, not
  `self.name`; an empty graph fails while selecting the first node. Its
  inherited constraint check remains unimplemented and it does not create a
  `checking_level` field unless dynamic config happened to add one.
- Python config/kwargs merging remains a cold factory concern. Duplicate keys
  created by explicit constructor arguments plus `**config/**kwargs` raise at
  Python call binding; C++ uses direct typed specs and does not retain a
  dynamic fixed-field map.

## Stable C++ surface for implementation

```cpp
struct NodeAttributeBinding { AttrId value_id = 0; };
struct NodeAttributePairBinding {
    NodeAttributeBinding virtual_graph;
    NodeAttributeBinding physical_graph;
};
struct NodeAttributeAssignment { Vertex node = 0; AttrValue value; };

enum class CheckingLevel : std::uint8_t { node, link, path, graph };

class NodeAttribute : public BaseAttribute {
public:
    explicit NodeAttribute(BaseAttributeSpec spec);
    NodeAttributeBinding bind(const Graph&) const;
    NodeAttributeBinding bind(const DiGraph&) const;
    const AttrValue& get(const Graph&, Vertex, NodeAttributeBinding) const;
    const AttrValue& get(const DiGraph&, Vertex, NodeAttributeBinding) const;
    void set_data(Graph&, const std::vector<NodeAttributeAssignment>&,
                  NodeAttributeBinding) const;
    void set_data(DiGraph&, const std::vector<NodeAttributeAssignment>&,
                  NodeAttributeBinding) const;
    void set_data_dense(Graph&, const std::vector<AttrValue>&,
                        NodeAttributeBinding, std::size_t workers = 1) const;
    void set_data_dense(DiGraph&, const std::vector<AttrValue>&,
                        NodeAttributeBinding, std::size_t workers = 1) const;
    std::vector<AttrValue> get_data(const Graph&, NodeAttributeBinding,
                                    std::size_t workers = 1) const;
    std::vector<AttrValue> get_data(const DiGraph&, NodeAttributeBinding,
                                    std::size_t workers = 1) const;
};

struct NodeStatusSpec { /* direct name/generation fields */ };
struct NodeExtremaSpec {
    std::string name;
    std::string originator_name;              // cold diagnostics only
    AttributeDefinitionId originator_id = 0;  // resolved once by registry owner
};
struct NodeResourceSpec { /* direct name/generation/restriction/checking fields */ };
struct NodePositionSpec { /* direct name/generation/radius/restriction fields */ };

class NodeStatusAttribute final : public NodeAttribute { /* typed spec */ };
class NodeExtremaAttribute final : public NodeAttribute {
public:
    std::vector<AttrValue> generate_from_resolved_originator(
        const Graph&, const NodeAttribute&, NodeAttributeBinding,
        std::size_t workers = 1) const;
};
class NodeResourceAttribute final : public NodeAttribute {
public:
    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_node, AttrId virtual_id,
        const AttrMap& physical_node, AttrId physical_id,
        ComparisonOperation method) const;
};

struct NodePositionValue {
    AttributeNumber x;
    AttributeNumber y;
    double radius = 0.0;
};
class NodePositionAttribute final : public NodeAttribute {
public:
    std::vector<NodePositionValue> generate_positions(
        const NetworkCardinality&, NumpyRandomState&,
        std::size_t workers = 1) const;
    std::vector<AttrValue> get_existing_pos_data(
        const Graph&, NodeAttributeBinding literal_pos_binding,
        std::size_t workers = 1) const;
};
```

Exact spec fields, typed error enums, resolver signatures, and Graph/DiGraph
overloads are finalized with the header. Fixed owner/kind/is-constraint values
are construction invariants, not mutable strings.

## ID and performance rules

- `bind(graph)` is the only name-to-`AttrId` boundary. Unrelated virtual and
  physical graphs must be bound separately; numeric IDs are never copied
  between registries.
- Every node loop receives a binding and uses only dense `Vertex`, direct
  `node_attrs(v)`, and `find/at/set(AttrId)`. No loop or worker may resolve,
  hash, or compare an attribute string.
- Sparse assignments remain sequential to preserve input/duplicate-write
  order and ignore out-of-range nodes like NetworkX. Dense unique-node writes
  validate length before mutation and may split disjoint nodes by configured
  worker width. Reads may gather into per-node slots in parallel and compact
  in node order afterward.
- Attribute values use frozen `AttrValue`; numeric checks dispatch once from
  bool/int64/double to completed `AttributeNumber` policy. Strings and recursive
  values fail through a typed error before numeric work.
- Position generation draws x, y, then radius sequentially from one stream.
  Only post-draw clipping/zipping may be parallel. Variant/lane dispatch occurs
  once outside each hot loop.
- Worker count is caller configuration: zero/one sequential, wider values
  count/hardware capped. No machine-specific auto threshold is embedded.

## Gate

Unit/differential coverage must include Graph and DiGraph, missing and partial
attributes, dense/sparse writes and no-partial-write failure, separate graph
registries with different IDs, all AttrValue lanes, resource hard/soft and
missing/non-numeric values, status/extrema invariants, all position branches,
three-call RNG continuation, clip boundaries/NaN/signed zero/reversed bounds,
workers `0/1/2/8`, and concurrent independent graphs.

Use only a few representative timings: dense set/get and position generation,
each with exact output/RNG checks and a small configured-worker smoke. Expand
only for an output mismatch or a row slower than Python. Record API, benchmark,
and performance here immediately when the leaf closes.

## Accepted verification and performance

The isolated unit and direct-source oracle cover both `Graph` and `DiGraph`,
sparse and dense mutation, short-input atomic failure, missing values, separate
graph registries, bool/int64/double/string lanes, status/extrema/resource/position
branches, exact resource arithmetic, NumPy clipping boundaries, three-draw RNG
continuation, workers `0/1/2/8`, and concurrent independent graphs. The oracle
passed 37 differential cases plus five explicitly recorded Python-only dynamic
boundaries (42 total). Resource access-order cases lock missing-value precedence
before numeric conversion for check and every update branch.

The compact benchmark used 100,000 nodes, seed 123, workers `1/2/8`, one
warm-up, and three samples. Every row passed exact checksum/output-byte gates;
position rows also retained the exact following RNG bits. Dense set/get was
`6.002x` to `8.888x` faster than Python. Position generation was `6.490x` to
`8.427x` faster. No benchmark tuning was performed after these gates passed.
The accepted benchmark source/result are frozen and must not be rerun or
updated during dependent-module work.

GCC 11 strict warnings for production/unit/harness/benchmark, ASan/UBSan/leaks,
Release full CTest 26/26, and frozen-component integrity all pass. Exact commands,
artifact hashes, and timing rows are in
`../results/node_attribute_2026-07-28.md`.
