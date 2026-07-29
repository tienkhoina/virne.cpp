# Component API: `network.attribute.graph_attribute`

State: **COMPLETE** on 2026-07-28.

Source: `../virne/virne/network/attribute/graph_attribute.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`DFA858918068A792CB0A673B400EE6F3F93107F1D17BAAD28F72546F50FBCEDE`,
4,308 bytes and 87 lines. Before the source read, the completed BaseAttribute,
attribute-method, LinkAttribute, frozen graph API, and port-wide performance
documents were read. Completed dependency implementations remain closed; only
their public headers were opened where constructor/error signatures were not
fully specified by the notes.

## Python behavior to retain

- `GraphAttribute.get` and `get_data` both validate that the dynamic name is not
  `None`, then return `network.graph[name]` directly. `set_data` assigns the
  same slot directly. An empty name is valid; a missing key fails at lookup.
- Status fixes owner/kind to graph/status and Information policy overwrites
  `is_constraint=false`.
- Extrema requires a non-`None` originator and is graph/extrema/non-constraint.
  Preserve the original method quirk: every owner other than node selects the
  link definition registry, so GraphExtrema delegates to a resolved
  `LinkAttribute`, not to a graph-attribute registry. Ordinary Python
  config/kwargs can also duplicate the explicit `originator` argument; that
  call-binding failure is a recorded Python-only boundary.
- Resource fixes graph/resource/constraint. Restriction precedence is
  `config.constraint_restrictions`, `config.restriction`,
  `kwargs.constraint_restrictions`, `kwargs.restriction`, then hard; checking
  level defaults to graph.
- Resource checking reads virtual and physical graph slots, in that order,
  before missing validation or numeric conversion, then applies the completed
  exact satisfiability policy. Python `None` is missing; native `AttrValue` has
  no null lane.
- Resource update validates type then operation before lookup. Its unusual
  observable behavior mutates the first graph by the second graph:
  `target += operand` or `target -= operand`. It reads target then operand
  before arithmetic, never reads `safe`, permits negative subtraction, and  `
  returns true. Self-alias doubles on add and becomes zero on subtract.

## Stable C++ surface

```cpp
class GraphAttribute;
struct GraphAttributeBinding {
    AttrId value_id = 0;
    const AttrMap* graph_identity = nullptr;
    const GraphAttribute* definition_identity = nullptr;
};
struct GraphAttributePairBinding {
    GraphAttributeBinding virtual_graph;
    GraphAttributeBinding physical_graph;
};
struct GraphAttributeConstSlot {
    const AttrMap* graph_attributes = nullptr;
    GraphAttributeBinding binding;
};
struct GraphAttributeMutableSlot {
    AttrMap* graph_attributes = nullptr;
    GraphAttributeBinding binding;
};

class GraphAttribute : public BaseAttribute {
public:
    explicit GraphAttribute(BaseAttributeSpec);
    GraphAttributeBinding bind(const Graph&) const;
    GraphAttributeBinding bind(const DiGraph&) const;
    const AttrValue& get(const Graph&, GraphAttributeBinding) const;
    const AttrValue& get(const DiGraph&, GraphAttributeBinding) const;
    const AttrValue& get_data(const Graph&, GraphAttributeBinding) const;
    const AttrValue& get_data(const DiGraph&, GraphAttributeBinding) const;
    void set_data(Graph&, const AttrValue&, GraphAttributeBinding) const;
    void set_data(DiGraph&, const AttrValue&, GraphAttributeBinding) const;
    std::vector<AttrValue> get_data_batch(
        const std::vector<GraphAttributeConstSlot>&,
        std::size_t workers = 1) const;
    void set_data_batch(
        const std::vector<GraphAttributeMutableSlot>&,
        const std::vector<AttrValue>&,
        std::size_t workers = 1) const;
};

class GraphStatusAttribute final : public GraphAttribute { /* typed spec */ };
class GraphExtremaAttribute final : public GraphAttribute {
public:
    std::vector<AttrValue> generate_from_resolved_originator(
        const Graph&, const LinkAttribute&, LinkAttributeBinding,
        std::size_t workers = 1) const;
    std::vector<AttrValue> generate_from_resolved_originator(
        const DiGraph&, const LinkAttribute&, LinkAttributeBinding,
        std::size_t workers = 1) const;
};
class GraphResourceAttribute final : public GraphAttribute {
public:
    bool update(AttrMap& target, AttrId target_id,
                const AttrMap& operand, AttrId operand_id,
                ResourceUpdateOperation, bool safe = true) const;
    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_graph, AttrId virtual_id,
        const AttrMap& physical_graph, AttrId physical_id,
        ComparisonOperation = ComparisonOperation::less_equal) const;
};
```

Exact typed specs, error codes/stages, accessors, and overload declarations are
finalized in the implementation header. Fixed owner/kind/is-constraint,
restriction, and checking-level values are direct invariants.

## ID, threading, and performance rules

- `bind(graph)` is the only dynamic value-name-to-`AttrId` boundary. Its two
  raw identity tokens make a binding local to both the graph map and typed
  attribute definition, so a same-range ID from another graph/field is rejected
  with pointer comparisons and no repeated name lookup. Virtual, physical, and
  every batch graph resolve independently. Originator name resolves once to an
  `AttributeDefinitionId` in the explicitly fixed link registry.
- Scalar graph storage validates compact identities, then uses direct
  `graph_attrs().at/set(id)`. Batch slots are identity-validated once before
  dispatch; workers touch only pointers, `AttrId`, direct `AttrMap`, pre-sized
  result slots, and indices. No worker or batch-slot validation resolves,
  hashes, or compares a string. Bindings are non-owning and must not outlive
  their graph or attribute definition; copies are rebound explicitly.
- Worker zero/one is sequential. Wider caller-configured values use
  deterministic contiguous blocks capped by item count and CPU allowance.
  Parallel writes require independent graph maps; duplicate mutable maps fall
  back to the canonical sequential order. Shared resource mutation is scalar
  because update/alias order is observable.
- Batch get/set is a native throughput extension over independent graph
  instances. It does not change the scalar Python surface or object-identity
  contract; scalar get/get_data continue returning a live const reference.

## Gate

Cover Graph/DiGraph, empty/different registry names, get/get_data/set
insert/overwrite/missing, every AttrValue lane and raw floating bits, status and
extrema invariants/link-registry quirk, resource restriction/checking fields,
all comparisons/restrictions, missing/nonnumeric access precedence, target-first
updates, ignored `safe`, negative result, bool promotion, typed overflow, and
self-alias. Batch tests cover empty/null/shape errors, duplicate-map sequential
fallback, workers `0/1/2/8`, exact order, and concurrent independent callers.
Record Python-only `None`, arbitrary-object, unbounded-int, duplicate-argument,
and reflection boundaries explicitly.

After exact differential passes, run one compact independent-graph roundtrip
benchmark against Python at configured workers `1/2/8`, one warm-up and three
samples. Exact tag/checksum/raw-bit output gates precede timing. Once it passes
and C++ is faster, freeze its source/result immediately and never rerun or tune
it during dependent-module work.

## Accepted verification and performance

The isolated unit covers Graph/DiGraph, all `AttrValue` lanes and raw floating
bits, live-reference semantics, empty/missing fields, independent and forged
bindings, fixed typed leaves, the GraphExtrema-to-Link quirk, resource
precedence/arithmetic/aliasing, batch validation/atomicity/duplicate-map order,
workers `0/1/2/8`, and concurrent independent callers. The direct-source oracle
passed 45 shared C++/Python cases plus four native batch/range cases and five
recorded Python-only dynamic boundaries (54 total).

The single frozen benchmark uses 20,000 independent graphs, raw64 double output,
workers `1/2/8`, one warm-up, and three samples. Exact type, byte count, and
checksum gates passed. C++ speedups are `2.595x`, `1.828x`, and `1.209x`
respectively. The initial worker-1 attempt exposed unnecessary sequential
duplicate detection/default construction; production removed those costs, the
same unchanged benchmark passed, and it was frozen immediately. Do not rerun,
retune, or update its source/result during dependent-module work.

Strict GCC 11 warnings for production/unit/harness/benchmark, ASan/UBSan/leaks
for unit/harness, Release CTest 27/27, and frozen-component integrity pass.
Exact hashes and timing rows are in
`../results/graph_attribute_2026-07-28.md`.
