# Component API: `core.controller.ConstraintChecker`

State: **COMPLETE / FROZEN** on 2026-07-29.

Python oracle: `../virne/virne/core/controller/constraint_checker.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`EA41EE9226CEF3F38CFFF30D7ABB276A7A9AF58FEE67CE78C3763413BDB9619A`,
7,984 bytes. Completed Solution, utils-network, BaseNetwork,
Virtual/PhysicalNetwork, and node/link/graph attribute documents were read
first. The Python leaf was opened once because no checker contract existed and
the C++ scaffold was empty.

## Exact Python behavior

- Generic checking visits every attribute in list order, never short-circuits,
  and returns `True` for an empty list. Duplicate names overwrite the earlier
  value while ordinary dict insertion position remains first.
- Graph checking delegates the complete graph-attribute list unchanged.
- Node checking first asserts that the physical node exists, then reads the
  virtual and physical node maps and performs the generic check.
- Link checking reads the virtual edge map before the physical edge map, then
  performs the generic check.
- Path checking converts consecutive physical nodes to ordered links first.
  It performs complete link-level checking for every physical link, then
  transposes per-link offsets to the Python attribute-first mapping, then
  checks path-level attributes against the ordered physical-map list. It never
  stops after a failed link or attribute. The final flag is link-level AND
  path-level.
- A path shorter than two nodes fails in the completed `path_to_links` helper.
  Python arbitrary mappings, monkey-patched check methods, custom result
  truthiness, unbounded node labels, and duplicate dynamic attribute objects
  remain language boundaries.

## Fixed-field and ID rule

Every checker category, result flag, link endpoint, error, operation, and
selection is a direct field or enum. Attribute names are genuinely dynamic;
the preparation boundary reads each selected virtual-registry entry's name
once to bind independent virtual/physical `AttrId` values. Registry IDs are
never copied between networks. Prepared and hot checking loops retain only
`AttributeRegistryId`, `AttrId`, typed concrete attribute pointers, direct
network pointers, vertices, and pre-sized result slots. No fixed field and no
hot-loop lookup uses a string-keyed map.

Unsupported dynamic combinations are rejected during preparation:
node-at-node must be `NodeResourceAttribute`, link-at-link must be
`LinkResourceAttribute`, link-at-path must be `LinkLatencyAttribute`, and graph
constraints must be `GraphResourceAttribute`. This records the Python errors
that would otherwise arise from calling an incompatible dynamic method.

## Stable native API

```cpp
namespace virne::core::controller {

using ConstraintId = network::attribute::AttributeRegistryId;

struct ConstraintLink { Vertex source, target; };
struct GraphConstraintSelection {
    ConstraintId output_id;
    const network::attribute::GraphResourceAttribute* attribute;
};
struct ConstraintCheckerSelection {
    std::vector<ConstraintId> node_at_node;
    std::vector<ConstraintId> link_at_link;
    std::vector<ConstraintId> link_at_path;
    std::vector<GraphConstraintSelection> graph;
};

struct ConstraintCheckResult {
    bool feasible;
    SolutionAttributeValues offsets; // direct slots by resolved ID
};
struct PhysicalLinkConstraintResult {
    ConstraintLink physical_link;
    SolutionAttributeValues offsets;
};
struct PathConstraintCheckResult {
    bool feasible;
    std::vector<PhysicalLinkConstraintResult> link_level;
    SolutionAttributeValues path_level;
};

struct NodeConstraintRequest { Vertex virtual_node, physical_node; };
struct LinkConstraintRequest {
    ConstraintLink virtual_link, physical_link;
};
struct PathConstraintRequest {
    ConstraintLink virtual_link;
    std::vector<Vertex> physical_path;
};

class ConstraintChecker {
public:
    explicit ConstraintChecker(ConstraintCheckerSelection);
    PreparedConstraintChecker prepare(
        const network::VirtualNetwork&,
        const network::PhysicalNetwork&) const;
};

class PreparedConstraintChecker {
public:
    ConstraintCheckResult check_graph_constraints() const;
    ConstraintCheckResult check_node_level_constraints(Vertex, Vertex) const;
    ConstraintCheckResult check_link_level_constraints(
        ConstraintLink, ConstraintLink) const;
    PathConstraintCheckResult check_path_level_constraints(
        ConstraintLink, const std::vector<Vertex>&) const;

    std::vector<ConstraintCheckResult> check_node_level_constraints_batch(
        const std::vector<NodeConstraintRequest>&,
        std::size_t workers = 1) const;
    std::vector<ConstraintCheckResult> check_link_level_constraints_batch(
        const std::vector<LinkConstraintRequest>&,
        std::size_t workers = 1) const;
    std::vector<PathConstraintCheckResult> check_path_level_constraints_batch(
        const std::vector<PathConstraintRequest>&,
        std::size_t workers = 1) const;
};
}
```

Preparation is the sole dynamic-name boundary and is not timed as a hot check.
Node/link/path selection IDs belong to the virtual network's typed registries.
Graph output IDs are direct slots, may be sparse or duplicate, and reject only
the reserved invalid ID; a duplicate retains its last checked offset.
Prepared objects are non-owning and must not outlive or move their networks or
selected graph definitions. Network/registry mutation may not race with a
prepared check.

Path preparation first preserves the frozen `path_to_links` validation, then
resolves the immutable virtual edge once before the first physical edge. Each
physical link is still visited in order and every selected attribute is still
checked after a failure. This removes repeated graph lookup without changing
the non-dynamic native result or first-error order.

Scalar operations retain Python evaluation and first-error order. Batch
extensions process complete independent scalar requests in deterministic
contiguous blocks; worker zero/one is sequential, wider values are capped by
request count, results retain input order, and the lowest failing request wins.
There is no host-derived automatic worker policy.

## Accepted gate

Cover empty/all-success/mixed hard-soft constraints, exact numeric types and
offsets, duplicate IDs, node/link/path/graph ordering, missing vertices/edges,
short paths, link/path combined failure, preparation family/level errors,
worker `0/1/2/8` equality and concurrent callers. Compare exact Python values
and runtime, run one compact checksum-gated benchmark only after correctness,
then freeze it permanently. Strict warnings, ASan/UBSan/leaks, CTest, and
frozen-foundation integrity are required.

The implementation passed all gates: 14 exact shared Python cases, seven
native unit groups, strict production/unit/harness compilation, ASan/UBSan and
leak detection, and targeted CTest 2/2 including frozen-foundation integrity.
The permanently frozen 32,768-request node-batch benchmark retained checksum
`11118347938320421286`; C++ was `46.384x`, `46.694x`, and `74.694x` faster at
configured workers `1/2/8`. See
`porting/results/constraint_checker_2026-07-29.md`. Do not rerun or update the
accepted benchmark.

## Integration worker correction (2026-07-30)

Batch APIs now dispatch contiguous blocks through the persistent deterministic
executor. The first error is still selected by request index, worker `0/1` is
canonical sequential, nested dispatch is sequential, and no host width is
selected. Focused batch/error/concurrent-caller units pass; the frozen benchmark
was not rerun.
