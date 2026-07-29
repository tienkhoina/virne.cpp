# Component API: `core.controller.ResourceUpdator`

State: **COMPLETE / FROZEN** on 2026-07-29. Do not rerun or update the
accepted benchmark.

Python oracle: `../virne/virne/core/controller/resource_updator.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`9B7C14F8C6EAA5E8BC50A723B727FEC7EFF8F1C7D05F7BEF0DA0CC941C15AC85`,
5,172 bytes. Completed Solution, ConstraintChecker, attribute-method,
node/link/graph attribute, and Base/Virtual/PhysicalNetwork documents were read
first. Completed dependency source stayed closed; only public headers were
opened where the notes omitted an update signature. The Python leaf was opened
because no ResourceUpdator contract or implementation existed.

## Exact Python behavior

- `update_resource` validates the operator before the owner. Add ignores
  `safe`; subtract checks availability only when `safe` is truthy. Node failure
  formats a diagnostic and rereads its current value, while link failure has no
  custom message. A successful update mutates one mapping slot and returns
  `None`.
- Node/link multi-resource updates visit the input dictionary in insertion
  order and invoke the scalar operation for each entry. They are sequential and
  may leave earlier attributes mutated when a later update fails.
- Path updating visits `link_resource_attrs` in constructor order. For each
  attribute it resolves the virtual link and invokes that attribute's ordered
  path update. Attribute order is outermost, physical-link order is innermost;
  failure retains every earlier mutation. An empty attribute list performs no
  virtual-link or path validation.
- Python assertion disabling, arbitrary mappings/numeric objects, monkey
  patching, unbounded integers/labels, and dynamic owner/operator strings are
  language boundaries. Native validation is typed and stable in every build.

## Fixed-field and ID rule

Every operation, owner implied by API, endpoint, result/error category,
resource amount, request, and safe flag is a direct field or enum. Selection IDs
belong to the virtual node/link registries. `prepare()` reads each selected
dynamic name once and binds the independent virtual/physical graph `AttrId`
values. Scalar and batch update loops receive only registry IDs, graph-local
value IDs, typed numeric variants, vertices, direct attribute maps, and typed
attribute pointers. No fixed field is stored in a string-keyed map, and no hot
loop resolves, hashes, or compares a string.

## Stable native API

```cpp
namespace virne::core::controller {

using ResourceId = network::attribute::AttributeRegistryId;

struct ResourceAmount {
    ResourceId resource_id;
    network::attribute::AttributeNumber value;
};
struct ResourceUpdatorSelection {
    std::vector<ResourceId> node_resources;
    std::vector<ResourceId> link_resources;
};
struct NodeResourceUpdateRequest {
    Vertex physical_node;
    std::vector<ResourceAmount> resources;
};
struct LinkResourceUpdateRequest {
    ConstraintLink physical_link;
    std::vector<ResourceAmount> resources;
};

class ResourceUpdator {
public:
    explicit ResourceUpdator(ResourceUpdatorSelection);
    PreparedResourceUpdator prepare(
        const network::VirtualNetwork&,
        network::PhysicalNetwork&) const;
};

class PreparedResourceUpdator {
public:
    void update_node_resource(
        Vertex, ResourceAmount, ResourceUpdateOperation, bool safe = true);
    void update_node_resources(
        Vertex, const std::vector<ResourceAmount>&,
        ResourceUpdateOperation, bool safe = true);
    void update_link_resource(
        ConstraintLink, ResourceAmount,
        ResourceUpdateOperation, bool safe = true);
    void update_link_resources(
        ConstraintLink, const std::vector<ResourceAmount>&,
        ResourceUpdateOperation, bool safe = true);
    void update_path_resources(
        ConstraintLink virtual_link, const std::vector<Vertex>& physical_path,
        ResourceUpdateOperation, bool safe = true);

    void update_node_resources_batch(
        const std::vector<NodeResourceUpdateRequest>&,
        ResourceUpdateOperation, bool safe = true, std::size_t workers = 1);
    void update_link_resources_batch(
        const std::vector<LinkResourceUpdateRequest>&,
        ResourceUpdateOperation, bool safe = true, std::size_t workers = 1);
};
}
```

Prepared objects are non-owning. They must not outlive or move their networks,
and network mutation may not race with another operation unless the batch API
owns that dispatch. Duplicate selected link IDs remain in the ordered path list
and therefore repeat the mutation like duplicate Python attribute objects.

## Mutation and threading contract

Scalar resource lists and paths remain sequential with Python partial-mutation
and first-error order. Path conversion is performed once after the first
virtual-link access; its outer attribute/inner link order remains unchanged.
Numeric conversion dispatches directly across bool/int64/double and reuses the
completed `update_resource_value` semantics; strings/recursive values fail
before mutation.

Batch worker zero/one delegates scalar requests in order. Wider configured
workers parallelize only disjoint physical nodes or undirected physical links.
A repeated target falls back to canonical sequential execution. The disjoint
case precomputes every request's final direct-slot values before committing, so
an error chooses the lowest request and performs no batch mutation; successful
commits touch disjoint maps in deterministic contiguous blocks. No host-derived
worker policy is embedded. Path batches are deliberately absent because shared
edge mutation order is public.

## Gate

Cover bool/int64/double arithmetic, add/sub aliases resolved at the cold caller,
safe/unsafe insufficient resources, overflow, missing/nonnumeric values,
selection family/range errors, independent registry order, duplicate IDs,
node/link/path order and partial failure, reversed links, empty/short/missing
paths, workers `0/1/2/8`, disjoint parallel equality, duplicate-target fallback,
lowest batch error, and concurrent independent networks. Use a compact exact
node/link batch benchmark once after differential correctness, then freeze it.

The gate passed 10 exact shared Python cases and seven native unit groups.
Release and strict GCC 11 production/unit/harness builds pass; ASan, UBSan,
leak detection, and targeted CTest (unit plus frozen-foundation integrity) pass.
The permanently frozen 32,768-update benchmark retained checksum
`17411705748429442498`. C++ workers `1/2/8` were respectively 33.113x,
5.542x, and 5.435x faster than Python. Worker count remains an explicit input;
the accepted case favors worker 1 and no automatic host policy is introduced.
See `porting/results/resource_updator_2026-07-29.md`.
