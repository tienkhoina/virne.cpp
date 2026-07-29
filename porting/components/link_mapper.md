# Component API: `core.controller.LinkMapper`

State: **COMPLETE / FROZEN** on 2026-07-29. Production, unit, differential,
benchmark sources, and accepted artifacts must not be edited or rerun.

Python oracle: `../virne/virne/core/controller/link_mapper.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`E7E5FB542D6FCA6A5C9A8BEBACE2C82E9BFEDEF1628CD8ADBB065C4F95D40237`,
28,660 bytes. Completed Solution, ConstraintChecker, ResourceUpdator,
TopologyAnalyzer, LinkAttribute, utils-network, BaseNetwork, and Graph notes
were read first. Completed dependency implementation remains closed. Solver,
Controller orchestration, system, and learning/ML are outside this leaf.

## Fixed-field and ID rule

Shortest-path method, safe/unsafe mode, reusable/inplace/record flags, virtual
and physical endpoints, paths, results, errors, operations, placeholder state,
and worker widths are direct fields or enums. Link resources and link/path
constraints are virtual-registry IDs. `prepare()` resolves every genuinely
dynamic resource name once to its virtual graph `AttrId`; all candidate,
physical-link, resource, pooling, violation, undo, and mapping loops retain only
vertices, registry IDs, graph-local IDs, typed numeric variants, byte masks,
and direct Solution tables. No fixed field uses a string-keyed map and no hot
loop resolves, hashes, stores, or compares a string.

Commit resources cold-collapse duplicate IDs in first-position order because
Python builds a dictionary. Topology/pruning selection retains duplicates
because Python applies those adjustments repeatedly. Link/path constraint order
is stored as compact ID vectors and hard membership is a direct byte mask.

## Stable native API

```cpp
using PhysicalPath = std::vector<Vertex>;
using PhysicalPaths = std::vector<PhysicalPath>;
using LinkPathRanker = std::function<void(PhysicalPaths&)>;

struct LinkMapperSelection {
    std::vector<ConstraintId> link_constraints;
    std::vector<ConstraintId> path_constraints;
    std::vector<ResourceId> link_resources;
    std::vector<ConstraintId> hard_constraints;
    bool reusable = false;
};

struct LinkRouteOptions {
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 1;
    double max_path_nodes = 1.0e6;
    std::size_t topology_constraint_workers = 1;
    std::size_t candidate_workers = 1;
    const LinkPathRanker* ranker = nullptr;
    bool allow_constraint_violation = false;
    bool record_constraint_violation = true;
};

struct LinkMappingOptions {
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 10;
    double max_path_nodes = 1.0e6;
    std::size_t topology_constraint_workers = 1;
    std::size_t candidate_workers = 1;
    bool inplace = true;
    bool allow_constraint_violation = false;
};

struct LinkRouteCheckInfo {
    bool placeholder = false;
    PathConstraintCheckResult constraints;
};
struct LinkRouteResult { bool routed; LinkRouteCheckInfo check; };

class LinkMapper {
public:
    explicit LinkMapper(LinkMapperSelection);
    const LinkMapperSelection& selection() const noexcept;
    PreparedLinkMapper prepare(
        const network::VirtualNetwork&,
        network::PhysicalNetwork&) const;
};

class PreparedLinkMapper {
public:
    LinkRouteResult route(
        ConstraintLink virtual_link, ConstraintLink physical_pair,
        Solution&, LinkRouteOptions = {});
    void record_route_constraint_violation(
        ConstraintLink, const LinkRouteCheckInfo&, Solution&) const;
    bool undo_route(ConstraintLink, Solution&);
    bool link_mapping(Solution&, LinkMappingOptions = {});
    bool link_mapping(
        const std::vector<ConstraintLink>& virtual_links,
        Solution&, LinkMappingOptions = {});
};
```

Prepared objects and optional rank callbacks are non-owning for the duration of
the call. Networks must not move or race mutation. Completed dependency
exceptions retain their typed forms; LinkMapper-specific invalid state uses a
direct `LinkMapperErrorCode`/`LinkMapperOperation` plus numeric context. At the
cold boundary, an out-of-range or non-resource `link_resources` ID is rejected
as `invalid_link_resource_selection` before dependency preparation.

## Python behavior retained

- Identical physical endpoints return immediately only when `reusable=true`;
  they neither write a path nor record violations. Otherwise a typed error is
  raised before mutation.
- Rerouting deletes old path-info entries and writes an empty path but does not
  restore old resources. Safe routing ranks once, checks paths in order, and
  selects the first feasible path. No path returns the typed zero placeholder;
  recording converts every link-level placeholder lane to `100.0`.
- Before commit, the entire selected physical-link path is written. Resource
  updates are physical-link-major/resource-major and info is written only after
  a successful edge update, preserving Python partial state on failure.
- Unsafe single-link routing accepts only `k_shortest`, `all_shortest`, and
  `k_shortest_length`; it ignores the ranker. It still safely commits the first
  feasible path. If none is feasible, it scores all constraints, takes the
  first minimum-violation path, commits with `safe=false`, and returns literal
  success.
- Recording replaces the four per-link pooled tables, sums positive per-edge
  link offsets, clamps path offsets at zero, lets a path lane override the same
  link lane, and accumulates the maximum selected hard violation. Re-recording
  double-counts the total. An empty hard match is a typed error after table
  mutation.
- Undo restores each physical link and erases its info in path order, then
  erases the path. Failure preserves the current path/info while earlier links
  remain restored and erased.
- Safe whole-link mapping clears only path/path-info tables, optionally clones
  the physical network, routes virtual links sequentially from Solution node
  slots, retains earlier work on failure, and leaves success flags unchanged.
  Missing/incomplete mappings become deterministic typed errors.

Python's non-empty unsafe whole-link mapping is not a valid behavior surface:
it passes an unsupported keyword, then contains tuple arithmetic and a
self-reference bug. Native rejects it before mutation. `route_v_links_with_mcf`
is an OR-Tools/SCIP solver path and remains explicitly deferred; no solver code
or dependency enters this target. Arbitrary Python mappings/numerics, monkey
patches, assertion removal, sparse labels, and callback object protocols remain
language boundaries.

## Threading and acceptance gate

Path generation/ranking is sequential. With explicit `candidate_workers>1`,
complete candidate path checks run in deterministic contiguous blocks and
write pre-sized outcome/error slots. The scan remains in original order: an
error before the first feasible path is rethrown, while errors after it are
ignored exactly as Python's early return. Worker zero/one is sequential. Path
commit, violation pooling, undo, and whole-link mapping remain sequential.
There is no host-derived worker policy.

The accepted gate covers same-node/reusable, safe first/none/all-infeasible,
ranking, reroute leak, partial commit, unsafe first-feasible/least-violation/tie,
exact sentinel and pooling types, repeated/empty-hard recording, undo partial
state, mapping order/clone/failure/cardinality, workers `0/1/2/8`, later-error
suppression, and concurrent independent callers. Differential is **17/17
PASS**; strict GCC 11, ASan/UBSan/leaks, and targeted CTest are all **PASS**.

The permanently frozen 1,024-path benchmark uses one warm-up and three samples
at candidate workers `1/2/8`. It retains checksum `14052633754962558449` and
27,830 output bytes. Python median is 8.861967 ms; C++ medians are 0.628866,
0.987043, and 1.720057 ms, respectively: **14.092x, 8.978x, and 5.152x**
faster. Fixture/preparation, subprocess startup, validation, serialization, and
fingerprinting are excluded. This is an AST-isolated LinkMapper leaf benchmark
with equivalent fake Python dependencies, not an end-to-end solver result.
See `porting/results/link_mapper_2026-07-29.md`.
