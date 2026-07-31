# Component API: `core.controller.Controller` lifecycle

State: **COMPLETE / FROZEN LIFECYCLE** on 2026-07-29, with additive
resource-only transaction and safe fixed-node-slot APIs on 2026-07-30 and
2026-07-31. Do not rerun or edit the accepted component differential or
benchmark artifacts.

Python oracle: `../virne/virne/core/controller/controller.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`352552B0312D18B42EB6904629ED43C76DF2D9C844DF59FE51AB8653FFAC02A5`,
37,463 bytes and 685 physical lines. The frozen ConstraintChecker,
ResourceUpdator, TopologyAnalyzer, NodeMapper, LinkMapper, Solution, network,
attribute, and Graph API documents were read before this contract. Their
accepted production and benchmark sources remain closed and frozen.
The completed ResourceUpdator source was opened once only after the first new
Controller benchmark exposed a non-equivalent fake Python updater and a slow
native nested-vector batch path. That permitted the exact oracle correction
and the direct-ID fast path below; no frozen dependency was changed.

## Port boundary

This component is the non-solver lifecycle seam required by Environment:

- prepare the updater/node/link lifecycle dependencies once for one
  VirtualNetwork and one mutable PhysicalNetwork; checker/topology preparation
  stays inside the completed mappers and is not duplicated;
- place one virtual node and route only its already-placed incident links;
- undo that placement and its incident routes in Python mutation order;
- deploy the resource information already stored in a successful Solution;
- release the resources referenced by a successful Solution;
- safely evaluate one complete insertion-ordered fixed node assignment by
  mapping every node first and every virtual link second;
- preserve the Python `undo_deploy` quirk: release resources but do not reset
  the caller's Solution.

MCF/OR-Tools, `bfs_deploy`, candidate-search helpers, unsafe whole-link
mapping, solver/system integration, observations, rewards, RL, Torch, and ML
are outside this lifecycle target. Safe and supported unsafe single-link
routing are delegated only to the completed LinkMapper enum surface. The
broken Python unsafe `deploy_with_node_slots` path is not fabricated.

## Fixed fields and ID rules

Configuration, shortest-path mode, operation, error, phase, worker widths,
node assignments, endpoints, and result metadata are direct fields/enums.
Node and link resources are registry IDs. Hard node constraints and hard
link/path constraints are separate ID vectors because node and link registry
IDs are local to different registries and equal numeric IDs must never be
interchanged.

`prepare()` and `prepare_mutation()` are the only dynamic attribute-name
resolution boundaries, through the completed dependencies. Controller hot
loops retain only `Vertex`, typed
endpoint pairs, registry IDs, graph-local direct slots, compact Solution entry
IDs, byte membership masks, and numeric variants. No fixed field uses a string
map. No neighbor, route, pool, deploy, release, or worker loop resolves,
hashes, stores, or compares an attribute name.

## Stable native API

```cpp
namespace virne::core::controller {

struct ControllerSelection {
    ConstraintCheckerSelection constraints;
    std::vector<ResourceId> node_resources;
    std::vector<ResourceId> link_resources;
    std::vector<ConstraintId> hard_node_constraints;
    std::vector<ConstraintId> hard_link_constraints;
    bool reusable = false;
};

struct ControllerWorkers {
    std::size_t topology_constraint_workers = 1;
    std::size_t candidate_workers = 1;
};

struct PlaceAndRouteOptions {
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 1;
    double max_path_nodes = 1.0e6;
    ControllerWorkers workers;
    bool allow_constraint_violation = false;
};

enum class ControllerFailurePhase : std::uint8_t { none, place, route };

struct PlaceAndRouteResult {
    bool succeeded = false;
    ControllerFailurePhase failure_phase = ControllerFailurePhase::none;
    NodePlacementResult placement;
    std::optional<LinkRouteResult> last_route;
    std::size_t attempted_routes = 0;
};

struct ControllerMutationOptions { std::size_t workers = 1; };

struct DeployWithNodeSlotsOptions {
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 10;
    double max_path_nodes = 1.0e6;
    ControllerWorkers workers;
};

class Controller {
public:
    explicit Controller(ControllerSelection);
    const ControllerSelection& selection() const noexcept;
    PreparedController prepare(
        const network::VirtualNetwork&,
        network::PhysicalNetwork&) const;
    PreparedControllerMutation prepare_mutation(
        const network::VirtualNetwork&,
        network::PhysicalNetwork&) const;
};

class PreparedControllerMutation {
public:
    bool deploy(const Solution&, ControllerMutationOptions = {});
    bool release(const Solution&, ControllerMutationOptions = {});
    void rollback(const Solution&, ControllerMutationOptions = {});
    void begin_transaction();
    void commit_transaction() noexcept;
    void rollback_transaction();
    bool transaction_active() const noexcept;
};

class PreparedController {
public:
    bool deploy_with_node_slots(
        const NodeSlots&, Solution&, DeployWithNodeSlotsOptions = {});
    PlaceAndRouteResult place_and_route(
        Vertex virtual_node, Vertex physical_node, Solution&,
        PlaceAndRouteOptions = {});
    bool undo_place_and_route(Vertex virtual_node, Solution&);
    bool deploy(Solution&, ControllerMutationOptions = {});
    bool release(const Solution&, ControllerMutationOptions = {});
    bool undo_deploy(const Solution&, ControllerMutationOptions = {});
};
}
```

Prepared objects are non-owning and must not outlive or move either network.
They do not retain the owning Controller. A prepared object owns the required
prepared dependency copies, so dynamic names are never rebound during
lifecycle calls. Network
mutation may not race with a lifecycle call. Independent prepared controllers
over independent networks may run concurrently.

`PreparedControllerMutation` prepares only ResourceUpdator plus numeric
resource/value IDs. Its exact checkpoint stores fixed target enum, node/edge
ID, `AttrId`, and original `AttrValue`; rollback does not depend on inverse
arithmetic. The System ownership contract is in
`transactional_solver_system.md`.

## Exact lifecycle semantics

- Safe placement writes the node single-step offsets immediately. Placement
  failure then sets `place_result=false` and `result=false`, without routing or
  rollback. Placed neighbors are examined in Graph adjacency order. A virtual
  link is skipped when either orientation already exists in `link_paths`.
- Incident links route sequentially. Failure retains the new node and all
  earlier successful routes, pools offsets through the failing route, updates
  the single-step maximum, sets `route_result=false` and `result=false`, and
  stops. Success never forces pre-existing result flags back to true.
- Unsafe single-step mode ignores collaborator result booleans, still pools
  and records offsets, and returns success unless a typed dependency exception
  occurs. Unsupported shortest modes fail at their completed dependency
  boundary after any earlier observable mutation.
- Link/path step offsets max-pool independently by numeric constraint ID. Both
  pooled values are completed before either is written. With no incident route,
  direct zero placeholders are copied. Only hard-node IDs participate in the
  node level and only hard-link IDs participate in link/path levels. All-negative
  hard lanes retain their negative maximum; an empty hard set yields zero.
- Undo verifies the placement before mutation, restores the node first, then
  snapshots and restores incident link paths in insertion order. Results from
  individual undo calls are ignored as in Python; exceptions retain partial
  restoration.
- Deploy returns false without mutation when `solution.result` is false. It
  subtracts `node_slots_info` in insertion order, then `link_paths_info` in
  insertion order. Release also returns false for an unsuccessful Solution;
  otherwise it adds node resources in `node_slots` order and link resources in
  `link_paths`/physical-path order. Neither operation auto-rolls back.
- `undo_deploy` calls release, constructs the same temporary reset candidate
  needed to retain Python's post-release error point, and leaves the caller's
  Solution unchanged.
- `deploy_with_node_slots` is the safe-only native counterpart of Python
  `_safely_deploy_with_node_slots`. Cardinality is checked before the complete
  value scan for `-1`; either failure changes only `place_result/result` to
  false. Valid entries retain `NodeSlots` insertion order. NodeMapper runs
  first with `l2s2`, `reusable=false`, `inplace=true`; only complete placement
  reaches LinkMapper, which runs `inplace=true` with typed shortest/k/worker
  options. Placement and route failures retain earlier partial mutations and
  flags; success sets only `result=true`. The broken unsafe branch remains
  unsupported.

## Threading contract

Worker zero/one is canonical sequential execution. Wider caller widths may use
the completed updater batch only when all physical node targets, or normalized
undirected physical-link targets, are disjoint. Duplicate targets fall back to
sequential mutation. A disjoint batch error is atomic in the completed updater;
Controller then replays scalar requests in original order so the same earlier
partial mutations and first error become observable. Node phase always
completes before link phase. Place and incident-route commits remain sequential;
their configured worker widths are forwarded only to completed independent
topology/candidate checks. There is no host-derived worker policy.

The fixed-node-slot API forwards caller widths to those same completed mapper
boundaries. Its reusable key/value vectors are direct `Vertex` buffers owned by
the prepared controller; after capacity is established there is no per-call
assignment-vector allocation and no string lookup.

The accepted disjoint fast path binds each selected resource to its physical
graph `AttrId` during preparation. It preflights every numeric update in Python
order without mutation, retains direct `AttrValue*` targets only after proving
node/edge-ID disjointness, then commits contiguous target ranges through the
persistent deterministic executor at the caller width. Any duplicate target,
missing info/value, invalid numeric lane, unsafe
arithmetic, or invalid endpoint falls back before mutation to the canonical
completed ResourceUpdator path. Thus ordinary partial-error semantics remain
unchanged while successful independent commits avoid per-request AttrMap
copies and nested `vector<ResourceAmount>` allocations.

## Frozen acceptance

Cover no-neighbor, placement failure, adjacency order, already-routed links,
multi-route success and middle failure, safe/unsafe behavior, sparse and
overlapping numeric IDs, zero/negative/hard/soft pooling, undo partial order,
deploy/release no-op and partial failures, round-trip restoration,
`undo_deploy`, workers `0/1/2/8`, duplicate-target fallback, and concurrent
independent callers. Run one compact differential and one runtime benchmark
after correctness; freeze both immediately. Existing component benchmarks are
never rerun.

The gate passed: 10/10 exact Python cases; focused production unit coverage,
workers `0/1/2/8`, duplicate-target fallback, scalar replay after batch error,
and eight concurrent independent controllers; strict GCC 11; ASan/UBSan/leak
detection; hot-string review; and targeted CTest with frozen-foundation
integrity. The permanently frozen 32,768-mutation deploy/release benchmark
retained deployed/restored checksums `17514356897791579542` /
`8486823302284311477`. Python took 33.624408 ms; C++ workers `1/2/8` took
3.892177 / 6.484015 / 11.785875 ms, respectively 8.639x / 5.186x / 2.853x
faster. See `../results/controller_2026-07-29.md`.

The additive transaction path and default Main exact-output/runtime signal are
recorded separately in
`../results/system_transaction_integration_2026-07-30.md`; the frozen
Controller benchmark above was not rerun.

The 2026-07-31 fixed-node-slot unit gate covers success, incomplete and `-1`
assignments, ordered partial placement/route failure, insertion order, and
exact worker equality at `0/1/2/8`; it passes without reopening the frozen
benchmark.
