# Component API: `core.controller.NodeMapper`

State: **COMPLETE / FROZEN** on 2026-07-29. Do not rerun or update the
accepted benchmark.

Python oracle: `../virne/virne/core/controller/node_mapper.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`8AD2AE077E61732DCB77FFA08269AAB630BDB6DF29969C983D09A3565BA860F9`.
The completed Solution, ConstraintChecker, ResourceUpdator, BaseNetwork,
VirtualNetwork, PhysicalNetwork, and graph API notes were read first. The
Python leaf was opened because both native files were empty and no contract
existed. Solver, system, routing, and learning/ML remain out of scope.

## Fixed-field and ID rule

Matching method, placement/mapping options, results, errors, operations,
candidates, nodes, and worker count are direct fields or enums. Selected node
resources and constraints are numeric registry IDs. `prepare()` resolves each
genuinely dynamic resource name to the virtual graph's `AttrId` once; placement,
candidate, resource, violation, and undo loops retain only IDs, vertices,
numeric variants, byte masks, and direct typed Solution members. No fixed
field is stored in a string-keyed map and no hot loop resolves, hashes, stores,
or compares a string.

Hard-constraint membership is a direct byte mask indexed by `ConstraintId`.
Constraint and resource iteration order is retained in compact ID vectors;
duplicate resource IDs collapse at the cold boundary exactly as Python's
resource dictionary comprehension does. The completed checker/updator remain
the only constraint and physical-resource engines.

## Stable native API

```cpp
namespace virne::core::controller {

enum class NodeMatchingMethod : std::uint8_t { greedy, l2s2 };

struct NodeMapperSelection {
    std::vector<ConstraintId> node_constraints;
    std::vector<ResourceId> node_resources;
    std::vector<ConstraintId> hard_constraints;
};

struct NodePlacementOptions {
    bool allow_constraint_violation = false;
    bool record_constraint_violation = true;
};

struct NodeMappingOptions {
    bool reusable = false;
    bool inplace = true;
    NodeMatchingMethod method = NodeMatchingMethod::greedy;
    bool allow_constraint_violation = false;
    std::size_t candidate_workers = 1;
};

struct NodePlacementResult {
    bool placed;
    ConstraintCheckResult check;
};

class NodeMapper {
public:
    explicit NodeMapper(NodeMapperSelection);
    const NodeMapperSelection& selection() const noexcept;
    PreparedNodeMapper prepare(
        const network::VirtualNetwork&,
        network::PhysicalNetwork&) const;
};

class PreparedNodeMapper {
public:
    NodePlacementResult place(
        Vertex virtual_node, Vertex physical_node, Solution&,
        NodePlacementOptions = {});
    void record_place_constraint_violation(
        Vertex virtual_node, const SolutionAttributeValues&, Solution&) const;
    bool undo_place(Vertex virtual_node, Solution&);
    bool node_mapping(
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<Vertex>& physical_nodes,
        Solution&, NodeMappingOptions = {});
};
}
```

Prepared objects are non-owning. They must not outlive or move either network;
network mutation may not race with placement. Dependency exceptions retain
their completed typed forms. NodeMapper-specific invalid state uses
`NodeMapperException` with direct error/operation enums and optional numeric
context fields.

## Python parity and native safety boundaries

- Safe placement checks every selected node constraint. Infeasibility changes
  no resource or slot. Success subtracts selected resources in order, then
  writes `node_slots` and `node_slots_info`.
- Unsafe placement still computes offsets, subtracts with `safe=false`, writes
  the same typed slots, and returns literal success.
- Recording first stores raw offsets, then non-negative violations, then adds
  the maximum selected hard offset clipped at zero. It accumulates on repeated
  calls and does not reset earlier totals. An empty matching hard-offset set is
  a typed error after the first two table writes, preserving Python's mutation
  order around `max([])`.
- Undo reads the mapped physical node and its stored typed resources, adds them
  in original resource order, and only then erases both entries. An update
  failure therefore retains the placement, as in Python.
- Mapping accepts only the typed `greedy` and `l2s2` modes, clears only node
  slots/info, copies the candidate vector, and never resets existing violation
  totals. `l2s2` fails on the first infeasible candidate. `greedy` selects the
  first feasible candidate. Failed mapping records the last reachable offsets
  and sets `place_result=false` and `result=false`; successful mapping leaves
  those pre-existing flags unchanged.
- `inplace=false` uses the completed `PhysicalNetwork::clone()` but still writes
  the caller's Solution, matching Python. Allow-violation mapping remains a
  typed unsupported operation and fails before clearing Solution state.
- Python's unbound-local failures for an empty/exhausted candidate list and its
  final assertion for an incomplete virtual-node list become deterministic
  typed native errors. Arbitrary mappings, custom numeric classes, monkey
  patches, and unbounded Python node/integer values remain language boundaries.

For valid typed input, wider `candidate_workers` may evaluate independent
physical candidates for one virtual node concurrently. Results and exceptions
are inspected in original candidate order, so a later exception is ignored
when an earlier feasible candidate would have stopped Python. Resource commit,
candidate removal, and virtual-node order remain sequential. Worker zero/one
is the canonical sequential path; wider values are explicit caller config and
are capped by candidate count, never chosen from the host.

## Frozen acceptance

The gate passed 12 exact shared Python cases and the native unit groups for
safe/unsafe placement, duplicate resources, violation mutation order, undo,
greedy/l2s2, reusable/clone behavior, empty and incomplete candidates, later
error suppression, workers `0/1/2/8`, and concurrent independent callers.
Production, unit, harness, and benchmark compile with strict GCC 11 warnings;
ASan, UBSan, leak detection, targeted CTest, frozen-foundation integrity, and
the hot-string audit all pass.

The permanently frozen greedy benchmark maps 32 virtual nodes across 2,080
physical candidates and performs at least 65,568 ordered candidate checks. It
retained checksum `15604526718891224062`. Python took `61.280082 ms`; C++ at
configured candidate workers `1/2/8` took `3.289713 / 16.008148 / 19.886030
ms`, respectively `18.628x / 3.828x / 3.082x` faster. The case favors worker
1; worker width remains caller configuration and no host-derived policy is
embedded. See `porting/results/node_mapper_2026-07-29.md`.

## Integration worker correction (2026-07-30)

The public API is unchanged. Candidate checks no longer create threads per
virtual node. Lists below 128 entries are canonical sequential. A larger search
probes up to eight leading candidates sequentially, then checks ordered windows
through the persistent executor. Each window is consumed in input order, so the
first feasible candidate/error remains exact and later windows stop after
success. Width `0/1` remains sequential and no host width is selected. Focused
equality/error/concurrent-caller units pass; the frozen benchmark was not rerun.
