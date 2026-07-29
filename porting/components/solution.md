# Component API: `core.solution`

State: **COMPLETE / FROZEN** on 2026-07-29.

Python oracle: `../virne/virne/core/solution.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`EC5D64EF6695350F718A818461FBF60934B5077D8ADD39BB7EE8BE18CFE1E78B`.
The completed `ClassDict`, `VirtualNetwork`, attribute, and graph documents
were read first. The oracle leaf was opened only because no Solution API note
or implementation existed.

## Native schema

`Solution` is a fixed model, not a `ClassDict` clone. IDs, request metadata,
result flags, counters, cost/revenue/demand metrics, violations, description,
actions, and reward are direct typed fields. The Python typo
`v_net_num_egdes` is emitted only by the compatibility serializer; native code
uses `v_net_num_edges`.

No fixed field is stored behind a string-keyed map, including at cold
boundaries. `description` is a string value, not a schema key. Arbitrary
Python-added fields are outside the typed native contract; the three known
later fields from Counter/Controller are direct optionals and retain Python's
reset-presence behavior.

The four ordered Python mappings use typed containers:

- `node_slots`: numeric virtual node -> physical node;
- `link_paths`: typed virtual endpoint pair -> ordered physical endpoint path;
- `node_slots_info` and `link_paths_info`: the same typed numeric keys with
  resource values indexed directly by completed attribute registry ID;
- constraint levels are a direct `node_level`/`link_level`/`path_level`
  struct. Dynamic attribute names resolve once in the future checker boundary,
  after which offsets use numeric IDs/direct slots.

Each ordered container keeps insertion order and offers a compact object-local
entry ID. Resolve a numeric key once when repeated access is needed; stable
phases then access entries directly by ID. Structural erase/clear invalidates
those IDs. No solution hot loop accepts a field-name string.

## Stable API

```cpp
struct SolutionMetadata {
    std::int64_t v_net_id;
    double v_net_lifetime;
    double v_net_arrival_time;
    std::size_t v_net_num_nodes;
    std::size_t v_net_num_edges;
};

class Solution {
public:
    explicit Solution(SolutionMetadata);
    static Solution from_v_net(const virne::network::VirtualNetwork&);
    static std::vector<Solution> from_metadata_batch(
        const std::vector<SolutionMetadata>&, std::size_t workers = 1);

    void reset();
    bool is_feasible() const noexcept;
    std::string repr() const;

    // Direct fixed fields and typed ordered mapping/constraint members.
};
```

Construction copies the five network facts, then calls `reset`. Reset retains
those facts and restores every other Python-initialized field exactly,
including `place_result=true`, `route_result=true`, empty ordered containers,
and both single-step extrema at negative infinity. Feasibility is exactly
`bool(result) && total_hard_constraint_violation <= 0`.

`from_v_net` requires the completed request ID, lifetime, and arrival-time
fields. A missing field is a typed construction error, corresponding to the
Python attribute failure. Arbitrary reflection, method shadowing, and updates
that replace a fixed field with an unrelated Python type remain explicit
language-boundary cases; native callers use typed direct assignment.

## Threading and frozen acceptance

One solution is sequential. `from_metadata_batch` and `reset_batch` fill or
reset independent
pre-sized slots at the caller width; zero/one is sequential, wider values are
bounded by item count, retain input/error order, and never auto-tune from the
host.

- Unit, concurrent callers, direct compact-ID access, missing-field order,
  worker `0/1/2/8`, strict GCC warnings, ASan, and UBSan: **PASS**.
- Exact differential: **PASS**, 17 shared + 2 native + 4 recorded boundaries
  = 23 classified cases.
- Frozen lifecycle benchmark: 32,768 solutions, one warm-up, three samples,
  exact entry/byte/feasible/checksum gates. C++ speedups at caller workers
  `1/2/8` are `6.760x / 6.005x / 6.419x`.

See [`solution_2026-07-29.md`](../results/solution_2026-07-29.md). The accepted
benchmark and all dependency benchmarks must not be rerun or updated.
