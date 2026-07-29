# Component API: `core.Counter`

State: **COMPLETE / FROZEN** on 2026-07-29.

Python oracle: `../virne/virne/core/counter.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`574745F86E99CD656CB0165330E3196B4F5EBF4EAA0687B076B8D9602DB4D637`,
12,463 bytes and 236 physical lines. The file was clean at that commit during
this audit.

The completed Solution, BaseNetwork, VirtualNetwork, PhysicalNetwork,
AttributeFactory, AttributeBenchmarkManager, VirtualNetworkEvent, and frozen
CSV contracts were read before opening this previously undocumented leaf.
Completed dependency implementation was not opened. `virne.utils.stats` is
not imported or called by Counter and is not a dependency.

The frozen `virne-python-oracle:py310-nonml` image contains NumPy 2.2.6 but no
Pandas, so importing the complete source there currently fails before the
class is defined. Core differential tests must isolate the exact `Counter`
class AST and inject only its typed dependencies. Record-summary parity needs
a separately pinned Pandas oracle; it must not mutate the frozen image or add
Pandas to native production.

## Scope and dependency boundary

This leaf owns:

- resource sums for a prepared BaseNetwork;
- partial and complete Solution revenue/cost mutation;
- virtual-link deployment cost from typed `link_paths_info`;
- typed aggregation of recorder rows; and
- the observable legacy `summary_csv` failure.

It reuses the completed network registries, graph-local attribute bindings,
Solution schema, VirtualEventType, and frozen CSV reader. It does not own
attribute construction, benchmark/max reduction, path construction, resource
mutation, recording, logging, controller orchestration, environments, solver,
system, plotting, or learning/ML.

`AttributeBenchmarkManager` is deliberately **not** a Counter runtime
dependency. Python Counter performs a NumPy sum over resource values; the
manager converts to float32 and computes maxima, which is observably different.
Likewise, VirtualNetwork's `noexcept` demand helpers convert data errors to
zero and therefore cannot implement Counter's throwing calculation methods.

The Python constructor accepts node/link/graph settings and config. It creates
node/link definitions, retains every created object, then selects definitions
whose exact type is `resource`. `graph_attrs_setting` is never read. `config`
is stored but never read by any Counter method. Native code must use the live
completed network registries instead of constructing a second attribute
object hierarchy or retaining an unused dynamic config tree.

## Fixed-field and ID rule

All metric names, node/link enable flags, event kind, result flags, summary
columns, operations, errors, worker widths, and returned fields are direct
members or enums. Node and link resource selections are registry-local
`AttributeRegistryId` vectors. Preparation validates the owning registry and
resolves each selected resource once to its graph-local `AttrId`.

Prepared resource, node, link, route, and summary loops use only direct
vertices, endpoint pairs, compact registry IDs, graph-local IDs, typed numeric
lanes, pre-resolved CSV column indexes, and direct Solution fields. No such
loop hashes, compares, retrieves, or stores a fixed field name. A genuinely
dynamic attribute or CSV column name may be bound once at the cold boundary;
all repeated row access is by numeric ID/index.

Node and link registry IDs are separate domains. Equal numeric IDs from two
networks or from node/link registries are not interchangeable. A prepared
counter is non-owning and cannot outlive or race mutation of its network.

## Proposed typed C++ API

All names below belong to `virne::core` unless qualified.

```cpp
using CounterResourceId =
    network::attribute::AttributeRegistryId;
using CounterNumber = std::variant<std::int64_t, double>;

struct CounterSelection {
    // nullopt selects every resource definition in registry order.
    // A present empty vector remains observably empty and later calculation
    // follows the Python empty-selection failure.
    std::optional<std::vector<CounterResourceId>> node_resources;
    std::optional<std::vector<CounterResourceId>> link_resources;
};

struct CounterOptions {
    std::size_t workers = 1U;
};

enum class CounterErrorCode : std::uint8_t {
    invalid_node_resource_selection,
    invalid_link_resource_selection,
    attribute_registry_mismatch,
    graph_binding_mismatch,
    virtual_network_required,
    missing_node_resource_value,
    missing_link_resource_value,
    non_numeric_resource_value,
    empty_node_resource_selection,
    empty_link_resource_selection,
    numeric_overflow,
    invalid_solution_node,
    invalid_solution_link,
    missing_route_info,
    missing_virtual_lifetime,
    empty_records,
    invalid_record_value,
    missing_record_column,
    legacy_summary_csv_binding,
};

enum class CounterOperation : std::uint8_t {
    prepare,
    sum_node_resources,
    sum_link_resources,
    count_link_cost,
    count_partial_solution,
    count_solution,
    summarize_records,
    summarize_csv,
};

class CounterException : public std::runtime_error {
public:
    CounterException(
        CounterErrorCode,
        CounterOperation,
        std::string message,
        std::optional<CounterResourceId> resource_id = std::nullopt,
        std::optional<SolutionNodeId> virtual_node = std::nullopt,
        std::optional<SolutionLink> virtual_link = std::nullopt,
        std::optional<SolutionLink> physical_link = std::nullopt,
        std::optional<std::size_t> row_index = std::nullopt);

    CounterErrorCode code() const noexcept;
    CounterOperation operation() const noexcept;
    const std::optional<CounterResourceId>& resource_id() const noexcept;
    const std::optional<SolutionNodeId>& virtual_node() const noexcept;
    const std::optional<SolutionLink>& virtual_link() const noexcept;
    const std::optional<SolutionLink>& physical_link() const noexcept;
    const std::optional<std::size_t>& row_index() const noexcept;
};

class PreparedCounter;

class Counter {
public:
    explicit Counter(CounterSelection selection = {});
    const CounterSelection& selection() const noexcept;

    // The BaseNetwork overload prepares generic resource sums. The
    // VirtualNetwork overload also enables count_* Solution methods without
    // RTTI and preserves late lifetime access.
    PreparedCounter prepare(
        const network::BaseNetwork& network) const;
    PreparedCounter prepare(
        const network::VirtualNetwork& virtual_network) const;
};

class PreparedCounter {
public:
    CounterNumber calculate_sum_network_resource(
        bool node = true,
        bool link = true,
        CounterOptions options = {}) const;
    CounterNumber calculate_sum_node_resource(
        CounterOptions options = {}) const;
    CounterNumber calculate_sum_link_resource(
        CounterOptions options = {}) const;

    CounterNumber calculate_v_net_link_cost(
        const Solution& solution) const;
    CounterNumber calculate_v_net_cost(
        const Solution& solution,
        CounterOptions options = {}) const;
    CounterNumber calculate_v_net_revenue(
        CounterOptions options = {}) const;

    void count_partial_solution(
        Solution& solution,
        CounterOptions options = {}) const;
    void count_solution(
        Solution& solution,
        CounterOptions options = {}) const;
};
```

`calculate_v_net_revenue` intentionally has no Solution argument because the
Python optional argument is ignored. `count_partial_solution` and
`count_solution` mutate the direct Solution and return `void`; Python's final
`to_dict()` allocation is a representation boundary already excluded by the
typed Solution contract. Differential tests serialize the direct fields after
the call.

Calling either Solution mutation method on a counter prepared through the
BaseNetwork overload raises `virtual_network_required` before mutation. The
VirtualNetwork overload
stores a direct non-owning VirtualNetwork pointer, but it must not eagerly
read lifetime: Python reads lifetime only after the main mutation sequence.

## Exact resource and numeric semantics

Resource definition order is Counter selection order; values within a row are
graph node/edge order. Python calls `np.array(rows).sum()` without an explicit
axis or dtype. The compatible typed domain therefore retains:

- bool-only and bool/int arrays promote to NumPy's integer sum lane;
- int64 arrays sum with NumPy 2.2.6 int64 wrap semantics, implemented with
  explicit unsigned arithmetic rather than C++ signed-overflow UB;
- a present double lane promotes the rectangular array to binary64 and uses
  the pinned NumPy flattened reduction order;
- a ragged, missing, or nonnumeric row fails at its original gather/array/sum
  stage; it is not converted to VirtualNetwork's fallback zero;
- node calculation finishes before link calculation; a disabled side is the
  literal integer zero and its data is not touched; and
- both disabled returns integer zero.

The eventual implementation must either share a documented exact NumPy-sum
primitive with VirtualNetwork or add a focused Counter-local primitive. It
must not use AttributeBenchmarkManager or an unordered/parallel reduction.

Manual Solution costs start at Python integer zero and add values strictly in
virtual-link, physical-link, then selected-resource order. bool/int/double
promotion follows Python scalar addition. Native supported totals are int64 or
binary64; Python arbitrary-precision aggregate integers beyond int64 are a
recorded language boundary. Native code detects overflow rather than invoking
undefined behavior.

Python `/` normalization produces binary64. Ratio fields alone guard an exact
zero cost and return literal zero when it is zero. No tolerance or epsilon is
introduced. NaN compares unequal to zero and therefore propagates through the
division branch like Python.

## `count_partial_solution` mutation contract

All node/link/path scans complete before the first Solution write. Scans use
Solution insertion order:

1. For each placed virtual node, sum all selected virtual node resources.
2. For each routed virtual link, add its selected virtual link-resource demand.
3. For each nonempty physical path, add every stored `link_paths_info` resource
   in physical-link then selected-resource order. An empty path has revenue but
   zero path cost.

After successful scans, fields are written in this exact order:

1. `v_net_node_revenue = raw_node_revenue / num_node_resources`;
2. `v_net_link_revenue = raw_link_revenue`;
3. `v_net_revenue = raw_node_revenue + raw_link_revenue`;
4. `v_net_link_cost`;
5. `v_net_path_cost = link_cost - link_revenue`;
6. `v_net_node_cost = raw_node_revenue / num_node_resources`;
7. `v_net_cost = node_cost + link_cost`;
8. `v_net_r2c_ratio`, guarded only by exact zero cost.

The total revenue deliberately uses the **unnormalized** node revenue even
though the node-revenue field is normalized. This inconsistency is observable
and retained. With zero selected node resources, Python evaluates the first
division before assignment and raises without writing any of these eight
fields. Lookup/numeric errors during scans likewise leave all Counter metric
fields unchanged.

Rerunning the method recomputes and overwrites these fields from the current
mapping/resource data; it does not clear any unrelated Solution field.

## `count_solution` mutation contract

The method writes in the following order:

1. `num_placed_nodes = node_slots.size()`;
2. `num_routed_links = link_paths.size()`;
3. normalized `v_net_node_demand`;
4. `v_net_link_demand`;
5. `v_net_demand = v_net_node_demand + old v_net_demand`.

Step five is the original Python bug: it does **not** add
`v_net_link_demand`. The previous direct Solution value is read after the first
four writes and retained exactly.

When `solution.result` is true, mutation continues in this order:

1. set `place_result=true`, `route_result=true`, `early_rejection=false`;
2. copy node/link demand to node/link revenue;
3. copy node revenue to node cost;
4. calculate and write link cost from `link_paths_info`;
5. write path cost, total revenue, total cost, then guarded r2c ratio.

Total cost is `v_net_revenue + v_net_path_cost`, which algebraically equals
node revenue plus routed link cost for ordinary finite values.

When `solution.result` is false, it writes zero to node revenue, link revenue,
total revenue, path cost, total cost, and r2c ratio in that order. It does not
change `place_result`, `route_result`, `early_rejection`, `v_net_node_cost`, or
`v_net_link_cost`; stale values in those fields remain observable.

After either branch, Python reads `v_net.lifetime` and writes, in order:

1. `v_net_time_revenue = v_net_revenue * lifetime`;
2. `v_net_time_cost = v_net_cost * lifetime`;
3. `v_net_time_rc_ratio = v_net_r2c_ratio * lifetime`.

A missing native optional lifetime becomes a typed error at that late stage;
all preceding mutations remain. Errors in successful link-cost calculation
also retain every earlier counter, demand, flag, revenue, and node-cost write.
No rollback is added.

## Other calculation methods

- `calculate_sum_network_resource` evaluates node then link according to its
  direct flags and returns their scalar sum.
- `calculate_sum_node_resource` and `calculate_sum_link_resource` expose their
  respective NumPy-compatible scalar result.
- `calculate_v_net_cost` evaluates node-resource sum first, then routed link
  cost, and adds them.
- `calculate_v_net_revenue` returns the node-plus-link network resource sum;
  its Python Solution argument has no effect.
- `calculate_v_net_link_cost` does not read its Python `v_net` argument. It
  traverses only Solution paths/info and selected link resources. A missing
  info entry fails at its exact path position after the local accumulator has
  advanced, but no Solution field is mutated by this calculation method.

## Typed record and summary API

Pandas dictionaries/DataFrames are a cold compatibility boundary. Native
recorder consumers use direct rows and results:

```cpp
struct CounterRecord {
    std::int64_t success_count = 0;
    std::int64_t virtual_network_count = 0;
    network::VirtualEventType event_type =
        network::VirtualEventType::leave;
    double v_net_r2c_ratio = 0.0;
    double total_time_revenue = 0.0;
    double total_time_cost = 0.0;
    double virtual_network_arrival_time = 0.0;
    bool early_rejection = false;
    bool place_result = true;
    bool route_result = true;
    double total_cost = 0.0;
    double total_revenue = 0.0;
    double physical_available_resource = 0.0;
    double physical_node_available_resource = 0.0;
    double physical_link_available_resource = 0.0;
    std::int64_t inservice_count = 0;
    double hard_constraint_violation = 0.0;
    double max_single_step_hard_constraint_violation = 0.0;
    std::optional<double> reward;
};

struct CounterRecords {
    std::vector<CounterRecord> rows;
    // Distinguishes an absent DataFrame column from a present column whose
    // selected rows are all missing/NaN.
    bool reward_column_present = false;
};

struct CounterSummary {
    double acceptance_rate = 0.0;
    double average_r2c_ratio = 0.0;
    double long_term_time_r2c_ratio = 0.0;
    double long_term_average_time_revenue = 0.0;
    std::int64_t success_count = 0;
    std::size_t early_rejection_count = 0U;
    std::size_t place_failure_count = 0U;
    std::size_t route_failure_count = 0U;
    double total_cost = 0.0;
    double total_revenue = 0.0;
    double total_time_revenue = 0.0;
    double total_time_cost = 0.0;
    double long_term_r2c_ratio = 0.0;
    double total_simulation_time = 0.0;
    double long_term_average_revenue = 0.0;
    double long_term_average_cost = 0.0;
    double minimum_physical_available_resource = 0.0;
    double minimum_physical_node_available_resource = 0.0;
    double minimum_physical_link_available_resource = 0.0;
    std::int64_t maximum_inservice_count = 0;
    double total_violation = 0.0;
    double total_max_single_step_violation = 0.0;
    double average_reward = 0.0;
};

CounterSummary summary_records(const CounterRecords& records);
CounterSummary summary_csv(const std::string& path);
```

`summary_records` requires at least one row. It uses the final positional row,
not the last arrival row, for success/vnet counts, totals, and simulation time.
Only exact `VirtualEventType::arrival` rows participate in average r2c,
average reward, and early/place/route failure counts. Resource minima,
inservice maximum, and both violation sums use all rows.

The output field order above matches Python dictionary insertion order. Ratios
have no zero guard; use IEEE binary64 division so zero denominators produce the
corresponding infinity/NaN rather than C++ integer division. Mean/min/max/sum
retain Pandas default skip-NaN behavior: an empty/all-NaN mean or min/max is
NaN, while an all-NaN sum is zero. A missing reward column produces literal
zero; a present column with no finite arrival values produces NaN.

`summary_csv` first executes `pd.read_csv(path, header=0)`, then calls the
instance method as `cls.summary_records(records)`. Because `summary_records`
is neither static nor a classmethod, every successful read reaches a missing
`records` binding `TypeError`. Native `summary_csv` preserves this as
`legacy_summary_csv_binding` after the frozen CSV reader successfully parses
the supported RFC4180 domain. File/open/parse errors occur first. Arbitrary
Pandas dialect, dtype inference, NA token, encoding, and parser behavior remain
a documented compatibility boundary. A future functional CSV summary must use
a differently named native extension and must not silently change this method.

## Error and partial-state rules

Counter-specific invalid state uses `CounterException`. Completed dependency
exceptions propagate unchanged unless Counter must attach a Solution key or
summary row index; any wrapper retains the dependency as a nested cause.

Preparation validates selected IDs in vector order and verifies exact resource
kind and registry/graph identity. It does not scan every graph value eagerly,
because Python missing/nonnumeric data fails at its natural calculation stage.
Raw registry IDs are intentionally relative to the network passed to
`prepare()`: a foreign numeric ID that aliases an in-range resource ID is not
distinguishable and remains an ID-domain caller boundary. Prepared identity
checks still detect later registry/graph replacement without string work.
Solution endpoint conversion is checked before graph access. Ordered map entry
IDs may be cached only while no structural Solution erase/clear occurs.

No method is transactional. Earlier direct Solution writes remain after a
later lifetime, path-info, numeric, or dependency error exactly as described
above. Pure calculation and summary methods build local results and publish
only on success.

## Parallel and ownership contract

- Worker count is direct caller configuration. Zero/one is sequential;
  explicit wider values are bounded by independent item count. No host-derived
  worker policy is embedded.
- Network row gathers may use completed deterministic workers after all
  selection/binding validation. The final NumPy-compatible numeric reduction
  remains in its exact deterministic order.
- Node-slot, link-path, path-info, Solution mutation, manual link-cost, and
  record-summary evaluation remain sequential because order, partial state,
  and first error are observable.
- An optional future batch API may process independent network/Solution pairs
  into pre-sized slots and surface the lowest input error. It must not mutate
  one shared Solution or network concurrently.
- Concurrent read-only use of one prepared immutable network with distinct
  Solutions is allowed. Mutation/read races on a Solution or mutation of the
  prepared network are unsupported.

## Explicit boundaries

- Pandas DataFrame identity, arbitrary dictionaries/columns/indexes, extension
  dtypes, object arrays, custom scalar protocols, and broad CSV inference;
- NumPy object/ragged arrays, arbitrary precision input integers, and numeric
  behavior outside bool/int64/binary64 resource lanes;
- arbitrary Python attribute objects, monkey patches, reflection, truthiness,
  and `Solution.to_dict()` object identity;
- constructor `graph_attrs_setting` and config observability beyond the fact
  that Python stores config; native fixed runtime state does not retain them;
- sparse/non-integral native vertices outside the completed graph/Solution
  representation; and
- Controller, Recorder, Logger, Environment, solver, system, plotting,
  OR-Tools, Torch, CUDA, and all learning/ML code.

## Completed implementation gate

The implementation gate is complete. The exact AST-isolated differential
passed all 20 shared cases against Python 3.10/NumPy 2.2.6. GCC 11 strict
compilation, the focused unit suite, ASan/UBSan/leak checks, hot-loop review,
and targeted frozen-integrity CTest all passed. The single accepted canonical
benchmark retained checksum `3910809078534895256` and measured C++ speedups
of 3.614x, 4.835x, and 4.372x at caller workers `1/2/8`. Its protocol and raw
timings are frozen in `porting/results/counter_benchmark_2026-07-29.json`; do
not rerun or edit that benchmark. See
`porting/results/counter_2026-07-29.md` for the compact handoff.

Completed coverage includes:

- default/all/explicit/empty resource selection, node/link registry separation,
  independent graph IDs, duplicates, wrong kind, and moved-network lifetime;
- bool/int64/double sums, mixed promotion, signed zero, subnormals, infinities,
  NaNs, int64 wrap, missing/ragged/nonnumeric rows, flags for node/link, and
  workers `0/1/2/8` with identical output/error order;
- partial count with empty paths, multiple resources, normalization mismatch,
  rerun overwrite, zero node-resource failure, and every lookup failure stage;
- complete count success/failure, old-demand bug, stale failure costs, exact
  flag mutation, missing lifetime after prior writes, link-info error partial
  state, zero-cost ratio, and repeated calls;
- every calculation helper and proof that revenue ignores Solution while link
  cost ignores the VirtualNetwork argument;
- record summary ordinary/empty/no-arrival/zero-denominator/NaN cases, final-row
  selection, arrival-only filters, reward absent/present/all-missing, output
  order, and the read-before-binding-error `summary_csv` behavior;
- concurrent independent callers, strict warning build, ASan/UBSan/leaks,
  exact direct-source differential, frozen dependency integrity, and a hot-loop
  audit proving no string lookup after preparation.

Only after exact compatibility passes, add one compact benchmark with one
warm-up and three samples at caller workers `1/2/8`. Fixture construction,
preparation, process startup, CSV parsing, serialization, and checksum work
remain outside timing. Gate exact numeric type/bits, Solution mutation order,
summary fields, output bytes, and final checksum before timing. Once accepted,
freeze it and do not rerun dependency benchmarks.
