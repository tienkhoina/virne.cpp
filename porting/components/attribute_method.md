# Component API: `network.attribute.attribute_method`

State: **COMPLETE for the independent typed-policy leaf** on 2026-07-28.

This document is the source-of-truth contract for the first attribute-model
leaf.  It freezes the behavior observed in the original Python mixins and the
native representation/performance boundary before production code is written.
No C++ implementation, CMake target, unit, oracle, or benchmark exists yet.

## Documentation-first rule

Before the one permitted source read, the audit read:

- `PORTING_STATUS.md`;
- `porting/NON_ML_COMPONENT_MAP.md`;
- `porting/PERFORMANCE_CONTRACT.md`;
- `porting/FROZEN_COMPONENTS.md`;
- the completed dataset RNG/core, topology generator, topological metric, and
  `utils.network` component API notes; and
- the frozen `graph/API.md` contract, especially its registry ownership and
  hot-loop `AttrId` rules.

There was no existing component note for this leaf, so reading the original
Python file and focused symbol callsites was necessary and permitted. Future
work must begin here. Do not reopen the implementation merely to rediscover
behavior. A focused source/header read is allowed only for an exact
differential mismatch or an API point explicitly listed as unresolved below;
record every new fact here in the same change.

## Source identity and scope

- Original source:
  `../virne/virne/network/attribute/attribute_method.py`.
- Original repository commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- The original checkout was clean during this audit.
- Exact source SHA-256:
  `e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f`.
- Exact source size: 5,543 bytes.
- Exact text extent: 118 LF-terminated physical lines.

The file defines four cooperative mixins. It does not define `__all__`.
`numpy`, `networkx`, and several `typing` names therefore remain visible in
the Python module namespace even though the method bodies use none of NumPy or
NetworkX directly. The package imports the mixins indirectly through wildcard
imports from node/link/graph attribute modules. C++ must export only the typed
attribute-method surface; Python namespace pollution is not an API to copy.

## Complete Python API inventory

| Class | Public/protected method | Ordinary result |
|---|---|---|
| `ResourceAttributeMethod` | `update(v, p, method='+', safe=True)` | mutates `p`; returns literal `True` |
| `ResourceAttributeMethod` | `generate_data(network)` | exact result of delegated `_generate_data(network)` |
| `ExtremaAttributeMethod` | `update(v, p, method='+', safe=True)` | literal `True`; no work |
| `ExtremaAttributeMethod` | `check(v_net, p_net, v_node_id, p_node_id, method='le')` | literal `True`; no work |
| `ExtremaAttributeMethod` | `generate_data(network)` | exact result of originator attribute `get_data(network)` |
| `InformationAttributeMethod` | cooperative `__init__(*args, **kwargs)` | initializes the next MRO class, then forces `is_constraint=False` |
| `ConstraintAttributeMethod` | cooperative `__init__(*args, **kwargs)` | initializes the next MRO class, then forces constraint fields |
| `ConstraintAttributeMethod` | `check_constraint_satisfiability(v, p, method='le')` | always raises `NotImplementedError` |
| `ConstraintAttributeMethod` | `_calculate_satisfiability_values(v_value, p_value, method='le')` | two-tuple `(flag, offset)` |

There are no module-level functions and no standalone concrete attribute
objects. The type hints are descriptive only: Python performs no argument type
checks and accepts arbitrary objects implementing the operations it invokes.

## Runtime dependencies and call graph

The Python module unconditionally imports NumPy and NetworkX at module load,
although all method bodies are built-in Python protocol code. `BaseNetwork` is
inside `TYPE_CHECKING` and is not imported at runtime. There is no Torch,
OmegaConf, solver, system, RNG, or filesystem dependency in this leaf.

The effective call graph is:

```text
ResourceAttributeMethod.update
  -> getattr(type/name)
  -> mapping reads + arithmetic/comparison + mapping write

ResourceAttributeMethod.generate_data
  -> getattr(generative/_generate_data)
  -> callable check
  -> concrete/BaseAttribute._generate_data(network)
     (future integration reuses completed dataset RNG)

ExtremaAttributeMethod.generate_data
  -> getattr(owner/originator)
  -> network.node_attrs or network.link_attrs
  -> originator_attribute.get_data(network)

InformationAttributeMethod.__init__
  -> next cooperative-MRO __init__

ConstraintAttributeMethod.__init__
  -> next cooperative-MRO __init__
  -> restriction validation

ConstraintAttributeMethod._calculate_satisfiability_values
  -> comparison
  -> subtraction / abs
  -> hard/soft restriction dispatch
```

The recommended isolated C++ policy target is standard-library-only. The
future BaseAttribute adapter will reuse the completed dataset RNG API; future
node/link/graph adapters will link the frozen graph target. This leaf must not
link either merely because Python imports broad packages.

## Exact observable behavior

Unless stated otherwise, Python exceptions and user-defined operator/mapping
side effects propagate immediately. No method rolls back an earlier side
effect. Exact Python exception messages below are part of the oracle record;
native C++ should additionally expose stable typed error code and operation
stage.

### `ResourceAttributeMethod.update`

1. Read `self.type` with `getattr(..., None)`, then `self.name` the same way.
2. Unless type equals the exact string `"resource"` and name is not `None`,
   raise:

   ```text
   TypeError: ResourceAttributeMethod requires 'type' == 'resource' and 'name' attribute in the main class.
   ```

   An empty name is accepted. Both attributes are read before validation.
3. Accept exactly `+`, `-`, `add`, and `sub`. Any other value raises before a
   mapping lookup or mutation:

   ```text
   NotImplementedError: Update method '<method>' is not supported.
   ```

4. `+`/`add` executes `p[name] += v[name]`. `safe` is neither read nor coerced
   to truth in this branch.
5. `-`/`sub` first evaluates `safe`. If it is truthy, read `v[name]` then
   `p[name]` and compare `v > p`. When true, format the message by reading both
   entries a second time, then raise without performing the subtraction:

   ```text
   ValueError: <name>: (v = <v-value>) > (p = <p-value>)
   ```

6. Otherwise execute `p[name] -= v[name]` and return literal `True`.

For ordinary dictionaries, successful augmented assignment reads the physical
value, reads the virtual value, computes, then writes the physical value. The
safe comparison adds a prior virtual-then-physical read. Missing keys,
unhashable names, failed truth conversion, comparison/arithmetic errors, and a
mapping setter error propagate at their native stage. Arbitrary mappings and
user-defined arithmetic are Python-only protocol behavior, but access order is
locked by probe tests so a native representation decision is explicit.

The safe guard applies only to subtraction. It does not reject negative
virtual values, non-finite values, or an addition that exceeds any maximum.

### `ResourceAttributeMethod.generate_data`

1. Read `self.generative` with default `False` and evaluate its truth value.
2. If truthy, read `self._generate_data` with default `None`.
3. If that value is not callable, raise:

   ```text
   NotImplementedError: ResourceAttributeMethod requires '_generate_data' method in the main class for generative attributes.
   ```

4. Otherwise call it once with the exact `network` object and return its exact
   result without copying, coercion, or length validation.
5. If `generative` is falsey, do not inspect `_generate_data`; raise:

   ```text
   NotImplementedError: Non-generative resource attribute must implement generate_data.
   ```

Callable exceptions and truth/getattr side effects propagate. The method does
not validate resource type or name. In normal classes the delegate is
`BaseAttribute._generate_data`, which is outside this leaf and later reuses the
completed typed dataset RNG rather than reimplementing generation.

### `ExtremaAttributeMethod`

`update` and `check` return literal `True` immediately. They do not inspect
`self`, arguments, method, or safe, and they perform no mutation. These are
observable no-op stubs, not implemented extrema arithmetic.

`generate_data` behaves as follows:

1. Read `self.owner`, then `self.originator`, each with default `None`.
2. If either is `None`, raise:

   ```text
   AttributeError: ExtremaAttributeMethod requires 'owner' and 'originator' attributes in the main class.
   ```

3. Only owner equal to the exact string `"node"` selects
   `network.node_attrs[originator]`.
4. Every other non-`None` owner selects
   `network.link_attrs[originator]`. This includes `"link"`, `"graph"`, an
   empty string, and arbitrary objects not equal to `"node"`.
5. Call the selected attribute's `get_data(network)` once and return its exact
   object without copying.

The non-node fallback is a locked implementation behavior. In particular,
`GraphExtremaAttribute` declares owner `"graph"` and therefore looks in the
link-attribute registry. Missing registries/keys, an absent `get_data`, and
delegate failures propagate unchanged.

### `InformationAttributeMethod.__init__`

The initializer calls `super().__init__(*args, **kwargs)` first. Only after it
returns does it assign literal `False` to `self.is_constraint`, overwriting a
same-named value that the next MRO initializer may have accepted from kwargs.
If `super` fails, the final assignment does not occur. This cooperative-MRO
ordering is exercised by status and extrema concrete classes.

### `ConstraintAttributeMethod.__init__`

The initializer also calls the next MRO initializer first. It then:

1. assigns literal `True` to `self.is_constraint`;
2. reads only `kwargs['restriction']`, defaulting to `"hard"`;
3. stores that value as `self.constraint_restrictions`; and
4. accepts only values equal to `"hard"` or `"soft"`.

An invalid value raises after the cooperative initializer and both assignments:

```text
ValueError: constraint_restrictions must be 'hard' or 'soft', got <value>
```

A direct `constraint_restrictions=` keyword is not consumed by this mixin and
does not replace the default. Current concrete resource constructors normalize
their config and pass the selected value as `restriction=`.

### Constraint checking and calculation

The base `check_constraint_satisfiability` method always raises, without
inspecting its arguments:

```text
NotImplementedError: The attribute has not implemented the check_constraint_satisfiability method
```

`_calculate_satisfiability_values` first validates `method`. Accepted spellings
are exactly:

| Spellings | Operation | Offset |
|---|---|---|
| `>=`, `ge` | `v_value >= p_value` | `p_value - v_value` |
| `<=`, `le` | `v_value <= p_value` | `v_value - p_value` |
| `eq` | `v_value == p_value` | `abs(v_value - p_value)` |

The source contains a later `"=="` branch, but the initial validator rejects
`"=="`; it is unreachable. Every unsupported method raises before numeric
operations:

```text
NotImplementedError: Used method <method>
```

The comparison is evaluated before the offset expression. Despite the
docstring's `Violation = Max(0, Offset)` claim, the returned offset is the raw
signed expression above and is never clamped.

After computing both values, restriction dispatch occurs:

- `hard` returns the computed `(flag, offset)`;
- `soft` returns `(True, offset)` but still performs the comparison first;
- any other current value raises after all numeric work:

  ```text
  ValueError: Unknown constraint restriction: <value>
  ```

Python does not coerce the hard flag to `bool`. Overloaded comparisons can
therefore return non-boolean objects, including NumPy arrays. Ordinary Virne
values are scalar ints/floats/bools, for which the result flag is boolean.
Python integer offsets are arbitrary precision; bool subtraction produces an
integer; any floating operand follows Python/NumPy floating rules. NaN,
infinity, and signed zero are not special-cased.

## Focused callsite inventory

No original test directly names these mixins or the protected calculator.
Their behavior is exercised indirectly by later layers, so the new isolated
matrix is mandatory.

| Consumer | Use of this leaf |
|---|---|
| `node_attribute.py` | status uses Information; extrema uses Extrema+Information; resource and position use Resource+Constraint |
| `link_attribute.py` | status/extrema mirror node; resource uses update/calculation; latency uses generation delegation and calculation |
| `graph_attribute.py` | graph status/extrema/resource use the mixins; graph resource overrides `update` |
| `BaseNetwork.generate_attrs_data` | calls node/link `generate_data` for generative or extrema attributes, then validates length and stores it |
| `LinkResourceAttribute.update_path` | converts a path to links and calls `ResourceAttributeMethod.update` once per physical link, in path order |
| concrete node/link/graph resource checks | obtain named values, then call `_calculate_satisfiability_values` |
| `LinkLatencyAttribute` | defaults comparison to `ge` and sums a physical path before calculation |
| `ConstraintChecker` | calls concrete checks in attribute-list order and records raw offsets by dynamic name |
| `Controller` | partitions attributes by `is_constraint`, `checking_level`, and `constraint_restrictions` |

Important integration facts:

- generic node/link resource updates in `ResourceUpdator` bypass this mixin;
  only link path updates call `ResourceAttributeMethod.update` today;
- GraphResource's override does not use this mixin's safe guard;
- NodePosition inherits the base constraint method but does not override it;
- Information attributes do not receive `constraint_restrictions`, while the
  current Controller reads that field from all node/link attributes; this is a
  known higher-layer inconsistency, not a reason to invent a value here; and
- the current attribute factory does not register graph-owner classes, even
  though their public Python classes exist.

## Proposed typed C++ boundary

The C++ design should use composition/policy structs, not reproduce Python's
cooperative dynamic mixin MRO. The following names are proposed and remain
**non-stable until implementation/API review**:

```cpp
namespace virne::network::attribute {

using AttributeDefinitionId = std::uint32_t;

enum class AttributeOwner : std::uint8_t {
    node,
    link,
    graph,
};

enum class AttributeKind : std::uint8_t {
    resource,
    extrema,
    status,
    position,
    latency,
};

enum class ResourceUpdateOperation : std::uint8_t {
    add,
    subtract,
};

enum class ComparisonOperation : std::uint8_t {
    greater_equal,
    less_equal,
    equal,
};

enum class ConstraintRestriction : std::uint8_t {
    hard,
    soft,
};

enum class AttributeNumberKind : std::uint8_t {
    boolean,
    integer,
    floating,
};

using AttributeNumber = std::variant<bool, std::int64_t, double>;

struct ResourceMethodSpec {
    AttributeDefinitionId definition_id = 0;
    bool generative = false;
};

enum class ExtremaOriginRegistry : std::uint8_t {
    node,
    link,
};

struct ExtremaMethodSpec {
    AttributeOwner declared_owner = AttributeOwner::node;
    ExtremaOriginRegistry origin_registry = ExtremaOriginRegistry::node;
    AttributeDefinitionId originator_id = 0;
};

struct InformationMethodSpec {
    static constexpr bool is_constraint = false;
};

struct ConstraintMethodSpec {
    static constexpr bool is_constraint = true;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
};

struct SatisfiabilityResult {
    bool flag = false;
    AttributeNumber offset = std::int64_t{0};
};

enum class AttributeMethodErrorCode : std::uint8_t {
    invalid_resource_state,
    unsupported_update_operation,
    insufficient_resource,
    non_generative_resource,
    missing_generator,
    missing_extrema_field,
    missing_originator,
    unsupported_comparison,
    invalid_restriction,
    invalid_numeric_type,
    numeric_range,
};

enum class AttributeMethodOperation : std::uint8_t {
    resolve_update,
    update_resource,
    generate_resource,
    resolve_extrema,
    generate_extrema,
    initialize_constraint,
    calculate_satisfiability,
};

class AttributeMethodException : public std::runtime_error {
public:
    AttributeMethodException(AttributeMethodErrorCode,
                             AttributeMethodOperation,
                             std::string message);
    AttributeMethodErrorCode code() const noexcept;
    AttributeMethodOperation operation() const noexcept;
};

ResourceUpdateOperation resource_update_operation_from_string(
    std::string_view value);
ComparisonOperation comparison_operation_from_string(std::string_view value);
ConstraintRestriction constraint_restriction_from_string(
    std::string_view value);

bool update_resource_value(
    const AttributeNumber& virtual_value,
    AttributeNumber& physical_value,
    ResourceUpdateOperation operation,
    bool safe = true,
    std::string_view diagnostic_name = {});

SatisfiabilityResult calculate_satisfiability_values(
    const AttributeNumber& virtual_value,
    const AttributeNumber& physical_value,
    ComparisonOperation operation,
    ConstraintRestriction restriction);

std::size_t double_satisfiability_batch_worker_count(
    std::size_t count,
    std::size_t configured_workers = 1) noexcept;

void calculate_satisfiability_values_double_batch(
    const std::vector<double>& virtual_values,
    const std::vector<double>& physical_values,
    ComparisonOperation operation,
    ConstraintRestriction restriction,
    std::vector<std::uint8_t>& flags,
    std::vector<double>& offsets,
    std::size_t workers = 1);

}  // namespace virne::network::attribute
```

`resource_update_operation_from_string` maps `+`/`add` and `-`/`sub` once.
`comparison_operation_from_string` maps `>=`/`ge`, `<=`/`le`, and `eq`; it
must deliberately reject `==` for parity. Restriction resolution accepts only
`hard`/`soft`. No string resolver may appear inside a node, edge, path,
candidate, constraint, or worker loop.

`AttributeNumber` is a boundary carrier, not the intended inner-loop storage.
At an attribute/network boundary, select `AttributeNumberKind` once and dispatch
to a bool/int64/double-specialized contiguous loop. Bool arithmetic promotes to
the integer lane as Python does. Check int64 overflow and report `numeric_range`
rather than invoking C++ signed overflow; Python arbitrary-size integers are a
documented representation boundary because frozen graph `AttrValue` is also
int64-bounded.

Mixed int64/double comparisons must reproduce Python's exact integer-versus-
binary64 ordering, including values beyond `2^53`, infinities, NaN, and signed
zero; casting the integer to double before comparison is not sufficient. Mixed
offset arithmetic does use the normal binary64 conversion because Python's
subtraction does, and raw result bits are differential-tested.

The double-only batch is a native throughput extension for independent scalar
constraints. All four vectors must already have the same size, so allocation
and schema conversion stay at the caller boundary. Comparison and restriction
are resolved enums and are dispatched once before the numeric loop; inner
loops touch only contiguous doubles/bytes and indices. Hard lanes preserve the
scalar comparison and offset result bits. Soft lanes deliberately skip the
discarded comparison and write literal true flags; the floating environment and
trapping signaling-NaN side effects are outside this primitive batch contract,
while every returned flag/offset bit remains Python-exact. Enum and shape
validation occurs before writes.

Worker width is a typed caller configuration, not a machine-specific automatic
policy. Values zero and one select the sequential route; wider values are
count- and affinity-capped. Output may legally alias either double input: each
SIMD/block loads its index range before overwriting it, and thread-construction
fallback never recomputes a completed block. Resource mutation, int64 overflow
paths, extrema delegation, and shared networks are deliberately not included in
this parallel extension.

On GCC/Clang x86, the batch runtime-dispatches once per contiguous block to
strict-IEEE AVX-512, then AVX2, with a portable scalar fallback. Operation and
restriction dispatch happens before the hot loop. There is no string, map,
registry, variant, allocation, or virtual dispatch per element.

Resource type is made valid by construction: a `ResourceMethodSpec` does not
carry a mutable `type` string that can cease to be `resource`. Python
reflection cases that mutate/remove `type`, `name`, or generator methods remain
oracle-only boundary characterization. Generation should use a concrete/CRTP
typed delegate from the future BaseAttribute class, not `std::function`,
`std::any`, or a string-keyed callback registry.

For extrema, preserve the Python fallback explicitly when resolving config:
declared owner `node` chooses the node definition registry; every other Python
owner chooses the link definition registry. Store that choice in
`ExtremaOriginRegistry`, then resolve the dynamic originator name once to
`AttributeDefinitionId`. This avoids name lookup during data generation while
keeping the graph-owner quirk visible.

## Graph and registry integration contract

The future node/link/graph adapter, not this independent numeric-policy leaf,
owns frozen graph integration:

- an attribute definition's dynamic output name is owned once for diagnostics
  and serialization;
- resolve it separately to frozen `AttrId` in the virtual and physical graph
  registries, because IDs from unrelated registries are not interchangeable;
- carry a typed pair such as `{virtual_attr_id, physical_attr_id}` through a
  check/update batch;
- inside an edge/path/node loop use only `AttrMap::find/at/set(AttrId)`, direct
  numeric references, `Vertex`, and stable edge IDs;
- resolve comparison/update/restriction strings to enums before the loop;
- resolve extrema originator names to a distinct attribute-definition ID; do
  not confuse that registry with a graph value `AttrId`; and
- retain the canonical name only on an error path, never hash it per element.

Resource generation later calls the stable dataset RNG surface using direct
`DistributionSpec`, `DatasetValueKind`, and one caller-owned
`NumpyRandomState`. This leaf must not create, seed, lock, or duplicate an RNG.

## Exact oracle and unit matrix

The Python oracle must verify the source hash, load only this file under a
private module name, and never import `virne` or `virne.network`. The pinned
oracle's real NumPy and NetworkX satisfy this file's two unconditional imports;
real Torch/OmegaConf must remain absent. Every record includes exact return
type/value or exception type/message/stage, mutated values, return identity,
and a probe access/call trace where order matters.

### Resource update

- valid `+`, `add`, `-`, and `sub`, plus every unsupported spelling including
  case variants and non-string values;
- wrong/missing type, missing/`None`/empty name, and validation-before-lookup;
- add with safe true/false/probe values proving safe is ignored;
- subtract below/equal/above availability, safe true/false, exact message, and
  no write on the guarded failure;
- missing virtual/physical keys and setter/arithmetic/comparison failures;
- mapping probes locking read/write order and the repeated reads used to format
  the guarded error;
- int/float/bool combinations, negative values, NaN, infinities, both signed
  zeros, int64 boundaries, and Python integers outside int64 as a recorded
  representation boundary;
- return literal `True` and exact final physical value/type.

### Resource generation

- false/missing/truthy generative flag;
- missing, `None`, non-callable, bound callable, and callable object delegate;
- exact network argument identity and returned object identity;
- delegated exception and getattr/truth/call traces;
- proof that `_generate_data` is not inspected for a falsey flag.

### Extrema

- no-op `update` and `check` with arguments that raise on any access, proving
  zero interaction;
- missing/`None` owner and originator, including read order;
- node, link, graph, empty, and unusual owners, locking non-node-to-link
  fallback;
- duplicate names across node/link registries proving the selected registry;
- missing originator, missing `get_data`, delegate failure, exact network
  identity, and returned object identity;
- integration cases for node/link/graph extrema concrete MROs.

### Information and constraint initialization

- cooperative-super argument forwarding and exact call/assignment order;
- preexisting/kwarg `is_constraint` overwritten to false or true;
- super failure before the final assignment;
- default, hard, soft, invalid, `None`, and direct
  `constraint_restrictions`-without-`restriction` cases;
- validation after the next initializer and exact invalid-restriction message;
- representative Node/Link/Graph concrete MRO constructions.

### Constraint calculation

- base check always raises without argument access;
- all five accepted spellings and rejected `==`/unknown/case variants;
- hard versus soft for true and false comparisons;
- exact raw signed offsets proving there is no `max(0, offset)` clamp;
- int/float/bool promotion, negative values, equality absolute offset, NaN,
  infinities, signed zero payload, and int64 range boundary;
- unknown/missing restriction after successful numeric work;
- comparison/subtraction/absolute-value probes and exception order;
- NumPy scalar/array behavior recorded as Python-only dynamic protocol scope;
- concrete node/link/graph resource and link-latency call-through cases.

C++-only tests must additionally cover every enum/error branch, variant lane,
overflow guard, move/copy behavior of specs, exact worker invariance if a batch
is accepted, concurrent independent callers, strict warnings, sanitizers, and
100-process stress. Frozen-integrity and the full CTest suite remain mandatory.

## Benchmark and multi-worker contract

Correctness and bit/type coverage precede timing. Normal development uses a few
representative warmed samples with exact checksums and compares Python,
sequential C++, and useful configured worker widths. Expand the timing corpus
only when output differs or a representative C++ row is unexpectedly slow.
This keeps semantic coverage broad without repeatedly tuning benchmark code.
Report median, compiler/runtime versions, CPU affinity, worker width,
result/mutation checksum, and exact input count. Fixture creation, process
startup, serialization, and checksum calculation remain outside the timed
numeric loop unless a row explicitly measures a boundary.

Canonical rows should include:

| Row | Timed work | Required checksum |
|---|---|---|
| resource add/subtract | large pre-resolved homogeneous int64 and double lanes | final physical values and return flags |
| guarded subtract | mixed pass/fail corpus measured separately from the exception path | comparison result, first failure, unchanged failed slot |
| hard satisfiability | large `ge`, `le`, and `eq` scalar corpus | flags, offset types, raw double bits |
| soft satisfiability | same corpus, proving masked flags but identical offsets | flags and raw offsets |
| extrema lookup/gather | future resolved definition IDs over a fixed network corpus | originator selection, value order, output identity class |
| delegation | one typed generation call versus Python method dispatch | result checksum; latency/report-only unless substantial |

One scalar update/check is sequential. A path touching shared physical
resources is also sequential because mutation order and first failure are
observable. Extrema delegation into one network is sequential because the
callee and return identity are external behavior.

The production double batch was retained after a representative 4,000,000-item
`hard/le` corpus showed configured width two at 6.074713 ms versus 7.079821 ms
sequential (1.165458x), with the same Python checksum for widths zero through
eight. The caller owns the worker configuration; production does not embed
that measured width or any size threshold. The extension uses pre-sized arrays,
input-order output, an explicit worker field, and affinity caps. It does not
parallelize shared mutations or perform string/registry lookup in a worker
loop.

A persistent executor is not justified merely to accelerate a handful of
scalar constraints. Document a sequential-only production decision if worker
creation/submission overhead loses after the full sweep.

## API decisions frozen for the first implementation

The pre-implementation review resolves the eight audit questions as follows.
Changing any item requires updating this contract before production code:

1. **Native numeric domain.** The native domain is exactly bool/int64/double,
   matching frozen graph storage. Arbitrary Python integers, NumPy scalars and
   arrays, and user-defined operators remain oracle-only dynamic protocol
   characterization. Integer overflow is a typed `numeric_range` failure and
   never C++ signed-overflow UB.
2. **Offset result lane.** The public result retains `AttributeNumber`: bool
   participates in the integer lane, integer-only arithmetic returns int64,
   and any double operand returns double. A later controller may add an
   explicit conversion boundary but this leaf does not canonicalize to double.
3. **Graph extrema fallback.** Preserve the original non-node-to-link fallback,
   including owner `graph`, through explicit `ExtremaOriginRegistry`. A future
   correction is a separate, named deviation.
4. **Equality spelling.** Reject `==`; only `eq` resolves to equality. This
   preserves the observable validator behavior rather than the unreachable
   later branch.
5. **Violation definition.** Return the raw signed offset and never clamp it,
   preserving code behavior rather than the inaccurate docstring.
6. **Constructor/MRO surface.** Typed C++ composition makes missing/mutated
   fields unrepresentable. Python reflection and cooperative-MRO failure probes
   remain oracle-only; this independent policy leaf adds no dynamic adapter.
7. **Higher-layer inconsistent fields.** Controller reads
   `constraint_restrictions`/`checking_level` from types that may not define
   them, and NodePosition inherits an unimplemented check. Those remain for
   their owning component notes and are not patched in this policy leaf.
8. **Graph classes in the factory.** Graph attribute classes exist but are not
   registered. This leaf neither registers nor removes them.

If implementation requires an exact signature from the future BaseAttribute,
attribute registry, or graph adapter, pause and document that owning API first.
Do not inspect or modify frozen graph source to solve it.

## Implemented target boundary

The intended first implementation turn is limited to:

- `vne_attribute_method`: enums, resolvers, typed numeric update/calculation,
  policy specs, and typed errors;
- `vne_attribute_method_unit`;
- `vne_attribute_method_harness` plus a direct-source Python comparator; and
- a checksum-gated benchmark/worker sweep.

Resource generation delegation and extrema registry integration require a
concrete attribute/registry owner and therefore belong with `BaseAttribute` and
node/link/graph attributes. Their Python reflection/MRO behavior is already
frozen by the oracle here; this leaf does not add a dynamic callback adapter.
No Torch, solver, system, learning code, second RNG, dynamic fixed-field map, or
frozen-library change was introduced.

## Completion evidence

- Isolated targets: `vne_attribute_method`, unit, and harness.
- Differential: **PASS 123/123**: 93 scalar/dynamic-MRO cases plus 30 raw-bit
  batch cases covering `ge/le/eq`, hard/soft, workers 0/1/2/3/8, signed zero,
  subnormal, infinities, five qNaN payloads, and sNaN.
- Unit coverage: int64/mixed-number boundaries, every batch lane at workers
  0..8, output/input aliasing, invalid shapes/enums, and eight simultaneous
  callers running 32 rounds each.
- Sanitizers and warnings: ASan, UBSan, leak checks, and
  `-Wall -Wextra -Wpedantic -Werror` all pass.
- Representative timing: every non-exception scalar row is faster than Python,
  from 19.044881x to 115.556359x; the exception-only failure path is report-only.
- Frozen graph/CSV/config/yaml-cpp and completed random code were not changed or
  rebuilt for this leaf.

Checked-in compact records:

- `porting/results/attribute_method_differential_2026-07-28.json`;
- `porting/results/attribute_method_benchmark_2026-07-28.json`;
- `porting/results/attribute_method_worker_sweep_2026-07-28.json`.

Stable production hashes:

- header: `7E64F19F8EF0D4CF670193197E927E4ABC9D8AE938193560E3C96CA91C36A209`;
- implementation: `A02426F39E203EDAF1A2D0770C3314743D9A0F1D744C005AA815BCFACBDDDF40`;
- unit: `0663E6DC4B9A6FB95E68A8A888A3810A5173B89D4825AE78AEE68A35678D422B`;
- harness binary: `640F6DB56AE35BD88D1370E7EEFE47D910E054334EBA9FC926B10CF37BD2A782`.
