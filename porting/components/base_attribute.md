# Component API: `network.attribute.base_attribute`

State: **COMPLETE / FROZEN API** on 2026-07-28.

Source: `../virne/virne/network/attribute/base_attribute.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`103C5C16126CA76E782C2191FF0811B95ED88A0C3637F3A61E84C0B22DF42E8E`.

## Python behavior to retain

- Construction stores `name`, `owner`, `type`, `generative` in that order;
  arbitrary kwargs are then assigned in caller order.
- `to_dict()` is a shallow instance-dictionary copy; `repr` follows insertion
  order and uses `str(value)`.
- Default `generate_data` and `update_data` raise
  `NotImplementedError("Subclasses must implement this method.")`.
- `_generate_data` rejects non-generative attributes first. Node owner uses
  `num_nodes`; every other owner uses `num_links`.
- Uniform/normal/exponential/poisson delegate the completed dataset RNG API;
  falsey dtype defaults to float.
- Customized draws NumPy uniform `[0,1)`, then applies
  `value * (max-min) + min`; it ignores dtype and requires numeric `min < max`.
- Unsupported distributions raise the original two-line message.
- `_get_config_value` is a dynamic Python boundary helper used by derived
  constructors. C++ derived configs use direct typed fields instead.

## Stable C++ surface

```cpp
struct NetworkCardinality {
    std::size_t num_nodes = 0;
    std::size_t num_links = 0;
};

struct BaseAttributeSpec {
    std::string name;
    AttributeOwner owner = AttributeOwner::node;
    AttributeKind kind = AttributeKind::status;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    std::optional<std::string> originator;
    std::optional<bool> is_constraint;
};

class BaseAttribute {
public:
    explicit BaseAttribute(BaseAttributeSpec spec);
    virtual ~BaseAttribute() = default;

    const BaseAttributeSpec& spec() const noexcept;
    virtual virne::utils::GeneratedData generate_data() const;
    virtual void update_data();
    virne::utils::GeneratedData generate_configured_data(
        const NetworkCardinality& network,
        NumpyRandomState& rng,
        std::size_t workers = 1) const;
    BaseAttributeSnapshot to_dict() const;
    std::string repr(std::string_view class_name = "BaseAttribute") const;
};

AttributeOwner attribute_owner_from_string(std::string_view);
std::string_view attribute_owner_name(AttributeOwner) noexcept;
AttributeKind attribute_kind_from_string(std::string_view);
std::string_view attribute_kind_name(AttributeKind) noexcept;
```

Fixed fields are direct members and discriminants are enums. `name` and
`originator` are owned once; future graph adapters resolve them once to compact
definition/graph IDs. No string lookup is allowed in generation loops.

Python arbitrary kwargs/reflection remain oracle characterization, not a reason
to store fixed derived schemas in a dynamic map. `BaseAttributeSnapshot` is the
typed cold serialization view in canonical field order; derived classes append
their own direct fields.

`BaseAttributeException` exposes direct `BaseAttributeErrorCode` and
`BaseAttributeOperation` fields. Resolver strings are accepted only at the cold
boundary. Generation dispatch uses enums/direct optionals; no map or string is
read in a sample loop.

## Locked generation details

- `generative == false` fails before cardinality, distribution validation, or
  RNG access. Node ownership selects `num_nodes`; link and graph select
  `num_links` exactly like Python.
- Uniform, normal, exponential, and poisson reuse the completed typed dataset
  RNG leaf. An absent dtype becomes floating. Missing normal `loc` or `scale`
  is forwarded as an explicit nonnumeric value so it fails before a draw,
  matching Python class attributes whose value is `None`.
- Python `BaseAttribute` never forwards the dataset-only `reciprocal` flag for
  poisson. This layer therefore clears that flag; callers that need reciprocal
  poisson must call the documented dataset API directly.
- Customized generation always returns the floating lane and ignores dtype.
  When both bounds are integer/bool, their positive difference is computed
  exactly in the full unsigned 64-bit range before the one conversion to
  `double`. This preserves Python integer subtraction near and above `2^53`.
- One `NumpyRandomState&` owns all draws in canonical NumPy order. Customized
  workers `0/1` are sequential; a configured width above one is capped by item
  count and available hardware, then transforms disjoint contiguous blocks
  only after the single-threaded draw. Worker creation failure completes the
  unlaunched blocks locally and joins every launched worker.
- `to_dict()` uses the documented canonical fixed-field order rather than a
  dynamic reflection map. `repr()` formats that snapshot and accepts the cold
  class-name boundary explicitly. Abstract Python `set_data/get_data` remain
  responsibilities of future typed node/link/graph adapters.

## Test and performance gate

The isolated unit covers resolvers, construction/snapshot/repr, default errors,
all distributions and dtype lanes, node/link/graph/zero sizing, invalid inputs,
RNG continuation, exact large-integer customized spans, workers `0/1/2/8`, and
four concurrent independent callers. The direct pinned Python oracle passed
32/32 cases (four static plus 28 generation/error/state cases).

The deliberately compact benchmark used 300,000 values, worker widths `1/2/8`,
one warm-up and three samples. All six rows beat Python: customized was
`1.441x` to `2.575x`; exponential-to-int was `1.404x` to `2.496x`, with worker
8 fastest for that parallel cast workload. Customized worker creation overhead
made width 1 fastest at this size; widths remain caller configuration, not a
host-specific automatic policy.

Strict GCC 11 warnings, ASan/UBSan/leaks, the isolated targets, full CTest
24/24, frozen integrity, and `git diff --check` pass. Exact commands, artifact
hashes, and timing rows are in `../results/base_attribute_2026-07-28.md`.
