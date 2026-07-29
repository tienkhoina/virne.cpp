# Component API: `network.attribute` factory

State: **COMPLETE AND FROZEN** on 2026-07-28.

Source: `../virne/virne/network/attribute/__init__.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`11AC0C2235BA02F6538427F3774C93766820EA640531C97C94F4B4DAF2D7B010`,
3,558 bytes and 75 lines. Completed BaseAttribute, NodeAttribute,
LinkAttribute, GraphAttribute, dataset, setting, and performance documents were
read first. Public dependency headers were opened only where the documents did
not contain exact constructor signatures.

This cold-boundary leaf must be complete before BaseNetwork. It converts raw
setting objects into direct typed specs once; graph/data hot loops never receive
the raw setting tree or factory strings.

## Python behavior to retain

- `ATTRIBUTES_DICT` contains exactly eight owner/type pairs: node/link status,
  node/link extrema, node/link resource, node position, and link latency. It
  intentionally contains no graph classes even though graph classes and
  `create_graph_attrs_from_dict` exist.
- `create_attr_from_dict` reads `name`, `owner`, then `type`; false/missing
  owner or type fails before lookup. Unsupported pairs fail before constructor
  invocation. Remaining keys are converted to strings and passed as keyword
  arguments in insertion order.
- Specific node/link/graph helpers call the general factory first and only then
  verify the resulting inheritance family. Thus a link spec passed to the node
  helper is constructed before the type error. Every graph helper call fails at
  the unsupported-pair lookup with the current registry.
- Setting-list helpers evaluate items strictly in input order. Duplicate names
  replace the value associated with the first insertion position. The first
  invalid item stops the call; no partial collection is returned.
- Constructor defaults and precedence are retained by the typed decode:
  `constraint_restrictions` precedes `restriction`; node/link/path checking
  levels are family defaults; extrema requires an originator; node position
  uses `min_r=0`, `max_r=1`; link latency treats distribution `position` as its
  position-generation mode and otherwise uses the configured distribution.
- A present `distribution: null` means no distribution. A present null
  restriction is not treated as absent: it fails as `invalid_restriction`;
  fallback/default applies only when the corresponding key is missing.
- Python arbitrary object keys, observable `str(key)` side effects, arbitrary
  extra instance attributes, non-string names, unhashable owner/type values,
  duplicate keyword collisions created only after `str(key)`, DictConfig
  interpolation, and Python `-O` removal of `assert` are dynamic boundaries.
  Native raw settings accept the completed `SettingObject` profile and reject
  unsupported field types with typed errors.

## Stable C++ surface

```cpp
struct AttributeFactorySpec {
    std::string name;
    AttributeOwner owner = AttributeOwner::node;
    AttributeKind kind = AttributeKind::status;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    std::optional<std::string> originator_name;
    std::optional<AttributeDefinitionId> originator_id;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    std::optional<CheckingLevel> checking_level;
    double minimum_radius = 0.0;
    double maximum_radius = 1.0;
    LatencyGenerationKind latency_generation =
        LatencyGenerationKind::configured;
    AttributeNumber minimum = 0.0;
    AttributeNumber maximum = 1.0;
};

AttributeFactorySpec attribute_factory_spec_from_setting(
    const virne::utils::SettingObject&);

std::unique_ptr<BaseAttribute> create_attribute(AttributeFactorySpec);
std::unique_ptr<NodeAttribute> create_node_attribute(AttributeFactorySpec);
std::unique_ptr<LinkAttribute> create_link_attribute(AttributeFactorySpec);
std::unique_ptr<GraphAttribute> create_graph_attribute(AttributeFactorySpec);

using AttributeRegistryId = AttributeDefinitionId;

class AttributeRegistry;
class NodeAttributeRegistry;
class LinkAttributeRegistry;

AttributeRegistry create_attributes_from_specs(
    const std::vector<AttributeFactorySpec>&, std::size_t workers = 1);
NodeAttributeRegistry create_node_attributes_from_specs(
    const std::vector<AttributeFactorySpec>&, std::size_t workers = 1);
LinkAttributeRegistry create_link_attributes_from_specs(
    const std::vector<AttributeFactorySpec>&, std::size_t workers = 1);

AttributeRegistry create_attributes_from_setting(
    const virne::utils::SettingList&, std::size_t workers = 1);
NodeAttributeRegistry create_node_attributes_from_setting(
    const virne::utils::SettingList&, std::size_t workers = 1);
LinkAttributeRegistry create_link_attributes_from_setting(
    const virne::utils::SettingList&, std::size_t workers = 1);
```

Each registry owns insertion-ordered entries and a compact name index. Its
public surface is `bind(string_view)`, checked `at(AttributeRegistryId)`,
`find(string_view)`, `entries()`, and `size()`. Node/link registries expose
typed attribute references, so BaseNetwork never downcasts in a graph loop.
Registries are movable and non-copyable because they uniquely own polymorphic
definitions. Exact error codes/stages and entry declarations are finalized in
the header.

## ID, threading, and performance rules

- Raw fixed key names are each resolved at most once per setting object. Decode
  immediately stores owner/kind/distribution/dtype/restriction/checking/
  generation as enums and all fixed values as direct fields.
- Registry insertion hashes a dynamic attribute name once. Repeated consumers
  bind once to `AttributeRegistryId` and use direct indexed entries thereafter.
  Duplicate replacement retains the first owned key storage and compact ID.
- Typed-spec construction can evaluate independent items in deterministic
  contiguous blocks at caller-configured workers `0/1/2/8`; zero/one is
  sequential. Results/errors occupy pre-sized slots, the lowest input error is
  rethrown, and ordered deduplication is a final sequential pass. Raw setting
  decode remains sequential because boundary coercion/failure order is public.
- Worker loops switch only on direct owner/kind enums and spec fields. They do
  not look up raw setting keys or registry names. No host-specific worker count
  is embedded; the caller supplies it.
- Extrema originator names are resolved against the completed collection once.
  A missing originator retains an invalid ID and is rejected only by the later
  generation/use stage, matching Python factory construction behavior.

## Accepted gate

Cover all eight registered pairs and constructor defaults, distribution/dtype
and numeric fields, restriction precedence, checking levels, position/latency
special fields, missing/false/unsupported owner/type, graph omission, every
specific-helper mismatch, invalid raw setting value types, empty settings,
duplicate overwrite/first order, compact-ID access/move lifetime, extrema
forward/back/missing references, workers `0/1/2/8`, lowest-index failures, and
concurrent independent batches. Record the dynamic Python boundaries above.

After exact differential passes, run one compact typed-spec construction
benchmark against the original Python factory at configured workers `1/2/8`,
one warm-up and three samples. Gate ordered names, owners/kinds, concrete class,
and direct fixed fields before timing. Once accepted, freeze the benchmark and
move immediately to BaseNetwork.

The gate passed 29 direct differential cases, three native typed cases, and
seven documented Python-only dynamic boundaries (39 total). Unit, exact
differential, ASan/UBSan/leak checks, strict production/unit/harness warnings,
full CTest 29/29, and frozen-foundation integrity pass.

The one accepted 32,768-spec benchmark is frozen. Python median was
114.526599 ms; C++ medians were 19.154425/17.849414/23.752271 ms at workers
`1/2/8`, or 5.979x/6.416x/4.822x faster. Worker count remains caller config;
no machine-specific automatic policy is embedded. See
`../results/attribute_factory_2026-07-28.md` for hashes and evidence. Do not
rerun or update the accepted benchmark source, driver, binary, or JSON.
