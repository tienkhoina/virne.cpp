# Component API: `network.physical_network`

State: **COMPLETE / FROZEN** on 2026-07-28.

Python source: `../virne/virne/network/physical_network.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`E37D48B2C1651B503931597B6CCA5620A413A11C3D003CF8C70AF89246E4CA5A`,
11,580 bytes. The completed BaseNetwork, attribute factory, topology, setting,
Random, graph/GML, and performance contracts were read before opening this
previously undocumented leaf.

## Python behavior

- Construction delegates to BaseNetwork. The topology wrapper changes only
  the dynamic default topology type from `path` to `waxman`.
- `from_setting` constructs the configured definitions first, then applies the
  optional seed to the process Python/NumPy streams. Torch/CUDA seeding belongs
  to the deliberately omitted ML boundary.
- A truthy existing `topology.file_path` is read as undirected GML with
  `label="id"`. Loaded topology/node/edge storage and graph metadata replace
  the generated storage; loaded `node_attrs_setting` and `link_attrs_setting`
  metadata are removed. Attribute names found on only the first loaded node
  or first loaded edge and absent from the configured registries are appended
  as non-generative node/link status definitions in encountered key order.
- Any exception in that supported load path falls back to generation. A
  missing/false path also generates; a truthy missing path first emits a
  warning. Every generation path requires `topology.num_nodes`. Missing it
  raises after construction/seeding and, after a load error, replaces the
  original load failure.
- Generation defaults missing `topology.type` to Waxman and delegates all
  topology and attribute generation in their established order. Shared
  Python-compatible topology and NumPy streams remain sequential.
- `to_gml` writes the BaseNetwork-prepared graph. `save_dataset` recursively
  creates the requested directory and defaults the file name to `p_net.gml`.
  `load_dataset` checks that exact joined path first, then delegates
  BaseNetwork GML restoration.
- Python stdout diagnostics, arbitrary `DictConfig` reflection/in-place
  `topology.num_nodes` mutation, arbitrary sparse/string node labels, exotic
  mapping truthiness, monkey-patching, and partial mutation after an exception
  inside NetworkX private-storage replacement are dynamic boundaries.

## Stable C++ API

All names are in `virne::network`.

```cpp
enum class PhysicalTopologyOrigin : std::uint8_t {
    generated,
    loaded_gml,
    generated_after_gml_error,
};

struct PhysicalNetworkBuildOptions {
    std::optional<std::uint32_t> seed;
    std::size_t factory_workers = 1;
    std::size_t attribute_workers = 1;
};

struct PhysicalNetworkBuildReport {
    PhysicalTopologyOrigin origin = PhysicalTopologyOrigin::generated;
    std::optional<std::string> requested_file;
    std::optional<std::string> gml_error;
};

class PhysicalNetwork final : public BaseNetwork {
public:
    PhysicalNetwork();
    explicit PhysicalNetwork(BaseNetworkConstruction);
    explicit PhysicalNetwork(BaseNetwork&&);
    PhysicalNetwork(PhysicalNetwork&&) noexcept;
    PhysicalNetwork& operator=(PhysicalNetwork&&) noexcept;
    PhysicalNetwork(const PhysicalNetwork&) = delete;
    PhysicalNetwork& operator=(const PhysicalNetwork&) = delete;

    static PhysicalNetwork from_setting(
        const virne::utils::SettingDocument&,
        const PhysicalNetworkBuildOptions& = {});
    static PhysicalNetwork from_setting(
        const virne::utils::SettingDocument&,
        RandomContext&,
        const PhysicalNetworkBuildOptions& = {});

    const PhysicalNetworkBuildReport& build_report() const noexcept;
    void to_gml(const std::string& path) const;
    void save_dataset(const std::string& directory,
                      const std::string& file_name = "p_net.gml") const;
    static PhysicalNetwork load_dataset(
        const std::string& directory,
        const std::string& file_name = "p_net.gml");
    PhysicalNetwork clone() const;
};
```

The overload without an explicit context uses `global_random_context()` and
calls its optional-seed operation, preserving Python module-global stream
continuation when the seed is absent. The explicit-context overload consumes
only that caller-owned context and optionally resets both streams first.

`build_report` replaces unconditional stdout with typed, directly accessible
diagnostics. `requested_file` is present for a truthy configured path;
`gml_error` is present only when an existing file failed and generation
succeeded. It is diagnostic cold data and is never read by graph loops.

## Decode, load, and ID rules

- The raw setting document must have an object root. Known topology keys are
  resolved once to `SettingKeyId`, validated, and decoded immediately into a
  direct `TopologyRequest`: `num_nodes`, `type`, `m`, `n`, `random_prob`,
  `wm_alpha`, `wm_beta`, optional native `max_attempts`, and `file_path`.
  Missing type is the enum `TopologyType::Waxman`. Type-specific numeric keys
  are decoded only for the selected topology; irrelevant keys and unknown
  keys are ignored like Python's permissive keyword path. A present null or
  nonnumeric relevant option is a typed cold-boundary error rather than being
  mistaken for a missing/default field.
- BaseNetwork configuration is decoded once through
  `base_network_construction_from_setting`; no generation loop retains or
  searches the generic setting tree. A successful GML path copies that typed
  `BaseNetworkConfig` to rebuild the move-only final registries; it does not
  decode the raw setting document a second time. Factory and attribute workers
  are direct caller fields. Zero/one selects the established sequential
  dependency path; wider values are forwarded unchanged and contain no
  machine policy.
- On supported GML success, configured definitions retain their registry
  order. First-sample unknown attributes are resolved once from the loaded
  graph registry and appended as typed `AttributeKind::status` specs. Loaded
  graph metadata is captured, the incoming graph-level value map is cleared,
  and the captured values are replayed after config metadata. Thus loaded
  values win while an overwritten config key keeps its original insertion
  position and loaded-only keys append in loaded order, as in Python. The two
  public setting snapshots are then erased while BaseNetwork keeps its private
  canonical factory snapshots.
- The frozen dense GML loader retains structural node `id` as an attribute,
  while NetworkX consumes that key when `label="id"`. PhysicalNetwork resolves
  that graph-local `AttrId` once and erases it by ID from every loaded node
  before sample discovery, preventing a spurious status definition and
  matching Python node data without string lookup in the node loop.
- Node/edge discovery loops walk direct ordered attribute IDs. A dynamic name
  is obtained once at this cold boundary, bound once to the factory registry,
  then represented by `AttributeRegistryId`/graph `AttrId`. No string lookup
  occurs in topology, generation, node, edge, or worker loops.
- GML input is limited to the frozen dense undirected `label="id"` domain.
  Unsupported labels/multigraphs/directed inputs follow the load-error fallback
  when generation config is available.

## Errors, order, and concurrency

Invalid setting shapes/types and a missing generation `num_nodes` fail before
topology generation. Dependency exceptions propagate at their ordered stage,
except an existing-file GML exception is retained in `build_report` only when
fallback generation succeeds. Filesystem/GML save errors propagate. A missing
dataset file raises `std::runtime_error` before invoking the loader.

One `from_setting` call mutates one network and one pair of RNG streams in
canonical order: optional reseed, topology attempts, node attribute draws,
then link attribute draws. Those operations are never parallelized across RNG
boundaries. Independent networks may be built concurrently only with distinct
`RandomContext` instances. The global-context overload requires serialized
caller access.

## Frozen acceptance gate

- Exact differential: **PASS**, 19 classified cases. The 11 shared cases cover
  generation at workers `0/1/2/8`, stream continuation, loaded/empty GML,
  malformed and missing-file fallback, missing-`num_nodes` precedence, and
  dataset load. Native clone/move is covered separately; seven deliberately
  unsupported dynamic Python behaviors remain recorded boundaries.
- Compact orchestration benchmark: **PASS** at 32,768 nodes after exact binary
  output gating. C++ is `4.831x`, `5.580x`, and `4.227x` faster at caller
  workers `1`, `2`, and `8`, respectively.
- The accepted JSON artifacts and summary in
  `porting/results/physical_network_2026-07-28.md` are frozen provenance. Do
  not rerun or update this benchmark while porting later modules.

## Current implementation record

- Production: `virne/network/physical_network.h` and `.cpp`.
- Isolated library: `vne_physical_network`, linked to the completed
  `vne_base_network`; aggregate `vne_network` links this target instead of
  recompiling the source.
- Unit/CTest: `vne_physical_network_unit` / test of the same name.
- Current SHA-256: header
  `765C2F1D9E2ED6BEFC81574F8B8BD2AFBB9B9C62F9D456CFC75AA19E959FAA30`,
  implementation
  `BD1736C4660B4623E91E710ABF368B2E537CCFBF7E32DF179E45DF6CA7EE67EF`,
  unit
  `5CA5328D6205703CF7BCF53E2362821CC4CF86E6BBB746E28F7DAA523CA1882E`.
- Release isolated build and direct unit: **PASS**. CTest: **PASS, 1/1**.
- `-Wall -Wextra -Wpedantic -Werror` syntax checks: **PASS** for production
  and unit. The production leaf also passes scoped `-Wconversion` and
  `-Wsign-conversion` with dependency headers treated as system inputs.
- Unit coverage includes configured workers `0/1/2/8`, seed and absent-seed
  continuation for both RNG streams, independent concurrent contexts, every
  topology family/default, generation failures, first-sample status ordering,
  loaded/config metadata precedence, structural-ID removal, move/clone
  rebinding, malformed/missing/empty GML branches, and dataset round trips.
- Differential and benchmark artifacts: **PASS / FROZEN**. Every accepted
  dependency and VirtualNetwork benchmark remains untouched and frozen.
