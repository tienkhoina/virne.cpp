# Component API: `virne.utils.config`

State: **COMPLETE / FROZEN** on 2026-07-29.

Python oracle: `../virne/virne/utils/config.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`AD1D32E6B2DB842C958240301535D356919DBD822E403D6A767CE2ED2BBFE787`.
The completed Config, Random, dataset, and attribute API notes are dependencies;
their implementations must not be rebuilt or reopened for ordinary use.

## Fixed-field rule

Every known timestamp, path input, simulation count, feature count, error, and
operation is a direct struct field or enum. No fixed field is stored in a
string-keyed map. Dataset attribute names remain genuinely dynamic output
values owned once by `DatasetAttributeSpec`. Attribute kinds are resolved to
`AttributeKind`; extracted membership is a fixed five-slot array indexed
directly by that ID. Each attribute group is counted in one pass without
string comparison or mapping lookup.

`Config`/YAML exists only as a cold dynamic compatibility boundary. The three
fixed run-directory paths are each read once and immediately returned as a
typed `RunDirectoryInput`. YAML nodes never enter simulation derivation.

## Stable native API

```cpp
enum class UtilsConfigErrorCode : std::uint8_t;
enum class UtilsConfigOperation : std::uint8_t;
class UtilsConfigException : public std::runtime_error;

struct RunIdTimestamp { int year, month, day, hour, minute, second; };
struct RunDirectoryInput {
    std::filesystem::path save_root_dir;
    std::string solver_name;
    std::string run_id;
};
struct SimulationAttributeKinds { std::vector<AttributeKind> kinds; };
struct ExtractedAttributeKinds { std::array<bool, 5> included; };

struct SimulationConfigInput {
    PhysicalDatasetSetting physical_dataset;
    VirtualDatasetSetting virtual_dataset;
    SimulationAttributeKinds physical_node_attributes;
    SimulationAttributeKinds physical_link_attributes;
    SimulationAttributeKinds virtual_node_attributes;
    SimulationAttributeKinds virtual_link_attributes;
    ExtractedAttributeKinds extracted_attribute_kinds;
    std::optional<DatasetScalar> seed;
};

struct FeatureConstructorSummary;  // five direct fields
struct SimulationConfigSummary;    // all Python output values are direct fields

std::string generate_run_id(
    const RunIdTimestamp&, std::string_view hostname, PyRandom&);
std::string generate_run_id(PyRandom&);
YAML::Node resolve_config_to_node(const Config&);
YAML::Node resolve_config_to_node(const YAML::Node&);
SimulationConfigSummary derive_simulation_config(
    const SimulationConfigInput&);
std::vector<SimulationConfigSummary> derive_simulation_configs_batch(
    const std::vector<SimulationConfigInput>&, std::size_t workers = 1);
std::filesystem::path get_run_id_dir(const RunDirectoryInput&);
RunDirectoryInput run_directory_input_from_config(const Config&);
```

## Behavior and ownership

- Run ID order is local timestamp formatting, hostname, then exactly one
  CPython-compatible `randint(0, 9999)` draw. Formatting is locale-independent.
  `PyRandom` remains caller-owned; concurrent callers must not share one stream
  without their own synchronization.
- `Config` resolution returns a fully resolved mapping. A raw YAML mapping
  preserves shared node state, which is the closest typed boundary to Python's
  exact plain-dict identity; no exact Python object-identity claim is made.
- Summary derivation calls the completed physical and virtual dataset path
  builders, then fills typed fields. The parallel batch processes complete
  inputs in caller-selected contiguous lanes without copying settings into
  intermediate request objects. Worker zero/one is sequential; wider values
  are capped only by input count. Results retain order and the lowest failing
  input index wins. Partial thread creation is exception-safe.
- Python's incremental DictConfig mutation and partial commits are a recorded
  language boundary. Native callers own the returned
  `SimulationConfigSummary`, `FeatureConstructorSummary`, and run path directly.

## Frozen gate

- Unit, strict production/unit compile, ASan/UBSan/leaks, and frozen-foundation
  integrity: **PASS**.
- Exact differential: **10 shared + 6 native + 5 boundaries = 21 PASS**.
- Frozen 2,048-config checksum: `6644728919515556009`.
- C++ speedup over Python at workers `1/2/8`:
  **36.342x / 52.867x / 43.241x**.

Evidence: `../results/utils_config_2026-07-29.md`. Accepted differential and
benchmark artifacts are permanently frozen and must not be rerun or updated.
