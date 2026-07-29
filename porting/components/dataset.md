# Component design: non-Torch `virne.utils.dataset`

State: **ALL NON-TORCH LEAVES COMPLETE / PYTHON-TORCH SEED FACADE DEFERRED**
on 2026-07-28.

This note is the source-of-truth contract for the completed core, NumPy-RNG,
and XML/graph/GML leaves plus the intentionally deferred seed boundary. The
implementation changed only the dataset utility, its isolated
targets/tests/tooling, and porting documentation. Frozen graph, CSV, config,
yaml-cpp, and random files remain unchanged.

## Documentation-first rule

The original `dataset.py` was read once because no component note existed.
Future work must begin here and must not reopen the Python or C++ implementation
merely to rediscover behavior.  Source may be opened again only for a focused
exact-differential mismatch or measured low-level optimization that cannot be
resolved from this contract.  Any newly learned fact must be added here in the
same change.

The audit also read `PORTING_STATUS.md`, `porting/NON_ML_COMPONENT_MAP.md`,
`PERFORMANCE_CONTRACT.md`, `FROZEN_COMPONENTS.md`, `README.md`, and the related
`components/manager.md` before opening Python.  External callsites were then
located by symbol-only Git grep without reopening `dataset.py`.

## Source identity and scope

- Original source: sibling `../virne/virne/utils/dataset.py`.
- Pinned original commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Exact checkout SHA-256:
  `269650EBCC373D7BDF79FA17346BD6F847973F17E60C1AC9BCAE7CFD97BF936F`.
- Exact checkout size: 9,635 bytes; 198 physical lines as decoded in the audit.
- The hash was checked from the same byte buffer used for the one permitted
  source read.
- This note covers every function in the file, but a same-named full
  `set_seed` implementation is not part of the non-Torch production phase.
- The graph, CSV, config, and yaml-cpp directories remain frozen inputs.  This
  audit did not touch them.

The module has no `__all__`.  Python `from .dataset import *` in
`virne.utils.__init__` therefore copies every top-level name not beginning with
an underscore, including imported modules/types as well as functions.  The C++
surface must intentionally export only typed dataset APIs; reproducing Python
namespace pollution (`os`, `random`, `np`, `torch`, typing names, and
OmegaConf names) is neither required nor desirable.

## Complete Python API inventory

| API | Return on ordinary success | Direct runtime dependency | Non-Torch disposition |
|---|---|---|---|
| `set_seed(seed: Optional[int] = None)` | `None` | `random`, NumPy, Torch CPU/CUDA/cuDNN, environment | mixed ML/global-state boundary; defer same-named facade |
| `generate_data_with_distribution(size: int, distribution: str, dtype: str, **kwargs)` | Python list from a NumPy array | NumPy legacy global RNG | **COMPLETE** through an explicit `NumpyRandomState&` boundary |
| `get_distribution_average(self, distribution, dtype, **kwargs)` | always `None` | none | independent compatibility leaf |
| `generate_file_name(config, epoch_id=0, extra_items=[], **kwargs)` | `str` | `vars`, mapping/string behavior | independent typed naming leaf |
| `get_p_net_dataset_dir_from_setting(p_net_setting, seed: Optional[int] = None)` | native joined path as `str` | `os.path`, two local helpers | independent after typed setting boundary |
| `get_v_nets_dataset_dir_from_setting(v_sim_setting, seed: Optional[int] = None)` | native joined path as `str` | `os.path`, two local helpers | independent after typed setting boundary |
| `get_distribution_parameters(distribution_dict)` | new Python list | mapping protocol | independent compatibility leaf |
| `get_parameters_string(parameters)` | `str` | sequence/iteration/string conversion | independent compatibility leaf |
| `preprocess_xml(topylogy_name, xml_source_fpath, gml_target_fpath)` | the exact created `networkx.Graph` | lazy NetworkX and `xml.dom.minidom`, filesystem | non-Torch, but graph/XML/GML dependent |

The spelling `topylogy_name` is the actual Python keyword name.  Passing
`topology_name=` raises an unexpected-keyword `TypeError`; that typo is an
observable Python boundary.

Top-level imports are `os`, `random`, NumPy, Torch, three typing names, and
`DictConfig`/`OmegaConf`.  The OmegaConf names, `Dict`, and `Union` are unused.
NetworkX and minidom are imported lazily only when `preprocess_xml` runs.

## Dependency and call graph

Internal calls are small and acyclic:

```text
set_seed                         -> random / NumPy / Torch / os.environ
generate_data_with_distribution -> NumPy global RNG
get_distribution_average        -> no calls (pass)
generate_file_name               -> vars + Python formatting
get_p_net_dataset_dir_from_setting
  -> get_distribution_parameters -> mapping access
  -> get_parameters_string       -> len / iteration / str
  -> os.path.exists/basename/join
get_v_nets_dataset_dir_from_setting
  -> get_distribution_parameters
  -> get_parameters_string
  -> os.path.join
preprocess_xml
  -> minidom.parse
  -> NetworkX Graph/add/write_gml
```

Repository callsite scan at the pinned commit found:

- `set_seed`: called by `network/dataset_generator.py`,
  `network/physical_network.py`, and
  `network/virtual_network_request_simulator.py`; imported by
  `core/environment.py` and `system/base_system.py`; patched by network tests.
- `generate_data_with_distribution`: used by
  `network/attribute/base_attribute.py` and
  `network/virtual_network_request_simulator.py`; imported by
  `network/attribute/link_attribute.py`.
- physical/virtual dataset directory builders: used by `utils/config.py`,
  `network/dataset_generator.py`, and `core/environment.py`, and exercised by
  `tests/network/test_dataset_generator.py`.
- `get_distribution_parameters` and `get_parameters_string`: called only by
  the two directory builders in this leaf and re-exported by `virne.utils`.
- `generate_file_name`, `get_distribution_average`, and `preprocess_xml`: no
  non-definition callsite was found; the public functions remain re-exported.

The original network tests patch `set_seed` and the two directory builders
while testing higher layers.  No direct original test covers distribution
values/RNG continuation, helper bugs, filename behavior, XML graph fields, or
GML bytes.  The future isolated matrix below is therefore mandatory rather
than supplemental.

This explains the required dependency order: pure helpers and naming can be
ported before attributes/network; NumPy-compatible generation must precede the
attribute model; the directory builders must precede network `Generator`; and
the mixed seed facade must be cut at the Torch boundary rather than importing
Torch into C++.

## Exact observable contracts

Unless stated otherwise, Python exceptions propagate immediately, earlier
global/filesystem/RNG side effects remain, there is no rollback, and successful
functions write neither stdout nor stderr.

### `set_seed`

`seed is None` returns immediately without calling any backend, changing an
environment variable, or probing CUDA.  For every other value, evaluation and
side-effect order is exactly:

1. `random.seed(seed)`;
2. `np.random.seed(seed)` on NumPy's legacy module-global RNG;
3. `torch.manual_seed(seed)`;
4. call `torch.cuda.is_available()`;
5. if true, call `torch.cuda.manual_seed_all(seed)`;
6. assign `torch.backends.cudnn.deterministic = True`;
7. assign `torch.backends.cudnn.benchmark = False`;
8. assign `os.environ['PYTHONHASHSEED'] = str(seed)`;
9. assign `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'`;
10. return `None`.

A failure stops at that exact stage and retains all earlier effects.  CUDA
backend fields are assigned even when `is_available()` returns false.  Seed
type/range validation is delegated sequentially to the three backends, so one
backend may accept a value that the next rejects.

Changing `PYTHONHASHSEED` in a running process does not reseed that process's
already-selected hash secret; the observable immediate effect is the
environment mapping (and therefore inheritance by later child processes).
The function mutates process-global RNG/backend/environment state and is not
safe to race with generation, hashing-dependent work, another seed call, or
unrelated Torch work.

The non-Torch C++ phase must not expose a function named `set_seed` that
silently seeds fewer domains.  Use an explicitly named native RNG/context API
until an API review decides how downstream non-ML callers express their seed.
Full Torch/CUDA state parity remains outside this phase.

### `generate_data_with_distribution`

Before reading keyword parameters or consuming RNG state, Python evaluates two
message-less assertions in order:

1. `distribution` is one of `uniform`, `normal`, `exponential`, `poisson`;
2. `dtype` is one of `int`, `float`, `bool`.

With ordinary (non-`-O`) Python, the first invalid discriminant raises
`AssertionError` and the NumPy RNG state is unchanged.  With `python -O`, the
assertions disappear: an unsupported distribution reaches the misspelled
`NotImplementedError` message containing `unsupporrted`, while some unsupported
dtypes can flow into NumPy `astype`.  Canonical testing uses assertions
enabled, with two optimized-interpreter boundary cases recorded separately.

Dispatch behavior is:

- `normal`: read `loc = kwargs.get('loc', 0.0)` then
  `scale = kwargs.get('scale', 1.0)` and call
  `np.random.normal(loc, scale, size)`.
- `uniform`: read `low` then `high` via `.get`, without defaults.  For
  `dtype == 'int'`, evaluate `high + 1` and call
  `np.random.randint(low, high + 1, size)`, making the configured high bound
  inclusive.  For `dtype == 'float'`, call
  `np.random.uniform(low, high, size)`.  For the accepted dtype `bool`, neither
  branch initializes `data`; the final access raises `UnboundLocalError`
  without consuming RNG state.  This bug is locked.
- `exponential`: read `scale = kwargs.get('scale')` with no default and call
  `np.random.exponential(scale, size)`.
- `poisson`: read `lam = kwargs.get('lam')`; if
  `kwargs.get('reciprocal', False)` is truthy, evaluate `1 / lam` before the
  NumPy call; then call `np.random.poisson(lam, size)`.

After a successful generator call, evaluation is always
`data.astype(dtype).tolist()`.  A cast failure occurs after the distribution
has already consumed RNG values.  Under the pinned 64-bit oracle, `int` means
NumPy's native integer dtype, `float` means float64, and `bool` becomes Python
booleans after `tolist`; return elements are Python native scalars.  NumPy also
accepts runtime sizes/shapes beyond the `int` annotation, but the proposed C++
boundary intentionally uses a one-dimensional count.

Missing/invalid parameters, negative size, divide-by-zero, overflow, NumPy
warnings/casts, and backend exception text remain NumPy-version sensitive.
Exact differential cases must pin NumPy 2.2.6 and compare every element/bit for
ordinary buffers; large buffers use byte count plus FNV checksum. They also
compare exception stage and pre/post global RNG continuation—not merely
statistics.

### `get_distribution_average`

The body is only `pass`.  After normal Python argument binding and caller-side
argument evaluation, every accepted call returns `None` without inspecting
`self`, `distribution`, `dtype`, or `kwargs`.  The seemingly method-like
`self` is an ordinary required positional parameter.  Missing arguments still
raise Python's binding `TypeError` before entering the function.

Do not invent an average formula under this compatibility name.  A useful
average calculation would be a separately named API with its own contract.

### `generate_file_name`

Observable order is:

1. if `config` is not an actual `dict` (dict subclasses count), replace it
   with `vars(config)`;
2. evaluate `extra_items + ['p_net_num_nodes', 'reusable']` and store the new
   value in local `items`;
3. never read `items` again;
4. read `config['solver_name']` and format
   `<solver>-records-<epoch_id>-`;
5. iterate `kwargs.items()` in insertion order and join each exact
   `f'{key}={value}'` with `-`;
6. append `.csv` and return.

With no kwargs the trailing separator remains, for example
`solver-records-0-.csv`.  Keys/values are formatted with ordinary Python
string conversion and are not escaped, so `-`, `=`, path separators, NULs,
newlines, and visually ambiguous values remain literal in the result.

The default `extra_items=[]` is one shared mutable Python object, but this
function does not mutate it.  The otherwise useless addition is observable:
an incompatible object or custom `__add__` can fail or cause side effects
before `solver_name` is read.  The typed C++ boundary omits this unused
parameter and allocation; that is a documented representation boundary, not a
reason to retain a hot-path no-op.

### `get_distribution_parameters`

The function calls `distribution_dict.get('distribution', None)` once.

- missing/`None` discriminator: return a new empty list;
- `exponential`: return `[distribution_dict['scale']]`;
- `poisson`: return `[distribution_dict['lam']]`;
- `uniform`: return `[distribution_dict['low'], distribution_dict['high']]`;
- `customized`: return `[distribution_dict['min'], distribution_dict['max']]`.

Any other non-`None` value—including `normal`, although normal generation is
supported—falls through with local `parameters` uninitialized and raises
`UnboundLocalError` at the return.  Missing required fields raise `KeyError` in
the listed access order.  The returned list retains the original parameter
objects; no numeric conversion or validation occurs.

### `get_parameters_string`

Python calls `len(parameters)` once for the empty check and, when non-empty,
again for the one-element check.  Results are:

- length zero: literal `None`;
- length one: `str(parameters[0])`;
- length greater than one: iterate the object, convert each value with `str`,
  join with `-`, and surround with square brackets.

There is no escaping or type restriction.  Custom `__len__`, indexing,
iteration, or `__str__` exceptions/side effects occur in that order.  Examples
include `[1-2]` for two numbers and `[a-b]` for the string `ab`.

### Physical-network dataset directory

`get_p_net_dataset_dir_from_setting` performs these steps in order:

1. read `p_net_setting['output']['save_dir']`;
2. scan every node attribute and read each `['name']` into an unused list;
3. scan every link attribute and read each `['name']` into another unused list;
4. repeatedly access the topology mapping to test whether `file_path` exists,
   is not `''`, `None`, or `'None'`, and satisfies `os.path.exists`;
5. if that test succeeds, use
   `os.path.basename(file_path).split('.')[0]` as the topology name; this uses
   the text before the **first** dot, not `pathlib`/filesystem `stem`;
6. otherwise format
   `<num_nodes>-<type>_[<wm_alpha>-<wm_beta>]` in that field order;
7. rescan node attributes, formatting each as
   `<name>_<get_parameters_string(get_distribution_parameters(spec))>` and
   joining with `-`;
8. do the same for link attributes;
9. concatenate topology, node, and link portions with two unconditional `-`
   separators;
10. append `-seed_<seed>` whenever `seed is not None` (zero and false count);
11. return `os.path.join(save_dir, middle)`.

The two initial attribute-name scans are unused but can raise or trigger custom
mapping side effects before the topology test; names are then read again.
Plain typed C++ settings deliberately eliminate the redundant scans.  Empty
attribute lists retain empty delimited portions.  `os.path.exists` converts
many stat failures to false, so those cases take the generated-topology branch.
Native separator, drive, absolute-component, path-like, symlink, and race
behavior belongs to `os.path` and must be profiled on the canonical platform.

### Virtual-network dataset directory

`get_v_nets_dataset_dir_from_setting` performs:

1. read `v_sim_setting['output']['save_dir']`;
2. build the node attribute string with the same helper nesting as above;
3. build the link attribute string;
4. format, in order,
   `<num_v_nets>-[<v_net_size.low>-<v_net_size.high>]-`
   `<topology.type>-<lifetime parameters>-<arrival_rate.lam>-`
   `<node attrs>-<link attrs>`;
5. append the same non-`None` seed suffix;
6. return `os.path.join(save_dir, middle)`.

Arrival rate uses only its direct `lam`; it does not call
`get_distribution_parameters`.  Lifetime does call the helper, so a lifetime
distribution of `normal` triggers the locked `UnboundLocalError`.  Empty
attribute collections retain trailing/consecutive separators.  All mapping
lookups, helper errors, and string conversions occur before the final join.

### `preprocess_xml`

At call time the function lazily imports NetworkX and minidom, then:

1. calls `minidom.parse(xml_source_fpath)`;
2. obtains all descendant `node` and `link` elements in document order;
3. fully extracts every node before extracting any edge. The XML node `id`
   attribute becomes the string-valued `label`; a missing `id` is an extraction
   error while explicit `id=""` is valid. Graph node IDs are dense enumeration
   indices `0..N-1`; `x` and `y` come from the first matching descendant and
   that element's first child's `.data`;
4. builds `label2id`; duplicate labels resolve to the last node ID;
5. fully extracts every edge in document order. The XML link `id` attribute
   becomes its string-valued `label`, with the same missing-versus-empty rule.
   Source/target labels use dictionary `.get`, so unknown labels become `None`
   until graph insertion;
6. read two `capacity` elements as `capacity_st` and `capacity_ts`;
7. **read those same two capacity elements again** as `cost_st` and `cost_ts`;
   no `cost` XML element is consulted.  This bug is locked;
8. only after all XML records were extracted, add all nodes and then all edges
   to one undirected simple `nx.Graph`;
9. set `G.graph['name'] = topylogy_name`;
10. stringify `gml_target_fpath` with an f-string and call `nx.write_gml`;
11. return that exact graph object only after the write succeeds.

Node coordinates and all edge fields remain strings.  Self-loops are allowed.
Parallel, reversed, or duplicate edges collapse according to simple-Graph
insertion/update order.  A `None` endpoint is rejected by the pinned NetworkX
at edge insertion; earlier edges may have mutated the local graph, but no graph
is returned and GML writing has not begun.

Missing attributes/tags, empty elements, malformed XML, file-like versus path
inputs, entity/parser behavior, unsupported GML values, and output failures
propagate from minidom/NetworkX.  Extraction failures occur before graph
mutation and before target writing.  A write failure occurs after graph
construction and may leave a created/truncated/partial target according to
NetworkX/filesystem behavior.  Exact GML bytes are version and platform
sensitive; the oracle pins NetworkX 3.4.2 and uses temporary targets.

## Torch-free oracle loading

Importing `virne` or `virne.utils` is forbidden: those eager paths reach solver
learning modules as well as this module's top-level Torch import.  The oracle
must:

1. verify the exact SHA-256 above before execution;
2. load only this file under a private module name;
3. install a controlled fake `torch` module in `sys.modules` before execution,
   with call-recording `manual_seed`, CUDA availability/manual-seed methods,
   and mutable cuDNN fields; assert that real Torch was never imported;
4. install a minimal fake `omegaconf` exposing sentinel `DictConfig` and
   `OmegaConf`, because both imports are unused by all functions;
5. use the pinned real NumPy and, only for XML cases, pinned NetworkX;
6. restore every replaced `sys.modules` entry after loading/testing.

For non-seed cases the Torch fake rejects any attribute access, proving that
the selected function stayed non-Torch.  Seed-contract tests use the recording
fake to verify exact call/assignment order, CUDA true/false branches, and staged
failures; they do **not** claim numerical Torch RNG parity.

Every case runs in an isolated process or snapshots/restores
`random.getstate()`, `np.random.get_state()`, both environment variables, and
all fake-backend fields.  XML/GML cases use only verified temporary paths.
This approach preserves the original module execution/export behavior better
than AST-copying function bodies and requires no ML package in the non-ML
oracle image.

## Non-Torch split and implementation order

Do not implement the file as one monolithic target merely because Python put
these functions together.

| Order | Proposed leaf | Contents | Dependencies / reason |
|---:|---|---|---|
| 1 | `vne_utils_dataset_core` | parameter helpers, locked average stub, filename builder | **COMPLETE**; C++17 standard library only |
| 2 | same core target | typed physical/virtual directory builders and deterministic batches | **COMPLETE**; filesystem + Threads, no graph/RNG/Torch |
| 3 | `vne_utils_dataset_rng` | distribution generation | **COMPLETE**; frozen `random_lib`, exact NumPy 2.2.6 stream/casts, parallel post-draw transform only |
| 4 | `vne_utils_dataset_xml` | XML record extraction, graph materialization, GML write | **COMPLETE**; Boost 1.85 RapidXML, frozen graph adapter, exact dataset-local NetworkX 3.4.2 writer |
| 5 | `vne_utils_dataset` aggregate | links the accepted non-Torch leaves | **COMPLETE**; interface-only, no implementation or global state |
| deferred | explicitly named seed adapter | native non-ML RNG context seeding | must not masquerade as full Python/Torch `set_seed` |

All non-Torch leaves and their interface-only aggregate are complete. Never
modify frozen `graph/` merely to make dataset serialization resemble NetworkX.

### RNG dependency decision (2026-07-28)

The required mechanism already exists; do not reimplement or vendor another
MT/distribution stack. `random/README.md` is the canonical API contract for
`random_lib::NumpyRandomState`. It provides scalar/vector legacy MT19937
`randint`, `uniform`, `normal`, `exponential`, and `poisson`, locks continuation
state, and already carries the accepted NumPy license notices and GCC 11.4
bulk-output optimization. The completed dataset RNG leaf links `random_lib`,
accepts one explicit stream, and adds only typed dataset dispatch plus NumPy
`astype(...).tolist()` conversion semantics. The existing NumPy-1.26 contract
was reverified against the pinned NumPy 2.2.6 oracle. No `random/` source or
completed random test was changed or rebuilt from scratch for the wrapper.

### Stable RNG API (completed 2026-07-28)

This section is the API source of truth for future callers. Read it instead of
opening `dataset_rng.cpp`; inspect implementation only for an unresolved exact
differential or a measured low-level optimization.

```cpp
enum class DatasetValueKind : std::uint8_t {
    integer,
    floating,
    boolean,
};

struct DistributionRequest {
    std::size_t count = 0;
    DatasetValueKind value_kind = DatasetValueKind::floating;
    DistributionSpec distribution;
};

struct GeneratedData {
    DatasetValueKind value_kind = DatasetValueKind::floating;
    std::variant<
        std::vector<std::int64_t>,
        std::vector<double>,
        std::vector<std::uint8_t>> values;
};

DatasetValueKind dataset_value_kind_from_string(std::string_view value);

GeneratedData generate_data_with_distribution(
    const DistributionRequest& request,
    NumpyRandomState& rng,
    std::size_t cast_workers = 0);
```

`dataset_value_kind_from_string` accepts exactly `"int"`, `"float"`, and
`"bool"`, resolves once, and returns the enum used by all subsequent loops.
`DistributionSpec::kind` must be `uniform`, `normal`, `exponential`, or
`poisson`; `none`, `customized`, and invalid enum values fail before a draw.
The result owns one contiguous typed vector. Boolean values use a dense
`uint8_t` lane containing only zero or one; callers must check `value_kind`
before selecting the variant alternative.

Parameter contract:

- normal uses `loc=0` and `scale=1` only when the option is disengaged; an
  explicitly present `std::monostate` is invalid rather than a default;
- uniform requires `low` and `high`; exponential requires `scale`; poisson
  requires `lambda`; both a disengaged option and present `monostate` count as
  missing for these required fields;
- numeric floating parameters accept `double`, `int64_t`, and `bool` with the
  same conversion used by the Python boundary; integer-uniform bounds accept
  only `int64_t` and `bool`;
- integer uniform treats the public `high` as inclusive. The
  `high == INT64_MAX` case uses the documented raw MT word hook and exact
  masked rejection, including the complete signed-int64 domain, because the
  frozen signed RNG API cannot represent the exclusive Python bound `2^63`;
- poisson applies `1 / lambda` before generation when `reciprocal` is true and
  reports a typed zero-division parameter failure before a draw;
- uniform+boolean deliberately reports
  `uniform_boolean_uninitialized` before reading either bound and consumes no
  RNG state, matching the original Python `UnboundLocalError` stage.

Cast contract matches the NumPy 2.2.6 payload exactly. Floating-to-int64
truncates toward zero; NaN, infinities, and values outside the representable
half-open range become `INT64_MIN`. Python emits seven locked `RuntimeWarning`
cases while producing those values; C++ produces the identical bytes without
a process-global warning side effect. Boolean conversion is `value != 0`, so
NaN is true and both signed zeros are false. Integer-to-double uses the native
binary64 cast. The normal interpreter contract is authoritative. Two separate
`python -O` tests record that disabled assertions can move invalid-distribution
or invalid-dtype failures across a draw; C++ intentionally does not emulate
that optimization-only control flow.

One caller-owned `NumpyRandomState` is the only RNG owner. Every draw remains
sequential and in NumPy order. Worker threads can touch only an already-owned
contiguous buffer; they never call, copy, seed, or lock the RNG. Exponential
integer/boolean generation uses `rng.random(count)` followed by the exact
`scale * -log(1-u)` transform fused with the final cast. Element-by-element
tests against the frozen `rng.exponential` path and the following RNG
continuation lock this optimization bit-for-bit.

`cast_workers=1` is sequential, values above one are explicit, and zero selects
the measured automatic policy. Widths are bounded by item count and Linux
process CPU affinity (`hardware_concurrency` fallback elsewhere):

| Work | Count | Automatic width |
|---|---:|---:|
| every native result and every non-exponential cast | all | 1 |
| exponential to int64 | `< 131,072` / `131,072..262,143` / `>= 262,144` | 1 / 3 / 7 |
| exponential to bool | `< 131,072` / `>= 131,072` | 1 / 7 |

A definite one-lane selection returns before querying CPU affinity; affinity is
needed only when an explicit or automatic width can exceed one.

The caller handles block zero and creates only `width-1` threads. Construction
and join overhead is inside public timing. If thread construction fails, all
already-started threads are joined before the exception propagates. Generation
has already advanced the caller's stream at that point, just as a Python cast
failure can occur after generation; no partially filled result is returned.
`std::bad_alloc` is preserved, while ordinary frozen-RNG backend failures map
to `DatasetException{rng_backend_failure, generate_values}`.

### Stable XML/GML API and dependency decision (completed 2026-07-28)

State: **COMPLETE** on 2026-07-28. Future callers must use this section and may
not reopen the XML or graph implementation merely to rediscover its surface.

No XML dependency is added or installed. The implementation reuses the pinned
workspace Boost 1.85 payload and parses one mutable file buffer in situ through
its bundled RapidXML adapter. Direct RapidXML avoids the second allocation tree
created by `boost::property_tree::read_xml`; because its header is under
`boost/property_tree/detail`, this is an explicit Boost-1.85 implementation pin
that must be retested with any Boost upgrade. The supported SNDlib input
encodings are UTF-8 and ISO-8859-1; other declared encodings are a typed native
boundary rejection. Parser/schema differential cases cover declaration/BOM,
entities, comments/CDATA, whitespace, malformed closing tags, and both accepted
encodings.

The stable public declarations belong in `virne/utils/dataset_xml.h`:

```cpp
struct XmlNodeRecord {
    std::string label;
    std::string x;
    std::string y;
};

struct XmlEdgeRecord {
    std::string label;
    std::string source_label;
    std::string target_label;
    std::string capacity_st;
    std::string capacity_ts;
    std::string cost_st;
    std::string cost_ts;
};

struct ParsedXmlTopology {
    std::vector<XmlNodeRecord> nodes;
    std::vector<XmlEdgeRecord> edges;
};

struct XmlTopologyRequest {
    std::string topology_name;
    std::filesystem::path xml_source_path;
    std::filesystem::path gml_target_path;
};

ParsedXmlTopology parse_sndlib_xml(const std::filesystem::path& source_path);

std::vector<ParsedXmlTopology> parse_sndlib_xml_batch(
    const std::vector<std::filesystem::path>& source_paths,
    std::size_t workers = 0);

Graph materialize_xml_topology(
    std::string_view topology_name,
    const ParsedXmlTopology& topology);

Graph preprocess_xml(const XmlTopologyRequest& request);
```

The parser scans XML element names only at the format boundary and stores known
schema values in direct record fields. It reproduces minidom's descendant
document order and literal first-child `.data` extraction: text, CDATA,
comment, and processing-instruction children all supply their own data and none
is skipped. Missing node/link `id` attributes are schema failures;
explicit empty IDs remain valid. It parses every node and edge record before
graph mutation, keeps values as strings, maps duplicate labels to the last
dense node ID, and deliberately copies the first two capacity values into both
capacity and cost fields. Actual XML `cost` elements remain ignored.

Materialization builds one frozen undirected simple `Graph`. It interns graph
attributes `name`, node attributes `label`/`x`/`y`, and edge attributes
`label`, `source_label`, `target_label`, `capacity_st`, `capacity_ts`,
`cost_st`, and `cost_ts` once before the corresponding loops. Loops then use
only direct record fields, dense `Vertex`, resolved `AttrId`, and
`AttrMap::set(AttrId, ...)`. The dynamic label table is reserved once and each
edge performs exactly one source and one target lookup before carrying native
IDs. Duplicate/reversed edges update the existing simple edge in input order;
self-loops remain supported.

`preprocess_xml` performs parse, materialization, graph `name` assignment, and
then the completed dataset-local NetworkX 3.4.2-compatible GML serializer. It
returns the graph only after a successful write. The frozen generic graph
writer was measured and found not byte-identical, so it remains untouched.
The byte gate is freshly generated NetworkX 3.4.2 output on Linux, not the
checkout's CRLF-normalized `Brain.gml`; exact attribute order, escaping,
indentation, and LF bytes are mandatory.

Errors map to the existing typed stages: file/XML syntax, invalid XML
characters/entities, and unsupported encoding use
`xml_parse_failure/parse_xml`; missing required attributes/descendants or first
data children use `xml_schema_failure/parse_xml`; a missing label lookup uses
`unknown_endpoint/materialize_graph`; other graph failures use
`graph_materialization_failure/materialize_graph`; writer failures use
`gml_write_failure/write_gml`. `std::bad_alloc` propagates unchanged.

Four typed safety boundaries intentionally differ from permissive or
side-effect-prone NetworkX behavior: only UTF-8 and ISO-8859-1 XML are accepted;
custom DTD entities are rejected; fixed GML keys `source`/`target` cannot be
aliased by input attributes; and `.gz`/`.bz2` targets are rejected rather than
silently introducing a compression dependency. These rejections occur before
opening the target, so an existing file stays unchanged.

`parse_sndlib_xml_batch` is the only worker extension. Documents are
independent and read-only; results/errors occupy pre-sized input-order slots,
and the lowest input-index failure is rethrown. If construction of a worker
thread throws, every already-created thread is joined before propagation.
Explicit widths are capped by document count and Linux affinity.

Worker zero uses the measured policy below. `representative bytes` is the first
input file's size, read once before worker loops; no path or string lookup is
performed in a parse hot loop.

| Documents | Representative bytes | Automatic lanes |
|---:|---:|---:|
| 0 | any | 0 |
| 1 | any | 1 |
| 2 | any | 2 |
| 3..7 | `< 196,608` / `>= 196,608` | 2 / 4 |
| 8 | `< 196,608` / `>= 196,608` | 3 / 4 |
| 9..23 | `< 196,608` / `>= 196,608` | 4 / 8 |
| 24 or more | any | 5 |

The selected width is still affinity- and count-capped. `preprocess_xml`
itself has no worker parameter because parallel target writes would change
observable side-effect/error order.

## Typed C++ surface

### Implemented core API (stable)

The stable declarations below are the integration contract. Ordinary callers
must use this section rather than reopen `dataset.h`/`.cpp`.

```cpp
enum class DistributionKind : std::uint8_t {
    none, uniform, normal, exponential, poisson, customized,
};
enum class DatasetValueKind : std::uint8_t {
    integer, floating, boolean,
};
enum class DatasetTopologyKind : std::uint8_t {
    path, star, grid_2d, waxman, random,
};
enum class DatasetErrorCode : std::uint8_t {
    invalid_distribution, invalid_value_kind, invalid_topology,
    missing_parameter, invalid_parameter, uniform_boolean_uninitialized,
    unsupported_parameter_distribution, rng_backend_failure,
    xml_parse_failure, xml_schema_failure, unknown_endpoint,
    graph_materialization_failure, gml_write_failure,
};
enum class DatasetOperation : std::uint8_t {
    resolve_distribution, resolve_topology, generate_values, cast_values,
    format_parameters, format_file_name, build_physical_path,
    build_virtual_path, parse_xml, materialize_graph, write_gml,
};

using DatasetAttrId = std::uint32_t;
using DatasetScalar = std::variant<
    std::monostate, std::int64_t, double, bool, std::string>;

struct DistributionSpec {
    DistributionKind kind = DistributionKind::none;
    std::optional<DatasetScalar> low, high, loc, scale, lambda;
    std::optional<DatasetScalar> minimum, maximum;
    bool reciprocal = false;
};
struct DatasetAttributeSpec {
    DatasetAttrId id = 0;
    std::string name;
    DistributionSpec distribution;
};
struct DatasetFileNameConfig { std::string solver_name; };
struct OrderedFileNameItem { std::string key; DatasetScalar value; };
struct DatasetFileNameRequest {
    DatasetFileNameConfig config;
    std::int64_t epoch_id = 0;
    std::vector<OrderedFileNameItem> ordered_items;
};
struct PhysicalTopologyDatasetSpec {
    std::optional<std::filesystem::path> file_path;
    std::int64_t num_nodes = 0;
    DatasetTopologyKind topology_type = DatasetTopologyKind::waxman;
    DatasetScalar wm_alpha = 0.5, wm_beta = 0.2;
};
struct PhysicalDatasetSetting {
    std::filesystem::path save_dir;
    PhysicalTopologyDatasetSpec topology;
    std::vector<DatasetAttributeSpec> node_attributes, link_attributes;
};
struct VirtualDatasetSetting {
    std::filesystem::path save_dir;
    std::int64_t num_virtual_networks = 0;
    std::int64_t size_low = 0, size_high = 0;
    DatasetTopologyKind topology_type = DatasetTopologyKind::random;
    DistributionSpec lifetime;
    DatasetScalar arrival_lambda = 0.0;
    std::vector<DatasetAttributeSpec> node_attributes, link_attributes;
};
struct PhysicalDatasetPathRequest {
    PhysicalDatasetSetting setting;
    std::optional<DatasetScalar> seed;
};
struct VirtualDatasetPathRequest {
    VirtualDatasetSetting setting;
    std::optional<DatasetScalar> seed;
};

class DatasetException : public std::runtime_error {
public:
    DatasetException(DatasetErrorCode, DatasetOperation, std::string,
        std::optional<std::size_t> input_index = std::nullopt,
        std::filesystem::path path = {});
    DatasetErrorCode code() const noexcept;
    DatasetOperation operation() const noexcept;
    const std::optional<std::size_t>& input_index() const noexcept;
    const std::filesystem::path& path() const noexcept;
};

DistributionKind distribution_kind_from_string(std::string_view);
DatasetTopologyKind dataset_topology_kind_from_string(std::string_view);
std::string_view dataset_topology_kind_name(DatasetTopologyKind);
std::string format_dataset_scalar(const DatasetScalar&);
std::vector<DatasetScalar> get_distribution_parameters(const DistributionSpec&);
std::string get_parameters_string(const std::vector<DatasetScalar>&);
std::optional<double> get_distribution_average(
    const DistributionSpec&, DatasetValueKind) noexcept;
std::string generate_file_name(const DatasetFileNameConfig&, std::int64_t,
    const std::vector<OrderedFileNameItem>&);
std::filesystem::path get_p_net_dataset_dir_from_setting(
    const PhysicalDatasetSetting&,
    const std::optional<DatasetScalar>& seed = std::nullopt);
std::filesystem::path get_v_nets_dataset_dir_from_setting(
    const VirtualDatasetSetting&,
    const std::optional<DatasetScalar>& seed = std::nullopt);
std::vector<std::string> generate_file_names_batch(
    const std::vector<DatasetFileNameRequest>&, std::size_t workers = 0);
std::vector<std::filesystem::path> get_p_net_dataset_dirs_batch(
    const std::vector<PhysicalDatasetPathRequest>&, std::size_t workers = 0);
std::vector<std::filesystem::path> get_v_nets_dataset_dirs_batch(
    const std::vector<VirtualDatasetPathRequest>&, std::size_t workers = 0);
```

`DatasetTopologyKind` deliberately keeps this leaf free of graph/topology
linkage. A future config/network boundary converts canonical `TopologyType`
once. Scalar variants retain original integer/float/bool spelling so path and
filename bytes do not drift (`500` must not become `500.0`). A present optional
seed containing `false` is distinct from an absent seed.

### Historical pre-port RNG/XML sketch (obsolete)

The older sketch below remains solely as design history. Its RNG and XML names
are obsolete and must not be used; the completed stable APIs above and public
dataset headers always win.

All names should live in `namespace virne::utils`.  Exact spellings may change
in API review, but any change must update this document before code.

```cpp
enum class DistributionKind : std::uint8_t {
    none,
    uniform,
    normal,
    exponential,
    poisson,
    customized,
};

enum class DatasetValueKind : std::uint8_t {
    integer,
    floating,
    boolean,
};

enum class DatasetErrorCode : std::uint8_t {
    invalid_distribution,
    invalid_value_kind,
    invalid_topology,
    missing_parameter,
    invalid_parameter,
    uniform_boolean_uninitialized,
    unsupported_parameter_distribution,
    rng_backend_failure,
    xml_parse_failure,
    xml_schema_failure,
    unknown_endpoint,
    graph_materialization_failure,
    gml_write_failure,
};

enum class DatasetOperation : std::uint8_t {
    resolve_distribution,
    resolve_topology,
    generate_values,
    cast_values,
    format_parameters,
    format_file_name,
    build_physical_path,
    build_virtual_path,
    parse_xml,
    materialize_graph,
    write_gml,
};

using DatasetAttrId = std::uint32_t;

using DatasetScalar = std::variant<
    std::monostate, std::int64_t, double, bool, std::string>;

struct DistributionSpec {
    DistributionKind kind = DistributionKind::none;
    std::optional<DatasetScalar> low;
    std::optional<DatasetScalar> high;
    std::optional<DatasetScalar> loc;
    std::optional<DatasetScalar> scale;
    std::optional<DatasetScalar> lambda;
    std::optional<DatasetScalar> minimum;
    std::optional<DatasetScalar> maximum;
    bool reciprocal = false;
};

struct DistributionRequest {
    std::size_t count = 0;
    DatasetValueKind value_kind;
    DistributionSpec distribution;
};

struct GeneratedData {
    DatasetValueKind value_kind;
    std::variant<
        std::vector<std::int64_t>,
        std::vector<double>,
        std::vector<std::uint8_t>> values;
};

struct DatasetAttributeSpec {
    DatasetAttrId id;          // converted from the future canonical AttrId
    std::string name;          // canonical dynamic output spelling, owned once
    DistributionSpec distribution;
};

struct DatasetFileNameConfig {
    std::string solver_name;
};

struct OrderedFileNameItem {
    std::string key;
    DatasetScalar value;
};

struct PhysicalTopologyDatasetSpec {
    std::optional<std::filesystem::path> file_path;
    std::int64_t num_nodes;
    DatasetTopologyKind topology_type;
    DatasetScalar wm_alpha;
    DatasetScalar wm_beta;
};

struct PhysicalDatasetSetting {
    std::filesystem::path save_dir;
    PhysicalTopologyDatasetSpec topology;
    std::vector<DatasetAttributeSpec> node_attributes;
    std::vector<DatasetAttributeSpec> link_attributes;
};

struct VirtualDatasetSetting {
    std::filesystem::path save_dir;
    std::int64_t num_virtual_networks;
    std::int64_t size_low;
    std::int64_t size_high;
    DatasetTopologyKind topology_type;
    DistributionSpec lifetime;
    DatasetScalar arrival_lambda;
    std::vector<DatasetAttributeSpec> node_attributes;
    std::vector<DatasetAttributeSpec> link_attributes;
};

struct XmlTopologyRequest {
    std::string topology_name;
    std::filesystem::path xml_source_path;
    std::filesystem::path gml_target_path;
};

class NumpyRandomState;
class DatasetGraph;

class DatasetException : public std::runtime_error {
public:
    DatasetErrorCode code() const noexcept;
    DatasetOperation operation() const noexcept;
    std::optional<std::size_t> input_index() const noexcept;
    const std::filesystem::path& path() const noexcept;
};

std::vector<DatasetScalar> get_distribution_parameters(
    const DistributionSpec& distribution);
std::string get_parameters_string(
    const std::vector<DatasetScalar>& parameters);
std::optional<double> get_distribution_average(
    const DistributionSpec& distribution,
    DatasetValueKind value_kind) noexcept; // locked empty result

std::string generate_file_name(
    const DatasetFileNameConfig& config,
    std::int64_t epoch_id,
    const std::vector<OrderedFileNameItem>& ordered_items);

std::filesystem::path get_p_net_dataset_dir_from_setting(
    const PhysicalDatasetSetting& setting,
    const std::optional<DatasetScalar>& seed = std::nullopt);
std::filesystem::path get_v_nets_dataset_dir_from_setting(
    const VirtualDatasetSetting& setting,
    const std::optional<DatasetScalar>& seed = std::nullopt);

GeneratedData generate_data_with_distribution(
    const DistributionRequest& request,
    NumpyRandomState& rng,
    std::size_t cast_workers = 0);

DatasetGraph preprocess_xml(const XmlTopologyRequest& request);
```

`PhysicalDatasetSetting`, `VirtualDatasetSetting`, and
`XmlTopologyRequest` must be direct-field structs finalized with their owning
components; they must not be generic `SettingObject`/string maps. The stable
seed boundary already uses `optional<DatasetScalar>`. `std::uint8_t` is an
internal dense boolean lane, not permission to expose integers as booleans.

Python accepts arbitrary objects and reflection in several APIs.  The C++
surface deliberately restricts values to documented scalars, ordered items,
native paths, and typed settings.  Python-only argument-binding/reflection
cases remain oracle boundary tests rather than dynamic C++ maps.

## Fixed fields, enums, IDs, and hot loops

The completed RNG leaf and future XML leaf follow `PERFORMANCE_CONTRACT.md`:

- Resolve distribution and value strings exactly once to
  `DistributionKind`/`DatasetValueKind`; branch once before a sample loop.
  No sample performs a string comparison, registry lookup, or kwargs lookup.
- Every known setting key (`save_dir`, topology fields, size, lifetime,
  arrival rate, seed) is a direct member.  Do not carry Python dictionaries or
  repeatedly hash fixed keys through directory-name construction.
- Attribute names are genuinely dynamic output data.  Resolve each to `AttrId`
  once at the attribute boundary and carry both compact ID and one owned
  canonical spelling.  A one-pass formatter consumes the spelling; it does not
  look it up by name again.
- Precompute/reserve filename and path segments.  Preserve ordered dynamic
  filename items as a contiguous vector; never use an unordered map because
  kwargs insertion order is observable.
- XML tags and fixed graph attributes are compile-time fields.  Parse into
  contiguous typed `XmlNodeRecord`/`XmlEdgeRecord` vectors.  Resolve graph
  `AttrId`s for `label`, `x`, `y`, `source_label`, `target_label`, both
  capacities, both costs, and graph `name` once before graph loops.
- Dense node enumeration IDs become native `NodeId`.  The label-to-ID table is
  genuinely dynamic and may hash labels once per edge lookup; convert the
  result to `NodeId` immediately and never repeat label lookup during graph
  materialization/GML writing.
- Preserve duplicate-label last-wins behavior and the capacity-as-cost bug.
- Eliminate Python's two unused physical-attribute scans and unused
  `extra_items` allocation in typed C++; their odd custom-object side effects
  are explicitly outside the native representation.

## Ownership, global state, and parallel safety

- Core naming/path helpers own returned strings/paths and retain no setting
  references.  Caller-owned settings and ordered item vectors must outlive only
  the call.
- Dataset generation writes into an owned contiguous result. The shared RNG
  draw phase remains sequential; only the post-draw transform/cast phase can
  use deterministic contiguous worker blocks as specified by the stable API.
- `set_seed` is process-global and must never run concurrently with generation
  or another seed operation.  Adding a lock would not make outside Python,
  NumPy, Torch, CUDA, and environment users atomic.
- `preprocess_xml` owns its graph result.  Source and target may not alias in a
  way that overwrites input.  Tests write only below an explicit temporary
  root; no workspace, frozen fixture, home, or repository file may be a target.
- Filesystem existence and GML write races are not retried.  Completed side
  effects before an error remain observable.

Safe external parallel extensions are limited to independent inputs:

- core filename/path batches may use pre-sized result/error slots and select
  the lowest input-index error; only add them if a 1..8 sweep wins;
- shared-stream distribution generation exposes only `cast_workers`; it never
  parallelizes RNG draws. Explicit and automatic widths affect conversion of
  an already generated buffer only, preserve index order, and leave RNG
  continuation identical;
- XML batches require disjoint source/target paths and independent graphs,
  pre-sized slots, input-order results, and lowest-index error selection;
- seed operations are never batched or parallelized.

## Exact unit and differential matrix

All differential records include return type/value, stable exception stage,
exact output bytes where relevant, global RNG state/continuation, environment
changes, filesystem tree, graph fields, and target-file bytes/checksum.
Platform-specific raw exception messages are diagnostic unless listed above.

### Import and seed boundary

- exact SHA/size and private direct load;
- real Torch/OmegaConf absent; fake-module trace proves import isolation;
- `seed=None` performs zero calls and zero changes;
- CUDA unavailable/available call and assignment order;
- staged failure at random, NumPy, Torch CPU, CUDA probe, CUDA all-device seed,
  each cuDNN assignment, string conversion, and each environment assignment;
- previous environment missing/present/empty values and exact overwrite;
- Python/NumPy pre/post states and deterministic continuation;
- invalid seed types/ranges classified at the backend that rejects them;
- explicit proof that runtime `PYTHONHASHSEED` mutation does not change the
  current process hash result.

### Distribution generation

- both assertions individually and together, with unchanged RNG state;
- canonical Python and two `python -O` boundary cases;
- normal defaults and explicit loc/scale;
- integer uniform inclusive high, float uniform, and locked bool-uniform
  `UnboundLocalError` with no draw;
- exponential/poisson required parameters, reciprocal false/true, zero and
  missing lambda;
- sizes zero, one, large, negative, and Python-only multidimensional shape;
- int/float/bool casts, overflow/NaN/Inf/warning cases;
- errors before generation versus errors during final cast, with exact RNG
  continuation;
- fixed seeds across all supported rows with element-by-element/float-bit
  equality and output checksum.

### Pure helpers and filename

- average stub with ordinary, unusual, and missing arguments;
- every supported parameter discriminator, missing required fields, `None`,
  `normal`, and unknown discriminator with locked error;
- parameter lists of length 0/1/many, strings, booleans, `None`, special
  characters, and staged `len`/index/iteration/string failures;
- dict, dict subclass, object `vars`, missing `__dict__`, missing solver name;
- empty/ordered kwargs and values containing `-`, `=`, separators, NUL,
  newline, Unicode, NaN/Inf, and custom string conversion;
- default and explicit epoch; mutable default reflection boundary;
- invalid/custom `extra_items.__add__` side effects before solver lookup.

### Directory builders

- minimal and full physical/virtual typed settings;
- empty/multiple node and link attributes, exact separator placement/order;
- every distribution parameter form and the `normal` helper failure;
- existing/non-existing/empty/`None`/`'None'` topology file path;
- filenames with zero/one/multiple dots and hidden names, proving first-dot
  semantics rather than filesystem `stem`;
- seed `None`, zero, negative, false, and normal integer;
- missing fields and side-effecting mapping fixtures that lock Python access
  order/redundant physical scans as representation-boundary evidence;
- relative/absolute components, spaces, Unicode, symlink, stat failure, native
  drive/separator behavior, and filesystem race classification;
- exact returned native bytes/string, with no directory creation.

### XML/graph/GML

- valid minimal and multi-node/multi-edge XML;
- exact dense IDs and string-valued node/edge/graph attributes;
- duplicate labels (last wins), duplicate/reversed edges, self-loop;
- unknown source/target labels and edge insertion stage;
- malformed XML, missing ID/x/y/source/target, missing first/second capacity,
  empty first child, extra descendant tags, and ignored cost tags;
- explicit capacity-to-cost bug assertion;
- source path/file-object Python boundary, relative/Unicode paths;
- existing target, unwritable/missing target parent, writer failure and partial
  target classification;
- exact returned graph semantics and exact NetworkX 3.4.2 GML bytes/hash;
- C++ graph semantic checksum separately from byte compatibility so a required
  intentional GML-format deviation cannot be hidden.

C++-only core tests additionally cover all core-reachable enum/error branches,
scalar formatting parity, move/copy behavior of typed settings, lowest-index
batch failure, no string lookup inside formatting loops, safe temporary paths,
and sanitizer/stress execution. RNG tests now cover all reachable RNG enum/error
branches, exact fused conversion, worker invariance, affinity caps, and huge
explicit worker requests. XML tests cover every listed XML/GML category plus
worker invariance, affinity caps, sanitizers, stress, and concurrent processes.

## Benchmark and worker policy

Correctness/checksum gates run before timing.  Fixture creation, source loading,
seeding unless it is the measured operation, serialization for comparison,
graph/tree hashing, process startup, and cleanup remain outside timed regions.
Use five warm-ups and at least 31 alternating Python/C++ samples, reporting
median, MAD, p95, compiler/library/runtime versions, CPU affinity, and all
checksums.

| Row | Work | Parallel policy |
|---|---|---|
| parameter extraction/string | large ordered corpus of typed specs | sequential baseline; pure deterministic batch sweep 1..8 only if worthwhile |
| filename generation | large ordered item corpus | safe independent batch, pre-sized outputs/errors |
| physical/virtual path generation | corpus with filesystem hits/misses | independent batch only on disjoint immutable inputs; filesystem cache reported |
| uniform/normal/exponential/poisson generation | equal counts and exact stream continuation | draw sequential; measured post-draw cast workers 1..8 + auto |
| XML parse/materialize | fixed XML corpus, no GML write timed | independent-document batch may sweep 1..8 |
| full XML→GML | fresh target per sample | report sequential first; batch only disjoint targets |
| seed orchestration | fake-backend call trace | correctness/latency report only; never parallel |

For distribution rows, compare direct callable payload separately from wrapper,
allocation, and conversion cost.  A C++ speedup is meaningless unless every
value bit and the subsequent RNG continuation match.  If native RNG semantics
are intentionally different, benchmark under a different API name and never
present it as an exact differential win.

The core, RNG, and XML sweeps are complete. Core automatic mode respects
affinity and uses the measured family/size policy: filename is sequential below
16,384 items and
uses up to five workers above it; physical/virtual batches are sequential
below 1,024, use up to four workers below 3,072, and up to six thereafter.
Explicit widths 1..8 remain available. Batch results are pre-sized, ordered,
and surface the lowest failing input index. RNG draws remain sequential, while
the stable cast-only policy above uses 3/7 lanes only for measured fused
exponential conversions. Seed APIs remain **sequential only / worker not
applicable**. XML batch auto mode uses the documented count/file-size policy,
and all 1..8 widths plus auto preserve graph bytes, order, and lowest-index
failure. No global executor exists merely to accelerate tiny strings.

## Targets and completion order

Completed isolated targets are:

- `vne_utils_dataset_core`, `vne_dataset_core_unit`,
  `vne_dataset_core_harness`.
- `vne_utils_dataset_rng`, `vne_dataset_rng_unit`,
  `vne_dataset_rng_harness`.
- `vne_utils_dataset_xml`, `vne_dataset_xml_unit`,
  `vne_dataset_xml_harness`.
- interface-only `vne_utils_dataset`, which links the three accepted leaves and
  adds no implementation or global state.

All three implementation turns are complete. None adds Torch, attribute-model
code, network-model code, or seed orchestration.

Do not mark any leaf complete until its API review, isolated strict build,
complete exact differential, canonical checksum-gated timing, safe worker
decision, ASan/UBSan/stress, full CTest, and frozen-foundation verification all
pass, followed immediately by component/result/status documentation.

## Core completion record

Completed on 2026-07-28:

- isolated standard-library/Threads target and exact stable API: **PASS**;
- exact differential: **PASS 60/60**, controlled fake Torch/OmegaConf only;
- Python scalar formatting: **PASS 16,395/16,395 binary64 bit patterns**;
- canonical five-warm-up/31-sample timing: **PASS 32/32 rows**, every C++
  median faster than Python, with invariant checksums/output bytes;
- explicit workers 1..8 plus automatic family/size policy: **PASS**;
- strict warnings, ASan/UBSan/leaks, and 100 stress iterations: **PASS**;
- full Release build and CTest: **PASS 20/20**, including frozen integrity;
- permanent evidence: `porting/results/dataset_core_2026-07-28.md` and the two
  dated JSON artifacts.

The core leaf is **COMPLETE**. The RNG and XML records below are complete; only
the same-named Python/Torch `set_seed` facade remains deferred exactly as
described above.

## RNG completion record

Completed on 2026-07-28:

- isolated `random_lib`-reusing target and stable typed API: **PASS**;
- exact NumPy 2.2.6 differential: **PASS 92/92**, with element/bit equality for
  ordinary buffers, byte count plus FNV for large buffers, warning
  classification, draw count, and `random()+normal()` continuation;
- Python optimized-boundary characterization: **PASS 2/2**;
- canonical five-warm-up/31-sample timing: **PASS 60/60 rows**, every C++
  median faster than Python with invariant checksum, bytes, and continuation;
- interleaved/rotated/reversed 1..8+auto sweeps at 192,000 and 600,000 items:
  **PASS 18/18 at each size**;
- strict warnings, ASan/UBSan/leaks, 100 stress iterations, and affinity-limited
  runs at one, two, and eight CPUs: **PASS**;
- full Release build and CTest: **PASS 21/21**, including frozen integrity;
- permanent evidence: `porting/results/dataset_rng_2026-07-28.md` and five
  dated JSON artifacts.

The NumPy RNG leaf is **COMPLETE**. `random/` and the frozen
graph/CSV/config/yaml-cpp foundations remain read-only dependencies.

## XML/graph/GML completion record

Completed on 2026-07-28:

- isolated Boost-1.85/frozen-graph target and stable typed API: **PASS**;
- exact NetworkX 3.4.2 differential: **PASS 57/57**, covering 24 compatibility,
  18 typed-error, and 14 batch cases plus the real Brain fixture;
- Brain fixture: 161 nodes, 332 XML link records, 166 simple graph edges, and
  exact 45,085-byte LF GML SHA-256
  `0F30D85C53ECC94461FE7DD00D627F1F91492428266F37528642F29EA2965A3F`;
- canonical five-warm-up/31-sample timing: every C++ row beats Python, from
  27.806x for sequential single-Brain parse through 149.433x for automatic
  16-document Brain parsing; exact graph/GML checksums gate every sample;
- explicit workers 1..8 plus automatic family/size policy: **PASS**; the final
  2/4/8/16/32/64-document sweep passed both Brain and synthetic corpora, auto
  beat sequential in all 12 policy rows and stayed within 25% of best explicit;
- strict warnings, ASan/leaks, UBSan no-recover, TSan with ASLR disabled, 100
  stress iterations, concurrent-process stress, and one/two/eight-CPU affinity
  runs: **PASS**;
- full Release build and CTest: **PASS 22/22**, including frozen integrity;
- permanent evidence: `porting/results/dataset_xml_2026-07-28.md` and three
  dated JSON artifacts.

The XML/graph/GML leaf and interface-only non-Torch dataset aggregate are
**COMPLETE**. Future work proceeds to the attribute-model phase. The
same-named Python/Torch seed facade stays explicitly out of this non-ML phase.
