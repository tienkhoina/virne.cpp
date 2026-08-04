# Virne Python to C++ port status

Last verified: 2026-08-03 (Asia/Saigon).

This file is the entry point for continuing the port. Read it and the linked
component note before opening the original Python implementation.

## Fixed references

- Original Python checkout: sibling `../virne`, commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- C++ baseline commit: `5e77c4447aa23f9698dd9b628b173168e3407909`.
- Non-ML oracle image:
  `virne-python-oracle:py310-nonml`, image ID
  `sha256:74cd8fcb09c2703a5b1b5568b2cc18d27912f6b5e3da77d9740233f438d7cddf`.
- Oracle runtime: CPython 3.10.20, NetworkX 3.4.2, NumPy 2.2.6,
  PyYAML 6.0.1.
- Machine-visible Docker CPUs during the recorded benchmark: 8.

The original package is not installed in the oracle image. Leaf source files
are loaded directly from the read-only sibling checkout so eager imports cannot
pull Torch, PyG, learning solvers, or a different copy of Virne.

The frozen Linux/Python reference and the controlled same-compiler old/final
native A/B are published in
`porting/results/hot_path_old_vs_new_2026-07-31.md`. All exact output gates
match; cross-platform medians remain separate from the native A/B ratios.

## Completion rule

All ports also follow `porting/PERFORMANCE_CONTRACT.md`: fixed schema uses
typed fields/enums, dynamic strings resolve once at the boundary, and hot loops
use IDs such as `AttrId` rather than string lookup.

A component is `COMPLETE` only when all of the following exist and pass:

1. an isolated C++ target that does not depend on unported stubs;
2. C++ unit tests;
3. differential tests against the exact Python source at the fixed commit;
4. same-fixture runtime measurements with warm-up, repeated samples and output
   checksums;
5. deterministic multithreading where it improves measured runtime, or a
   documented sequential-only decision where order and side effects are public;
6. a component note under `porting/components/`.

## Current matrix

| Layer | Component | State | Evidence |
|---|---|---|---|
| Frozen foundation | `graph/` | FROZEN | `porting/FROZEN_COMPONENTS.md` |
| Frozen foundation | `csv/` | FROZEN | `porting/FROZEN_COMPONENTS.md` |
| Frozen foundation | `config/` + `libs/yaml-cpp/` | FROZEN | `porting/FROZEN_COMPONENTS.md` |
| Leaf utility | `virne.utils.network` | **COMPLETE** | `porting/components/utils_network.md` |
| Leaf utility | `class_dict` | **COMPLETE** | `porting/components/class_dict.md` |
| Leaf utility | `setting` | **COMPLETE** | `porting/components/setting.md` |
| Leaf utility | `stats` | **COMPLETE** | `porting/components/stats.md` |
| Leaf utility | `virtualize` | DEFERRED / VISUALIZATION-ONLY | `porting/components/virtualize.md`; eager Matplotlib demo, no core API |
| Leaf utility | `manager` | **COMPLETE** | `porting/components/manager.md`, `porting/results/manager_2026-07-28.md` |
| Leaf utility | `virne.utils.config` | **COMPLETE** | `porting/components/utils_config.md`, `porting/results/utils_config_2026-07-29.md` |
| Leaf utility | `dataset` non-Torch core | **COMPLETE** | `porting/components/dataset.md`, `porting/results/dataset_core_2026-07-28.md` |
| Leaf utility | `dataset` NumPy RNG | **COMPLETE** | `porting/components/dataset.md`, `porting/results/dataset_rng_2026-07-28.md` |
| Leaf utility | `dataset` XML/GML | **COMPLETE** | `porting/components/dataset.md`, `porting/results/dataset_xml_2026-07-28.md` |
| Leaf utility | non-Torch `dataset` aggregate | **COMPLETE** | interface-only `vne_utils_dataset`; Python/Torch seed facade remains out of scope |
| Topology | `topology_generator` | **COMPLETE** | `porting/components/topology_generator.md` |
| Topology | `topological_metric_calculator` | **COMPLETE** | `porting/components/topological_metric_calculator.md` |
| Attribute model | `attribute_method` typed-policy leaf | **COMPLETE** | `porting/components/attribute_method.md`, `porting/results/attribute_method_2026-07-28.md` |
| Attribute model | `BaseAttribute` | **COMPLETE** | `porting/components/base_attribute.md`, `porting/results/base_attribute_2026-07-28.md` |
| Attribute model | `NodeAttribute` | **COMPLETE** | `porting/components/node_attribute.md`, `porting/results/node_attribute_2026-07-28.md` |
| Attribute model | `LinkAttribute` | **COMPLETE** | `porting/components/link_attribute.md`, `porting/results/link_attribute_2026-07-28.md` |
| Attribute model | `GraphAttribute` | **COMPLETE** | `porting/components/graph_attribute.md`, `porting/results/graph_attribute_2026-07-28.md` |
| Attribute model | `AttributeBenchmarkManager` | **COMPLETE** | `porting/components/attribute_benchmark_manager.md`, `porting/results/attribute_benchmark_manager_2026-07-28.md` |
| Attribute model | `AttributeFactory` | **COMPLETE** | `porting/components/attribute_factory.md`, `porting/results/attribute_factory_2026-07-28.md` |
| Network model | `BaseNetwork` | **COMPLETE** | `porting/components/base_network.md`, `porting/results/base_network_2026-07-28.md` |
| Network model | `PhysicalNetwork` / `VirtualNetwork` | **COMPLETE** | `porting/results/physical_network_2026-07-28.md`, `porting/results/virtual_network_2026-07-28.md` |
| Network model | `VirtualNetworkEvent` | **COMPLETE** | `porting/components/virtual_network_event.md`, `porting/results/virtual_network_event_2026-07-29.md` |
| Network model | `VirtualNetworkRequestSimulator` | **COMPLETE** | `porting/components/virtual_network_request_simulator.md`, `porting/results/virtual_network_request_simulator_2026-07-29.md` |
| Network model | dataset `Generator` / changeable workload | **COMPLETE** | `porting/components/dataset_generator.md`, `porting/results/dataset_generator_2026-07-29.md` |
| Core model | `Solution` | **COMPLETE** | `porting/components/solution.md`, `porting/results/solution_2026-07-29.md` |
| Core/controller | `ConstraintChecker`, `ResourceUpdator`, `TopologyAnalyzer`, `NodeMapper`, `LinkMapper` | **COMPLETE** | frozen result notes under `porting/results/*_2026-07-29.md` |
| Core/controller | `Controller` non-solver lifecycle | **COMPLETE** | `porting/components/controller.md`, `porting/results/controller_2026-07-29.md` |
| Core runtime | `Counter` | **COMPLETE** | `porting/components/counter.md`, `porting/results/counter_2026-07-29.md` |
| Core runtime | `Recorder` | **COMPLETE** | non-ML typed API/result docs; RL only has an optional cold extension seam |
| Core runtime | `Logger` | **COMPLETE** | non-ML console/file/CSV; optional sinks remain cold and dependency-free |
| Core runtime | environments | **COMPLETE / FROZEN (NON-RL)** | `porting/components/environment.md`, `porting/results/environment_2026-07-29.md` |
| Non-ML solver | `solver.rank.LinkRank` | **COMPLETE / FROZEN** | `porting/components/link_rank.md`, `porting/results/link_rank_2026-07-29.md` |
| Non-ML solver | `solver.rank.NodeRank` | **COMPLETE / FROZEN** | `porting/components/node_rank.md`, `porting/results/node_rank_2026-07-29.md` |
| Non-ML solver | `solver.base_solver` | **COMPLETE / FROZEN** | `porting/components/base_solver.md`, `porting/results/base_solver_2026-07-29.md` |
| Non-ML solver | all eight classes in `solver.heuristic.node_rank` | **COMPLETE / FROZEN** | `porting/components/heuristic_node_rank.md`, order-only and combined-variant artifacts under `porting/results/` |
| Non-ML solver | BFS, joint place-route and complete 14-solver heuristic registry | **COMPLETE / COLLECTIVE GATE PASS** | `porting/components/heuristic_registry.md`, `vne_heuristic_all_unit` |
| Non-ML solver | exact (`mip`, `d_round`, `r_round`) | **IMPLEMENTED / FOCUSED GATES PASS** | `porting/components/exact_solvers.md`, `porting/results/exact_solvers_2026-07-31.md`; 1/1 Docker GCC 11 unit; focused MIP structural match + 1.955914x median; all three native Main smokes pass; full parity remains open |
| Non-ML solver | `exact_with_risk` | **IMPLEMENTED / FOCUSED OBJECTIVE GATE PASS** | `porting/components/exact_with_risk.md`; copied exact MIP feasibility rows with normalized scarcity/balance/bridge tie-breaker; equal-hop and shorter-route dominance tests pass |
| Non-ML solver | meta-heuristic (`ga_meta`, `sa_meta`, `ts_meta`, `pso_meta`, `aco_meta`) | **IMPLEMENTED / FOCUSED GATE PASS** | `porting/components/meta_heuristic.md`, `porting/results/meta_heuristic_2026-08-03.md`, `vne_meta_heuristic_unit`; ML/RL remains deferred |
| System | online/offline/changeable/time-window + main config runtime | **IMPLEMENTED / EXACT ONLINE GATE PASS** | `porting/components/system.md`, `porting/results/system_main_e2e_differential_2026-07-30.json`, `porting/results/system_transaction_integration_2026-07-30.md` |
| ML runtime boundary | vendored LibTorch 2.6.0 CPU + CUDA-compatible probe | **IMPLEMENTED / NATIVE GATE PASS** | `DEPENDENCIES.md`, `porting/components/libtorch.md`, `porting/results/libtorch_probe_2026-08-04.md`; Python oracle check is ready but awaits Docker daemon recovery; learning/RL remains deferred |
| Learning/ML | `solver/learning/**` | OUT OF SCOPE | explicitly deferred |

## Completed behavior

`virne.utils.network` ports all five original functions:

- `path_to_links`;
- `get_bfs_tree_level`;
- `flatten_recurrent_dict`;
- `flatten_dict_list_for_gml`;
- `sanitize_attr_setting`.

The C++ extension `get_bfs_tree_levels` evaluates independent sources in a
deterministic batch. A persistent executor reuses up to eight workers. The
recorded 3-round worker sweep selected eight workers for BFS and GML batches;
path conversion remains sequential by default because wider fan-out did not
reliably beat the contiguous loop.

Canonical 5-warm-up/31-sample results passed 15 differential groups and every
benchmark checksum. Selected automatic variants were 34.17x faster than Python
for path conversion, 21.60x for 50 BFS sources, 24.68x for all 500 BFS sources,
5.97x for GML flattening, and the sequential ports were 8.71x/6.41x faster for
recursive flattening/sanitizing. Full data is in
`porting/results/utils_network_2026-07-27.md`.

`virne.utils.class_dict` now ports the insertion-ordered, shallow-update and
deep-snapshot data contract. Truly dynamic names resolve once to an
object-local `ClassFieldId`; repeated access is direct by ID or typed reference.
The four historical solution mapping names are classified on insertion, so no
string comparison occurs in the snapshot loop. This dynamic leaf is not a
replacement for typed fixed schemas: the future `Solution` port must use
direct fields and enums.

Deterministic `from_dict_batch`/`to_dict_batch` extensions reuse a persistent
executor, retain input/error order, handle concurrent callers, and make
reentrant batch copies sequential. Automatic mode stays sequential below
8,192 aggregate top-level fields and uses up to eight affinity-bounded workers
above it; nested-heavy callers retain an explicit override. The gate passed
16/16 exact data cases, sanitizer and warning-clean builds, and every timing
checksum. Canonical explicit-eight speedups range from 2.12x for construction
to 422.71x for compact-ID lookup. See `porting/components/class_dict.md`.

`network.topology.topology_generator` now ports path, star, non-periodic grid,
connected Erdős-Rényi and connected Waxman generation. Dynamic type strings
resolve once to `TopologyType`; fixed batch requests store that enum directly.
The wrapper preserves global Python RNG continuation, unbounded default retry,
the original positional Waxman parameter quirk, and NetworkX grid neighbor
order. A seeded deterministic batch extension uses measured 5/6-worker auto
policies bounded by CPU affinity. The gate passed 33 exact differential cases,
14/14 full CTests and all timing checks. See
`porting/components/topology_generator.md`.

`network.topology.topological_metric_calculator` now ports degree, closeness,
eigenvector, and betweenness centrality with exact float32-before-normalize
semantics. Fixed options/results are direct fields and metric dispatch uses
`TopologicalMetricKind`; no attribute or metric string enters a graph loop.
Closeness sources run independently, while parallel Brandes stores bounded
source contribution blocks and reduces them in Python source order so every
float bit remains deterministic. The measured auto policy uses eight workers
for either expensive metric alone and seven when both are enabled on the
reference eight-CPU cpuset.

The gate passed 92/92 exact Python differential cases (including a 64-graph
ordered corpus), workers 1 through 8, and all five performance rows. Canonical
speedups ranged from 65.59x to 158.68x with identical float32 checksums. See
`porting/components/topological_metric_calculator.md`.

`virne.utils.setting` now ports raw JSON/YAML parsing, exact Python-compatible
serialization, file modes/conversion, arbitrary-precision integers, aliases,
and deterministic batch parse/dump.  Fixed format/mode/value/error categories
are enums/direct fields.  Dynamic keys are owned once, indexed by
`SettingKeyId`, and stored in dense value lanes; repeated access never hashes a
string.  JSON uses reusable thread-local Boost SAX state, a one-pass exact
number classifier, zero-copy complete number tokens, and a flat dynamic-key
boundary index.  YAML reuses the frozen yaml-cpp parser and a synchronized
compact-ID sort cache.

The final gate passed 41 physical fixtures/82 exact transforms, 20 compatible
synthetic cases, 12 security/error rejections, 17,513 exact float-bit cases,
all file/batch contracts, sanitizer/stress, and worker 1..8 plus automatic.
Every canonical row beat Python; speedups include 1.658x JSON parse, 15.350x
YAML parse, and 187.758x YAML batch dump.  See
`porting/results/setting_2026-07-28.md`.

`virne.utils.stats` now ports `test_running_time` as the typed
`RunningTimeFunction` wrapper with exact wall-clock, formatting, forwarding,
return-identity, exception-order, and byte-output behavior.  Callable, clock,
sink, name, and cached prefix are direct fields; production has no dynamic map,
`std::function`, global lock, Threads dependency, or worker API.  Independent
wrapper throughput was nevertheless tested at 1/2/4/8 workers without changing
side-effect order inside any wrapper.

The stats gate passed 22/22 differential cases, strict/sanitizer/stress/full
CTest, and a five-warmup/31-sample Docker sweep with invariant return/output
checksums.  Total-wrapper speedups range from 52.028x at one worker to 170.820x
at eight; pure wrapper overhead is 45.106x to 145.665x faster.  See
`porting/results/stats_2026-07-28.md`.

`virne.utils.manager` now ports `delete_temp_files`, `clean_save_dir`, and
`delete_empty_dir` with direct fixed config fields, enum error/operation
categories, and native paths. It deliberately preserves the original
non-empty `delete_temp_files` type-error behavior and every sequential
enumeration, stdout, first-error, and partial-deletion side effect. Python's
unsafe algorithm-symlink escape is the one mandatory safety deviation: C++
returns `unsafe_path_escape` without deleting outside the supplied root.

POSIX builds use libc `opendir`/`readdir`/`stat`/`rmdir` fast paths; other
platforms use the `std::filesystem` fallback. A regular run entry that Python
passes to `shutil.rmtree` maps its `NotADirectoryError` to typed C++
`remove_failed`/`remove_run_tree`. There is no production worker API because
parallel deletion would change observable behavior.

The manager gate passed 24/24 compatible differential cases plus the recorded
safety deviation, strict warnings, ASan/UBSan/leak checks, 100 stress runs, and
full CTest 19/19 including frozen integrity. In the five-warmup/31-sample
canonical run, all success-path rows beat Python by 1.186x to 1.514x. The
legacy exception-only row measured 0.886x and is report-only with no speed
gate. See `porting/results/manager_2026-07-28.md`.

The non-Torch `dataset` core now ports parameter extraction/string helpers,
the locked empty average stub, filename generation, and typed physical/virtual
dataset path builders. Its production schema uses direct fixed fields,
`DistributionKind`/`DatasetTopologyKind` enums, compact `DatasetAttrId`, and a
scalar variant that preserves Python integer/float/bool spelling. No fixed key
is hashed in a hot loop, and dynamic attribute names are owned once.

The core differential passed 60/60 cases plus 16,395 exact binary64-to-Python
string cases without importing real Torch. The five-warm-up/31-sample
canonical run passed all 32 scalar/worker rows; scalar speedups are 4.493x to
14.404x and the fastest physical/virtual batch medians use six workers. Auto
mode is family/size aware and every worker width 1..8 preserves checksum,
output bytes, order, and lowest-index error semantics. Strict warnings,
ASan/UBSan/leaks, 100 stress runs, full CTest 20/20, and frozen integrity pass.
See `porting/results/dataset_core_2026-07-28.md`.

The non-Torch `dataset` NumPy RNG leaf now ports normal, uniform, exponential,
and poisson generation through one caller-owned `NumpyRandomState`. Fixed
distribution/value discriminants are enums and every parameter is a direct
field; strings resolve once at the public boundary and no hot sample/cast loop
performs string or map lookup. The frozen `random_lib` implementation was
reused unchanged.

The gate passed 92/92 exact NumPy 2.2.6 cases, with element/bit equality for
ordinary buffers and byte count plus FNV for large buffers, plus seven locked
Python cast warnings, error stage, draw count, and subsequent
`random()+normal()` continuation. Two Python `-O` boundary cases are recorded
separately. RNG draws remain sequential; only the exact exponential
transform+int/bool cast uses measured contiguous workers. Automatic mode is
affinity bounded: below 131,072 items it is sequential, medium int uses three
lanes, medium bool uses seven, and both use seven from 262,144 onward.

The canonical interleaved five-warm-up/31-sample run passed all 60 timing rows
with exact checksum/state gates; C++ speedups range from 1.645x to 8.999x.
Separate 192,000- and 600,000-item sweeps passed 18/18 rows each. Strict
warnings, ASan/UBSan/leaks, 100 stress runs, one/two/eight-CPU affinity tests,
full CTest 21/21, and frozen integrity all pass. See
`porting/results/dataset_rng_2026-07-28.md`.

The dataset XML/GML leaf parses UTF-8/ISO-8859-1 SNDlib documents directly
through the pinned Boost 1.85 RapidXML buffer, materializes the frozen simple
`Graph`, and writes exact NetworkX 3.4.2-compatible GML through a dataset-local
serializer. Fixed XML and graph fields are direct members and resolved
`AttrId`s; dynamic endpoint labels are each resolved once before the edge hot
path. Frozen graph code was not changed.

The exact differential passed 57/57 cases. The real Brain fixture produced 161
nodes, 166 simple edges, and exact 45,085-byte GML. In the canonical
five-warm-up/31-sample run, sequential single-document parsing was 27.806x
faster than Python, automatic 16-Brain parsing was 149.433x faster, automatic
synthetic parsing was 107.445x faster, and full Brain XML-to-GML was 37.787x
faster. All results retain exact graph/output checksums.

The 1..8 plus auto sweep covered 2/4/8/16/32/64 documents for both Brain and
synthetic corpora. Auto beat sequential in all 12 policy rows and stayed within
25% of the best explicit width. Strict warnings, ASan/leaks, UBSan, TSan, stress,
affinity checks, full CTest 22/22, and frozen integrity all pass. See
`porting/results/dataset_xml_2026-07-28.md`.

The independent `network.attribute.attribute_method` policy leaf now exposes
fixed enums/direct specs, exact bool/int64/double resource arithmetic,
Python-exact mixed int/double comparison, typed errors, and a contiguous
double satisfiability batch. Dynamic method strings resolve once; the batch
hot loop uses only direct pointers/bytes and runtime-dispatches to strict-IEEE
AVX-512/AVX2 with a scalar fallback.

The gate passed 93/93 scalar/dynamic-MRO cases and 30/30 raw-bit batch cases.
The batch matrix covers all three comparisons, hard/soft, configured workers
0/1/2/3/8, signed zero, subnormal, infinities, five qNaN payloads, and sNaN.
Unit tests additionally cover workers 0..8, input/output aliasing, eight
concurrent callers, int64 overflow, and mixed values beyond `2^53`.

Worker width is a caller-supplied typed config: zero/one are sequential and
wider values are affinity/count capped; no host-specific automatic threshold is
embedded. A representative 4,000,000-item exact corpus demonstrated 2-worker
parallel execution at 1.165458x sequential. All normal timing rows beat Python
by 19.044881x to 115.556359x. See
`porting/results/attribute_method_2026-07-28.md`.

The typed `network.attribute.BaseAttribute` generation leaf is complete. Fixed
owner/kind/distribution/value discriminants are enums and every schema field is
direct. Standard generation reuses the frozen dataset/RNG API; customized
generation preserves exact Python integer subtraction before float conversion
and can transform disjoint contiguous blocks with a caller-configured width
after the canonical sequential RNG draw.

The direct source gate passed 32/32 static/generation/error/state cases. The
compact 300,000-value worker smoke passed all six rows: C++ was 1.404x to
2.575x faster than Python, and exponential-to-int reached 2.496x at worker 8.
Strict warnings, ASan/UBSan/leaks, full CTest 24/24, frozen integrity, and diff
checks pass. See `porting/components/base_attribute.md` and
`porting/results/base_attribute_2026-07-28.md`.

The typed `network.attribute.NodeAttribute` adapter is complete for undirected
and directed graphs, together with status, extrema, resource, and position
leaves. Each dynamic attribute name resolves once per graph into `AttrId`; all
node hot loops use only the ID, dense vertex indices, and direct attribute maps.
Fixed owner/kind/restriction/checking-level fields are direct enums or members.
Sparse writes remain ordered; dense independent writes/reads and position
post-processing accept caller-configured contiguous workers while RNG draws
remain in exact NumPy order.

The direct gate passed 37 exact C++/Python cases and recorded five Python-only
dynamic boundaries. The compact six-row timing smoke passed all checksum, byte,
and RNG-state gates: dense set/get was 6.002x to 8.888x faster, and position
generation was 6.490x to 8.427x faster. Strict warnings, ASan/UBSan/leaks, full
CTest 26/26, and frozen integrity pass. Its accepted benchmark is frozen. See
`porting/results/node_attribute_2026-07-28.md`.

The typed `network.attribute.LinkAttribute` adapter is complete for undirected
and directed graphs together with status, extrema, resource, and latency
leaves. Dynamic link and position names resolve once into graph-local `AttrId`
values; edge, matrix, aggregation, distance, and path hot loops use descriptors,
IDs, direct fields, and typed enums only. Dense independent operations accept a
caller-configured worker width, while ordered sparse/path mutation stays
sequential to retain Python-visible side effects.

The direct gate passed 35 C++/Python cases plus five recorded Python-only
boundaries. All six compact timing rows retained exact checksums and output
bytes: dense roundtrip was 5.722x to 7.408x faster than Python and
position-derived latency was 8.550x to 72.780x faster. Strict warnings,
ASan/UBSan/leaks, full CTest 26/26, and frozen integrity pass. The accepted
benchmark is frozen and must not be rerun or updated. See
`porting/results/link_attribute_2026-07-28.md`.

The typed `network.attribute.GraphAttribute` adapter is complete for Graph and
DiGraph scalar metadata, independent-graph batch access, status, extrema, and
resource leaves. A name resolves once per graph into `AttrId`; compact graph-map
and definition identity tokens prevent cross-graph/cross-field ID reuse without
string work. Batch validation occurs before dispatch, while workers touch only
pointer/ID/direct slots. Duplicate targets and observable shared resource
mutation remain sequential.

The direct gate passed 45 shared C++/Python cases, four native extension cases,
and five recorded Python-only boundaries (54 total). The single three-row
frozen benchmark retained exact raw64 bytes/checksum and beat Python by
1.209x to 2.595x at configured workers `1/2/8`. Strict warnings,
ASan/UBSan/leaks, full CTest 27/27, and frozen integrity pass. See
`porting/results/graph_attribute_2026-07-28.md`.

The typed `network.attribute.AttributeBenchmarkManager` leaf is complete for
prepared node/link/link-sum reductions, ordered compact-ID maps, and the global
identity-preserving cache. Fixed descriptor/matrix/group/error fields are direct
members and enums. Names are owned once and resolve to
`AttributeBenchmarkId`; row workers use only float pointers, numeric row IDs,
fixed dimensions/repetition, and pre-sized result slots. Direct-link column
duplication is virtual, while insertion and duplicate overwrite stay sequential
in Python order.

The gate passed 19 shared Python cases, seven native cases, and six recorded
Python-only boundaries (32 total), including exact NumPy 2.2.6 signed-zero and
NaN payload bits. Its single frozen benchmark retained checksum
`16589509004670834835` and beat Python by 9.171x to 11.477x at configured
workers `1/2/8`. Strict warnings, ASan/UBSan/leaks, full CTest 28/28, and frozen
foundation integrity pass. See
`porting/results/attribute_benchmark_manager_2026-07-28.md`.

The typed `network.attribute.AttributeFactory` boundary is complete for the
eight Python-registered node/link owner-kind pairs. Raw fixed keys resolve once
into `AttributeFactorySpec`; registries own first-position/last-value duplicate
semantics and compact `AttributeRegistryId`s. Spec workers use deterministic
contiguous blocks at caller-configured widths, while raw setting decode and
ordered deduplication remain sequential to preserve public failure order.

The gate passed 29 direct Python cases, three native typed cases, and seven
recorded Python-only boundaries (39 total). Its single frozen 32,768-spec
benchmark retained checksum `13127048606653777947` and beat Python by 5.979x,
6.416x, and 4.822x at workers `1/2/8`. Strict production/unit/harness checks,
ASan/UBSan/leaks, full CTest 29/29, and frozen integrity pass. See
`porting/results/attribute_factory_2026-07-28.md`.

The non-ML `network.BaseNetwork` model is complete. It composes the frozen
Graph/topology/attribute APIs through direct typed config fields, compact
factory definition IDs, graph-local value IDs, identity-bearing bindings, and
caller-configured workers. Ordered merge/generation/partial-mutation, stale
cached cardinalities, views/clone/GML/setting behavior, and prepared benchmark
delegation are locked. The measured unnormalized sum adapter consumes frozen
row-major sparse COO instead of materializing a dense quadratic matrix.

The gate passed 29 exact Python cases plus 11 recorded Python-only boundaries
(40 total). Its single frozen 8,192-element benchmark retained exact
type/bit/order checksums: get is 3.245x-6.326x faster, set is 2.398x-6.495x
faster, and manager preparation/reduction is 67.875x-103.983x faster at workers
`1/2/8`. Strict warnings, ASan/UBSan/leaks, full CTest 30/30, and frozen
integrity pass. See `porting/results/base_network_2026-07-28.md`.

The typed `network.VirtualNetworkEvent` leaf is complete. Fixed ID, event type,
request ID, and time fields are direct; batch construction writes pre-sized
slots using caller-configured workers, and stable time sorting preserves input
order for ties. No string lookup or automatic worker selection enters an event
hot loop.

The accepted differential passed 13 shared cases plus one native extension and
six recorded Python-only dynamic boundaries. Its single frozen 131,072-event
benchmark retained entry count, byte count, and checksum at workers `1/2/8`,
beating Python by 47.206x, 53.776x, and 34.298x. See
`porting/results/virtual_network_event_2026-07-29.md`.

The non-ML `VirtualNetworkRequestSimulator` and dataset `Generator` chain is
complete. Both decode fixed config keys once into typed fields, preserve the
single ordered Python/NumPy streams, and forward only caller-configured worker
widths to deterministic completed leaves. Simulator hot paths use contiguous
arrangement/event buffers and numeric request/event indexes. Generator reuses
the frozen Config, Random, dataset-path, PhysicalNetwork, and simulator APIs;
its four-stage changeable workload moves completed request quarters and renews
events once without reparsing config or introducing Torch.

The simulator passed 24 classified core cases and 19 classified I/O/cache
cases. Its frozen 65,536-request benchmark is 1.986x-2.807x faster for
arrangement and 20.735x-26.236x faster for event scheduling at caller workers
`1/2/8`. Generator passed 19 classified differential cases; its frozen
512-request benchmark is 1.840x-2.004x faster for ordinary generation and
2.501x-2.778x faster for the changeable workload. See
`porting/components/network_generation_chain.md`.

The fixed-schema `core.Solution` leaf is complete. Every known field is a
direct member; node/link mappings use typed numeric keys and compact
object-local entry IDs, while resource/constraint values are indexed directly
by completed attribute registry ID. No fixed field name is stored in a string
map. Batch construct/reset uses only caller-configured workers and retains
input/error order.

The gate passed 17 shared Python cases, two native ID/slot cases, and four
recorded Python-language boundaries. Its frozen 32,768-solution lifecycle
benchmark retained checksum `4149871601928940857` and was 6.760x, 6.005x, and
6.419x faster at workers `1/2/8`. See
`porting/results/solution_2026-07-29.md`.

The fixed-schema `virne.utils.config` leaf is complete. Simulation and feature
summaries, timestamps, run-directory inputs, errors, and operations are direct
fields/enums. Extracted attribute membership is a five-slot array indexed by
resolved `AttributeKind`; all four attribute groups are counted in one pass
without string lookup. YAML remains a cold compatibility boundary only, and
each fixed Config path is resolved once into a typed field.

The gate passed 10 shared Python cases, six native cases, and five documented
language boundaries (21 total). Its permanently frozen 2,048-config benchmark
retained checksum `6644728919515556009` and was 36.342x, 52.867x, and 43.241x
faster at caller workers `1/2/8`. Strict compilation, ASan/UBSan/leaks, targeted
CTest, and frozen integrity pass. See
`porting/results/utils_config_2026-07-29.md`.

The typed `core.controller.ConstraintChecker` leaf is complete. Selection IDs
belong to the virtual typed registries; preparation resolves each genuinely
dynamic name once against independent virtual/physical graphs and hot checks
retain only concrete attribute pointers, graph-local IDs, vertices, and direct
result slots. Fixed categories, flags, endpoints, errors, operations, and
requests are direct fields/enums, with no fixed field or hot lookup in a
string-keyed map. Graph output IDs are direct sparse slots with deterministic
last-write behavior.

The gate passed 14 exact shared Python cases and seven native unit groups,
including independent registry ordering, hard/soft graph/node/link/path
constraints, worker `0/1/2/8` equality, lowest-error ordering, and concurrent
callers. Its permanently frozen 32,768-request benchmark retained checksum
`11118347938320421286` and was 46.384x, 46.694x, and 74.694x faster at caller
workers `1/2/8`. Strict compilation, ASan/UBSan/leaks, targeted CTest, and
frozen integrity pass. See
`porting/results/constraint_checker_2026-07-29.md`.

The fixed-schema `core.controller.ResourceUpdator` leaf is complete. Selection
uses virtual registry IDs; preparation resolves genuinely dynamic names once
to graph-local IDs and typed attribute pointers. Fixed operations, owners,
targets, amounts, errors, and requests are direct fields/enums, with no
string-keyed map or lookup in hot update loops. Scalar/path order matches
Python; caller-configured batch workers parallelize only disjoint targets and
retain deterministic lowest-error behavior.

The gate passed 10 exact shared Python cases and seven native unit groups. Its
permanently frozen 32,768-update benchmark retained checksum
`17411705748429442498` and was 33.113x, 5.542x, and 5.435x faster at caller
workers `1/2/8`. The accepted workload favors worker 1; width remains a config
input. Strict compilation, ASan/UBSan/leaks, targeted CTest, and frozen
integrity pass. See `porting/results/resource_updator_2026-07-29.md`.

The fixed-schema `core.controller.TopologyAnalyzer` leaf is complete. Six path
modes and every option/request/status are direct fields/enums. Resource names
bind once to independent virtual/physical IDs; path, edge, mask, predicate, and
worker loops use only vertices, edge IDs, `AttrId`, typed pointers, and direct
slots. First/available mode reconstructs completed raw-order BFS predecessors
to preserve NetworkX unweighted-Dijkstra FIFO ties; a benchmark-discovered
cyclic tie became a permanent differential without changing frozen Graph.

The gate passed 24 exact shared Python cases and nine native unit groups. Its
permanently frozen 4,096-query benchmark retained checksum
`10025764477037659827` and was 83.768x, 147.126x, and 177.706x faster at caller
workers `1/2/8`. Strict compilation, ASan/UBSan/leaks, targeted CTest, and
frozen integrity pass. See
`porting/results/topology_analyzer_2026-07-29.md`.

The fixed-schema `core.controller.NodeMapper` leaf is complete. Matching
methods, options, results, errors, operations, and workers are direct
fields/enums. Dynamic node resources bind once at preparation; placement,
candidate, violation, resource, and undo loops use only vertices, registry and
graph-local IDs, typed numeric lanes, direct Solution tables, and a byte hard
constraint mask. Greedy candidate checking supports caller-configured workers
while consuming results/errors in exact Python order; commits remain
sequential.

The gate passed 12 exact shared Python cases plus native boundary/concurrency
groups. Its permanently frozen 32-node/2,080-candidate benchmark retained
checksum `15604526718891224062` and was 18.628x, 3.828x, and 3.082x faster at
candidate workers `1/2/8`. The workload favors worker 1 and no automatic host
policy was introduced. Strict compilation, ASan/UBSan/leaks, targeted CTest,
hot-string audit, and frozen integrity pass. See
`porting/results/node_mapper_2026-07-29.md`.

The fixed-schema `core.controller.LinkMapper` leaf is complete. Routing
methods, options, results, errors, operations, paths, endpoints, flags, and
workers are direct fields/enums. Dynamic resource names bind only during
preparation; every candidate, edge, resource, pooling, undo, and mapping loop
uses registry/graph-local IDs, vertices, typed numeric lanes, masks, and direct
Solution tables. Candidate checks accept a caller-configured width while
preserving original path/error order; all observable mutations remain
sequential.

The gate passed 17 exact shared Python cases plus native boundary/concurrency
groups. Its permanently frozen 1,024-candidate benchmark retained checksum
`14052633754962558449` and was 14.092x, 8.978x, and 5.152x faster at candidate
workers `1/2/8`. Strict compilation, ASan/UBSan/leaks, targeted CTest,
hot-string audit, and frozen integrity pass. The MCF/SCIP solver surface remains
deferred. See `porting/results/link_mapper_2026-07-29.md`.

The fixed-schema `core.Counter` leaf is complete. Its exact differential passed
20 shared cases and the single permanently frozen benchmark retained checksum
`3910809078534895256`, measuring 3.614x, 4.835x, and 4.372x speedups at caller
workers `1/2/8`. Strict GCC 11, ASan/UBSan/leaks, focused unit, hot-ID audit,
and targeted frozen-integrity CTest passed. See
`porting/results/counter_2026-07-29.md`.

The non-ML `core.Recorder` leaf is complete. Fixed configuration, event,
state, record, path, operation, and error schema use direct fields/enums;
resource reductions retain prepared Counter IDs. Membership hot paths use
dense physical-node slots and preserve Python first-touch order without string
lookups. Dynamic ClassDict extras are deep-copied only at the cold record
boundary. Event mutation remains ordered and sequential; caller workers are
forwarded to independent Counter reductions and fixed-row CSV serialization.

The gate passed 6 exact shared non-ML Python cases plus the focused native
unit surface at workers `0/1/2/8`. Its single permanently frozen prepared
arrival/deep-snapshot benchmark retained checksum `8168332940057982619` and
was 24.907x, 17.061x, and 7.404x faster at workers `1/2/8`. Strict GCC 11,
ASan/UBSan/leaks, targeted CTest, path/schema/deep-snapshot checks, and hot-ID
review passed. RL/reward calculation, feature/training logic, solver/system,
Torch, and logging sinks were not implemented; only a cold optional summary
extension seam was prepared. See `porting/results/recorder_2026-07-29.md`.

The non-ML `core.Logger` leaf is complete. Backends, levels, paths, entries,
errors, and operations are direct fields/enums. Dynamic metric names register
once to object-local dense IDs; progress visibility is precomputed, so batch
format/CSV loops use only direct slots. Console/file/CSV output is synchronized
and ordered, while valid batch formatting uses caller workers and one ordered
CSV append.

The gate passed 9 exact Python cases plus ANSI/CRLF/schema/partial-error and
eight-caller native coverage. Its permanently frozen 4,096-row benchmark
retained FNV64 `3834733185215270441` and was 14.878x, 16.951x, and 13.689x
faster at workers `1/2/8`. Strict GCC 11, ASan/UBSan/leaks, targeted CTest, and
hot-ID review passed. WandB/TensorBoard/Torch/RL/solver/system were not linked;
only the cold generic sink seam was prepared. See
`porting/results/logger_2026-07-29.md`.

The fixed-schema non-solver `core.controller.Controller` lifecycle is
complete. Configuration, path modes, failure phase, workers, endpoints, and
results are direct fields/enums; dynamic resource names bind once and every
neighbor/resource/pool loop uses vertices, edge IDs, registry IDs, graph-local
`AttrId`s, direct slots, and masks. Place/route/undo and error mutation remain
sequential. Deploy/release parallelize only preflighted disjoint node/edge-ID
targets and fall back before mutation for every unsafe or duplicate case.

The gate passed 10/10 exact Python cases, focused worker/partial/concurrency
unit coverage, strict GCC 11, ASan/UBSan/leaks, hot-ID review, and targeted
CTest. Its permanently frozen 32,768-mutation deploy/release benchmark retained
deployed/restored checksums `17514356897791579542` /
`8486823302284311477` and was 8.639x, 5.186x, and 2.853x faster at workers
`1/2/8`. MCF, BFS deployment/search, candidate search, solver/system, RL, and
ML remain outside the target. See `porting/results/controller_2026-07-29.md`.

The typed non-RL `core.BaseEnvironment` and `SolutionStepEnvironment` lifecycle
is complete. Reset resolves dynamic request IDs once into dense event/request
slots and rebuilds prepared Controller/Counter views after every owned-network
replacement. Arrival/deploy/reject/Recorder/transit and leave/release mutation
retain Python order; fixed state and errors are direct fields/enums. A direct
request-index-to-history-index slot avoids an extra hash lookup on leave, while
frozen Recorder retains its own numeric compatibility index. Failed automatic
leaves can resume through `drain_leaves()` without hiding partial state.

The exact differential passed 9/9 shared cases, and unit coverage passed
workers `0/1/2/8`, sparse request IDs, reset/reprepare, rejection priorities,
typed failures, recovery and concurrent environments. Strict GCC 11,
ASan/UBSan/leaks and targeted CTest pass. Its permanently frozen 96-request,
192-event benchmark retained checksum `17358322786803582063`, physical raw64
checksum `5251282115348753471`, and 72,481 output bytes. It was 146.854x,
38.061x and 9.739x faster at caller workers `1/2/8`. Observation, reward,
JointPR, solver/system, RL and ML remain outside this component. See
`porting/results/environment_2026-07-29.md`.

The non-ML `solver.rank.LinkRank` leaf is complete. Methods, options, errors,
operations, selection, edge identity, and result entries are direct
fields/enums. Dynamic link-resource names resolve outside this API; preparation
binds registry IDs once to graph-local IDs, and every gather/reduction loop uses
only edge descriptors, typed IDs, numeric variants, and direct slots. Wider
caller widths split independent edge score columns deterministically; ordering
remains sequential and exactly Python-compatible, including classified NaNs.

The exact differential passed 12/12 shared cases at workers `1/2/8`; the
pinned Python BaseNetwork FFD typo is a recorded boundary. Strict GCC 11,
ASan/UBSan/leaks, targeted CTest, hot-ID review, and the exhaustive Timsort
probe passed. Its permanently frozen 131,072-edge x 8-resource benchmark
retained checksum `10478239091350211214` and was 2.133x, 2.208x, and 2.227x
faster at workers `1/2/8`. See
`porting/results/link_rank_2026-07-29.md`.

The non-ML `solver.rank.NodeRank` leaf is complete. Its eight methods, resource
selection, parameters, options, errors, result variants, prepared bindings, and
worker widths are typed fields/enums. Dynamic resource names resolve before
this API; every node/resource/adjacency/matrix hot loop uses only vertices,
registry/graph-local IDs, numeric lanes, and direct slots. Caller widths
`0/1` are sequential; wider widths split only deterministic independent blocks.
Random continuation, convergence reductions, Dijkstra, and Python-compatible
finite/NaN ordering remain sequential where observable order matters.

The exact differential passed 13/13 shared cases at workers `1/2/8`, including
raw binary64 values, NPS tuple lanes, ordered node IDs, and NumPy RNG
continuation. Strict GCC 11, ASan/UBSan/leaks, targeted CTest, hot-ID review,
and the generic CPython 3.10.20 Timsort probe passed. Its permanently frozen
131,072-node x 8-resource FFD benchmark retained checksum
`11449996351475094403`; C++ was 1.277x, 1.279x, and 1.312x faster at workers
`1/2/8`. See `porting/results/node_rank_2026-07-29.md`.

The non-ML `solver.base_solver` foundation is complete. Fixed config, ranking,
mapping, path, action, counter, category, operation, and error state are direct
fields/enums. Controller/Counter are retained as const references;
Recorder/Logger as mutable references. Dynamic solver names are owned only by
the startup registry, resolve once to stable `SolverId`, and never enter the
frozen descriptor/factory creation path. Explicit registration replaces
Python import side effects; learning categories remain dependency-free enum
seams only.

The exact AST-isolated differential passed 13/13 shared cases at native
workers `1/2/8`. Focused unit/concurrency coverage, strict GCC 11 production,
unit and harness builds, ASan/UBSan/leaks, targeted CTest, aggregate solver
integration, frozen-component integrity, and the hot-ID audit passed. Its
permanently frozen 32-descriptor / 4,096-holder benchmark retained 419,498
bytes and checksum `13751587758314786690`; C++ was 11.174x, 9.906x, and
10.401x faster at workers `1/2/8`. See
`porting/results/base_solver_2026-07-29.md`.

All eight non-ML solvers in `solver.heuristic.node_rank` are complete and
frozen: order, random, GRC, FFD, NRM, PL, NEA and random-walk. Standard variants
reuse the typed `BaseNodeRankSolver`; random rank borrows an explicit
caller-owned NumPy-compatible stream; PL/NEA use prepared numeric constraint
IDs and exact CPython integer-set traversal. Fixed state never enters a string
map and every dynamic solver name is confined to cold registration.

The combined exact AST-isolated differential passes ten cases at workers
`0/1/2/8`, including sparse-ID candidate ties and RNG continuation. On the
single frozen combined fixture every workers=1 solver beats Python by
`1.174x` to `75.990x`; wider workers remain caller choices because the small
fixture exposes fan-out overhead. The earlier order-only gate remains frozen
separately. See `porting/components/heuristic_node_rank.md`; do not rerun or
edit either accepted benchmark.

The remaining canonical `solver/heuristic` registry is complete. Three BFS
solvers reuse the prepared Controller and NodeRank APIs; three joint
place-route solvers use CPython-compatible candidate order, caller-owned
`PyRandom` where required, live FFD reranking and `k=1` routing. The public
central registrar returns 14 direct `SolverId` fields, including every class
grouped inside Python's `node_rank.py`. Dynamic names resolve only once.

The one-shot Docker GCC Release gate is 1/1 PASS. It creates and solves all 14
factories, compares exact workers=1/4 output and both RNG continuations, and
checks BFS/joint partial rollback. Its compact native catalog signal was
70.8778 ms at workers=1 and 36.7232 ms at workers=4 (1.930x faster). Existing
frozen Python/C++ node-rank benchmarks were not rerun. See
`porting/components/heuristic_registry.md`.

The native `solver.exact` leaf now imports workspace-local OR-Tools 9.15.6755
through the explicit `virne_ortools` target and registers `mip`, `d_round` and
`r_round` behind direct `ExactSolverIds` fields. Its split-capable integer-flow
formulation and compact meta-flow GLOP paths extend Python's hard-coded `cpu`/`bw` model
to every selected node and link resource: one placement/flow is shared while
every resource receives an independent capacity row. Dynamic names bind once
across independent graph registries; model and extraction loops retain only
graph-local IDs and dense
numeric buffers. A focused two-node/two-link-resource unit target covers
registry order, second-lane rejection, forced split flow, reusable aggregate
capacity, exact-integer/path guards, immutable solve and mutable deploy/release.
OR-Tools thread count is a direct `native.workers.exact` field resolved once
before the request loop. Floating-only link lanes use continuous path flow;
integral lanes retain exact GCD-scaled journals.
It passes 1/1 Docker GCC 11 CTest, including typed split journals and explicit
self-loop/tiny-positive-demand guards. A separate one-resource MIP smoke matches the
accepted/node-slot/path-edge structure and measures Python 16.8899 ms versus
native 8.6353 ms median (1.955914x). Native multi-resource Main smokes pass for
all three names. This is not canonical or full parity: the packages use
different OR-Tools versions, concrete symmetric mappings are not compared,
and cross-language Main/rounding differentials remain open. See
`porting/components/exact_solvers.md` and the linked focused result.

The non-ML system/main runtime now composes the completed generator,
environment, registry and node-rank solvers from Hydra-style overrides. Online
is exact against pinned Python; offline/changeable/time-window define typed
semantics for incomplete Python TODO paths. Default output is a compact JSON
summary plus a rate-limited Python-style progress bar. Progress receives direct
Recorder fields, uses global epoch labels for changeable stages, and does no
string lookup in request loops. Host strict and Docker GCC 11 units pass; the
representative `order_rank` end-to-end differential is exact and records both
Python/native time. The later default seed-0 `ffd_rank` verification, with no
explicit node/request-count override, is also exact at 100 nodes, 528 links,
1,000 requests, 752 accepted and 248 rejected; its single GCC 11 runtime signal
is 51.994x faster than Python. See `porting/components/system.md` and
`porting/results/system_transaction_integration_2026-07-30.md`.

Main now also supports the shipped multi-resource CPU/GPU/RAM groups. Virtual
controller IDs bind once to independent physical IDs, while Counter selects
typed resource IDs independently per registry, avoiding the former collision
with interleaved `max_cpu/max_gpu/max_ram` extrema. The seed-0 20-node/8-request
`ffd_rank` full report is exact against Python (5 accepted, 3 rejected), and a
CTest locks the Python-compatible `+p_net_setting` / `+v_sim_setting` CLI.
Progress frames now use one buffered write and one explicit flush, detect the
TTY width and compact to one non-wrapping physical row; main disables C/C++
stdio synchronization and detaches stream ties before any I/O.
Main also materializes Python's derived simulation/feature fields and snapshots
the resolved application config once into a fixed cold field. Hydra internals
and native worker/output controls are split from the Python-compatible
`Config:` view; saved `config.yaml` therefore has the same application schema,
while `native_config.yaml` retains the C++ extension. Both are resolved only at
the cold boundary, and no YAML traversal reaches a hot loop.

The experimental `exact_with_risk` leaf is now registered as a fourth exact
solver. It copies the exact MIP feasibility formulation and changes only the
objective: a normalized marginal fragmentation surrogate combines residual
resource scarcity, multi-resource residual skew and a cold bridge-criticality
flag. The route-flow term is kept at coefficient one while the complete risk
domain is normalized below one, so shortest route length remains strictly
dominant for integral flow. Floating-only flow uses a two-phase route lock;
only proven-optimal results are accepted. Its focused unit confirms an
equal-hop lower-risk choice and rejects a longer low-risk alternative. See
`porting/components/exact_with_risk.md`.

## Next component

Finish the concrete compatibility review and distinct Python
LP-rounding/RNG differentials before changing the
exact state to `COMPLETE`; the focused structural timing is not a substitute.
After that, the next independent solver leaf is `solver/meta_heuristic`;
specialized Controller MCF paths remain separate scope. Reuse the completed
registries and all frozen graph/controller/random/config APIs. The LibTorch
2.6.0 runtime is now vendored behind an opt-in ABI boundary and its CPU output
probe is compared with the Python 2.6.0 oracle. ML/RL implementation remains
deferred; the non-RL environment continues to use its exact Torch-free gate.
