# Port verification workflow

Start at `../PORTING_STATUS.md`. The frozen foundation and non-ML dependency map
are documented in this directory; do not rediscover them by scanning the whole
repository unless the pinned source commit changes.

All new code must also follow `PERFORMANCE_CONTRACT.md`: fixed schema uses
direct fields/enums, dynamic strings resolve once, and graph attribute hot
loops use pre-resolved `AttrId` values.

Accepted benchmark artifacts for every completed component are frozen. Commands
retained below are provenance only: do not rerun or update those benchmarks.
Continue to run unit/differential/sanitizer/integration checks as needed. A new
component gets one compact benchmark after correctness passes, then it is frozen
immediately.

## Build and unit/regression tests

The existing toolchain container mounts `virne.cpp` at `/work`:

```powershell
docker start virne-cpp-dev
docker exec virne-cpp-dev cmake -S /work `
  -B /work/.docker-smoke/build-port-release `
  -DCMAKE_BUILD_TYPE=Release
docker exec virne-cpp-dev cmake --build `
  /work/.docker-smoke/build-port-release -j 4
```

The frozen GML test has an existing absolute dependency on `/virne/datasets`.
Run the complete suite in a temporary toolchain container with the original
checkout mounted read-only:

```powershell
$cppRoot = (Resolve-Path .\virne.cpp).Path
$pythonRoot = (Resolve-Path .\virne).Path
docker run --rm `
  --mount "type=bind,source=$cppRoot,target=/work" `
  --mount "type=bind,source=$pythonRoot,target=/virne,readonly" `
  --workdir /work virne-cpp-toolchain:gcc11.4 `
  ctest --test-dir /work/.docker-smoke/build-port-release --output-on-failure
```

## Build the non-ML Python oracle

```powershell
docker build --tag virne-python-oracle:py310-nonml `
  .\virne.cpp\porting\oracle
```

The Dockerfile pins the CPython base digest and exact NumPy/NetworkX versions.
It does not install original Virne or any ML dependency.

## Optional LibTorch runtime/probe

LibTorch 2.6.0+cpu is vendored at `libs/libtorch`; the normal non-ML build does
not link it. Configure the isolated target only when checking the ML runtime
boundary:

```powershell
docker start virne-engine-keepalive
docker exec virne-cpp-dev cmake -S /work -B /work/build-libtorch `
  -DCMAKE_BUILD_TYPE=Release -DVIRNE_ENABLE_LIBTORCH=ON `
  -DVIRNE_LIBTORCH_ROOT=/work/libs/libtorch
docker exec virne-cpp-dev cmake --build /work/build-libtorch -j 4 `
  --target vne_libtorch_probe
docker exec virne-cpp-dev ctest --test-dir /work/build-libtorch `
  -R '^vne_libtorch_probe$' --output-on-failure
```

Compare the fixed CPU tensor output with the Python oracle (the oracle wheel is
Torch 2.6.0+cu124, so version/capability fields are informational):

```powershell
$cppRoot = (Resolve-Path .\virne.cpp).Path
docker run --rm `
  --mount "type=bind,source=$cppRoot,target=/work" `
  --workdir /work virne-cpu:latest `
  python /work/porting/compare_libtorch_probe.py `
    --native /work/build-libtorch/vne_libtorch_probe --threads 1 --device cpu `
    --output /work/porting/results/libtorch_probe_2026-08-04.json
```

For a CUDA LibTorch archive, set `VIRNE_LIBTORCH_ROOT` to that archive's local
path and run the probe with `--device cuda` on a CUDA-enabled runner. Keep the
thread-pool width fixed at process/worker startup; do not mutate global Torch
thread settings inside a hot request loop. The exact non-RL environment gate
is independent and remains:

```text
python /workspace/cpp/porting/compare_environment.py \
  --source /src/virne/virne/core/environment.py \
  --harness /workspace/cpp/.docker-smoke/build-port-release/porting/vne_environment_harness
```

## Differential and canonical timing

Run from the directory containing sibling `virne` and `virne.cpp` checkouts:

```powershell
$cppRoot = (Resolve-Path .\virne.cpp).Path
$pythonRoot = (Resolve-Path .\virne).Path

docker run --rm --network none --read-only --cpuset-cpus=0-7 `
  --tmpfs /tmp:rw,noexec,nosuid,size=64m `
  --mount "type=bind,source=$cppRoot,target=/workspace/cpp,readonly" `
  --mount "type=bind,source=$pythonRoot,target=/src/virne,readonly" `
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 `
  --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 `
  virne-python-oracle:py310-nonml `
  python /workspace/cpp/porting/compare_utils_network.py `
    --cpp /workspace/cpp/.docker-smoke/build-port-release/porting/vne_utils_network_harness `
    --python-source /src/virne/virne/utils/network.py `
    --fixture /src/virne/datasets/topology/Waxman500.gml `
    --warmups 5 --repetitions 31 --workers 8
```

The script exits nonzero for a differential mismatch, checksum mismatch, missing
timing row, or if no C++ variant beats Python for any function.

## Worker sweep

Use the same mounts/environment but replace the final Python command with:

```text
python /workspace/cpp/porting/sweep_utils_network_workers.py \
  --cpp /workspace/cpp/.docker-smoke/build-port-release/porting/vne_utils_network_harness \
  --python-source /src/virne/virne/utils/network.py \
  --fixture /src/virne/datasets/topology/Waxman500.gml \
  --workers 1,2,3,4,5,6,7,8 --warmups 5 --repetitions 31 --rounds 3
```

Keep fixture construction, GML parsing, process/container startup, checksums and
serialization outside timed regions. Always pass differential/checksum gates
before interpreting a speedup.

## Topology generator gate

Using the same oracle image, mounts, network/thread restrictions, and cpuset as
the utils command above:

```text
python /workspace/cpp/porting/compare_topology_generator.py \
  --cpp /workspace/cpp/.docker-smoke/build-port-release/porting/vne_topology_generator_harness \
  --python-source /src/virne/virne/network/topology/topology_generator.py \
  --warmups 5 --repetitions 31 --workers 7
```

`--workers 0` exercises the production automatic family policy. The canonical
record uses one explicit worker count for comparable batch rows; the separate
auto-policy validation uses zero.

Run the worker sweep with:

```text
python /workspace/cpp/porting/sweep_topology_generator_workers.py \
  --cpp /workspace/cpp/.docker-smoke/build-port-release/porting/vne_topology_generator_harness \
  --python-source /src/virne/virne/network/topology/topology_generator.py \
  --workers 1,2,3,4,5,6,7,8 --warmups 2 --repetitions 5 --rounds 3
```

The comparator pins the exact Python source SHA-256, direct-loads only that
file, rejects an unexpected NetworkX version/Torch import, compares exact graph
and RNG state before timing, and requires every benchmark checksum to match.

## Topological metric calculator gate

Using the same pinned oracle image and read-only source mount, run:

```text
python /workspace/cpp/porting/compare_topological_metric_calculator.py \
  --cpp /workspace/cpp/.docker-smoke/build-port-release/porting/vne_topological_metric_calculator_harness \
  --python-source /src/virne/virne/network/topology/topological_metric_calculator.py \
  --warmups 5 --repetitions 31 --workers 8
```

`--workers 0` validates the production automatic policy. The canonical
explicit-width run and automatic validation both require exact optional fields,
shape, exceptions, all float32 payloads, corpus metadata, and timed checksums.

Run the complete worker sweep with:

```text
python /workspace/cpp/porting/sweep_topological_metric_workers.py \
  --cpp /workspace/cpp/.docker-smoke/build-port-release/porting/vne_topological_metric_calculator_harness \
  --workers 1,2,3,4,5,6,7,8 \
  --warmups 3 --repetitions 11 --rounds 3
```

After the full sweep, the recorded finalist confirmation used workers
`1,4,5,6,7,8`, five warm-ups, 31 samples, and five rounds.

The metric comparator direct-loads the exact 4,261-byte Python leaf source,
pins its SHA-256 and NumPy/NetworkX versions, rejects Torch imports, and compares
92 cases without numeric tolerance. The sweep rotates and reverses worker order
between rounds and rejects any checksum drift.

## ClassDict gate

The current verified ClassDict Release build is `/work/build`. Configure and
build its isolated targets in the persistent toolchain container with:

```powershell
docker start virne-cpp-dev
docker exec virne-cpp-dev cmake -S /work -B /work/build `
  -DCMAKE_BUILD_TYPE=Release
docker exec virne-cpp-dev cmake --build /work/build -j 4 `
  --target vne_class_dict_unit vne_class_dict_harness
docker exec virne-cpp-dev ctest --test-dir /work/build `
  -R '^vne_class_dict_unit$' --output-on-failure
```

Using the same oracle image, read-only mounts, disabled network, thread
environment, and eight-CPU cpuset as the earlier gates, run the canonical
comparison with:

```text
python /workspace/cpp/porting/compare_class_dict.py \
  --cpp /workspace/cpp/build/porting/vne_class_dict_harness \
  --python-source /src/virne/virne/utils/class_dict.py \
  --workers 8 --warmups 5 --repetitions 31
```

Repeat with `--workers 0` to validate the production automatic policy. The
comparator pins the exact Python source SHA-256, direct-loads only that leaf,
compares 16 tagged data/identity cases exactly, and rejects any timed checksum
drift.

Run the final worker sweep with:

```text
python /workspace/cpp/porting/sweep_class_dict_workers.py \
  --harness /workspace/cpp/build/porting/vne_class_dict_harness \
  --workers 1,2,3,4,5,6,7,8 \
  --warmups 5 --repetitions 31 --rounds 5 --batch-items 512
```

The sweep always includes automatic mode, rotates and reverses execution order,
and requires invariant checksums across every worker and round. A separate
`--batch-items 64` run checks the sequential side of the 8,192-top-level-field
automatic threshold for the 64-field fixture. Use an explicit width for
nested-heavy data because the automatic estimate intentionally counts only
top-level fields.

## Setting gate

The complete API, compatibility boundaries, and exact three-root fixture list
are in `components/setting.md`.  Build only the isolated targets, then run the
41-fixture/17,513-float canonical comparator in the pinned oracle container:

```text
cmake --build /work/build -j4 --target vne_setting_unit vne_setting_harness
ctest --test-dir /work/build -R '^vne_setting_unit$' --output-on-failure
python /workspace/cpp/porting/compare_setting.py \
  --cpp /workspace/bin/vne_setting_harness \
  --python-source /workspace/src/virne/utils/setting.py \
  --fixture-root /workspace/src/settings \
  --fixture-root /workspace/src/test-settings \
  --fixture-root /workspace/src/cpp-settings \
  --expected-fixtures 41 --workers 8 --warmups 5 --repetitions 31 \
  --operations 256 --batch-operations 2 --batch-size 64 \
  --id-iterations 1000000 --random-float-cases 16384 \
  --json-output /workspace/setting_compare.json
python /workspace/cpp/porting/sweep_setting_workers.py \
  --harness /workspace/bin/vne_setting_harness \
  --fixture /workspace/src/settings/main.yaml \
  --workers 1,2,3,4,5,6,7,8 --warmups 5 --repetitions 31 \
  --rounds 3 --batch-size 256 --operations 2
```

Both commands require exact output/checksum invariants before timing is
interpreted.  The accepted artifacts are under `results/` with date
`2026-07-28`.

## Stats gate

`stats` is a standard-library-only production leaf.  Its multi-worker test
uses independent wrappers solely as an external throughput exercise; the
production API intentionally has no worker control.

```text
cmake --build /work/build -j4 --target vne_stats_unit vne_stats_harness
ctest --test-dir /work/build -R '^vne_stats_unit$' --output-on-failure
python /workspace/cpp/porting/compare_stats.py \
  --harness /workspace/bin/vne_stats_harness \
  --python-root /workspace/src
python /workspace/cpp/porting/sweep_stats_workers.py \
  --harness /workspace/bin/vne_stats_harness \
  --python-root /workspace/src --iterations 20000 \
  --warmups 5 --repeats 31 --workers 1 2 4 8 \
  --json-output /workspace/stats_sweep.json
```

The sweep excludes thread construction/teardown and rejects any return,
output, byte-count, callable-count, or clock-count mismatch.  Full results are
in `results/stats_2026-07-28.md`.

## Manager gate

`manager` is a standard-library/platform-libc filesystem leaf. Build only its
isolated unit, differential harness, and benchmark against the existing build
tree; completed dependencies do not need rebuilding:

```text
cmake --build /work/build -j4 --target \
  vne_manager_unit vne_manager_harness vne_manager_benchmark
ctest --test-dir /work/build -R '^vne_manager_unit$' --output-on-failure
```

Using the same read-only source mounts, disabled network, `/tmp` tmpfs, and
eight-CPU policy described above, run the exact differential before timing:

```text
python /workspace/cpp/porting/compare_manager.py \
  --harness /workspace/bin/vne_manager_harness \
  --python-source /workspace/src/virne/utils/manager.py
python /workspace/cpp/porting/benchmark_manager.py \
  --harness /workspace/bin/vne_manager_benchmark \
  --python-source /workspace/src/virne/utils/manager.py \
  --warmups 5 --repetitions 31 \
  --json-output /tmp/manager_compare_2026-07-28.json
```

Both programs create and destroy only fresh temporary fixture trees. The
differential requires 24/24 exact compatibility cases and separately records
the mandatory `unsafe_path_escape` deviation for an algorithm-directory
symlink that Python would follow outside the supplied root. It compares native
sequential order, exact stdout, typed failure point, partial side effects, and
the complete before/after tree.

There is intentionally no manager worker sweep or production worker control.
Parallel enumeration/deletion would change stdout order, first-error order,
and which irreversible deletions precede a failure. The canonical
five-warmup/31-sample result passed all success-path speed gates; the legacy
exception-only row is report-only. Accepted results and binary/artifact hashes
are in `results/manager_2026-07-28.md`.

## Dataset core gate

The completed dataset core is the Torch-free helper/naming/path leaf. Build
only its isolated targets; do not rebuild completed graph/CSV/YAML components:

```text
cmake --build /work/build -j4 --target \
  vne_dataset_core_unit vne_dataset_core_harness
ctest --test-dir /work/build -R '^vne_dataset_core_unit$' --output-on-failure
```

Copy the final harness and pinned Python source into the existing non-ML oracle
container, then run exact differential before timing:

```text
python /workspace/cpp/porting/compare_dataset_core.py \
  --harness /workspace/bin/vne_dataset_core_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --json-output /workspace/dataset_core_differential_2026-07-28.json
python /workspace/cpp/porting/benchmark_dataset_core.py \
  --harness /workspace/bin/vne_dataset_core_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --warmups 5 --repetitions 31 \
  --workers 1 2 3 4 5 6 7 8 0 --scale 1 \
  --json-output /workspace/dataset_core_compare_2026-07-28.json
```

The comparator must report 60/60 core cases, 16,395 exact binary64 string
cases, and controlled fake Torch/OmegaConf only. The benchmark checks exact
checksum and output bytes for every sample; process/fixture/hash work is
excluded, while production thread creation/join is deliberately included.

Worker zero is automatic. Filename batches stay sequential below 16,384
items, then use up to five lanes. Physical/virtual path batches stay
sequential below 1,024, use up to four lanes below 3,072, and up to six lanes
thereafter. Explicit widths 1..8 must remain output-invariant. Accepted API,
timings, hashes, and the original leaf boundaries are in
`components/dataset.md` and `results/dataset_core_2026-07-28.md`.

## Dataset RNG gate

The completed RNG leaf reuses `random_lib`; do not rebuild or modify the frozen
RNG implementation. Build only the wrapper targets against the existing build
tree:

```text
cmake --build /work/build -j4 --target \
  vne_utils_dataset_rng vne_dataset_rng_unit vne_dataset_rng_harness
ctest --test-dir /work/build -R '^vne_dataset_rng_unit$' --output-on-failure
```

Run the exact oracle and Python optimized-boundary check before timing:

```text
python /workspace/cpp/porting/compare_dataset_rng.py \
  --harness /workspace/bin/vne_dataset_rng_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --json-output /workspace/dataset_rng_differential_2026-07-28.json
python -O /workspace/cpp/porting/check_dataset_rng_optimized.py \
  --python-source /workspace/src/virne/utils/dataset.py \
  --json-output /workspace/dataset_rng_optimized_2026-07-28.json
```

The canonical comparison and final worker-policy sweeps are:

```text
python /workspace/cpp/porting/benchmark_dataset_rng.py \
  --harness /workspace/bin/vne_dataset_rng_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --warmups 5 --repetitions 31 --workers 1 0 2 3 4 5 6 7 8 \
  --performance-gate \
  --json-output /workspace/dataset_rng_compare_2026-07-28.json
python /workspace/cpp/porting/benchmark_dataset_rng.py \
  --harness /workspace/bin/vne_dataset_rng_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --warmups 5 --repetitions 31 --workers 1 0 2 3 4 5 6 7 8 \
  --scale 0.64 --kinds exponential_int exponential_bool \
  --worker-policy-gate \
  --json-output /workspace/dataset_rng_worker_medium_2026-07-28.json
python /workspace/cpp/porting/benchmark_dataset_rng.py \
  --harness /workspace/bin/vne_dataset_rng_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --warmups 5 --repetitions 31 --workers 1 0 2 3 4 5 6 7 8 \
  --scale 2 --kinds exponential_int exponential_bool \
  --worker-policy-gate \
  --json-output /workspace/dataset_rng_worker_large_2026-07-28.json
```

Worker widths are interleaved and their order is rotated/reversed at every
sample so thermal drift cannot masquerade as an automatic-policy win. Every
sample gates exact Adler-32, output bytes, and the following two RNG values
before timing is accepted. Thread construction/join and allocation are timed;
process startup, seeding, checksum, and continuation verification are not.
`--worker-policy-gate` accepts only both exponential cast kinds, exact widths
0..8, seed 123, scale 0.64 or 2, and at least five warm-ups/31 repetitions; an
incomplete invocation cannot pass vacuously.
Accepted API, worker thresholds, hashes, and results are in
`components/dataset.md` and `results/dataset_rng_2026-07-28.md`.

## Dataset XML/GML gate

The completed XML leaf reuses the frozen graph API and pinned Boost 1.85
RapidXML header. Do not rebuild or modify graph/CSV/config/yaml-cpp/random.
Build only the isolated leaf tests and harness against the existing tree:

```text
cmake --build /work/build -j4 --target \
  vne_dataset_xml_unit vne_dataset_xml_harness
ctest --test-dir /work/build -R '^vne_dataset_xml_unit$' --output-on-failure
```

Run exact differential before timing. The checked-out `Brain.gml` has CRLF
normalization; the comparator independently generates the canonical Linux LF
oracle and therefore does not mistake checkout line endings for serializer
parity:

```text
python /workspace/cpp/porting/compare_dataset_xml.py \
  --harness /workspace/bin/vne_dataset_xml_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --brain-xml /workspace/src/datasets/topology/Brain.xml \
  --brain-gml /workspace/src/datasets/topology/Brain.gml \
  --json-output /workspace/dataset_xml_differential_2026-07-28.json
```

The canonical Python/C++ comparison and complete worker sweep are:

```text
python /workspace/cpp/porting/benchmark_dataset_xml.py \
  --harness /workspace/bin/vne_dataset_xml_harness \
  --python-source /workspace/src/virne/utils/dataset.py \
  --brain-xml /workspace/src/datasets/topology/Brain.xml \
  --workers 1 0 2 3 4 5 6 7 8 \
  --warmups 5 --repetitions 31 --batch-documents 16 \
  --performance-gate --worker-policy-gate \
  --json-output /workspace/dataset_xml_benchmark_2026-07-28.json
python /workspace/cpp/porting/sweep_dataset_xml_workers.py \
  --harness /workspace/bin/vne_dataset_xml_harness \
  --brain-xml /workspace/src/datasets/topology/Brain.xml \
  --documents 2 4 8 16 32 64 \
  --workers 1 0 2 3 4 5 6 7 8 \
  --warmups 5 --repetitions 31 --policy-gate \
  --json-output /workspace/dataset_xml_worker_sweep_2026-07-28.json
```

Worker widths are rotated/reversed and Python placement alternates across
samples. Parse, allocation, thread construction/join, graph materialization,
and the production GML write are inside their corresponding timers; fixture
creation, process startup, cleanup, and checksum calculation are outside.
Every accepted sample first gates exact graph/output bytes.

Worker zero is the measured document-count/first-file-size policy documented in
`components/dataset.md`. Explicit widths 1..8 are affinity- and count-capped.
The worker-policy gate requires the exact widths, five warm-ups, 31 samples, and
16 documents; the sweep gate requires the exact six document counts. In the
accepted results, C++ is 27.806x to 149.433x faster on the reported Brain rows,
107.445x faster for automatic synthetic batch parsing, and 37.787x faster for
full Brain XML-to-GML. Accepted API, typed safety boundaries, hashes, and all
evidence are in `components/dataset.md` and
`results/dataset_xml_2026-07-28.md`.

## Attribute-method gate

Read `components/attribute_method.md` before using the completed leaf. Build
only its isolated targets:

```text
cmake --build /work/build -j8 --target \
  vne_attribute_method_unit vne_attribute_method_harness
ctest --test-dir /work/build -R '^vne_attribute_method_unit$' \
  --output-on-failure
```

Run the broad semantic gate first, then a compact representative timing signal:

```text
python /workspace/cpp/porting/compare_attribute_method.py \
  --source /workspace/src/virne/network/attribute/attribute_method.py \
  --harness /workspace/bin/vne_attribute_method_harness \
  --output /workspace/attribute_method_differential_2026-07-28.json
python /workspace/cpp/porting/benchmark_attribute_method.py \
  --source /workspace/src/virne/network/attribute/attribute_method.py \
  --harness /workspace/bin/vne_attribute_method_harness \
  --count 10000 --warmups 1 --repetitions 3 \
  --output /workspace/attribute_method_benchmark_2026-07-28.json
python /workspace/cpp/porting/sweep_attribute_method_workers.py \
  --source /workspace/src/virne/network/attribute/attribute_method.py \
  --harness /workspace/bin/vne_attribute_method_harness \
  --kinds hard_le_double --counts 4000000 \
  --warmups 1 --repetitions 3 \
  --output /workspace/attribute_method_worker_sweep_2026-07-28.json
```

The comparator must pass 123 combined cases; timing is deliberately compact
unless output differs or C++ is unexpectedly slow. Worker count is a typed
caller config, not an automatic policy: zero/one are sequential and wider
values are count/affinity capped. Accepted API, coverage, hashes, and results
are in `components/attribute_method.md` and
`results/attribute_method_2026-07-28.md`.

## BaseAttribute gate

Read `components/base_attribute.md`; do not reopen completed dependency source.
Build only the isolated leaf targets:

```text
cmake --build /work/build -j8 --target \
  vne_base_attribute_unit vne_base_attribute_harness \
  vne_base_attribute_benchmark
ctest --test-dir /work/build -R '^vne_base_attribute_unit$' \
  --output-on-failure
```

Run the exact direct-source gate before the deliberately short timing smoke:

```text
python /workspace/cpp/porting/compare_base_attribute.py \
  --source /workspace/src/virne/network/attribute/base_attribute.py \
  --dataset-source /workspace/src/virne/utils/dataset.py \
  --harness /workspace/bin/vne_base_attribute_harness \
  --output /workspace/base_attribute_differential_2026-07-28.json
python /workspace/cpp/porting/benchmark_base_attribute.py \
  --source /workspace/src/virne/network/attribute/base_attribute.py \
  --dataset-source /workspace/src/virne/utils/dataset.py \
  --benchmark /workspace/bin/vne_base_attribute_benchmark \
  --count 300000 --workers 1 2 8 --warmups 1 --repetitions 3 \
  --output /workspace/base_attribute_benchmark_2026-07-28.json
```

The accepted gate is 32 direct cases and six timing rows. All timing rows must
retain the exact output checksum and following RNG bits and beat Python. API,
worker semantics, hashes, and results are in `components/base_attribute.md`
and `results/base_attribute_2026-07-28.md`.

## NodeAttribute gate

Read `components/node_attribute.md`; all completed dependency APIs are frozen.
Build only the isolated leaf targets against the existing tree:

```text
cmake --build /work/build -j8 --target \
  vne_node_attribute_unit vne_node_attribute_harness \
  vne_node_attribute_benchmark
ctest --test-dir /work/build -R '^vne_node_attribute_unit$' \
  --output-on-failure
```

Historical accepted direct-source and compact timing commands (do not rerun the
frozen NodeAttribute benchmark):

```text
python /workspace/cpp/porting/compare_node_attribute.py \
  --source /workspace/src/virne/network/attribute/node_attribute.py \
  --base-source /workspace/src/virne/network/attribute/base_attribute.py \
  --method-source /workspace/src/virne/network/attribute/attribute_method.py \
  --dataset-source /workspace/src/virne/utils/dataset.py \
  --harness /workspace/bin/vne_node_attribute_harness \
  --output /workspace/node_attribute_differential_2026-07-28.json
python /workspace/cpp/porting/benchmark_node_attribute.py \
  --source /workspace/src/virne/network/attribute/node_attribute.py \
  --base-source /workspace/src/virne/network/attribute/base_attribute.py \
  --method-source /workspace/src/virne/network/attribute/attribute_method.py \
  --dataset-source /workspace/src/virne/utils/dataset.py \
  --benchmark /workspace/bin/vne_node_attribute_benchmark \
  --count 100000 --workers 1 2 8 --warmups 1 --repetitions 3 \
  --output /workspace/node_attribute_benchmark_2026-07-28.json
```

The accepted gate is 37 differential cases plus five recorded Python-only
boundaries and six timing rows. Every row checks exact output checksum/bytes;
position also checks RNG continuation. API, ID rules, hashes, and timings are
in `components/node_attribute.md` and
`results/node_attribute_2026-07-28.md`.

## LinkAttribute gate

Read `components/link_attribute.md`; all completed dependency APIs and the
accepted LinkAttribute benchmark are frozen. The isolated unit/differential
targets may still be used for regression, but do not rerun or update its timing
driver or result.

The accepted gate is 35 direct C++/Python cases plus five recorded Python-only
boundaries (40 total). Its six frozen timing rows cover dense edge roundtrip and
position-derived latency at workers `1/2/8`, with exact checksum/output-byte
gates. Dense roundtrip is `5.722x` to `7.408x` faster than Python and position
latency is `8.550x` to `72.780x` faster. API, ID rules, hashes, and timings are
in `components/link_attribute.md` and
`results/link_attribute_2026-07-28.md`.

## GraphAttribute gate

Read `components/graph_attribute.md`; its accepted benchmark is frozen and is
provenance only. Unit/differential/sanitizer/integration checks may be rerun,
but do not rerun or update the timing driver/result.

The accepted gate is 45 shared C++/Python cases, four native extension cases,
and five Python-only boundaries (54 total). The one frozen independent-graph
roundtrip workload covers workers `1/2/8`, exact raw64 checksum/output bytes,
and C++ speedups of `1.209x` to `2.595x`. API, binding identity rules, hashes,
and timings are in `components/graph_attribute.md` and
`results/graph_attribute_2026-07-28.md`.

## AttributeBenchmarkManager gate

Read `components/attribute_benchmark_manager.md`; its accepted benchmark is
frozen and is provenance only. Unit/differential/sanitizer/integration checks
may be rerun, but do not rerun or update its timing driver/result.

The accepted gate is 19 shared Python cases, seven native extension/error
cases, and six Python-only boundaries (32 total). The one frozen prepared-row
workload covers workers `1/2/8`, exact ordered UTF-8/raw64 output, and C++
speedups of `9.171x` to `11.477x`. API, compact-ID/thread/cache rules, hashes,
and timings are in `components/attribute_benchmark_manager.md` and
`results/attribute_benchmark_manager_2026-07-28.md`.

## AttributeFactory gate

Read `components/attribute_factory.md` before using the completed boundary.
The accepted benchmark is frozen and is provenance only; do not rerun or edit
its timing source, driver, binary, or JSON. Unit/differential/sanitizer and
integration regression checks remain allowed.

The accepted gate is 29 shared Python cases, three native typed cases, and
seven Python-only dynamic boundaries (39 total). The one frozen 32,768-spec
workload covers caller-configured workers `1/2/8`, exact ordered direct-field
output, and C++ speedups of `5.979x`, `6.416x`, and `4.822x`. API, compact-ID,
duplicate, worker/error semantics, hashes, and timings are in
`components/attribute_factory.md` and
`results/attribute_factory_2026-07-28.md`.

## BaseNetwork gate

Read `components/base_network.md` before using the completed model. Its
accepted benchmark is frozen provenance; never rerun or edit its source,
driver, binary, JSON, or timings. Correctness/sanitizer/integration regression
checks remain allowed.

The accepted gate is 29 exact shared Python cases plus 11 Python-only
boundaries (40 total). The frozen 8,192-element get/set/manager workload covers
workers `1/2/8`, raw type/bit/order gates, compact definition/value IDs, and the
measured sparse ordered manager adapter. Get is `3.245x-6.326x`, set is
`2.398x-6.495x`, and manager is `67.875x-103.983x` faster than Python. API,
threading, cache/clone/view/serialization rules, hashes, and timings are in
`components/base_network.md` and `results/base_network_2026-07-28.md`.

## LinkRank gate

Read `components/link_rank.md` before using the completed leaf. Its accepted
benchmark is frozen provenance; never rerun or edit its source, driver, binary,
JSON, result, or timings. Focused correctness and sanitizer regressions remain
allowed when a downstream change requires them.

The accepted exact differential is 12/12 shared cases at workers `1/2/8` with
raw binary64 score equality. The CPython 3.10.20 ordering probe passed 4,569
cases / 445,868 entries plus every qNaN/sNaN permutation. The frozen
131,072-edge x 8-resource FFD workload retained checksum
`10478239091350211214`; C++ was `2.133x`, `2.208x`, and `2.227x` faster at
workers `1/2/8`. API, boundaries, hashes, and timings are in
`components/link_rank.md`, `results/link_rank_2026-07-29.md`,
`results/link_rank_differential_2026-07-29.json`, and
`results/link_rank_benchmark_2026-07-29.json`.

## NodeRank gate

Read `components/node_rank.md` before using the completed leaf. Its accepted
benchmark and exact differential are frozen provenance; never rerun or edit
their drivers, binaries, JSON, result, or timings. Focused unit, sanitizer, and
integration regressions remain allowed when required by a downstream change.

The accepted exact differential is 13/13 shared cases at workers `1/2/8`, with
ordered node IDs, raw binary64 equality, NPS distance/score lanes, and NumPy RNG
continuation. Strict GCC 11, ASan/UBSan/leaks, targeted CTest, hot-ID review,
and the generic CPython 3.10.20 Timsort probe passed. The frozen 131,072-node x
8-resource FFD workload retained checksum `11449996351475094403`; C++ was
`1.277x`, `1.279x`, and `1.312x` faster at workers `1/2/8`. API, boundaries,
hashes, and timings are in `components/node_rank.md`,
`results/node_rank_2026-07-29.md`,
`results/node_rank_differential_2026-07-29.json`, and
`results/node_rank_benchmark_2026-07-29.json`.

## BaseSolver gate

Read `components/base_solver.md` before building a concrete solver. The
accepted differential and cold-start benchmark are frozen provenance; never
rerun or edit their source, drivers, JSON, result, or timings. Focused unit,
sanitizer, and integration regressions remain allowed when a downstream leaf
requires them.

The accepted exact differential is 13/13 shared cases at native workers
`1/2/8`. Unit/concurrency coverage, strict GCC 11 production/unit/harness,
ASan/UBSan/leaks, targeted CTest, aggregate solver integration, frozen
integrity, and the hot-ID audit passed. The frozen 32-descriptor / 4,096-holder
workload retained 419,498 bytes and checksum `13751587758314786690`; C++ was
`11.174x`, `9.906x`, and `10.401x` faster at workers `1/2/8`. API, boundaries,
hashes, and timings are in `components/base_solver.md`,
`results/base_solver_2026-07-29.md`,
`results/base_solver_differential_2026-07-29.json`, and
`results/base_solver_benchmark_2026-07-29.json`.

## Node-rank solver gate

Read `components/heuristic_node_rank.md` before consuming the completed typed
solver API. Both the original OrderRank artifacts and the later combined
eight-solver benchmark are frozen provenance; never rerun or edit their
benchmark source, drivers, JSON, results, or timings. Focused unit,
differential, sanitizer, and integration regressions may still run when
required by a downstream solver leaf.

The focused unit/concurrency, strict/sanitizer/CTest/frozen-integrity and
hot-ID gates passed. The exact AST-isolated differential passed six shared
cases at native workers `0/1/2/8`, covering five Solution paths plus typed
empty-virtual rank precedence. The frozen benchmark is deliberately labeled a
conservative mixed-dependency microbenchmark: Python executes the exact target
class AST with lightweight deterministic rank/mapper doubles, while native
executes the production solver pipeline. Its exact gate is 64 outputs, 87,752
bytes, and checksum `9328970994111537605`.

Python sequential time was 87,527.734 ns/solve. Native worker 1 was 56,442.344
ns/solve, or `1.551x` faster; explicit workers 2 and 8 were slower at
1,957,763.281 and 6,946,745.063 ns/solve. Those widths remain deterministic
caller configuration, not an automatic policy or acceptance requirement for
this small fixture. API, boundaries, and frozen evidence are in
`components/heuristic_node_rank.md`,
`results/heuristic_node_rank_differential_2026-07-29.json`, and
`results/heuristic_node_rank_benchmark_2026-07-29.json`.

The combined differential passes ten cases for all eight registered node-rank
solvers at workers `0/1/2/8`; every workers=1 row in its frozen compact fixture
beats Python. Its artifacts are the
`results/heuristic_node_rank_variants_*_2026-07-30.json` files.

## Complete heuristic registry gate

Read `components/heuristic_registry.md` before using or extending heuristic
solvers. The central registry has 14 direct IDs: all eight classes grouped in
`node_rank.py`, all three BFS solvers and all three joint place-route solvers.
The unfinished undecorated `ego_network.py`/`fit.py` prototypes and C++ stubs
without Python provenance are intentionally not compatibility entries.

The single collective Docker Release CTest is 1/1 PASS. It locks the complete
catalog, runs every factory, compares exact workers=1/4 output and RNG
continuation, and covers BFS/joint partial rollback. Its compact native timing
was 70.8778 ms at workers=1 and 36.7232 ms at workers=4 (1.930x faster). The
frozen Python/C++ node-rank evidence was not rerun.

## Meta-heuristic solver gate

Read `components/meta_heuristic.md` before using the five typed registry
entries `ga_meta`, `sa_meta`, `ts_meta`, `pso_meta` and `aco_meta`. The focused
unit executes every algorithm and compares worker widths 1/4 for exact
solution signatures and `PyRandom` continuation. Candidate generation stays
on the coordinator; independent evaluations use the existing deterministic
executor. The single focused result is frozen in
`results/meta_heuristic_2026-08-03.md`; the tiny fixture is intentionally not a
production throughput claim.
