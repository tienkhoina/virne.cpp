# ObservationHandler and FeatureConstructor

Ported from the Python `virne.solver.learning.obs_handler` and
`virne.solver.learning.rl_core.feature_constructor` modules. The implementation
is split to mirror the source layout:

- `virne/solver/learning/obs_handler.h/.cpp`: stateless observation API;
- `virne/solver/learning/obs_handler_detail.h`: prepared ID-only hot-path API;
- `virne/solver/learning/rl_core/feature_constructor.h/.cpp`: feature registry,
  cached preparation, and feature concatenation.

## API

`ObservationHandler` exposes node/link attributes, link aggregation and
filters, bidirectional edge index/pairs, average distance, physical/virtual
status, virtual demands, and topological metric columns. Dense outputs are
row-major `DenseMatrixF32`, `DenseMatrixI64`, `DenseVectorF32`, or
`DenseVectorBool` and use NumPy-compatible shapes/dtypes.

`FeatureConstructorKind` has the fixed IDs `p_net`, `p_net_v_node`, and
`p_net_v_net`. Convert configuration names once with
`feature_constructor_kind_from_string`. Construct a move-only
`FeatureConstructor` with stable physical/virtual network references, then
call `construct(solution, curr_v_node_id)`. `construct_v_node_features` keeps
Python's terminal empty-vector behavior when `curr_v_node_id >= num_v_nodes`.

`FeatureConstructorOptions::workers` is required to be positive. Attribute
names are resolved to registry/graph IDs in the constructor and never appear
in feature loops. Benchmarks and topology metrics can be supplied through
`FeatureConstructorPreparedData`; otherwise they are prepared once.

The Python call-site keyword mismatch for aggregated `mean` is intentionally
preserved for differential compatibility: physical and virtual-network mean
columns remain unnormalized, while min/max use link benchmarks and sum uses
link-sum benchmarks.

## Verification

The frozen fixture covers multi-resource CPU/GPU/RAM plus BW/latency, all
optional flags, empty placement, terminal virtual-node behavior, status and
topological observations, and one-vs-four-worker determinism.

From the repository root inside `virne-cpp-dev`:

```sh
cmake -S /work -B /work/build-ml-linux
cmake --build /work/build-ml-linux \
  --target vne_feature_constructor_unit vne_feature_constructor_harness \
  --parallel 4
ctest --test-dir /work/build-ml-linux -R '^vne_feature_constructor_unit$' \
  --output-on-failure
```

Run the Python oracle with both repositories mounted (the oracle source is
never copied into the C++ tree):

```sh
docker run --rm \
  --mount type=bind,source="<repo>/virne.cpp",target=/work \
  --mount type=bind,source="<repo>/virne",target=/python \
  -w /work virne-cpu:latest \
  python /work/porting/compare_feature_constructor.py \
    --harness /work/build-ml-linux/porting/vne_feature_constructor_harness \
    --python-source /python/virne/solver/learning/rl_core/feature_constructor.py \
    --obs-source /python/virne/solver/learning/obs_handler.py \
    --json-output /work/porting/results/feature_constructor_differential.json
```

The harness also supports a bounded timing sample without changing the exact
fixture:

```sh
/work/build-ml-linux/porting/vne_feature_constructor_harness \
  --benchmark --threads 4 --warmups 2 --repetitions 5 --iterations 25
```

The frozen 2026-08-05 gate passes 8/8 exact cases and the native unit. With
one caller worker, the median prepared construction is 18,129.65 ns versus
404,280.15 ns in Python (22.299x Python/native time ratio). Four workers are
deterministic but slower on this six-node fixture (7,802,548.1 ns versus
453,117.85 ns); this is expected fan-out startup overhead, so worker width
remains a caller configuration for larger graphs. Both timings retain output
checksum `6515734063109927193`. Full JSON is in
`porting/results/feature_constructor_benchmark_t1_2026-08-05.json` and
`porting/results/feature_constructor_benchmark_t4_2026-08-05.json`.

Benchmark numbers are reported per repetition in nanoseconds; construction
setup (benchmark/metric preparation) is intentionally reported separately by
the caller and is not mixed into the hot-call timing.
