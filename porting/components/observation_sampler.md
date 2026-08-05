# Typed observation sampling

## Status

This leaf ports the tensor and fixed-dictionary paths of
`get_observations_sample` from `virne/solver/learning/utils.py`. Python
dictionary observations are replaced by the existing fixed
`DenseObservationBatch`, `GeneralTensorObservation`,
`MlpTensorObservation`, `MlpFcTensorObservation`, and
`AttentionTensorObservation` structures. Repeated sampling therefore uses
direct fields; no string or map lookup enters the hot path.

## API

Header: `virne/solver/learning/rl_core/observation_sampler.h`

- `sample_observations(tensor, indices, device)` selects a tensor batch.
- The fixed-observation overloads select every present field and preserve
  optional fields.
- `sample_observations(pyg_lite::Batch, indices, device)` selects graphs and
  rebuilds valid batch offsets through the frozen PyG-lite API.
- Indices must be a one-dimensional `int64` tensor. Duplicate, reordered, and
  Python-style negative indices are preserved.
- Undefined tensors, scalar sources, invalid indices, mixed-device fields, and
  mixed leading batch sizes raise `ObservationSampleException` with a typed
  error code.

Indices move to the source device and negative values are normalized once per
fixed observation. Each field then uses leading-dimension `index_select`, and
the result moves directly to the requested CPU or CUDA device. CUDA dispatch
is source-compatible but was not hardware-gated on this host.

The homogeneous `Batch` overload implements Python's valid graph-valued
dictionary branch by selecting graphs and rebatching them. The original
top-level `Batch` branch calls `.to()` on PyG's returned list and is malformed,
so that bug is not reproduced. Heterogeneous/dual-graph observation wrappers
remain a separate integration leaf. Homogeneous graph selection currently
uses the frozen PyG-lite cold extraction/rebatch boundary; a prepared offset
cache should replace its host metadata path before graph PPO training.

## Verification and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_observation_sampler_unit vne_observation_sampler_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_observation_sampler_unit$' --output-on-failure
python porting/compare_observation_sampler.py `
  --native .\build-rollout-win-msvc\porting\Release\vne_observation_sampler_harness.exe `
  --device cpu --threads 1 --warmups 1 --repetitions 3 --iterations 10 `
  --benchmark `
  --json-output porting/results/observation_sampler_differential_2026-08-05.json
```

Windows MSVC Release passed the unit and `15/15` exact Python differential
cases. The comparator AST-extracts the exact sibling
`utils.py:get_observations_sample` function and records its SHA-256. The single
frozen `4096 -> 1024` five-field batch measurement produced the same five
checksums: native median `59,080 ns/call`, Python median `234,540 ns/call`, or
`3.970x` native speedup. Independent batches may be
sampled by caller-owned workers; a single tensor gather is intentionally left
to LibTorch's configured execution backend.
