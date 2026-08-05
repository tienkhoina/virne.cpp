# Prepared graph observation sampling

## Status

Complete for the homogeneous, dual-graph, and fixed six-relation heterogeneous
observation batches used by Virne. Dynamic Python dictionary names terminate at
the conversion boundary; production sampling uses direct fields, typed node and
relation IDs, and fixed arrays only.

## API

Header: `virne/solver/learning/rl_core/graph_observation_sampler.h`

- `prepare_graph_indices` normalizes duplicate, reordered, and Python-style
  negative graph IDs once.
- `PreparedBatchObservationSampler` samples homogeneous `pyg_lite::Batch`.
- `PreparedDualGraphObservationSampler` samples `p_net`, `v_net`, and the three
  general tensor fields together.
- `PreparedHeteroGraphObservationSampler` samples the base graph and stores only
  the overridden `v-connect-v` and `p-connect-p` relations for the two augmented
  views. `view(HeteroObservationViewId)` exposes a shallow typed view.

Constructors validate and cache host pointer metadata once. `prepare_selection`
builds device-resident node/edge gathers and offsets outside the hot path.
`sample(prepared_selection, device)` then performs only `index_select`, in-place
endpoint offsets, six fixed relation slots, and an optional CPU/CUDA transfer.
Plans carry source tensor identities, so a plan from another same-sized batch is
rejected. Source tensors, plans, and returned shallow metadata/view tensors must
remain read-only while caller workers sample them concurrently.

## Verification and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_graph_observation_sampler_unit `
           vne_graph_observation_sampler_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_graph_observation_sampler_unit$' --output-on-failure
python porting/compare_graph_observation_sampler.py `
  --native build-rollout-win-msvc/porting/Release/vne_graph_observation_sampler_harness.exe `
  --device cpu --threads 1 --benchmark --warmups 1 `
  --repetitions 3 --iterations 10
```

Windows MSVC Release passed the focused unit and `99/99` bit-exact differential
records: homogeneous and dual batches, every general field, three heterogeneous
views, both node types, and all six relations. The comparator AST-extracts and
hash-pins Python's exact `get_observations_sample`. PyG is unavailable on this
host, so the function runs against a minimal `Data`/`Batch` boundary implementing
the same select-and-rebatch contract; the underlying PyG-lite batching kernels
have their own frozen Python/PyG differential gate.

The single frozen prepared-only `256 -> 128` homogeneous measurement excludes
cold plan construction and matches every output checksum. Native median was
`96,280 ns/call`; Python median was `1,894,850 ns/call`, a `19.681x` native
speedup. CUDA-compatible ATen dispatch and a conditional CUDA unit smoke are
present, but this host had no CUDA hardware gate. Independent minibatches may be
sampled by caller-owned workers without shared mutation.

Full evidence:
`porting/results/graph_observation_sampler_differential_2026-08-05.json`.
