# Typed MLP inference session

## Status

Complete for the inference/evaluation composition used by the Python MLP
solver: dense observation conversion, actor/critic forward, action
distribution, greedy/sample selection, action evaluation and value estimate.
Optimizer, rollout storage and training remain out of scope.

## API

Header: `virne/solver/learning/rl_core/mlp_inference.h`

- `MlpInferenceSession(configuration, action_options, device)` constructs the
  typed policy once through `mlp_policy_builder` and enters evaluation mode.
- The overload accepting an existing `MlpActorCritic` reuses already loaded
  weights and moves them to the requested device.
- `forward`, `action_logits` and `values` accept either one
  `DenseObservation` or a `DenseObservationBatch`.
- `get_action_prob_dist`, `select_action` and `evaluate_actions` reuse the
  frozen typed `policy_action` boundary. `evaluate_actions` keeps Python's
  present-mask and temperature quirks through that shared API.
- `estimate_value(DenseObservation)` returns the Python scalar contract;
  `estimate_values(DenseObservationBatch)` keeps the batched tensor contract.

The session resolves all fixed fields before the hot path. No string/map lookup,
optimizer, hidden worker or per-request model construction is used. The
forward and value paths run under no-grad; caller-owned LibTorch thread pools
and independent batch workers remain supported. CPU and CUDA use the same
ATen source; this host gate used CPU LibTorch.

Shapes follow the existing leaves: raw `[B,N,Fp]` and `[B,Fv]` become MLP
`[B,N,Fp+Fv]`, actor logits are `[B,N]`, raw critic output is `[B,1]`, and
`evaluate_actions` returns squeezed values/log-probabilities/entropy `[B]`.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_mlp_inference_unit vne_mlp_inference_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_mlp_inference_unit$' --output-on-failure
& .\.deps-cache\python-torch26\Scripts\python.exe `
  porting\compare_mlp_inference.py `
  --native build-rollout-win-msvc\porting\Release\vne_mlp_inference_harness.exe `
  --benchmark --warmups 1 --repetitions 3 `
  --json-output porting\results\mlp_inference_differential_2026-08-06.json
```

MSVC Release passed the focused unit and `3/3` exact cases against the
hash-pinned Python MLP policy, `RLSolver` action methods and mask helper. The
cases cover single/batched observations, multi-resource-shaped features,
masked and unmasked greedy evaluation, output shapes and fixed tensor bits.

The single end-to-end `32 x 500` batch measured a native median of
`25,541,900 ns/call` versus Python `38,531,000 ns/call`, a `1.509x` native
speedup. Both timings include raw physical/virtual feature concatenation;
logits, values, log-probabilities and entropy checksums were identical.
Full evidence:
`porting/results/mlp_inference_differential_2026-08-06.json`.
