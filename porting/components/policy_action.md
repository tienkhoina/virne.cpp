# Typed policy action and evaluation boundary

## Status

Complete for the inference/evaluation surface used by `BaseRLSolver`:
`get_action_prob_dist`, `select_action`, and `evaluate_actions`. The C++ API
keeps tensors on their current CPU/CUDA device and preserves batch shape instead
of copying through NumPy.

## API

Header: `virne/solver/learning/rl_core/policy_action.h`

- `PolicyActionOptions` stores temperature, mask enablement, and maskable-policy
  choice as direct fixed fields.
- `policy_action_distribution` returns candidate logits and their probabilities
  under no-grad semantics.
- `select_policy_action` performs typed sampled or greedy selection through
  `PolicyActionMode`; sampled actions always use candidate logits, matching
  Python. Log probability uses the candidate or raw distribution according to
  the two fixed policy flags.
- `evaluate_policy_actions` returns optional values, log probability, entropy,
  masked-action mass, and predictor BCE through fixed fields.

The implementation preserves two source quirks exactly: evaluation applies a
present mask even when `mask_actions` is false, and its candidate softmax does
not apply temperature. Categorical probability normalization, epsilon clamp,
entropy, and multinomial RNG continuation match Torch 2.6. Float16, BFloat16,
Float32, and Float64 logits are accepted. Actions must have the exact logits
batch shape without the event dimension; this is the Virne rollout invariant
and avoids a broadcast allocation in the hot path.

No string/map lookup or hidden worker exists. Use batched logits for throughput;
independent caller workers may evaluate disjoint tensors. Sampling through the
default Torch generator has the same shared-RNG contract as Python, so ordered
determinism should use one batched sampling call rather than concurrent calls.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_policy_action_unit vne_policy_action_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_policy_action_unit$' --output-on-failure
& .\.deps-cache\python-torch26\Scripts\python.exe `
  porting\compare_policy_action.py `
  --native build-rollout-win-msvc\porting\Release\vne_policy_action_harness.exe `
  --device cpu --threads 1 --benchmark --warmups 1 `
  --repetitions 3 --iterations 10
```

Windows MSVC Release passed the focused unit and `23/23` exact records against
AST-extracted, hash-pinned Python source running Torch `2.6.0+cpu`. Coverage
includes distribution, greedy/sample RNG, masks and policy flags, values,
entropy, auxiliary metrics, predictor BCE/autograd, scalar Float64, plus native
FP16/BF16 and conditional CUDA smoke checks.

The single frozen `256x256` evaluation measured `579,090 ns/call` native versus
`1,027,340 ns/call` Python, a `1.774x` native speedup, with all three output
checksums equal. CUDA-compatible ATen dispatch is present; this host gate used
the installed CPU LibTorch package. Full evidence:
`porting/results/policy_action_differential_2026-08-06.json`.
