# Typed MLP policy construction adapter

## Status

Complete for `PolicyBuilder.build_mlp_policy` up to policy construction and
`.to(agent.device)`. Optimizer creation and all RL training state remain out of
scope.

## API

Header: `virne/solver/learning/rl_core/mlp_policy_builder.h`

- `make_mlp_actor_critic_options(configuration)` maps the prepared typed policy
  dimensions and general network settings to `MlpActorCriticOptions`.
- `build_mlp_policy(configuration, device)` constructs `MlpActorCritic` and
  moves every registered parameter and buffer to the requested LibTorch
  `torch::Device`.

The mapping preserves Python's fixed `feature_dim = p_net_x_dim +
v_node_x_dim`, `action_dim = p_net_num_nodes`, default three layers, embedding,
dropout, batch norm, constructor RNG consumption, parameter/buffer names and
training mode. Fixed fields are direct typed members; the valid path has no
string, map, YAML lookup or worker. CPU and CUDA use the same source. LibTorch
intra-op/inter-op thread pools remain caller-owned because construction is a
cold boundary.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_mlp_policy_builder_unit vne_mlp_policy_builder_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_mlp_policy_builder_unit$' --output-on-failure
python porting\compare_mlp_policy_builder.py `
  --native build-rollout-win-msvc\porting\Release\vne_mlp_policy_builder_harness.exe `
  --benchmark --warmups 1 --repetitions 3
```

Windows MSVC Release passed the focused unit and `3/3` exact cases against the
hash-pinned AST body of Python `PolicyBuilder.build_mlp_policy` plus the
already-pinned dimension/config helpers. The unit additionally checks direct
construction state, RNG continuation, output shapes, CPU placement, validation
and conditional CUDA placement.

The single frozen prepared batch of 65,536 typed configs measured a native
median of `48.996 ns/config` versus Python `13,288.304 ns/config`, a `271.212x`
speedup, with checksum `16947531349973059386` on both sides. This timing covers
only option resolution; module construction is verified for correctness but is
not re-benchmarked because the underlying `MlpActorCritic` leaf is frozen.
Full evidence:
`porting/results/mlp_policy_builder_differential_2026-08-06.json`.
