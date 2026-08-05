# Typed policy feature/config resolver

## Status

Complete for `PolicyBuilder.get_feature_dim_config`,
`PolicyBuilder.get_general_nn_config`, and the five dimension helpers in
`policy_builder.py`. This is a cold policy-construction utility only; it does
not build an optimizer or run RL training.

## API

Header: `virne/solver/learning/rl_core/policy_config.h`

- `PolicyFeatureFlags` stores the four feature switches as direct bool fields.
- `derive_policy_feature_dimensions(summary, flags)` consumes the shared typed
  `utils::FeatureConstructorSummary` and returns the five feature dimensions
  plus physical node count as direct `int64` fields.
- `validate_policy_general_network_config` forwards embedding dimension,
  dropout probability, and batch norm after cold validation.
- `derive_policy_configuration` combines both fixed structures.

The valid path has no string/map lookup, heap allocation, YAML access, or
worker. It preserves Python's intentional quirk that degree and extra topology
flags do not increase `v_node_x_dim`. Counts and arithmetic are checked before
conversion to LibTorch's signed shape domain. The shared summary lives in the
lightweight `virne/utils/feature_constructor_summary.h`, so this resolver does
not link or initialize config, random, graph, or LibTorch libraries.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_policy_config_unit vne_policy_config_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_policy_config_unit$' --output-on-failure
python porting\compare_policy_config.py `
  --native build-rollout-win-msvc\porting\Release\vne_policy_config_harness.exe `
  --benchmark --warmups 1 --repetitions 3
```

Windows MSVC Release passed the focused unit and `20/20` exact cases against
seven AST-extracted, hash-pinned Python APIs. Coverage includes the default
multi-resource dimensions `14/1/15/1/11/500`, all 16 flag combinations, zero
and multiple attributes, the `v_node` quirk, and general NN field forwarding.

The single frozen prepared batch of 65,536 configs measured a native median of
`21.510 ns/config` versus Python `5,519.547 ns/config`, a `256.600x` speedup,
with checksum `13714288253493205289` on both sides. Full evidence:
`porting/results/policy_config_differential_2026-08-06.json`.
