# RL reward calculators

## Status

Complete for the four registered calculators in
`solver/learning/rl_core/reward_calculator.py`: `gradual_intermediate`,
`adaptive_intermediate`, `fixed_intermediate`, and `vanilla`.

## API

Header: `virne/solver/learning/rl_core/reward_calculator.h`

- `RewardCalculatorKind` is the fixed four-value registry enum.
- `RewardCalculatorOptions` stores the selected kind and direct
  `intermediate_reward`.
- `RewardCalculatorRegistry::kind_from_name` is the one-time config boundary;
  `create` returns a typed calculator and `list_registered` returns the fixed
  enum array.
- `BaseRewardCalculator::compute(p_net, v_net, solution)` updates both direct
  `Solution::v_net_reward` and the calculator's accumulated reward.
- `current_place_progress` uses direct `Solution::node_slots.size()` and the
  fixed virtual-network node count.

The physical network argument is retained for Python API compatibility; the
four current formulas do not inspect it. The unregistered commented
`get_node_load_balance` experiment remains deferred because it requires a
dynamic resource-attribute policy.

## Formula and threading contract

Successful requests return `v_net_r2c_ratio`. Partial placement/routing uses
the selected calculator's intermediate rule; failures use the corresponding
negative progress/weight. `intermediate_reward` accepts `-1` or a non-negative
number, matching the Python assertion. Zero-node adaptive requests are
rejected with a typed error instead of an untyped divide-by-zero.

The compute path is intentionally sequential: each call mutates one solution
and one calculator accumulator. Independent prepared calculators/solutions may
be run by caller-owned workers; no hidden global executor or string lookup is
introduced.

## Gate and performance

```text
docker exec virne-cpp-dev /work/build-ml-linux/porting/vne_reward_calculator_unit
docker exec virne-cpp-dev ctest --test-dir /work/build-ml-linux \
  -R '^vne_reward_calculator_unit$' --output-on-failure
```

The comparator passed `4/4` exact formula/registry outputs. A prepared fixed
reward call benchmark (two warmups, five samples, 1,000 iterations, one
thread) measured native `3.186 ns/call` versus Python `61.224 ns/call`, or
`19.217x` Python/native speedup. Full samples are in
`porting/results/reward_calculator_differential_2026-08-05.json` and
`porting/results/reward_calculator_benchmark_2026-08-05.json`.
