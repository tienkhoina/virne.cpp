# RL rollout returns and advantages

## Status

The standalone numeric helpers from `virne/solver/learning/rl_core/buffer.py`
are ported behind `VIRNE_ENABLE_ML`.  They deliberately cover the three public
return modes used by the Python buffer (`gae`, `mc`, and `td`) and the two cost
return modes (`reachability` and `cumulative`).  Rollout storage, replay, and
training integration remain separate leaves.

## API

Header: `virne/solver/learning/rl_core/rollout_returns.h`

- `compute_returns_with_gae(items, values, dones, last_value, gamma, gae_lambda)`
  returns typed `ReturnAdvantageResult` with dense `returns` and `advantages`.
- `compute_returns_with_mc(items, dones, gamma)` computes discounted returns
  and resets the recurrence at each terminal step.
- `compute_returns_with_td(items, values, dones, last_value, gamma)` preserves
  Python's one-step TD-error values exposed as `returns`.
- `cost_return_method_from_name` resolves the two cold configuration names to
  `CostReturnMethod`; no string lookup is performed in a numeric loop.
- `compute_cost_returns(costs, dones, gamma, method)` accepts a const cost
  vector and returns a new dense vector.  Cumulative mode clamps negative costs
  in the calculation, matching Python's observable `costs` transformation for
  the returned values without mutating a caller-owned vector.

Length mismatches and unknown method names throw `RolloutReturnException` with
the fixed `RolloutReturnErrorCode`.  Empty input is valid and returns empty
vectors.  `dones` is a compact `uint8_t` terminal lane; callers should pass
0/1 values.  The typed length check is an intentional safer boundary for the
Python helpers' otherwise implicit zip/index behavior on malformed input.
Negative cumulative costs become zero, including non-finite values that fail
the Python `>= 0` test; the reachability reduction keeps Python's comparison
ordering for NaN values.

## Performance and threading contract

All recurrences run in reverse source order and use direct vector indices.
MC/TD/cost results are pre-sized rather than using Python's `insert(0)`; this
keeps the same values while making the native path O(n).  A single recurrence
cannot be split without changing floating-point order, so the helpers are
sequential by design.  Independent episodes or prepared buffers may be
scheduled by caller-owned workers; there is no hidden global executor or lock.

## Gate and benchmark

```text
docker exec virne-cpp-dev /work/build-ml-linux/porting/vne_rollout_returns_unit
docker exec virne-cpp-dev ctest --test-dir /work/build-ml-linux \
  -R '^vne_rollout_returns_unit$' --output-on-failure

# Windows MSVC equivalent used for the recorded gate:
cmake --build build-rollout-win-msvc --config Release \
  --target vne_rollout_returns_unit vne_rollout_returns_harness -- /m:4
ctest --test-dir build-rollout-win-msvc -C Release \
  -R '^vne_rollout_returns_unit$' --output-on-failure
```

The differential harness compares every element of the fixed mixed-terminal
fixture against a source-level Python oracle.  The Windows MSVC Release gate
passed `5/5` exact output groups and the focused CTest passed.  On the fixed
4,096-step GAE fixture (two warmups, five samples, 25 iterations), native
median time was `34,348 ns/call` versus Python `1,058,224 ns/call`
(`30.809x` Python/native).  The output checksum is
`13995964860393899778`.  Full reports are stored in
`porting/results/rollout_returns_differential_2026-08-05.json` and
`porting/results/rollout_returns_benchmark_2026-08-05.json`.
