# RL rollout buffer storage and lifecycle

## Status

The scalar rollout storage/lifecycle surface from
`solver/learning/rl_core/buffer.py` is ported as a typed column store.  It
reuses the completed `rollout_returns` leaf for GAE/MC/TD and cost recurrences.
The unused Python `PrioritizedReplayBuffer` remains deferred: its source is
missing the NumPy import, has no active call site, and needs a separate RNG and
sampling contract.

## API

Header: `virne/solver/learning/rl_core/rollout_buffer.h`

- `RolloutBuffer` exposes the fixed basic lanes (`observations`, `actions`,
  `rewards`, `dones`, `next_observations`, `logprobs`, `values`), calculated
  `advantages`/`returns`, and the six known extension lanes as direct fields.
- `ObservationId` and `TensorId` are stable handles into caller-owned typed
  arenas.  Split/merge therefore move compact IDs and scalar columns rather
  than copying network graphs, tensors, or Python-style dictionaries.
- `add`, `reset`/`clear`, `reserve`, `get_subbuffer`, `split_with_ratios`,
  `split_with_instance`, and `merge` preserve Python ordering.  `curr_idx`
  increments only in `add`; subbuffers start at zero and merge does not adjust
  it, matching the original public side effect.
- `RolloutBufferMethod` selects GAE, MC, or TD without a string in any numeric
  loop.  `rollout_buffer_method_from_name` is the one-time config boundary.
- `RolloutExtensionId` covers `hidden_states`, `cell_states`, `action_mask`,
  `entropies`, `feasibility_flags`, and `feasibility_budgets`.  The first four
  are enabled by default like Python; `extend` resolves optional names once and
  resets that fixed lane.
- `RolloutBufferWithCost` adds the four safe-RL lanes and preserves cumulative
  in-place cost clamping plus repeated-call prepend behavior.
- `RolloutBufferWithAvgReturn` preserves the original GAE/TD mean-reward rules,
  including clearing the final terminal flag, GAE reward mutation, and TD's
  intentional lack of a gamma multiplier.

Malformed lane lengths, indices, ratios, methods, and extension states use
`RolloutBufferException` with a fixed `RolloutBufferErrorCode`.  Ratio splits
floor every partition except the final remainder; instance splits emit only
segments ending in a terminal transition and drop an incomplete tail.  They
also preserve Python's historical absolute-index slicing quirk when there are
three or more terminal transitions (the native implementation does this with
a bounded cursor, without repeated front erases).

The C++ subbuffer implementation deliberately keeps known extension/cost
schema instead of reproducing Python's subclass/dynamic-extension
`AttributeError`.  Repeated registration is idempotent at the schema level and
clears the lane, avoiding Python's duplicate-name double-merge bug.

Cost-buffer selection and merge preflight every populated lane before any
mutation. Repeated `compute_cost_returns` calls retain Python's complete
`k * transition_count` generations, and self-merge duplicates those generations
without out-of-range access. Base-class lifecycle and return calculation are
virtual so cost/average variants remain correct through a `RolloutBuffer&`.

## Performance and threading contract

All known columns are contiguous vectors.  Dynamic strings stop at the enum
resolver; every transition, select, split, merge, and recurrence loop uses
direct fields, compact IDs, and numeric indices.  `reserve` lets environment
code remove growth allocations before collection.

Mutation of one buffer and recurrence order are sequential and observable.
Independent episode buffers may be collected or transformed by caller-owned
workers, then merged in source order.  This leaf does not create a hidden
executor, lock, or machine-specific automatic worker policy.

## Gate and benchmark

```text
cmake --build build-rollout-win-msvc --config Release \
  --target vne_rollout_buffer_unit -- /m:1
cmake --build build-rollout-win-msvc --config Release \
  --target vne_rollout_buffer_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release \
  -R '^vne_rollout_buffer_unit$' --output-on-failure
```

The Windows MSVC Release gate passed `12/12` exact output groups and the
focused CTest.  On the prepared 4,096-row to 1,024-index subbuffer selection
(one warmup, three samples, ten iterations), native median time was
`18,280 ns/call` versus Python `571,100 ns/call` (`31.242x` Python/native).
The output checksum is `16108351205255056931`.  Full reports are stored under
`porting/results/rollout_buffer_differential_2026-08-05.json` and
`porting/results/rollout_buffer_benchmark_2026-08-05.json`.
