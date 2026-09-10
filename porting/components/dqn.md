# DQN and DoubleDQN

Updated 2026-09-10. **The CPU instance DQN/DoubleDQN cluster is complete within
the boundary below.** Exact gates and paired benchmarks cover the update,
action selection and checkpoint integration. Cluster acceptance adds five
new scenarios and reuses accepted reports without rerunning them.

## Native API

`rl_core/dqn_update.{h,cpp}` provides `DqnTrainer`. It accepts prepared batch
lanes and online/target callbacks. Target parameters and buffers are resolved
by name once at construction, then synchronization copies prebound tensors.
Both algorithms sample with replacement, compute the source Bellman target,
apply MSE loss, clip policy gradients, run Adam, and clear the supplied batch.
The callback runs after incrementing the update count and before buffer clear.

DoubleDQN chooses the next action using the online network, then gathers its
value from the target network. Source behaviors remain explicit: the target is
synchronized on update zero, target maxima are unmasked, target BatchNorm stays
in the source training mode, and this source's update clears its rollout buffer.
Persistent/prioritized replay is a separate remaining component.

`rl_core/dqn_action.{h,cpp}` provides epsilon-greedy selection with a lazy policy
callback. Random exploration does not execute the actor and samples the source's
full physical-node action range. Greedy selection honors the two mask flags.
The result includes the original Float32 `[1,1]` zero log-probability field.

`NumpyRandomState::snapshot/restore` adds MT19937 keys, position and Gaussian
cache state without changing any existing random algorithm. Invalid positions
are rejected before state mutation. No old random test or benchmark was rerun.

## Exact gates

| Report | Cases | Records | Result |
| --- | ---: | ---: | --- |
| `results/dqn_update_differential_2026-09-09.json` | 4 | 1656 | PASS |
| `results/dqn_action_differential_2026-09-09.json` | 4 | 45 | PASS |
| `results/learning_session_dqn_2026-09-09.json` | 5 | 354 | PASS |
| `results/dqn_cluster_complete_2026-09-10.json` (new cases) | 5 | 520 | PASS |

The update gate calls the original `DQNSolver.update` and
`DoubleDQNSolver.update` methods for four successive updates each, comparing
loss, counters, logging order, all policy/target parameters and buffers,
gradients, Adam moments and RNG tails. Two cases use unchanged Python behavior.
The two BatchNorm cases use the documented feature-last correction in memory:
flatten leading dimensions before BatchNorm, then restore the shape. This also
provides focused coverage for the earlier C++ MLP normalization repair.

The action gate calls the original epsilon schedule and selection methods. It
compares all actions, epsilon values, policy-call counts, RNG keys and counters
around a restored state. A separate new resume case retains the cached Gaussian.
`std::exp` initially differed from NumPy by one ULP at several schedule steps;
the pinned Torch Float64 CPU `exp` closes that native discrepancy. Python's
epsilon calculation was not patched or approximated.

The grouped gate checks restored BatchNorm in evaluation with distinct online
and target weights, a pending graph rollout flushed after restore, reseeding
before and after initialization, and rejection of an incompatible checkpoint.
Only early reseeding failed in the initial run: the setter changed construction
configuration and prevented restoration into a fresh session with the original
options. Native code now stores that pending seed as separate RNG state. Only
the failed case was rerun; the final report reuses the four passing cases and
verifies all 71 original learning source hashes. The initial failure remains
in `results/dqn_cluster_2026-09-10.json`.

Grouped gate: `validate_dqn_cluster.py` with `vne_dqn_cluster_harness`. Its final
report records C++ source hashes, native hash, reused evidence and exact output
hashes. No accepted benchmark or old test suite was rerun.

## Frozen update benchmark

`results/dqn_update_benchmark_2026-09-09.json` reuses the accepted differential
and times one complete DoubleDQN update, including target synchronization.
Policy/target creation, input setup and output serialization are excluded.
Three warmups and five samples run at one Torch thread in the CPU-2 oracle.

| Runtime | Median |
| --- | ---: |
| C++ | 0.401028 ms |
| Python | 2.848491 ms |

Measured Python/C++ ratio: **7.102973x**. Final loss and all parameters match;
the common output SHA256 is
`f213e5244a88351c5b2e5bd8f166b250521bf8535da0fc478e5d8ec345f11ad7`.
The complete DoubleDQN episode benchmark is recorded separately below.

`results/learning_session_dqn_benchmark_2026-09-09.json` measures three episodes
and a final rollout flush, excluding construction, one warm solve and output
serialization. It uses the same one-thread CPU-2 setup, three warmups and five
samples, reusing the accepted integration report.

| Runtime | Median | p95 |
| --- | ---: | ---: |
| C++ | 6.112622 ms | 9.769736 ms |
| Python | 96.434462 ms | 114.925197 ms |

Measured Python/C++ ratio: **15.776284x**. Actions, final loss, online and target
parameters have identical output SHA256
`a71faa09c2ebcc4ac5f2577b5f3562ea21af8a9b6c30fb9e4340a50b77d93d65`.
This is the small recorded fixture, not a general large-network speed estimate.

Build targets are `vne_dqn_update_harness` and `vne_dqn_action_harness`.
Oracles are `compare_dqn_update.py` and `compare_dqn_action.py`; only the former's
`--benchmark-from` mode was used for this frozen timing. Container/library paths
follow `learning_session.md`. The original Python checkout remains read-only.

## Integration boundary

`LearningSession` now has `UpdateAlgorithm::dqn` and `double_dqn`, with typed
`DqnConfiguration` for exploration and target intervals. It stores next
observations and terminal flags, and checkpoints add target state plus the
independent NumPy exploration stream. Three integration cases cover DQN,
DoubleDQN and GCN-DoubleDQN; two more compare native checkpoint continuation
against uninterrupted Python, including target parameters and exploration state.
The final cluster gate covers pending graph rollout, BatchNorm, reseed timing
and configuration mismatch, including subsequent training updates.
Recurrent DQN is explicitly rejected until that policy/agent combination is
implemented. Main solver registration, online events and priority replay remain
on the full RL backlog; no CUDA or new MSVC gate is claimed.
