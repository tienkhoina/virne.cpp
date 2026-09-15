# Native instance learning session

Updated 2026-09-09. The instance learning workflow has a passing differential
and benchmark gate. This is not a claim that the entire RL package is complete.

## API and ownership

`rl_core/learning_session.h` exposes `LearningSession::learn`, `solve`, `update`,
pending/update counters, parameter export and checkpoint byte payloads.
`InstanceEnvironment` owns physical/virtual networks, mapping and resource state.
The new session owns policy, optimizer, prepared observations and training state.

The public session/environment boundary is tensor-free and carries typed numeric
fields. Core graph targets retain their existing C++11 string ABI; LibTorch
targets retain ABI 0. Torch linkage is private to `vne_learning_session` so native
graph callers can link both without changing the ABI of frozen libraries.

Observation preparation reuses the accepted dense `TensorConvertor`, creates
PyG-lite batches, and checks schema consistency when collecting sampled records.
Registered policy choices resolve to typed variants before the episode loop.
Action selection, critic evaluation, environment transitions, terminal bootstrap,
GAE/MC/TD returns, successful rollout merge, threshold updates and final flush are
implemented. The initial CPU gate uses one Torch thread and one CPU affinity.
Episode order and RNG state are observable, so this workflow remains sequential.

## Differential evidence

| Report | Cases | Byte records | Result |
| --- | ---: | ---: | --- |
| `results/learning_session_ppo_2026-09-09.json` | 1 | 78 | PASS |
| `results/learning_session_remaining_2026-09-09.json` | 10 | 780 | PASS |
| `results/learning_session_checkpoint_2026-09-09.json` | 3 | 90 | PASS |
| `results/learning_session_dqn_2026-09-09.json` | 5 | 354 | PASS |

The eleven new cases cover PG, A2C, PPO, ARPPO, the source's PPO-like DDPG update,
CNN, GCN/MLP, GAT/MLP, attention, dual GCN, and an infeasible instance. Each case
compares three learning episodes, an explicit final update, greedy solve and
sampled solve. Outputs include actions, values, log probabilities, rewards,
returns, terminal state, pending counts, update count, loss and all parameters.

The oracle executes the original `InstanceAgent.learn_with_instance`,
`TensorConvertor`, `RLSolver.select_action`/`estimate_value`, `RolloutBuffer` and
actual update methods. Source hashes and the 34-file environment loader evidence
are recorded. Existing accepted test suites were not executed again.

The original Python checkout stays read-only. Confirmed environment fixes from
the instance gate apply. An additional in-memory oracle correction maps constant
average distances to zero: the source divides by `max - min == 0`, creates NaN
policy logits and raises from `Categorical`. C++ now corrects this branch in
`obs_handler.cpp`; valid nonconstant normalization is unchanged. Historical
observation gates that preserved these NaNs are superseded for this branch only.

## Frozen benchmark

`results/learning_session_benchmark_2026-09-09.json` reuses the passing PPO report;
`differential_rerun` is false. It measures three sequential PPO episodes plus
flushing remaining rollout. Model/environment construction, one warm solve,
container startup and output serialization are excluded on both sides.

| Runtime | Median | p95 |
| --- | ---: | ---: |
| C++ | 7.887987 ms | 8.920421 ms |
| Python | 94.315164 ms | 99.578296 ms |

Measured Python/C++ ratio: **11.956810x**. Three warmups, five timed samples,
Torch 2.6.0 CPU, one thread, CPU affinity `[2]`. Final actions, loss and parameters
match exactly; both output SHA256 values are
`6b7c6cc4ae15d7cdc4dc5c5b23b47646e63d30dd1cefe520c16f5fde18cbd6f9`.
These small-fixture measurements do not predict large-network or GPU speedups.

## Checkpoints and remaining work

Checkpoint implementation now serializes model buffers/parameters, Adam moments
and groups, CPU RNG, counters, configuration, and pending dense/graph/recurrent
observations. Restore uses a temporary session and preserves the current state
and RNG on failure. Three new resume cases (MLP, GCN and dual-GCN) compare the
next two episodes against the uninterrupted Python workflow, including optimizer
updates across a pending batch. Native assertions also compare restored and
uninterrupted sessions, reject a malformed archive and verify state/RNG rollback.
These checks add 90 exact records without rerunning the eleven accepted cases.
Checkpoint serialization is a native extension; its archive bytes are not
claimed to match Python's checkpoint format. Serialization is outside the PPO
benchmark workload.

The main ML runtime uses the `VIRNE_NATIVE_CHECKPOINT_2` framing. Each tensor is
stored with `torch::pickle_save`, so loading does not enter LibTorch's nested
TorchScript/JIT archive path. A checkpoint with optimizer state restores Adam
moments and groups; a checkpoint that marks the optimizer absent emits a
warning and keeps the algorithm's freshly initialized optimizer. Legacy or
invalid checkpoints are also warning-only at the main CLI boundary and leave
the run on native defaults.

The DQN/DoubleDQN branch now includes online/target policy updates, lazy
epsilon-greedy actions, next observations and NumPy RNG checkpoint state. Its
five integration/resume cases and frozen 15.776x episode benchmark are detailed
in `dqn.md`. Failed initialization is also checked separately: four invalid
attempts leave no partial policy or optimizer behind.

Recurrent episode integration and auxiliary cost/lambda learning still need their
own gates. Heterogeneous policies, safe-RL updates, searchers and the remaining
advanced registrations remain on the active backlog. Main CLI registration and
CPU train/evaluate/checkpoint orchestration are covered by
`../results/main_cli_learning_cluster_2026-09-15.json`.
The public session currently runs on CPU. No new MSVC or CUDA parity is claimed.

## Reproduction provenance

Do not rerun accepted gates or benchmark rows. The commands below document how
the new results were produced, using existing images and `build-ml-linux`.

```text
docker exec virne-rl-build cmake --build /work/build-ml-linux --target vne_learning_session_harness -j 2
docker exec --env LD_LIBRARY_PATH= virne-rl-oracle python /workspace/virne.cpp/porting/compare_learning_session.py --native /workspace/virne.cpp/build-ml-linux/porting/vne_learning_session_harness --output /workspace/virne.cpp/porting/results/learning_session_ppo_2026-09-09.json --cases ppo
docker exec --env LD_LIBRARY_PATH= virne-rl-oracle python /workspace/virne.cpp/porting/benchmark_learning_session.py --native /workspace/virne.cpp/build-ml-linux/porting/vne_learning_session_harness --parity-report /workspace/virne.cpp/porting/results/learning_session_ppo_2026-09-09.json --output /workspace/virne.cpp/porting/results/learning_session_benchmark_2026-09-09.json
```

The temporary oracle container mounts C++ at `/workspace/virne.cpp` and original
Python read-only at `/workspace/virne`. Its image is
`virne-python-oracle:py310-rl-cpu`; affinity is CPU 2 and numeric-library thread
counts are 1. Keep `LD_LIBRARY_PATH` empty for Python. The comparator sets the
native child's library path to the checkout's LibTorch directory. Setting that
path globally causes Python to import the C++ distribution's incompatible
`libtorch_python.so`. No new image or duplicate build tree was needed here.
