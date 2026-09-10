# Active RL completion work

Updated 2026-09-10. This supplements the frozen 2026-09-08 continuation report.
The objective remains completing the RL port, not only adding policy leaves.

## User constraints

- Preserve every original Python source file under sibling `../virne`.
- Reuse completed tests/benchmarks as evidence; do not rerun closed gates.
- Match valid Python output/state/gradients exactly at a fixed Torch thread width.
- Confirmed Python bugs may be corrected in C++. Record the evidence and use
  explicit corrected expectations for those cases, without editing Python source.
- Reuse existing Docker images/build trees and keep disk usage small.
- Finish a coherent cluster, run its new cases together, and only repair
  behavior when a failing case demonstrates the discrepancy.

## Current state

| Area | State | Evidence / next action |
| --- | --- | --- |
| Earlier neural/PyG and CNN/GNN-MLP gates | Frozen historical evidence | `results/learning_continuation_2026-09-08.md`; do not rerun |
| Attention, dual-GNN (including cost/lambda), legacy/configurable seq2seq | Differential and benchmark PASS | 27 cases, 3655 byte records; 25 unchanged-source cases and two explicit residual-initialization corrections; `results/rl_policy_remaining_benchmark_2026-09-09.json`: 3.445x |
| Adam + PG/A2C/PPO/ARPPO/source-DDPG update loop | Differential and benchmark PASS | 11 cases and 984 exact byte records; `results/training_update_benchmark_2026-09-09.json`: 3.668x; benchmark reused the passing parity report |
| C++ repairs for confirmed source bugs | Focused regression gates PASS | `components/native_repairs.md`: 3 new cases, 135 records for single-layer MLP, natural DeepEdge initialization and recurrent clamp; DQN gate covers feature-last BatchNorm; session covers constant-distance normalization |
| Shared optimizer | Native worker wrapper differential and benchmark PASS | `components/shared_adam.md`; 3 cases, 156 records; four workers apply 64 packets with coherent snapshots; 9.028x measured sequential packet benchmark |
| Complete typed policy registry/builders | Implemented, verification pending | 16 source registry names; `policy_registry.{h,cpp}` and `policy_builder.{h,cpp}`; resolved dimensions and per-child optimizer groups |
| Heterogeneous encoder/policy and EdgeGNN | Pending | Existing typed HeteroData/HeteroConv are the substrate; lazy initialization/order needs its own gate |
| RL instance environment | Differential PASS, benchmark pending | `results/instance_environment_differential_2026-09-09.json`: 15 trajectories, 1890 byte records using 34 actual Python source files; original graph neighbor order is part of the fixture |
| Online RL environment + solver registry | COMPLETE within documented CPU scope | `components/online_learning.md`; 13 groups, 1361 byte records, 34 names, seven online transition cases. Environment 48.805x; registered PPO epoch 10.345x; final artifact `results/online_cluster_final_2026-09-10.json` |
| Native instance learning lifecycle | Differential and benchmark PASS; checkpoint resume PASS | `components/learning_session.md`; 11 end-to-end cases, 858 byte records, 11.957x measured PPO workflow; 3 additional checkpoint cases, 90 records, include pending rollout and RNG rollback |
| Agents/searchers/full learning lifecycle | In progress | Native learn/solve/update available; restart/beam/search state and main integration remain |
| DQN/DoubleDQN update variants | Differential and benchmark PASS | `results/dqn_update_differential_2026-09-09.json`: 4 cases, 1656 records; `results/dqn_update_benchmark_2026-09-09.json`: 7.103x |
| DQN/DoubleDQN CPU instance cluster | COMPLETE within documented scope | `components/dqn.md`; action 45, update 1656, session 354 and grouped checkpoint 520 records. One early-reseed defect fixed after failure; only that case rerun. DoubleDQN episodes 15.776x |
| Safe-RL update variants | Pending | Extend real update workflow with explicit numerical/state comparisons |
| Concrete solver registrations/full RL integration | 34 sequential names and native epoch complete; broader integration pending | Main CLI/config training, OnlineAgent admission loop, distributed/recurrent/safe families, searchers and remaining registrations still require their own clusters |

## Confirmed source defects and native choices

- `SharedAdam.step()` on the pinned Torch 2.6 raises:
  `API has changed, state_steps argument must contain a list of singleton tensors`.
  Native optimizer uses scalar tensor counters.
- `DeepEdgeFeatureGAT.weight_N` is created with uninitialized storage and no
  reset. C++ defaults to orthogonal initialization. Dual-GNN deliberately defers
  that extra reset because its parent already initializes all weights, preserving
  valid original dual-policy RNG/state behavior.
- MLP with feature-last rank-3 input sends the node dimension to BatchNorm1d's
  channel slot. C++ flattens leading dimensions while normalizing features.
- Original single-layer MLP indexes an empty hidden-dimension list. Native
  single-layer MLP is a linear output layer with the same orthogonal weight rule.
- Configurable seq2seq caps node IDs at 100 even when its embedding table is
  smaller. Native caps at the actual last embedding index.
- Python Controller candidate filtering computes link feasibility but does not
  use the resulting suitable-node set. The RL environment port must account for
  this confirmed bug in its implemented candidate masks.
- `ObservationHandler.get_average_distance` divides by zero when all distances
  are equal and nonzero. A newly sampled PPO trajectory produces NaN logits and
  Python `Categorical` raises. Native constant-distance normalization now returns
  zeros. The new session oracle patches only that branch in memory. Historical
  NaN-preserving observation results remain historical and were not rerun.

## Build state

After closing the 2026-09-10 online/registry cluster, Docker reports four images
using 2.341 GB, three containers using about 291 kB, no volumes and **zero build
cache**. The unused builder cache was pruned (Docker reported 426.4 MB). Existing
images/build trees and the reusable build/oracle containers were retained.

The authoritative final artifact is `results/online_cluster_final_2026-09-10.json`.
The preliminary `online_cluster_complete_*` registry timing and
`online_cluster_accepted_*` registry timing are invalid scope diagnostics, as
explained in `results/online_registry_benchmark_scope_correction_2026-09-10.md`.
The environment benchmark under `online_cluster_complete_*` is valid and reused.

`virne-cpp-dev` and `virne-rl-build` were restarted after Docker stopped.
The latter uses the same GCC image and a 96 MiB tmpfs over LibTorch headers.
After every container restart, refill that volatile cache before compiling:

```text
docker exec virne-rl-build cp -a /reference_libtorch_headers/. /work/libs/libtorch/include/
```

The workspace is still mounted at `/work`, and `build-ml-linux` is reused.
No new images are necessary for the current numerical gates. The temporary
`virne-rl-build` container should be removed after active build work completes.

Do not infer completion from a target merely building. Record actual parity,
corrected-case expectations, benchmark output checks and remaining integration
gaps before updating `PORTING_STATUS.md`.
