# CNN and GNN/MLP actor-critic policies

## Surface

`vne_learning_cnn_policy` ports the three active classes in `cnn_policy.py`:
`CnnActorCritic`, Actor and Critic. Native names are `CnnActorCritic`,
`CnnActor` and `CnnCritic` to avoid colliding with the existing MLP classes.
The input is the direct tensor `p_net_x`. Conv/ReLU/Flatten order, registered
`actor.net.0.*` / `critic.net.0.*` state and actor-before-critic construction
match Python. `action_dim` and `embedding_dim` are inert in this architecture.
CNN critic returns `[B]`, unlike the MLP/GNN critic's `[B,1]`.

`vne_learning_gnn_mlp_policy` ports GnnMlpEncoder, Actor, Critic and all three
registered GCN/GAT/DeepEdgeFeatureGAT policy classes from `gnn_mlp_policy.py`.
Native GcnMlpActorCritic/GatMlpActorCritic/DeepEdgeFeatureGATActorCritic wrappers
reuse one implementation selected by `neural_network::GnnKind`.
`GnnMlpObservation` has direct `p_net` Batch and `v_node_x` fields.

The encoder constructs the GNN before its virtual-node MLP and returns the
dense physical embedding plus the broadcast virtual embedding. Actor/Critic
each have independent encoders and heads. The actor squeezes the final
channel and the critic averages the dense node dimension, including padded
nodes, exactly as Python does. `p_net_num_nodes` is retained as an inert
constructor option. Batch sizes and maximum nodes come from prepared Batch
metadata; no per-layer string lookups or device-side size inference is added.

## Behavior boundaries

- Native libraries depend only on the existing C++ MLP/GNN/PyG-lite/LibTorch
  targets. Python/PyG is solely the external reference runtime.
- GNN `batch_norm=True` retains Python's dense node-axis behavior. The
  fixture covers both a matching hidden/node width and a configuration that
  raises a BatchNorm shape error. No architecture repair is applied.
- DeepEdgeGAT with middle layers requires explicit shared residual weights
  because the source leaves them uninitialized. The training fixture fills
  every parameter identically; the two-layer fixture compares natural seeded
  initialization and the exact RNG tail.
- Module `train/eval`, registered parameter/buffer state, dtype/device transfer
  and autograd are inherited from LibTorch. CUDA hardware is unverified.
- Complete policy/configuration registries, optimizer construction, agents and
  training-loop integration remain separate work. A policy class existing in
  C++ is not a completed RL solver.

## New verification and benchmark

The isolated `vne_cnn_gnn_mlp_policy_harness` builds with GCC strict warnings
(`-Wall -Wextra -Wpedantic -Werror`). The companion
`compare_cnn_gnn_mlp_policy.py` loads SHA-pinned Python definitions and the
actual BaseActorCritic/registry classes with package imports isolated.

Every tensor is serialized as name, dtype, shape and complete raw bytes;
comparison is literal byte equality, without numeric tolerance. Cases cover
CNN batched/unbatched execution, inert options, double backward, all three GNN
families in eval/train, actor/critic parameters, input/parameter gradients,
BatchNorm buffers and dropout RNG. Only compact SHA-256 summaries are persisted.

The new representative benchmark times paired CNN plus GAT/MLP actor/critic
forward calls with prebuilt inputs, three warmups, five repetitions and ten
iterations. Output bytes are checked outside timing. It is an aggregate
forward measurement, not evidence that every GCN/DeepEdgeGAT method or backward
pass is faster. Results live in
`porting/results/cnn_gnn_mlp_policy_*_2026-09-08.json`.

Accepted result: **12/12** cases and **577** byte-exact records at each fixed
thread width, 1 and 4. The aggregate forward benchmark is **3.820x / 3.780x**
faster than Python. Thread width changes some parity tensors; use a fixed width,
with one thread as the faster reproducible reference for this fixture. The
frozen table and hashes are in `porting/results/learning_continuation_2026-09-08.md`.
