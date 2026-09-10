# Learning port continuation map

Source inventory: 2026-09-08. This is an API/dependency audit, not a test run.
Implementation progress updated 2026-09-10; detailed live gates are tracked in
`RL_COMPLETION_PROGRESS.md`.
The original Python files are read-only references. Frozen parity/timing
artifacts from earlier components have not been rerun.

`results/learning_api_inventory_2026-09-08.json` records every source hash,
import, declaration, signature and source line. It contains 71 Python files,
214 class declarations and 646 function/method declarations, including nested
factories and constructors. Those counts do not imply completion percentages.

Regenerate only the inventory when the Python source changes:

```text
python porting/audit_learning_modules.py --source ../virne/virne/solver/learning \
  --output porting/results/learning_api_inventory_2026-09-08.json
```

## Dependency order and remaining work

| Source group | Files | Current boundary | Remaining implementation/gate |
| --- | ---: | --- | --- |
| `neural_network` | 10 | Earlier primitives frozen; remaining valid leaves now have 27/27 fixed-thread exact cases and paired timing | Broken reference prototypes and unused generic PyG surface are explicit boundaries |
| `rl_policy` | 8 | MLP, CNN/GNN-MLP, attention, dual cost/lambda and recurrent leaves have exact gates; typed 16-name registry implemented | Complete registry/builder gate and recurrent episode integration |
| `rl_core` | 17 | Instance/online environments, policy-gradient and DQN sessions, checkpoints and SharedAdam have exact gates | Remaining graph/RNN conversion surface, OnlineAgent admission training, searchers, safe RL and priority replay |
| `reinforcement_learning` | 24 | 34 sequential solver names registered and a complete native online epoch verified | Main CLI/config training, remaining A3C/recurrent/safe registrations, hetero encoder and EdgeGNNConv, MCTS and advanced solvers |
| `unsupervised_learning` | 8 | GraphViNE message passing primitive frozen | GAE/VGAE/ARGA/ARGVA and negative sampling, clustering integration, Hopfield model/solver |
| Learning root | 4 | Observation helpers and selected `utils.py` functions frozen | Graph augmentation, remaining normalization/network conversion/link-pair helpers and export integration |

1. Finish focused gates for corrected single-layer MLP, natural DeepEdge residual
   initialization and small recurrent embedding IDs; reuse closed neural gates.
2. Complete recurrent/heterogeneous episode integration below
   `reinforcement_learning`, reusing the verified policy leaves and typed batches.
3. Complete remaining `rl_core/tensor_convertor.py` graph/RNN paths and
   `policy_builder.py` policy/optimizer construction. Existing samplers do not
   establish conversion parity for every raw Python observation shape.
4. The online transition/solver-registry cluster is closed with 13 groups,
   1,361 exact records and two paired benchmarks. Keep these gates frozen while
   adding OnlineAgent admission training and main CLI/config integration. The
   earlier instance gate covers 15 trajectories; its standalone benchmark is pending.
5. Complete `instance_agent.py` integration, `online_agent.py`, `safe_instance_agent.py` and
   `searcher.py`. Establish deterministic action/RNG and branch/restart order
   before adding workers.
6. Complete `safe_rl_solver.py`, distributed worker
   composition and replay storage. PG/A2C/PPO/ARPPO/source-DDPG, DQN/DoubleDQN
   updates, DQN episode/checkpoint integration and the native SharedAdam owner already have paired exact/timing
   evidence. Do not rerun those closed gates.
7. Connect `solver_maker.py`, concrete solver wrappers, hetero/CONAL, MCTS and
   advanced policies. Only then claim a native RL solver training/inference
   workflow in the main runtime.
8. Complete augmentation and the unsupervised subtrees with their own paired
   fixtures and benchmarks.

## Concrete subgroups

- Implemented policy leaves: `rl_policy/att_policy.py` (AttEncoder/Actor/Critic),
  `dual_gnn_policy.py` (NetEncoder, BiGnnBaseModel, nine policy variants),
  `gcn_seq2seq_policy.py` (encoder, decoder, attention, recurrent state),
  `base_policy.py` (registry/shared encoder composition). The duplicate
  `rl_core/base_policy.py` is a separate legacy implementation, not an alias.
- Training classes: PG, A2C, PPO, ARPPO, A3C, DPG/DDPG, DQN/DoubleDQN in
  `rl_core/rl_solver.py`; fixed-penalty, Lagrangian, neural/robust Lagrangian,
  reward-CPO and adaptive statewise PPO in `safe_rl_solver.py`.
- Environments: solution, placement, joint placement/routing, node-pair and
  node-slots variants exist in both instance and online modules.
- Searchers: random, greedy, one-sample, restart, multi-sample, beam,
  recoverable and one-shot. They depend on clone/rollback and policy state,
  not just categorical action sampling.
- Heterogeneous subtree: `hetero_gnn_solver/edge_gnn.py`, `net.py`,
  `instance_env.py`, `solver.py`. The existing HeteroConv and observation
  samplers cover the substrate, not these complete model/solver modules.
- Recurrent subtree: `gnn_seq2seq_solver/policy_with_encoder.py` and
  `gnn_seq2seq_solver.py`, plus `seq2seq_solver.py`.
- Other concrete wrappers: `mlp_solver.py`, `cnn_solver.py`, `gnn_mlp_solver.py`,
  `att_solver.py`, `dual_gnn_solver.py`, `safe_rl_solver/{net,solver}.py`,
  `advanced_solvers/multi_policies.py`, `mcts_solver/{node,mcts}.py`.
- Empty source placeholders: `advanced_policy.py`, `flag_solver.py` and
  `tracks.py`. Their empty contents are recorded; they are not working solvers
  awaiting an invented implementation. `__init__.py` exports are integration
  work, not independent numerical leaves.

## Python limitations to preserve explicitly

- `ResLinearNet` calls `super` with an unrelated class; construction fails.
- `NNConvNet` omits the required edge network; `PNAConvNet` passes a shifted
  constructor argument list. Neither supplies a valid reference forward.
  The separate custom `graph_conv.NNConv` also refers to missing `F_in`/`nn`.
- `GraphConvNet` passes `heads`, but its GAT factories read `num_heads`.
  Its viable wrapper always uses one head, even when the requested count is 7.
- `DeepEdgeFeatureGAT.weight_N` tensors are allocated without initialization.
  The native default now uses orthogonal initialization; explicit corrected
  expectations are used where needed. Dual-GNN defers this extra initialization
  because its parent already resets every weight.
- `DenseToSparse` modifies row/column indices after using them to gather edge
  weights. The new Python backward probe fails for differentiable edge weights;
  node-feature backward is a separate valid path. Native differentiable edge
  weights are an extension and are not claimed as shared Python behavior.
- GNN/MLP `batch_norm=True` applies BatchNorm1d to the dense node/head layout.
  Native feature-last normalization flattens the leading dimensions. The new
  DQN gate covers corrected output, gradient and running-buffer behavior while
  leaving every original Python file unchanged.
- `Seq2SeqActorCriticWrapper` lacks normal module initialization and invokes
  the wrapped constructor without an instance. Audit this constructor before
  assuming it supplies a valid recurrent policy reference.
- `SharedAdam` initializes `state['step']` as an integer, which Torch 2.6 rejects.
  Native scalar tensor counters and the omitted AMSGrad maximum lane have now
  passed a dedicated differential and benchmark gate.

## Acceptance contract

Each new leaf needs the full public API audit, an isolated native target,
exact paired output/state/gradient checks where Python supports them, and a
representative timing after correctness passes. Record unsupported or broken
reference branches separately. Resolve dynamic names once to enums/IDs and
keep hot paths on direct fields. CUDA hardware and full RL training remain
unverified until their own gates exist.
