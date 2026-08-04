# MLP actor/critic policy

`MlpActorCritic`, `Actor`, and `Critic` port the Torch-only policy classes in
`virne/solver/learning/rl_policy/mlp_policy.py`. They reuse the completed
`MLPNet` leaf and do not depend on BasePolicy, the solver registry, PyG,
feature construction, or training control.

## C++ API

`virne/solver/learning/rl_policy/mlp_policy.h` exposes the fixed schema:

```cpp
struct MlpActorCriticOptions {
    std::int64_t feature_dim;
    std::int64_t action_dim;
    std::int64_t num_layers = 3;
    std::int64_t embedding_dim = 128;
    double dropout_prob = 0.0;
    bool batch_norm = false;
};

MlpActorCritic policy(options);
torch::Tensor logits = policy->act(p_net_x);
torch::Tensor values = policy->evaluate(p_net_x);
```

`Actor` and `Critic` are also public module holders with
`forward(const torch::Tensor& p_net_x)`. For policy-shaped input
`p_net_x[B,N,F]`, Actor returns `[B,N]` by squeezing its final dimension and
Critic returns `[B,1]` by averaging over dimension 1. The composite exposes
public registered `actor` and `critic` modules.

The Python observation mapping is intentionally resolved outside this hot
boundary. C++ receives the fixed `p_net_x` Tensor directly; there is no string
map lookup in Actor, Critic, or the composite calls.

## Compatibility contract

Construction order and registration preserve Python checkpoint names:

```text
actor.net.lins.N.*
critic.net.lins.N.*
```

Actor is constructed before Critic, including all MLP default-bias and
orthogonal-weight RNG consumption. `action_dim` is validated and retained for
API/config compatibility but, exactly like Python, does not affect either
network. `dropout_prob` is forwarded to MLPNet and remains compatibility-only
because the original MLP does not create Dropout modules.

The supported production domain has positive feature/action dimensions and at
least two MLP layers. BatchNorm matches Python for rank-2 input. The original
BatchNorm layout is not compatible with the usual rank-3 policy Tensor, so the
normal policy configuration keeps `batch_norm=false`; the C++ port does not
silently change that behavior.

## Performance, threads, and devices

All configuration is typed and resolved during construction. The inference
path performs only direct module/Tensor operations. Configure LibTorch
intra-op/inter-op pools once before the first tensor operation. Do not combine
wide Torch pools with multiple outer policy workers without an explicit CPU
budget, because that oversubscribes the same cores.

The modules do not hard-code CPU. A matching CUDA LibTorch payload can move the
policy and input to CUDA without API changes; the accepted exact gate remains
Windows CPU until a CUDA runner records its separate deterministic tolerance
gate.

## Verification

Build only the policy targets in the existing ML build tree:

```powershell
cmake --build build-ml-win-msvc --config Release `
  --target vne_mlp_policy_unit vne_mlp_policy_harness --parallel 1
ctest --test-dir build-ml-win-msvc -C Release `
  -R '^vne_mlp_policy_unit$' --output-on-failure
```

Run the pinned differential and compact timing gate:

```powershell
& .\.deps-cache\python-torch26\Scripts\python.exe `
  .\porting\compare_mlp_policy.py `
  --harness .\build-ml-win-msvc\porting\Release\vne_mlp_policy_harness.exe `
  --python-source ..\virne\virne\solver\learning\rl_policy\mlp_policy.py `
  --mlp-source ..\virne\virne\solver\learning\neural_network\mlp.py `
  --threads 1 --warmups 3 --repetitions 5 --iterations 20
```

The comparator verifies both source hashes, AST-extracts only the three policy
classes with dependency-free base/registry stubs, and rejects any Virne/PyG
import. State, RNG continuation, input and outputs are compared bit-exactly
before timing is accepted.
