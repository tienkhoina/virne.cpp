# Typed GCN network wrapper

## Status

Complete for Python `neural_network.gnn.GCNConvNet` and `GraphPooling` add,
mean and max modes. Generic NNConv/PNA, GAT wrappers and attention pooling are
separate leaves. No optimizer or RL training loop is included.

## API

Header: `virne/solver/learning/neural_network/gnn.h`

- `GCNConvNetOptions` stores all fixed dimensions, layer count, BatchNorm,
  dropout, dense-return and typed `GraphPoolingKind` fields.
- `graph_pooling_kind_from_name` resolves cold `add`/`sum`, `mean`, `max` and
  `none` configuration strings once. `GraphPooling::forward` then dispatches
  by enum to the completed PyG-lite global pools.
- `GCNConvNet::forward(Data)` accepts a cold generic graph boundary.
  `forward(Batch)` uses `num_graphs` and `max_num_nodes` already resolved by
  collation, avoiding device-side size inference in the hot dense path.
- Layers are stored by numeric ID and registered under Python-compatible
  `conv_N`, `norm_N` and `dout_N` names. Forward uses no string lookup.

The wrapper reuses the completed PyG-lite `GCNConv`, `to_dense_batch` and
global pooling kernels. A one-layer network maps input directly to output;
deeper networks use the fixed embedding width, BatchNorm/Dropout, and
LeakyReLU only between layers. All GCN linear weights receive Python's
post-construction orthogonal initialization in layer order. ATen operations
remain CPU/CUDA-neutral and callers own the LibTorch thread pool.

## Gate and performance

MSVC Release:

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_gcn_conv_net_unit vne_gcn_conv_net_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_gcn_conv_net_unit$' --output-on-failure
```

Same-container Python/PyG and GCC 11 differential:

```powershell
docker run --rm `
  --mount "type=bind,source=D:\Virne to cpp\virne.cpp,target=/work" `
  --mount "type=bind,source=D:\Virne to cpp\virne,target=/virne,readonly" `
  -w /work virne-cpu:latest `
  python porting/compare_gcn_conv_net.py `
    --native /work/build-ml-linux/porting/vne_gcn_conv_net_harness `
    --benchmark --warmups 1 --repetitions 3 `
    --json-output /work/porting/results/gcn_conv_net_differential_2026-08-06.json
```

Both focused units passed. The AST-isolated Python/PyG 2.7 gate passed `5/5`
cases for sparse, dense+BatchNorm, sum, mean and max outputs; all state-dict and
output bits match. The one prepared 32-graph / 512-node measurement produced
native median `912,943 ns/call` versus Python `1,001,247 ns/call`, a `1.097x`
native speedup; output checksum `19487954877254352` matched.

Full evidence:
`porting/results/gcn_conv_net_differential_2026-08-06.json`.
