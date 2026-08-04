# PyG-lite homogeneous core

## Status

Complete for the homogeneous PyG operations currently exercised by Virne.
The implementation depends only on LibTorch/ATen; PyG 2.7 remains the Python
oracle and is not a C++ runtime dependency.

## API

Header: `virne/solver/learning/pyg_lite/pyg_lite.h`

- `Data`: direct fixed fields `x`, `edge_index`, `edge_attr`, `batch`, `ptr`;
  `validate`, cold `validate_indices`, deep `clone`, and `to`.
- `Batch`: adds direct `edge_ptr`, `num_graphs`, and `max_num_nodes` metadata;
  `from_data_list`, O(K) `index_select`, deep `clone`, and `to`.
- `scatter_dim0`: typed sum/mean/minimum/maximum/product/any reductions.
- Edge utilities: `sort_edge_index`, self-loop add/remove/remaining, and
  `segment_softmax`.
- Batching/pooling: `to_dense_batch`, `global_add_pool`, `global_mean_pool`, and
  `global_max_pool`. The `Batch` dense overload consumes pre-resolved dimensions
  and does not synchronize the device to infer them.
- GNN: `gcn_norm`, `GCNConv`, and homogeneous fixed-channel `GATConv`, including
  edge features and attention output.

Fixed Python attributes are typed direct members. Dynamic names are not stored
in maps. Hot callers must pass known node/pool/batch sizes or use `Batch`
metadata instead of asking a device tensor to infer a size.

## Device and performance contract

Math tensors inherit device and dtype from their inputs. No hot mathematical
path stages data through CPU. Linux CPU edge sorting uses a radix fast path;
duplicate keys fall back to the ordering used by the Python oracle. Windows,
CUDA, and other ATen backends use the native device sort.

`Batch::index_select` and inferred-size overloads are cold boundary APIs and may
synchronize. Call `validate_indices` when constructing a graph, not in every
forward. `GCNConv(cached=true)` is for one static/transductive topology; cache
identity and tensor version are checked and module `to(...)` clears it. For a
stack of GAT layers, prepare self-loops once and construct layers with
`add_self_loops=false` when the topology is immutable.

The same source builds against CPU or CUDA LibTorch. The unit test automatically
repeats its device smoke on CUDA when `torch::cuda::is_available()` is true.
The current runner has no CUDA device, so CUDA compilation/dispatch compatibility
is implemented but not claimed as a physical-device gate.

## Gates

Linux:

```text
cmake --build build-ml-linux --target vne_pyg_lite_unit vne_pyg_lite_harness
./build-ml-linux/porting/vne_pyg_lite_unit
```

Windows/MSVC:

```powershell
cmake --build build-ml-win-msvc --config Release `
  --target vne_pyg_lite_unit vne_pyg_lite_harness --parallel 1
.\build-ml-win-msvc\porting\Release\vne_pyg_lite_unit.exe
```

The differential gate records 59 tensors covering data/batch operations,
reductions, edge utilities, pooling, GCN/GAT outputs, attention, and gradients.
See `porting/results/pyg_lite_2026-08-04.md` for the frozen timing.

## Deferred surface

Generic `MessagePassing`, heterogeneous data/convolution, and lazy bipartite
GAT channels are separate components. Unused PyG APIs are intentionally not
ported. Invalid Virne `NNConv`/`PNAConv` call sites are also outside this gate.
