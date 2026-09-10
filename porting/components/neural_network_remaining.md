# Remaining neural network primitives

## Surface

Headers are under `virne/solver/learning/neural_network/` and the isolated
aggregate target is `vne_learning_neural_network_remaining`.

| Python API | C++ API / contract |
| --- | --- |
| `multipler.MultiplerNet` | Typed state dimension; registered `l1/l2/l3`; LeakyReLU/Softplus forward and autograd |
| `loss.BarlowTwinsContrastiveLoss` | Optional lambda, normalization, epsilon; corrected sample standard deviation and matching backward graph creation order |
| `res.ResNetBlock` | Two registered 1x1 convolutions and residual; batched/unbatched input; float32/float64 |
| `res.ResLinearNet` | Explicit throwing constructor for the broken Python `super` call |
| `sinkhorn.SinkhornNetwork` | `forward`, `forward_log`, `forward_log_dummy`; matrix/batch input, nrows/ncols, supplied marginals, dummy rows, log-space sequential/batched execution |
| `sinkhorn.Sinkhorn` / `logsumexp` | `row_norm`, `col_norm`, forward, dimension/keepdim; Python's operation sequence and epsilon placement |
| `gnn.DenseToSparse`, `to_sparse_batch` | Node-major layout, ordered edge indices, float32 edge weights, direct batch tensor; valid node-feature backward |
| `gnn.GraphAttentionPooling` | `forward`, `get_coefs`, orthogonal weight and state name; explicit graph count avoids device synchronization |
| `gnn.GraphPooling` | Typed stateless or attention module; nested `pooling.weight` state path |
| `gnn.GraphConvNet`, `GATConvNet`, `EdgeFusionGATConvNet` | Typed layer stack and pooling; optional/scalar GAT edge attributes, dropout, BatchNorm, dense output; registered state order |
| `gnn.DeepEdgeFeatureGAT` | Start/middle/end GAT, residual matrices, alpha/theta, pooling/dense output; optional edge dimensions |
| `gnn.get_gnn_class` | Boundary resolver `gnn_kind_from_name`; fixed enum dispatch |

The earlier MLP, attention, neural-tensor, plain GCN and stateless pooling
gates remain frozen. The new GCN case tests attention composition only.
The generic Python `get_conv`/private initialization hooks become typed native
construction; there is no dynamic subclass/reflection protocol in hot loops.

## Corrected state and gradient behavior

`EdgeFusionGATConv` now creates/registers `pre_nns` before projections, resets
both aliased `lin_src` and `lin_dst`, and uses PyG's default Kaiming/uniform
initialization for the pre-network. Both state-dict names remain registered;
fixture filling deduplicates the shared TensorImpl just like Python parameters.
Native optimizer construction must likewise deduplicate aliases returned by
LibTorch module traversal. The new gradient gate does not claim an optimizer
step/checkpoint update gate.
The historical message-passing benchmark is not rerun or overwritten.

Barlow Twins creates the centered tensors before their standard-deviation
denominators. This preserves Python's autograd node order and float32 gradient
accumulation, not just forward values.

Sinkhorn retains the Python distinction between batched and sequential dummy
row cleanup when the padded matrix was already square. Zero active rows are
accepted. `epsilon`, `log_forward` and call `dtype` retain the reference's
inert behavior instead of silently selecting another algorithm.

## Limits

- The source's NNConvNet/PNAConvNet and custom NNConv constructors are broken;
  they are not usable convolution implementations or registry entries.
- DeepEdgeGAT supports the viable one-head shape path. Python leaves residual
  weights uninitialized. The original gate used shared weights; the native
  default now initializes these matrices orthogonally and has a separate
  seeded forward/backward gate in `native_repairs.md`. Two-layer models have
  no such matrices; dual-GNN retains its own parent initialization sequence.
- The GAT wrappers inherit Python's `heads`/`num_heads` mismatch and use one
  head. Do not silently change this architecture when loading checkpoints.
- Python DenseToSparse cannot backpropagate through differentiable edge
  weights because it mutates saved indices. The native implementation avoids
  that mutation; only node-feature gradients are part of the shared gate.
- Numeric dimensions are positive, layers are nonempty, and dropout is in
  `[0,1]` at the native boundary. Generic PyG lazy/bipartite custom-message
  protocols are outside the documented homogeneous contract.
- Operations preserve tensor devices. Physical CUDA execution is unverified.

## Verification

The continuation extends the previously unfinished aggregate fixture with
training/backward, BatchNorm buffers, dropout/RNG, repeated shared-parameter
reset, unbatched/double convolution, weighted/rectangular/dummy Sinkhorn,
optional/scalar GAT edges and zero-theta/no-edge DeepEdgeGAT cases. Tensor
shape, dtype and raw-byte FNV checksums are compared without tolerance against
six SHA-pinned Python files. Existing accepted tests are not rerun.

New artifacts: `porting/results/neural_network_remaining_*_2026-09-08.json`.
Accepted result: **27/27** exact cases at fixed thread widths 1 and 4, strict
GCC checks passed, and **2.685x / 2.384x** aggregate forward speedup.
See `porting/results/learning_continuation_2026-09-08.md` for the frozen table
and the explicit lack of bit invariance when changing thread width.
The benchmark is one representative forward pipeline of ten valid leaves;
it does not establish per-API or backward speedups. Correctness is required
before warmups/timing. Source, commands and accepted timing are frozen once
the paired benchmark passes.
