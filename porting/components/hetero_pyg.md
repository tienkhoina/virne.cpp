# PyG-lite heterogeneous data and convolution

## Status

Complete for the two-node-type, six-relation surface used by Virne's
heterogeneous GAT encoder. Python PyG 2.7 is the oracle; C++ uses only
LibTorch/ATen and fixed IDs.

## API

Headers:

- `virne/solver/learning/pyg_lite/hetero_data.h`
- `virne/solver/learning/pyg_lite/bipartite_gat.h`
- `virne/solver/learning/pyg_lite/hetero_conv.h`

`HeteroData` stores direct `x`, optional `batch`/`ptr`, and a fixed array of
`HeteroRelationData`. `HeteroNodeType` has `physical` and `virtual_node`;
`HeteroRelationId` has `p/v_connect`, `v/p_mapping`, and `v/p_imaginary`.
`hetero_relation_id(source, kind, target)` resolves a dynamic boundary choice
once; forward loops use the enum ID and array index.

`HeteroBatch::from_data_list` concatenates each node type, creates direct batch
and cumulative pointer tensors, and offsets relation endpoints by source and
target node counts. `validate`, `clone`, and `to` are cold boundary operations.

`BipartiteGATConv` implements tuple-channel GATConv with separate source and
target linear projections, edge features, configurable heads/concat,
self-loop fill, attention output, residual, bias, CPU, and CUDA-compatible
ATen dispatch. `HeteroConv` owns one typed module per enabled relation and
groups destination messages with `stack(...).sum(dim=0)`, matching PyG's
floating-point reduction order.

## Performance contract

Inputs should be validated while constructing a batch or encoder. The hot
`HeteroConv::forward` path intentionally does not repeat endpoint validation;
this matches PyG and avoids device synchronization. Relation IDs, tensor
references, and fixed channel options are resolved before the forward loop.
LibTorch intra-op workers are selected with `torch::set_num_threads` (or the
caller-wide worker configuration); the fixed six-relation reduction remains
deterministic.

## Gates and benchmark

Linux Docker gate:

```text
docker exec virne-cpp-dev cmake --build /work/build-ml-linux \
  --target vne_hetero_pyg_unit vne_hetero_pyg_harness -j 4
docker exec virne-cpp-dev /work/build-ml-linux/porting/vne_hetero_pyg_unit
docker exec virne-cpp-dev ctest --test-dir /work/build-ml-linux \
  -R '^vne_hetero_pyg_unit$' --output-on-failure
```

The exact fixture has 12 summaries covering batching, IDs, bipartite output,
attention, self-loops, and both heterogeneous destinations. The Python
comparator passed `12/12` bit-exact outputs with checksum
`4525087838765020817`.

Same fixture, LibTorch/PyTorch 2.6 CPU, prepared forward only:

| threads | native median | Python median | Python/native |
|---:|---:|---:|---:|
| 1 | 566,128 ns/call | 2,010,359 ns/call | 3.551x |
| 4 | 491,911 ns/call | 1,769,675 ns/call | 3.598x |

The full samples and exact reports are in
`porting/results/hetero_pyg_differential_2026-08-05.json` and
`porting/results/hetero_pyg_benchmark_2026-08-05.json`.
