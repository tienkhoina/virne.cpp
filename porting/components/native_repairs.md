# Confirmed native neural repairs

Updated 2026-09-10. This focused gate covers three repairs already present in
the native neural modules. Original Python source files remain unchanged.
The oracle applies only the explicit corrections below in memory, so its
result is exact parity with corrected expectations rather than a claim that
the broken source paths run successfully without correction.

| Case | Confirmed source defect | Native behavior and oracle correction |
| --- | --- | --- |
| `single_layer` | `MLPNet(num_layers=1)` indexes the empty hidden-dimension list in the first-layer branch. | One input-to-output Linear followed by Identity; original orthogonal initialization and state names are retained. The case requests BatchNorm and dropout to check the final-layer contract. |
| `residual_initialization` | `DeepEdgeFeatureGAT` allocates residual matrices with uninitialized `torch.Tensor`; the initialization call is commented out. | Initialize those matrices orthogonally after constructing the original GAT layers. The case exercises a three-layer model with its natural seeded initialization, edge features and backward. |
| `recurrent_clamp` | The configurable decoder uses a minimum clamp bound of 100 even when its embedding has only `p_net_num_nodes + 1` rows. | Clamp to the physical-node count. With four physical nodes, action IDs 77 and 101 map to the reserved embedding row 4. Encoder outputs, logits, hidden state, value and backward are compared. |

The DeepEdge repair is the default for direct construction. The native dual
GNN policy suppresses that intermediate initialization because its parent
already initializes every parameter, preserving the original parent RNG
sequence. This focused gate does not repeat the accepted dual-policy gate.

## Accepted exact gate

[`results/native_repairs_differential_2026-09-10.json`](../results/native_repairs_differential_2026-09-10.json)
reports **3 cases, 135 records, PASS** on CPU Torch 2.6.0, one Torch thread.
The comparator checks output, input and parameter gradients, registered
parameters/buffers, dtype/shape, raw tensor bytes and the subsequent RNG draw.
The Python and native output SHA256 are both
`f1db80ef0e48215e355163eac64306bc3dfbaeae6beb0291bb7496381f6cd486`.

The source SHA256 values in the report were rechecked against the original
Python files after the accepted gate. They still match. No further source
change was needed after this gate passed.

Build target: `vne_native_repairs_harness`. Comparator:
`porting/compare_native_repairs.py --native <binary> --output <report>`.
These are regression cases for previously measured modules, so accepted
aggregate benchmarks were not rerun. Existing measurements remain in
[`neural_network_remaining.md`](neural_network_remaining.md) and the
[`RL continuation report`](../results/learning_continuation_2026-09-08.md).
They do not establish a separate speedup for these repaired edge cases.

The three-case gate does not cover the separate feature-axis BatchNorm repair
on rank-three MLP input; do not infer that coverage from `single_layer`, whose
final layer correctly has no BatchNorm.
