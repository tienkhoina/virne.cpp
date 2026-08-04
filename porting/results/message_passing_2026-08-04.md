# Bounded message-passing gate - 2026-08-04

Implementation is complete for the homogeneous custom layers used by Virne:

- `GraphViNEConv`: weighted mean propagation, two typed linear transforms,
  optional L2 normalization.
- `EdgeFusionGATConv`: add propagation, optional edge features, one
  `Linear -> ReLU -> Linear` message network, and typed self-loop fill.

The focused unit covers generic propagation, closed-form GraphViNE output,
EdgeFusion output shape/autograd, and edge-feature self-loop construction. No
full Virne model wiring is required for this bounded gate: the two Python
layer classes were loaded directly and compared with the native fixture.

## Differential and timing

The oracle used Torch 2.6.0+cu124 and PyG 2.7.0. Python and C++ ran in the
same CPU container with Torch fixed to one worker and both processes pinned to
one CPU core. C++ used GCC 14 `-O3`; each median used two warmups, five
repetitions, and three forward iterations.

| Layer | Python median | C++ median | C++ speedup | Output |
|---|---:|---:|---:|---|
| GraphViNEConv | 0.403187 ms | 0.436154 ms | 0.924x | PASS |
| EdgeFusionGATConv | 3.851177 ms | 2.329804 ms | 1.653x | PASS |

Parity passed **3/3** records with maximum absolute error `0`. The records
are GraphViNE output, normalized GraphViNE output, and EdgeFusion output.
The C++ signatures and Python signatures match for both timed layers.

GraphViNE's direct ATen path improved from the prior 0.692x result to 0.924x
by avoiding the generic propagation gather of unused target features. The
remaining difference is retained as an honest fixture result; no interface
or output trade-off was introduced. EdgeFusion's homogeneous path also
reuses the single Python projection and measured 1.653x faster than Python.

Machine-readable output: `porting/results/message_passing_benchmark_2026-08-04.json`.
