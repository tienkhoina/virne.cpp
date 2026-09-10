# Bounded homogeneous message passing

## Status

The PyG-lite propagation boundary now covers the two custom homogeneous
operators used by Virne: `GraphViNEConv` and `EdgeFusionGATConv`. It is a
LibTorch/ATen component and does not add PyG or `torch_scatter` as a native
dependency.

The 2026-09-08 continuation corrected EdgeFusion's seeded initialization and
shared-projection reset order. Its broader state/RNG/backward gate is recorded
in `components/neural_network_remaining.md` and
`results/learning_continuation_2026-09-08.md`. The historical output benchmark
below remains frozen and was not rerun.

## API

Header: `virne/solver/learning/pyg_lite/message_passing.h`

- `PropagationOptions` stores the typed aggregation enum and optional target
  node count. Callers should provide the count for static graphs.
- `PropagationBatch` exposes direct `source_index`, `target_index`, `x_j`,
  `x_i`, `edge_attr`, and `target_num_nodes` fields.
- `prepare_homogeneous_propagation` performs boundary validation and device
  native gathers using source row `edge_index[0]` and destination row
  `edge_index[1]`.
- `propagate_homogeneous` is a header-only typed callback boundary. The
  callback creates an `[E, ...]` message tensor; `scatter_dim0` performs the
  requested reduction. No string dispatch or per-edge C++ loop is used.
- `GraphViNEConv` implements mean aggregation, optional scalar edge weights,
  `lin_rel(mean) + lin_root(x)`, and optional row-wise L2 normalization.
- `EdgeFusionGATConv` implements the Virne layer's actual add-aggregation
  path: source/target projections, optional edge projection, the one
  `Linear -> ReLU -> Linear` message network, head reshape, and bias. The
  Python attention coefficient code is disabled, so no attention state is
  fabricated in C++.

Fixed Python fields are typed direct members. Python string options are
resolved at the boundary to `ScatterReduce` and `SelfLoopFillKind` enums.
`EdgeFusionGATConv` intentionally accepts the homogeneous Virne path with
`heads == 1`; bipartite/lazy channels and heterogeneous data are separate
components.

## Device/performance contract

All feature and edge tensors remain on their input device. Index gathers,
message construction, and reduction are ATen operations and preserve
autograd. Passing `target_num_nodes` avoids any device-to-host inference in a
forward call. Range validation belongs at graph construction via
`Data::validate_indices`; it is not repeated in this hot propagation path.

Self-loop preparation uses the existing typed PyG-lite loop utilities. For a
static topology, prepare loops once and construct the layer with
`add_self_loops = false` when the caller owns that lifecycle.

## Differential and timing gate

`porting/tests/message_passing_unit.cpp` checks weighted mean propagation,
GraphViNE output against its closed-form identity-weight case, EdgeFusion
shape/autograd, and edge-feature self-loop handling. Build and run only the
focused target:

```powershell
cmake --build build-ml-win-msvc --config Release `
  --target vne_message_passing_unit --parallel 1
.\build-ml-win-msvc\porting\Release\vne_message_passing_unit.exe
```

The focused test is not a system benchmark. Timing and full Python
differential are measured against the isolated Python layer classes with the
same deterministic fixture. The 2026-08-04 CPU gate used Torch 2.6.0+cu124
and PyG 2.7.0 in the oracle container, GCC 14 `-O3` for C++, one Torch worker,
and CPU affinity pinned to one core for both processes. Each median used two
warmups, five repetitions, and three forward iterations per repetition.

| Layer | Python median | C++ median | C++ speedup | Output |
|---|---:|---:|---:|---|
| GraphViNEConv | 0.403187 ms | 0.436154 ms | 0.924x | PASS |
| EdgeFusionGATConv | 3.851177 ms | 2.329804 ms | 1.653x | PASS |

The differential gate covers three output tensors (GraphViNE output,
normalized output, and EdgeFusion output): **3/3 passed**, maximum absolute
error `0`. GraphViNE remains below 1x because the Python oracle's
`torch_scatter` reduction is highly optimized for this fixture; its C++ path
uses a direct ATen gather/reduction and does not allocate the unused `x_i`
endpoint. EdgeFusion is faster while preserving the active Virne add path.

The machine-readable result is
`porting/results/message_passing_benchmark_2026-08-04.json`; the short report
is `porting/results/message_passing_2026-08-04.md`.

Reproduce the focused native gate with:

```powershell
cmake --build build-ml-win-msvc --config Release `
  --target vne_message_passing_unit vne_message_passing_harness --parallel 1
.\build-ml-win-msvc\porting\Release\vne_message_passing_unit.exe
```

The Python differential command is kept in the result report because the
oracle harness must be linked against the container's Python Torch shared
libraries. It is not a native PyG dependency.
