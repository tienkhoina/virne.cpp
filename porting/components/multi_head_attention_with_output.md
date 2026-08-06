# Typed multi-head attention with output

## Status

Complete for Python `neural_network.att.MultiHeadAttentionWithOutput`. This
is an inference/training primitive only; it does not add RL training code.

## API

Header: `virne/solver/learning/neural_network/attention.h`

- `MultiHeadAttentionWithOutputOptions` stores `embedding_dim`, `num_heads`,
  `d_k`, `d_v` and dropout probability as fixed fields. `d_k == 0` and
  `d_v == 0` resolve once to `embedding_dim`, matching Python `None`.
- `MultiHeadAttentionWithOutput` registers `W_q`, `W_k`, `W_v`, `dropout` and
  `fc` with the original state-dict names and LibTorch linear initialization.
- `forward(query, key, value, attention_mask)` returns
  `MultiHeadAttentionOutput { output, weights }`. Inputs are rank-3
  `[batch, sequence, embedding_dim]`; weights are `[B,H,Q,K]` and output is
  `[B,Q,embedding_dim]`.
- A zero-valued mask entry is filled with `-1e9` before softmax. ATen dispatch
  is CPU/CUDA-neutral; callers own thread-pool configuration. The hot path
  uses only typed module handles and fixed dimensions.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_multi_head_attention_with_output_unit `
  vne_multi_head_attention_with_output_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_multi_head_attention_with_output_unit$' --output-on-failure
& .\.deps-cache\python-torch26\Scripts\python.exe `
  porting\compare_multi_head_attention_with_output.py `
  --native build-rollout-win-msvc\porting\Release\vne_multi_head_attention_with_output_harness.exe `
  --benchmark --warmups 1 --repetitions 3 `
  --json-output porting\results\multi_head_attention_with_output_differential_2026-08-06.json
```

MSVC Release passed the focused unit and `3/3` exact AST-isolated Python
cases: default dimensions, explicit dimensions with a four-dimensional mask,
and a broadcast `[Q,K]` mask. Parameter, output and attention-weight float32
bits match. The one `[32,64,64]` / four-head measurement produced native
median `4,367,800 ns/call` versus Python `5,032,400 ns/call`, a `1.152x`
native speedup; output checksum `8777342082330918831` matched.

Full evidence:
`porting/results/multi_head_attention_with_output_differential_2026-08-06.json`.
