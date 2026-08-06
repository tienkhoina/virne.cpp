# Typed multi-head self-attention compatibility

## Status

Complete for Python `neural_network.att.MultiHeadSelfAttention`. This is the
compatibility primitive used by future attention policies; it does not add RL
training or optimizer code.

## API

Header: `virne/solver/learning/neural_network/attention.h`

- `MultiHeadSelfAttentionOptions` stores input, embedding, value/key widths and
  head count as direct integer fields.
- `MultiHeadSelfAttention` registers `W_query` and `W_key` with the original
  state names and uses the same Torch uniform initialization sequence.
- `forward(query, key, value, attention_mask)` preserves Python's current
  behavior: `key=None` falls back to query, while value and attention mask are
  accepted for API compatibility but are not used by this source class.

The output is the mean-over-head compatibility tensor `[B,Q,K]`, after the
same key-dimension scaling and `tanh(...)*10` transform. No string/map lookup
or worker is present in the forward loop. The module uses standard ATen
dispatch on CPU or CUDA; callers own thread-pool configuration.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_multi_head_attention_unit vne_multi_head_attention_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_multi_head_attention_unit$' --output-on-failure
& .\.deps-cache\python-torch26\Scripts\python.exe `
  porting\compare_multi_head_attention.py `
  --native build-rollout-win-msvc\porting\Release\vne_multi_head_attention_harness.exe `
  --benchmark --warmups 1 --repetitions 3 `
  --json-output porting\results\multi_head_attention_differential_2026-08-06.json
```

MSVC Release passed the focused unit and `3/3` exact AST-isolated Python cases
for self, cross and explicit-dimension attention. Both parameter tensors and
output float32 bits match, including the Torch RNG initialization sequence.
The single `[32,64,64]` / four-head measurement produced native median
`1,872,000 ns/call` versus Python `3,859,600 ns/call`, a `2.062x` speedup;
the output checksum was identical.

Full evidence:
`porting/results/multi_head_attention_differential_2026-08-06.json`.
