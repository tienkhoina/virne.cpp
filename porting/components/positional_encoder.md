# Typed positional encoder

## Status

Complete for Python `neural_network.att.PositionalEncoder` in add/concat
evaluation paths. It is a neural-network primitive only; no RL training or
optimizer code is included.

## API

Header: `virne/solver/learning/neural_network/attention.h`

- `PositionalEncoderOptions` stores embedding width, maximum sequence length,
  dropout probability and the fixed `PositionalEncodingMethod` enum.
- `PositionalEncoder` registers the `[1,max_len,embedding_dim]` sinusoidal
  buffer under the same `pe` name as Python.
- `forward([B,L,D])` uses a repeated prefix of the buffer, then performs direct
  add or concatenation and applies the registered dropout module in train mode.

The method string is resolved once at the boundary; no string branch or map
lookup remains in forward. The implementation is CPU/CUDA-neutral and callers
own LibTorch thread-pool configuration. Valid Python attention inputs use an
even positive embedding dimension and `L <= max_len`; invalid shapes fail
before tensor composition.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_positional_encoder_unit vne_positional_encoder_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_positional_encoder_unit$' --output-on-failure
& .\.deps-cache\python-torch26\Scripts\python.exe `
  porting\compare_positional_encoder.py `
  --native build-rollout-win-msvc\porting\Release\vne_positional_encoder_harness.exe `
  --benchmark --warmups 1 --repetitions 3 `
  --json-output porting\results\positional_encoder_differential_2026-08-06.json
```

MSVC Release passed the focused unit and `3/3` exact AST-isolated Python cases
for add, concat and a longer sequence. Buffer and output float32 bits match.
The single `[32,500,128]` measurement produced native median
`2,362,900 ns/call` versus Python `5,354,100 ns/call`, a `2.266x` speedup;
the output checksum was identical.

Full evidence:
`porting/results/positional_encoder_differential_2026-08-06.json`.
