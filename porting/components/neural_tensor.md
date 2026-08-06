# Typed neural tensor network

## Status

Complete for Python `neural_network.neural_tensor.NeuralTensorNetwork`. This
is an inference/training primitive only; no optimizer or RL loop is included.

## API

Header: `virne/solver/learning/neural_network/neural_tensor.h`

- `NeuralTensorNetworkOptions` stores the fixed `input_features` and
  `tensor_neurons` dimensions.
- `NeuralTensorNetwork` registers `weight_matrix`, `weight_matrix_block` and
  `bias` under the Python state-dict names and applies orthogonal initialization
  in the original parameter order.
- `forward(embedding_1, embedding_2)` accepts `[B,input_features]` tensors and
  returns the non-negative similarity scores `[B,tensor_neurons]`.

The implementation resolves dimensions once and uses direct tensor handles in
the hot path. Matmul, transpose, reshape, concatenation and ReLU are ATen
operations and therefore dispatch unchanged on CPU or CUDA; callers own the
LibTorch thread pool.

## Gate and performance

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_neural_tensor_unit vne_neural_tensor_harness -- /m:1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_neural_tensor_unit$' --output-on-failure
& .\.deps-cache\python-torch26\Scripts\python.exe `
  porting\compare_neural_tensor.py `
  --native build-rollout-win-msvc\porting\Release\vne_neural_tensor_harness.exe `
  --benchmark --warmups 1 --repetitions 3 `
  --json-output porting\results\neural_tensor_differential_2026-08-06.json
```

MSVC Release passed the focused unit and `3/3` exact AST-isolated Python
cases. All three parameter tensors and output float32 bits match, including
the rank-3 orthogonal initialization. The one `[32,64]` / 32-neuron
measurement produced native median `152,200 ns/call` versus Python
`227,000 ns/call`, a `1.491x` native speedup; output checksum
`8497801754783203451` matched.

Full evidence:
`porting/results/neural_tensor_differential_2026-08-06.json`.
