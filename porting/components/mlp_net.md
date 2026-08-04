# MLPNet

`MLPNet` is the first Torch-backed ML leaf. It ports the Python class in
`virne/solver/learning/neural_network/mlp.py` without importing the RL, PyG,
feature-construction, or training stacks.

## C++ API

`virne/solver/learning/neural_network/mlp.h` exposes:

```cpp
using MlpHiddenDims = std::variant<
    std::monostate, std::int64_t, std::vector<std::int64_t>>;

struct MlpOptions {
    std::int64_t input_dim = 0;
    std::int64_t output_dim = 0;
    std::int64_t num_layers = 2;
    MlpHiddenDims embedding_dims = std::monostate{};
    bool batch_norm = false;
    double dropout_prob = 1.0;
};

MLPNet model(MlpOptions{...});
torch::Tensor output = model->forward(input);
```

The input may have any leading dimensions and must end in `input_dim`; the
output preserves those leading dimensions and ends in `output_dim`.
`embedding_dims` exactly mirrors Python's three accepted forms:

- `std::monostate`: every hidden width is `2 * input_dim`;
- one `int64_t`: repeat that width for all `num_layers - 1` hidden layers;
- `vector<int64_t>`: one explicit width per hidden layer.

The valid ported domain has positive dimensions and `num_layers >= 2`.
Invalid options fail at construction. `options()` and
`resolved_hidden_dims()` expose the typed immutable configuration.

## Exact semantics

Each hidden layer is
`Linear -> BatchNorm1d/Identity -> LeakyReLU(negative_slope=0.01)`; the final
layer is `Linear -> Identity`. Linear biases retain the LibTorch/Python
default initialization. After every layer module has been constructed, only
Linear weights receive `orthogonal_(gain=1)`, preserving Python RNG
consumption and continuation exactly.

The public Sequential is registered as `lins`, preserving state names such as
`lins.0.weight`, `lins.1.running_mean`, and `lins.3.bias`. The Python argument
`dropout_prob` is deliberately stored but does not create a Dropout module;
changing it must not change state or output. This is compatibility behavior,
not an optimization omission.

## Typed boundary and performance

All fixed options are direct typed fields. The dynamic hidden-width form is
resolved once during construction to a compact vector; `forward` executes the
prebuilt Sequential and performs no string lookup, registry lookup, allocation
of modules, or configuration parsing in the hot path. Callers should pass a
typed Tensor directly rather than a string-keyed observation map.

The leaf creates no worker threads. Configure LibTorch intra-op and inter-op
pools once before the first tensor operation; never change them per request.
The canonical differential/benchmark uses one thread. Request-level
parallelism must keep independent Tensor/model state and deterministic input
order.

The module is device-agnostic: moving the model and input to the same CUDA
device enables CUDA execution when built against a matching CUDA LibTorch
payload. The frozen exact gate is CPU-only; CUDA parity must be recorded on a
CUDA runner before it is claimed.

`vne_learning_mlp` is built only with `VIRNE_ENABLE_ML=ON`, which itself
requires `VIRNE_ENABLE_LIBTORCH=ON`. It links the isolated
`virne_libtorch` target and inherits the archive's ABI and runtime handling.
The default non-ML build remains Torch-free; Torch types must not leak into
the frozen non-ML libraries.

## Build and verification

From the `virne.cpp` root in a Visual Studio Developer PowerShell:

```powershell
$torchRoot = (Resolve-Path .\libs\libtorch-win).Path
cmake -S . -B build-ml-win-msvc -G "Visual Studio 17 2022" -A x64 `
  -DVIRNE_ENABLE_LIBTORCH=ON `
  -DVIRNE_ENABLE_ML=ON `
  -DVIRNE_LIBTORCH_ROOT="$torchRoot"

cmake --build build-ml-win-msvc --config Release `
  --target vne_mlp_net_unit vne_mlp_net_harness

ctest --test-dir build-ml-win-msvc -C Release `
  -R '^vne_mlp_net_unit$' --output-on-failure
```

Run the exact Python/native gate and write its machine-readable report:

```powershell
& .\.deps-cache\python-torch26\Scripts\python.exe `
  .\porting\compare_mlp_net.py `
  --harness .\build-ml-win-msvc\porting\Release\vne_mlp_net_harness.exe `
  --python-source ..\virne\virne\solver\learning\neural_network\mlp.py `
  --threads 1 --warmups 3 --repetitions 5 --iterations 20 `
  --json-output .\.deps-cache\mlp_net_parity_windows.json
```

The comparator loads the pinned Python file directly, so importing it cannot
trigger Virne's package-level PyG installation side effects. It exits nonzero
on any state, RNG, tensor metadata, byte-checksum, or benchmark-output
mismatch.
