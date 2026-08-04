# LibTorch runtime boundary

## Status

The 2.6.0 shared CPU LibTorch archive is vendored at `libs/libtorch` and is
resolved only by the opt-in CMake option `VIRNE_ENABLE_LIBTORCH=ON`. The
default build and the frozen non-RL environment do not link Torch. This is a
dependency/runtime gate, not an implementation of `virne/solver/learning`.

## CMake/API

`virne_libtorch` is an interface target backed by the archive's own
`TorchConfig.cmake`. `VIRNE_LIBTORCH_ROOT` overrides the default
`libs/libtorch` path and may point at the matching official CUDA archive on a
CUDA runner. The only current executable is `vne_libtorch_probe`:

```text
vne_libtorch_probe [--threads N] [--device cpu|cuda|auto]
```

The probe sets the LibTorch intra-op and inter-op pools once before the first
operation, seeds the global CPU/CUDA generator, computes a fixed `float64`
`3x4 @ 4x3` result, and emits one JSON object. `--device cpu` is the parity
mode; `--device cuda` requires `torch::cuda::is_available()`, while `auto`
selects CUDA only when the runtime reports it. A caller must configure the
thread pools at process/worker startup, not inside a hot request loop.

Output fields are stable and machine-readable:

| Field | Meaning |
|---|---|
| `libtorch_version` | archive `build-version` |
| `device`, `cuda_available` | selected device and runtime capability |
| `threads`, `interop_threads` | effective LibTorch pool widths |
| `shape`, `dtype` | fixed probe tensor metadata |
| `checksum`, `output_bytes`, `sum_bits` | FNV-1a/bit-exact output digest |

The C++ probe uses typed tensor operations and does not expose dynamic string
maps in a hot loop. Future ML leaves should keep Torch at this explicit ABI
boundary; do not mix the archive's ABI-0 target into the ABI-1 non-RL libraries
without a separate reviewed interface.

## Python output gate

`porting/compare_libtorch_probe.py` runs the same CPU expression with the
Python oracle's Torch 2.6.0 wheel and compares `shape`, `dtype`, `checksum`,
`output_bytes`, and `sum_bits` exactly. Version/device capability fields are
reported but are not treated as equal because the oracle wheel is the CUDA
build (`2.6.0+cu124`) while the vendored development payload is CPU.

## Environment gate

The non-RL `BaseEnvironment`/`SolutionStepEnvironment` API is already frozen
without Torch. Run `porting/compare_environment.py` after building to compare
the complete lifecycle output (including physical-resource restoration) with
the pinned Python source. This keeps environment parity independent of the
future ML/RL tensor boundary.
