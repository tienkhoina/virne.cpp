# Action-mask helpers

## Status

The two small helpers from `virne/solver/learning/utils.py` are ported as a
typed LibTorch boundary.  They mask logits with `-1e8` and probabilities with
zero, preserving the input dtype and device.  Python list masks are accepted
once at the boundary through a `std::vector<uint8_t>` overload; all repeated
masking work uses tensors and fixed overloads.

## API

Header: `virne/solver/learning/rl_core/action_mask.h`

- `apply_mask_to_logit(values)` and `apply_mask_to_prob(values)` return the
  same tensor handle when no mask is supplied.
- Tensor-mask overloads normalize with `bool`, move to `values.device()`, and
  reshape to `values.sizes()` before `where`.  A mismatched element count
  raises `ActionMaskException(shape_mismatch)` before the LibTorch kernel.
- `std::vector<uint8_t>` overloads mirror Python's `BoolTensor(list)` path and
  are intended only for cold input/config conversion.
- The implementation has no dynamic string lookup or hidden worker.  Caller
  code can batch independent tensors or use LibTorch's configured thread pool.

## Device and correctness

The operations are device-neutral for CPU and CUDA LibTorch.  A mask with a
different device is moved to the values device; a different element count is
rejected by the same reshape contract as Python. Float16 overflow of the
`-1e8` sentinel is represented as negative infinity exactly like Python's cast.
Float16 and BFloat16 values are covered by the focused native unit; Float32 and
Float64 values, boolean/integer/float masks, mixed/all-true/all-false masks,
vector masks, and identity calls remain covered by the frozen differential gate.

## Gate and benchmark

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_action_mask_unit vne_action_mask_harness --parallel 1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_action_mask_unit$' --output-on-failure
python porting/compare_action_mask.py `
  --native .\build-rollout-win-msvc\porting\Release\vne_action_mask_harness.exe `
  --device cpu --threads 1 --warmups 2 --repetitions 5 --iterations 25 `
  --json-output porting/results/action_mask_differential_YYYY-MM-DD.json
```

The Windows MSVC Release CTest passed.  The comparator reports `9/9` exact
Python cases.  On the one compact `256x1024` logits case (one warmup, three
samples, ten iterations), native median time was `212,630 ns/call` versus
Python `221,350 ns/call` (`1.041x` native speedup); the output checksum is
`9213111756712424923`.  The full report is stored in
`porting/results/action_mask_differential_2026-08-05.json` and the focused
benchmark summary in `porting/results/action_mask_benchmark_2026-08-05.json`.
