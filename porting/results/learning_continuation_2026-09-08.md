# Learning continuation: 2026-09-08

## Accepted new evidence

- Remaining neural primitives: **27/27 exact Python/C++ cases** at each fixed
  Torch thread count, 1 and 4. Shape, dtype, state, RNG tails, forward output,
  input gradients and parameter gradients use raw-byte checksums without
  tolerance. This completes the previously unfinished EdgeFusion state gate
  and adds the new lifecycle/optional-input coverage.
- CNN and GNN/MLP policies: **12/12 cases**, **577 records**, **89,072 serialized
  bytes** compared literally at each thread count. Natural initialization,
  actor/critic outputs, double precision, training gradients, BatchNorm and
  error behavior are covered. Non-finite reference tensors are rejected.
- GCC 11.4 Release builds passed. New policy targets/harness compiled with
  `-Wall -Wextra -Wpedantic -Werror`; syntax-only checks with the same strict
  flags passed for gnn, loss, multipler, res, sinkhorn and message_passing.
- Existing accepted unit/differential/benchmark suites were not rerun. The
  previous uncompleted neural comparator was extended; only this unfinished
  group and new policy fixtures were executed. Old result files are preserved.

## Paired benchmarks

No compiler was running during the measurements. Both sides ran sequentially
inside the same CPU oracle container, with a common affinity mask and fixed
intra-op threads; inter-op threads were 1. Three warmups and five repetitions
were used. The policy workload batches ten iterations per repetition; neural
timing measures one ten-leaf pipeline per repetition. Tensor serialization and
checksum comparison are outside the timed region.

| New representative forward pipeline | Threads / affinity | C++ median (ms) | Python median (ms) | Python/C++ |
| --- | --- | ---: | ---: | ---: |
| Ten remaining neural leaves | 1 / CPU 0 | 1.506410 | 4.044451 | **2.685x** |
| Ten remaining neural leaves | 4 / CPUs 0-3 | 2.691930 | 6.417104 | **2.384x** |
| CNN + GAT/MLP actor/critic | 1 / CPU 0 | 0.548194 | 2.094355 | **3.820x** |
| CNN + GAT/MLP actor/critic | 4 / CPUs 0-3 | 0.645025 | 2.437873 | **3.780x** |

These rows are now frozen. They establish aggregate forward performance for
these fixtures, not per-method or backward/training throughput speedups.
Four threads are slower than one for both native workloads. Use a fixed
single-thread Torch configuration as the reproducible reference; no automatic
worker policy was introduced.

Exact Python/C++ equality holds **within each fixed thread width**. It does
not imply bit invariance when changing thread width: ResNet forward and some
BatchNorm/GAT training tensors differ between 1 and 4 threads on both sides.
Policy parity SHA-256 also changes across widths, while the policy benchmark
output happens to remain identical. This is explicitly not an accepted
cross-thread bit-invariance gate or a claim that parallelism helps this fixture.

| Checksum | Threads 1 | Threads 4 |
| --- | --- | --- |
| Neural timed output FNV-1a | `4063719262803839771` | `4448702900771770593` |
| Policy raw parity SHA-256 | `c95e4817036c10355fa400404e1dd283e1199f6d218f2df6552048cc1e2f8ebf` | `5e51f48d3d8c59dbc26abb033c3cf911e83b4234247ed475b5227482b595c90a` |
| Policy timed output SHA-256 | `8c00944eda95fbc11e7caddfa5d8128be3c1bd148e992941398a29fef8e8e525` | same |

## Artifacts

- `neural_network_remaining_differential_2026-09-08.json`
- `neural_network_remaining_benchmark_t1_2026-09-08.json`
- `neural_network_remaining_benchmark_t4_2026-09-08.json`
- `cnn_gnn_mlp_policy_differential_2026-09-08.json`
- `cnn_gnn_mlp_policy_benchmark_t1_2026-09-08.json`
- `cnn_gnn_mlp_policy_benchmark_t4_2026-09-08.json`
- `learning_api_inventory_2026-09-08.json` (AST inventory only)

Python source hashes are embedded in the JSON reports. Native/harness SHA-256:

| File (relative to repository root) | SHA-256 |
| --- | --- |
| `build-ml-linux/porting/vne_neural_network_remaining_harness` | `8e6ec8d61acbbf803f14c80f6d4d2a97d51afa7200ea7265d3e75baf8adb0c02` |
| `build-ml-linux/porting/vne_cnn_gnn_mlp_policy_harness` | `ff1e63cc2c6c2f515c0f8f82a8b4c6a4d1604b6c40f1eaaec3820e95507fc4b6` |
| `porting/compare_neural_network_remaining.py` | `13866eb782cad70bd87f5cc3c9e986c21567f2bce681fc7dde076b446ecd9e79` |
| `porting/neural_network_remaining_harness.cpp` | `c0ef6bdb8cd45fd489b7162b28bfe63f7b589a06e291ff22042158c7ad8fe687` |
| `porting/compare_cnn_gnn_mlp_policy.py` | `29d7f42fe1a26e8a8f0d8d8ac14663315f6cd144ef9a763643d107aad4f1084c` |
| `porting/cnn_gnn_mlp_policy_harness.cpp` | `d8119f179cf2b76544df6eb4bbf9aa2a3bd7249d733c588babeea7f057649941` |
| `porting/tensor_bytes.h` | `5286bdf95a5a2a485ac31992b63be0ffc1e3607d35b4ac64e9c6b0d48afaa1d8` |

## Recorded commands (provenance, not a rerun request)

```powershell
docker exec virne-cpp-dev cmake -S /work -B /work/build-ml-linux `
  -DCMAKE_BUILD_TYPE=Release -DVIRNE_ENABLE_LIBTORCH=ON -DVIRNE_ENABLE_ML=ON `
  -DVIRNE_LIBTORCH_ROOT=/work/libs/libtorch
docker exec virne-cpp-dev cmake --build /work/build-ml-linux `
  --target vne_neural_network_remaining_harness vne_cnn_gnn_mlp_policy_harness -j 2

docker run --rm --network none --cpuset-cpus=0 `
  --mount 'type=bind,source=D:\Virne to cpp\virne.cpp,target=/work' `
  --mount 'type=bind,source=D:\Virne to cpp\virne,target=/virne,readonly' `
  --workdir /work virne-cpu:latest python porting/compare_neural_network_remaining.py `
  --native build-ml-linux/porting/vne_neural_network_remaining_harness `
  --benchmark --warmups 3 --repetitions 5 --threads 1 `
  --output porting/results/neural_network_remaining_benchmark_t1_2026-09-08.json

docker run --rm --network none --cpuset-cpus=0 `
  --mount 'type=bind,source=D:\Virne to cpp\virne.cpp,target=/work' `
  --mount 'type=bind,source=D:\Virne to cpp\virne,target=/virne,readonly' `
  --workdir /work virne-cpu:latest python porting/compare_cnn_gnn_mlp_policy.py `
  --native build-ml-linux/porting/vne_cnn_gnn_mlp_policy_harness `
  --benchmark --threads 1 `
  --output porting/results/cnn_gnn_mlp_policy_benchmark_t1_2026-09-08.json
```

The paired 4-thread invocations used `--cpuset-cpus=0-3`, `--threads 4` and
the corresponding `t4` result filename, with every other option unchanged.

## Reconstructed environment and limits

- Toolchain: `virne-cpp-toolchain:gcc11.4`, image
  `41ed2c98f7d5c5b6f3c48587502e463d2bcf837355eee2ebb731b80422f48a93`.
  Its GCC/libstdc++ compile guard passed: GCC 11.4.0, `_GLIBCXX_RELEASE=11`,
  `__GLIBCXX__=20230528`. Vendored LibTorch 2.6.0 CPU uses ABI 0.
- Python non-ML: `virne-python-oracle:py310-nonml`, image
  `cf0257c8792a8e7c1b9242e8da9ada0df8a0687d7b37476fb35ffdd7a4be080d`.
- CPU ML: both `virne-cpu:latest` and
  `virne-python-oracle:py310-torch26-pyg27-cpu` reference image
  `67b30d3b4155a1854caa5c3d7ee93d08980bdd5fc59c117b7b7c07a5744b820a`.
  CPython 3.10.20, Torch 2.6.0+cpu, PyG 2.7.0, torch_scatter 2.1.2+pt26cpu,
  torch_sparse 0.6.18+pt26cpu, NumPy 2.2.6 and SciPy 1.15.3.
- Docker reported 8 CPUs. After unused build-cache cleanup it reported three
  images totaling 1.736 GB, no local volumes and zero build cache. Only
  `virne-cpp-dev` remains running; the oracle containers used `--rm`.
- This CPU image replaces the old CUDA-backed oracle for new results. The old
  performance artifacts are not relabeled as CPU-wheel measurements.
- CUDA hardware, new MSVC builds, optimizer steps/checkpoint updates, full RL
  environments/agents/training and the remaining policies are not verified by
  these new gates. See `../LEARNING_COMPONENT_MAP.md` for the concrete backlog.
