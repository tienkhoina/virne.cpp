# Results: non-Torch `virne.utils.dataset` NumPy RNG — 2026-07-28

State: **RNG LEAF COMPLETE**. The non-Torch dataset core and XML/graph/GML leaf
were completed separately. Only the same-named Python/Torch `set_seed` facade
remains explicitly unimplemented.

## Locked identity

- Original Python leaf: `virne/utils/dataset.py` at commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Python source SHA-256:
  `269650EBCC373D7BDF79FA17346BD6F847973F17E60C1AC9BCAE7CFD97BF936F`;
  exact size 9,635 bytes.
- Final Release unit SHA-256:
  `B3E70A569A6BE4DA6D1EA7C5EF07C7C24FD412C5557B2595915F271F947CF301`.
- Final Release harness SHA-256:
  `104474C2E434DB8FDF723890C3A39ED46CCE23F4D33C8E583ED386F0319875D2`.
- Differential artifact: `dataset_rng_differential_2026-07-28.json`, SHA-256
  `9D236D63A3D2B1BC0BED3E1C9A7C0641877B8C5D97915D2E8DC5C2682C251C9E`.
- Python-optimized boundary artifact: `dataset_rng_optimized_2026-07-28.json`,
  SHA-256
  `8811FBF0ABAB4FB5A97D332CB2132196F03679BAE6F5B3476B80235EC92E618C`.
- Canonical timing artifact: `dataset_rng_compare_2026-07-28.json`, SHA-256
  `A2BBEBED3E9556B7829B0793BBC1516809E992E706740E981BB3CF69A3DD3C02`.
- Medium worker artifact: `dataset_rng_worker_medium_2026-07-28.json`, SHA-256
  `0EF851167E8E5BE8243F40DDB76B0C035F291186EF16BB76F73651B21C011DD2`.
- Large worker artifact: `dataset_rng_worker_large_2026-07-28.json`, SHA-256
  `5C9F454533F29B8C71A27E3F9B95A3FA5815ED1A051A06E24D05258AC710FF26`.
- Runtime: CPython 3.10.20, NumPy 2.2.6, GCC 11.4.0 on
  `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41`, affinity
  CPUs 0–7.

The host copies of all five JSON artifacts were compared with `sha256sum` in
the oracle container and match byte-for-byte.

## API and representation contract

The production target is `vne_utils_dataset_rng`; it links the completed
`vne_utils_dataset_core`, frozen `random_lib`, and `Threads::Threads`. Public
types and exact behavior are documented in `porting/components/dataset.md`.
The only generation entry point is:

```cpp
GeneratedData generate_data_with_distribution(
    const DistributionRequest& request,
    NumpyRandomState& rng,
    std::size_t cast_workers = 0);
```

`DistributionRequest`, `DistributionSpec`, and `GeneratedData` use direct fixed
members. Distribution and result type are `DistributionKind` and
`DatasetValueKind`; boundary strings resolve once. Result storage is exactly
one contiguous `vector<int64_t>`, `vector<double>`, or dense
`vector<uint8_t>`. There is no string-keyed fixed schema and no string/map
lookup in a draw, sample, transform, cast, or worker loop.

One caller-owned `NumpyRandomState` is the sole stream owner. All generation is
sequential and preserves NumPy draw order. Parallelism is limited to contiguous
post-draw transform/cast blocks; workers never access the RNG. The integer and
boolean exponential fast path fuses the exact `scale * -log(1-u)` transform
with the final cast. Direct tests compare every element with frozen
`NumpyRandomState::exponential` and verify the following stream state.

The wrapper preserves integer-uniform inclusive high, including
`high == INT64_MAX` and the full signed-int64 domain through the documented raw
MT-word hook. It also preserves normal defaults, required-parameter order,
explicit `None`/`monostate`, poisson reciprocal behavior, and the original
uniform+boolean uninitialized-data failure before bounds or draws.

NumPy floating-to-int64 payload semantics are exact: finite in-range values
truncate toward zero and NaN/infinity/out-of-range values become `INT64_MIN`.
Python emits seven locked `RuntimeWarning` records while C++ intentionally has
no process-global warning channel; output bytes and continuation remain exact.

## Exact differential

The direct-loaded pinned Python leaf passed **92/92 cases** against the final
Release harness. Real Torch was neither installed nor imported; only controlled
fake Torch/OmegaConf modules satisfied the original file's eager imports.

The corpus locks:

- normal default sizes 0/1/2/7/257, custom location/scale, int/bool casts,
  zero scale, signed zero, NaN/infinities, huge values, and exact int64 cast
  boundaries;
- float/integer uniform, inclusive high, near/full signed int64 ranges, and the
  uniform+boolean exception before valid, absent, partial, or string bounds;
- exponential/poisson float/int/bool, zero/special values, reciprocal lambda,
  large poisson-to-double rounding, required absent versus explicit `None`,
  negative parameters, and zero-size error timing;
- invalid fixed enums and normal string/explicit-`None` parameters;
- worker-invariant normal buffers at 300,000 values, exponential buffers at
  192,000 and 600,000 values, and exact continuation after every case;
- seven exact Python warning type/message records.

Every ordinary successful value is compared element-by-element or by exact
binary64 bits. Large buffers use a byte-count plus FNV checksum. Every case
then compares explicitly sequenced `random()` and `normal()` bit patterns, so
matching output without matching stream consumption cannot pass.

An additional `python -O` subprocess gate passed **2/2** cases. It records the
assertion-disabled original behavior: customized distribution rejects before a
draw through the misspelled `unsupporrted` path, while normal with an invalid
dtype can reject during the final cast after a draw. These are characterized
Python boundaries, not a request for C++ to disable typed prevalidation.

## Canonical timing

The canonical protocol uses five warm-ups and 31 measured samples. Python/C++
call order alternates inside each pair. Worker widths are interleaved, rotated,
and reversed at every sample to prevent thermal/scheduler drift across separate
width blocks. Allocation, generation, cast, thread creation, and join are
inside the timer. Process startup, RNG construction/seed, checksum, and the two
continuation draws are outside it. Adler-32, output bytes, and both continuation
bit patterns gate every accepted sample.

The six native rows below use worker 1; the six conversion rows show production
automatic mode. Times are total median/MAD/p95 milliseconds for 300,000 values.

| Row | Python median / MAD / p95 ms | C++ median / MAD / p95 ms | Speedup |
|---|---:|---:|---:|
| normal → float | 10.732661 / 0.759016 / 14.573369 | 3.854504 / 0.294355 / 5.371471 | 2.784x |
| uniform → float | 6.688779 / 0.482484 / 9.495920 | 1.161703 / 0.063701 / 1.486620 | 5.758x |
| uniform → int | 6.998338 / 0.422687 / 7.944313 | 0.777697 / 0.036669 / 1.015635 | 8.999x |
| uniform → int, `INT64_MAX` high | 7.782507 / 0.801983 / 9.906279 | 2.006295 / 0.259485 / 2.982204 | 3.879x |
| exponential → float | 9.155122 / 0.317942 / 21.640241 | 2.082712 / 0.204603 / 3.076837 | 4.396x |
| poisson → int | 20.345100 / 0.378524 / 23.372763 | 10.701525 / 0.514404 / 12.686423 | 1.901x |
| normal → int, auto | 10.668629 / 0.708174 / 21.152277 | 5.269346 / 0.432415 / 10.389395 | 2.025x |
| normal → bool, auto | 8.371552 / 0.509141 / 10.565361 | 4.378997 / 0.309804 / 5.398824 | 1.912x |
| exponential → int, auto | 8.550389 / 0.465438 / 12.474213 | 3.007950 / 0.301176 / 3.946175 | 2.843x |
| exponential → bool, auto | 7.075937 / 0.439256 / 9.545656 | 2.147528 / 0.204006 / 3.228685 | 3.295x |
| poisson → float, auto | 23.777890 / 1.119601 / 31.605758 | 12.820125 / 0.665255 / 17.009086 | 1.855x |
| poisson → bool, auto | 21.039671 / 1.052829 / 25.084623 | 11.923654 / 0.454834 / 16.253226 | 1.765x |

All **60/60** canonical rows—six native rows and six conversion rows at every
explicit width 1..8 plus auto—beat Python. The complete speedup range is
**1.645x to 8.999x**.

## Worker policy

Automatic mode remains sequential for native results and every non-exponential
conversion. It is affinity bounded and parallelizes only fused exponential
int/bool transformation:

- fewer than 131,072 values: one lane;
- 131,072–262,143: int uses three lanes, bool uses seven;
- 262,144 or more: both use seven lanes.

The final independent sweeps use the same five-warm-up/31-sample interleaved
protocol. “Best explicit” excludes auto; every one of the 18 rows per size also
beats Python.

| Output / values | Sequential C++ ms | Auto C++ ms | Best explicit width / ms | Auto Python speedup |
|---|---:|---:|---:|---:|
| int / 192,000 | 2.520463 | 2.093627 | 5 / 1.926759 | 2.446x |
| bool / 192,000 | 2.061392 | 1.707211 | 5 / 1.646052 | 2.921x |
| int / 600,000 | 8.111355 | 5.420485 | 7 / 5.214415 | 2.899x |
| bool / 600,000 | 6.111475 | 4.044105 | 5 / 4.091810 | 3.598x |

Auto must beat sequential and stay within 15% of the best explicit width at
every parallel-policy gate. Explicit widths remain available and are capped by
item count plus Linux process affinity (hardware concurrency elsewhere).

## Engineering gates

- Isolated Release build and direct unit: **PASS**.
- Strict production/unit/harness compile with
  `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror`: **PASS**.
- ASan with leak detection, unit and all harness cases: **PASS**.
- UBSan with no recovery, unit and all harness cases: **PASS**.
- Repeated Release unit stress, 100 iterations: **PASS**.
- Release unit constrained with `taskset` to one, two, and eight CPUs: **PASS**.
- Full Release build and repository CTest: **PASS 21/21**.
- Frozen graph/CSV/config/yaml-cpp integrity: **PASS**.
- `random/` source and tests unchanged; strict/sanitizer gates reused the
  existing frozen archive rather than rebuilding that dependency: **PASS**.
- `git diff --check`: **PASS** after final documentation.

Result: the non-Torch dataset **NumPy RNG leaf is COMPLETE on 2026-07-28**.
The XML/graph/GML leaf was subsequently completed and is recorded independently
in `dataset_xml_2026-07-28.md`; the port now proceeds to attribute methods.
