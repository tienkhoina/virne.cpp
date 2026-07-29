# `attribute_method` completion record — 2026-07-28

State: **COMPLETE for the independent typed-policy leaf**.

Read `porting/components/attribute_method.md` for the stable API and exact
Python behavior. Future components should use that document before opening this
implementation or the original Python source.

## Scope

The leaf provides typed owners/kinds/specs, string-to-enum boundary resolvers,
bool/int64/double resource update, exact mixed numeric comparison, constraint
satisfiability, typed errors, and a contiguous double batch extension.
Generation/extrema callbacks require `BaseAttribute` plus definition/graph
registries and remain in that owning layer; their Python MRO and fallback quirks
are already frozen by this leaf's oracle.

No graph, CSV, config, yaml-cpp, random, solver, system, or ML source changed.

## Data and hot-loop rules

- Fixed owner/kind/update/comparison/restriction/error/operation values are
  enums or direct struct fields.
- Dynamic method strings resolve once at the public boundary.
- `AttributeNumber` is a boundary variant; homogeneous double batch loops use
  direct contiguous pointers and bytes only.
- The x86 fast path runtime-dispatches to strict-IEEE AVX-512, then AVX2;
  other targets use the scalar implementation.
- There is no string, map, registry, variant, allocation, or virtual dispatch
  per batch element.
- Output may alias either input vector. Thread-creation fallback calculates only
  blocks that did not already finish, so aliased input is never transformed
  twice.

## Worker configuration

`workers` is a typed caller/config field. Zero and one are sequential; larger
values are capped by item count and visible CPU affinity. Production contains
no host-specific automatic threshold. A representative exact 4,000,000-item
`hard/le` run measured:

| Configured workers | Effective | Median ms |
|---:|---:|---:|
| 0 | 1 | 6.870821 |
| 1 | 1 | 7.079821 |
| 2 | 2 | 6.074713 |
| 3 | 3 | 6.386566 |
| 4 | 4 | 7.343987 |
| 5 | 5 | 6.892936 |
| 6 | 6 | 6.634807 |
| 7 | 7 | 7.444011 |
| 8 | 8 | 7.590657 |

Width two was 1.165458x faster than width one on this corpus. This is a
configuration example, not a compiled-in policy. All widths produced Python
checksum `16492953567995315926`.

## Exactness coverage

The direct-source oracle locks original commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, source SHA-256
`e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f`.

- Scalar/resource/resolver/calculation/dynamic-MRO: 93/93.
- Batch raw-bit cases: 30/30.
- Combined: 123/123.
- Batch matrix: `ge/le/eq × hard/soft × workers 0/1/2/3/8`.
- Values include both signed zeros, subnormals, infinities, five qNaN payloads,
  and stable sNaN cases.
- Unit additionally covers every width 0..8, bool/int64/double boundaries,
  values beyond `2^53`, both int64 ends, overflow errors, output/input aliasing,
  invalid enums/shapes, and eight concurrent callers with 32 rounds each.

Soft batch lanes skip the discarded comparison for throughput. Returned flags
and offset bits are exact; floating-environment/trapping side effects are not a
batch API guarantee. Scalar APIs preserve the original evaluation behavior.

## Representative Python/C++ timing

The quick timing signal used one warm-up and three samples at 10,000 ordinary
operations; all checksums were exact. Timing is intentionally compact because
semantic coverage is handled by the larger differential/unit matrix.

| Row | C++ speedup over Python |
|---|---:|
| resource add int | 50.670980x |
| resource subtract double | 115.556359x |
| guarded subtract int | 85.662914x |
| hard ge int | 24.545575x |
| hard le double | 23.057900x |
| hard eq double | 19.044881x |
| soft ge int | 22.670592x |

The exception-only guard-failure row was 0.944572x and is report-only. It does
not represent the normal numeric hot path and did not justify changing exact
error construction.

## Build and validation

- Isolated Release build/unit/harness: PASS.
- Warning-clean `-Wall -Wextra -Wpedantic -Werror`: PASS.
- ASan, UBSan, and leak unit: PASS.
- Focused CTest: PASS.
- Differential: PASS 123/123.
- Concurrent/alias unit: PASS.

Compact evidence files:

- `attribute_method_differential_2026-07-28.json`;
- `attribute_method_benchmark_2026-07-28.json`;
- `attribute_method_worker_sweep_2026-07-28.json`.

Raw container artifacts had SHA-256 values `651d125e...f0bd`,
`e80d6c13...d6a8`, and `85fe984a...77d` respectively.

## Stable hashes

- `attribute_method.h`:
  `7E64F19F8EF0D4CF670193197E927E4ABC9D8AE938193560E3C96CA91C36A209`
- `attribute_method.cpp`:
  `A02426F39E203EDAF1A2D0770C3314743D9A0F1D744C005AA815BCFACBDDDF40`
- unit source:
  `0663E6DC4B9A6FB95E68A8A888A3810A5173B89D4825AE78AEE68A35678D422B`
- harness source:
  `774410B89B439137ECDDFE702732D8C18F6CB900BF3C030894A99776F8C5CAF9`
- harness binary:
  `640F6DB56AE35BD88D1370E7EEFE47D910E054334EBA9FC926B10CF37BD2A782`
