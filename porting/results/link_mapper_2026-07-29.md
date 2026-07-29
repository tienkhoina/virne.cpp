# LinkMapper port result - 2026-07-29

Status: **COMPLETE / FROZEN**. The accepted production, unit, harness,
comparator, benchmark sources, and result artifacts must not be edited or
rerun.

## API and design

The stable typed API is documented in
[`../components/link_mapper.md`](../components/link_mapper.md). Fixed methods,
options, results, errors, operations, paths, endpoints, flags, and worker widths
are direct fields/enums. Dynamic resource names bind only during `prepare()`;
candidate, edge, resource, pooling, undo, and mapping loops retain registry
IDs, graph-local `AttrId`s, vertices, typed numeric variants, masks, and direct
Solution tables. A non-resource selection is rejected at this cold typed
boundary. The only production string bind found by the audit is in
`prepare()`; no hot loop uses a string or string-keyed fixed schema.

Candidate path checks may run at the caller's configured width and publish into
pre-sized slots. The original path/error order remains observable. Generation,
ranking, commit, violation recording, undo, and whole-link mapping remain
sequential. Unsafe whole-link mapping is a broken Python surface and is rejected
before mutation; the OR-Tools/SCIP MCF path remains deferred with solver work.

## Correctness and safety

- Python source SHA-256:
  `E7E5FB542D6FCA6A5C9A8BEBACE2C82E9BFEDEF1628CD8ADBB065C4F95D40237`.
- AST-isolated differential: **PASS 17/17**, including workers `0/1/2/8`.
- Native unit: **PASS**, covering safe/unsafe route, exact pooling and
  mutation order, reroute/undo partial state, mapping/clone/cardinality, typed
  boundaries, error suppression, and concurrent independent callers.
- Strict GCC 11 `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -Werror`: **PASS** for production, unit, harness, and benchmark.
- ASan + UBSan + leak detection: **PASS**.
- Targeted CTest: **2/2 PASS** (`vne_link_mapper_unit` and frozen integrity).
- `git diff --check`: **PASS**.

Differential artifact: `link_mapper_differential_2026-07-29.json`.

## Permanently frozen benchmark

Accepted invocation: 1,024 equal-length two-hop candidates, 2,048 physical
edges, candidate workers `1/2/8`, one warm-up, and three timed samples. The
first 1,023 candidates are infeasible and the last is feasible. Complete
ordered route state is checksum-gated before accepting every row. Setup,
process startup, validation, serialization, and fingerprinting are excluded
from the reported route time.

| Candidate workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 8.861967 ms | 0.628866 ms | 14.092x |
| 2 | 8.861967 ms | 0.987043 ms | 8.978x |
| 8 | 8.861967 ms | 1.720057 ms | 5.152x |

Fingerprint: checksum `14052633754962558449`, output bytes `27830`.
Benchmark artifact: `link_mapper_benchmark_2026-07-29.json`. The benchmark is
an AST-isolated LinkMapper leaf comparison with equivalent Python dependency
fakes, not a full solver/system benchmark. Worker 1 wins this compact workload;
the implementation preserves the caller's configured width and embeds no
machine-specific auto policy.

## Frozen file hashes

| File | SHA-256 |
|---|---|
| `virne/core/controller/link_mapper.h` | `AB09436AD19EFD7626BBA17EE9B5D765DCC962251B9198B1BDCB74E62F823943` |
| `virne/core/controller/link_mapper.cpp` | `C3B491DB94D09B6CDB0F558A0AD6603FEE17069E4E256929F056C9A8D532956C` |
| `porting/link_mapper_unit.cpp` | `1F4E7DC760EE600FF8D02F0BB0DE04B9D0BB6998A45EBCFBE048598D8A80154E` |
| `porting/link_mapper_harness.cpp` | `58851E1853D2FC70A3950516F92C7AE91D52DCC4A8F8FD4F287703C9CC213228` |
| `porting/compare_link_mapper.py` | `06A7CDB6BF6EF15D1D336D1A01197E7A5D22E5EE07D4335A85308ADDDCB9AF2D` |
| `porting/link_mapper_benchmark.cpp` | `F2C338D285E936C1B2976DE9E81D01DA37EE9FF7D9A0E4A84920E26B9484CB49` |
| `porting/benchmark_link_mapper.py` | `CAFC05C4C30FA8AD3AE8ADEC8FDB83332049A6DB84D8BE6D167F5A7EC0583DDD` |
| differential artifact | `FC2EEB3AC8F99F0C72C8734AFBCE1B582079B00B6A8690038BD1FC3C0EDC38F8` |
| benchmark artifact | `13E21651741EE9355DE0AB403DD8477A139EEC405960F18AE50E4F8DE9F4A04F` |
