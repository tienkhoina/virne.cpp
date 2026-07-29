# LinkAttribute result - 2026-07-28

Status: **COMPLETE**. Scope is the typed link-attribute adapter and its status,
extrema, resource, and latency leaves; graph attributes and network models are
not part of this leaf.

## Correctness

- Isolated unit: PASS for `Graph`/`DiGraph`, edge order, loops/reversed pairs,
  sparse/dense mutation, atomic short-input failure, separate registries,
  matrices and all aggregations, non-finite values, typed leaves, path
  validation/partial mutation, position resolution, workers `0/1/2/8`, and
  concurrent independent graphs.
- Direct Python oracle: PASS 35 differential cases plus five recorded
  Python-only protocol boundaries (40 total), against source SHA
  `a95cfd2b8e2b46d4de23f70934ca942de502e674e2a7eaa139482df640e8646f`.
  Values use exact tags/bytes or binary64 bits. The SciPy-free pinned image uses
  a controlled dense sparse facade only for matrix materialization consumed by
  this leaf; the frozen graph sparse API has its own differential gate.

## Frozen compact benchmark

Protocol: CPython 3.10.20, NetworkX 3.4.2, NumPy 2.2.6, 50,000 edges,
one warm-up, three samples, and configured workers `1/2/8`. Fixture/process and
checksum work are excluded. Every row first passed exact checksum/output bytes.

| Workload | Workers | Python median | C++ median | Speedup |
|---|---:|---:|---:|---:|
| dense edge roundtrip | 1 | 105.988164 ms | 17.214467 ms | 6.157x |
| dense edge roundtrip | 2 | 105.988164 ms | 14.307932 ms | 7.408x |
| dense edge roundtrip | 8 | 105.988164 ms | 18.523222 ms | 5.722x |
| position-derived latency | 1 | 443.752286 ms | 8.709616 ms | 50.950x |
| position-derived latency | 2 | 443.752286 ms | 6.097162 ms | 72.780x |
| position-derived latency | 8 | 443.752286 ms | 51.901279 ms | 8.550x |

Worker width is caller configuration; no machine-specific auto threshold is
embedded. Ordered sparse/path mutation remains sequential. These accepted
benchmark sources, commands, JSON, checksums, and timings are frozen: do not
rerun, retune, or update them during dependent-module work.

## Performance contract

- Fixed owner/kind/restriction/checking-level/aggregation/generation fields are
  direct typed members or enums.
- Dynamic link and position names resolve once into graph-local `AttrId`s.
  Edge, matrix, aggregation, distance, and path hot loops use only descriptors,
  IDs, and direct attribute storage.
- Independent dense work uses deterministic contiguous blocks. Path mutation
  remains ordered to preserve Python's observable partial side effects.

## Validation

- GCC 11 strict warnings (`-Wall -Wextra -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -Werror`): PASS for production, unit, harness, and
  benchmark.
- ASan + UBSan + leak checks: PASS for unit and harness.
- Release full CTest: PASS 26/26; its test list contains no benchmark target.
- Frozen-component integrity: PASS. Scoped whitespace validation for final
  Node/Link production, unit, and harness files: PASS. Repository-wide
  `diff --check` remains noisy from pre-existing CRLF/trailing whitespace in
  unrelated files and was not rewritten.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `link_attribute.h` | `61AC7ACB1600349080E08F9BDC25EEF1D8B6518358FD85E0EA2384B413441E21` |
| `link_attribute.cpp` | `9ADC59D9EB30F94BC00BB8160386D53642816F7D24F24B63AD0BAC502CFBD076` |
| unit source | `AF985516C4D745FA8E9A12AFB0259EC12A91E9E53F43B5997DA4E3D23B68E18C` |
| harness source | `DAAD62DDDF86F244E06E68D654796D886D07CF9947DC4E9F3ECB0F8CA0746529` |
| benchmark source | `E1AA7DE22F53700966E27B04CC92AED093764386B64EA55BEF6FB1D9ACFDE1AE` |
| comparator source | `8411376238A90BE3D675A9FEB4287DE34268DBB7B23833FBEB5B4E8B33FA586E` |
| benchmark driver | `8B76298FD3DAF26DAC351C6B1070372DB658CCD9B5BA8C485E5609F5A9585C03` |
| oracle harness binary | `D4776B808FA70837FB77768D9019412ABA912228B52BE8766D5424AB3D5043BA` |
| benchmark binary | `8FCD81592A994F05A82914635D01243EF3FFB5D8B7F9E5D8BF82DAE4F2981878` |
| differential JSON | `7F9CCAAE626EDA3D5D8F2B79A5868326B0838281A3B0D2E1FE4847239793A9EA` |
| benchmark JSON | `81394929BE6D2A426CABC430B0E90A944D6594FE86C5199E44ED314F5F74BB31` |
