# VirtualNetwork result - 2026-07-28

Status: **COMPLETE / FROZEN**.

## API

`VirtualNetwork` keeps request ID, arrival, lifetime, and maximum latency in
direct optional fields. Resource definitions
bind once to compact registry/graph IDs; demand loops use typed values and
preserve Python ordering and cached-total behavior. Caller-selected workers
`0/1/2/8`, clone/move, GML, failure-to-zero, cache invalidation, and independent
concurrent networks are covered.

## Differential

**PASS**, 20 classified cases: 14 shared Python/native cases, one native move
extension, and five explicit Python-only dynamic boundaries. Fixed metadata,
assignment, demand ordering/types, worker invariance, stale cache, invalidation,
ragged/nonnumeric fallback, clone, and empty/no-resource behavior match the
accepted oracle.

## Frozen benchmark

The accepted workload has 32,768 elements and 131,070 entries, one warm-up and
three samples. Fixture creation, process startup, and fingerprinting are
excluded; raw64 output and checksum `13238708660769685600` are gated before
timing. The benchmark was not rerun during this closeout.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 67.354577 ms | 9.724267 ms | 6.926x |
| 2 | 67.354577 ms | 8.869010 ms | 7.594x |
| 8 | 67.354577 ms | 13.552468 ms | 4.970x |

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Python source | `0B73E73FEB43793559976F08FD93ED227698B810ED8741EA5BCC1534ADB3768C` |
| `virtual_network.h` | `18B5752CDBCB5C149598A8ED1C79FD7F9B7382FA55EAC334F244D5592E7C14CF` |
| `virtual_network.cpp` | `465778F3E785FA890526D0E59A7C2BB4FD357F6B28839310BD062506E182C2CF` |
| unit source | `093BBC426527900867AD9A27CF63E3AD7D4F916048F2375BF090B9216D30CB8B` |
| harness source | `51AAB26076569B03A119888E7346420871BB133582D10EA530D79C97F0894E1C` |
| comparator source | `8EBDA0881008DE60061404899EF7119371EB4C341A815A3FE11853E0A22AF701` |
| benchmark source | `7F774E5C882FABE9F2A38B951B95EB931E06E93D53AFEE93F8CC6DB4689C6019` |
| benchmark driver | `E8AE9C54426B4073D31D55E22307605B65FECB9A451214B6BD30D189A4BDC3AB` |
| differential JSON | `7882EA26FC57D007868B979BD317241A500618AF9BA716EB092593D2389BC8CA` |
| benchmark JSON | `6DC49F150BC84D7F97EE2229C9F16DBCEC78D0B804BAC596AA1F62DC8FF5CD12` |
| accepted benchmark artifact | `88416F20143DB8C380852AEDA12C63B838BFA2DD65E146BAF2374DD4EB45369F` |
