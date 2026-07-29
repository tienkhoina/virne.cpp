# VirtualNetworkEvent result - 2026-07-29

Status: **COMPLETE / FROZEN**.

## API

`VirtualNetworkEvent` stores fixed `id`, `VirtualEventType`, request ID, and
time fields directly. Typed constructors, accessors, invariant-preserving
setters, and `repr()` are exposed together with deterministic batch creation
and stable time sorting. Caller-selected workers `0/1/2/8` use pre-sized
indexed output; no dynamic string lookup or automatic worker policy enters the
hot path.

## Differential

**PASS**, 20 classified cases: 13 shared Python/native cases, one native NaN
rejection extension, and six recorded Python-only dynamic boundaries. Values,
validation/error order, setters, representation, stable ties, configured
worker invariance, and lowest-input-index batch errors match the accepted
oracle where the typed domains overlap.

## Frozen benchmark

The accepted construct-and-sort workload has 131,072 events, one warm-up and
three samples. Fixture creation, process startup, and fingerprinting are
excluded; all rows retain 131,072 entries, 3,276,800 output bytes, and checksum
`15302777944810256441`. The benchmark is frozen after this first PASS.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 349.346420 ms | 7.400389 ms | 47.206x |
| 2 | 349.346420 ms | 6.496323 ms | 53.776x |
| 8 | 349.346420 ms | 10.185598 ms | 34.298x |

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Pinned Python source | `970E63F9DAC59F60E2ED1786606DC87D3271AF062BA1D8D6C67AEF8D3C7478E1` |
| Recorded benchmark program | `2A8542E66B790D159E390D183FA9BC6B4AA09593720E90B751E1D17AAF2CBB7B` |
| Differential JSON | `238B195044D649F26BE8ADE944219A1B22AB0C2724C11EB6A2E752B787808A03` |
| Benchmark JSON | `0F099788C3B03101780364A14DA114360D5728ED950A6A2A10B07CBC42227431` |
