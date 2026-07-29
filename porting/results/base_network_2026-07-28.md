# BaseNetwork result - 2026-07-28

Status: **COMPLETE**. The native undirected non-ML BaseNetwork owns a frozen
Graph, typed node/link AttributeFactory registries, compact graph bindings,
Python-compatible cached cardinalities, ordered generation/data adapters,
views/clone/GML/setting boundaries, and the prepared AttributeBenchmarkManager
adapter. PhysicalNetwork and VirtualNetwork remain later modules.

## API and correctness

- Fixed config, owner/kind/layout/aggregation/error fields are direct typed
  members/enums. Dynamic definition names bind once to `AttributeRegistryId`
  and graph values bind once to graph-local `AttrId`; hot loops use IDs/direct
  slots and validate registry/graph identity.
- Unit coverage includes merge/dedup, raw decode, move/clone/rebind, stale
  caches, topology/generation partial mutation, selection precedence, dense and
  sparse setters, row/adjacency/aggregation adapters, existence order, views,
  and exact manager delegation at workers `0/1/2/8` plus concurrent networks.
- Differential: PASS for 29 shared Python cases. Eleven dynamic Python-only
  boundaries are recorded separately (40 total), including the `None`
  originator dictionary-key behavior outside the native string-key domain.
- The measured unnormalized sum path aggregates frozen row-major sparse COO
  by compact value ID, preserving dense column-sum bit order without an
  `O(n^2)` temporary.

## Frozen compact benchmark

Protocol: 8,192 elements, resolved `get`, resolved `set`, and prepared manager
operations; one warm-up, three samples, workers `1/2/8`. Fixture/input setup,
process startup, and fingerprinting are excluded. Raw value types/bits/order,
entry counts, output bytes, and checksums are gated before timing.

| Operation | Workers | Python median | C++ median | Speedup |
|---|---:|---:|---:|---:|
| get | 1 | 20.519328 ms | 3.243619 ms | 6.326x |
| get | 2 | 20.519328 ms | 3.933634 ms | 5.216x |
| get | 8 | 20.519328 ms | 6.322977 ms | 3.245x |
| set | 1 | 20.447408 ms | 3.148405 ms | 6.495x |
| set | 2 | 20.447408 ms | 4.593106 ms | 4.452x |
| set | 8 | 20.447408 ms | 8.525752 ms | 2.398x |
| manager | 1 | 588.476812 ms | 5.659331 ms | 103.983x |
| manager | 2 | 588.476812 ms | 8.670046 ms | 67.875x |
| manager | 8 | 588.476812 ms | 8.281702 ms | 71.057x |

Checksums are `3693647448735006061` for get,
`2294184608147702832` for set, and `7234129939705340940` for manager. This
accepted benchmark is now provenance only and must not be rerun or updated.

## Validation

- GCC 11 `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
  -Werror`: PASS for production, unit, harness, and benchmark.
- ASan + UBSan + leak checks: PASS for the isolated unit.
- Release full CTest: PASS 30/30, including frozen-foundation integrity.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `base_network.h` | `BCA9972FE9733EFC059222A6C382DCED145BAA93858C86C74109B874D6E301DF` |
| `base_network.cpp` | `08CA2DBB1F9D67887EE18DE97E33DAC2098AE26932476D413AEA38F61B954B57` |
| unit source | `42819A664FA61E3CE8D50A7AE977DC2799CC43314AE7F7EF75DB7EA871546C7F` |
| harness source | `35B836598B30C8F12E926741596AAA6BCD2E85C0C9463BE8C3C78CE2D3B00D39` |
| comparator source | `D97FEB94EF216752F7B468A17491EAEE47B0B648855619754550CD1D705B3ADD` |
| benchmark source | `3E34BFD23446267A26B2E0F59056F740FC96706D27F91772E73FB52D951F8813` |
| benchmark driver | `B475A5536A039A3A84D7B0296578E1292D107575A49CE4AAD574875FFDB1D8CE` |
| harness binary | `F17C29823F1867D77BD78300E3DE6EF366CED70500417FE44FC7D14AEFC0CFDE` |
| benchmark binary | `2ACC7804648F0AA0F9343D1DF5CD59B5E31B1FA8ED86A2F5454F3833DF784621` |
| differential JSON | `26508287B9A638171F98ED9D383092E2B4D293D52CC4429606ACA61102882F6A` |
| benchmark JSON | `2CF8B98C5632D399EF283DB440C3B83508D96ABFCD7CCD049B369D8257436839` |
