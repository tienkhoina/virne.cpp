# AttributeBenchmarkManager result - 2026-07-28

Status: **COMPLETE**. Scope is prepared node/link/link-sum benchmark reduction,
ordered compact-ID output maps, caller-configured row parallelism, and the
process-global identity-preserving cache. The future `BaseNetwork` adapter owns
data gathering and must reuse this leaf.

## Correctness

- Isolated strict unit: PASS for shape/overflow/error stages, zip truncation,
  extrema/resource/originator behavior, duplicate insertion order, compact-ID
  access/copy/move, exact float32 reduction, workers `0/1/2/8`, concurrency,
  optional group order, and cache identity/concurrency.
- Direct Python oracle: PASS for 19 shared cases plus seven native extension
  cases. Six dynamic/integration boundaries are recorded separately (32 total).
- NumPy 2.2.6 comparisons retain ordered keys and raw binary64 result bits,
  including signed zero, qNaN/sNaN payloads, repetition, and the 16-lane plus
  scalar-tail reduction paths.

## Frozen compact benchmark

Protocol: 4,096 prepared rows by 128 float32 columns, virtual direct-link
column repetition two, one warm-up, three samples, configured workers `1/2/8`.
Fixture construction, process startup, and checksum calculation are excluded.
Every row has checksum `16589509004670834835`, 80,810 output bytes, and 4,096
ordered entries.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 17.032365 ms | 1.857281 ms | 9.171x |
| 2 | 17.032365 ms | 1.837471 ms | 9.269x |
| 8 | 17.032365 ms | 1.484077 ms | 11.477x |

This benchmark passed on its first run. Its source, driver, binary, JSON,
checksum, and timings are frozen and must not be rerun, retuned, or updated
during dependent-module work.

## Performance contract

- Fixed descriptor, matrix, group, error, and operation fields are direct typed
  members/enums.
- Dynamic output names resolve once into `AttributeBenchmarkId`; repeated
  consumers use `at(id)`. Workers receive only float pointers, row indices,
  fixed dimensions/repetition, and pre-sized output slots.
- The direct-link second column copy is virtual and allocates no duplicate
  matrix. Output insertion and duplicate overwrite remain sequential in source
  order. Worker width is caller configuration and independent rows use
  deterministic contiguous blocks.
- Cache mutation is mutex-protected and returns the exact stored
  `shared_ptr<AttributeBenchmarks>` identity.

## Validation

- GCC 11 strict `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -Werror`: PASS for production, unit, harness, and benchmark.
- ASan + UBSan + leak checks: PASS for unit and harness.
- Release full CTest: PASS 28/28; frozen foundation integrity: PASS.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `attribute_benchmark_manager.h` | `0CE6648F08F82926B83FEDDF3FF9B44064F186804137269A1B86680A207ECB33` |
| `attribute_benchmark_manager.cpp` | `79AE5FDA7B87C8CE16286114DEB241FB69FD0BFB0C53B4EF57440D3238EDCE42` |
| unit source | `A59206D1DF8282058302BCA1F3AB284075C75FDB44326827D2D5617125C22FCD` |
| harness source | `398E299B9AD96DAB7CBAD7818A628FCB9D0DE0FA9D82F19EEB72400B28C20D2A` |
| comparator source | `F689DB8B08F701E14549432ADE61B17CEBE8D3F1B2D4731301DCFE186A5DC7C4` |
| benchmark source | `3FFAA7D72FA63186C40FC54523C6FD97FB7BCDDDFB407A4BC59DA9F8A2CD6A08` |
| benchmark driver | `C47E66497F4B2AB7C0732311C5BE15174F1AD4938C5E261F846BA0025FEFDAE9` |
| oracle harness binary | `5828A6A89CCE620988AE041813B18F1B097F2918ED92808213BA84133116E1DD` |
| benchmark binary | `BAE326406927C599E5C7E3C9AD93B41E80BE70DB98F16A842289E72E6BAB1B92` |
| differential JSON | `B0AF1A3AE86C742FE0BCB2D904CC839D9906C01F0A5EA6DC5553BF897A395735` |
| benchmark JSON | `47E1FCCF6D47853B40D62939E58E717D6CD98060125758E8BDF9BE2D9132CEB5` |
