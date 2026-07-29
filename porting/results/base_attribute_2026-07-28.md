# BaseAttribute result — 2026-07-28

Status: **COMPLETE**. Scope is the independent typed BaseAttribute generation
leaf only; node/link/graph adapters, network mutation, ML, solver, and system
code were not started.

## Correctness

- Isolated unit: PASS. Coverage includes all fixed resolvers/fields, snapshot
  and repr, default errors, every standard distribution/dtype lane,
  node/link/graph/zero cardinality, customized validation, exact integer spans
  near `2^62`, RNG continuation, workers `0/1/2/8`, and four concurrent callers.
- Direct oracle: PASS 32/32 against pinned Python source SHA
  `103c5c16126ca76e782c2191ff0811b95ed88a0c3637f3a61e84c0b22df42e8e`
  and dataset source SHA
  `269650ebcc373d7bdf79fa17346bd6f847973f17e60c1ac9bcae7cfd97bf936f`.
  Successful buffers compare every int/bool value or every double raw bit;
  every case also compares the following NumPy RNG value. Error cases lock the
  typed C++ error and RNG continuation.
- Oracle findings fixed before closure: integer customized spans are subtracted
  exactly before floating conversion; absent normal parameters fail before a
  draw; the dataset-only poisson reciprocal flag is ignored at this layer.

## Compact benchmark

Protocol: NumPy 2.2.6, 300,000 outputs, seed 123, one warm-up, three samples,
worker widths `1/2/8`; process startup, checksums, and continuation checks are
outside timed regions. No additional benchmark tuning was performed after all
rows passed.

| Workload | Workers | Python median | C++ median | Speedup |
|---|---:|---:|---:|---:|
| customized float | 1 | 3.871002 ms | 1.503554 ms | 2.574568x |
| customized float | 2 | 3.871002 ms | 2.403897 ms | 1.610303x |
| customized float | 8 | 3.871002 ms | 2.685478 ms | 1.441457x |
| exponential → int | 1 | 8.303804 ms | 5.914410 ms | 1.403995x |
| exponential → int | 2 | 8.303804 ms | 3.534242 ms | 2.349529x |
| exponential → int | 8 | 8.303804 ms | 3.326736 ms | 2.496081x |

Customized draws remain sequential for exact RNG order; configured workers
only transform disjoint blocks afterward. At this representative size thread
startup outweighed that transform, while the completed dataset cast path scaled
and worker 8 was fastest. The API keeps width as caller configuration.

## Validation

- GCC 11 strict compile (`-Wall -Wextra -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -Werror`) for production, unit, harness, and
  benchmark: PASS. Frozen dependency sources were linked from the existing
  build rather than modified for unrelated warnings.
- ASan + UBSan + leak check for unit and harness: PASS.
- Release full CTest: PASS 24/24.
- Frozen integrity and frozen git status for `graph/`, `csv/`, `config/`,
  `libs/yaml-cpp/`, and `random/`: PASS / clean.
- `git diff --check`: PASS.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `base_attribute.h` | `F93CD721ACB7458F92E7581077F4E83EDAFE9A12C48EC489383EF6DC4A906509` |
| `base_attribute.cpp` | `FBBFFD5193FB19994ADDA9112B09B7BCC494ECDEC4BE14FF6E9E151CCA9037C1` |
| unit source | `DA65669BF8CCE88AC4C086BEC6B7501457E7B35A9DD7068E9DCEC4B0DB62BFB7` |
| harness source | `C512AED3E7A3FF6A77094E7BAFCC07A6ACA217AA3CD999CEF506ED110AA8DA45` |
| benchmark source | `3785D3B665446CAA9F9C7F5885898DD082148D94BFD842A9E25C6A26ACB011F0` |
| comparator source | `5FA9780585052D1FD9B0C170C14F200FD58D5FD16BCF6ADB01C4E6EF9FF57F16` |
| benchmark driver | `3A096843BB8108526D1CE38BFAC084C62B75F2E27E307FDCBDC20FFB46D82422` |
| oracle harness binary | `6F29269C19F20F64F79A2742ABA9CF95C36EA6C7C8C04BDD4EC486C018112A00` |
| benchmark binary | `3D57F657B0711E281BE4D3F029D94E67ED87B2416C142DECF326D0F9B6AE81C6` |
| differential JSON | `7DA802496AB1D8875E66A4A79BA2A987B1B342B234A3D0E48D27BE27BB3C454B` |
| benchmark JSON | `303E813D7AB1B2BC341C9E8C187A055A86D99B17383FFD4DB73AE0F74F334C3F` |
