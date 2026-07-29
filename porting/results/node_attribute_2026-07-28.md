# NodeAttribute result - 2026-07-28

Status: **COMPLETE**. Scope is the typed node-attribute adapter and its status,
extrema, resource, and position leaves. Link/graph attributes, network models,
solver, system, and learning code were not included in this leaf.

## Correctness

- Isolated unit: PASS. It covers Graph/DiGraph, separate attribute registries,
  sparse duplicate writes and ignored out-of-range nodes, dense atomic length
  validation, every supported `AttrValue` lane, missing/partial data, typed
  errors, status/extrema invariants, resource updates/checks, position fallback,
  NumPy clip edge cases, workers `0/1/2/8`, and concurrent independent graphs.
- Direct Python oracle: PASS 37 differential cases against node source SHA
  `e90e286320e59ebbba9957b701f6f12acdc1785e4821de80ccc5a0ab0f3ee56a`,
  BaseAttribute SHA
  `103c5c16126ca76e782c2191ff0811b95ed88a0c3637f3a61e84c0b22df42e8e`,
  and attribute-method SHA
  `e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f`.
  Five Python-only dynamic-protocol boundaries are recorded separately, for 42
  total cases. Values compare by exact tag/integer/string bytes or double bits.
- Position generation compares all x/y/radius values and the following RNG
  value after the canonical x, y, radius draw sequence. Dense/sparse adapter
  output preserves NetworkX node order.

## Compact benchmark

Protocol: CPython 3.10.20, NetworkX 3.4.2, NumPy 2.2.6, 100,000 nodes, seed
123, one warm-up, three samples, and worker widths `1/2/8`. Fixture creation,
process startup, checksums, and continuation checks are excluded. Every sample
first passes exact checksum, output-byte, and (where applicable) RNG-state gates.

| Workload | Workers | Python median | C++ median | Speedup |
|---|---:|---:|---:|---:|
| dense set/get roundtrip | 1 | 189.573237 ms | 21.328620 ms | 8.888x |
| dense set/get roundtrip | 2 | 189.573237 ms | 31.587641 ms | 6.002x |
| dense set/get roundtrip | 8 | 189.573237 ms | 22.226209 ms | 8.529x |
| position generation | 1 | 37.762409 ms | 5.818109 ms | 6.490x |
| position generation | 2 | 37.762409 ms | 4.481054 ms | 8.427x |
| position generation | 8 | 37.762409 ms | 5.262022 ms | 7.176x |

The caller controls worker width. Dense roundtrip is sequentially fastest on
this fixture because `AttrValue` cloning and thread creation dominate; position
post-processing benefits at width 2 in this sample. The API intentionally embeds no
machine-specific automatic threshold.

## Performance contract

- Fixed owner/kind/restriction/checking-level/operation fields are direct enums
  or typed members.
- A dynamic attribute name is resolved once by `bind(graph)` into an `AttrId`.
  Every node hot loop receives that ID and directly accesses `node_attrs(v)`; no
  loop or worker hashes, resolves, or compares a string.
- Virtual and physical graphs are bound separately because their registries can
  assign different numeric IDs. Sparse writes remain sequential to preserve
  observable order; dense unique-node work and position post-processing may use
  disjoint contiguous worker blocks. RNG draws remain canonical and sequential.

## Validation

- GCC 11 strict compile (`-Wall -Wextra -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -Werror`) for production, unit, harness, and
  benchmark: PASS. Vendored Boost is treated as a system include.
- ASan + UBSan + leak checks for unit and harness: PASS.
- Release full CTest: PASS 26/26 after the dependent LinkAttribute leaf.
- Frozen component integrity: PASS.
- The strict gate found one signed iterator conversion only in the unit test;
  it was corrected with the container `difference_type`. A later semantic audit
  also fixed resource missing-vs-nonnumeric access precedence for check, add,
  unsafe subtract, and safe subtract; four direct oracle cases lock this order.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `node_attribute.h` | `3D3C4AAAC10BA0398EC6BA05867602D8F495E45834F4EE2EC07F94EB6D3F1F90` |
| `node_attribute.cpp` | `FE273B744833AE0B19F8A75417C8312A2B3C516E6B3072D731488E2503FACB60` |
| unit source | `7275A5D570581307527092A81D30A03E86B8CCD91005FAE8A28F13F48DE5A9DB` |
| harness source | `52650820B5AFD8B642994DA58D74D38515D142EEBC212E5C770FB8D5B3A3AD93` |
| benchmark source | `BE84312DCB30D454BDD29582EA22321EBC9A7B464C4C60314790388B0C2C976D` |
| comparator source | `69E400969F7DEBB0B8AB7BC46DDEFC34245BFB5A55ED5DE928D8BE95D38C0F3D` |
| benchmark driver | `E97BFBC44C85DD45963DE87BF3AFA57BE5C2841A4777B76DEA84DDF781EF850B` |
| oracle harness binary | `0EDF08A800DAEEE1C797DF76E8D9687B2763F10A0C73194F24508C202E2FEDCD` |
| benchmark binary | `79A3C078FE2B164EEC9A0A067C8DA3D6DDF3D62B588AA16A4753D39E87C97916` |
| differential JSON | `DE8CF46C6CD7448725F074DA09E0349DD675560E830B16015D9EA27311D6329D` |
| benchmark JSON | `72EF1B3A8B2770C5BCA9E995EC99286A03973D7A74E150775D2D97BCD9C16A94` |
