# GraphAttribute result - 2026-07-28

Status: **COMPLETE**. Scope is scalar and independent-batch graph metadata plus
status, extrema, and resource leaves.

## Correctness

- Isolated unit: PASS for Graph/DiGraph, every `AttrValue` lane/raw double bit,
  graph-local definition bindings, fixed fields, extrema link delegation,
  resource missing/conversion/update order, ignored `safe`, aliasing/overflow,
  batch atomicity/order, workers `0/1/2/8`, and concurrent callers.
- Direct Python oracle: PASS 45 shared cases plus four native extension cases;
  five Python-only boundaries are recorded separately (54 total). Source SHA is
  `dfa858918068a792cb0a673b400ee6f3f93107f1d17baad28f72546f50fbcede`.

## Frozen compact benchmark

Protocol: 20,000 independent graphs, raw64 double output, one warm-up, three
samples, configured workers `1/2/8`; fixture creation, binding, process startup,
and checksum calculation are excluded. Checksum is
`5125455356797706885`, output is 160,000 bytes for every row.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 13.767438 ms | 5.305297 ms | 2.595x |
| 2 | 13.767438 ms | 7.530612 ms | 1.828x |
| 8 | 13.767438 ms | 11.388209 ms | 1.209x |

The first timing attempt exposed sequential-only `unordered_set` construction
and double initialization in batch get. Production removed those measured
costs; the unchanged protocol then passed. These accepted benchmark sources,
binary, JSON, checksums, and timings are now frozen and must not be rerun or
updated during dependent-module work.

## Performance contract

- Fixed owner/kind/restriction/checking-level fields are direct typed members.
- `bind` resolves a dynamic name once to `AttrId` and carries compact map and
  definition identity tokens. Scalar validation is pointer-only; batch validates
  slots before dispatch, and workers use only pointer, ID, direct `AttrMap`, and
  pre-sized indices—no string/hash/registry lookup.
- Worker count is caller configuration. Duplicate mutable maps fall back to
  canonical sequential order; shared resource mutation remains scalar.

## Validation

- GCC 11 strict warnings-as-errors: PASS for production, unit, harness, and
  benchmark.
- ASan + UBSan + leak checks: PASS for unit and harness.
- Release full CTest: PASS 27/27; frozen-component integrity: PASS.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `graph_attribute.h` | `821790079AF447805ADBE44ED49527E1A1BA04CB4C147D7A24C2BC4285920F1A` |
| `graph_attribute.cpp` | `E85C157649426D34BD3CCD3DF38D8957B72C9291B624EE4330EA945CE904E031` |
| unit source | `1EEA1782760620920CCD5B25298C6E6072129086E855E5C1137CC1E43E14F9B4` |
| harness source | `77B53E5510201CC856762FBC6EC5AFE7CF736CC67198C560DCA986DF8B63CC70` |
| benchmark source | `DEDC998B3D3AFCAB5C67F7D245F43AB0FF1F955B571F7BD5827CE12F8CB68181` |
| comparator source | `C848FF15EEFE0FA6A3D1167BF1A23A331D9CD8AE43400C4F1B570AD319532DCF` |
| benchmark driver | `7278548921E9E1CBA372BEAEE7FB55E911124EE8E0E5B45489E7FF3DFBBC16DE` |
| oracle harness binary | `283B794F445C5727BF7D9420A74737F414089E0C6B6029CA3CB5E406AFB681BC` |
| benchmark binary | `23C70BDE125ABDAD61A2A07A3DBEFDC354FA08E83A79E6C3DE39605B18ADF270` |
| differential JSON | `43D9EC58D180043EEB9A82FEC5A6359EAF3776430402483B74E8E80C636715F2` |
| benchmark JSON | `46634CEF7C4AE7C02CEEF729D5EBBD3FCB6B4E2AA3A7A4ED861BD1C2C10C024C` |
