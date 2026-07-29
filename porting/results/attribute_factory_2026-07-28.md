# AttributeFactory result - 2026-07-28

Status: **COMPLETE**. This leaf decodes raw settings once, constructs the eight
registered node/link attribute pairs, and owns insertion-ordered typed
registries with compact definition IDs. Graph-owner factory pairs remain
unsupported exactly as in the pinned Python registry.

## API and correctness

- Stable API: `AttributeFactorySpec`, raw-setting decoder, general/node/link/
  graph create helpers, move-only general/node/link registries, and setting/spec
  batch helpers with caller-configured workers.
- Fixed owner/kind/distribution/dtype/restriction/checking/generation fields are
  enums/direct members. Dynamic names hash once on insertion and bind once to
  `AttributeRegistryId`; hot consumers use direct indexed entries.
- Differential: PASS for 29 shared Python cases plus three native typed cases;
  seven Python-only dynamic boundaries are recorded separately (39 total).
  This includes all eight pairs, defaults, null/invalid fields, graph omission,
  duplicate order, compact IDs, extrema resolution, workers `0/1/2/8`, and
  deterministic lowest-index errors.

## Frozen compact benchmark

Protocol: 32,768 typed specs, one warm-up, three samples, workers `1/2/8`.
Fixture/fingerprint and process startup are excluded; every sample first gates
ordered type/direct-field output at checksum `13127048606653777947`,
10,326,911 bytes, and 32,768 entries.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 114.526599 ms | 19.154425 ms | 5.979x |
| 2 | 114.526599 ms | 17.849414 ms | 6.416x |
| 8 | 114.526599 ms | 23.752271 ms | 4.822x |

The benchmark passed and is now provenance only. Do not rerun, retune, edit,
or replace its source, driver, binary, JSON, checksum, or timings.

## Validation

- Strict GCC 11 warnings-as-errors: PASS for production, unit, and differential
  harness. The already accepted benchmark source remains immutable.
- ASan + UBSan + leak checks: PASS for the isolated unit.
- Release full CTest: PASS 29/29, including frozen-foundation integrity.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `attribute_factory.h` | `0CC906A308C8A6FEF920571629854C4EEB11295867FE8BBE7C6762D28EAED5E3` |
| `attribute_factory.cpp` | `3B550270E9278C081BA9CD66C493274C8CF996F30B9910CAB36DC113F7B2A0E0` |
| unit source | `BD897548E06A26F638F548B4A66EF367CCA7136F4A4FFFBDCCBE3B0E4B100879` |
| harness source | `6B646AD22265EFF9A1446AE4FC42DE18B5A1FE65CEE187E33185B7CC4E9CCD0E` |
| comparator source | `D94AED98F623ECD228DBDEEE42081CF558DC949E4A9911C07774954B96DF09A5` |
| benchmark source | `5B8A1A710B911496017265DED24B7BC7EBB5AACFCC1E6A49B646A4E691013BF3` |
| benchmark driver | `261C575119473E32B2ECD8AC309E375E37F589289750ADBDC954F9755D8D473B` |
| harness binary | `EC25A6E44CA7EEB92BA60DCC5A9BC859AB08B31DB60FC72DD0C13A0D25FF574A` |
| benchmark binary | `AC9C7757D64FA8DE8A3A9B67D489877304B0C0A31C914701B35A55BC84ACE222` |
| differential JSON | `F4D44D3EBED609771C53A4674BF5BB1948222FB9D2ABBEBE55F53050A3555CF4` |
| benchmark JSON | `7E22016F950BF018EFAE4135CA524CBCA223170B1E89B057C79320C514B4F43A` |
