# ResourceUpdator acceptance - 2026-07-29

State: **COMPLETE / FROZEN**. Do not rerun or update the accepted benchmark.

## API and design

- `ResourceUpdatorSelection` contains virtual-registry `ResourceId` values.
  `prepare()` binds each genuinely dynamic name once against the independent
  virtual and physical graphs. Hot updates retain only typed attribute
  pointers, graph-local `AttrId` values, vertices, direct maps, and numeric
  variants.
- Fixed owners, operations, requests, amounts, targets, errors, indexes, and
  flags are direct fields/enums. Production code has no string-keyed map or
  string lookup in an update loop; strings are limited to cold diagnostics and
  completed dependency `bind()` calls during preparation.
- Scalar lists and paths preserve Python first-error and partial-mutation
  order. Path updates use selected-attribute order outside physical-link order;
  duplicate selected resource IDs deliberately repeat the mutation.
- Node/link batches take a caller-configured worker count. `0/1` and duplicate
  targets use canonical sequential order. Wider disjoint batches preflight
  direct map copies, report the lowest failing request without committing, and
  then commit disjoint targets in deterministic contiguous blocks.
- Prepared instances are non-owning and must not outlive or race external
  mutation of their virtual/physical networks.

## Correctness and safety

- Python source SHA-256:
  `9B7C14F8C6EAA5E8BC50A723B727FEC7EFF8F1C7D05F7BEF0DA0CC941C15AC85`.
- AST-isolated differential: **10/10 shared cases PASS**, covering node, link,
  reversed undirected link, path ordering, duplicates, partial failure, empty
  selection, unsafe subtraction, and bool promotion.
- Native unit: seven groups covering independent registry order, typed family
  and range errors, bool/int64/double arithmetic, insufficient/missing/
  nonnumeric/overflow failures, partial mutation, paths, workers `0/1/2/8`,
  duplicate fallback, disjoint atomic failure, lowest-error selection, and
  concurrent independent networks.
- Release and strict GCC 11 production/unit/harness gates pass. ASan, UBSan,
  and leak detection pass. Targeted CTest: **2/2 PASS** (ResourceUpdator unit
  plus frozen foundation integrity).

## Permanently frozen compact benchmark

Disjoint node batch, 32,768 updates, one warm-up, three samples. Fixture and
preparation are outside timing. Every row retained `entry_count=32768`,
`value_sum=3407839`, and checksum `17411705748429442498`.

| Runtime | Median | Speedup vs Python |
|---|---:|---:|
| Python scalar loop | 83.935910 ms | 1.000x |
| C++ workers 1 | 2.534859 ms | 33.113x |
| C++ workers 2 | 15.145464 ms | 5.542x |
| C++ workers 8 | 15.444391 ms | 5.435x |

All configured widths beat Python. This workload favors worker 1; worker count
is intentionally a config input, so no automatic tuning is embedded.

Artifacts:

- Differential JSON SHA-256:
  `C9342A1DDA5D3425F4D17D71D1F9850A2A6E1AA89CC61DB41149E2C270B48C1E`.
- Benchmark JSON SHA-256:
  `4A6F740828B5DF514E9879212940A5934DA46A73EAF8CDB9881C91265368BC1D`.
- Production header/source SHA-256:
  `A10F6C896F590DEA45830B4EA9D195258382BF7B382D41AFE1238614C39C2E42` /
  `33F2A4E836F3C846E778B04D34DA72FB3B1951516880CB5327BE4FC2954D1C1B`.
- Unit/harness SHA-256:
  `64C0BB54D3814419CBAC634772FD7E442787966D1EC71EED732A11CA03FFAFD2` /
  `5B17D2F7EE9044B4C081EC9FA21567EB76E8487D3A9DAF11EDD52C46364C601E`.
- Comparator/benchmark-driver SHA-256:
  `5028CC40DC3114EDC1B5373D972C9E27A82E01C5B357DD050342C26239481B7C` /
  `84FDBEA6F37E02A6D4976CD92BC8D9AC05E0DB4C7D6C114A6D3EAE4F52986A7E`.
