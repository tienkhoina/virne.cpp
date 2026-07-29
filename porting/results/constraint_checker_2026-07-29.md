# ConstraintChecker acceptance — 2026-07-29

State: **COMPLETE / FROZEN**. Do not rerun or update the accepted benchmark.

## API and design

- `ConstraintCheckerSelection` uses virtual-registry `ConstraintId` values.
  `prepare()` resolves each dynamic name once against the independent virtual
  and physical graphs, then hot checks retain only typed attribute pointers,
  graph-local `AttrId` values, direct network pointers, vertices, and result
  slots.
- Graph output IDs are direct, may be sparse or duplicate, and reject the
  reserved invalid ID. Duplicate writes retain the last value as Python does.
- Scalar graph/node/link/path checks preserve Python order and do not
  short-circuit. Path checking resolves the immutable virtual edge once and
  uses ordered physical-map pointers for link and latency checks.
- Node/link/path batches accept caller-configured workers. `0/1` is
  sequential; wider values use deterministic contiguous blocks, retain input
  order, and report the lowest failing request. No automatic host policy is
  embedded.
- All fixed categories, flags, endpoints, errors, operations, and request
  fields are direct fields/enums. Production checker code contains no
  string-keyed map; strings exist only in cold diagnostics and inside the
  completed dependency `bind()` calls during preparation.

## Correctness and safety

- Python source SHA-256:
  `EA41EE9226CEF3F38CFFF30D7ABB276A7A9AF58FEE67CE78C3763413BDB9619A`.
- AST-isolated differential: **14/14 shared cases PASS**, including exact
  bool/int64/double-bit offsets, graph/node/link/path, reversed undirected
  links, combined link/path failures, and empty selections.
- Native unit: seven groups covering hard/soft policies, sparse and duplicate
  IDs, independent virtual/physical registry order, preparation/runtime error
  metadata, workers `0/1/2/8`, lowest-error ordering, and eight concurrent
  callers.
- Release unit and targeted CTest: **2/2 PASS** (checker plus frozen foundation
  integrity). Production, unit, and harness strict GCC 11 warning gates pass.
  ASan, UBSan, and leak detection pass.

## Permanently frozen compact benchmark

Node batch, 32,768 requests, one warm-up, three samples. Preparation and fixture
construction are outside timing; the timed region includes every check and the
exact output checksum.

| Runtime | Median | Speedup vs Python |
|---|---:|---:|
| Python scalar loop | 243.836906 ms | 1.000x |
| C++ workers 1 | 5.256938 ms | 46.384x |
| C++ workers 2 | 5.222010 ms | 46.694x |
| C++ workers 8 | 3.264485 ms | 74.694x |

All rows retained `entry_count=32768`, `feasible_count=24576`, and checksum
`11118347938320421286`.

Artifacts:

- `constraint_checker_differential_2026-07-29.json` SHA-256
  `2856226BA8CA3663B053D3A781D301CAB33626A9A41B922018DE34E0E235F1D0`.
- `constraint_checker_benchmark_2026-07-29.json` SHA-256
  `08013E210A6080F7FEC973578DB752F675996536C63E45DFB62D5ECB2A561DF8`.
- Production header/source SHA-256:
  `6A1465F09EFB53D6BBAA7763D485AD34537CE6E2AA8E68C0FF49EAC7E1720AEA` /
  `589068590E1DD00E063E30CC47A1D7D0F89E81714AB228DBFA7ED6D8B19697CE`.
