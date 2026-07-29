# TopologyAnalyzer acceptance - 2026-07-29

State: **COMPLETE / FROZEN**. Do not rerun or update the accepted benchmark.

## API and design

- The six path modes, endpoints, `k`, maximum path-node count, worker widths,
  requests, operations, and errors are direct fields/enums. Unique dynamic
  link-resource names bind once during `prepare()` to independent virtual and
  physical registry/value IDs; duplicate selections then reuse those prepared
  records in original order.
- Path, edge, predicate, candidate, and worker loops use only `Vertex`, stable
  edge IDs, graph-local `AttrId`, typed attribute pointers, direct maps/masks,
  and pre-sized result slots. Production has no string-keyed map or repeated
  string lookup; strings are limited to cold diagnostics and the one prepare
  binding boundary.
- First/available shortest path uses completed raw-order BFS plus direct
  predecessor reconstruction to reproduce NetworkX
  `dijkstra_path(weight=None)` FIFO ties. A discovered cyclic-tie mismatch in
  the broader benchmark corpus was converted into two permanent four-node
  differential cases; frozen Graph code was not edited.
- Available/pruned mask construction and complete request batches accept
  caller-configured workers. Zero/one is sequential; wider widths use
  deterministic contiguous blocks and input-order results. Constraint BFS
  remains sequential because edge-check and neighbor order are observable.
- Available views read live virtual/physical values. Pruned views snapshot the
  adjusted virtual map while physical values remain live. Prepared analyzers
  and views are non-owning.

## Correctness and safety

- Python source SHA-256:
  `665519C5C4BF50C2318E2D22D30679881DDA0EDCBC0B618C4FB2055AC5A01B28`.
- AST-isolated differential: **24/24 shared cases PASS**, covering six modes,
  ordinary and cyclic tie order, negative/zero `k`, node-count cutoff,
  first-path-only maximum, no-path, hard/soft available/BFS behavior, prune
  ratio/div/order/equality/duplicates, and empty selection.
- Native unit: nine groups covering source-equals-target, invalid endpoints,
  live/snapshot views, independent registry IDs, workers `0/1/2/8`, batch
  order/invalid-enum metadata, and concurrent callers.
- Release, strict GCC 11 production/unit/harness/benchmark, ASan, UBSan, and
  leak gates pass. Targeted CTest: **2/2 PASS** (TopologyAnalyzer unit plus
  frozen-foundation integrity).

## Permanently frozen compact benchmark

First-shortest batch on a 512-node path graph, 4,096 requests, one warm-up,
three samples. The unique-path corpus makes exact ordered output a prerequisite
to timing. All rows retained `path_count=4096`, `vertex_count=716112`, and
checksum `10025764477037659827`.

| Runtime | Median | Speedup vs Python |
|---|---:|---:|
| Python scalar loop | 1409.883334 ms | 1.000x |
| C++ workers 1 | 16.830783 ms | 83.768x |
| C++ workers 2 | 9.582848 ms | 147.126x |
| C++ workers 8 | 7.933787 ms | 177.706x |

Artifacts:

- Differential JSON SHA-256:
  `8246D74C83B0BD02C19AAE8182C4E880D437C0BDB3F07B35CF2BC0FE66E4FC7A`.
- Benchmark JSON SHA-256:
  `86525F9C0098BBC2FA377A8EC39ADDD346505D1862284B09F2D8F4947ECA68C6`.
- Production header/source SHA-256:
  `9A830DC9359912E6191D08E16F709FC465A8302BF66B4E14DDD24174BC641885` /
  `1AE697B58A529F34C75FD7F1C88A5740211A634A45E1D26A11537978C4497399`.
- Unit/harness SHA-256:
  `3D56ADA6A84F99E1E96CF526ACBE20FB48F00EDF92201F2C14E55CA2B87F0A41` /
  `4E28B0563ED96BAD1AC072123586A2491011D0E1A10B9F9A3669EB3C1B941FA5`.
- Comparator/benchmark C++/benchmark driver SHA-256:
  `15E699401BFF923B5E93ADE6B8971FC9C7EB3F64F66597C43823EB274F160FED` /
  `1E257087CDAD7F9CDDD1894DBEA224B20BDF93A5BA7D7AD723E04D15D167DF9A` /
  `A8F689AED295AE3EE687A032A51F3E9A70A1C7A22CBB03DCDC453C20246F24C2`.
