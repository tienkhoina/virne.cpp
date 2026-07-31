# Internal hot-path regression — 2026-07-31

Status: **PASS**. Public APIs, fixed-field layouts, dynamic attribute-ID
contracts, output order, error order, and transaction semantics are unchanged.
The accepted Python/Linux artifacts remain frozen.

The final controlled ClangCL A/B, checksums, toolchain flags, and sample
protocol are published in
`porting/results/hot_path_old_vs_new_2026-07-31.md`. Key final speedups versus
the same-compiler old implementation at eight workers are:

| Module | Old ms | Final ms | Speedup |
|---|---:|---:|---:|
| Counter | 21.1854 | 20.1579 | 1.051x |
| Recorder | 8.6792 | 5.4048 | 1.606x |
| Controller | 5.1718 | 3.3370 | 1.550x |
| NodeMapper | 26.4855 | 14.2869 | 1.854x |
| LinkMapper | 2.7254 | 2.5227 | 1.080x |
| Environment | 539.8481 | 42.9124 | 12.580x |

Focused units passed for all six modules. A separate 20,000-call stress sweep
also passed while changing worker width from 0 through 9, confirming exact
single-visit block coverage.

## Internal changes

- persistent deterministic workers replace repeated thread construction;
- task state is snapshotted consistently under one mutex, with atomic
  completion and exception-safe pool growth;
- inactive workers park immediately and active workers use only a bounded
  pause window;
- caller-local reusable scratch removes repeated hot-loop allocations;
- Recorder derives the whole-network total from one node and one link scan;
- Controller reuses numeric-ID mutation/checkpoint and mapper/path buffers;
- Environment avoids redundant physical clones and summary copies;
- LinkMapper uses larger ordered path windows to amortize barriers.

Worker width remains explicit configuration. No host-derived auto-tuning,
string-keyed hot-loop access, meta-heuristic, or ML implementation was added.
