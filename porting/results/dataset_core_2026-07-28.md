# Results: non-Torch `virne.utils.dataset` core — 2026-07-28

State: **CORE COMPLETE**. NumPy-compatible generation, `set_seed`, XML parsing,
graph materialization, and GML output remain explicitly unimplemented.

## Locked identity

- Original Python leaf: `virne/utils/dataset.py` at commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Python source SHA-256:
  `269650EBCC373D7BDF79FA17346BD6F847973F17E60C1AC9BCAE7CFD97BF936F`;
  exact size 9,635 bytes.
- Final Release unit SHA-256:
  `D21488081E9DAB86BABC158961FE137DF184AEF86F817FADB719DBD319C525F5`.
- Final Release harness SHA-256:
  `622A778A47A0F300BC99A80FE74A8993DC20A5B49ACEF8556B76D34DDE9E9B28`.
- Differential artifact:
  `porting/results/dataset_core_differential_2026-07-28.json`, SHA-256
  `193020AB73900B3019C50AE002F1F729741AA4D4D4C709A50402F9B91E9ED237`.
- Canonical timing artifact:
  `porting/results/dataset_core_compare_2026-07-28.json`, SHA-256
  `394E735E98A4657581F1654E5562D71906F9F78C329E2C54AAEE45F7B3BC3287`.
- Runtime: CPython 3.10.20, NumPy 2.2.6, GCC 11.4.0 on
  `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41`, eight
  visible CPUs.

## Exact differential

The direct-loaded Python leaf passed **60/60** typed compatibility cases.
The loader verified the source bytes first, installed controlled fake `torch`
and `omegaconf` modules, and proved no real Torch backend was imported or
touched by the selected core APIs.

The corpus locks:

- every parameter-helper discriminator, missing-parameter order, the locked
  `normal`/`UnboundLocalError` mapping, and the always-empty average stub;
- Python scalar spelling for integers, floats, booleans, `None`, strings,
  Unicode, embedded NUL/newline, NaN, infinities, and signed zero;
- filename item order and literal unescaped separators;
- generated/existing/missing/hidden/multi-dot physical topology paths, empty
  attribute groups, integer/boolean/negative seeds, dynamic Unicode names,
  physical and virtual golden paths, and the customized/normal lifetime paths;
- identical filename/physical/virtual batch checksums for explicit workers
  1, 2, 4, 8 and automatic mode.

An additional **16,395 binary64 bit-pattern cases** compared C++ formatting
against Python `str(float)` exactly. They include deterministic boundaries,
subnormals, signed zeros, infinities/NaNs, and 16,384 seeded random payloads.
No tolerance or normalization was used.

## Canonical timing

The canonical protocol used five warm-ups and 31 paired alternating measured
samples. Fixture construction, source loading, process startup, hashing, and
verification were outside the timer. Thread construction and joining are
inside the timed C++ batch API, so worker results include their real public
overhead. Every measured sample matched checksum and output byte count before
its time was accepted.

Scalar medians/MAD/p95 are total milliseconds for each row's operation count.

| Row | Operations | Python median / MAD / p95 ms | C++ median / MAD / p95 ms | Speedup |
|---|---:|---:|---:|---:|
| parameter extraction + string | 100,000 | 55.873823 / 1.197563 / 64.622433 | 4.853469 / 0.168226 / 6.581262 | 11.512x |
| filename | 30,000 | 52.307604 / 1.228176 / 63.912988 | 3.631546 / 0.205757 / 6.306620 | 14.404x |
| physical generated path | 20,000 | 109.379132 / 3.474836 / 136.799314 | 17.657638 / 0.575984 / 24.084006 | 6.194x |
| physical existing-file path | 10,000 | 75.396894 / 5.298241 / 114.630133 | 16.780460 / 1.134689 / 27.679053 | 4.493x |
| virtual path | 20,000 | 92.562303 / 4.710280 / 110.023295 | 14.850263 / 0.781688 / 18.151935 | 6.233x |

Batch cells below are `C++ median ms / Python speedup`. Filename runs four
8,192-item calls; physical and virtual each run three 4,096-item calls.

| Requested workers | Filename | Physical | Virtual |
|---:|---:|---:|---:|
| 1 | 4.215528 / 12.628x | 14.532111 / 4.710x | 12.880301 / 4.571x |
| 2 | 4.848263 / 10.671x | 10.729517 / 6.590x | 9.426343 / 6.569x |
| 3 | 4.642806 / 11.143x | 9.320189 / 7.438x | 8.522122 / 7.359x |
| 4 | 4.559417 / 11.272x | 8.212192 / 8.295x | 7.522302 / 7.913x |
| 5 | 5.102622 / 10.359x | 7.900216 / 8.669x | 7.280607 / 8.292x |
| 6 | 5.203810 / 10.074x | **7.861640 / 8.685x** | **7.208163 / 8.008x** |
| 7 | 6.106462 / 8.423x | 8.009469 / 8.682x | 7.484669 / 7.472x |
| 8 | 6.173734 / 8.319x | 9.393767 / 7.457x | 8.432081 / 6.616x |
| auto | 4.401567 / 11.926x | 7.887645 / 8.846x | 7.330752 / 7.781x |

All **32/32 timing rows** beat Python. Accepted semantic facts are invariant:

| Family | Checksum | Output bytes |
|---|---:|---:|
| filename batch | `15786229347802482891` | 452,436 |
| physical batch | `7757782050140820735` | 501,398 |
| virtual batch | `11657498433134710547` | 411,286 |

Automatic mode is family/size aware, based on rotated/reversed exploratory
1..8 sweeps at 512, 1,024, 2,048, 4,096, 8,192, and 16,384 items:

- filename remains sequential below 16,384 because thread/allocator overhead
  wins at the canonical 8,192 items; at 16,384 it uses five workers, measured
  at 14.710 ms versus 16.008 ms for one worker;
- path batches remain sequential below 1,024, use four workers through 3,071,
  and use up to six workers thereafter; automatic mode respects CPU affinity;
- explicit widths remain available for controlled external tuning.

## API and performance contract

The isolated `vne_utils_dataset_core` target links only C++17 and
`Threads::Threads`. It implements the parameter helpers, locked empty average,
filename builder, typed physical/virtual path builders, and deterministic batch
extensions. It does not expose a misleading partial `set_seed`, RNG generator,
XML parser, graph adapter, or GML writer.

All fixed configuration is stored in direct members. Distribution/topology
strings resolve once to enums. Dynamic attribute spelling is owned once beside
compact `DatasetAttrId`; hot formatting loops use the direct enum, scalar, ID,
and string fields without mapping/string-key lookup. Path builders use direct
append formatting and avoid the original Python's redundant attribute scans.
Parallel batches pre-size outputs, preserve input order, and report the
lowest-index typed failure.

## Engineering gates

- Isolated Release build and unit: **PASS**.
- Strict `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror`
  production/unit/harness builds: **PASS**.
- ASan, UBSan, and leak detection: **PASS**.
- Repeated unit stress, 100 iterations: **PASS**.
- Full Release build and repository CTest: **PASS 20/20**.
- Frozen graph/CSV/config/yaml-cpp integrity: **PASS**.
- `git diff --check`: **PASS** after documentation finalization.

Result: the non-Torch dataset **core leaf is COMPLETE on 2026-07-28**. Future
work begins from `porting/components/dataset.md`; the next dataset decisions
are the NumPy legacy RNG compatibility boundary and, separately, XML/GML.
