# NodeMapper port result — 2026-07-29

Status: **COMPLETE / FROZEN**. The accepted benchmark, source, unit, harness,
comparator, and artifacts must not be rerun or edited.

## API and design

The stable API is documented in
[`../components/node_mapper.md`](../components/node_mapper.md). Fixed matching
methods, flags, results, errors, operations, candidates, and worker width are
direct fields/enums. Dynamic resource names bind only at `prepare()`; hot
placement/mapping/undo loops retain `Vertex`, `ResourceId`, `ConstraintId`,
`AttrId`, direct Solution tables, numeric variants, and a byte hard-constraint
mask. The hot-string audit found only the cold `bind_node_attribute` call and
no string-keyed map.

`greedy` may check independent candidates concurrently at the explicit caller
width. Results/errors are consumed in original candidate order, including the
rule that a later error is ignored after an earlier feasible candidate.
Resource mutation, candidate removal, and virtual-node progress remain
sequential. `l2s2` remains scalar because failure of its first candidate is
public behavior.

## Correctness and safety

- Python source SHA-256:
  `8AD2AE077E61732DCB77FFA08269AAB630BDB6DF29969C983D09A3565BA860F9`.
- AST-isolated differential: **PASS 12/12**, including workers `0/1/2/8`.
- Native unit: **PASS**, including duplicate-resource collapse, empty-hard-set
  mutation order, failure flags, clone/reuse, later candidate errors, and
  concurrent independent callers.
- Strict GCC 11 `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -Werror`: **PASS** for production, unit, harness, and benchmark.
- ASan + UBSan + leak detection: **PASS**.
- Targeted CTest: **2/2 PASS** (`vne_node_mapper_unit` and frozen integrity).
- `git diff --check`: **PASS**.

Differential artifact:
`node_mapper_differential_2026-07-29.json`.

## Permanently frozen benchmark

One warm-up and three samples; fixture/preparation, subprocess startup, and
fingerprinting are excluded from timed work. Exact ordered slots, resource
info, offsets, violations, result flags, and every final physical capacity are
gated before timing.

| Candidate workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 61.280082 ms | 3.289713 ms | 18.628x |
| 2 | 61.280082 ms | 16.008148 ms | 3.828x |
| 8 | 61.280082 ms | 19.886030 ms | 3.082x |

Corpus: 32 virtual nodes, 2,048 initially infeasible candidates plus 32 valid
candidates, at least 65,568 checks. Fingerprint: checksum
`15604526718891224062`, output bytes `9744`, entry count `2208`.

Benchmark artifact: `node_mapper_benchmark_2026-07-29.json`. This workload
favors worker 1; the implementation deliberately preserves the caller's
configured width instead of embedding benchmark-selected auto-tuning.

## Frozen file hashes

| File | SHA-256 |
|---|---|
| `virne/core/controller/node_mapper.h` | `7F1D64615E65E3473A95846CFA3E4727026E2E947D00091F0F2F523BE2BEB473` |
| `virne/core/controller/node_mapper.cpp` | `B01842C2B981D6E9CB836E9B6C1F44C873A0191EB639D0657052D5A1E6B5EFF9` |
| `porting/node_mapper_unit.cpp` | `4C3F6E084E887982B4AA1EB1BA574ED1A7BD6FB38AB354BDFF00323FDB2FC0CC` |
| `porting/node_mapper_harness.cpp` | `D8477517F432BB46296EC056185EB13D632606ACCFBC8C85153E6FB6AB52E986` |
| `porting/compare_node_mapper.py` | `B5DA7A347700D5C566C39BC4D5501D631FE2483FEC20331051F106F2624676AA` |
| `porting/node_mapper_benchmark.cpp` | `9117EA4BCE9937F4D36398AD8D09E991C536546ADE648759C30A43925781C8B8` |
| `porting/benchmark_node_mapper.py` | `C3429590C912F20ED72DBC7BBFB473013942DA97EFC896BE350FD64E40CF1996` |
| differential artifact | `0E7C401D4F34B3C2B2C153E916DB9D180981474E8F666969E9C4B443261089D1` |
| benchmark artifact | `CE95423F1B6EEDF89CD3DA121EF1981A73B3927DF2FEF55A859134A111D83FE2` |
