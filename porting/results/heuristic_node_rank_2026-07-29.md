# HeuristicNodeRank result - 2026-07-29

Status: **COMPLETE / FROZEN**.

## Acceptance evidence

- The real Controller, NodeRanker, NodeMapper, LinkMapper, Network and Solution
  stack passes unit coverage for API/registration, config and worker forwarding,
  successful and failed solves, retained partial output, unchanged input,
  multi-link graph/path order, workers `0/1/2/8`, and concurrent instances.
- Differential: **PASS**, six shared cases including five solution cases, for
  native workers `0/1/2/8`. The exact Python `BaseNodeRankSolver` and
  `OrderRankSolver` AST nodes are pinned by source hash.
- Strict production, unit, harness and benchmark builds pass. ASan, UBSan and
  leak checks pass. Targeted CTest is **2/2 PASS** and the aggregate build
  passes.
- The direct-ID audit passes: fixed fields are accessed directly, prepared
  numeric IDs remain in hot loops, and `order_rank` is confined to the cold
  registration boundary. Worker width remains caller configuration with no
  automatic tuning.

## Frozen benchmark

This is an **AST-isolated mixed-dependency conservative microbenchmark**, not
a full original-stack end-to-end benchmark. Native timing covers the full C++
solve pipeline, including the per-solve physical-network clone and real frozen
rank/controller/mappers. Python timing executes the exact target solver classes
with lightweight deterministic rank/mapper doubles and NetworkX shortest-path
routing. Registration, fixture preparation, serialization and output gates are
outside the measured region.

The single sparse fixture has 16 virtual nodes/17 links and 32 physical
nodes/31 links, greedy matching and BFS shortest paths. It uses one warm-up,
three samples and 64 solves per sample. Every row passes the same ordered
slot/path gate: **64 entries**, **87,752 bytes**, checksum
`9328970994111537605`.

| Runtime | Workers | Median total (ns / 64) | ns/solve | Speedup vs Python |
|---|---:|---:|---:|---:|
| Python | sequential | 5,601,775 | 87,527.734375 | 1.0000000x |
| C++ | 1 | 3,612,310 | 56,442.34375 | 1.5507459x |
| C++ | 2 | 125,296,850 | 1,957,763.28125 | 0.0447080x |
| C++ | 8 | 444,591,684 | 6,946,745.0625 | 0.0125998x |

The wider-worker overhead on this compact fixture is recorded as observed; it
does not trigger auto-tuning or a hard-coded worker policy. This benchmark and
its JSON are frozen and must not be rerun or updated.

## SHA-256 provenance

| Artifact | SHA-256 |
|---|---|
| Python target `../virne/virne/solver/heuristic/node_rank.py` | `44A6F63F1A1798935453C057C6981A692F5DC4F03BDE6A6535607AECBDC61389` |
| Public API `virne/solver/heuristic/node_rank.h` | `BD3512AAF16C40E23538381D59A7F391EFDCAD81FA05709484F8E1CA48588B71` |
| Production `virne/solver/heuristic/node_rank.cpp` | `0D2334449FD83371854C114C3519962D092EC98AC5A0BD3824167CE3F9729A64` |
| Unit `porting/heuristic_node_rank_unit.cpp` | `B9BD487C7DD1F3F561169D8E7C8BE09A0ED9F901E23A53B8F9FDE65421CE3DF8` |
| Harness `porting/heuristic_node_rank_harness.cpp` | `916162588BA68E024EE23E87260598BD8FCC044090FFD5CB2AE52736E8760AE3` |
| Differential driver `porting/compare_heuristic_node_rank.py` | `AC0D2CFF54045FCD8A1C15F0B000283953DEE49E862E0BEFD182023AFC182EC2` |
| Native benchmark `porting/heuristic_node_rank_benchmark.cpp` | `846F66E821F3E498F7439BFBC9FFE8B07C1426D050EA61FD007D1A508155DC59` |
| Benchmark driver `porting/benchmark_heuristic_node_rank.py` | `BB8BE27A626C011DF96C085A4C43F0D32EAB3159D5CC7490B63BA5F961D5C96A` |
| Differential JSON | `7788734245E206588C37371368548F587E3A8FE386E87F07FBC0841E049A5F20` |
| Benchmark JSON | `6F0D6D14BD9F255F7225CEB203FE6521DF962E141AC02D485CEF99C1788C47D1` |
| Component API note | `D7C1BB4B20526BEE2764BFFE993AFE9EA2D64D8B6FBAC010AAB3B1740135F81E` |

The result note itself is omitted from the table because a file cannot contain
its own stable cryptographic hash.
