# `core.environment` non-RL handoff — 2026-07-29

State: **COMPLETE / FROZEN**.

## API and scope

`BaseEnvironment` owns typed reset/event preparation, const physical-network
access, prepared Controller/Counter arrays, Recorder history, direct event and
arrival-record slots, automatic leave transit, recovery through
`drain_leaves()`, and typed summaries. `SolutionStepEnvironment::step` consumes
a caller-built Solution. Fixed lifecycle fields are direct members/enums;
dynamic request IDs resolve once during reset and every event hot loop uses
dense numeric slots.

Solver registries, system wrappers, JointPR interaction, observation, reward,
action masks, features, Torch, RL and ML are not linked.

## Correctness gate

- Exact Python differential: **9/9 PASS**, workers `1/2/8`, source SHA-256
  `6004CFF2114E504E30C5490232763F71CB9E7799216C4F6CE8BADC27A0E42B34`.
- Unit: **PASS**, including workers `0/1/2/8`, sparse request IDs,
  accepted/rejected/leave/reset/error/recovery and concurrent environments.
- Strict GCC 11 (`-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -Werror`): **PASS**.
- ASan/UBSan/leaks on Environment production plus the full unit: **PASS**.
- CTest `frozen_component_integrity|vne_environment_unit`: **2/2 PASS**.

## Frozen benchmark

96 requests, 192 ordered events, 64 accepted; one warm-up and three samples.
The timer contains arrival admission/count/deploy/record/transit and automatic
leave release/record only.

| Runtime | Median ms | Speedup |
|---|---:|---:|
| Python AST oracle | 9675.094 | 1.000x |
| C++ workers 1 | 65.883 | 146.854x |
| C++ workers 2 | 254.199 | 38.061x |
| C++ workers 8 | 993.396 | 9.739x |

Exact gates: 192 records, 72,481 bytes, checksum
`17358322786803582063`, final physical raw64 checksum
`5251282115348753471`. Wider workers remain caller configuration; this compact
workload correctly favors worker 1 and no automatic worker policy is embedded.

## Provenance hashes

- `virne/core/environment.h`:
  `CE821B41DAF17C76496220B5C17FBA8580123F9FE2D42A71E8AB2FD47488E21D`
- `virne/core/environment.cpp`:
  `2EE1C2D9B29A8AA97E3578B781983FBC1E34BE8686E2E6CE710B5C8883C29FF8`
- `porting/environment_unit.cpp`:
  `DD8DEB8178679703F1DCA2D74C3112B71FF52B39D43B3E69A07EAA69E1EFC049`
- `porting/environment_harness.cpp`:
  `5D2911F4CA8D4861075B7BEE04A16AD3957525045A73E2915917A58758AC0AAD`
- `porting/compare_environment.py`:
  `2186F59FC7335B75CF0DB7F527C66A27D038BBA05B9032362BBC410946D3EF1A`
- `porting/environment_benchmark.cpp`:
  `90A9452DA9C2BA0F7A56A452ECDB6963576A39D72A408F25061181AA08B8C874`
- `porting/benchmark_environment.py`:
  `E258DA259C7055A60219287A9732A1A8E2A03B57169379C71C8E54A015248266`
- differential JSON:
  `0E4D6ED68A40466454049F7A39F298239C2D009E18C3D43993E4B38F1DCA3074`
- benchmark JSON:
  `22426F3ADD7BBA4565EE560AAEC6CB955911D39A04F7701AD41F141E79DB6185`

The accepted benchmark, its code and JSON are frozen; do not rerun or revise
them while porting later modules.
