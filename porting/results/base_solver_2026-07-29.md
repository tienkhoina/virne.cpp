# BaseSolver result - 2026-07-29

Status: **PASS / COMPLETE / FROZEN**.

Scope is the independent non-ML `solver.base_solver` foundation: a typed
solver configuration snapshot, non-owning runtime collaborator references,
base ready/solve behavior, compact `SolverId` registry, explicit factory
registration, and reserved category seams for later optional modules. Concrete
solvers, systems, MCF/OR-Tools, RL, Torch, CUDA, and learning imports were not
linked.

## Stable API

`SolverDependencies` holds const `Controller` and `Counter` references plus
mutable `Recorder` and `Logger` references. `Solver` owns one fixed
`SolverConfig`, exposes the original collaborators directly, and maintains its
arrived-request counter. `SolverRegistry` owns descriptors/factories in
registration order, resolves a dynamic name once to `SolverId`, freezes before
concurrent reads, and creates caller-owned solver instances by ID.
`SolverErrorCode`, `SolverOperation`, and `SolverException` provide concrete
typed parse, construction, registry, ID, factory, overflow, and base-solve
failures.

## Correctness and safety

- Exact Python differential: `13/13` shared cases at native workers `1/2/8`.
- Focused unit coverage passed configuration/path behavior, collaborator
  identity and constness, counter overflow, base ready/solve, all categories,
  registry ordering/errors/freeze, direct IDs, null and throwing factories,
  register-vs-freeze linearization, and concurrent frozen resolve/list/create
  readers at workers `1/2/8`.
- Strict GCC 11 warnings-as-errors production, unit, and harness builds passed.
- ASan/UBSan/leak checks passed; targeted CTest passed `2/2`.
- Hot-ID audit passed: name strings are registration/resolve boundaries only;
  repeated lookup and creation use compact direct slots. Registration uses
  geometric vector growth rather than per-item `reserve(size()+1)`, avoiding
  quadratic relocation.

After the benchmark was accepted, registration-only vector growth was fixed
and the transition/factory/concurrent-reader unit coverage was expanded. The
release unit, strict production/unit/harness compiles, ASan/UBSan/leak unit,
targeted CTest, aggregate solver build, and exact differential were rerun on
the hashes below and passed. The frozen benchmark was not rerun or edited:
descriptor registration and fixture construction are outside its timed path,
so the accepted measurements and checksum remain unchanged provenance.

## Frozen benchmark

Fixture: 32 pre-registered descriptors and 4,096 typed solver holders, one
warm-up and three samples. Fixture construction, process startup,
serialization, and checksum calculation were outside timing.

| Route | Workers | Median ms | Speedup |
| --- | ---: | ---: | ---: |
| Python | 1 | 17.813469 | 1.000x |
| C++ | 1 | 1.594245 | 11.174x |
| C++ | 2 | 1.798216 | 9.906x |
| C++ | 8 | 1.712720 | 10.401x |

Every route produced 4,096 entries, 419,498 bytes, and checksum
`13751587758314786690`.

## Provenance

- Python source commit: `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Python source SHA-256:
  `196573E631654A6D14888685B93977601412F110A9B5620398850CCE121A2806`.
- `base_solver.h`: `1738C394669F756842179D7466CF0BE4F44092B91BE5A5764471A05A67E83CE8`.
- `base_solver.cpp`: `6A0205D78ECC45E75CDE4F27D3CBA41CDE8CF7B7323565EEB16A9938371EA4C4`.
- Unit: `B3BFF0108203FA09DB497FECA4DF712CDEFDD9F606626FDCF70CBAFFF5618FE3`.
- Harness: `6B254A2E2EC8F2482EC3E0011AD450D75EE42452E946F5CB7217684973F4CB39`.
- Differential driver:
  `B8476947F97442DCDBE00DE605AC52B3C5819426FE121D43A0BB7BB922972673`.
- Benchmark binary source:
  `7ACCB0A42BB6EC432C345A66FDBCE059B8FAC91C62B508A7B88FEE13123EE7CB`.
- Benchmark driver:
  `F81D820538E2E0E8923CB1187134371B3AABE8D71D57A9028DA03873D2EA3919`.
- Differential JSON:
  `018F11B9A065BB07F09240F8DA7AC9707D567BF0508BF31778884CD31E6315F2`.
- Benchmark JSON:
  `4E66A7400A383580A89F43637BAA98D6BCF5524DD720EA7B6AD5D96C8CCA38C3`.

The benchmark source, driver, JSON, measurements, and this accepted result are
now provenance. Do not rerun, edit, or reinterpret them while porting later
leaves.
