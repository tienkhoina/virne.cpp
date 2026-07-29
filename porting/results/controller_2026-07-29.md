# Controller lifecycle acceptance — 2026-07-29

Status: **PASS / FROZEN**.

Scope is the non-solver lifecycle only: prepared place-and-route for completed
non-MCF path modes, undo, deploy, release, and Python-compatible undo-deploy.
MCF, BFS deployment/search, candidate search, system, RL, Torch, and ML are not
linked.

- Exact Python differential: **10/10 PASS**, including partial mutation,
  rollback order, failed/no-op lifecycle, and workers `1/2/8`.
- Production/unit/harness/benchmark strict GCC 11: **PASS**.
- Focused unit, eight independent concurrent callers, ASan, UBSan, leak
  detection, and `frozen_component_integrity|vne_controller_unit`: **PASS**.
- Hot-loop audit: fixed fields are direct; dynamic resource names bind once;
  neighbor/resource/pool/batch loops use vertices, edge IDs, registry IDs,
  graph-local `AttrId`s, direct slots, and byte masks.

The permanently frozen benchmark performs 8,192 node and 8,192 disjoint-link
updates in both deploy and release (32,768 mutations), with one warm-up and
three samples. It runs exact `Controller.deploy/release` plus exact
`ResourceUpdator` Python AST methods on a NetworkX graph. Both the deployed and
restored checksums match native exactly.

| Runtime | Median | Speedup |
|---|---:|---:|
| Python | 33.624408 ms | 1.000x |
| C++ worker 1 | 3.892177 ms | 8.639x |
| C++ worker 2 | 6.484015 ms | 5.186x |
| C++ worker 8 | 11.785875 ms | 2.853x |

Deployed checksum: `17514356897791579542`; restored checksum:
`8486823302284311477`; serialized numeric output: 131,072 bytes.

Frozen SHA-256:

- `controller.h`: `08F31D62067EADFB3DCD9B90A316552E5EADA001F50636A92A77338CFCA7372F`
- `controller.cpp`: `4342A54F598A8FA88EB6CD561A60F967240054C8CC95807B8095822337628DC9`
- `controller_unit.cpp`: `3F829328852300B165A990ECF0F5CE34CDE97DCEB985B13C7A9C67D1203F42E5`
- `controller_harness.cpp`: `AD1C741702C2F13D68112AF9CFEBA476E3024FF3F2C1F8ED8C2A888E099268F0`
- `compare_controller.py`: `48B8048A2BD83F55F9D4456A73740B39962575505E2BC31CD43F0C49D5FF4C57`
- `controller_benchmark.cpp`: `768AB16D0CDCF20B4AEBE1024E602E2722FEC290972BD76D6F0001C81AA5B3A0`
- `benchmark_controller.py`: `BF3974D8E8DFC6D075BB8F491EC5B3462DA490D03E21C325E104CDE5D57FE637`
- differential JSON: `8144DEBF9C0290ED3DB0DB4353674EDE45ECFE85D6655A973E94A213D0ADA7E5`
- benchmark JSON: `A20D1A0883380BF1BA757C4B4E912B5614B237E970226166E0596884E81E8D6C`

Do not edit or rerun the accepted production, differential, or benchmark
surface. Continue from `porting/components/controller.md` and this result.
