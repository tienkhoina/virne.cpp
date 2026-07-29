# PhysicalNetwork result - 2026-07-28

Status: **COMPLETE / FROZEN**. The non-ML native leaf implements typed physical
network construction, generated or GML-loaded topology, Python-compatible RNG
ordering, dataset persistence, typed load diagnostics, and independent clone.

## API and correctness

- `PhysicalNetwork::from_setting` accepts either the global RNG context or a
  caller-owned `RandomContext`, plus direct worker fields in
  `PhysicalNetworkBuildOptions`. Fixed options are typed fields; dynamic names
  bind once to compact registry/graph IDs before graph loops.
- `PhysicalNetworkBuildReport` exposes generated, loaded-GML, or
  generated-after-GML-error origin without stdout parsing. `to_gml`,
  `save_dataset`, `load_dataset`, and `clone` complete the persistence and
  ownership boundary.
- Differential: **PASS**, 19 classified cases: 11 shared Python/C++ cases, one
  native clone/move extension, and seven explicit Python-only dynamic
  boundaries. Shared coverage includes workers `0/1/2/8`, stream
  continuation, GML success/empty/error fallback, missing `num_nodes`, and
  dataset load.

## Frozen compact benchmark

Protocol: 32,768 nodes; one warm-up and three samples; fixture/input creation,
process startup, and fingerprinting excluded. Binary output is gated before
acceptance. All rows match checksum `18010873477772571912`, 65,535 entries,
and 524,304 output bytes.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 100.674790 ms | 20.839576 ms | 4.831x |
| 2 | 100.674790 ms | 18.041472 ms | 5.580x |
| 8 | 100.674790 ms | 23.817123 ms | 4.227x |

This accepted benchmark is provenance only and must not be rerun or updated.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Python `physical_network.py` | `E37D48B2C1651B503931597B6CCA5620A413A11C3D003CF8C70AF89246E4CA5A` |
| Python `base_network.py` | `94CB1185B5E6F0046D9393F97AF1EE0DD1C0D688596B102A0EC3CC98314ADDCE` |
| `physical_network.h` | `765C2F1D9E2ED6BEFC81574F8B8BD2AFBB9B9C62F9D456CFC75AA19E959FAA30` |
| `physical_network.cpp` | `BD1736C4660B4623E91E710ABF368B2E537CCFBF7E32DF179E45DF6CA7EE67EF` |
| unit source | `5CA5328D6205703CF7BCF53E2362821CC4CF86E6BBB746E28F7DAA523CA1882E` |
| benchmark source | `AC87C9017377B8F79022DBD7675F3E73798534D9907503F1AA3336953080DB48` |
| benchmark driver | `FBB4E21B2DECF29F41325768E41E3D5E87861C537735151B4072C07ECD92A6CE` |
| differential JSON | `E5EDBBEC981FFBDF317F29DD8D993535AF7CF9B671FDB5733C42516753E16123` |
| benchmark JSON | `6EE8BD3154A883CFA58D4BECACDBD765F0E6D49B36AFEF44FAE71146C8D8D0F3` |
| accepted benchmark artifact | `84E555A54C47915AB2CB9E72850BEC4B5B2ADA425278D431CB66C381A6B07DBC` |
