# Main transaction integration verification — 2026-07-30

Status: **PASS**. This is one end-to-end verification sample, not a replacement
or rerun of any frozen component benchmark.

## Case

- `setting/main.yaml`, `solver.solver_name=ffd_rank`, seed `0`;
- no override for `p_net_setting.topology.num_nodes`,
  `v_sim_setting.num_v_nets`, or virtual-network size;
- save backends, progress, and trace capture disabled while timing;
- native component worker widths fixed to `1`.

Both runtimes produced exactly:

```text
physical_nodes=100
physical_links=528
virtual_requests=1000
accepted=752
rejected=248
acceptance_rate=0.752
long_term_r2c_ratio=0.48211107669531345
```

## Runtime signal

| Runtime | Setup | System run | Run speedup |
|---|---:|---:|---:|
| Python oracle container | 78.756 ms | 24,024.558 ms | 1.000x |
| C++ GCC 11 container | 125.175 ms | 462.063 ms | 51.994x |
| C++ host clang-cl | 81.731 ms | 957.774 ms | 25.084x |

Timing is a single signal and includes normal machine-load variance. Exact
output is the gate; the measured speedup is supporting evidence only.

## Integration changes covered

- mutable node-rank solvers map directly on the Environment p-net;
- exact resource checkpoints replace graph clones and inverse-add rollback;
- successful committed solutions are not deployed twice;
- Recorder reuses the already-counted arrival fields;
- repeated worker batches use the persistent deterministic executor;
- OfflineSystem clones its immutable p-net snapshot once per run and rolls the
  same checkpoint back after every independent request.

Focused host CTest passed `5/5` for Controller, all node-rank variants, System,
transactional System, and Main config. GCC 11 Docker executions of
`vne_system_unit` and `vne_transactional_system_unit` also passed. Frozen
graph/CSV/YAML/config/random benchmarks and their artifacts were not run or
modified.
