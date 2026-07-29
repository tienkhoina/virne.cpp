# Dataset Generator result - 2026-07-29

Status: **COMPLETE / FROZEN**. Exact differential output was accepted before
the compact timing run.

## Differential

**PASS**: 15 shared cases, zero native-only cases, and four explicit
boundaries: native seed width, persistence/filesystem effects, the deferred
Torch seed boundary, and typed Config in place of arbitrary reflection.
`torch_seed_calls` is zero.

## Frozen compact benchmark

Protocol: 512 requests; one warm-up and three repetitions; process startup,
fixture creation, fingerprinting, and output gating excluded from timing.
Workers are caller configured and never auto-tuned.

| Workload | Workers | Python median | C++ median | Speedup |
|---|---:|---:|---:|---:|
| ordinary | 1 | 27.879044 ms | 15.149291 ms | 1.840x |
| ordinary | 2 | 27.879044 ms | 14.365843 ms | 1.941x |
| ordinary | 8 | 27.879044 ms | 13.911506 ms | 2.004x |
| changeable | 1 | 144.576812 ms | 52.036351 ms | 2.778x |
| changeable | 2 | 144.576812 ms | 52.492845 ms | 2.754x |
| changeable | 8 | 144.576812 ms | 57.798831 ms | 2.501x |

Ordinary rows share checksum `12807557851346020205`; changeable rows share
checksum `16807312341349659821`. The accepted benchmark is provenance only and
must not be rerun or updated.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Python `dataset_generator.py` | `43D5DBE625FCD15F273067700B3C9D0B69CF931E064F6542C65802B0A4BA4E5C` |
| differential JSON | `13C6611CAE472A0093B580366CC7B84D4B78156B000266EDBAB060F44C571F3D` |
| benchmark JSON | `E9191074D2A5DC25EA4011D778DAECD02E1CABFDBA9A4EB93233128E14445285` |
