# VirtualNetworkRequestSimulator result - 2026-07-29

Status: **COMPLETE / FROZEN**.

## API

The simulator decodes raw settings once into direct typed config fields,
reuses the completed RNG/topology/attribute/network/setting/GML APIs, and uses
numeric event indexes in hot paths. Caller-configured workers cover typed
arrangement, attribute/event work, and deterministic I/O. Persistence retains
the deep source setting and exact Python layout/cache quirks.

`VirtualSimulationOutput` includes cold optional `save_dir` plus event/setting
filenames. `release_v_nets() && noexcept` transfers generated requests to the
Generator without copying. `save_dir` was added after benchmark acceptance;
the measured arrangement and scheduling hot paths were untouched.

## Evidence

- Core differential: **PASS**, 18 shared + 2 native + 4 boundaries = 24
  classified cases.
- I/O/cache differential: **PASS**, 16 shared + 0 native + 3 boundaries =
  artifact `case_count` 19.
- Exact state/output, RNG continuation, workers `0/1/2/8`, GML/YAML semantics,
  layout/error order, event cardinality, and deep `seed_`/non-`seed_` cache
  behavior are gated.

## Frozen benchmark

The accepted workload has 65,536 requests, one warm-up, and three samples.
Fixture creation, process startup, and fingerprints are excluded; exact
entry-count, output-byte, and checksum gates pass before timing.

| Hot path | Workers | Python median | C++ median | Speedup |
|---|---:|---:|---:|---:|
| Arrangement | 1 | 6.993699 ms | 2.491562 ms | 2.807x |
| Arrangement | 2 | 6.993699 ms | 3.338535 ms | 2.095x |
| Arrangement | 8 | 6.993699 ms | 3.520900 ms | 1.986x |
| Event schedule | 1 | 346.722896 ms | 15.298536 ms | 22.664x |
| Event schedule | 2 | 346.722896 ms | 13.215411 ms | 26.236x |
| Event schedule | 8 | 346.722896 ms | 16.721802 ms | 20.735x |

This benchmark is frozen forever and must not be rerun or updated.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Core differential JSON | `B4EDFAF749547A3B14C46BBABFCD36429CED49A39586755CE3C6E71054AA923B` |
| I/O differential JSON | `459B819994BE547461D8A9963B1BF2FA5FEDD7745E35C335C902CFF10CE2313B` |
| Benchmark JSON | `BBAB8200818F8B3B5FA877107513CD8C5D353E276A1C2A9D1AB17823F962BCB1` |
