# Component API: `network.dataset_generator`

State: **COMPLETE / FROZEN** on 2026-07-29.

This is the final non-ML orchestration leaf. It reuses the completed Config,
Random, dataset-path, PhysicalNetwork, and VirtualNetworkRequestSimulator
components; it introduces no parser, RNG, path formatter, Torch dependency, or
worker auto-tuning.

## Stable API

All native names are in `virne::network`:

```cpp
enum class GeneratorSeedMode {
    compatibility_root_seed, composed_experiment_seed,
};
enum class GeneratorPersistence { memory_only, save };

struct GeneratorSelection;
struct GeneratorWorkers;
struct GeneratedDataset;

class Generator {
public:
    static GeneratedDataset generate_dataset(
        const Config&, RandomContext&, const GeneratorSelection&,
        const GeneratorWorkers&, GeneratorSeedMode);
    static PhysicalNetwork generate_p_net_dataset_from_config(...);
    static VirtualNetworkRequestSimulator
        generate_v_nets_dataset_from_config(...);
    static VirtualNetworkRequestSimulator
        generate_changeable_v_nets_dataset_from_config(...);
};
```

`GeneratedDataset` owns optional move-only physical and virtual results.
`GeneratorWorkers` carries direct physical factory/attribute fields and the
completed `VirtualSimulationWorkers`; values are caller configuration.

## Config, RNG, and changeable workload

- A selected config subtree crosses `Config::get_raw` once, is resolved once,
  and is converted through the completed YAML/Setting adapter. Generation and
  hot loops then use typed fields, enums, numeric IDs, and direct slots.
- `compatibility_root_seed` reads `seed`; `composed_experiment_seed` reads
  `experiment.seed`. There is no fallback search. Missing/null seed preserves
  stream continuation. Physical generation precedes virtual generation, and a
  present seed resets each selected leaf exactly as Python does.
- The changeable variant constructs four typed stage configs, generates every
  request in each stage in canonical RNG order, and moves only the selected
  quarter via `release_v_nets() &&`. Events are renewed once after merging.
  Its save directory follows the final size-scaled stage while the persisted
  simulator setting remains the original setting.
- Shared Python/NumPy streams and stochastic request order remain sequential.
  Only completed deterministic inner worker paths receive caller widths;
  Torch/CUDA and all ML behavior remain outside this component.

## Frozen acceptance

- Differential: **PASS**, 15 shared Python/C++ cases, zero native-only cases,
  and four recorded boundaries. Selection, root/composed seed modes, absent
  seed continuation, workers `0/1/2/8`, normal/changeable generation, missing
  subtrees, and invalid stage cardinality are covered.
- Compact benchmark: **PASS**, 512 requests, workers `1/2/8`. Ordinary
  speedups are `1.840x / 1.941x / 2.004x`; changeable speedups are
  `2.778x / 2.754x / 2.501x`.
- Full results and provenance are in
  [`dataset_generator_2026-07-29.md`](../results/dataset_generator_2026-07-29.md).
  The accepted benchmark is frozen and must not be rerun or updated.
