#pragma once

#include "physical_network.h"
#include "virtual_network_request_simulator.h"

#include <cstddef>
#include <cstdint>
#include <optional>

class Config;
class RandomContext;

namespace virne::network {

// The original standalone Generator reads config.seed. Composed Virne
// settings keep the same value at experiment.seed. Callers select the
// boundary explicitly; no fallback search is performed.
enum class GeneratorSeedMode : std::uint8_t {
    compatibility_root_seed,
    composed_experiment_seed,
};

enum class GeneratorPersistence : std::uint8_t {
    memory_only,
    save,
};

struct GeneratorSelection {
    bool physical_network = true;
    bool virtual_networks = true;
    GeneratorPersistence persistence = GeneratorPersistence::memory_only;
};

struct GeneratorWorkers {
    std::size_t physical_factory_workers = 1U;
    std::size_t physical_attribute_workers = 1U;
    VirtualSimulationWorkers virtual_simulation;
};

struct GeneratedDataset {
    std::optional<PhysicalNetwork> physical_network;
    std::optional<VirtualNetworkRequestSimulator> virtual_networks;
};

class Generator final {
public:
    static GeneratedDataset generate_dataset(
        const Config& config,
        RandomContext& random,
        const GeneratorSelection& selection = {},
        const GeneratorWorkers& workers = {},
        GeneratorSeedMode seed_mode =
            GeneratorSeedMode::compatibility_root_seed);

    static PhysicalNetwork generate_p_net_dataset_from_config(
        const Config& config,
        RandomContext& random,
        GeneratorPersistence persistence = GeneratorPersistence::memory_only,
        const GeneratorWorkers& workers = {},
        GeneratorSeedMode seed_mode =
            GeneratorSeedMode::compatibility_root_seed);

    static VirtualNetworkRequestSimulator
    generate_v_nets_dataset_from_config(
        const Config& config,
        RandomContext& random,
        GeneratorPersistence persistence = GeneratorPersistence::memory_only,
        const GeneratorWorkers& workers = {},
        GeneratorSeedMode seed_mode =
            GeneratorSeedMode::compatibility_root_seed);

    static VirtualNetworkRequestSimulator
    generate_changeable_v_nets_dataset_from_config(
        const Config& config,
        RandomContext& random,
        GeneratorPersistence persistence = GeneratorPersistence::memory_only,
        const GeneratorWorkers& workers = {},
        GeneratorSeedMode seed_mode =
            GeneratorSeedMode::compatibility_root_seed);
};

}  // namespace virne::network
