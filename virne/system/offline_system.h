#pragma once

#include "base_system.h"

#include <functional>

namespace virne::system {

struct OfflineRunConfig {
    std::size_t num_simulations = 1U;
    bool capture_solutions = true;
    std::size_t counter_workers = 1U;
    SystemProgressSink* progress = nullptr;
};

// Offline requests are evaluated independently against one immutable
// physical-network snapshot. The mutable Solver is intentionally serialized:
// RNG consumption and solver state therefore remain identical for every
// caller worker spelling.
class OfflineSystem final {
public:
    OfflineSystem(
        const network::PhysicalNetwork& physical_network,
        const network::VirtualNetworkRequestSimulator& simulator,
        solver::Solver& solver);

    SystemRunResult run(const OfflineRunConfig& config = {});

    const network::PhysicalNetwork& physical_network() const noexcept;
    const network::VirtualNetworkRequestSimulator& simulator() const noexcept;
    solver::Solver& solver_instance() const noexcept;

private:
    std::reference_wrapper<const network::PhysicalNetwork> physical_network_;
    std::reference_wrapper<
        const network::VirtualNetworkRequestSimulator> simulator_;
    std::reference_wrapper<solver::Solver> solver_;
};

} // namespace virne::system
