#pragma once

#include "base_system.h"

namespace virne::system {

class OnlineSystem final : public BaseSystem {
public:
    using BaseSystem::BaseSystem;

    SystemRunResult run(
        RandomContext& random,
        const SystemRunConfig& config = {});
};

} // namespace virne::system
