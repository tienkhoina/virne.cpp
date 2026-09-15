#pragma once

#include "../solver/learning/solver_registry.h"

#include <cstddef>
#include <filesystem>
#include <string>

class Config;

namespace virne::system {

// Decode only after the main registry has identified a learning solver. All
// fields are graph-ABI safe; this header does not expose a Torch dependency.
struct MainLearningConfig {
    solver::learning::LearningSolverOptions solver_options;
    std::size_t num_train_epochs = 0U;
    std::filesystem::path pretrained_model_path;
    std::string model_dir_name = "models";
    bool random_training_seed = true;
    std::size_t save_interval = 10U;
    std::size_t eval_interval = 10U;
    std::size_t log_interval = 1U;
    bool inference_sample = false;
};

MainLearningConfig main_learning_config_from_hydra(const Config& config);

} // namespace virne::system
