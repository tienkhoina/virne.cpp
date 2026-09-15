#include "main_learning_config.h"

#include "config/config.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace virne::system {
namespace {
namespace rl = solver::learning::rl_core;

[[noreturn]] void invalid(const std::string& path, const std::string& reason) {
    throw std::invalid_argument(path + ": " + reason);
}

std::int64_t positive_integer(const Config& config, const std::string& path,
    const std::int64_t fallback) {
    const auto value = config.get<std::int64_t>(path, fallback);
    if (value <= 0) invalid(path, "must be positive");
    return value;
}

std::size_t count(const Config& config, const std::string& path,
    const std::int64_t fallback, const bool allow_zero = false) {
    const auto value = config.get<std::int64_t>(path, fallback);
    if (value < 0 || (!allow_zero && value == 0))
        invalid(path, allow_zero ? "must be non-negative" : "must be positive");
    if (static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max())
        invalid(path, "exceeds native size range");
    return static_cast<std::size_t>(value);
}

double real(const Config& config, const std::string& path, const double fallback,
    const double minimum = 0.0, const double maximum = std::numeric_limits<double>::max()) {
    const auto value = config.get<double>(path, fallback);
    if (!std::isfinite(value) || value < minimum || value > maximum)
        invalid(path, "must be finite and within the supported range");
    return value;
}

std::optional<double> optional_real(const Config& config, const std::string& path) {
    const auto node = config.get_raw(path);
    if (!node.IsDefined() || node.IsNull()) return std::nullopt;
    return real(config, path, 0.0);
}

rl::InstanceEnvironmentOptions::Reward reward(const std::string& name) {
    using Reward = rl::InstanceEnvironmentOptions::Reward;
    if (name == "gradual_intermediate") return Reward::gradual;
    if (name == "adaptive_intermediate") return Reward::adaptive;
    if (name == "fixed_intermediate") return Reward::fixed;
    if (name == "vanilla") return Reward::vanilla;
    invalid("rl.reward_calculator.name", "unsupported reward calculator '" + name + "'");
}

core::controller::ShortestPathMethod shortest(const std::string& name) {
    using Method = core::controller::ShortestPathMethod;
    if (name == "first_shortest") return Method::first_shortest;
    if (name == "k_shortest") return Method::k_shortest;
    if (name == "k_shortest_length") return Method::k_shortest_length;
    if (name == "all_shortest") return Method::all_shortest;
    if (name == "bfs_shortest") return Method::bfs_shortest;
    if (name == "available_shortest") return Method::available_shortest;
    invalid("solver.shortest_method", "unsupported shortest path method '" + name + "'");
}

void require_disabled(const Config& config, const std::string& path, const char* reason) {
    if (config.get<bool>(path, false)) invalid(path, reason);
}
} // namespace

MainLearningConfig main_learning_config_from_hydra(const Config& config) {
    MainLearningConfig result;
    auto& session = result.solver_options.session;
    auto& training = session.training;
    auto& environment = result.solver_options.environment;

    // The vendored native runtime is CPU-only, matching Python's fallback
    // behavior when CUDA is unavailable. Keep the source flag accepted so a
    // normal Hydra config remains runnable without a redundant override.
    require_disabled(config, "training.distributed_training", "distributed main-runtime training is not implemented");
    require_disabled(config, "rl.if_use_baseline_solver", "baseline-filtered experience is not implemented in the main learning runtime");
    require_disabled(config, "rl.if_allow_baseline_unsafe_solve", "unsafe baseline solving is not implemented in the main learning runtime");
    require_disabled(config, "solver.allow_rejection", "registered learning policies require physical-node actions");
    require_disabled(config, "solver.allow_revocable", "registered learning policies require physical-node actions");

    const auto strategy = config.get<std::string>("inference.decode_strategy", "greedy");
    if (strategy == "greedy") result.inference_sample = false;
    else if (strategy == "sample" || strategy == "sampling") result.inference_sample = true;
    else invalid("inference.decode_strategy", "main learning supports greedy or sample");
    if (positive_integer(config, "inference.k_searching", 1) != 1)
        invalid("inference.k_searching", "main learning supports one rollout; set inference.k_searching=1");

    result.num_train_epochs = count(config, "training.num_train_epochs", 50, true);
    result.random_training_seed = config.get<bool>("training.if_use_random_training_seed", true);
    result.save_interval = count(config, "training.save_interval", 10);
    result.eval_interval = count(config, "training.eval_interval", 10);
    result.log_interval = count(config, "training.log_interval", 1);
    result.pretrained_model_path = config.get<std::string>("solver.pretrained_model_path", "");
    result.model_dir_name = config.get<std::string>("training.model_dir_name", "models");
    if (result.model_dir_name.empty()) invalid("training.model_dir_name", "must not be empty");

    // BaseSystem does not forward training.num_workers, training.seed, nn layer
    // counts or the PPO constructor kwargs. Match those effective source
    // defaults; native.learning is the explicit extension for changing them.
    const auto seed = config.get<std::int64_t>("experiment.seed", 0);
    if (seed < 0 || static_cast<std::uint64_t>(seed) > std::numeric_limits<std::uint32_t>::max())
        invalid("experiment.seed", "must fit an unsigned 32-bit seed");
    session.seed = static_cast<std::uint64_t>(seed);
    session.embedding_dim = positive_integer(config, "nn.embedding_dim", 128);
    session.num_layers = positive_integer(config, "native.learning.num_layers", 3);
    session.dropout = real(config, "nn.dropout_prob", 0.0, 0.0, 1.0);
    session.batch_norm = config.get<bool>("nn.batch_norm", false);
    session.target_steps = positive_integer(config, "rl.target_steps", 256);
    session.include_failed = config.get<bool>("rl.if_use_negative_sample", true);
    session.weight_decay = real(config, "rl.weight_decay", 1e-5);
    session.gamma = real(config, "rl.gamma", 0.99, 0.0, 1.0);
    session.gae_lambda = real(config, "native.learning.gae_lambda", 0.98, 0.0, 1.0);
    if (config.contains("native.learning.intraop_threads")) {
        const auto threads = positive_integer(config, "native.learning.intraop_threads", 1);
        if (threads > std::numeric_limits<int>::max()) invalid("native.learning.intraop_threads", "exceeds native integer range");
        session.intraop_threads = static_cast<int>(threads);
    }
    if (config.contains("native.learning.max_episode_steps"))
        session.max_episode_steps = count(config, "native.learning.max_episode_steps", 1);

    session.learning_rates.actor = real(config, "rl.learning_rate.actor", 1e-3);
    session.learning_rates.critic = real(config, "rl.learning_rate.critic", 1e-3);
    session.learning_rates.encoder = optional_real(config, "rl.learning_rate.encoder");
    session.learning_rates.cost_critic = optional_real(config, "rl.learning_rate.cost_critic");
    session.learning_rates.lambda_net = optional_real(config, "rl.learning_rate.lambda_net");
    training.batch_size = positive_integer(config, "training.batch_size", 128);
    training.repeat_times = real(config, "native.learning.repeat_times", 10.0);
    training.eps_clip = real(config, "native.learning.eps_clip", 0.2);
    training.normalize_returns = config.get<bool>("rl.norm_reward", false);
    training.normalize_advantage = config.get<bool>("native.learning.norm_advantage", true);
    training.clip_grad = config.get<bool>("native.learning.clip_grad", true);
    training.max_grad_norm = real(config, "native.learning.max_grad_norm", 1.0);
    training.critic_coefficient = real(config, "rl.coef_critic_loss", 0.5);
    training.entropy_coefficient = real(config, "rl.coef_entropy_loss", 0.01);
    training.mask_coefficient = real(config, "rl.coef_mask_loss", 0.01);
    training.target_kl = optional_real(config, "rl.target_kl");

    // Until the session exposes both original masking switches, reject changes
    // to them rather than silently executing a different policy distribution.
    if (!config.get<bool>("rl.mask_actions", true) || !config.get<bool>("rl.maskable_policy", true))
        invalid("rl.mask_actions/rl.maskable_policy", "this runtime requires both action-masking switches enabled");
    if (real(config, "native.learning.softmax_temp", 1.0) != 1.0)
        invalid("native.learning.softmax_temp", "the learning session currently requires temperature 1");

    const auto extracted = config.get<std::vector<std::string>>(
        "rl.feature_constructor.extracted_attr_types", {"resource"});
    if (extracted != std::vector<std::string>{"resource"})
        invalid("rl.feature_constructor.extracted_attr_types", "only resource features are implemented by the instance environment");
    environment.node_status = config.get<bool>("rl.feature_constructor.if_use_node_status_flags", true);
    environment.aggregated_links = config.get<bool>("rl.feature_constructor.if_use_aggregated_link_attrs", true);
    environment.degree = config.get<bool>("rl.feature_constructor.if_use_degree_metric", true);
    environment.other_metrics = config.get<bool>("rl.feature_constructor.if_use_more_topological_metrics", true);
    environment.reward = reward(config.get<std::string>("rl.reward_calculator.name", "fixed_intermediate"));
    environment.intermediate_reward = config.get<double>("rl.reward_calculator.intermediate_reward", 0.1);
    if (!std::isfinite(environment.intermediate_reward)
        || (environment.intermediate_reward < 0 && environment.intermediate_reward != -1.0))
        invalid("rl.reward_calculator.intermediate_reward", "must be non-negative or -1");
    environment.reusable = config.get<bool>("solver.reusable", false);
    environment.shortest_method = shortest(config.get<std::string>("solver.shortest_method", "k_shortest"));
    environment.k_shortest = positive_integer(config, "solver.k_shortest", 10);
    environment.workers = count(config, "native.workers.learning_environment", 1);

    result.solver_options = solver::learning::resolve_learning_solver_options(
        config.get<std::string>("solver.solver_name"), std::move(result.solver_options));
    result.solver_options.inference_sample = result.inference_sample;
    return result;
}

} // namespace virne::system
