#include "learning_session.h"
#include "policy_builder.h"
#include "policy_observation.h"
#include "rollout_returns.h"
#include "training_update.h"
#include "dqn_update.h"
#include "dqn_action.h"
#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace virne::solver::learning::rl_core {
namespace p = rl_policy;
namespace {
using Output = torch::serialize::OutputArchive;
using Input = torch::serialize::InputArchive;
torch::Tensor read_tensor(Input& archive, const std::string& name) {
    torch::Tensor value; archive.read(name, value, true); return value;
}
void write_optional(Output& archive, const std::string& name, const torch::Tensor& value) {
    archive.write(name + "_present", torch::tensor(static_cast<std::int64_t>(value.defined())), true);
    if (value.defined()) archive.write(name, value.detach(), true);
}
torch::Tensor read_optional(Input& archive, const std::string& name) {
    return read_tensor(archive, name + "_present").item<std::int64_t>() ? read_tensor(archive, name) : torch::Tensor{};
}
template <class T> std::vector<T> tensor_vector(const torch::Tensor& value, const torch::ScalarType dtype) {
    if (value.dim() != 1 || value.scalar_type() != dtype || !value.is_cpu()) throw std::invalid_argument("invalid checkpoint vector");
    const auto data = value.contiguous();
    if (!data.numel()) return {};
    return {data.data_ptr<T>(), data.data_ptr<T>() + data.numel()};
}

constexpr char checkpoint_magic[] = "VIRNE_NATIVE_CHECKPOINT_2";

void append_u64(std::vector<std::uint8_t>& output, const std::uint64_t value) {
    for (unsigned shift = 0; shift != 64U; shift += 8U)
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
}

void append_blob(std::vector<std::uint8_t>& output, const std::vector<char>& blob) {
    append_u64(output, static_cast<std::uint64_t>(blob.size()));
    output.insert(output.end(), blob.begin(), blob.end());
}

void append_tensor(std::vector<std::uint8_t>& output, const torch::Tensor& value) {
    if (!value.defined()) {
        append_u64(output, std::numeric_limits<std::uint64_t>::max());
        return;
    }
    append_blob(output, torch::pickle_save(torch::IValue(value.detach().cpu())));
}

void append_observation(std::vector<std::uint8_t>& output,
    const p::PolicyObservation& observation) {
    const std::vector<torch::Tensor> fields{
        observation.p_net_x, observation.v_node_x, observation.v_net_x,
        observation.current_virtual_node, observation.virtual_net_size,
        observation.hidden_state, observation.p_node_id,
        observation.encoder_outputs,
        observation.recurrent_mask.value_or(torch::Tensor{}),
        observation.action_mask.value_or(torch::Tensor{}), observation.p_net.x,
        observation.p_net.edge_index,
        observation.p_net.edge_attr.value_or(torch::Tensor{}),
        observation.v_net.x, observation.v_net.edge_index,
        observation.v_net.edge_attr.value_or(torch::Tensor{})};
    for (const auto& field : fields) append_tensor(output, field);
}

class CheckpointReader final {
public:
    explicit CheckpointReader(const std::vector<std::uint8_t>& bytes)
        : bytes_(bytes) {}

    std::uint64_t u64() {
        if (offset_ + 8U > bytes_.size()) throw std::invalid_argument("truncated learning checkpoint");
        std::uint64_t result = 0U;
        for (unsigned shift = 0; shift != 64U; shift += 8U)
            result |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
        return result;
    }

    torch::Tensor tensor() {
        const auto size = u64();
        if (size == std::numeric_limits<std::uint64_t>::max()) return torch::Tensor{};
        if (size > bytes_.size() - offset_ || size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            throw std::invalid_argument("invalid learning checkpoint tensor size");
        std::vector<char> blob(size);
        if (size != 0U) std::memcpy(blob.data(), bytes_.data() + offset_, static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
        const auto value = torch::pickle_load(blob);
        if (!value.isTensor()) throw std::invalid_argument("learning checkpoint blob is not a tensor");
        return value.toTensor();
    }

    void require_end() const {
        if (offset_ != bytes_.size()) throw std::invalid_argument("learning checkpoint has trailing data");
    }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0U;
};
void write_observation(Output& archive, const std::string& root, const p::PolicyObservation& o) {
    const std::vector<std::pair<const char*, torch::Tensor>> fields{
        {"physical_x", o.p_net_x}, {"vnode_x", o.v_node_x}, {"virtual_x", o.v_net_x},
        {"current", o.current_virtual_node}, {"size", o.virtual_net_size}, {"hidden", o.hidden_state},
        {"node_id", o.p_node_id}, {"encoded", o.encoder_outputs},
        {"recurrent_mask", o.recurrent_mask.value_or(torch::Tensor{})}, {"action_mask", o.action_mask.value_or(torch::Tensor{})},
        {"p_x", o.p_net.x}, {"p_edges", o.p_net.edge_index}, {"p_attr", o.p_net.edge_attr.value_or(torch::Tensor{})},
        {"v_x", o.v_net.x}, {"v_edges", o.v_net.edge_index}, {"v_attr", o.v_net.edge_attr.value_or(torch::Tensor{})}};
    for (const auto& field : fields) write_optional(archive, root + field.first, field.second);
}
p::PolicyObservation read_observation(Input& archive, const std::string& root) {
    p::PolicyObservation o;
    o.p_net_x = read_optional(archive, root + "physical_x"); o.v_node_x = read_optional(archive, root + "vnode_x");
    o.v_net_x = read_optional(archive, root + "virtual_x"); o.current_virtual_node = read_optional(archive, root + "current");
    o.virtual_net_size = read_optional(archive, root + "size"); o.hidden_state = read_optional(archive, root + "hidden");
    o.p_node_id = read_optional(archive, root + "node_id"); o.encoder_outputs = read_optional(archive, root + "encoded");
    const auto recurrent_mask = read_optional(archive, root + "recurrent_mask"), mask = read_optional(archive, root + "action_mask");
    if (recurrent_mask.defined()) o.recurrent_mask = recurrent_mask;
    if (mask.defined()) o.action_mask = mask;
    const auto graph = [&](const std::string& graph_prefix) {
        pyg_lite::Data data;
        data.x = read_optional(archive, root + graph_prefix + "_x"); data.edge_index = read_optional(archive, root + graph_prefix + "_edges");
        const auto attributes = read_optional(archive, root + graph_prefix + "_attr");
        if (attributes.defined()) data.edge_attr = attributes;
        return data.x.defined() ? pyg_lite::Batch::from_data_list({data}) : pyg_lite::Batch{};
    };
    o.p_net = graph("p"); o.v_net = graph("v"); return o;
}

p::PolicyObservation read_portable_observation(CheckpointReader& reader) {
    p::PolicyObservation o;
    o.p_net_x = reader.tensor(); o.v_node_x = reader.tensor();
    o.v_net_x = reader.tensor(); o.current_virtual_node = reader.tensor();
    o.virtual_net_size = reader.tensor(); o.hidden_state = reader.tensor();
    o.p_node_id = reader.tensor(); o.encoder_outputs = reader.tensor();
    const auto recurrent_mask = reader.tensor();
    const auto mask = reader.tensor();
    if (recurrent_mask.defined()) o.recurrent_mask = recurrent_mask;
    if (mask.defined()) o.action_mask = mask;
    const auto graph = [&reader](const bool physical) {
        pyg_lite::Data data;
        data.x = reader.tensor(); data.edge_index = reader.tensor();
        const auto attributes = reader.tensor();
        if (attributes.defined()) data.edge_attr = attributes;
        static_cast<void>(physical);
        return data.x.defined() ? pyg_lite::Batch::from_data_list({data}) : pyg_lite::Batch{};
    };
    o.p_net = graph(true); o.v_net = graph(false);
    return o;
}
}

class LearningSession::Impl {
public:
    LearningSessionOptions options;
    std::unique_ptr<p::Policy> policy;
    std::unique_ptr<Adam> optimizer;
    std::unique_ptr<PolicyGradientTrainer> trainer;
    std::unique_ptr<p::Policy> target_policy;
    std::unique_ptr<DqnTrainer> q_trainer;
    std::unique_ptr<DqnActionSelector> q_selector;
    std::optional<std::uint32_t> pending_exploration_seed;
    std::vector<p::PolicyObservation> observations, next_observations;
    std::vector<std::int64_t> actions;
    std::vector<std::uint8_t> dones;
    std::vector<double> rewards, returns, log_probabilities;
    std::optional<double> last_actor_loss, last_critic_loss, last_entropy_loss,
        last_approximate_kl, last_gradient_norm, last_log_probability_mean,
        last_old_log_probability_mean, last_value_mean, last_return_mean,
        last_advantage_mean, last_reward_mean;
    std::int64_t last_applied_update_steps = 0;

    explicit Impl(LearningSessionOptions value) : options(std::move(value)) {
        if (options.target_steps <= 0 || options.num_layers <= 0) throw std::invalid_argument("invalid learning-session limits");
        if (options.intraop_threads) {
            if (*options.intraop_threads <= 0) throw std::invalid_argument("invalid Torch thread count");
            torch::set_num_threads(*options.intraop_threads);
        }
        torch::manual_seed(options.seed);
    }
    bool q_learning() const noexcept {
        return options.training.algorithm == UpdateAlgorithm::dqn || options.training.algorithm == UpdateAlgorithm::double_dqn;
    }
    std::int64_t update_time() const noexcept {
        return q_trainer ? q_trainer->update_time() : (trainer ? trainer->update_time() : 0);
    }
    void initialize(const InstanceObservation& observation) {
        PolicyConfiguration config;
        auto& dims = config.feature_dimensions;
        dims.p_net_num_nodes = static_cast<std::int64_t>(observation.features.p_net.x.rows);
        dims.p_net_x_dim = static_cast<std::int64_t>(observation.features.p_net.x.columns);
        dims.p_net_edge_dim = static_cast<std::int64_t>(observation.features.p_net.edge_attr.columns);
        if (observation.features.v_node_x) dims.v_node_x_dim = static_cast<std::int64_t>(observation.features.v_node_x->values.size());
        if (observation.features.v_net) {
            dims.v_net_x_dim = static_cast<std::int64_t>(observation.features.v_net->x.columns);
            dims.v_net_edge_dim = static_cast<std::int64_t>(observation.features.v_net->edge_attr.columns);
        }
        config.neural_network = {options.embedding_dim, options.dropout, options.batch_norm};
        auto constructor = policy_options_from_configuration(options.policy, config);
        constructor.num_layers = options.num_layers;
        initialize(constructor);
    }
    void initialize(const p::PolicyOptions& constructor) {
        if (q_learning() && p::policy_descriptor(options.policy).observation == p::ObservationKind::recurrent)
            throw std::invalid_argument("recurrent DQN episode integration is not implemented");
        auto created_policy = std::make_unique<p::Policy>(constructor);
        auto created_optimizer = build_policy_optimizer(*created_policy, options.learning_rates, options.weight_decay);
        if (q_learning()) {
            auto generator = at::detail::getDefaultCPUGenerator(); const auto rng = generator.get_state();
            auto created_target = std::make_unique<p::Policy>(constructor); generator.set_state(rng);
            auto created_trainer = std::make_unique<DqnTrainer>(*created_optimizer, created_policy->module(), created_target->module(),
                [this](const torch::Tensor& indices, const bool next) {
                    return policy->act(gather_policy_observations(next ? next_observations : observations, indices));
                },
                [this](const torch::Tensor& indices, const bool next) {
                    return target_policy->act(gather_policy_observations(next ? next_observations : observations, indices));
                }, DqnOptions{options.training.batch_size, options.dqn.target_update_interval, options.gamma,
                    options.training.max_grad_norm, options.training.algorithm == UpdateAlgorithm::double_dqn});
            auto created_selector = std::make_unique<DqnActionSelector>(DqnExplorationOptions{options.dqn.epsilon_start, options.dqn.epsilon_end,
                options.dqn.epsilon_decay, constructor.physical_nodes, options.dqn.mask_actions, options.dqn.maskable_policy,
                pending_exploration_seed.value_or(options.dqn.seed)});
            policy = std::move(created_policy); optimizer = std::move(created_optimizer);
            target_policy = std::move(created_target); q_trainer = std::move(created_trainer); q_selector = std::move(created_selector);
            return;
        }
        auto created_trainer = std::make_unique<PolicyGradientTrainer>(*created_optimizer,
            [this](const torch::Tensor& indices, const torch::Tensor& chosen_actions) {
                const auto batch = gather_policy_observations(observations, indices);
                const auto logits = policy->act(batch);
                const auto values = policy->evaluate(batch);
                return evaluate_policy_actions(logits, chosen_actions, values, batch.action_mask, std::nullopt, true);
            }, options.training);
        policy = std::move(created_policy); optimizer = std::move(created_optimizer); trainer = std::move(created_trainer);
    }
    torch::Tensor configuration_integers() const {
        const auto& t = options.training;
        std::vector<std::int64_t> values{1, static_cast<std::int64_t>(options.policy),
            static_cast<std::int64_t>(t.algorithm), options.embedding_dim, options.num_layers, options.target_steps,
            t.batch_size, t.normalize_returns, t.normalize_advantage, t.clip_grad, options.batch_norm,
            options.include_failed, static_cast<std::int64_t>(options.advantage),
            options.max_episode_steps.has_value(), static_cast<std::int64_t>(options.max_episode_steps.value_or(0)),
            options.intraop_threads.value_or(torch::get_num_threads()), t.target_kl.has_value(),
            options.learning_rates.actor.has_value(), options.learning_rates.critic.has_value(),
            options.learning_rates.encoder.has_value(), options.learning_rates.cost_critic.has_value(),
            options.learning_rates.lambda_net.has_value()};
        if (q_learning()) values.insert(values.end(), {options.dqn.target_update_interval, options.dqn.seed,
            options.dqn.mask_actions, options.dqn.maskable_policy});
        return torch::tensor(values, torch::kInt64);
    }
    torch::Tensor configuration_reals() const {
        const auto& t = options.training; const auto& r = options.learning_rates;
        std::vector<double> values{options.dropout, options.weight_decay, options.gamma, options.gae_lambda,
            t.repeat_times, t.eps_clip, t.max_grad_norm, t.critic_coefficient, t.entropy_coefficient, t.mask_coefficient,
            t.target_kl.value_or(0), r.actor.value_or(0), r.critic.value_or(0), r.encoder.value_or(0),
            r.cost_critic.value_or(0), r.lambda_net.value_or(0)};
        if (q_learning()) values.insert(values.end(), {options.dqn.epsilon_start, options.dqn.epsilon_end, options.dqn.epsilon_decay});
        return torch::tensor(values, torch::kFloat64);
    }
    std::vector<std::uint8_t> portable_checkpoint() const {
        if (!policy) throw std::logic_error("cannot checkpoint an uninitialized learning session");
        torch::NoGradGuard no_grad;
        std::vector<std::uint8_t> result(checkpoint_magic, checkpoint_magic + sizeof(checkpoint_magic) - 1U);
        append_tensor(result, configuration_integers());
        append_tensor(result, configuration_reals());
        const auto& policy_options = policy->options();
        append_tensor(result, torch::tensor(std::vector<std::int64_t>{policy_options.physical_nodes,
            policy_options.physical_features, policy_options.virtual_node_features,
            policy_options.virtual_features, policy_options.physical_edge_features.value_or(-1),
            policy_options.virtual_edge_features.value_or(-1)}, torch::kInt64));
        const auto parameters = unique_parameters(policy->module());
        append_u64(result, parameters.size());
        for (const auto& parameter : parameters) append_tensor(result, parameter);
        const auto buffers = policy->module().named_buffers(true);
        append_u64(result, buffers.size());
        for (const auto& buffer : buffers) append_tensor(result, buffer.value());
        append_tensor(result, torch::tensor(static_cast<std::int64_t>(policy->module().is_training())));
        append_tensor(result, torch::tensor(update_time()));
        append_tensor(result, at::detail::getDefaultCPUGenerator().get_state());
        const auto state = optimizer->snapshot();
        append_u64(result, 1U);
        append_u64(result, state.moments.size());
        for (const auto& moment : state.moments) {
            append_tensor(result, moment.step); append_tensor(result, moment.exp_avg);
            append_tensor(result, moment.exp_avg_sq); append_tensor(result, moment.max_exp_avg_sq);
        }
        std::vector<double> groups;
        for (const auto& group : state.options) groups.insert(groups.end(), {group.lr, group.beta1,
            group.beta2, group.eps, group.weight_decay, static_cast<double>(group.amsgrad),
            static_cast<double>(group.maximize)});
        append_tensor(result, torch::tensor(groups, torch::kFloat64));
        append_tensor(result, torch::tensor(actions, torch::kInt64));
        append_tensor(result, torch::tensor(rewards, torch::kFloat64));
        append_tensor(result, torch::tensor(returns, torch::kFloat64));
        append_tensor(result, torch::tensor(log_probabilities, torch::kFloat64));
        append_u64(result, observations.size());
        for (const auto& observation : observations)
            append_observation(result, observation);
        append_u64(result, q_learning() ? 1U : 0U);
        if (q_learning()) {
            const auto target_parameters = unique_parameters(target_policy->module());
            append_u64(result, target_parameters.size());
            for (const auto& parameter : target_parameters) append_tensor(result, parameter);
            const auto target_buffers = target_policy->module().named_buffers(true);
            append_u64(result, target_buffers.size());
            for (const auto& buffer : target_buffers) append_tensor(result, buffer.value());
            append_tensor(result, torch::tensor(static_cast<std::int64_t>(target_policy->module().is_training())));
            append_tensor(result, torch::tensor(dones, torch::kUInt8));
            append_u64(result, next_observations.size());
            for (const auto& observation : next_observations)
                append_observation(result, observation);
            const auto exploration = q_selector->snapshot();
            append_tensor(result, torch::tensor(std::vector<std::int64_t>(
                exploration.random.key.begin(), exploration.random.key.end()), torch::kInt64));
            append_tensor(result, torch::tensor(std::vector<std::int64_t>{
                exploration.steps_done, exploration.random.position,
                exploration.random.has_gaussian}, torch::kInt64));
            append_tensor(result, torch::tensor(std::vector<double>{
                exploration.epsilon, exploration.random.gaussian}, torch::kFloat64));
        }
        return result;
    }
    void restore_portable(const std::vector<std::uint8_t>& bytes) {
        const std::size_t magic_size = sizeof(checkpoint_magic) - 1U;
        if (bytes.size() < magic_size || !std::equal(checkpoint_magic,
                checkpoint_magic + magic_size, bytes.begin()))
            throw std::invalid_argument("invalid native learning checkpoint");
        std::vector<std::uint8_t> body(bytes.begin() + static_cast<std::ptrdiff_t>(magic_size), bytes.end());
        CheckpointReader reader(body);
        if (!torch::equal(reader.tensor(), configuration_integers()) ||
            !torch::equal(reader.tensor(), configuration_reals()))
            throw std::invalid_argument("checkpoint learning configuration mismatch");
        const auto dims = tensor_vector<std::int64_t>(reader.tensor(), torch::kInt64);
        if (dims.size() != 6U) throw std::invalid_argument("invalid checkpoint dimensions");
        p::PolicyOptions policy_options;
        policy_options.kind = options.policy; policy_options.physical_nodes = dims[0];
        policy_options.physical_features = dims[1]; policy_options.virtual_node_features = dims[2];
        policy_options.virtual_features = dims[3];
        if (dims[4] >= 0) policy_options.physical_edge_features = dims[4];
        if (dims[5] >= 0) policy_options.virtual_edge_features = dims[5];
        policy_options.embedding_dim = options.embedding_dim; policy_options.num_layers = options.num_layers;
        policy_options.dropout = options.dropout; policy_options.batch_norm = options.batch_norm;
        initialize(policy_options);
        torch::NoGradGuard no_grad;
        const auto parameters = unique_parameters(policy->module());
        const auto parameter_count = reader.u64();
        if (parameter_count != parameters.size()) throw std::invalid_argument("checkpoint policy parameter count mismatch");
        for (const auto& parameter : parameters) {
            const auto source = reader.tensor();
            if (source.sizes() != parameter.sizes()) throw std::invalid_argument("checkpoint policy parameter shape mismatch");
            parameter.copy_(source.to(parameter.options()));
        }
        const auto buffers = policy->module().named_buffers(true);
        if (reader.u64() != buffers.size()) throw std::invalid_argument("checkpoint policy buffer count mismatch");
        for (const auto& buffer : buffers) {
            const auto source = reader.tensor();
            if (source.sizes() != buffer.value().sizes()) throw std::invalid_argument("checkpoint policy buffer shape mismatch");
            buffer.value().copy_(source.to(buffer.value().options()));
        }
        policy->train(reader.tensor().item<std::int64_t>() != 0);
        const auto restored_update_time = reader.tensor().item<std::int64_t>();
        const auto restored_rng = reader.tensor();
        const bool has_optimizer = reader.u64() != 0U;
        auto state = optimizer->snapshot();
        if (has_optimizer) {
            try {
                if (reader.u64() != state.moments.size()) throw std::invalid_argument("checkpoint optimizer parameter count mismatch");
                for (auto& moment : state.moments) {
                    moment.step = reader.tensor(); moment.exp_avg = reader.tensor();
                    moment.exp_avg_sq = reader.tensor(); moment.max_exp_avg_sq = reader.tensor();
                }
                const auto groups = tensor_vector<double>(reader.tensor(), torch::kFloat64);
                if (groups.size() != state.options.size() * 7U) throw std::invalid_argument("checkpoint optimizer group count mismatch");
                for (std::size_t i = 0; i < state.options.size(); ++i) state.options[i] = {groups[i*7], groups[i*7+1], groups[i*7+2],
                    groups[i*7+3], groups[i*7+4], groups[i*7+5] != 0, groups[i*7+6] != 0};
                optimizer->restore(state);
            } catch (const std::exception& error) {
                std::clog << "warning: learning checkpoint optimizer ignored: " << error.what()
                          << "; using the algorithm optimizer defaults\n";
            }
        } else {
            std::clog << "warning: learning checkpoint has no optimizer state; "
                      << "using the algorithm optimizer defaults\n";
        }
        actions = tensor_vector<std::int64_t>(reader.tensor(), torch::kInt64);
        rewards = tensor_vector<double>(reader.tensor(), torch::kFloat64);
        returns = tensor_vector<double>(reader.tensor(), torch::kFloat64);
        log_probabilities = tensor_vector<double>(reader.tensor(), torch::kFloat64);
        if (actions.size() != rewards.size() || actions.size() != returns.size() ||
            actions.size() != log_probabilities.size())
            throw std::invalid_argument("checkpoint rollout lengths differ");
        const auto observation_count = reader.u64();
        observations.reserve(static_cast<std::size_t>(observation_count));
        for (std::uint64_t i = 0; i < observation_count; ++i)
            observations.push_back(read_portable_observation(reader));
        if (q_learning() != (reader.u64() != 0U)) throw std::invalid_argument("checkpoint learning algorithm mismatch");
        if (q_learning()) {
            const auto target_parameters = unique_parameters(target_policy->module());
            if (reader.u64() != target_parameters.size()) throw std::invalid_argument("checkpoint target parameter count mismatch");
            for (const auto& parameter : target_parameters) parameter.copy_(reader.tensor().to(parameter.options()));
            const auto target_buffers = target_policy->module().named_buffers(true);
            if (reader.u64() != target_buffers.size()) throw std::invalid_argument("checkpoint target buffer count mismatch");
            for (const auto& buffer : target_buffers) buffer.value().copy_(reader.tensor().to(buffer.value().options()));
            target_policy->train(reader.tensor().item<std::int64_t>() != 0);
            dones = tensor_vector<std::uint8_t>(reader.tensor(), torch::kUInt8);
            const auto next_observation_count = reader.u64();
            next_observations.reserve(static_cast<std::size_t>(next_observation_count));
            for (std::uint64_t i = 0; i < next_observation_count; ++i)
                next_observations.push_back(read_portable_observation(reader));
            const auto keys = tensor_vector<std::int64_t>(reader.tensor(), torch::kInt64);
            const auto integers = tensor_vector<std::int64_t>(reader.tensor(), torch::kInt64);
            const auto reals = tensor_vector<double>(reader.tensor(), torch::kFloat64);
            if (keys.size() != 624U || integers.size() != 3U || reals.size() != 2U)
                throw std::invalid_argument("invalid checkpoint exploration state");
            DqnExplorationState exploration;
            for (std::size_t i = 0; i < keys.size(); ++i) {
                if (keys[i] < 0 || keys[i] > UINT32_MAX)
                    throw std::invalid_argument("invalid checkpoint MT19937 word");
                exploration.random.key[i] = static_cast<std::uint32_t>(keys[i]);
            }
            exploration.steps_done = integers[0];
            exploration.random.position = static_cast<std::uint32_t>(integers[1]);
            exploration.random.has_gaussian = integers[2] != 0;
            exploration.epsilon = reals[0]; exploration.random.gaussian = reals[1];
            q_selector->restore(exploration);
        }
        if (q_trainer) q_trainer->set_update_time(restored_update_time); else trainer->set_update_time(restored_update_time);
        auto generator = at::detail::getDefaultCPUGenerator(); generator.set_state(restored_rng);
        reader.require_end();
    }
    std::vector<std::uint8_t> checkpoint() const {
        if (!policy) throw std::logic_error("cannot checkpoint an uninitialized learning session");
        torch::NoGradGuard no_grad;
        Output archive;
        archive.write("configuration_i", configuration_integers(), true);
        archive.write("configuration_f", configuration_reals(), true);
        const auto& o = policy->options();
        archive.write("dimensions", torch::tensor(std::vector<std::int64_t>{o.physical_nodes, o.physical_features,
            o.virtual_node_features, o.virtual_features, o.physical_edge_features.value_or(-1), o.virtual_edge_features.value_or(-1)}, torch::kInt64), true);
        // Persist raw tensors instead of nesting a TorchScript module archive.
        // LibTorch 2.6 may abort in its JIT regex loader when that nested
        // archive is reopened from a different process.
        const auto parameters = unique_parameters(policy->module());
        archive.write("parameter_count", torch::tensor(static_cast<std::int64_t>(parameters.size()), torch::kInt64), true);
        for (std::size_t i = 0; i < parameters.size(); ++i)
            archive.write("parameter_" + std::to_string(i), parameters[i].detach().cpu(), true);
        const auto buffers = policy->module().named_buffers(true);
        archive.write("buffer_count", torch::tensor(static_cast<std::int64_t>(buffers.size()), torch::kInt64), true);
        for (std::size_t i = 0; i < buffers.size(); ++i)
            archive.write("buffer_" + std::to_string(i), buffers[i].value().detach().cpu(), true);
        archive.write("training", torch::tensor(static_cast<std::int64_t>(policy->module().is_training())), true);
        archive.write("update_time", torch::tensor(update_time()), true);
        archive.write("rng", at::detail::getDefaultCPUGenerator().get_state(), true);
        const auto state = optimizer->snapshot();
        archive.write("moment_count", torch::tensor(static_cast<std::int64_t>(state.moments.size())), true);
        for (std::size_t i = 0; i < state.moments.size(); ++i) {
            const auto prefix = "moment_" + std::to_string(i) + "_";
            write_optional(archive, prefix + "step", state.moments[i].step);
            write_optional(archive, prefix + "mean", state.moments[i].exp_avg);
            write_optional(archive, prefix + "variance", state.moments[i].exp_avg_sq);
            write_optional(archive, prefix + "maximum", state.moments[i].max_exp_avg_sq);
        }
        std::vector<double> groups;
        for (const auto& group : state.options) groups.insert(groups.end(), {group.lr, group.beta1, group.beta2,
            group.eps, group.weight_decay, static_cast<double>(group.amsgrad), static_cast<double>(group.maximize)});
        archive.write("optimizer_present", torch::tensor(static_cast<std::int64_t>(1), torch::kInt64), true);
        archive.write("optimizer_groups", torch::tensor(groups, torch::kFloat64), true);
        archive.write("actions", torch::tensor(actions, torch::kInt64), true);
        archive.write("rewards", torch::tensor(rewards, torch::kFloat64), true);
        archive.write("returns", torch::tensor(returns, torch::kFloat64), true);
        archive.write("logprobs", torch::tensor(log_probabilities, torch::kFloat64), true);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            write_observation(archive, "observation_" + std::to_string(i) + "_", observations[i]);
        }
        if (q_learning()) {
            const auto target_parameters = unique_parameters(target_policy->module());
            archive.write("target_parameter_count", torch::tensor(static_cast<std::int64_t>(target_parameters.size()), torch::kInt64), true);
            for (std::size_t i = 0; i < target_parameters.size(); ++i)
                archive.write("target_parameter_" + std::to_string(i), target_parameters[i].detach().cpu(), true);
            const auto target_buffers = target_policy->module().named_buffers(true);
            archive.write("target_buffer_count", torch::tensor(static_cast<std::int64_t>(target_buffers.size()), torch::kInt64), true);
            for (std::size_t i = 0; i < target_buffers.size(); ++i)
                archive.write("target_buffer_" + std::to_string(i), target_buffers[i].value().detach().cpu(), true);
            archive.write("target_training", torch::tensor(static_cast<std::int64_t>(target_policy->module().is_training())), true);
            archive.write("dones", torch::tensor(dones, torch::kUInt8), true);
            for (std::size_t i = 0; i < next_observations.size(); ++i) {
                write_observation(archive, "next_observation_" + std::to_string(i) + "_", next_observations[i]);
            }
            const auto state = q_selector->snapshot();
            archive.write("exploration_key", torch::tensor(std::vector<std::int64_t>(state.random.key.begin(), state.random.key.end()), torch::kInt64), true);
            archive.write("exploration_i", torch::tensor(std::vector<std::int64_t>{state.steps_done, state.random.position, state.random.has_gaussian}, torch::kInt64), true);
            archive.write("exploration_f", torch::tensor(std::vector<double>{state.epsilon, state.random.gaussian}, torch::kFloat64), true);
        }
        std::ostringstream stream(std::ios::out | std::ios::binary); archive.save_to(stream);
        const auto bytes = stream.str(); return {bytes.begin(), bytes.end()};
    }
    void restore(Input& archive) {
        if (!torch::equal(read_tensor(archive, "configuration_i"), configuration_integers())
            || !torch::equal(read_tensor(archive, "configuration_f"), configuration_reals()))
            throw std::invalid_argument("checkpoint learning configuration mismatch");
        const auto dims = tensor_vector<std::int64_t>(read_tensor(archive, "dimensions"), torch::kInt64);
        if (dims.size() != 6) throw std::invalid_argument("invalid checkpoint dimensions");
        p::PolicyOptions o;
        o.kind = options.policy; o.physical_nodes = dims[0]; o.physical_features = dims[1];
        o.virtual_node_features = dims[2]; o.virtual_features = dims[3];
        if (dims[4] >= 0) o.physical_edge_features = dims[4];
        if (dims[5] >= 0) o.virtual_edge_features = dims[5];
        o.embedding_dim = options.embedding_dim; o.num_layers = options.num_layers;
        o.dropout = options.dropout; o.batch_norm = options.batch_norm;
        initialize(o);
        torch::NoGradGuard no_grad;
        torch::Tensor parameter_count;
        if (!archive.try_read("parameter_count", parameter_count))
            throw std::invalid_argument("checkpoint does not contain portable policy tensors");
        const auto parameters = unique_parameters(policy->module());
        if (parameter_count.item<std::int64_t>() != static_cast<std::int64_t>(parameters.size()))
            throw std::invalid_argument("checkpoint policy parameter count mismatch");
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            const auto source = read_tensor(archive, "parameter_" + std::to_string(i));
            if (source.sizes() != parameters[i].sizes()) throw std::invalid_argument("checkpoint policy parameter shape mismatch");
            parameters[i].copy_(source.to(parameters[i].options()));
        }
        torch::Tensor buffer_count;
        if (!archive.try_read("buffer_count", buffer_count))
            throw std::invalid_argument("checkpoint does not contain policy buffers");
        const auto buffers = policy->module().named_buffers(true);
        if (buffer_count.item<std::int64_t>() != static_cast<std::int64_t>(buffers.size()))
            throw std::invalid_argument("checkpoint policy buffer count mismatch");
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            const auto source = read_tensor(archive, "buffer_" + std::to_string(i));
            if (source.sizes() != buffers[i].value().sizes()) throw std::invalid_argument("checkpoint policy buffer shape mismatch");
            buffers[i].value().copy_(source.to(buffers[i].value().options()));
        }
        policy->train(read_tensor(archive, "training").item<std::int64_t>() != 0);
        auto state = optimizer->snapshot();
        torch::Tensor optimizer_marker;
        const bool has_optimizer = archive.try_read("optimizer_present", optimizer_marker)
            ? optimizer_marker.item<std::int64_t>() != 0
            : archive.try_read("moment_count", optimizer_marker);
        if (has_optimizer) {
            try {
                const auto moment_count = archive.try_read("moment_count", optimizer_marker)
                    ? optimizer_marker.item<std::int64_t>() : -1;
                if (moment_count != static_cast<std::int64_t>(state.moments.size()))
                    throw std::invalid_argument("checkpoint optimizer parameter count mismatch");
                for (std::size_t i = 0; i < state.moments.size(); ++i) {
                    const auto prefix = "moment_" + std::to_string(i) + "_";
                    state.moments[i] = {read_optional(archive, prefix + "step"), read_optional(archive, prefix + "mean"),
                        read_optional(archive, prefix + "variance"), read_optional(archive, prefix + "maximum")};
                }
                const auto groups = tensor_vector<double>(read_tensor(archive, "optimizer_groups"), torch::kFloat64);
                if (groups.size() != state.options.size() * 7) throw std::invalid_argument("checkpoint optimizer group count mismatch");
                for (std::size_t i = 0; i < state.options.size(); ++i) state.options[i] = {groups[i*7], groups[i*7+1], groups[i*7+2],
                    groups[i*7+3], groups[i*7+4], groups[i*7+5] != 0, groups[i*7+6] != 0};
                optimizer->restore(state);
            } catch (const std::exception& error) {
                std::clog << "warning: learning checkpoint optimizer ignored: " << error.what()
                          << "; using the algorithm optimizer defaults\n";
            }
        } else {
            std::clog << "warning: learning checkpoint has no optimizer state; using the algorithm optimizer defaults\n";
        }
        const auto updates = read_tensor(archive, "update_time").item<std::int64_t>();
        if (q_trainer) q_trainer->set_update_time(updates); else trainer->set_update_time(updates);
        actions = tensor_vector<std::int64_t>(read_tensor(archive, "actions"), torch::kInt64);
        rewards = tensor_vector<double>(read_tensor(archive, "rewards"), torch::kFloat64);
        returns = tensor_vector<double>(read_tensor(archive, "returns"), torch::kFloat64);
        log_probabilities = tensor_vector<double>(read_tensor(archive, "logprobs"), torch::kFloat64);
        if (rewards.size() != actions.size() || returns.size() != actions.size() || log_probabilities.size() != actions.size())
            throw std::invalid_argument("checkpoint rollout lengths differ");
        for (std::size_t i = 0; i < actions.size(); ++i) {
            observations.push_back(read_observation(archive, "observation_" + std::to_string(i) + "_"));
        }
        if (q_learning()) {
            torch::Tensor target_parameter_count;
            if (!archive.try_read("target_parameter_count", target_parameter_count))
                throw std::invalid_argument("checkpoint does not contain portable target policy tensors");
            const auto target_parameters = unique_parameters(target_policy->module());
            if (target_parameter_count.item<std::int64_t>() != static_cast<std::int64_t>(target_parameters.size()))
                throw std::invalid_argument("checkpoint target parameter count mismatch");
            for (std::size_t i = 0; i < target_parameters.size(); ++i) {
                const auto source = read_tensor(archive, "target_parameter_" + std::to_string(i));
                if (source.sizes() != target_parameters[i].sizes()) throw std::invalid_argument("checkpoint target parameter shape mismatch");
                target_parameters[i].copy_(source.to(target_parameters[i].options()));
            }
            const auto target_buffers = target_policy->module().named_buffers(true);
            const auto target_buffer_count = read_tensor(archive, "target_buffer_count").item<std::int64_t>();
            if (target_buffer_count != static_cast<std::int64_t>(target_buffers.size()))
                throw std::invalid_argument("checkpoint target buffer count mismatch");
            for (std::size_t i = 0; i < target_buffers.size(); ++i) {
                const auto source = read_tensor(archive, "target_buffer_" + std::to_string(i));
                if (source.sizes() != target_buffers[i].value().sizes()) throw std::invalid_argument("checkpoint target buffer shape mismatch");
                target_buffers[i].value().copy_(source.to(target_buffers[i].value().options()));
            }
            target_policy->train(read_tensor(archive, "target_training").item<std::int64_t>() != 0);
            dones = tensor_vector<std::uint8_t>(read_tensor(archive, "dones"), torch::kUInt8);
            if (dones.size() != actions.size()) throw std::invalid_argument("checkpoint DQN terminal count mismatch");
            for (std::size_t i = 0; i < actions.size(); ++i) {
                next_observations.push_back(read_observation(archive, "next_observation_" + std::to_string(i) + "_"));
            }
            const auto keys = tensor_vector<std::int64_t>(read_tensor(archive, "exploration_key"), torch::kInt64);
            const auto integers = tensor_vector<std::int64_t>(read_tensor(archive, "exploration_i"), torch::kInt64);
            const auto reals = tensor_vector<double>(read_tensor(archive, "exploration_f"), torch::kFloat64);
            if (keys.size() != 624 || integers.size() != 3 || reals.size() != 2 || integers[1] < 0 || integers[1] > 624)
                throw std::invalid_argument("invalid checkpoint exploration shape");
            DqnExplorationState state;
            for (std::size_t i = 0; i < keys.size(); ++i) {
                if (keys[i] < 0 || keys[i] > UINT32_MAX) throw std::invalid_argument("invalid checkpoint MT19937 word");
                state.random.key[i] = static_cast<std::uint32_t>(keys[i]);
            }
            state.steps_done = integers[0]; state.random.position = static_cast<std::uint32_t>(integers[1]);
            state.random.has_gaussian = integers[2] != 0; state.epsilon = reals[0]; state.random.gaussian = reals[1];
            q_selector->restore(state);
        }
        auto generator = at::detail::getDefaultCPUGenerator(); generator.set_state(read_tensor(archive, "rng"));
    }
    std::optional<double> update() {
        if (observations.empty()) return std::nullopt;
        policy->train();
        last_actor_loss.reset(); last_critic_loss.reset(); last_entropy_loss.reset();
        last_approximate_kl.reset(); last_gradient_norm.reset();
        last_log_probability_mean.reset(); last_old_log_probability_mean.reset();
        last_value_mean.reset(); last_return_mean.reset(); last_advantage_mean.reset();
        last_reward_mean.reset(); last_applied_update_steps = 0;
        torch::Tensor loss;
        if (q_learning()) {
            DqnBatch batch{torch::tensor(actions, torch::kInt64), torch::tensor(rewards, torch::kFloat32), torch::tensor(dones, torch::kFloat32)};
            const auto update_result = q_trainer->update(batch);
            loss = update_result.loss;
            last_gradient_norm = update_result.gradient_norm.item<double>();
            last_applied_update_steps = 1;
            next_observations.clear(); dones.clear();
        } else {
            TrainingBatch batch{torch::tensor(actions, torch::kInt64), torch::tensor(rewards, torch::kFloat32),
                torch::tensor(returns, torch::kFloat32), torch::tensor(log_probabilities, torch::kFloat32)};
            const auto update_result = trainer->update(batch);
            loss = update_result.loss;
            const auto scalar = [](const torch::Tensor& value) -> std::optional<double> {
                return value.defined() ? std::optional<double>(value.item<double>()) : std::nullopt;
            };
            last_actor_loss = scalar(update_result.actor_loss);
            last_critic_loss = scalar(update_result.critic_loss);
            last_entropy_loss = scalar(update_result.entropy);
            last_approximate_kl = scalar(update_result.approximate_kl);
            last_gradient_norm = scalar(update_result.gradient_norm);
            last_log_probability_mean = scalar(update_result.log_probability_mean);
            last_old_log_probability_mean = scalar(update_result.old_log_probability_mean);
            last_value_mean = scalar(update_result.value_mean);
            last_return_mean = scalar(update_result.return_mean);
            last_advantage_mean = scalar(update_result.advantage_mean);
            last_reward_mean = scalar(update_result.reward_mean);
            last_applied_update_steps = update_result.applied_steps;
        }
        observations.clear(); actions.clear(); rewards.clear(); returns.clear(); log_probabilities.clear();
        return loss.item<double>();
    }
    EpisodeLearningResult run(InstanceEnvironment& env, const bool learn, const bool sample) {
        auto raw = env.reset();
        if (!policy) initialize(raw);
        policy->train(learn);
        const auto kind = p::policy_descriptor(options.policy).observation;
        auto observation = prepare_policy_observation(raw, kind);
        torch::Tensor encoded, hidden, recurrent_mask;
        std::int64_t previous = static_cast<std::int64_t>(raw.features.p_net.x.rows);
        if (kind == p::ObservationKind::recurrent) {
            torch::NoGradGuard no_grad;
            encoded = policy->encode(observation).transpose(0, 1);
            hidden = policy->last_rnn_state().permute({1, 0, 2});
            recurrent_mask = torch::ones({1, encoded.size(1)}, torch::kBool);
        }
        const auto set_recurrent = [&](p::PolicyObservation& value) {
            if (kind != p::ObservationKind::recurrent) return;
            value.hidden_state = hidden;
            value.encoder_outputs = encoded;
            value.recurrent_mask = recurrent_mask;
            value.p_node_id = torch::tensor({previous}, torch::kInt64);
        };
        EpisodeLearningResult result;
        std::vector<p::PolicyObservation> episode_observations, episode_next_observations;
        while (true) {
            if (options.max_episode_steps && result.actions.size() >= *options.max_episode_steps) throw std::runtime_error("configured RL episode step limit exceeded");
            set_recurrent(observation);
            std::int64_t action;
            double value, log_probability;
            {
                torch::NoGradGuard no_grad;
                if (q_learning()) {
                    if (env.num_actions() != policy->options().physical_nodes) throw std::invalid_argument("DQN requires a physical-node action space");
                    const auto selected = q_selector->select([&] { return policy->act(observation); }, observation.action_mask, sample);
                    action = selected.action; log_probability = selected.log_probability.item<double>();
                } else {
                    const auto logits = policy->act(observation);
                    if (logits.numel() != env.num_actions()) throw std::invalid_argument("policy output and environment action space differ");
                    const auto selected = select_policy_action(logits, observation.action_mask,
                        sample ? PolicyActionMode::sample : PolicyActionMode::greedy);
                    action = selected.actions.item<std::int64_t>();
                    log_probability = selected.log_probabilities.item<double>();
                }
                value = policy->evaluate(observation).squeeze(-1).item<double>();
                if (kind == p::ObservationKind::recurrent) hidden = policy->last_rnn_state().permute({1, 0, 2});
            }
            const auto step = env.step(action);
            episode_observations.push_back(std::move(observation));
            result.actions.push_back(action); result.values.push_back(value); result.log_probabilities.push_back(log_probability);
            result.rewards.push_back(step.reward); result.dones.push_back(step.done ? 1 : 0);
            previous = action;
            observation = prepare_policy_observation(step.observation, kind);
            if (q_learning()) episode_next_observations.push_back(observation);
            if (step.done) { result.solution = step.info; break; }
        }
        set_recurrent(observation);
        {
            torch::NoGradGuard no_grad;
            result.last_value = policy->evaluate(observation).squeeze(-1).item<double>();
        }
        switch (options.advantage) {
        case AdvantageMethod::gae:
            result.returns = compute_returns_with_gae(result.rewards, result.values, result.dones,
                result.last_value, options.gamma, options.gae_lambda).returns; break;
        case AdvantageMethod::monte_carlo:
            result.returns = compute_returns_with_mc(result.rewards, result.dones, options.gamma); break;
        case AdvantageMethod::temporal_difference:
            result.returns = compute_returns_with_td(result.rewards, result.values, result.dones, result.last_value, options.gamma); break;
        }
        if (learn && (options.include_failed || result.solution.result)) {
            observations.insert(observations.end(), std::make_move_iterator(episode_observations.begin()), std::make_move_iterator(episode_observations.end()));
            actions.insert(actions.end(), result.actions.begin(), result.actions.end());
            rewards.insert(rewards.end(), result.rewards.begin(), result.rewards.end());
            returns.insert(returns.end(), result.returns.begin(), result.returns.end());
            log_probabilities.insert(log_probabilities.end(), result.log_probabilities.begin(), result.log_probabilities.end());
            if (q_learning()) {
                next_observations.insert(next_observations.end(), std::make_move_iterator(episode_next_observations.begin()), std::make_move_iterator(episode_next_observations.end()));
                dones.insert(dones.end(), result.dones.begin(), result.dones.end());
            }
            if (static_cast<std::int64_t>(observations.size()) >= options.target_steps) {
                result.update_loss = update();
                result.actor_loss = last_actor_loss;
                result.critic_loss = last_critic_loss;
                result.entropy_loss = last_entropy_loss;
                result.approximate_kl = last_approximate_kl;
                result.gradient_norm = last_gradient_norm;
                result.log_probability_mean = last_log_probability_mean;
                result.old_log_probability_mean = last_old_log_probability_mean;
                result.value_mean = last_value_mean;
                result.return_mean = last_return_mean;
                result.advantage_mean = last_advantage_mean;
                result.reward_mean = last_reward_mean;
                result.learning_rate = optimizer->options(0).lr;
                result.applied_update_steps = last_applied_update_steps;
            }
        }
        result.update_time = update_time();
        result.pending_steps = static_cast<std::int64_t>(observations.size());
        if (q_selector) {
            const auto state = q_selector->snapshot(); result.exploration_epsilon = state.epsilon; result.exploration_steps = state.steps_done;
        }
        return result;
    }
};

LearningSession::LearningSession(LearningSessionOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
LearningSession::~LearningSession() = default;
LearningSession::LearningSession(LearningSession&&) noexcept = default;
LearningSession& LearningSession::operator=(LearningSession&&) noexcept = default;
EpisodeLearningResult LearningSession::learn(InstanceEnvironment& environment) { return impl_->run(environment, true, true); }
EpisodeLearningResult LearningSession::solve(InstanceEnvironment& environment, const bool sample) { return impl_->run(environment, false, sample); }
std::optional<double> LearningSession::update() { return impl_->update(); }
std::int64_t LearningSession::pending_steps() const noexcept { return static_cast<std::int64_t>(impl_->observations.size()); }
std::int64_t LearningSession::update_time() const noexcept { return impl_->update_time(); }
void LearningSession::set_sampling_seed(const std::uint64_t seed) { torch::manual_seed(seed); }
void LearningSession::set_exploration_seed(const std::uint32_t seed) {
    if (impl_->q_selector) impl_->q_selector->seed(seed); else impl_->pending_exploration_seed = seed;
}
std::vector<double> LearningSession::parameter_values() const {
    std::vector<double> result;
    if (!impl_->policy) return result;
    for (const auto& parameter : unique_parameters(impl_->policy->module())) {
        const auto data = parameter.detach().cpu().to(torch::kFloat64).contiguous().view({-1});
        const auto* pointer = data.data_ptr<double>();
        result.insert(result.end(), pointer, pointer + data.numel());
    }
    return result;
}
std::vector<double> LearningSession::target_parameter_values() const {
    std::vector<double> result;
    if (!impl_->target_policy) return result;
    for (const auto& parameter : unique_parameters(impl_->target_policy->module())) {
        const auto data = parameter.detach().cpu().to(torch::kFloat64).contiguous().view({-1});
        const auto* pointer = data.data_ptr<double>(); result.insert(result.end(), pointer, pointer + data.numel());
    }
    return result;
}
std::vector<std::uint8_t> LearningSession::checkpoint() const { return impl_->portable_checkpoint(); }
void LearningSession::restore_checkpoint(const std::vector<std::uint8_t>& bytes) {
    auto generator = at::detail::getDefaultCPUGenerator();
    const auto rng_before = generator.get_state();
    try {
        const std::size_t magic_size = sizeof(checkpoint_magic) - 1U;
        if (bytes.size() < magic_size || !std::equal(checkpoint_magic,
                checkpoint_magic + magic_size, bytes.begin()))
            throw std::invalid_argument("legacy learning checkpoint is unsupported; using the algorithm defaults");
        auto restored = std::make_unique<Impl>(impl_->options);
        restored->restore_portable(bytes);
        impl_ = std::move(restored);
    } catch (...) {
        generator.set_state(rng_before); throw;
    }
}

}  // namespace virne::solver::learning::rl_core
