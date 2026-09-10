#include "learning_session.h"
#include "policy_builder.h"
#include "policy_observation.h"
#include "rollout_returns.h"
#include "training_update.h"
#include "dqn_update.h"
#include "dqn_action.h"
#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>
#include <iterator>
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
void write_observation(Output& archive, const p::PolicyObservation& o) {
    const std::vector<std::pair<const char*, torch::Tensor>> fields{
        {"physical_x", o.p_net_x}, {"vnode_x", o.v_node_x}, {"virtual_x", o.v_net_x},
        {"current", o.current_virtual_node}, {"size", o.virtual_net_size}, {"hidden", o.hidden_state},
        {"node_id", o.p_node_id}, {"encoded", o.encoder_outputs},
        {"recurrent_mask", o.recurrent_mask.value_or(torch::Tensor{})}, {"action_mask", o.action_mask.value_or(torch::Tensor{})},
        {"p_x", o.p_net.x}, {"p_edges", o.p_net.edge_index}, {"p_attr", o.p_net.edge_attr.value_or(torch::Tensor{})},
        {"v_x", o.v_net.x}, {"v_edges", o.v_net.edge_index}, {"v_attr", o.v_net.edge_attr.value_or(torch::Tensor{})}};
    for (const auto& field : fields) write_optional(archive, field.first, field.second);
}
p::PolicyObservation read_observation(Input& archive) {
    p::PolicyObservation o;
    o.p_net_x = read_optional(archive, "physical_x"); o.v_node_x = read_optional(archive, "vnode_x");
    o.v_net_x = read_optional(archive, "virtual_x"); o.current_virtual_node = read_optional(archive, "current");
    o.virtual_net_size = read_optional(archive, "size"); o.hidden_state = read_optional(archive, "hidden");
    o.p_node_id = read_optional(archive, "node_id"); o.encoder_outputs = read_optional(archive, "encoded");
    const auto recurrent_mask = read_optional(archive, "recurrent_mask"), mask = read_optional(archive, "action_mask");
    if (recurrent_mask.defined()) o.recurrent_mask = recurrent_mask;
    if (mask.defined()) o.action_mask = mask;
    const auto graph = [&](const std::string& prefix) {
        pyg_lite::Data data;
        data.x = read_optional(archive, prefix + "_x"); data.edge_index = read_optional(archive, prefix + "_edges");
        const auto attributes = read_optional(archive, prefix + "_attr");
        if (attributes.defined()) data.edge_attr = attributes;
        return data.x.defined() ? pyg_lite::Batch::from_data_list({data}) : pyg_lite::Batch{};
    };
    o.p_net = graph("p"); o.v_net = graph("v"); return o;
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
    std::vector<std::uint8_t> checkpoint() const {
        if (!policy) throw std::logic_error("cannot checkpoint an uninitialized learning session");
        torch::NoGradGuard no_grad;
        Output archive, model;
        archive.write("configuration_i", configuration_integers(), true);
        archive.write("configuration_f", configuration_reals(), true);
        const auto& o = policy->options();
        archive.write("dimensions", torch::tensor(std::vector<std::int64_t>{o.physical_nodes, o.physical_features,
            o.virtual_node_features, o.virtual_features, o.physical_edge_features.value_or(-1), o.virtual_edge_features.value_or(-1)}, torch::kInt64), true);
        policy->module().save(model); archive.write("policy", model);
        archive.write("training", torch::tensor(static_cast<std::int64_t>(policy->module().is_training())), true);
        archive.write("update_time", torch::tensor(update_time()), true);
        archive.write("rng", at::detail::getDefaultCPUGenerator().get_state(), true);
        const auto state = optimizer->snapshot();
        archive.write("moment_count", torch::tensor(static_cast<std::int64_t>(state.moments.size())), true);
        for (std::size_t i = 0; i < state.moments.size(); ++i) {
            Output moment;
            write_optional(moment, "step", state.moments[i].step);
            write_optional(moment, "mean", state.moments[i].exp_avg);
            write_optional(moment, "variance", state.moments[i].exp_avg_sq);
            write_optional(moment, "maximum", state.moments[i].max_exp_avg_sq);
            archive.write("moment_" + std::to_string(i), moment);
        }
        std::vector<double> groups;
        for (const auto& group : state.options) groups.insert(groups.end(), {group.lr, group.beta1, group.beta2,
            group.eps, group.weight_decay, static_cast<double>(group.amsgrad), static_cast<double>(group.maximize)});
        archive.write("optimizer_groups", torch::tensor(groups, torch::kFloat64), true);
        archive.write("actions", torch::tensor(actions, torch::kInt64), true);
        archive.write("rewards", torch::tensor(rewards, torch::kFloat64), true);
        archive.write("returns", torch::tensor(returns, torch::kFloat64), true);
        archive.write("logprobs", torch::tensor(log_probabilities, torch::kFloat64), true);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            Output record; write_observation(record, observations[i]); archive.write("observation_" + std::to_string(i), record);
        }
        if (q_learning()) {
            Output target; target_policy->module().save(target); archive.write("target_policy", target);
            archive.write("target_training", torch::tensor(static_cast<std::int64_t>(target_policy->module().is_training())), true);
            archive.write("dones", torch::tensor(dones, torch::kUInt8), true);
            for (std::size_t i = 0; i < next_observations.size(); ++i) {
                Output record; write_observation(record, next_observations[i]); archive.write("next_observation_" + std::to_string(i), record);
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
        Input model; archive.read("policy", model); policy->module().load(model);
        policy->train(read_tensor(archive, "training").item<std::int64_t>() != 0);
        auto state = optimizer->snapshot();
        if (read_tensor(archive, "moment_count").item<std::int64_t>() != static_cast<std::int64_t>(state.moments.size()))
            throw std::invalid_argument("checkpoint optimizer parameter count mismatch");
        for (std::size_t i = 0; i < state.moments.size(); ++i) {
            Input record; archive.read("moment_" + std::to_string(i), record);
            state.moments[i] = {read_optional(record, "step"), read_optional(record, "mean"),
                read_optional(record, "variance"), read_optional(record, "maximum")};
        }
        const auto groups = tensor_vector<double>(read_tensor(archive, "optimizer_groups"), torch::kFloat64);
        if (groups.size() != state.options.size() * 7) throw std::invalid_argument("checkpoint optimizer group count mismatch");
        for (std::size_t i = 0; i < state.options.size(); ++i) state.options[i] = {groups[i*7], groups[i*7+1], groups[i*7+2],
            groups[i*7+3], groups[i*7+4], groups[i*7+5] != 0, groups[i*7+6] != 0};
        optimizer->restore(state);
        const auto updates = read_tensor(archive, "update_time").item<std::int64_t>();
        if (q_trainer) q_trainer->set_update_time(updates); else trainer->set_update_time(updates);
        actions = tensor_vector<std::int64_t>(read_tensor(archive, "actions"), torch::kInt64);
        rewards = tensor_vector<double>(read_tensor(archive, "rewards"), torch::kFloat64);
        returns = tensor_vector<double>(read_tensor(archive, "returns"), torch::kFloat64);
        log_probabilities = tensor_vector<double>(read_tensor(archive, "logprobs"), torch::kFloat64);
        if (rewards.size() != actions.size() || returns.size() != actions.size() || log_probabilities.size() != actions.size())
            throw std::invalid_argument("checkpoint rollout lengths differ");
        for (std::size_t i = 0; i < actions.size(); ++i) {
            Input record; archive.read("observation_" + std::to_string(i), record); observations.push_back(read_observation(record));
        }
        if (q_learning()) {
            Input target; archive.read("target_policy", target); target_policy->module().load(target);
            target_policy->train(read_tensor(archive, "target_training").item<std::int64_t>() != 0);
            dones = tensor_vector<std::uint8_t>(read_tensor(archive, "dones"), torch::kUInt8);
            if (dones.size() != actions.size()) throw std::invalid_argument("checkpoint DQN terminal count mismatch");
            for (std::size_t i = 0; i < actions.size(); ++i) {
                Input record; archive.read("next_observation_" + std::to_string(i), record); next_observations.push_back(read_observation(record));
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
        torch::Tensor loss;
        if (q_learning()) {
            DqnBatch batch{torch::tensor(actions, torch::kInt64), torch::tensor(rewards, torch::kFloat32), torch::tensor(dones, torch::kFloat32)};
            loss = q_trainer->update(batch).loss;
            next_observations.clear(); dones.clear();
        } else {
            TrainingBatch batch{torch::tensor(actions, torch::kInt64), torch::tensor(rewards, torch::kFloat32),
                torch::tensor(returns, torch::kFloat32), torch::tensor(log_probabilities, torch::kFloat32)};
            loss = trainer->update(batch).loss;
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
            if (static_cast<std::int64_t>(observations.size()) >= options.target_steps) result.update_loss = update();
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
std::vector<std::uint8_t> LearningSession::checkpoint() const { return impl_->checkpoint(); }
void LearningSession::restore_checkpoint(const std::vector<std::uint8_t>& bytes) {
    auto generator = at::detail::getDefaultCPUGenerator();
    const auto rng_before = generator.get_state();
    try {
        Input archive; archive.load_from(reinterpret_cast<const char*>(bytes.data()), bytes.size(), torch::kCPU);
        auto restored = std::make_unique<Impl>(impl_->options);
        restored->restore(archive);
        impl_ = std::move(restored);
    } catch (...) {
        generator.set_state(rng_before); throw;
    }
}

}  // namespace virne::solver::learning::rl_core
