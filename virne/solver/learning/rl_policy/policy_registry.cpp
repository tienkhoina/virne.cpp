#include "policy_registry.h"
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace virne::solver::learning::rl_policy {
namespace {
constexpr std::array<PolicyDescriptor, 16> catalog{{
    {PolicyKind::cnn, "cnn", ObservationKind::cnn, false, false},
    {PolicyKind::gcn_mlp, "gcn_mlp", ObservationKind::gnn_mlp, false, false},
    {PolicyKind::gat_mlp, "gat_mlp", ObservationKind::gnn_mlp, false, false},
    {PolicyKind::deep_edge_gat_mlp, "deep_edge_gat_mlp", ObservationKind::gnn_mlp, false, false},
    {PolicyKind::mlp, "mlp", ObservationKind::dense, false, false},
    {PolicyKind::attention, "att", ObservationKind::attention, false, false},
    {PolicyKind::gcn_seq2seq, "gcn_seq2seq", ObservationKind::recurrent, false, false},
    {PolicyKind::dual_gcn, "dual_gcn", ObservationKind::dual, false, false},
    {PolicyKind::dual_gat, "dual_gat", ObservationKind::dual, false, false},
    {PolicyKind::dual_deep_edge_gat, "dual_deep_edge_gat", ObservationKind::dual, false, false},
    {PolicyKind::dual_gcn_cost, "dual_gcn_with_cost", ObservationKind::dual, true, false},
    {PolicyKind::dual_gat_cost, "dual_gat_with_cost", ObservationKind::dual, true, false},
    {PolicyKind::dual_deep_edge_gat_cost, "dual_deep_edge_gat_with_cost", ObservationKind::dual, true, false},
    {PolicyKind::dual_gcn_cost_lambda, "dual_gcn_with_cost_and_lambda", ObservationKind::dual, true, true},
    {PolicyKind::dual_gat_cost_lambda, "dual_gat_with_cost_and_lambda", ObservationKind::dual, true, true},
    {PolicyKind::dual_deep_edge_gat_cost_lambda, "dual_deep_edge_gat_with_cost_and_lambda", ObservationKind::dual, true, true},
}};

DualObservation dual_observation(const PolicyObservation& o) { return {o.v_net, o.p_net, o.current_virtual_node}; }
Seq2SeqObservation recurrent_observation(const PolicyObservation& o) {
    return {o.p_net, o.v_net_x, o.hidden_state, o.p_node_id, o.encoder_outputs, o.recurrent_mask};
}
template <class Model, class Fn>
torch::Tensor dispatch_observation(Model& model, const PolicyObservation& o, Fn&& fn) {
    using T = std::decay_t<Model>;
    if constexpr (std::is_same_v<T, MlpActorCritic> || std::is_same_v<T, CnnActorCritic>) return fn(model, o.p_net_x);
    else if constexpr (std::is_same_v<T, GnnMlpActorCritic>) return fn(model, GnnMlpObservation{o.p_net, o.v_node_x});
    else if constexpr (std::is_same_v<T, AttActorCritic>) return fn(model, AttObservation{o.p_net_x, o.v_node_x});
    else if constexpr (std::is_same_v<T, DualActorCritic>) return fn(model, dual_observation(o));
    else return fn(model, recurrent_observation(o));
}
}

const std::array<PolicyDescriptor, 16>& registered_policies() noexcept { return catalog; }
PolicyKind policy_kind_from_name(const std::string_view name) {
    for (const auto& entry : catalog) if (entry.name == name) return entry.kind;
    throw std::invalid_argument("unknown actor-critic policy");
}
const PolicyDescriptor& policy_descriptor(const PolicyKind kind) {
    const auto index = static_cast<std::size_t>(kind);
    if (index >= catalog.size()) throw std::invalid_argument("invalid actor-critic policy ID");
    return catalog[index];
}

Policy::Policy(PolicyOptions o, const torch::Device& device) : options_(o) {
    const auto& descriptor = policy_descriptor(o.kind);
    if (o.physical_features < 0 || o.virtual_node_features < 0
        || o.physical_features > std::numeric_limits<std::int64_t>::max() - o.virtual_node_features)
        throw std::invalid_argument("invalid merged feature dimensions");
    const auto merged_features = o.physical_features + o.virtual_node_features;
    switch (descriptor.observation) {
    case ObservationKind::dense:
        model_ = MlpActorCritic(MlpActorCriticOptions{merged_features, o.physical_nodes,
            o.num_layers, o.embedding_dim, o.dropout, o.batch_norm}); break;
    case ObservationKind::cnn:
        model_ = CnnActorCritic(CnnActorCriticOptions{merged_features, o.physical_nodes, o.embedding_dim}); break;
    case ObservationKind::gnn_mlp: {
        const auto kind = o.kind == PolicyKind::gcn_mlp ? neural_network::GnnKind::gcn
            : (o.kind == PolicyKind::gat_mlp ? neural_network::GnnKind::gat : neural_network::GnnKind::deep_edge_gat);
        model_ = GnnMlpActorCritic(kind, GnnMlpPolicyOptions{o.physical_nodes, o.physical_features,
            o.virtual_node_features, o.num_layers, o.embedding_dim, o.physical_edge_features, o.dropout, o.batch_norm}); break;
    }
    case ObservationKind::attention:
        model_ = AttActorCritic(AttPolicyOptions{o.physical_nodes, o.physical_features,
            o.virtual_node_features, o.num_layers, o.embedding_dim, o.dropout, o.batch_norm}); break;
    case ObservationKind::dual: {
        const auto slot = (static_cast<std::size_t>(o.kind) - static_cast<std::size_t>(PolicyKind::dual_gcn)) % 3;
        const auto kind = slot == 0 ? neural_network::GnnKind::gcn : (slot == 1 ? neural_network::GnnKind::gat : neural_network::GnnKind::deep_edge_gat);
        model_ = DualActorCritic(DualPolicyOptions{o.physical_nodes, o.physical_features, o.physical_edge_features,
            o.virtual_features, o.virtual_edge_features, o.embedding_dim, o.dropout, o.batch_norm, o.num_layers,
            kind, o.virtual_gnn, o.physical_gnn, descriptor.cost, descriptor.multiplier}); break;
    }
    case ObservationKind::recurrent:
        model_ = GnnSeq2SeqActorCritic(Seq2SeqOptions{o.physical_nodes, o.physical_features, o.physical_edge_features,
            o.virtual_features, o.embedding_dim, o.dropout, o.batch_norm,
            neural_network::GnnKind::gcn, Seq2SeqFlavor::legacy_gcn}); break;
    }
    module().to(device);
}
torch::nn::Module& Policy::module() {
    return std::visit([](auto& model) -> torch::nn::Module& { return *model; }, model_);
}
void Policy::train(const bool enabled) { module().train(enabled); }
torch::Tensor Policy::act(const PolicyObservation& obs) {
    return std::visit([&](auto& model) { return dispatch_observation(model, obs,
        [](auto& value, const auto& input) { return value->act(input); }); }, model_);
}
torch::Tensor Policy::evaluate(const PolicyObservation& obs) {
    return std::visit([&](auto& model) { return dispatch_observation(model, obs,
        [](auto& value, const auto& input) { return value->evaluate(input); }); }, model_);
}
torch::Tensor Policy::evaluate_cost(const PolicyObservation& obs) {
    auto* model = std::get_if<DualActorCritic>(&model_);
    if (!model || !(*model)->cost_critic) throw std::logic_error("policy has no cost critic");
    return (*model)->cost_critic->forward(dual_observation(obs));
}
torch::Tensor Policy::evaluate_lambda(const PolicyObservation& obs) {
    auto* model = std::get_if<DualActorCritic>(&model_);
    if (!model || !(*model)->lambda_net) throw std::logic_error("policy has no lambda network");
    return (*model)->lambda_net->forward(dual_observation(obs));
}
torch::Tensor Policy::encode(const PolicyObservation& obs) {
    auto* model = std::get_if<GnnSeq2SeqActorCritic>(&model_);
    if (!model) throw std::logic_error("policy has no recurrent encoder");
    return (*model)->encode(recurrent_observation(obs));
}
torch::Tensor Policy::last_rnn_state() const {
    const auto* model = std::get_if<GnnSeq2SeqActorCritic>(&model_);
    return model ? (*model)->get_last_rnn_state() : torch::Tensor{};
}
void Policy::set_last_rnn_state(torch::Tensor state) {
    auto* model = std::get_if<GnnSeq2SeqActorCritic>(&model_);
    if (!model) throw std::logic_error("policy has no recurrent state");
    (*model)->set_last_rnn_hidden(std::move(state));
}

}  // namespace virne::solver::learning::rl_policy
