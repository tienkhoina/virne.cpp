#include "policy_builder.h"
#include <stdexcept>

namespace virne::solver::learning::rl_core {

rl_policy::PolicyOptions policy_options_from_configuration(const rl_policy::PolicyKind kind,
    const PolicyConfiguration& configuration) {
    const auto general = validate_policy_general_network_config(configuration.neural_network);
    const auto& f = configuration.feature_dimensions;
    rl_policy::PolicyOptions o;
    o.kind = kind; o.physical_nodes = f.p_net_num_nodes;
    o.physical_features = f.p_net_x_dim; o.virtual_features = f.v_net_x_dim;
    o.virtual_node_features = f.v_node_x_dim; o.embedding_dim = general.embedding_dim;
    o.dropout = general.dropout_prob; o.batch_norm = general.batch_norm;
    const auto schema = rl_policy::policy_descriptor(kind).observation;
    // Original GCN/GAT-MLP builders omit edge dimensions; DeepEdge and dual
    // builders explicitly pass them. Preserve that distinction at this boundary.
    if (schema == rl_policy::ObservationKind::dual || kind == rl_policy::PolicyKind::deep_edge_gat_mlp) {
        if (f.p_net_edge_dim > 0) o.physical_edge_features = f.p_net_edge_dim;
        if (f.v_net_edge_dim > 0) o.virtual_edge_features = f.v_net_edge_dim;
    }
    return o;
}

std::unique_ptr<rl_policy::Policy> build_policy(const rl_policy::PolicyKind kind,
    const PolicyConfiguration& configuration, const torch::Device& device) {
    return std::make_unique<rl_policy::Policy>(policy_options_from_configuration(kind, configuration), device);
}

std::unique_ptr<Adam> build_policy_optimizer(rl_policy::Policy& policy,
    const PolicyLearningRates& rates, const double weight_decay) {
    if (!rates.actor || !rates.critic) throw std::invalid_argument("learning rates require actor and critic entries");
    std::vector<AdamParameterGroup> groups;
    for (const auto& child : policy.module().named_children()) {
        std::optional<double> rate;
        const auto& name = child.key();
        if (name == "actor") rate = rates.actor;
        else if (name == "critic") rate = rates.critic;
        else if (name == "encoder") rate = rates.encoder;
        else if (name == "cost_critic") rate = rates.cost_critic;
        else if (name == "lambda_net") rate = rates.lambda_net;
        if (!rate) throw std::invalid_argument("missing learning rate for policy child: " + name);
        AdamOptions options;
        options.lr = *rate; options.weight_decay = weight_decay;
        groups.push_back({unique_parameters(*child.value()), options});
    }
    if (groups.empty()) throw std::invalid_argument("policy has no parameter groups");
    return std::make_unique<Adam>(std::move(groups));
}

}  // namespace virne::solver::learning::rl_core
