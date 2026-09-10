#include "dual_gnn_policy.h"
#include <torch/torch.h>
#include <cmath>
#include <unordered_set>

namespace virne::solver::learning::rl_policy {

NetEncoderImpl::NetEncoderImpl(const std::int64_t features, std::optional<std::int64_t> edges,
    const std::int64_t embedding, const double dropout, const bool norm,
    const std::int64_t layers, const neural_network::GnnKind kind,
    const bool initialize_residual_weights) {
    init_lin = register_module("init_lin", torch::nn::Linear(features, embedding));
    net_gnn = register_graph_module(*this, "net_gnn", kind, neural_network::GraphConvNetOptions{
        embedding, embedding, layers, embedding, 1, edges, norm, dropout, false,
        neural_network::GraphPoolingKind::none}, initialize_residual_weights);
    register_module("mean_pool", std::make_shared<torch::nn::Module>());
    att_pool = register_module("att_pool", neural_network::GraphPoolingModule(
        neural_network::GraphPoolingKind::attention, embedding));
}

NetEncoding NetEncoderImpl::forward(const pyg_lite::Batch& input) {
    const auto initial = init_lin->forward(input.x);
    auto copied = input.clone();
    copied.x = initial;
    const auto nodes = graph_forward(net_gnn, copied);
    const auto mean = pyg_lite::global_mean_pool(nodes, input.batch, input.num_graphs);
    const auto attention = att_pool->forward(nodes, input.batch, input.num_graphs);
    const auto global = mean + attention;
    const auto dense = pyg_lite::to_dense_batch(nodes, input.batch, 0.0, input.max_num_nodes, input.num_graphs).values;
    const auto initial_dense = pyg_lite::to_dense_batch(initial, input.batch, 0.0, input.max_num_nodes, input.num_graphs).values;
    return {nodes, global, dense + global.unsqueeze(1) + initial_dense, initial_dense};
}

BiGnnBaseModelImpl::BiGnnBaseModelImpl(DualPolicyOptions o) {
    v_net_encoder = register_module("v_net_encoder", NetEncoder(o.v_net_x_dim, o.v_net_edge_dim,
        o.embedding_dim, o.dropout_prob, o.batch_norm, o.num_layers, o.v_kind.value_or(o.kind), false));
    p_net_encoder = register_module("p_net_encoder", NetEncoder(o.p_net_x_dim, o.p_net_edge_dim,
        o.embedding_dim, o.dropout_prob, o.batch_norm, o.num_layers, o.p_kind.value_or(o.kind), false));
    lin = register_module("lin", torch::nn::Sequential());
    linear_in_ = torch::nn::Linear(o.embedding_dim, o.embedding_dim);
    lin->push_back(linear_in_);
    lin->push_back(torch::nn::ReLU());
    linear_out_ = torch::nn::Linear(o.embedding_dim, 1);
    lin->push_back(linear_out_);
    init_parameters();
}

void BiGnnBaseModelImpl::init_parameters() {
    torch::NoGradGuard no_grad;
    std::unordered_set<const c10::TensorImpl*> seen;
    // Initialization is a cold reflection boundary; forward only uses typed fields.
    for (const auto& parameter : named_parameters(true)) {
        if (!seen.insert(parameter.value().unsafeGetTensorImpl()).second) continue;
        if (parameter.key().find("weight") != std::string::npos) {
            const double bound = 1.0 / std::sqrt(static_cast<double>(parameter.value().size(-1)));
            parameter.value().uniform_(-bound, bound);
        } else if (parameter.key().find("bias") != std::string::npos) {
            parameter.value().zero_();
        }
    }
}

torch::Tensor BiGnnBaseModelImpl::forward(const DualObservation& obs) {
    const auto v = v_net_encoder->forward(obs.v_net);
    const auto p = p_net_encoder->forward(obs.p_net);
    const auto index = obs.curr_v_node_id.unsqueeze(1).unsqueeze(1).to(torch::kInt64);
    const auto current = v.dense.gather(1, index.expand({v.dense.size(0), -1, v.dense.size(-1)})).squeeze(1);
    const auto with_global = p.dense + v.graph.unsqueeze(1);
    const auto fused = with_global + current.unsqueeze(1);
    const auto hidden = torch::relu(torch::linear(fused, linear_in_->weight, linear_in_->bias));
    return torch::linear(hidden, linear_out_->weight, linear_out_->bias).squeeze(-1);
}

DualActorImpl::DualActorImpl(DualPolicyOptions o) {
    encoder = register_module("encoder", BiGnnBaseModel(o));
}
torch::Tensor DualActorImpl::forward(const DualObservation& obs) { return encoder->forward(obs); }
DualCriticImpl::DualCriticImpl(DualPolicyOptions o) {
    encoder = register_module("encoder", BiGnnBaseModel(o));
}
torch::Tensor DualCriticImpl::forward(const DualObservation& obs) { return encoder->forward(obs).mean(-1, true); }
DualActorCriticImpl::DualActorCriticImpl(DualPolicyOptions o) {
    actor = register_module("actor", DualActor(o));
    critic = register_module("critic", DualCritic(o));
    auto auxiliary = o;
    auxiliary.v_kind.reset();
    auxiliary.p_kind.reset();
    if (o.cost_critic) cost_critic = register_module("cost_critic", DualCritic(auxiliary));
    if (o.lambda_net) lambda_net = register_module("lambda_net", DualCritic(auxiliary));
}
torch::Tensor DualActorCriticImpl::act(const DualObservation& obs) { return actor->forward(obs); }
torch::Tensor DualActorCriticImpl::evaluate(const DualObservation& obs) { return critic->forward(obs); }

}  // namespace virne::solver::learning::rl_policy
