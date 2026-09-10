#pragma once

#include "policy_types.h"

#include "mlp_policy.h"
#include "cnn_policy.h"
#include "gnn_mlp_policy.h"
#include "att_policy.h"
#include "dual_gnn_policy.h"
#include "seq2seq_policy.h"
#include <array>
#include <string_view>
#include <variant>

namespace virne::solver::learning::rl_policy {

struct PolicyOptions {
    PolicyKind kind = PolicyKind::mlp;
    std::int64_t physical_nodes = 0, physical_features = 0, virtual_node_features = 0, virtual_features = 0;
    std::optional<std::int64_t> physical_edge_features, virtual_edge_features;
    std::int64_t embedding_dim = 128, num_layers = 3;
    double dropout = 0.0;
    bool batch_norm = false;
    std::optional<neural_network::GnnKind> physical_gnn, virtual_gnn;
};
struct PolicyObservation {
    torch::Tensor p_net_x, v_node_x, v_net_x, current_virtual_node, virtual_net_size;
    pyg_lite::Batch p_net, v_net;
    torch::Tensor hidden_state, p_node_id, encoder_outputs;
    std::optional<torch::Tensor> recurrent_mask, action_mask;
};

class Policy final {
public:
    explicit Policy(PolicyOptions options, const torch::Device& device = torch::kCPU);
    torch::Tensor act(const PolicyObservation& observation);
    torch::Tensor evaluate(const PolicyObservation& observation);
    torch::Tensor evaluate_cost(const PolicyObservation& observation);
    torch::Tensor evaluate_lambda(const PolicyObservation& observation);
    torch::Tensor encode(const PolicyObservation& observation);
    torch::Tensor last_rnn_state() const;
    void set_last_rnn_state(torch::Tensor state);
    void train(bool enabled = true);
    torch::nn::Module& module();
    const PolicyOptions& options() const noexcept { return options_; }
private:
    using Model = std::variant<MlpActorCritic, CnnActorCritic, GnnMlpActorCritic,
        AttActorCritic, DualActorCritic, GnnSeq2SeqActorCritic>;
    PolicyOptions options_;
    Model model_{MlpActorCritic{nullptr}};
};

}  // namespace virne::solver::learning::rl_policy
