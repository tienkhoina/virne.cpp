#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace virne::solver::learning::rl_policy {
enum class PolicyKind : std::uint8_t {
    cnn, gcn_mlp, gat_mlp, deep_edge_gat_mlp, mlp, attention, gcn_seq2seq,
    dual_gcn, dual_gat, dual_deep_edge_gat,
    dual_gcn_cost, dual_gat_cost, dual_deep_edge_gat_cost,
    dual_gcn_cost_lambda, dual_gat_cost_lambda, dual_deep_edge_gat_cost_lambda,
};
enum class ObservationKind : std::uint8_t { dense, cnn, gnn_mlp, attention, dual, recurrent };
struct PolicyDescriptor { PolicyKind kind; std::string_view name; ObservationKind observation; bool cost, multiplier; };
const std::array<PolicyDescriptor, 16>& registered_policies() noexcept;
PolicyKind policy_kind_from_name(std::string_view name);
const PolicyDescriptor& policy_descriptor(PolicyKind kind);
}  // namespace virne::solver::learning::rl_policy
