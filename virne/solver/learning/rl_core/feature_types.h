#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace virne::solver::learning::rl_core {

struct DenseMatrixF32 { std::size_t rows = 0U, columns = 0U; std::vector<float> values; };
struct DenseMatrixI64 { std::size_t rows = 0U, columns = 0U; std::vector<std::int64_t> values; };
struct DenseVectorF32 { std::vector<float> values; };
struct DenseVectorI32 { std::vector<std::int32_t> values; };
struct DenseVectorBool { std::vector<std::uint8_t> values; };

enum class FeatureConstructorKind : std::uint8_t { p_net, p_net_v_node, p_net_v_net };
struct NetworkFeatures { DenseMatrixF32 x; DenseMatrixI64 edge_index; DenseMatrixF32 edge_attr; };
struct FeatureObservation {
    NetworkFeatures p_net;
    std::optional<DenseVectorF32> v_node_x;
    std::optional<NetworkFeatures> v_net;
};

}  // namespace virne::solver::learning::rl_core
