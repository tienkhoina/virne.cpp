#pragma once

#include "pyg_lite.h"

#include <cstdint>
#include <optional>

namespace virne::solver::learning::pyg_lite {

struct BipartiteGATConvOptions {
    std::int64_t source_in_channels = 0;
    std::int64_t target_in_channels = 0;
    std::int64_t out_channels = 0;
    std::int64_t heads = 1;
    bool concat = true;
    double negative_slope = 0.2;
    double dropout = 0.0;
    bool add_self_loops = true;
    std::optional<std::int64_t> edge_dim;
    SelfLoopFill fill_value = SelfLoopFill::reduce(SelfLoopFillKind::mean);
    bool bias = true;
    bool residual = false;
};

class BipartiteGATConvImpl final : public torch::nn::Module {
public:
    explicit BipartiteGATConvImpl(BipartiteGATConvOptions options);

    void reset_parameters();
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;

    torch::Tensor forward(
        const torch::Tensor& x_source,
        const torch::Tensor& x_target,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_attr = std::nullopt);

    GATForwardResult forward_with_attention(
        const torch::Tensor& x_source,
        const torch::Tensor& x_target,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_attr = std::nullopt);

    const BipartiteGATConvOptions& options() const noexcept { return options_; }

    // These names match PyG's tuple-input GATConv state surface.
    torch::nn::Linear lin_src{nullptr};
    torch::nn::Linear lin_dst{nullptr};
    torch::nn::Linear lin_edge{nullptr};
    torch::nn::Linear res{nullptr};
    torch::Tensor att_src;
    torch::Tensor att_dst;
    torch::Tensor att_edge;
    torch::Tensor bias;

private:
    BipartiteGATConvOptions options_;
};
TORCH_MODULE(BipartiteGATConv);

}  // namespace virne::solver::learning::pyg_lite
