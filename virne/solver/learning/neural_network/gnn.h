#pragma once

#include "../pyg_lite/pyg_lite.h"

#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace virne::solver::learning::neural_network {

enum class GraphPoolingKind : std::uint8_t {
    none,
    add,
    mean,
    max,
};

GraphPoolingKind graph_pooling_kind_from_name(std::string_view name);

class GraphPooling final {
public:
    explicit GraphPooling(GraphPoolingKind kind);

    torch::Tensor forward(
        const torch::Tensor& node_embeddings,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> size = std::nullopt) const;

    [[nodiscard]] GraphPoolingKind kind() const noexcept { return kind_; }

private:
    GraphPoolingKind kind_ = GraphPoolingKind::none;
};

struct GCNConvNetOptions {
    std::int64_t input_dim = 0;
    std::int64_t output_dim = 0;
    std::int64_t embedding_dim = 128;
    std::int64_t num_layers = 3;
    bool batch_norm = true;
    double dropout_prob = 1.0;
    bool return_batch = false;
    GraphPoolingKind pooling = GraphPoolingKind::none;
};

class GCNConvNetImpl final : public torch::nn::Module {
public:
    explicit GCNConvNetImpl(GCNConvNetOptions options);

    torch::Tensor forward(const pyg_lite::Data& input);
    torch::Tensor forward(const pyg_lite::Batch& input);

    [[nodiscard]] const GCNConvNetOptions& options() const noexcept {
        return options_;
    }

private:
    struct Layer {
        pyg_lite::GCNConv conv{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::Dropout dropout{nullptr};
        bool has_norm = false;
        bool has_dropout = false;
    };

    torch::Tensor forward_impl(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> max_num_nodes,
        std::optional<std::int64_t> batch_size);

    GCNConvNetOptions options_{};
    std::vector<Layer> layers_;
    std::optional<GraphPooling> graph_pooling_;
};

TORCH_MODULE(GCNConvNet);

}  // namespace virne::solver::learning::neural_network
