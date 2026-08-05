#pragma once

#include "bipartite_gat.h"
#include "hetero_data.h"

#include <array>
#include <cstdint>

namespace virne::solver::learning::pyg_lite {

struct HeteroConvOptions {
    std::array<BipartiteGATConvOptions, kHeteroRelations> relation_options{};
    std::array<bool, kHeteroRelations> enabled{};
};

struct HeteroConvOutput {
    std::array<torch::Tensor, kHeteroNodeTypes> x{};

    torch::Tensor& node(HeteroNodeType type) noexcept {
        return x[hetero_node_index(type)];
    }
    const torch::Tensor& node(HeteroNodeType type) const noexcept {
        return x[hetero_node_index(type)];
    }
};

class HeteroConvImpl final : public torch::nn::Module {
public:
    explicit HeteroConvImpl(HeteroConvOptions options);

    HeteroConvOutput forward(const HeteroData& data);
    const HeteroConvOptions& options() const noexcept { return options_; }

    BipartiteGATConv& relation(HeteroRelationId id) noexcept {
        return convolutions_[hetero_relation_index(id)];
    }
    const BipartiteGATConv& relation(HeteroRelationId id) const noexcept {
        return convolutions_[hetero_relation_index(id)];
    }

private:
    HeteroConvOptions options_;
    std::array<BipartiteGATConv, kHeteroRelations> convolutions_;
};
TORCH_MODULE(HeteroConv);

}  // namespace virne::solver::learning::pyg_lite
