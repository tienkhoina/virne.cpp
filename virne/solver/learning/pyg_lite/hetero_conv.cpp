#include "hetero_conv.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace virne::solver::learning::pyg_lite {

namespace {

std::pair<HeteroNodeType, HeteroNodeType> endpoints(HeteroRelationId id) {
    switch (id) {
    case HeteroRelationId::p_connect_p:
        return {HeteroNodeType::physical, HeteroNodeType::physical};
    case HeteroRelationId::v_connect_v:
        return {HeteroNodeType::virtual_node, HeteroNodeType::virtual_node};
    case HeteroRelationId::v_mapping_p:
        return {HeteroNodeType::virtual_node, HeteroNodeType::physical};
    case HeteroRelationId::p_mapping_v:
        return {HeteroNodeType::physical, HeteroNodeType::virtual_node};
    case HeteroRelationId::v_imaginary_p:
        return {HeteroNodeType::virtual_node, HeteroNodeType::physical};
    case HeteroRelationId::p_imaginary_v:
        return {HeteroNodeType::physical, HeteroNodeType::virtual_node};
    }
    throw PygLiteException(ErrorCode::unsupported_option, "unknown hetero relation");
}

}  // namespace

HeteroConvImpl::HeteroConvImpl(HeteroConvOptions options)
    : options_(std::move(options)),
      convolutions_{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr} {
    for (std::size_t index = 0U; index < kHeteroRelations; ++index) {
        if (!options_.enabled[index]) continue;
        convolutions_[index] = register_module(
            "relation_" + std::to_string(index),
            BipartiteGATConv(options_.relation_options[index]));
    }
}

HeteroConvOutput HeteroConvImpl::forward(const HeteroData& data) {
    // Data is validated when it enters HeteroBatch/encoder setup.  Keeping
    // validation out of this hot forward loop matches PyG and avoids device
    // synchronizations from endpoint checks on every relation.
    HeteroConvOutput result;
    std::array<std::vector<torch::Tensor>, kHeteroNodeTypes> messages;
    for (std::size_t index = 0U; index < kHeteroRelations; ++index) {
        if (!options_.enabled[index] || !data.relations[index].present()) continue;
        const auto id = static_cast<HeteroRelationId>(index);
        const auto [source, target] = endpoints(id);
        auto& module = convolutions_[index];
        if (!module) {
            throw PygLiteException(
                ErrorCode::unsupported_option,
                "heterogeneous relation module is not initialized");
        }
        const auto output = module->forward(
            data.node(source),
            data.node(target),
            data.relations[index].edge_index,
            data.relations[index].edge_attr);
        messages[hetero_node_index(target)].push_back(output);
    }
    for (std::size_t type = 0U; type < kHeteroNodeTypes; ++type) {
        if (messages[type].empty()) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "heterogeneous convolution produced no message for a node type");
        }
        // PyG HeteroConv groups relation outputs with stack(...).sum(dim=0).
        // Preserve that reduction order for bit-level oracle parity.
        result.x[type] = torch::stack(messages[type], 0).sum(0);
    }
    return result;
}

}  // namespace virne::solver::learning::pyg_lite
