#include "hetero_data.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace virne::solver::learning::pyg_lite {

namespace {

void require_defined(const torch::Tensor& value, const char* name) {
    if (!value.defined()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            std::string(name) + " must be a defined tensor");
    }
}

void validate_edge_index(
    const torch::Tensor& edge_index,
    const std::array<torch::Tensor, kHeteroNodeTypes>& nodes,
    HeteroNodeType source,
    HeteroNodeType target) {
    require_defined(edge_index, "heterogeneous edge_index");
    if (edge_index.dim() != 2 || edge_index.size(0) != 2
        || edge_index.scalar_type() != torch::kInt64) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "heterogeneous edge_index must have shape [2, E] and int64 dtype");
    }
    const auto& source_x = nodes[hetero_node_index(source)];
    const auto& target_x = nodes[hetero_node_index(target)];
    if (source_x.device() != edge_index.device()
        || target_x.device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "heterogeneous nodes and edge_index must share a device");
    }
    if (edge_index.numel() == 0) return;
    const auto source_max = edge_index.select(0, 0).max().item<std::int64_t>();
    const auto target_max = edge_index.select(0, 1).max().item<std::int64_t>();
    if (source_max < 0 || target_max < 0
        || source_max >= source_x.size(0)
        || target_max >= target_x.size(0)) {
        throw PygLiteException(
            ErrorCode::invalid_index,
            "heterogeneous edge_index contains an out-of-range endpoint");
    }
}

void validate_optional_batch(
    const std::optional<torch::Tensor>& batch,
    const torch::Tensor& x,
    const char* name) {
    if (!batch.has_value()) return;
    require_defined(*batch, name);
    if (batch->dim() != 1 || batch->size(0) != x.size(0)
        || batch->scalar_type() != torch::kInt64
        || batch->device() != x.device()) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            std::string(name) + " must be an int64 vector with one entry per node");
    }
}

void validate_optional_ptr(
    const std::optional<torch::Tensor>& ptr,
    const torch::Tensor& x,
    const char* name) {
    if (!ptr.has_value()) return;
    require_defined(*ptr, name);
    if (ptr->dim() != 1 || ptr->scalar_type() != torch::kInt64
        || ptr->device() != x.device() || ptr->numel() < 1
        || ptr->select(0, 0).item<std::int64_t>() != 0
        || ptr->select(0, ptr->size(0) - 1).item<std::int64_t>() != x.size(0)) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            std::string(name) + " must be an int64 cumulative node pointer");
    }
}

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

torch::Tensor cat_or_empty(
    const std::vector<torch::Tensor>& values,
    const torch::TensorOptions& options,
    std::int64_t rows,
    std::int64_t columns) {
    if (!values.empty()) return torch::cat(values, 0);
    return torch::empty({rows, columns}, options);
}

}  // namespace

std::size_t hetero_node_index(const HeteroNodeType type) noexcept {
    return static_cast<std::size_t>(type);
}

std::size_t hetero_relation_index(const HeteroRelationId relation) noexcept {
    return static_cast<std::size_t>(relation);
}

std::optional<HeteroRelationId> hetero_relation_id(
    const HeteroNodeType source,
    const HeteroRelationKind kind,
    const HeteroNodeType target) noexcept {
    if (kind == HeteroRelationKind::connect && source == target) {
        return source == HeteroNodeType::physical
            ? HeteroRelationId::p_connect_p
            : HeteroRelationId::v_connect_v;
    }
    if (kind == HeteroRelationKind::mapping) {
        if (source == HeteroNodeType::virtual_node
            && target == HeteroNodeType::physical) {
            return HeteroRelationId::v_mapping_p;
        }
        if (source == HeteroNodeType::physical
            && target == HeteroNodeType::virtual_node) {
            return HeteroRelationId::p_mapping_v;
        }
    }
    if (kind == HeteroRelationKind::imaginary) {
        if (source == HeteroNodeType::virtual_node
            && target == HeteroNodeType::physical) {
            return HeteroRelationId::v_imaginary_p;
        }
        if (source == HeteroNodeType::physical
            && target == HeteroNodeType::virtual_node) {
            return HeteroRelationId::p_imaginary_v;
        }
    }
    return std::nullopt;
}

std::int64_t HeteroData::num_nodes(const HeteroNodeType type) const {
    const auto& value = node(type);
    require_defined(value, "heterogeneous node features");
    if (value.dim() < 1) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "heterogeneous node features must have a node dimension");
    }
    return value.size(0);
}

void HeteroData::validate() const {
    for (std::size_t index = 0U; index < kHeteroNodeTypes; ++index) {
        require_defined(x[index], "heterogeneous node features");
        if (x[index].dim() != 2) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "heterogeneous node features must have shape [N, F]");
        }
        validate_optional_batch(batch[index], x[index], "heterogeneous batch");
        validate_optional_ptr(ptr[index], x[index], "heterogeneous ptr");
    }
    for (std::size_t index = 0U; index < kHeteroRelations; ++index) {
        if (!relations[index].present()) continue;
        const auto id = static_cast<HeteroRelationId>(index);
        const auto [source, target] = endpoints(id);
        validate_edge_index(relations[index].edge_index, x, source, target);
        if (relations[index].edge_attr.has_value()) {
            const auto& edge_attr = *relations[index].edge_attr;
            require_defined(edge_attr, "heterogeneous edge_attr");
            if (edge_attr.dim() < 1
                || edge_attr.size(0) != relations[index].edge_index.size(1)
                || edge_attr.device() != relations[index].edge_index.device()) {
                throw PygLiteException(
                    ErrorCode::invalid_shape,
                    "heterogeneous edge_attr leading dimension/device mismatch");
            }
        }
    }
}

HeteroData HeteroData::clone() const {
    HeteroData result;
    for (std::size_t index = 0U; index < kHeteroNodeTypes; ++index) {
        result.x[index] = x[index].clone();
        if (batch[index].has_value()) result.batch[index] = batch[index]->clone();
        if (ptr[index].has_value()) result.ptr[index] = ptr[index]->clone();
    }
    for (std::size_t index = 0U; index < kHeteroRelations; ++index) {
        result.relations[index].edge_index = relations[index].edge_index.clone();
        if (relations[index].edge_attr.has_value()) {
            result.relations[index].edge_attr = relations[index].edge_attr->clone();
        }
    }
    return result;
}

HeteroData HeteroData::to(
    const torch::Device& device,
    const bool non_blocking,
    const bool copy) const {
    HeteroData result;
    for (std::size_t index = 0U; index < kHeteroNodeTypes; ++index) {
        result.x[index] = x[index].to(device, x[index].scalar_type(), non_blocking, copy);
        if (batch[index].has_value()) {
            result.batch[index] = batch[index]->to(
                device, batch[index]->scalar_type(), non_blocking, copy);
        }
        if (ptr[index].has_value()) {
            result.ptr[index] = ptr[index]->to(
                device, ptr[index]->scalar_type(), non_blocking, copy);
        }
    }
    for (std::size_t index = 0U; index < kHeteroRelations; ++index) {
        result.relations[index].edge_index = relations[index].edge_index.to(
            device, relations[index].edge_index.scalar_type(), non_blocking, copy);
        if (relations[index].edge_attr.has_value()) {
            result.relations[index].edge_attr = relations[index].edge_attr->to(
                device, relations[index].edge_attr->scalar_type(), non_blocking, copy);
        }
    }
    return result;
}

HeteroBatch HeteroBatch::from_data_list(
    const std::vector<HeteroData>& data_list) {
    HeteroBatch result;
    result.num_graphs = static_cast<std::int64_t>(data_list.size());
    if (data_list.empty()) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "HeteroBatch requires at least one graph");
    }
    for (const auto& item : data_list) item.validate();
    for (std::size_t type = 0U; type < kHeteroNodeTypes; ++type) {
        std::vector<torch::Tensor> features;
        std::vector<torch::Tensor> batch_values;
        std::vector<std::int64_t> pointers{0};
        features.reserve(data_list.size());
        batch_values.reserve(data_list.size());
        std::int64_t total = 0;
        result.max_num_nodes[type] = 0;
        for (std::size_t graph = 0U; graph < data_list.size(); ++graph) {
            const auto& value = data_list[graph].x[type];
            features.push_back(value);
            const auto count = value.size(0);
            total += count;
            pointers.push_back(total);
            result.max_num_nodes[type] = std::max(result.max_num_nodes[type], count);
            batch_values.push_back(torch::full(
                {count}, static_cast<std::int64_t>(graph),
                torch::TensorOptions().device(value.device()).dtype(torch::kInt64)));
        }
        result.x[type] = torch::cat(features, 0);
        result.batch[type] = torch::cat(batch_values, 0);
        result.node_ptr[type] = torch::tensor(
            pointers,
            torch::TensorOptions().device(result.x[type].device()).dtype(torch::kInt64));
        result.ptr[type] = result.node_ptr[type];
    }
    for (std::size_t relation_index = 0U;
         relation_index < kHeteroRelations; ++relation_index) {
        const auto id = static_cast<HeteroRelationId>(relation_index);
        const auto [source, target] = endpoints(id);
        std::vector<torch::Tensor> edge_indices;
        std::vector<torch::Tensor> edge_attrs;
        bool has_edge_attr = false;
        bool has_missing_attr = false;
        std::int64_t source_offset = 0;
        std::int64_t target_offset = 0;
        for (const auto& item : data_list) {
            const auto& relation = item.relations[relation_index];
            if (!relation.present()) {
                edge_indices.push_back(torch::empty(
                    {2, 0}, item.node(source).options().dtype(torch::kInt64)));
                has_missing_attr = true;
            } else {
                auto edge_index = relation.edge_index + torch::tensor(
                    {source_offset, target_offset},
                    relation.edge_index.options()).view({2, 1});
                edge_indices.push_back(std::move(edge_index));
                if (relation.edge_attr.has_value()) {
                    edge_attrs.push_back(*relation.edge_attr);
                    has_edge_attr = true;
                } else {
                    has_missing_attr = true;
                }
            }
            source_offset += item.num_nodes(source);
            target_offset += item.num_nodes(target);
        }
        if (has_edge_attr && has_missing_attr) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "a relation cannot mix present and absent edge_attr in a batch");
        }
        if (!edge_indices.empty()) {
            result.relations[relation_index].edge_index = torch::cat(edge_indices, 1);
            if (has_edge_attr) result.relations[relation_index].edge_attr = torch::cat(edge_attrs, 0);
        }
    }
    result.validate();
    return result;
}

void HeteroBatch::validate() const {
    HeteroData::validate();
    if (num_graphs <= 0) {
        throw PygLiteException(ErrorCode::inconsistent_batch, "num_graphs must be positive");
    }
    for (std::size_t type = 0U; type < kHeteroNodeTypes; ++type) {
        require_defined(node_ptr[type], "heterogeneous node_ptr");
        if (node_ptr[type].dim() != 1
            || node_ptr[type].size(0) != num_graphs + 1
            || node_ptr[type].scalar_type() != torch::kInt64
            || node_ptr[type].device() != x[type].device()) {
            throw PygLiteException(
                ErrorCode::inconsistent_batch,
                "heterogeneous node_ptr shape/device mismatch");
        }
        if (node_ptr[type].select(0, node_ptr[type].size(0) - 1)
                .item<std::int64_t>() != x[type].size(0)) {
            throw PygLiteException(
                ErrorCode::inconsistent_batch,
                "heterogeneous node_ptr does not cover node features");
        }
    }
}

HeteroBatch HeteroBatch::clone() const {
    HeteroBatch result;
    const HeteroData base = HeteroData::clone();
    static_cast<HeteroData&>(result) = base;
    for (std::size_t type = 0U; type < kHeteroNodeTypes; ++type) {
        result.node_ptr[type] = node_ptr[type].clone();
    }
    result.max_num_nodes = max_num_nodes;
    result.num_graphs = num_graphs;
    return result;
}

HeteroBatch HeteroBatch::to(
    const torch::Device& device,
    const bool non_blocking,
    const bool copy) const {
    HeteroBatch result;
    const HeteroData base = HeteroData::to(device, non_blocking, copy);
    static_cast<HeteroData&>(result) = base;
    for (std::size_t type = 0U; type < kHeteroNodeTypes; ++type) {
        result.node_ptr[type] = node_ptr[type].to(
            device, node_ptr[type].scalar_type(), non_blocking, copy);
    }
    result.max_num_nodes = max_num_nodes;
    result.num_graphs = num_graphs;
    return result;
}

}  // namespace virne::solver::learning::pyg_lite
