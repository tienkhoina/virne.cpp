#include "graph_observation_sampler.h"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <optional>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

using pyg_lite::HeteroNodeType;
using pyg_lite::HeteroRelationId;

std::vector<std::int64_t> host_i64_vector(
    const torch::Tensor& source,
    const char* name) {
    if (!source.defined() || source.dim() != 1
        || source.scalar_type() != torch::kInt64) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            std::string(name) + " must be a one-dimensional int64 tensor");
    }
    const torch::Tensor host = source.to(torch::kCPU).contiguous();
    const auto* begin = host.const_data_ptr<std::int64_t>();
    return {begin, begin + host.numel()};
}

void validate_cumulative_ptr(
    const std::vector<std::int64_t>& ptr,
    const std::int64_t graph_count,
    const std::int64_t expected_end,
    const char* name) {
    if (ptr.size() != static_cast<std::size_t>(graph_count + 1)
        || ptr.empty() || ptr.front() != 0 || ptr.back() != expected_end
        || expected_end < 0
        || !std::is_sorted(ptr.begin(), ptr.end())
        || std::any_of(ptr.begin(), ptr.end(), [expected_end](const auto value) {
               return value < 0 || value > expected_end;
           })) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            std::string(name) + " must be monotonic and cover its tensor");
    }
}

const void* tensor_identity(const torch::Tensor& value) noexcept {
    return value.defined()
        ? static_cast<const void*>(value.unsafeGetTensorImpl())
        : nullptr;
}

void validate_homogeneous_edge_partitions(
    const pyg_lite::Batch& source,
    const std::vector<std::int64_t>& node_ptr,
    const std::vector<std::int64_t>& edge_ptr) {
    if (source.edge_index.numel() == 0) return;
    const auto host = source.edge_index.to(torch::kCPU).contiguous();
    const auto* endpoints = host.const_data_ptr<std::int64_t>();
    const auto total_edges = host.size(1);
    for (std::int64_t graph = 0; graph < source.num_graphs; ++graph) {
        const auto node_begin = node_ptr[static_cast<std::size_t>(graph)];
        const auto node_end = node_ptr[static_cast<std::size_t>(graph + 1)];
        const auto edge_begin = edge_ptr[static_cast<std::size_t>(graph)];
        const auto edge_end = edge_ptr[static_cast<std::size_t>(graph + 1)];
        for (auto edge = edge_begin; edge < edge_end; ++edge) {
            const auto from = endpoints[edge];
            const auto to = endpoints[total_edges + edge];
            if (from < node_begin || from >= node_end
                || to < node_begin || to >= node_end) {
                throw GraphObservationSampleException(
                    GraphObservationSampleErrorCode::malformed_batch,
                    "homogeneous batch edges must stay inside edge_ptr graph");
            }
        }
    }
}

std::pair<HeteroNodeType, HeteroNodeType> relation_endpoints(
    const HeteroRelationId id) {
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
    throw GraphObservationSampleException(
        GraphObservationSampleErrorCode::malformed_batch,
        "unknown heterogeneous relation ID");
}

std::int64_t graph_for_node(
    const std::int64_t node,
    const std::vector<std::int64_t>& ptr) {
    if (node < 0 || ptr.empty() || node >= ptr.back()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            "heterogeneous relation endpoint is outside prepared pointers");
    }
    const auto upper = std::upper_bound(ptr.begin(), ptr.end(), node);
    if (upper == ptr.begin()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            "heterogeneous relation endpoint is outside prepared pointers");
    }
    return static_cast<std::int64_t>(upper - ptr.begin() - 1);
}

std::vector<std::int64_t> prepare_hetero_edge_ptr(
    const pyg_lite::HeteroBatch& source,
    const HeteroRelationId relation,
    const std::array<std::vector<std::int64_t>,
                     pyg_lite::kHeteroNodeTypes>& node_ptr) {
    const auto relation_index = pyg_lite::hetero_relation_index(relation);
    const auto& data = source.relations[relation_index];
    std::vector<std::int64_t> counts(
        static_cast<std::size_t>(source.num_graphs), 0);
    if (!data.present()) {
        return std::vector<std::int64_t>(
            static_cast<std::size_t>(source.num_graphs + 1), 0);
    }
    const torch::Tensor host = data.edge_index.to(torch::kCPU).contiguous();
    if (host.dim() != 2 || host.size(0) != 2
        || host.scalar_type() != torch::kInt64) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            "heterogeneous edge_index must have shape [2, E]");
    }
    const auto [source_type, target_type] = relation_endpoints(relation);
    const auto& source_ptr = node_ptr[pyg_lite::hetero_node_index(source_type)];
    const auto& target_ptr = node_ptr[pyg_lite::hetero_node_index(target_type)];
    const auto* values = host.const_data_ptr<std::int64_t>();
    const std::int64_t edges = host.size(1);
    std::int64_t previous_graph = -1;
    for (std::int64_t edge = 0; edge < edges; ++edge) {
        const auto source_graph = graph_for_node(values[edge], source_ptr);
        const auto target_graph = graph_for_node(values[edges + edge], target_ptr);
        if (source_graph != target_graph || source_graph < previous_graph) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::malformed_batch,
                "heterogeneous batch edges must be graph-local and ordered");
        }
        ++counts[static_cast<std::size_t>(source_graph)];
        previous_graph = source_graph;
    }
    std::vector<std::int64_t> result;
    result.reserve(counts.size() + 1U);
    result.push_back(0);
    for (const auto count : counts) result.push_back(result.back() + count);
    return result;
}

void require_index_count(
    const PreparedGraphIndices& indices,
    const std::int64_t graph_count) {
    if (indices.source_graph_count() != graph_count
        || indices.values().empty()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::graph_count_mismatch,
            "prepared graph indices belong to a different source batch");
    }
    for (const auto index : indices.values()) {
        if (index < 0 || index >= graph_count) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::invalid_indices,
                "prepared graph index is outside the source batch");
        }
    }
}

void validate_general_count(
    const GeneralTensorObservation& source,
    const std::int64_t graph_count) {
    const std::array<const std::optional<torch::Tensor>*, 3> fields{
        &source.curr_v_node_id, &source.action_mask, &source.v_net_size};
    std::optional<torch::Device> device;
    for (const auto* field : fields) {
        if (!field->has_value()) continue;
        const auto& value = **field;
        if (!value.defined() || value.dim() < 1 || value.size(0) != graph_count) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::graph_count_mismatch,
                "general observation field count differs from graph batch");
        }
        if (device.has_value() && value.device() != *device) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::malformed_batch,
                "general observation fields must share one source device");
        }
        device = value.device();
    }
}

torch::Tensor ptr_tensor(
    const std::vector<std::int64_t>& values,
    const torch::Device& device) {
    return torch::tensor(
        values,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
}

torch::Tensor move_if_needed(
    const torch::Tensor& value,
    const torch::Device& device) {
    if (value.device() == device) return value;
    return value.to(device, value.scalar_type(), false, false);
}

}  // namespace

GraphObservationSampleException::GraphObservationSampleException(
    const GraphObservationSampleErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

PreparedGraphIndices prepare_graph_indices(
    const std::vector<std::int64_t>& indices,
    const std::int64_t source_graph_count) {
    if (source_graph_count <= 0 || indices.empty()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::invalid_indices,
            "graph sampling requires a non-empty index lane and source batch");
    }
    PreparedGraphIndices result;
    result.source_graph_count_ = source_graph_count;
    result.values_.reserve(indices.size());
    for (auto index : indices) {
        if (index < 0) index += source_graph_count;
        if (index < 0 || index >= source_graph_count) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::invalid_indices,
                "graph sample index is outside the source batch");
        }
        result.values_.push_back(index);
    }
    return result;
}

PreparedGraphIndices prepare_graph_indices(
    const torch::Tensor& indices,
    const std::int64_t source_graph_count) {
    if (!indices.defined() || indices.dim() != 1
        || indices.scalar_type() != torch::kInt64) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::invalid_indices,
            "graph sample indices must be a one-dimensional int64 tensor");
    }
    return prepare_graph_indices(
        host_i64_vector(indices, "graph sample indices"), source_graph_count);
}

PreparedBatchObservationSampler::PreparedBatchObservationSampler(
    pyg_lite::Batch source)
    : source_(std::move(source)) {
    source_.validate();
    node_ptr_ = host_i64_vector(*source_.ptr, "homogeneous node ptr");
    edge_ptr_ = host_i64_vector(source_.edge_ptr, "homogeneous edge ptr");
    validate_cumulative_ptr(
        node_ptr_, graph_count(), source_.x.size(0), "homogeneous node ptr");
    validate_cumulative_ptr(
        edge_ptr_, graph_count(), source_.edge_index.size(1),
        "homogeneous edge ptr");
    validate_homogeneous_edge_partitions(source_, node_ptr_, edge_ptr_);
}

pyg_lite::Batch PreparedBatchObservationSampler::sample(
    const torch::Tensor& indices,
    const torch::Device& device) const {
    return sample(prepare_graph_indices(indices, graph_count()), device);
}

PreparedBatchSelection PreparedBatchObservationSampler::prepare_selection(
    const PreparedGraphIndices& indices) const {
    require_index_count(indices, graph_count());
    if (node_ptr_.size() != static_cast<std::size_t>(graph_count() + 1)
        || edge_ptr_.size() != static_cast<std::size_t>(graph_count() + 1)
        || node_ptr_.back() != source_.x.size(0)
        || edge_ptr_.back() != source_.edge_index.size(1)) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            "prepared homogeneous batch metadata changed after preparation");
    }

    std::vector<std::int64_t> node_indices;
    std::vector<std::int64_t> edge_indices;
    std::vector<std::int64_t> edge_offsets;
    std::vector<std::int64_t> batch;
    std::vector<std::int64_t> output_node_ptr{0};
    std::vector<std::int64_t> output_edge_ptr{0};
    std::size_t selected_nodes = 0U;
    std::size_t selected_edges = 0U;
    for (const auto graph_id : indices.values()) {
        const auto graph = static_cast<std::size_t>(graph_id);
        selected_nodes += static_cast<std::size_t>(
            node_ptr_[graph + 1U] - node_ptr_[graph]);
        selected_edges += static_cast<std::size_t>(
            edge_ptr_[graph + 1U] - edge_ptr_[graph]);
    }
    node_indices.reserve(selected_nodes);
    edge_indices.reserve(selected_edges);
    edge_offsets.reserve(selected_edges);
    batch.reserve(selected_nodes);
    output_node_ptr.reserve(indices.values().size() + 1U);
    output_edge_ptr.reserve(indices.values().size() + 1U);
    std::int64_t node_offset = 0;
    std::int64_t max_nodes = 0;
    for (std::size_t output_graph = 0U;
         output_graph < indices.values().size(); ++output_graph) {
        const auto graph = static_cast<std::size_t>(indices.values()[output_graph]);
        const auto node_begin = node_ptr_[graph];
        const auto node_count = node_ptr_[graph + 1U] - node_begin;
        const auto edge_begin = edge_ptr_[graph];
        const auto edge_count = edge_ptr_[graph + 1U] - edge_begin;
        for (std::int64_t node = 0; node < node_count; ++node) {
            node_indices.push_back(node_begin + node);
            batch.push_back(static_cast<std::int64_t>(output_graph));
        }
        for (std::int64_t edge = 0; edge < edge_count; ++edge) {
            edge_indices.push_back(edge_begin + edge);
            edge_offsets.push_back(node_offset - node_begin);
        }
        node_offset += node_count;
        output_node_ptr.push_back(node_offset);
        output_edge_ptr.push_back(output_edge_ptr.back() + edge_count);
        max_nodes = std::max(max_nodes, node_count);
    }

    PreparedBatchSelection result;
    result.source_graph_count = graph_count();
    result.output_graph_count = static_cast<std::int64_t>(indices.values().size());
    result.source_x_identity = tensor_identity(source_.x);
    result.source_edge_identity = tensor_identity(source_.edge_index);
    result.node_indices = ptr_tensor(node_indices, source_.x.device());
    result.edge_indices = ptr_tensor(edge_indices, source_.edge_index.device());
    result.edge_offsets = ptr_tensor(edge_offsets, source_.edge_index.device());
    result.batch = ptr_tensor(batch, source_.x.device());
    result.ptr = ptr_tensor(output_node_ptr, source_.x.device());
    result.edge_ptr = ptr_tensor(output_edge_ptr, source_.x.device());
    result.max_num_nodes = max_nodes;
    return result;
}

pyg_lite::Batch PreparedBatchObservationSampler::sample(
    const PreparedGraphIndices& indices,
    const torch::Device& device) const {
    return sample(prepare_selection(indices), device);
}

pyg_lite::Batch PreparedBatchObservationSampler::sample(
    const PreparedBatchSelection& selection,
    const torch::Device& device) const {
    if (selection.source_graph_count != graph_count()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::graph_count_mismatch,
            "homogeneous selection belongs to a different source batch");
    }
    if (selection.source_x_identity != tensor_identity(source_.x)
        || selection.source_edge_identity != tensor_identity(source_.edge_index)) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::foreign_selection,
            "homogeneous selection belongs to a different prepared batch");
    }
    pyg_lite::Batch result;
    result.x = source_.x.index_select(0, selection.node_indices);
    result.edge_index = source_.edge_index.index_select(1, selection.edge_indices);
    if (selection.edge_offsets.numel() != 0) {
        result.edge_index.add_(selection.edge_offsets.unsqueeze(0));
    }
    if (source_.edge_attr.has_value()) {
        result.edge_attr = source_.edge_attr->index_select(
            0, selection.edge_indices);
    }
    result.batch = selection.batch;
    result.ptr = selection.ptr;
    result.edge_ptr = selection.edge_ptr;
    result.num_graphs = selection.output_graph_count;
    result.max_num_nodes = selection.max_num_nodes;
    result.x = move_if_needed(result.x, device);
    result.edge_index = move_if_needed(result.edge_index, device);
    if (result.edge_attr.has_value()) {
        result.edge_attr = move_if_needed(*result.edge_attr, device);
    }
    result.batch = move_if_needed(*result.batch, device);
    result.ptr = move_if_needed(*result.ptr, device);
    result.edge_ptr = move_if_needed(result.edge_ptr, device);
    return result;
}

PreparedHeteroBatchObservationSampler::PreparedHeteroBatchObservationSampler(
    pyg_lite::HeteroBatch source)
    : source_(std::move(source)) {
    source_.validate();
    for (std::size_t type = 0U; type < pyg_lite::kHeteroNodeTypes; ++type) {
        node_ptr_[type] = host_i64_vector(
            source_.node_ptr[type], "heterogeneous node ptr");
        node_identity_[type] = tensor_identity(source_.x[type]);
        validate_cumulative_ptr(
            node_ptr_[type], graph_count(), source_.x[type].size(0),
            "heterogeneous node ptr");
    }
    for (std::size_t relation = 0U;
         relation < pyg_lite::kHeteroRelations; ++relation) {
        edge_ptr_[relation] = prepare_hetero_edge_ptr(
            source_, static_cast<HeteroRelationId>(relation), node_ptr_);
    }
}

pyg_lite::HeteroBatch PreparedHeteroBatchObservationSampler::sample(
    const torch::Tensor& indices,
    const torch::Device& device) const {
    return sample(prepare_graph_indices(indices, graph_count()), device);
}

namespace {

PreparedHeteroRelationSelection build_relation_selection(
    const pyg_lite::HeteroRelationData& relation,
    const HeteroRelationId relation_id,
    const std::array<std::vector<std::int64_t>,
                     pyg_lite::kHeteroNodeTypes>& node_ptr,
    const std::array<const void*, pyg_lite::kHeteroNodeTypes>&
        node_identity,
    const std::vector<std::int64_t>& edge_ptr,
    const PreparedGraphIndices& indices) {
    if (!relation.present()
        || edge_ptr.size()
            != static_cast<std::size_t>(indices.source_graph_count() + 1)
        || edge_ptr.back() != relation.edge_index.size(1)) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::malformed_batch,
            "prepared heterogeneous relation metadata changed");
    }
    const auto [source_type, target_type] = relation_endpoints(relation_id);
    const auto source_slot = pyg_lite::hetero_node_index(source_type);
    const auto target_slot = pyg_lite::hetero_node_index(target_type);
    std::vector<std::int64_t> edge_indices;
    std::vector<std::int64_t> source_offsets;
    std::vector<std::int64_t> target_offsets;
    std::size_t selected_edges = 0U;
    for (const auto graph_id : indices.values()) {
        const auto graph = static_cast<std::size_t>(graph_id);
        selected_edges += static_cast<std::size_t>(
            edge_ptr[graph + 1U] - edge_ptr[graph]);
    }
    edge_indices.reserve(selected_edges);
    source_offsets.reserve(selected_edges);
    target_offsets.reserve(selected_edges);
    std::int64_t output_source_offset = 0;
    std::int64_t output_target_offset = 0;
    for (const auto graph_id : indices.values()) {
        const auto graph = static_cast<std::size_t>(graph_id);
        const auto edge_begin = edge_ptr[graph];
        const auto edge_count = edge_ptr[graph + 1U] - edge_begin;
        const auto source_begin = node_ptr[source_slot][graph];
        const auto target_begin = node_ptr[target_slot][graph];
        const auto source_delta = output_source_offset - source_begin;
        const auto target_delta = output_target_offset - target_begin;
        for (std::int64_t edge = 0; edge < edge_count; ++edge) {
            edge_indices.push_back(edge_begin + edge);
            source_offsets.push_back(source_delta);
            target_offsets.push_back(target_delta);
        }
        output_source_offset += node_ptr[source_slot][graph + 1U] - source_begin;
        output_target_offset += node_ptr[target_slot][graph + 1U] - target_begin;
    }
    std::vector<std::int64_t> offsets;
    offsets.reserve(source_offsets.size() + target_offsets.size());
    offsets.insert(offsets.end(), source_offsets.begin(), source_offsets.end());
    offsets.insert(offsets.end(), target_offsets.begin(), target_offsets.end());
    PreparedHeteroRelationSelection result;
    result.source_edge_identity = tensor_identity(relation.edge_index);
    result.source_x_identity = node_identity;
    result.relation_id = relation_id;
    result.edge_indices = ptr_tensor(edge_indices, relation.edge_index.device());
    result.endpoint_offsets = ptr_tensor(offsets, relation.edge_index.device())
                                  .reshape({2, static_cast<std::int64_t>(
                                                   edge_indices.size())});
    return result;
}

torch::Device general_source_device(
    const GeneralTensorObservation& source) {
    if (source.curr_v_node_id.has_value()) return source.curr_v_node_id->device();
    if (source.action_mask.has_value()) return source.action_mask->device();
    if (source.v_net_size.has_value()) return source.v_net_size->device();
    return torch::Device(torch::kCPU);
}

GeneralTensorObservation sample_general_prepared(
    const GeneralTensorObservation& source,
    const torch::Tensor& indices,
    const torch::Device& device) {
    GeneralTensorObservation result;
    const auto select = [&](const std::optional<torch::Tensor>& value)
        -> std::optional<torch::Tensor> {
        if (!value.has_value()) return std::nullopt;
        return move_if_needed(value->index_select(0, indices), device);
    };
    result.curr_v_node_id = select(source.curr_v_node_id);
    result.action_mask = select(source.action_mask);
    result.v_net_size = select(source.v_net_size);
    return result;
}

}  // namespace

PreparedHeteroBatchSelection
PreparedHeteroBatchObservationSampler::prepare_selection(
    const PreparedGraphIndices& indices) const {
    require_index_count(indices, graph_count());
    PreparedHeteroBatchSelection result;
    result.source_graph_count = graph_count();
    result.output_graph_count = static_cast<std::int64_t>(indices.values().size());

    for (std::size_t type = 0U; type < pyg_lite::kHeteroNodeTypes; ++type) {
        result.source_x_identity[type] = tensor_identity(source_.x[type]);
        const auto& prepared_ptr = node_ptr_[type];
        if (prepared_ptr.size() != static_cast<std::size_t>(graph_count() + 1)
            || prepared_ptr.back() != source_.x[type].size(0)) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::malformed_batch,
                "prepared heterogeneous node metadata changed");
        }
        std::vector<std::int64_t> node_indices;
        std::vector<std::int64_t> batch;
        std::vector<std::int64_t> output_ptr{0};
        std::size_t selected_nodes = 0U;
        for (const auto graph_id : indices.values()) {
            const auto graph = static_cast<std::size_t>(graph_id);
            selected_nodes += static_cast<std::size_t>(
                prepared_ptr[graph + 1U] - prepared_ptr[graph]);
        }
        node_indices.reserve(selected_nodes);
        batch.reserve(selected_nodes);
        output_ptr.reserve(indices.values().size() + 1U);
        std::int64_t total = 0;
        std::int64_t max_nodes = 0;
        for (std::size_t output_graph = 0U;
             output_graph < indices.values().size(); ++output_graph) {
            const auto graph = static_cast<std::size_t>(indices.values()[output_graph]);
            const auto begin = prepared_ptr[graph];
            const auto count = prepared_ptr[graph + 1U] - begin;
            for (std::int64_t node = 0; node < count; ++node) {
                node_indices.push_back(begin + node);
                batch.push_back(static_cast<std::int64_t>(output_graph));
            }
            total += count;
            output_ptr.push_back(total);
            max_nodes = std::max(max_nodes, count);
        }
        result.node_indices[type] = ptr_tensor(
            node_indices, source_.x[type].device());
        result.batch[type] = ptr_tensor(batch, source_.x[type].device());
        result.node_ptr[type] = ptr_tensor(output_ptr, source_.x[type].device());
        result.max_num_nodes[type] = max_nodes;
    }

    for (std::size_t relation_index = 0U;
         relation_index < pyg_lite::kHeteroRelations; ++relation_index) {
        const auto relation = static_cast<HeteroRelationId>(relation_index);
        const auto& source_relation = source_.relations[relation_index];
        result.source_edge_identity[relation_index] =
            tensor_identity(source_relation.edge_index);
        if (!source_relation.present()) continue;
        result.relations[relation_index] = build_relation_selection(
            source_relation, relation, node_ptr_, node_identity_,
            edge_ptr_[relation_index], indices);
    }
    return result;
}

PreparedHeteroRelationSelection
PreparedHeteroBatchObservationSampler::prepare_relation_selection(
    const pyg_lite::HeteroRelationData& relation,
    const HeteroRelationId relation_id,
    const PreparedGraphIndices& indices) const {
    require_index_count(indices, graph_count());
    pyg_lite::HeteroBatch temporary = source_;
    temporary.relations[pyg_lite::hetero_relation_index(relation_id)] = relation;
    const auto prepared_ptr = prepare_hetero_edge_ptr(
        temporary, relation_id, node_ptr_);
    return build_relation_selection(
        relation, relation_id, node_ptr_, node_identity_, prepared_ptr,
        indices);
}

pyg_lite::HeteroRelationData
PreparedHeteroBatchObservationSampler::sample_relation(
    const pyg_lite::HeteroRelationData& relation,
    const HeteroRelationId relation_id,
    const PreparedHeteroRelationSelection& selection,
    const torch::Device& device) const {
    if (selection.source_edge_identity != tensor_identity(relation.edge_index)
        || selection.relation_id != relation_id) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::foreign_selection,
            "heterogeneous relation selection belongs to another relation");
    }
    for (std::size_t type = 0U; type < pyg_lite::kHeteroNodeTypes; ++type) {
        if (selection.source_x_identity[type]
            != tensor_identity(source_.x[type])) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::foreign_selection,
                "heterogeneous relation selection belongs to another node batch");
        }
    }
    pyg_lite::HeteroRelationData result;
    result.edge_index = relation.edge_index.index_select(
        1, selection.edge_indices);
    if (selection.endpoint_offsets.numel() != 0) {
        result.edge_index.add_(selection.endpoint_offsets);
    }
    if (relation.edge_attr.has_value()) {
        result.edge_attr = relation.edge_attr->index_select(
            0, selection.edge_indices);
    }
    result.edge_index = move_if_needed(result.edge_index, device);
    if (result.edge_attr.has_value()) {
        result.edge_attr = move_if_needed(*result.edge_attr, device);
    }
    return result;
}

pyg_lite::HeteroBatch PreparedHeteroBatchObservationSampler::sample(
    const PreparedGraphIndices& indices,
    const torch::Device& device) const {
    return sample(prepare_selection(indices), device);
}

pyg_lite::HeteroBatch PreparedHeteroBatchObservationSampler::sample(
    const PreparedHeteroBatchSelection& selection,
    const torch::Device& device) const {
    if (selection.source_graph_count != graph_count()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::graph_count_mismatch,
            "heterogeneous selection belongs to a different source batch");
    }
    for (std::size_t type = 0U; type < pyg_lite::kHeteroNodeTypes; ++type) {
        if (selection.source_x_identity[type]
            != tensor_identity(source_.x[type])) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::foreign_selection,
                "heterogeneous selection belongs to a different node batch");
        }
    }
    for (std::size_t relation = 0U;
         relation < pyg_lite::kHeteroRelations; ++relation) {
        if (selection.source_edge_identity[relation]
            != tensor_identity(source_.relations[relation].edge_index)) {
            throw GraphObservationSampleException(
                GraphObservationSampleErrorCode::foreign_selection,
                "heterogeneous selection belongs to a different relation batch");
        }
    }
    pyg_lite::HeteroBatch result;
    result.num_graphs = selection.output_graph_count;
    for (std::size_t type = 0U; type < pyg_lite::kHeteroNodeTypes; ++type) {
        result.x[type] = move_if_needed(source_.x[type].index_select(
            0, selection.node_indices[type]), device);
        result.batch[type] = move_if_needed(selection.batch[type], device);
        result.node_ptr[type] = move_if_needed(
            selection.node_ptr[type], device);
        result.ptr[type] = result.node_ptr[type];
        result.max_num_nodes[type] = selection.max_num_nodes[type];
    }
    for (std::size_t relation = 0U;
         relation < pyg_lite::kHeteroRelations; ++relation) {
        if (!source_.relations[relation].present()) continue;
        result.relations[relation] = sample_relation(
            source_.relations[relation],
            static_cast<HeteroRelationId>(relation),
            selection.relations[relation],
            device);
    }
    return result;
}

PreparedDualGraphObservationSampler::PreparedDualGraphObservationSampler(
    DualGraphTensorObservation source)
    : p_net_(std::move(source.p_net)),
      v_net_(std::move(source.v_net)),
      general_(std::move(source.general)) {
    if (p_net_.graph_count() != v_net_.graph_count()) {
        throw GraphObservationSampleException(
            GraphObservationSampleErrorCode::graph_count_mismatch,
            "dual graph batches must contain the same number of graphs");
    }
    validate_general_count(general_, graph_count());
}

DualGraphTensorObservation PreparedDualGraphObservationSampler::sample(
    const torch::Tensor& indices,
    const torch::Device& device) const {
    return sample(prepare_graph_indices(indices, graph_count()), device);
}

PreparedDualGraphSelection
PreparedDualGraphObservationSampler::prepare_selection(
    const PreparedGraphIndices& indices) const {
    require_index_count(indices, graph_count());
    PreparedDualGraphSelection result;
    result.graph_indices = indices;
    result.p_net = p_net_.prepare_selection(indices);
    result.v_net = v_net_.prepare_selection(indices);
    result.general_indices = ptr_tensor(
        indices.values(), general_source_device(general_));
    return result;
}

DualGraphTensorObservation PreparedDualGraphObservationSampler::sample(
    const PreparedGraphIndices& indices,
    const torch::Device& device) const {
    return sample(prepare_selection(indices), device);
}

DualGraphTensorObservation PreparedDualGraphObservationSampler::sample(
    const PreparedDualGraphSelection& selection,
    const torch::Device& device) const {
    require_index_count(selection.graph_indices, graph_count());
    DualGraphTensorObservation result{
        p_net_.sample(selection.p_net, device),
        v_net_.sample(selection.v_net, device),
        sample_general_prepared(general_, selection.general_indices, device)};
    return result;
}

pyg_lite::HeteroBatch HeteroGraphTensorObservation::view(
    const HeteroObservationViewId id) const {
    pyg_lite::HeteroBatch result = base;
    if (id == HeteroObservationViewId::virtual_augmented) {
        result.relations[pyg_lite::hetero_relation_index(
            HeteroRelationId::v_connect_v)] = virtual_connect_augmented;
    } else if (id == HeteroObservationViewId::physical_augmented) {
        result.relations[pyg_lite::hetero_relation_index(
            HeteroRelationId::p_connect_p)] = physical_connect_augmented;
    }
    return result;
}

PreparedHeteroGraphObservationSampler::PreparedHeteroGraphObservationSampler(
    HeteroGraphTensorObservation source)
    : base_(source.base),
      virtual_connect_augmented_(source.virtual_connect_augmented),
      physical_connect_augmented_(source.physical_connect_augmented),
      general_(std::move(source.general)) {
    source.view(HeteroObservationViewId::virtual_augmented).validate();
    source.view(HeteroObservationViewId::physical_augmented).validate();
    for (std::size_t type = 0U; type < pyg_lite::kHeteroNodeTypes; ++type) {
        node_ptr_[type] = host_i64_vector(
            source.base.node_ptr[type], "heterogeneous observation node ptr");
        node_identity_[type] = tensor_identity(source.base.x[type]);
    }
    const auto virtual_view = source.view(HeteroObservationViewId::virtual_augmented);
    const auto physical_view = source.view(HeteroObservationViewId::physical_augmented);
    virtual_edge_ptr_ = prepare_hetero_edge_ptr(
        virtual_view, HeteroRelationId::v_connect_v, node_ptr_);
    physical_edge_ptr_ = prepare_hetero_edge_ptr(
        physical_view, HeteroRelationId::p_connect_p, node_ptr_);
    validate_general_count(general_, graph_count());
}

HeteroGraphTensorObservation PreparedHeteroGraphObservationSampler::sample(
    const torch::Tensor& indices,
    const torch::Device& device) const {
    return sample(prepare_graph_indices(indices, graph_count()), device);
}

PreparedHeteroGraphSelection
PreparedHeteroGraphObservationSampler::prepare_selection(
    const PreparedGraphIndices& indices) const {
    require_index_count(indices, graph_count());
    PreparedHeteroGraphSelection result;
    result.graph_indices = indices;
    result.base = base_.prepare_selection(indices);
    result.virtual_connect_augmented = build_relation_selection(
        virtual_connect_augmented_, HeteroRelationId::v_connect_v,
        node_ptr_, node_identity_, virtual_edge_ptr_, indices);
    result.physical_connect_augmented = build_relation_selection(
        physical_connect_augmented_, HeteroRelationId::p_connect_p,
        node_ptr_, node_identity_, physical_edge_ptr_, indices);
    result.general_indices = ptr_tensor(
        indices.values(), general_source_device(general_));
    return result;
}

HeteroGraphTensorObservation PreparedHeteroGraphObservationSampler::sample(
    const PreparedGraphIndices& indices,
    const torch::Device& device) const {
    return sample(prepare_selection(indices), device);
}

HeteroGraphTensorObservation PreparedHeteroGraphObservationSampler::sample(
    const PreparedHeteroGraphSelection& selection,
    const torch::Device& device) const {
    require_index_count(selection.graph_indices, graph_count());
    HeteroGraphTensorObservation result{
        base_.sample(selection.base, device),
        base_.sample_relation(
            virtual_connect_augmented_,
            HeteroRelationId::v_connect_v,
            selection.virtual_connect_augmented, device),
        base_.sample_relation(
            physical_connect_augmented_,
            HeteroRelationId::p_connect_p,
            selection.physical_connect_augmented, device),
        sample_general_prepared(general_, selection.general_indices, device)};
    return result;
}

}  // namespace virne::solver::learning::rl_core
