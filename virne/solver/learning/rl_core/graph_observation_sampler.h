#pragma once

#include "tensor_convertor.h"
#include "../pyg_lite/hetero_data.h"

#include <torch/types.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::solver::learning::rl_core {

enum class GraphObservationSampleErrorCode : std::uint8_t {
    invalid_indices,
    graph_count_mismatch,
    foreign_selection,
    malformed_batch,
};

class GraphObservationSampleException final : public std::runtime_error {
public:
    GraphObservationSampleException(
        GraphObservationSampleErrorCode code,
        std::string message);

    GraphObservationSampleErrorCode code() const noexcept { return code_; }

private:
    GraphObservationSampleErrorCode code_;
};

// Dynamic Tensor/list indices resolve once into this typed, normalized lane.
// Every graph sampler thereafter consumes direct integer slots only.
class PreparedGraphIndices final {
public:
    std::int64_t source_graph_count() const noexcept {
        return source_graph_count_;
    }

    const std::vector<std::int64_t>& values() const noexcept {
        return values_;
    }

private:
    friend PreparedGraphIndices prepare_graph_indices(
        const torch::Tensor&, std::int64_t);
    friend PreparedGraphIndices prepare_graph_indices(
        const std::vector<std::int64_t>&, std::int64_t);

    std::int64_t source_graph_count_ = 0;
    std::vector<std::int64_t> values_;
};

PreparedGraphIndices prepare_graph_indices(
    const torch::Tensor& indices,
    std::int64_t source_graph_count);

PreparedGraphIndices prepare_graph_indices(
    const std::vector<std::int64_t>& indices,
    std::int64_t source_graph_count);

// Selection plans are cold, device-resident gather metadata. Build them in a
// caller worker when minibatch IDs are generated; hot sample() then performs
// only index_select, fixed relation loops, and optional device transfer.
struct PreparedBatchSelection {
    std::int64_t source_graph_count = 0;
    std::int64_t output_graph_count = 0;
    std::shared_ptr<const std::uint8_t> owner_identity;
    torch::Tensor node_indices;
    torch::Tensor edge_indices;
    torch::Tensor edge_offsets;
    torch::Tensor batch;
    torch::Tensor ptr;
    torch::Tensor edge_ptr;
    std::int64_t max_num_nodes = 0;
};

struct PreparedHeteroRelationSelection {
    std::shared_ptr<const std::uint8_t> owner_identity;
    const void* source_edge_identity = nullptr;
    pyg_lite::HeteroRelationId relation_id =
        pyg_lite::HeteroRelationId::p_connect_p;
    torch::Tensor edge_indices;
    torch::Tensor endpoint_offsets;
};

struct PreparedHeteroBatchSelection {
    std::int64_t source_graph_count = 0;
    std::int64_t output_graph_count = 0;
    std::shared_ptr<const std::uint8_t> owner_identity;
    std::array<torch::Tensor, pyg_lite::kHeteroNodeTypes> node_indices{};
    std::array<torch::Tensor, pyg_lite::kHeteroNodeTypes> batch{};
    std::array<torch::Tensor, pyg_lite::kHeteroNodeTypes> node_ptr{};
    std::array<std::int64_t, pyg_lite::kHeteroNodeTypes> max_num_nodes{};
    std::array<PreparedHeteroRelationSelection,
               pyg_lite::kHeteroRelations> relations{};
};

class PreparedBatchObservationSampler final {
public:
    explicit PreparedBatchObservationSampler(pyg_lite::Batch source);

    std::int64_t graph_count() const noexcept { return source_.num_graphs; }

    PreparedBatchSelection prepare_selection(
        const PreparedGraphIndices& indices) const;

    pyg_lite::Batch sample(
        const PreparedBatchSelection& selection,
        const torch::Device& device = torch::kCPU) const;

    pyg_lite::Batch sample(
        const PreparedGraphIndices& indices,
        const torch::Device& device = torch::kCPU) const;

    pyg_lite::Batch sample(
        const torch::Tensor& indices,
        const torch::Device& device = torch::kCPU) const;

private:
    pyg_lite::Batch source_;
    std::shared_ptr<const std::uint8_t> owner_identity_ =
        std::make_shared<const std::uint8_t>(0);
    std::vector<std::int64_t> node_ptr_;
    std::vector<std::int64_t> edge_ptr_;
};

class PreparedHeteroBatchObservationSampler final {
public:
    explicit PreparedHeteroBatchObservationSampler(pyg_lite::HeteroBatch source);

    std::int64_t graph_count() const noexcept { return source_.num_graphs; }

    PreparedHeteroBatchSelection prepare_selection(
        const PreparedGraphIndices& indices) const;

    PreparedHeteroRelationSelection prepare_relation_selection(
        const pyg_lite::HeteroRelationData& relation,
        pyg_lite::HeteroRelationId relation_id,
        const PreparedGraphIndices& indices) const;

    pyg_lite::HeteroBatch sample(
        const PreparedHeteroBatchSelection& selection,
        const torch::Device& device = torch::kCPU) const;

    pyg_lite::HeteroRelationData sample_relation(
        const pyg_lite::HeteroRelationData& relation,
        pyg_lite::HeteroRelationId relation_id,
        const PreparedHeteroRelationSelection& selection,
        const torch::Device& device = torch::kCPU) const;

    pyg_lite::HeteroBatch sample(
        const PreparedGraphIndices& indices,
        const torch::Device& device = torch::kCPU) const;

    pyg_lite::HeteroBatch sample(
        const torch::Tensor& indices,
        const torch::Device& device = torch::kCPU) const;

private:
    friend class PreparedHeteroGraphObservationSampler;

    pyg_lite::HeteroBatch source_;
    std::shared_ptr<const std::uint8_t> owner_identity_ =
        std::make_shared<const std::uint8_t>(0);
    std::array<std::vector<std::int64_t>, pyg_lite::kHeteroNodeTypes>
        node_ptr_{};
    std::array<std::vector<std::int64_t>, pyg_lite::kHeteroRelations>
        edge_ptr_{};
};

struct DualGraphTensorObservation {
    pyg_lite::Batch p_net;
    pyg_lite::Batch v_net;
    GeneralTensorObservation general;
};

enum class HeteroObservationViewId : std::uint8_t {
    base = 0,
    virtual_augmented = 1,
    physical_augmented = 2,
};

struct HeteroGraphTensorObservation {
    pyg_lite::HeteroBatch base;
    pyg_lite::HeteroRelationData virtual_connect_augmented;
    pyg_lite::HeteroRelationData physical_connect_augmented;
    GeneralTensorObservation general;

    pyg_lite::HeteroBatch view(HeteroObservationViewId id) const;
};

struct PreparedDualGraphSelection {
    PreparedGraphIndices graph_indices;
    PreparedBatchSelection p_net;
    PreparedBatchSelection v_net;
    torch::Tensor general_indices;
};

struct PreparedHeteroGraphSelection {
    PreparedGraphIndices graph_indices;
    PreparedHeteroBatchSelection base;
    PreparedHeteroRelationSelection virtual_connect_augmented;
    PreparedHeteroRelationSelection physical_connect_augmented;
    torch::Tensor general_indices;
};

class PreparedDualGraphObservationSampler final {
public:
    explicit PreparedDualGraphObservationSampler(
        DualGraphTensorObservation source);

    std::int64_t graph_count() const noexcept {
        return p_net_.graph_count();
    }

    PreparedDualGraphSelection prepare_selection(
        const PreparedGraphIndices& indices) const;

    DualGraphTensorObservation sample(
        const PreparedDualGraphSelection& selection,
        const torch::Device& device = torch::kCPU) const;

    DualGraphTensorObservation sample(
        const PreparedGraphIndices& indices,
        const torch::Device& device = torch::kCPU) const;

    DualGraphTensorObservation sample(
        const torch::Tensor& indices,
        const torch::Device& device = torch::kCPU) const;

private:
    PreparedBatchObservationSampler p_net_;
    PreparedBatchObservationSampler v_net_;
    GeneralTensorObservation general_;
};

class PreparedHeteroGraphObservationSampler final {
public:
    explicit PreparedHeteroGraphObservationSampler(
        HeteroGraphTensorObservation source);

    std::int64_t graph_count() const noexcept {
        return base_.graph_count();
    }

    PreparedHeteroGraphSelection prepare_selection(
        const PreparedGraphIndices& indices) const;

    HeteroGraphTensorObservation sample(
        const PreparedHeteroGraphSelection& selection,
        const torch::Device& device = torch::kCPU) const;

    HeteroGraphTensorObservation sample(
        const PreparedGraphIndices& indices,
        const torch::Device& device = torch::kCPU) const;

    HeteroGraphTensorObservation sample(
        const torch::Tensor& indices,
        const torch::Device& device = torch::kCPU) const;

private:
    PreparedHeteroBatchObservationSampler base_;
    pyg_lite::HeteroRelationData virtual_connect_augmented_;
    pyg_lite::HeteroRelationData physical_connect_augmented_;
    std::array<std::vector<std::int64_t>, pyg_lite::kHeteroNodeTypes>
        node_ptr_{};
    std::vector<std::int64_t> virtual_edge_ptr_;
    std::vector<std::int64_t> physical_edge_ptr_;
    GeneralTensorObservation general_;
};

}  // namespace virne::solver::learning::rl_core
