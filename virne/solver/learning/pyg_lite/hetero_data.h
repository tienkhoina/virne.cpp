#pragma once

#include "pyg_lite.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace virne::solver::learning::pyg_lite {

// The active Virne hetero graph has two fixed node types and three fixed
// relation names.  A compact relation ID replaces PyG's tuple/string maps in
// every forward loop.
enum class HeteroNodeType : std::uint8_t {
    physical = 0,
    virtual_node = 1,
};

enum class HeteroRelationKind : std::uint8_t {
    connect = 0,
    mapping = 1,
    imaginary = 2,
};

enum class HeteroRelationId : std::uint8_t {
    p_connect_p = 0,
    v_connect_v = 1,
    v_mapping_p = 2,
    p_mapping_v = 3,
    v_imaginary_p = 4,
    p_imaginary_v = 5,
};

constexpr std::size_t kHeteroNodeTypes = 2U;
constexpr std::size_t kHeteroRelations = 6U;

std::size_t hetero_node_index(HeteroNodeType type) noexcept;
std::size_t hetero_relation_index(HeteroRelationId relation) noexcept;

std::optional<HeteroRelationId> hetero_relation_id(
    HeteroNodeType source,
    HeteroRelationKind kind,
    HeteroNodeType target) noexcept;

struct HeteroRelationData {
    torch::Tensor edge_index;
    std::optional<torch::Tensor> edge_attr;

    bool present() const noexcept { return edge_index.defined(); }
};

struct HeteroData {
    std::array<torch::Tensor, kHeteroNodeTypes> x{};
    std::array<std::optional<torch::Tensor>, kHeteroNodeTypes> batch{};
    std::array<std::optional<torch::Tensor>, kHeteroNodeTypes> ptr{};
    std::array<HeteroRelationData, kHeteroRelations> relations{};

    torch::Tensor& node(HeteroNodeType type) noexcept {
        return x[hetero_node_index(type)];
    }
    const torch::Tensor& node(HeteroNodeType type) const noexcept {
        return x[hetero_node_index(type)];
    }
    HeteroRelationData& relation(HeteroRelationId id) noexcept {
        return relations[hetero_relation_index(id)];
    }
    const HeteroRelationData& relation(HeteroRelationId id) const noexcept {
        return relations[hetero_relation_index(id)];
    }

    std::int64_t num_nodes(HeteroNodeType type) const;
    void validate() const;
    HeteroData clone() const;
    HeteroData to(
        const torch::Device& device,
        bool non_blocking = false,
        bool copy = false) const;
};

struct HeteroBatch : HeteroData {
    std::array<torch::Tensor, kHeteroNodeTypes> node_ptr{};
    std::array<std::int64_t, kHeteroNodeTypes> max_num_nodes{};
    std::int64_t num_graphs = 0;

    static HeteroBatch from_data_list(
        const std::vector<HeteroData>& data_list);
    void validate() const;
    HeteroBatch clone() const;
    HeteroBatch to(
        const torch::Device& device,
        bool non_blocking = false,
        bool copy = false) const;
};

}  // namespace virne::solver::learning::pyg_lite
