#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::solver::learning::pyg_lite {

enum class ErrorCode : std::uint8_t {
    invalid_tensor,
    invalid_shape,
    invalid_dtype,
    invalid_index,
    inconsistent_batch,
    unsupported_option,
};

class PygLiteException final : public std::runtime_error {
public:
    PygLiteException(ErrorCode code, std::string message);
    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

// Fixed PyG fields remain direct members. Undefined/dynamic Python attributes
// are deliberately not represented by a string map.
struct Data {
    torch::Tensor x;
    torch::Tensor edge_index;
    std::optional<torch::Tensor> edge_attr;
    std::optional<torch::Tensor> batch;
    std::optional<torch::Tensor> ptr;

    std::int64_t num_nodes() const;
    std::int64_t num_edges() const;
    void validate() const;
    void validate_indices(std::optional<std::int64_t> node_count = std::nullopt) const;
    Data clone() const;
    Data to(
        const torch::Device& device,
        bool non_blocking = false,
        bool copy = false) const;
};

struct Batch : Data {
    // Fixed collation metadata. edge_ptr makes graph extraction O(K) slices
    // instead of O(K*E) edge ownership scans.
    torch::Tensor edge_ptr;
    std::int64_t num_graphs = 0;
    std::int64_t max_num_nodes = 0;

    static Batch from_data_list(const std::vector<Data>& data_list);
    void validate() const;
    std::vector<Data> index_select(const torch::Tensor& graph_indices) const;
    Batch clone() const;
    Batch to(
        const torch::Device& device,
        bool non_blocking = false,
        bool copy = false) const;
};

enum class ScatterReduce : std::uint8_t {
    sum,
    mean,
    minimum,
    maximum,
    product,
    any,
};

// Virne's PyG paths reduce along the leading node/edge dimension. Keeping the
// native API explicit avoids a dynamic dimension/string dispatch in hot code.
torch::Tensor scatter_dim0(
    const torch::Tensor& src,
    const torch::Tensor& index,
    std::int64_t dim_size,
    ScatterReduce reduce = ScatterReduce::sum);

struct SortedEdges {
    torch::Tensor edge_index;
    std::optional<torch::Tensor> edge_attr;
    torch::Tensor permutation;
};

SortedEdges sort_edge_index(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr = std::nullopt,
    std::optional<std::int64_t> num_nodes = std::nullopt,
    bool sort_by_row = true);

enum class SelfLoopFillKind : std::uint8_t {
    one,
    scalar,
    tensor,
    add,
    mean,
    minimum,
    maximum,
    product,
};

struct SelfLoopFill {
    SelfLoopFillKind kind = SelfLoopFillKind::one;
    double scalar = 1.0;
    std::optional<torch::Tensor> value;

    static SelfLoopFill scalar_value(double value) noexcept;
    static SelfLoopFill tensor_value(torch::Tensor value);
    static SelfLoopFill reduce(SelfLoopFillKind kind);
};

struct LoopResult {
    torch::Tensor edge_index;
    std::optional<torch::Tensor> edge_attr;
};

LoopResult remove_self_loops(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr = std::nullopt);

LoopResult add_self_loops(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr = std::nullopt,
    SelfLoopFill fill = {},
    std::optional<std::int64_t> num_nodes = std::nullopt);

LoopResult add_remaining_self_loops(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr = std::nullopt,
    SelfLoopFill fill = {},
    std::optional<std::int64_t> num_nodes = std::nullopt);

torch::Tensor segment_softmax(
    const torch::Tensor& src,
    const torch::Tensor& index,
    std::optional<std::int64_t> num_nodes = std::nullopt);

struct DenseBatchResult {
    torch::Tensor values;
    torch::Tensor mask;
};

DenseBatchResult to_dense_batch(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch = std::nullopt,
    double fill_value = 0.0,
    std::optional<std::int64_t> max_num_nodes = std::nullopt,
    std::optional<std::int64_t> batch_size = std::nullopt);

// Synchronization-free hot overload: collation has already resolved the two
// dynamic dimensions and guarantees max_num_nodes is sufficient.
DenseBatchResult to_dense_batch(
    const Batch& batch,
    double fill_value = 0.0);

torch::Tensor global_add_pool(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch = std::nullopt,
    std::optional<std::int64_t> size = std::nullopt);

torch::Tensor global_mean_pool(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch = std::nullopt,
    std::optional<std::int64_t> size = std::nullopt);

torch::Tensor global_max_pool(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch = std::nullopt,
    std::optional<std::int64_t> size = std::nullopt);

struct GcnNormResult {
    torch::Tensor edge_index;
    torch::Tensor edge_weight;
};

GcnNormResult gcn_norm(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_weight,
    std::int64_t num_nodes,
    bool improved = false,
    bool add_loops = true,
    std::optional<torch::Dtype> dtype = std::nullopt);

struct GCNConvOptions {
    std::int64_t in_channels = 0;
    std::int64_t out_channels = 0;
    bool improved = false;
    bool cached = false;
    bool add_self_loops = true;
    bool normalize = true;
    bool bias = true;
};

class GCNConvImpl final : public torch::nn::Module {
public:
    explicit GCNConvImpl(GCNConvOptions options);

    void reset_parameters();
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_weight = std::nullopt);

    const GCNConvOptions& options() const noexcept { return options_; }

    torch::nn::Linear lin{nullptr};
    torch::Tensor bias;

private:
    void clear_cached_normalization() noexcept;

    GCNConvOptions options_;
    std::optional<GcnNormResult> cached_normalization_;
    std::optional<std::int64_t> cached_num_nodes_;
    torch::Tensor cached_input_edge_index_;
    std::optional<torch::Tensor> cached_input_edge_weight_;
    std::uint32_t cached_edge_index_version_ = 0;
    std::uint32_t cached_edge_weight_version_ = 0;
};
TORCH_MODULE(GCNConv);

struct GATConvOptions {
    std::int64_t in_channels = 0;
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

struct GATForwardResult {
    torch::Tensor output;
    torch::Tensor edge_index;
    torch::Tensor attention;
};

class GATConvImpl final : public torch::nn::Module {
public:
    explicit GATConvImpl(GATConvOptions options);

    void reset_parameters();
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_attr = std::nullopt);
    GATForwardResult forward_with_attention(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_attr = std::nullopt);

    const GATConvOptions& options() const noexcept { return options_; }

    torch::nn::Linear lin{nullptr};
    torch::nn::Linear lin_edge{nullptr};
    torch::nn::Linear res{nullptr};
    torch::Tensor att_src;
    torch::Tensor att_dst;
    torch::Tensor att_edge;
    torch::Tensor bias;

private:
    GATForwardResult forward_impl(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_attr);

    GATConvOptions options_;
};
TORCH_MODULE(GATConv);

}  // namespace virne::solver::learning::pyg_lite
