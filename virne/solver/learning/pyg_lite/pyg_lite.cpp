#include "pyg_lite.h"

#include <ATen/ops/argsort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
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

void validate_edge_index(const torch::Tensor& edge_index) {
    require_defined(edge_index, "edge_index");
    if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_index must have shape [2, E]");
    }
    if (edge_index.scalar_type() != torch::kInt64) {
        throw PygLiteException(
            ErrorCode::invalid_dtype,
            "edge_index must use int64 indices");
    }
}

void validate_edge_attr(
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& edge_attr) {
    if (!edge_attr.has_value()) {
        return;
    }
    require_defined(*edge_attr, "edge_attr");
    if (edge_attr->dim() < 1 || edge_attr->size(0) != edge_index.size(1)) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_attr leading dimension must equal the edge count");
    }
    if (edge_attr->device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "edge_index and edge_attr must be on the same device");
    }
}

std::int64_t infer_num_nodes(
    const torch::Tensor& edge_index,
    const std::optional<std::int64_t> explicit_count) {
    if (explicit_count.has_value()) {
        if (*explicit_count < 0) {
            throw PygLiteException(
                ErrorCode::invalid_index,
                "num_nodes must be non-negative");
        }
        return *explicit_count;
    }
    if (edge_index.numel() == 0) {
        return 0;
    }
    return edge_index.max().item<std::int64_t>() + 1;
}

torch::Tensor broadcast_dim0(
    const torch::Tensor& index,
    const torch::Tensor& src) {
    std::vector<std::int64_t> shape(
        static_cast<std::size_t>(src.dim()), 1);
    shape.front() = -1;
    return index.view(shape).expand_as(src);
}

std::vector<std::int64_t> reduced_shape(
    const torch::Tensor& src,
    const std::int64_t dim_size) {
    std::vector<std::int64_t> shape(src.sizes().begin(), src.sizes().end());
    shape.front() = dim_size;
    return shape;
}

ScatterReduce loop_reduce(const SelfLoopFillKind kind) {
    switch (kind) {
    case SelfLoopFillKind::add:
        return ScatterReduce::sum;
    case SelfLoopFillKind::mean:
        return ScatterReduce::mean;
    case SelfLoopFillKind::minimum:
        return ScatterReduce::minimum;
    case SelfLoopFillKind::maximum:
        return ScatterReduce::maximum;
    case SelfLoopFillKind::product:
        return ScatterReduce::product;
    default:
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "self-loop fill is not a reduction");
    }
}

torch::Tensor compute_loop_attr(
    const torch::Tensor& edge_index,
    const torch::Tensor& edge_attr,
    const std::int64_t num_nodes,
    const SelfLoopFill& fill) {
    std::vector<std::int64_t> shape(edge_attr.sizes().begin(), edge_attr.sizes().end());
    shape.front() = num_nodes;

    switch (fill.kind) {
    case SelfLoopFillKind::one:
        return torch::ones(shape, edge_attr.options());
    case SelfLoopFillKind::scalar:
        return torch::full(shape, fill.scalar, edge_attr.options());
    case SelfLoopFillKind::tensor: {
        if (!fill.value.has_value() || !fill.value->defined()) {
            throw PygLiteException(
                ErrorCode::invalid_tensor,
                "tensor self-loop fill requires a defined value");
        }
        torch::Tensor value = fill.value->to(
            edge_attr.device(), edge_attr.scalar_type(), false, false);
        if (value.dim() != edge_attr.dim()) {
            value = value.unsqueeze(0);
        }
        return value.expand(shape).contiguous();
    }
    case SelfLoopFillKind::add:
    case SelfLoopFillKind::mean:
    case SelfLoopFillKind::minimum:
    case SelfLoopFillKind::maximum:
    case SelfLoopFillKind::product:
        return scatter_dim0(
            edge_attr,
            edge_index.select(0, 1),
            num_nodes,
            loop_reduce(fill.kind));
    }
    throw PygLiteException(
        ErrorCode::unsupported_option,
        "invalid self-loop fill kind");
}

torch::Tensor move_pool_dimension_to_front(
    const torch::Tensor& x,
    std::int64_t& original_dimension) {
    original_dimension = x.dim() == 1 ? 0 : x.dim() - 2;
    if (original_dimension == 0) {
        return x;
    }
    return torch::movedim(x, original_dimension, 0);
}

torch::Tensor restore_pool_dimension(
    const torch::Tensor& x,
    const std::int64_t original_dimension) {
    if (original_dimension == 0) {
        return x;
    }
    return torch::movedim(x, 0, original_dimension);
}

std::int64_t pool_size(
    const torch::Tensor& batch,
    const std::optional<std::int64_t> explicit_size) {
    if (explicit_size.has_value()) {
        if (*explicit_size < 0) {
            throw PygLiteException(
                ErrorCode::invalid_index,
                "pool size must be non-negative");
        }
        return *explicit_size;
    }
    return batch.numel() == 0
        ? 0
        : batch.max().item<std::int64_t>() + 1;
}

void glorot_last_two(torch::Tensor value) {
    if (!value.defined()) {
        return;
    }
    const double bound = std::sqrt(
        6.0 / static_cast<double>(
            value.size(-2) + value.size(-1)));
    torch::NoGradGuard guard;
    value.uniform_(-bound, bound);
}

std::uint32_t tensor_version(const torch::Tensor& value) {
    const c10::VariableVersion& counter =
        value.unsafeGetTensorImpl()->version_counter();
    return counter.enabled() ? counter.current_version() : 0U;
}

torch::Tensor argsort_edge_key(const torch::Tensor& key) {
#if defined(__GLIBCXX__) && !defined(_WIN32)
    if (key.device().is_cpu()
        && key.scalar_type() == torch::kInt64
        && key.is_contiguous()) {
        const std::int64_t count = key.numel();
        if (count < 2) {
            return torch::arange(count, key.options());
        }

        // Unique graph-edge keys use a stable LSD radix pass. It is linear,
        // allocation-bounded, and materially faster than generic argsort for
        // Virne's sparse CPU graphs. If duplicate keys are found afterwards,
        // fall back to libstdc++ introsort: PyTorch's unstable CPU argsort
        // uses the same equal-key permutation, which preserves exact parity.
        torch::Tensor current = torch::empty({count}, key.options());
        torch::Tensor scratch = torch::empty({count}, key.options());
        auto* current_data = current.data_ptr<std::int64_t>();
        auto* scratch_data = scratch.data_ptr<std::int64_t>();
        const auto* key_data = key.data_ptr<std::int64_t>();
        std::int64_t maximum = 0;
        for (std::int64_t index = 0; index < count; ++index) {
            if (key_data[index] < 0) {
                return at::argsort(key, false, 0, false);
            }
            maximum = std::max(maximum, key_data[index]);
            current_data[index] = index;
        }

        constexpr std::uint32_t radix_bits = 11;
        constexpr std::size_t radix_size = std::size_t{1} << radix_bits;
        constexpr std::uint64_t radix_mask = radix_size - 1U;
        std::uint32_t shift = 0;
        do {
            std::array<std::int64_t, radix_size> offsets{};
            for (std::int64_t position = 0; position < count; ++position) {
                const std::uint64_t value = static_cast<std::uint64_t>(
                    key_data[current_data[position]]);
                ++offsets[(value >> shift) & radix_mask];
            }
            std::int64_t prefix = 0;
            for (std::int64_t& offset : offsets) {
                const std::int64_t bucket_size = offset;
                offset = prefix;
                prefix += bucket_size;
            }
            for (std::int64_t position = 0; position < count; ++position) {
                const std::int64_t original = current_data[position];
                const std::uint64_t value = static_cast<std::uint64_t>(
                    key_data[original]);
                scratch_data[offsets[(value >> shift) & radix_mask]++] = original;
            }
            std::swap(current, scratch);
            current_data = current.data_ptr<std::int64_t>();
            scratch_data = scratch.data_ptr<std::int64_t>();
            shift += radix_bits;
        } while (shift < 64U
                 && (static_cast<std::uint64_t>(maximum) >> shift) != 0U);

        bool has_duplicate = false;
        for (std::int64_t position = 1; position < count; ++position) {
            if (key_data[current_data[position - 1]]
                == key_data[current_data[position]]) {
                has_duplicate = true;
                break;
            }
        }
        if (has_duplicate) {
            std::iota(current_data, current_data + count, std::int64_t{0});
            std::sort(
                current_data,
                current_data + count,
                [key_data](const std::int64_t left, const std::int64_t right) {
                    return key_data[left] < key_data[right];
                });
        }
        return current;
    }
#endif
    // CUDA, Windows/MSVC and other backends retain ATen's device-native
    // implementation and its backend-specific duplicate ordering.
    return at::argsort(key, false, 0, false);
}

}  // namespace

PygLiteException::PygLiteException(ErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

std::int64_t Data::num_nodes() const {
    require_defined(x, "x");
    if (x.dim() < 1) {
        throw PygLiteException(ErrorCode::invalid_shape, "x must have a node dimension");
    }
    return x.size(0);
}

std::int64_t Data::num_edges() const {
    validate_edge_index(edge_index);
    return edge_index.size(1);
}

void Data::validate() const {
    require_defined(x, "x");
    if (x.dim() < 1) {
        throw PygLiteException(ErrorCode::invalid_shape, "x must have rank at least one");
    }
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_attr);
    if (x.device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "x and edge_index must be on the same device");
    }
    if (batch.has_value()) {
        if (!batch->defined() || batch->dim() != 1
            || batch->size(0) != x.size(0)
            || batch->scalar_type() != torch::kInt64
            || batch->device() != x.device()) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "batch must be an int64 vector with one entry per node");
        }
    }
    if (ptr.has_value()) {
        if (!ptr->defined() || ptr->dim() != 1
            || ptr->scalar_type() != torch::kInt64
            || ptr->device() != x.device()) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "ptr must be a one-dimensional int64 tensor");
        }
    }
}

void Data::validate_indices(
    const std::optional<std::int64_t> node_count) const {
    validate();
    const std::int64_t nodes = node_count.value_or(num_nodes());
    if (nodes < 0) {
        throw PygLiteException(
            ErrorCode::invalid_index,
            "node_count must be non-negative");
    }
    if (edge_index.numel() == 0) return;

    // Deliberately a cold graph-boundary check. Calling item() here prevents
    // invalid CUDA indices from surfacing later as a device-side assert that
    // poisons the process context.
    const std::int64_t minimum = edge_index.min().item<std::int64_t>();
    const std::int64_t maximum = edge_index.max().item<std::int64_t>();
    if (minimum < 0 || maximum >= nodes) {
        throw PygLiteException(
            ErrorCode::invalid_index,
            "edge_index contains a node outside [0, node_count)");
    }
}

Data Data::clone() const {
    validate();
    Data result{};
    result.x = x.clone();
    result.edge_index = edge_index.clone();
    if (edge_attr.has_value()) result.edge_attr = edge_attr->clone();
    if (batch.has_value()) result.batch = batch->clone();
    if (ptr.has_value()) result.ptr = ptr->clone();
    return result;
}

Data Data::to(
    const torch::Device& device,
    const bool non_blocking,
    const bool copy) const {
    validate();
    auto move = [&](const torch::Tensor& value) {
        return value.to(
            device, value.scalar_type(), non_blocking, copy);
    };
    Data result{};
    result.x = move(x);
    result.edge_index = move(edge_index);
    if (edge_attr.has_value()) result.edge_attr = move(*edge_attr);
    if (batch.has_value()) result.batch = move(*batch);
    if (ptr.has_value()) result.ptr = move(*ptr);
    return result;
}

Batch Batch::from_data_list(const std::vector<Data>& data_list) {
    if (data_list.empty()) {
        throw PygLiteException(
            ErrorCode::inconsistent_batch,
            "Batch.from_data_list requires at least one graph");
    }

    const bool with_edge_attr = data_list.front().edge_attr.has_value();
    std::vector<torch::Tensor> xs;
    std::vector<torch::Tensor> edge_indices;
    std::vector<torch::Tensor> edge_attrs;
    std::vector<torch::Tensor> batches;
    xs.reserve(data_list.size());
    edge_indices.reserve(data_list.size());
    edge_attrs.reserve(data_list.size());
    batches.reserve(data_list.size());
    std::vector<std::int64_t> ptr_values;
    ptr_values.reserve(data_list.size() + 1U);
    ptr_values.push_back(0);
    std::vector<std::int64_t> edge_ptr_values;
    edge_ptr_values.reserve(data_list.size() + 1U);
    edge_ptr_values.push_back(0);

    std::int64_t node_offset = 0;
    std::int64_t edge_offset = 0;
    std::int64_t max_num_nodes = 0;
    for (std::size_t graph_id = 0; graph_id < data_list.size(); ++graph_id) {
        const Data& data = data_list[graph_id];
        data.validate();
        if (data.edge_attr.has_value() != with_edge_attr) {
            throw PygLiteException(
                ErrorCode::inconsistent_batch,
                "edge_attr must be present for all graphs or none");
        }
        if (data.x.device() != data_list.front().x.device()
            || data.x.scalar_type() != data_list.front().x.scalar_type()) {
            throw PygLiteException(
                ErrorCode::inconsistent_batch,
                "batched node tensors must share device and dtype");
        }
        xs.push_back(data.x);
        edge_indices.push_back(data.edge_index + node_offset);
        if (with_edge_attr) edge_attrs.push_back(*data.edge_attr);
        batches.push_back(torch::full(
            {data.num_nodes()},
            static_cast<std::int64_t>(graph_id),
            data.edge_index.options()));
        node_offset += data.num_nodes();
        edge_offset += data.num_edges();
        max_num_nodes = std::max(max_num_nodes, data.num_nodes());
        ptr_values.push_back(node_offset);
        edge_ptr_values.push_back(edge_offset);
    }

    Batch result{};
    result.x = torch::cat(xs, 0);
    result.edge_index = torch::cat(edge_indices, 1);
    if (with_edge_attr) result.edge_attr = torch::cat(edge_attrs, 0);
    result.batch = torch::cat(batches, 0);
    result.ptr = torch::tensor(
        ptr_values,
        torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(result.x.device()));
    result.edge_ptr = torch::tensor(
        edge_ptr_values,
        torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(result.x.device()));
    result.num_graphs = static_cast<std::int64_t>(data_list.size());
    result.max_num_nodes = max_num_nodes;
    return result;
}

void Batch::validate() const {
    Data::validate();
    if (num_graphs < 0 || max_num_nodes < 0
        || !ptr.has_value()
        || !edge_ptr.defined()
        || edge_ptr.dim() != 1
        || edge_ptr.scalar_type() != torch::kInt64
        || edge_ptr.device() != x.device()
        || ptr->numel() != num_graphs + 1
        || edge_ptr.numel() != num_graphs + 1) {
        throw PygLiteException(
            ErrorCode::inconsistent_batch,
            "Batch fixed metadata is incomplete or inconsistent");
    }
}

std::vector<Data> Batch::index_select(const torch::Tensor& graph_indices) const {
    validate();
    if (!batch.has_value() || !ptr.has_value() || !edge_ptr.defined()) {
        throw PygLiteException(
            ErrorCode::inconsistent_batch,
            "Batch.index_select requires batch, ptr and edge_ptr fields");
    }
    if (!graph_indices.defined() || graph_indices.dim() != 1
        || graph_indices.scalar_type() != torch::kInt64) {
        throw PygLiteException(
            ErrorCode::invalid_index,
            "graph_indices must be a one-dimensional int64 tensor");
    }

    const torch::Tensor selected = graph_indices.to(torch::kCPU).contiguous();
    const torch::Tensor host_ptr = ptr->to(torch::kCPU).contiguous();
    const torch::Tensor host_edge_ptr = edge_ptr.to(torch::kCPU).contiguous();
    if (num_graphs < 0
        || host_ptr.numel() != num_graphs + 1
        || host_edge_ptr.numel() != num_graphs + 1) {
        throw PygLiteException(
            ErrorCode::inconsistent_batch,
            "Batch ptr metadata does not match num_graphs");
    }
    const auto* selected_data = selected.data_ptr<std::int64_t>();
    const auto* ptr_data = host_ptr.data_ptr<std::int64_t>();
    const auto* edge_ptr_data = host_edge_ptr.data_ptr<std::int64_t>();
    if (ptr_data[0] != 0 || ptr_data[num_graphs] != x.size(0)
        || edge_ptr_data[0] != 0
        || edge_ptr_data[num_graphs] != edge_index.size(1)) {
        throw PygLiteException(
            ErrorCode::inconsistent_batch,
            "Batch ptr endpoints do not match tensor sizes");
    }
    for (std::int64_t graph_id = 0; graph_id < num_graphs; ++graph_id) {
        if (ptr_data[graph_id] > ptr_data[graph_id + 1]
            || edge_ptr_data[graph_id] > edge_ptr_data[graph_id + 1]) {
            throw PygLiteException(
                ErrorCode::inconsistent_batch,
                "Batch ptr metadata must be monotonic");
        }
    }
    std::vector<Data> result;
    result.reserve(static_cast<std::size_t>(selected.numel()));
    for (std::int64_t position = 0; position < selected.numel(); ++position) {
        const std::int64_t graph_id = selected_data[position];
        if (graph_id < 0 || graph_id >= num_graphs) {
            throw PygLiteException(
                ErrorCode::invalid_index,
                "graph index is out of range");
        }
        const std::int64_t begin = ptr_data[graph_id];
        const std::int64_t end = ptr_data[graph_id + 1];
        const std::int64_t edge_begin = edge_ptr_data[graph_id];
        const std::int64_t edge_end = edge_ptr_data[graph_id + 1];

        Data data{};
        data.x = x.slice(0, begin, end);
        data.edge_index = edge_index.slice(1, edge_begin, edge_end) - begin;
        if (edge_attr.has_value()) {
            data.edge_attr = edge_attr->slice(0, edge_begin, edge_end);
        }
        result.push_back(std::move(data));
    }
    return result;
}

Batch Batch::to(
    const torch::Device& device,
    const bool non_blocking,
    const bool copy) const {
    validate();
    const Data moved = Data::to(device, non_blocking, copy);
    Batch result{};
    result.x = moved.x;
    result.edge_index = moved.edge_index;
    result.edge_attr = moved.edge_attr;
    result.batch = moved.batch;
    result.ptr = moved.ptr;
    result.edge_ptr = edge_ptr.to(
        device, edge_ptr.scalar_type(), non_blocking, copy);
    result.num_graphs = num_graphs;
    result.max_num_nodes = max_num_nodes;
    return result;
}

Batch Batch::clone() const {
    validate();
    const Data copied = Data::clone();
    Batch result{};
    result.x = copied.x;
    result.edge_index = copied.edge_index;
    result.edge_attr = copied.edge_attr;
    result.batch = copied.batch;
    result.ptr = copied.ptr;
    result.edge_ptr = edge_ptr.clone();
    result.num_graphs = num_graphs;
    result.max_num_nodes = max_num_nodes;
    return result;
}

torch::Tensor scatter_dim0(
    const torch::Tensor& src,
    const torch::Tensor& index,
    const std::int64_t dim_size,
    const ScatterReduce reduce) {
    require_defined(src, "src");
    require_defined(index, "index");
    if (src.dim() < 1 || index.dim() != 1 || index.size(0) != src.size(0)) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "scatter index must contain one entry per source row");
    }
    if (index.scalar_type() != torch::kInt64 || index.device() != src.device()) {
        throw PygLiteException(
            ErrorCode::invalid_dtype,
            "scatter index must be int64 on the source device");
    }
    if (dim_size < 0) {
        throw PygLiteException(ErrorCode::invalid_index, "scatter dim_size is negative");
    }

    const std::vector<std::int64_t> shape = reduced_shape(src, dim_size);
    switch (reduce) {
    case ScatterReduce::any: {
        torch::Tensor result = torch::zeros(shape, src.options());
        return result.scatter_(0, broadcast_dim0(index, src), src);
    }
    case ScatterReduce::sum: {
        torch::Tensor result = torch::zeros(shape, src.options());
        // This API is intentionally fixed to dim 0. index_add_ avoids
        // materializing the broadcast [E,...] index for rank-2 GCN/pooling
        // lanes. For rank >=3, ATen's scatter kernel is faster on the common
        // [E, heads, channels] GAT layout, so retain the specialized branch.
        if (src.dim() <= 2) return result.index_add_(0, index, src);
        return result.scatter_add_(0, broadcast_dim0(index, src), src);
    }
    case ScatterReduce::mean: {
        torch::Tensor count = torch::zeros({dim_size}, src.options());
        count.index_add_(
            0, index, torch::ones({src.size(0)}, src.options()));
        count.clamp_min_(1);
        torch::Tensor result = torch::zeros(shape, src.options());
        if (src.dim() <= 2) {
            result.index_add_(0, index, src);
        } else {
            result.scatter_add_(0, broadcast_dim0(index, src), src);
        }
        std::vector<std::int64_t> count_shape(
            static_cast<std::size_t>(src.dim()), 1);
        count_shape.front() = dim_size;
        const torch::Tensor divisor = count.view(count_shape);
        if (result.is_floating_point() || result.is_complex()) {
            return result / divisor;
        }
        return result.div(divisor, c10::string_view("floor"));
    }
    case ScatterReduce::minimum: {
        torch::Tensor result = torch::zeros(shape, src.options());
        return result.scatter_reduce_(
            0, broadcast_dim0(index, src), src, "amin", false);
    }
    case ScatterReduce::maximum: {
        torch::Tensor result = torch::zeros(shape, src.options());
        return result.scatter_reduce_(
            0, broadcast_dim0(index, src), src, "amax", false);
    }
    case ScatterReduce::product: {
        torch::Tensor result = torch::ones(shape, src.options());
        return result.scatter_reduce_(
            0, broadcast_dim0(index, src), src, "prod", true);
    }
    }
    throw PygLiteException(ErrorCode::unsupported_option, "invalid scatter reduction");
}

SortedEdges sort_edge_index(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr,
    const std::optional<std::int64_t> num_nodes,
    const bool sort_by_row) {
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_attr);
    const std::int64_t nodes = infer_num_nodes(edge_index, num_nodes);
    const std::int64_t primary = sort_by_row ? 0 : 1;
    const std::int64_t secondary = 1 - primary;

    torch::Tensor permutation;
    if (nodes <= 0
        || nodes <= std::numeric_limits<std::int64_t>::max() / std::max<std::int64_t>(1, nodes)) {
        const torch::Tensor key =
            edge_index.select(0, primary) * nodes
            + edge_index.select(0, secondary);
        // PyG 2.7 index_sort defaults to stable=false. This matters for
        // duplicate edges because edge attributes must follow its exact
        // permutation rather than insertion order.
        permutation = argsort_edge_key(key);
    } else {
        const torch::Tensor secondary_order = at::argsort(
            edge_index.select(0, secondary), true, 0, false);
        const torch::Tensor primary_values = edge_index
            .select(0, primary)
            .index_select(0, secondary_order);
        permutation = secondary_order.index_select(
            0, at::argsort(primary_values, true, 0, false));
    }

    SortedEdges result{};
    result.edge_index = edge_index.index_select(1, permutation);
    if (edge_attr.has_value()) {
        result.edge_attr = edge_attr->index_select(0, permutation);
    }
    result.permutation = permutation;
    return result;
}

SelfLoopFill SelfLoopFill::scalar_value(const double value) noexcept {
    SelfLoopFill result{};
    result.kind = SelfLoopFillKind::scalar;
    result.scalar = value;
    return result;
}

SelfLoopFill SelfLoopFill::tensor_value(torch::Tensor value) {
    SelfLoopFill result{};
    result.kind = SelfLoopFillKind::tensor;
    result.value = std::move(value);
    return result;
}

SelfLoopFill SelfLoopFill::reduce(const SelfLoopFillKind kind) {
    (void)loop_reduce(kind);
    SelfLoopFill result{};
    result.kind = kind;
    return result;
}

LoopResult remove_self_loops(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr) {
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_attr);
    const torch::Tensor mask = edge_index.select(0, 0).ne(edge_index.select(0, 1));
    LoopResult result{};
    result.edge_index = edge_index.index({torch::indexing::Slice(), mask});
    if (edge_attr.has_value()) result.edge_attr = edge_attr->index({mask});
    return result;
}

LoopResult add_self_loops(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr,
    const SelfLoopFill fill,
    const std::optional<std::int64_t> num_nodes) {
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_attr);
    const std::int64_t nodes = infer_num_nodes(edge_index, num_nodes);
    const torch::Tensor loop_nodes = torch::arange(
        nodes, edge_index.options());
    const torch::Tensor loop_index = loop_nodes.unsqueeze(0).repeat({2, 1});
    LoopResult result{};
    result.edge_index = torch::cat({edge_index, loop_index}, 1);
    if (edge_attr.has_value()) {
        result.edge_attr = torch::cat(
            {*edge_attr, compute_loop_attr(edge_index, *edge_attr, nodes, fill)},
            0);
    }
    return result;
}

LoopResult add_remaining_self_loops(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr,
    const SelfLoopFill fill,
    const std::optional<std::int64_t> num_nodes) {
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_attr);
    const std::int64_t nodes = infer_num_nodes(edge_index, num_nodes);
    const torch::Tensor row = edge_index.select(0, 0);
    const torch::Tensor col = edge_index.select(0, 1);
    const torch::Tensor mask = row.ne(col);
    const torch::Tensor loop_nodes = torch::arange(nodes, edge_index.options());
    const torch::Tensor loop_index = loop_nodes.unsqueeze(0).repeat({2, 1});

    LoopResult result{};
    result.edge_index = torch::cat(
        {edge_index.index({torch::indexing::Slice(), mask}), loop_index}, 1);
    if (edge_attr.has_value()) {
        torch::Tensor loop_attr = compute_loop_attr(
            edge_index, *edge_attr, nodes, fill);
        const torch::Tensor inverse = mask.logical_not();
        // Empty index_copy_ is a no-op. Avoid inverse.any().item(), which
        // would otherwise force a device synchronization on every CUDA call.
        loop_attr.index_copy_(
            0,
            row.index({inverse}),
            edge_attr->index({inverse}));
        result.edge_attr = torch::cat({edge_attr->index({mask}), loop_attr}, 0);
    }
    return result;
}

torch::Tensor segment_softmax(
    const torch::Tensor& src,
    const torch::Tensor& index,
    const std::optional<std::int64_t> num_nodes) {
    const std::int64_t nodes = num_nodes.has_value()
        ? *num_nodes
        : (index.numel() == 0 ? 0 : index.max().item<std::int64_t>() + 1);
    const torch::Tensor maximum = scatter_dim0(
        src.detach(), index, nodes, ScatterReduce::maximum);
    torch::Tensor output = (
        src - maximum.index_select(0, index)).exp();
    const torch::Tensor denominator = scatter_dim0(
        output, index, nodes, ScatterReduce::sum) + 1e-16;
    return output / denominator.index_select(0, index);
}

namespace {

DenseBatchResult to_dense_batch_impl(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch,
    const double fill_value,
    std::optional<std::int64_t> max_num_nodes,
    std::optional<std::int64_t> batch_size,
    const bool max_num_nodes_is_sufficient) {
    require_defined(x, "x");
    if (x.dim() < 1) {
        throw PygLiteException(ErrorCode::invalid_shape, "x must have rank at least one");
    }
    if (!batch.has_value() && !max_num_nodes.has_value()) {
        return {
            x.unsqueeze(0),
            torch::ones({1, x.size(0)},
                         x.options().dtype(torch::kBool))};
    }
    if (!batch.has_value()) {
        batch = torch::zeros(
            {x.size(0)}, x.options().dtype(torch::kInt64));
    }
    if (!batch->defined() || batch->dim() != 1
        || batch->size(0) != x.size(0)
        || batch->scalar_type() != torch::kInt64
        || batch->device() != x.device()) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "batch must be an int64 vector with one value per node");
    }
    if (!batch_size.has_value()) {
        if (batch->numel() == 0) {
            throw PygLiteException(
                ErrorCode::invalid_index,
                "cannot infer batch_size from an empty batch");
        }
        batch_size = batch->max().item<std::int64_t>() + 1;
    }
    if (*batch_size < 0) {
        throw PygLiteException(ErrorCode::invalid_index, "batch_size is negative");
    }
    const torch::Tensor counts = scatter_dim0(
        torch::ones({x.size(0)}, batch->options()),
        *batch,
        *batch_size,
        ScatterReduce::sum);
    const torch::Tensor cumulative = torch::cat(
        {torch::zeros({1}, batch->options()), counts.cumsum(0)}, 0);
    // Supplying max_num_nodes is the production fast path: it keeps shape
    // resolution entirely on-device and avoids counts.max().item().
    const bool infer_max_num_nodes = !max_num_nodes.has_value();
    if (infer_max_num_nodes) {
        max_num_nodes = counts.numel() == 0
            ? 0
            : counts.max().item<std::int64_t>();
    }
    if (*max_num_nodes < 0) {
        throw PygLiteException(ErrorCode::invalid_index, "max_num_nodes is negative");
    }

    torch::Tensor local = torch::arange(
        batch->size(0), batch->options())
        - cumulative.index_select(0, *batch);
    torch::Tensor indices = local + (*batch * *max_num_nodes);
    torch::Tensor filtered_x = x;
    if (!infer_max_num_nodes && !max_num_nodes_is_sufficient) {
        // An explicit bound may truncate over-sized graphs, so retain PyG's
        // exact filtering semantics. With an inferred bound, observed_max is
        // by definition sufficient and the zero-copy path is used.
        const torch::Tensor keep = local.lt(*max_num_nodes);
        filtered_x = x.index({keep});
        indices = indices.index({keep});
    }

    std::vector<std::int64_t> flat_shape(x.sizes().begin(), x.sizes().end());
    flat_shape.front() = *batch_size * *max_num_nodes;
    torch::Tensor values = torch::full(flat_shape, fill_value, x.options());
    values.index_copy_(0, indices, filtered_x);
    std::vector<std::int64_t> dense_shape;
    dense_shape.reserve(static_cast<std::size_t>(x.dim()) + 1U);
    dense_shape.push_back(*batch_size);
    dense_shape.push_back(*max_num_nodes);
    dense_shape.insert(dense_shape.end(), x.sizes().begin() + 1, x.sizes().end());
    values = values.view(dense_shape);

    torch::Tensor mask = torch::zeros(
        {*batch_size * *max_num_nodes},
        x.options().dtype(torch::kBool));
    mask.index_fill_(0, indices, true);
    return {values, mask.view({*batch_size, *max_num_nodes})};
}

}  // namespace

DenseBatchResult to_dense_batch(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch,
    const double fill_value,
    std::optional<std::int64_t> max_num_nodes,
    std::optional<std::int64_t> batch_size) {
    return to_dense_batch_impl(
        x,
        std::move(batch),
        fill_value,
        max_num_nodes,
        batch_size,
        false);
}

DenseBatchResult to_dense_batch(
    const Batch& batch,
    const double fill_value) {
    batch.validate();
    if (!batch.batch.has_value() || batch.num_graphs < 0
        || batch.max_num_nodes < 0) {
        throw PygLiteException(
            ErrorCode::inconsistent_batch,
            "dense Batch metadata is incomplete");
    }
    return to_dense_batch_impl(
        batch.x,
        batch.batch,
        fill_value,
        batch.max_num_nodes,
        batch.num_graphs,
        true);
}

torch::Tensor global_add_pool(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch,
    const std::optional<std::int64_t> size) {
    require_defined(x, "x");
    const std::int64_t dimension = x.dim() == 1 ? 0 : x.dim() - 2;
    if (!batch.has_value()) return x.sum(dimension, x.dim() <= 2);
    std::int64_t original = 0;
    const torch::Tensor front = move_pool_dimension_to_front(x, original);
    return restore_pool_dimension(
        scatter_dim0(front, *batch, pool_size(*batch, size), ScatterReduce::sum),
        original);
}

torch::Tensor global_mean_pool(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch,
    const std::optional<std::int64_t> size) {
    require_defined(x, "x");
    const std::int64_t dimension = x.dim() == 1 ? 0 : x.dim() - 2;
    if (!batch.has_value()) return x.mean(dimension, x.dim() <= 2);
    std::int64_t original = 0;
    const torch::Tensor front = move_pool_dimension_to_front(x, original);
    return restore_pool_dimension(
        scatter_dim0(front, *batch, pool_size(*batch, size), ScatterReduce::mean),
        original);
}

torch::Tensor global_max_pool(
    const torch::Tensor& x,
    std::optional<torch::Tensor> batch,
    const std::optional<std::int64_t> size) {
    require_defined(x, "x");
    const std::int64_t dimension = x.dim() == 1 ? 0 : x.dim() - 2;
    if (!batch.has_value()) {
        return std::get<0>(x.max(dimension, x.dim() <= 2));
    }
    std::int64_t original = 0;
    const torch::Tensor front = move_pool_dimension_to_front(x, original);
    return restore_pool_dimension(
        scatter_dim0(front, *batch, pool_size(*batch, size), ScatterReduce::maximum),
        original);
}

GcnNormResult gcn_norm(
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_weight,
    const std::int64_t num_nodes,
    const bool improved,
    const bool add_loops,
    const std::optional<torch::Dtype> dtype) {
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_weight);
    if (add_loops) {
        LoopResult loops = add_remaining_self_loops(
            edge_index,
            edge_weight,
            SelfLoopFill::scalar_value(improved ? 2.0 : 1.0),
            num_nodes);
        edge_weight = std::move(loops.edge_attr);
        if (!edge_weight.has_value()) {
            edge_weight = torch::ones(
                {loops.edge_index.size(1)},
                torch::TensorOptions()
                    .device(edge_index.device())
                    .dtype(dtype.value_or(torch::kFloat32)));
        }
        const torch::Tensor row = loops.edge_index.select(0, 0);
        const torch::Tensor col = loops.edge_index.select(0, 1);
        torch::Tensor degree = scatter_dim0(
            *edge_weight, col, num_nodes, ScatterReduce::sum);
        torch::Tensor inverse = degree.pow(-0.5);
        inverse.masked_fill_(torch::isinf(inverse), 0.0);
        return {
            loops.edge_index,
            inverse.index_select(0, row) * *edge_weight
                * inverse.index_select(0, col)};
    }
    if (!edge_weight.has_value()) {
        edge_weight = torch::ones(
            {edge_index.size(1)},
            torch::TensorOptions()
                .device(edge_index.device())
                .dtype(dtype.value_or(torch::kFloat32)));
    }
    const torch::Tensor row = edge_index.select(0, 0);
    const torch::Tensor col = edge_index.select(0, 1);
    torch::Tensor degree = scatter_dim0(
        *edge_weight, col, num_nodes, ScatterReduce::sum);
    torch::Tensor inverse = degree.pow(-0.5);
    inverse.masked_fill_(torch::isinf(inverse), 0.0);
    return {
        edge_index,
        inverse.index_select(0, row) * *edge_weight
            * inverse.index_select(0, col)};
}

GCNConvImpl::GCNConvImpl(GCNConvOptions options)
    : options_(options) {
    if (options_.in_channels <= 0 || options_.out_channels <= 0) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "GCNConv channel counts must be positive");
    }
    if (options_.add_self_loops && !options_.normalize) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "GCNConv cannot add self-loops without normalization");
    }
    lin = register_module(
        "lin",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.in_channels, options_.out_channels).bias(false)));
    if (options_.bias) {
        bias = register_parameter(
            "bias", torch::empty({options_.out_channels}));
    }
    reset_parameters();
}

void GCNConvImpl::reset_parameters() {
    torch::nn::init::xavier_uniform_(lin->weight);
    if (bias.defined()) {
        torch::NoGradGuard guard;
        bias.zero_();
    }
    clear_cached_normalization();
}

void GCNConvImpl::clear_cached_normalization() noexcept {
    cached_normalization_.reset();
    cached_num_nodes_.reset();
    cached_input_edge_index_ = torch::Tensor();
    cached_input_edge_weight_.reset();
    cached_edge_index_version_ = 0;
    cached_edge_weight_version_ = 0;
}

void GCNConvImpl::to(
    torch::Device device,
    torch::Dtype dtype,
    const bool non_blocking) {
    torch::nn::Module::to(device, dtype, non_blocking);
    clear_cached_normalization();
}

void GCNConvImpl::to(const torch::Dtype dtype, const bool non_blocking) {
    torch::nn::Module::to(dtype, non_blocking);
    clear_cached_normalization();
}

void GCNConvImpl::to(torch::Device device, const bool non_blocking) {
    torch::nn::Module::to(device, non_blocking);
    clear_cached_normalization();
}

torch::Tensor GCNConvImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_weight) {
    require_defined(x, "x");
    if (x.dim() != 2 || x.size(1) != options_.in_channels) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "GCNConv x must have shape [N, in_channels]");
    }
    validate_edge_index(edge_index);
    validate_edge_attr(edge_index, edge_weight);
    if (x.device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "GCNConv x and edge_index must be on the same device");
    }
    const std::optional<torch::Tensor> input_edge_weight = edge_weight;
    const std::uint32_t input_edge_index_version = tensor_version(edge_index);
    const std::uint32_t input_edge_weight_version = edge_weight.has_value()
        ? tensor_version(*edge_weight)
        : 0U;
    torch::Tensor normalized_index = edge_index;
    if (options_.normalize) {
        const bool cache_usable = cached_normalization_.has_value()
            && cached_num_nodes_.has_value()
            && *cached_num_nodes_ == x.size(0)
            && cached_normalization_->edge_index.defined()
            && cached_normalization_->edge_weight.defined()
            && cached_normalization_->edge_index.device() == x.device()
            && cached_normalization_->edge_weight.device() == x.device()
            && cached_normalization_->edge_weight.scalar_type()
                == x.scalar_type()
            && cached_normalization_->edge_index.size(0) == 2;
        const bool same_input = cache_usable
            && cached_input_edge_index_.defined()
            && cached_input_edge_index_.unsafeGetTensorImpl()
                == edge_index.unsafeGetTensorImpl()
            && cached_edge_index_version_ == input_edge_index_version
            && cached_input_edge_weight_.has_value()
                == input_edge_weight.has_value()
            && (!input_edge_weight.has_value()
                || (cached_input_edge_weight_->unsafeGetTensorImpl()
                        == input_edge_weight->unsafeGetTensorImpl()
                    && cached_edge_weight_version_
                        == input_edge_weight_version));
        if (same_input) {
            normalized_index = cached_normalization_->edge_index;
            edge_weight = cached_normalization_->edge_weight;
        } else {
            if (cached_normalization_.has_value()) {
                // Device/dtype, graph identity or in-place version changed.
                // Never feed stale topology into the next forward.
                clear_cached_normalization();
            }
            GcnNormResult normalized = gcn_norm(
                edge_index,
                std::move(edge_weight),
                x.size(0),
                options_.improved,
                options_.add_self_loops,
                x.scalar_type());
            normalized_index = normalized.edge_index;
            edge_weight = normalized.edge_weight;
            const bool cache_allowed = options_.cached
                && (!input_edge_weight.has_value()
                    || !input_edge_weight->requires_grad());
            if (cache_allowed) {
                cached_normalization_ = normalized;
                cached_num_nodes_ = x.size(0);
                cached_input_edge_index_ = edge_index;
                cached_input_edge_weight_ = input_edge_weight;
                cached_edge_index_version_ = input_edge_index_version;
                cached_edge_weight_version_ = input_edge_weight_version;
            }
        }
    }
    const torch::Tensor projected = lin->forward(x);
    const torch::Tensor source = normalized_index.select(0, 0);
    const torch::Tensor target = normalized_index.select(0, 1);
    torch::Tensor messages = projected.index_select(0, source);
    if (edge_weight.has_value()) {
        messages = messages * edge_weight->view({-1, 1});
    }
    torch::Tensor output = scatter_dim0(
        messages, target, x.size(0), ScatterReduce::sum);
    if (bias.defined()) output = output + bias;
    return output;
}

GATConvImpl::GATConvImpl(GATConvOptions options)
    : options_(std::move(options)) {
    if (options_.in_channels <= 0 || options_.out_channels <= 0
        || options_.heads <= 0) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "GATConv channel/head counts must be positive");
    }
    if (options_.dropout < 0.0 || options_.dropout > 1.0) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "GATConv dropout must lie in [0, 1]");
    }
    const std::int64_t projected = options_.heads * options_.out_channels;
    lin = register_module(
        "lin",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.in_channels, projected).bias(false)));
    att_src = register_parameter(
        "att_src", torch::empty({1, options_.heads, options_.out_channels}));
    att_dst = register_parameter(
        "att_dst", torch::empty({1, options_.heads, options_.out_channels}));
    if (options_.edge_dim.has_value()) {
        if (*options_.edge_dim <= 0) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "GATConv edge_dim must be positive");
        }
        lin_edge = register_module(
            "lin_edge",
            torch::nn::Linear(torch::nn::LinearOptions(
                *options_.edge_dim, projected).bias(false)));
        att_edge = register_parameter(
            "att_edge", torch::empty({1, options_.heads, options_.out_channels}));
    }
    const std::int64_t total = options_.concat
        ? projected
        : options_.out_channels;
    if (options_.residual) {
        res = register_module(
            "res",
            torch::nn::Linear(torch::nn::LinearOptions(
                options_.in_channels, total).bias(false)));
    }
    if (options_.bias) {
        bias = register_parameter("bias", torch::empty({total}));
    }
    reset_parameters();
}

void GATConvImpl::reset_parameters() {
    torch::nn::init::xavier_uniform_(lin->weight);
    if (lin_edge) torch::nn::init::xavier_uniform_(lin_edge->weight);
    if (res) torch::nn::init::xavier_uniform_(res->weight);
    glorot_last_two(att_src);
    glorot_last_two(att_dst);
    glorot_last_two(att_edge);
    if (bias.defined()) {
        torch::NoGradGuard guard;
        bias.zero_();
    }
    clear_cached_topology();
}

void GATConvImpl::clear_cached_topology() noexcept {
    cached_input_edge_index_ = torch::Tensor();
    cached_input_edge_attr_.reset();
    cached_edge_index_ = torch::Tensor();
    cached_edge_attr_.reset();
    cached_num_nodes_.reset();
    cached_edge_index_version_ = 0;
    cached_edge_attr_version_ = 0;
}

void GATConvImpl::to(
    torch::Device device,
    torch::Dtype dtype,
    const bool non_blocking) {
    torch::nn::Module::to(device, dtype, non_blocking);
    clear_cached_topology();
    if (options_.fill_value.kind == SelfLoopFillKind::tensor
        && options_.fill_value.value.has_value()) {
        options_.fill_value.value = options_.fill_value.value->to(
            device, dtype, non_blocking, false);
    }
}

void GATConvImpl::to(const torch::Dtype dtype, const bool non_blocking) {
    torch::nn::Module::to(dtype, non_blocking);
    clear_cached_topology();
    if (options_.fill_value.kind == SelfLoopFillKind::tensor
        && options_.fill_value.value.has_value()) {
        options_.fill_value.value = options_.fill_value.value->to(
            dtype, non_blocking, false);
    }
}

void GATConvImpl::to(torch::Device device, const bool non_blocking) {
    torch::nn::Module::to(device, non_blocking);
    clear_cached_topology();
    if (options_.fill_value.kind == SelfLoopFillKind::tensor
        && options_.fill_value.value.has_value()) {
        options_.fill_value.value = options_.fill_value.value->to(
            device,
            options_.fill_value.value->scalar_type(),
            non_blocking,
            false);
    }
}

GATForwardResult GATConvImpl::forward_impl(
    const torch::Tensor& x,
    const torch::Tensor& original_edge_index,
    std::optional<torch::Tensor> edge_attr) {
    require_defined(x, "x");
    validate_edge_index(original_edge_index);
    if (x.dim() != 2 || x.size(1) != options_.in_channels) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "GATConv x must have shape [N, in_channels]");
    }
    validate_edge_attr(original_edge_index, edge_attr);
    if (x.device() != original_edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "GATConv x and edge_index must be on the same device");
    }

    const std::int64_t nodes = x.size(0);
    const std::int64_t heads = options_.heads;
    const std::int64_t channels = options_.out_channels;
    const torch::Tensor projected = lin->forward(x).view({-1, heads, channels});
    const torch::Tensor alpha_source = (projected * att_src).sum(-1);
    const torch::Tensor alpha_target = (projected * att_dst).sum(-1);

    torch::Tensor edge_index = original_edge_index;
    if (options_.add_self_loops) {
        const std::optional<torch::Tensor> input_edge_attr = edge_attr;
        const bool fill_requires_grad =
            options_.fill_value.kind == SelfLoopFillKind::tensor
            && options_.fill_value.value.has_value()
            && options_.fill_value.value->requires_grad();
        const bool cache_allowed = !fill_requires_grad
            && (!input_edge_attr.has_value()
                || !input_edge_attr->requires_grad());
        const std::uint32_t edge_index_version =
            tensor_version(original_edge_index);
        const std::uint32_t edge_attr_version = input_edge_attr.has_value()
            ? tensor_version(*input_edge_attr)
            : 0U;
        const bool cache_hit = cache_allowed
            && cached_edge_index_.defined()
            && cached_num_nodes_.has_value()
            && *cached_num_nodes_ == nodes
            && cached_input_edge_index_.defined()
            && cached_input_edge_index_.unsafeGetTensorImpl()
                == original_edge_index.unsafeGetTensorImpl()
            && cached_edge_index_version_ == edge_index_version
            && cached_input_edge_attr_.has_value() == input_edge_attr.has_value()
            && (!input_edge_attr.has_value()
                || (cached_input_edge_attr_->unsafeGetTensorImpl()
                        == input_edge_attr->unsafeGetTensorImpl()
                    && cached_edge_attr_version_ == edge_attr_version));
        if (cache_hit) {
            edge_index = cached_edge_index_;
            edge_attr = cached_edge_attr_;
        } else {
            LoopResult without = remove_self_loops(
                edge_index, std::move(edge_attr));
            LoopResult with = add_self_loops(
                without.edge_index,
                std::move(without.edge_attr),
                options_.fill_value,
                nodes);
            edge_index = std::move(with.edge_index);
            edge_attr = std::move(with.edge_attr);
            if (cache_allowed) {
                cached_input_edge_index_ = original_edge_index;
                cached_input_edge_attr_ = input_edge_attr;
                cached_edge_index_ = edge_index;
                cached_edge_attr_ = edge_attr;
                cached_num_nodes_ = nodes;
                cached_edge_index_version_ = edge_index_version;
                cached_edge_attr_version_ = edge_attr_version;
            } else {
                clear_cached_topology();
            }
        }
    }

    const torch::Tensor source = edge_index.select(0, 0);
    const torch::Tensor target = edge_index.select(0, 1);
    torch::Tensor alpha = alpha_source.index_select(0, source)
        + alpha_target.index_select(0, target);
    if (edge_attr.has_value() && lin_edge) {
        torch::Tensor attributes = *edge_attr;
        if (attributes.dim() == 1) attributes = attributes.view({-1, 1});
        const torch::Tensor projected_edges = lin_edge->forward(attributes)
            .view({-1, heads, channels});
        alpha = alpha + (projected_edges * att_edge).sum(-1);
    }
    alpha = torch::leaky_relu(alpha, options_.negative_slope);
    alpha = segment_softmax(alpha, target, nodes);
    alpha = torch::dropout(alpha, options_.dropout, is_training());

    const torch::Tensor messages = alpha.unsqueeze(-1)
        * projected.index_select(0, source);
    torch::Tensor output = scatter_dim0(
        messages, target, nodes, ScatterReduce::sum);
    output = options_.concat
        ? output.view({-1, heads * channels})
        : output.mean(1);
    if (res) output = output + res->forward(x);
    if (bias.defined()) output = output + bias;
    return {output, edge_index, alpha};
}

torch::Tensor GATConvImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr) {
    return forward_impl(x, edge_index, std::move(edge_attr)).output;
}

GATForwardResult GATConvImpl::forward_with_attention(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr) {
    return forward_impl(x, edge_index, std::move(edge_attr));
}

}  // namespace virne::solver::learning::pyg_lite
