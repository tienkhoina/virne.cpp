#include "../../virne/solver/learning/pyg_lite/pyg_lite.h"

#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_shape(
    const torch::Tensor& value,
    const std::vector<std::int64_t>& expected,
    const std::string& message) {
    require(value.sizes().vec() == expected, message);
}

void require_equal(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message) {
    require_shape(actual, expected.sizes().vec(), message + " (shape)");
    require(actual.scalar_type() == expected.scalar_type(), message + " (dtype)");
    require(actual.device() == expected.device(), message + " (device)");
    require(torch::equal(actual, expected), message + " (values)");
}

void require_close(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message,
    const double rtol = 1e-6,
    const double atol = 1e-7) {
    require_shape(actual, expected.sizes().vec(), message + " (shape)");
    require(actual.device() == expected.device(), message + " (device)");
    require(
        torch::allclose(actual, expected, rtol, atol),
        message + " (values)");
}

void require_finite(const torch::Tensor& value, const std::string& message) {
    require(value.defined(), message + " (undefined)");
    require(torch::isfinite(value).all().item<bool>(), message + " (non-finite)");
}

torch::TensorOptions index_options(const torch::Device& device) {
    return torch::TensorOptions().dtype(torch::kInt64).device(device);
}

torch::TensorOptions value_options(
    const torch::Device& device,
    const torch::Dtype dtype = torch::kFloat32) {
    return torch::TensorOptions().dtype(dtype).device(device);
}

void test_data_and_batch() {
    const torch::Device device(torch::kCPU);
    const auto indices = index_options(device);
    const auto values = value_options(device, torch::kFloat64);

    pyg::Data first{};
    first.x = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, values);
    first.edge_index = torch::tensor(
        {{0, 1, 1}, {1, 0, 1}}, indices);
    first.edge_attr = torch::tensor({{0.1}, {0.2}, {0.3}}, values);
    first.validate();
    require(first.num_nodes() == 2, "Data node count mismatch");
    require(first.num_edges() == 3, "Data edge count mismatch");

    pyg::Data second{};
    second.x = torch::tensor({{5.0, 6.0}}, values);
    second.edge_index = torch::empty({2, 0}, indices);
    second.edge_attr = torch::empty({0, 1}, values);

    pyg::Data third{};
    third.x = torch::tensor({{7.0, 8.0}, {9.0, 10.0}}, values);
    third.edge_index = torch::tensor({{0}, {1}}, indices);
    third.edge_attr = torch::tensor({{0.4}}, values);

    const pyg::Batch batch = pyg::Batch::from_data_list({first, second, third});
    batch.validate();
    require(batch.num_graphs == 3, "Batch graph count mismatch");
    require_equal(
        batch.x,
        torch::tensor(
            {{1.0, 2.0},
             {3.0, 4.0},
             {5.0, 6.0},
             {7.0, 8.0},
             {9.0, 10.0}},
            values),
        "Batch node concatenation mismatch");
    require_equal(
        batch.edge_index,
        torch::tensor({{0, 1, 1, 3}, {1, 0, 1, 4}}, indices),
        "Batch edge offset mismatch");
    require(batch.edge_attr.has_value(), "Batch dropped edge attributes");
    require_close(
        *batch.edge_attr,
        torch::tensor({{0.1}, {0.2}, {0.3}, {0.4}}, values),
        "Batch edge attribute concatenation mismatch");
    require(batch.batch.has_value(), "Batch vector is absent");
    require_equal(
        *batch.batch,
        torch::tensor({0, 0, 1, 2, 2}, indices),
        "Batch assignment vector mismatch");
    require(batch.ptr.has_value(), "Batch pointer vector is absent");
    require_equal(
        *batch.ptr,
        torch::tensor({0, 2, 3, 5}, indices),
        "Batch pointer vector mismatch");
    require_equal(
        batch.edge_ptr,
        torch::tensor({0, 3, 3, 4}, indices),
        "Batch edge pointer vector mismatch");
    require(batch.max_num_nodes == 2, "Batch max node count mismatch");

    const std::vector<pyg::Data> selected = batch.index_select(
        torch::tensor({2, 0}, indices));
    require(selected.size() == 2U, "Batch index_select result count mismatch");
    require_equal(selected[0].x, third.x, "Batch index_select graph 2 nodes mismatch");
    require_equal(
        selected[0].edge_index,
        third.edge_index,
        "Batch index_select graph 2 edges mismatch");
    require_close(
        *selected[0].edge_attr,
        *third.edge_attr,
        "Batch index_select graph 2 attributes mismatch");
    require_equal(selected[1].x, first.x, "Batch index_select graph 0 nodes mismatch");
    require_equal(
        selected[1].edge_index,
        first.edge_index,
        "Batch index_select graph 0 edges mismatch");

    const pyg::Batch copied = batch.to(device, false, true);
    require(copied.num_graphs == batch.num_graphs, "Batch.to lost graph count");
    require_equal(copied.x, batch.x, "Batch.to changed node values");
    require_equal(copied.edge_index, batch.edge_index, "Batch.to changed edges");
    require(copied.x.data_ptr() != batch.x.data_ptr(), "Batch.to(copy=true) aliased x");

    const pyg::Batch cloned = batch.clone();
    require(cloned.num_graphs == batch.num_graphs, "Batch.clone lost graph count");
    require_equal(cloned.x, batch.x, "Batch.clone changed node values");
    require_equal(cloned.edge_index, batch.edge_index, "Batch.clone changed edges");
    require(cloned.x.data_ptr() != batch.x.data_ptr(), "Batch.clone aliased x");
    require(
        cloned.edge_index.data_ptr() != batch.edge_index.data_ptr(),
        "Batch.clone aliased edge_index");
    require_equal(cloned.edge_ptr, batch.edge_ptr, "Batch.clone changed edge_ptr");
    require(
        cloned.edge_ptr.data_ptr() != batch.edge_ptr.data_ptr(),
        "Batch.clone aliased edge_ptr");
}

void test_scatter_reductions() {
    const auto values = torch::TensorOptions().dtype(torch::kFloat64);
    const torch::Tensor source = torch::tensor(
        {{2.0, -1.0}, {4.0, 3.0}, {-6.0, 5.0}}, values);
    const torch::Tensor index = torch::tensor({0, 1, 0}, torch::kInt64);

    require_equal(
        pyg::scatter_dim0(source, index, 3, pyg::ScatterReduce::sum),
        torch::tensor({{-4.0, 4.0}, {4.0, 3.0}, {0.0, 0.0}}, values),
        "scatter sum mismatch");
    require_equal(
        pyg::scatter_dim0(source, index, 3, pyg::ScatterReduce::mean),
        torch::tensor({{-2.0, 2.0}, {4.0, 3.0}, {0.0, 0.0}}, values),
        "scatter mean mismatch");
    require_equal(
        pyg::scatter_dim0(source, index, 3, pyg::ScatterReduce::minimum),
        torch::tensor({{-6.0, -1.0}, {4.0, 3.0}, {0.0, 0.0}}, values),
        "scatter minimum mismatch");
    require_equal(
        pyg::scatter_dim0(source, index, 3, pyg::ScatterReduce::maximum),
        torch::tensor({{2.0, 5.0}, {4.0, 3.0}, {0.0, 0.0}}, values),
        "scatter maximum mismatch");
    require_equal(
        pyg::scatter_dim0(source, index, 3, pyg::ScatterReduce::product),
        torch::tensor({{-12.0, -5.0}, {4.0, 3.0}, {1.0, 1.0}}, values),
        "scatter product mismatch");

    const torch::Tensor flags = torch::tensor(
        {0, 1, 1}, torch::TensorOptions().dtype(torch::kBool));
    const torch::Tensor unique = torch::tensor({0, 1, 2}, torch::kInt64);
    require_equal(
        pyg::scatter_dim0(flags, unique, 4, pyg::ScatterReduce::any),
        torch::tensor(
            {0, 1, 1, 0}, torch::TensorOptions().dtype(torch::kBool)),
        "scatter any mismatch");
}

void test_edge_sorting() {
    const torch::Tensor edge_index = torch::tensor(
        {{2, 0, 1, 0, 2}, {0, 2, 0, 1, 0}}, torch::kInt64);
    const torch::Tensor edge_attr = torch::tensor(
        {20, 2, 10, 1, 21}, torch::kInt64);

    const pyg::SortedEdges by_row = pyg::sort_edge_index(
        edge_index, edge_attr, 3, true);
    require_equal(
        by_row.permutation,
        torch::tensor({3, 1, 2, 0, 4}, torch::kInt64),
        "row-major edge permutation mismatch");
    require_equal(
        by_row.edge_index,
        torch::tensor(
            {{0, 0, 1, 2, 2}, {1, 2, 0, 0, 0}}, torch::kInt64),
        "row-major edge ordering mismatch");
    require_equal(
        *by_row.edge_attr,
        torch::tensor({1, 2, 10, 20, 21}, torch::kInt64),
        "row-major edge attribute ordering mismatch");

    const pyg::SortedEdges by_column = pyg::sort_edge_index(
        edge_index, edge_attr, 3, false);
    require_equal(
        by_column.permutation,
        torch::tensor({2, 0, 4, 3, 1}, torch::kInt64),
        "column-major edge permutation mismatch");
    require_equal(
        *by_column.edge_attr,
        torch::tensor({10, 20, 21, 1, 2}, torch::kInt64),
        "column-major edge attribute ordering mismatch");
}

void test_self_loop_utilities() {
    const torch::Tensor edge_index = torch::tensor(
        {{0, 1, 1, 2, 2}, {0, 0, 2, 2, 1}}, torch::kInt64);
    const torch::Tensor edge_attr = torch::tensor(
        {{10.0}, {20.0}, {30.0}, {40.0}, {50.0}});

    const pyg::LoopResult removed = pyg::remove_self_loops(edge_index, edge_attr);
    require_equal(
        removed.edge_index,
        torch::tensor({{1, 1, 2}, {0, 2, 1}}, torch::kInt64),
        "remove_self_loops edge mismatch");
    require_equal(
        *removed.edge_attr,
        torch::tensor({{20.0}, {30.0}, {50.0}}),
        "remove_self_loops attribute mismatch");

    const pyg::LoopResult scalar = pyg::add_self_loops(
        edge_index,
        edge_attr,
        pyg::SelfLoopFill::scalar_value(7.0),
        3);
    require_shape(scalar.edge_index, {2, 8}, "add_self_loops edge shape mismatch");
    require_equal(
        scalar.edge_index.slice(1, 5),
        torch::tensor({{0, 1, 2}, {0, 1, 2}}, torch::kInt64),
        "add_self_loops appended edge mismatch");
    require_equal(
        scalar.edge_attr->slice(0, 5),
        torch::tensor({{7.0}, {7.0}, {7.0}}),
        "add_self_loops scalar fill mismatch");

    const pyg::LoopResult tensor_fill = pyg::add_self_loops(
        edge_index,
        edge_attr,
        pyg::SelfLoopFill::tensor_value(torch::tensor({8.0})),
        3);
    require_equal(
        tensor_fill.edge_attr->slice(0, 5),
        torch::tensor({{8.0}, {8.0}, {8.0}}),
        "add_self_loops tensor fill mismatch");

    const pyg::LoopResult mean = pyg::add_self_loops(
        edge_index,
        edge_attr,
        pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean),
        3);
    require_equal(
        mean.edge_attr->slice(0, 5),
        torch::tensor({{15.0}, {50.0}, {35.0}}),
        "add_self_loops mean fill mismatch");

    const pyg::LoopResult remaining = pyg::add_remaining_self_loops(
        edge_index,
        edge_attr,
        pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean),
        3);
    require_equal(
        remaining.edge_index,
        torch::tensor(
            {{1, 1, 2, 0, 1, 2}, {0, 2, 1, 0, 1, 2}},
            torch::kInt64),
        "add_remaining_self_loops edge mismatch");
    require_equal(
        *remaining.edge_attr,
        torch::tensor({{20.0}, {30.0}, {50.0}, {10.0}, {50.0}, {40.0}}),
        "add_remaining_self_loops preserved fill mismatch");
}

void test_segment_softmax() {
    const auto values = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor source = torch::tensor(
        {{1001.0, 1000.0},
         {1002.0, 1002.0},
         {1003.0, 999.0},
         {1004.0, 1000.0}},
        values);
    source.set_requires_grad(true);
    const torch::Tensor index = torch::tensor({0, 0, 1, 0}, torch::kInt64);
    const torch::Tensor output = pyg::segment_softmax(source, index, 2);

    const double first_denominator = std::exp(-3.0) + std::exp(-2.0) + 1.0;
    const double second_denominator = 2.0 * std::exp(-2.0) + 1.0;
    const torch::Tensor expected = torch::tensor(
        {{std::exp(-3.0) / first_denominator,
          std::exp(-2.0) / second_denominator},
         {std::exp(-2.0) / first_denominator,
          1.0 / second_denominator},
         {1.0, 1.0},
         {1.0 / first_denominator,
          std::exp(-2.0) / second_denominator}},
        values);
    require_close(output, expected, "segment softmax mismatch", 1e-12, 1e-12);
    require_close(
        pyg::scatter_dim0(output, index, 2, pyg::ScatterReduce::sum),
        torch::ones({2, 2}, values),
        "segment softmax group sums mismatch",
        1e-12,
        1e-12);

    const torch::Tensor weights = torch::arange(1.0, 9.0, values).view({4, 2});
    (output * weights).sum().backward();
    require_finite(source.grad(), "segment softmax input gradient");
}

void test_dense_batch_and_pools() {
    const torch::Tensor x = torch::tensor(
        {{1.0, 10.0},
         {2.0, 20.0},
         {3.0, 30.0},
         {4.0, 40.0},
         {5.0, 50.0}});
    const torch::Tensor batch = torch::tensor({0, 0, 1, 2, 2}, torch::kInt64);

    const pyg::DenseBatchResult dense = pyg::to_dense_batch(
        x, batch, -1.0, 2, 4);
    require_equal(
        dense.values,
        torch::tensor(
            {{{1.0, 10.0}, {2.0, 20.0}},
             {{3.0, 30.0}, {-1.0, -1.0}},
             {{4.0, 40.0}, {5.0, 50.0}},
             {{-1.0, -1.0}, {-1.0, -1.0}}}),
        "to_dense_batch values mismatch");
    require_equal(
        dense.mask,
        torch::tensor(
            {{1, 1}, {1, 0}, {1, 1}, {0, 0}},
            torch::TensorOptions().dtype(torch::kBool)),
        "to_dense_batch mask mismatch");

    const pyg::DenseBatchResult truncated = pyg::to_dense_batch(
        x, batch, 0.0, 1, 3);
    require_equal(
        truncated.values,
        torch::tensor({{{1.0, 10.0}}, {{3.0, 30.0}}, {{4.0, 40.0}}}),
        "to_dense_batch truncation mismatch");
    const pyg::DenseBatchResult unbatched = pyg::to_dense_batch(x);
    require_equal(unbatched.values, x.unsqueeze(0), "unbatched dense values mismatch");
    require_equal(
        unbatched.mask,
        torch::ones({1, 5}, torch::TensorOptions().dtype(torch::kBool)),
        "unbatched dense mask mismatch");

    require_equal(
        pyg::global_add_pool(x, batch, 4),
        torch::tensor(
            {{3.0, 30.0}, {3.0, 30.0}, {9.0, 90.0}, {0.0, 0.0}}),
        "global add pool mismatch");
    require_equal(
        pyg::global_mean_pool(x, batch, 4),
        torch::tensor(
            {{1.5, 15.0}, {3.0, 30.0}, {4.5, 45.0}, {0.0, 0.0}}),
        "global mean pool mismatch");
    require_equal(
        pyg::global_max_pool(x, batch, 4),
        torch::tensor(
            {{2.0, 20.0}, {3.0, 30.0}, {5.0, 50.0}, {0.0, 0.0}}),
        "global max pool mismatch");
    require_equal(
        pyg::global_add_pool(x),
        torch::tensor({{15.0, 150.0}}),
        "unbatched global add pool mismatch");
    require_equal(
        pyg::global_mean_pool(x),
        torch::tensor({{3.0, 30.0}}),
        "unbatched global mean pool mismatch");
    require_equal(
        pyg::global_max_pool(x),
        torch::tensor({{5.0, 50.0}}),
        "unbatched global max pool mismatch");

    const torch::Tensor rank_three = torch::arange(0.0, 12.0).view({2, 3, 2});
    require_equal(
        pyg::global_add_pool(
            rank_three,
            torch::tensor({0, 1, 0}, torch::kInt64),
            2),
        torch::tensor(
            {{{4.0, 6.0}, {2.0, 3.0}},
             {{16.0, 18.0}, {8.0, 9.0}}}),
        "rank-three global add pool dimension mismatch");
}

void test_gcn_normalization_and_convolution() {
    const torch::Tensor edge_index = torch::tensor(
        {{0, 1, 1}, {1, 0, 2}}, torch::kInt64);
    const pyg::GcnNormResult normalized = pyg::gcn_norm(
        edge_index, std::nullopt, 3, false, true, torch::kFloat64);
    require_equal(
        normalized.edge_index,
        torch::tensor(
            {{0, 1, 1, 0, 1, 2}, {1, 0, 2, 0, 1, 2}},
            torch::kInt64),
        "gcn_norm edge mismatch");
    require_close(
        normalized.edge_weight,
        torch::full({6}, 0.5, torch::TensorOptions().dtype(torch::kFloat64)),
        "gcn_norm unit-weight mismatch",
        1e-12,
        1e-12);

    const torch::Tensor weighted_edges = torch::tensor(
        {{0, 1, 1}, {1, 0, 1}}, torch::kInt64);
    const torch::Tensor weights = torch::tensor(
        {2.0, 3.0, 4.0}, torch::TensorOptions().dtype(torch::kFloat64));
    const pyg::GcnNormResult improved = pyg::gcn_norm(
        weighted_edges, weights, 2, true, true, torch::kFloat64);
    const double root30 = std::sqrt(30.0);
    require_close(
        improved.edge_weight,
        torch::tensor(
            {2.0 / root30, 3.0 / root30, 2.0 / 5.0, 4.0 / 6.0},
            torch::TensorOptions().dtype(torch::kFloat64)),
        "gcn_norm weighted/improved mismatch",
        1e-12,
        1e-12);

    pyg::GCNConvOptions options{};
    options.in_channels = 2;
    options.out_channels = 2;
    options.cached = true;
    pyg::GCNConv model(options);
    {
        torch::NoGradGuard guard;
        model->lin->weight.copy_(torch::eye(2));
        model->bias.copy_(torch::tensor({0.25, -0.5}));
    }
    torch::Tensor x = torch::tensor(
        {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    x.set_requires_grad(true);
    const torch::Tensor output = model->forward(x, edge_index);
    require_close(
        output,
        torch::tensor({{2.25, 2.5}, {2.25, 2.5}, {4.25, 4.5}}),
        "GCNConv forward mismatch");
    output.square().sum().backward();
    require_finite(x.grad(), "GCNConv input gradient");
    for (const auto& item : model->named_parameters(true)) {
        require_finite(item.value().grad(), "GCNConv parameter gradient: " + item.key());
    }

    // cached=true is safe across a new tensor and an in-place topology
    // mutation: identity/version changes must invalidate normalization.
    torch::Tensor changed_edges = torch::tensor(
        {{0, 2, 1}, {2, 1, 0}}, torch::kInt64);
    pyg::GCNConvOptions fresh_options = options;
    fresh_options.cached = false;
    pyg::GCNConv fresh(fresh_options);
    {
        torch::NoGradGuard guard;
        fresh->lin->weight.copy_(model->lin->weight);
        fresh->bias.copy_(model->bias);
        require_close(
            model->forward(x.detach(), changed_edges),
            fresh->forward(x.detach(), changed_edges),
            "GCNConv cache ignored new topology");
        changed_edges.index_put_({0, 0}, 1);
        require_close(
            model->forward(x.detach(), changed_edges),
            fresh->forward(x.detach(), changed_edges),
            "GCNConv cache ignored in-place topology mutation");
    }
}

void fill_deterministic_parameters(torch::nn::Module& module) {
    torch::NoGradGuard guard;
    std::int64_t offset = 1;
    for (auto& item : module.named_parameters(true)) {
        torch::Tensor values = torch::arange(
            offset,
            offset + item.value().numel(),
            item.value().options()).view(item.value().sizes());
        item.value().copy_(values * 0.013 - 0.2);
        offset += item.value().numel();
    }
}

void test_gat_convolution() {
    pyg::GATConvOptions options{};
    options.in_channels = 2;
    options.out_channels = 2;
    options.heads = 2;
    options.concat = true;
    options.dropout = 0.0;
    options.add_self_loops = true;
    options.edge_dim = 1;
    options.bias = true;
    options.residual = true;
    pyg::GATConv model(options);
    fill_deterministic_parameters(*model);

    torch::Tensor x = torch::tensor(
        {{1.0, -1.0}, {0.5, 2.0}, {-2.0, 1.5}});
    x.set_requires_grad(true);
    const torch::Tensor edge_index = torch::tensor(
        {{0, 1, 2, 2}, {1, 2, 0, 2}}, torch::kInt64);
    const torch::Tensor edge_attr = torch::tensor({{1.0}, {2.0}, {3.0}, {4.0}});
    const pyg::GATForwardResult result = model->forward_with_attention(
        x, edge_index, edge_attr);

    require_shape(result.output, {3, 4}, "GATConv output shape mismatch");
    require_shape(result.edge_index, {2, 6}, "GATConv self-loop edge shape mismatch");
    require_shape(result.attention, {6, 2}, "GATConv attention shape mismatch");
    require_finite(result.output, "GATConv output");
    require_finite(result.attention, "GATConv attention");
    const torch::Tensor target = result.edge_index.select(0, 1);
    require_close(
        pyg::scatter_dim0(
            result.attention, target, 3, pyg::ScatterReduce::sum),
        torch::ones({3, 2}),
        "GATConv attention normalization mismatch");

    result.output.square().mean().backward();
    require_finite(x.grad(), "GATConv input gradient");
    for (const auto& item : model->named_parameters(true)) {
        require_finite(item.value().grad(), "GATConv parameter gradient: " + item.key());
    }

    // Immutable inference topology is cached after the first call. Compare
    // the cache-hit output with a fresh module, then mutate both inputs to
    // ensure tensor version checks invalidate the cached edge transform.
    pyg::GATConv uncached(options);
    {
        torch::NoGradGuard guard;
        auto target_parameters = uncached->named_parameters(true);
        for (const auto& item : model->named_parameters(true)) {
            target_parameters[item.key()].copy_(item.value());
        }
    }
    const torch::Tensor cached_output = model->forward(
        x.detach(), edge_index, edge_attr);
    const torch::Tensor reference_output = uncached->forward(
        x.detach(), edge_index, edge_attr);
    require_close(
        cached_output, reference_output,
        "GATConv topology cache hit mismatch");
    torch::Tensor changed_edges = edge_index.clone();
    torch::Tensor changed_attrs = edge_attr.clone();
    changed_edges.index_put_({0, 0}, 1);
    changed_attrs.index_put_({0, 0}, 9.0);
    const torch::Tensor changed_output = model->forward(
        x.detach(), changed_edges, changed_attrs);
    const torch::Tensor changed_reference = uncached->forward(
        x.detach(), changed_edges, changed_attrs);
    require_close(
        changed_output, changed_reference,
        "GATConv topology cache invalidation mismatch");

    // Grad-tracked edge attributes must never be retained by the cache: the
    // loop-fill path remains differentiable for training callers.
    torch::Tensor grad_attrs = edge_attr.detach().clone();
    grad_attrs.set_requires_grad(true);
    const torch::Tensor grad_output = model->forward(
        x.detach(), edge_index, grad_attrs);
    grad_output.square().mean().backward();
    require_finite(grad_attrs.grad(), "GATConv edge attribute gradient");

    pyg::GATConvOptions mean_options{};
    mean_options.in_channels = 2;
    mean_options.out_channels = 3;
    mean_options.heads = 2;
    mean_options.concat = false;
    mean_options.add_self_loops = false;
    mean_options.bias = false;
    pyg::GATConv mean_model(mean_options);
    const torch::Tensor mean_output = mean_model->forward(x.detach(), edge_index);
    require_shape(mean_output, {3, 3}, "GATConv non-concat output shape mismatch");
    require_finite(mean_output, "GATConv non-concat output");
}

void test_validation_boundaries() {
    bool invalid_data = false;
    try {
        pyg::Data data{};
        data.x = torch::zeros({2, 3});
        data.edge_index = torch::zeros({3, 2}, torch::kInt64);
        data.validate();
    } catch (const pyg::PygLiteException& error) {
        invalid_data = error.code() == pyg::ErrorCode::invalid_shape;
    }
    require(invalid_data, "Data accepted malformed edge_index");

    bool invalid_index = false;
    try {
        pyg::Data data{};
        data.x = torch::zeros({2, 3});
        data.edge_index = torch::tensor({{0, 2}, {1, 0}}, torch::kInt64);
        data.validate_indices();
    } catch (const pyg::PygLiteException& error) {
        invalid_index = error.code() == pyg::ErrorCode::invalid_index;
    }
    require(invalid_index, "Data accepted an out-of-range node index");

    bool empty_batch = false;
    try {
        (void)pyg::Batch::from_data_list({});
    } catch (const pyg::PygLiteException& error) {
        empty_batch = error.code() == pyg::ErrorCode::inconsistent_batch;
    }
    require(empty_batch, "Batch accepted an empty data list");

    bool invalid_gcn = false;
    try {
        pyg::GCNConvOptions options{};
        options.in_channels = 2;
        options.out_channels = 2;
        options.normalize = false;
        options.add_self_loops = true;
        pyg::GCNConv model(options);
        (void)model;
    } catch (const pyg::PygLiteException& error) {
        invalid_gcn = error.code() == pyg::ErrorCode::unsupported_option;
    }
    require(invalid_gcn, "GCNConv accepted loops without normalization");
}

// This intentionally uses no CPU-only value extraction. Running the same body
// on CUDA catches accidental host tensors, host indices, and device-mismatched
// module parameters while retaining a cheap smoke-test footprint.
void test_device_smoke(const torch::Device& device) {
    const auto indices = index_options(device);
    const auto values = value_options(device);

    pyg::Data graph{};
    graph.x = torch::tensor(
        {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}}, values);
    graph.edge_index = torch::tensor(
        {{0, 1, 2}, {1, 2, 0}}, indices);
    graph.edge_attr = torch::tensor({{1.0F}, {2.0F}, {3.0F}}, values);
    graph.validate();

    const pyg::Batch batch = pyg::Batch::from_data_list({graph, graph});
    require(batch.x.device() == device, "device smoke Batch moved x");
    require(batch.edge_index.device() == device, "device smoke Batch moved edges");
    require(batch.batch->device() == device, "device smoke Batch moved batch index");

    const torch::Tensor scatter = pyg::scatter_dim0(
        graph.x,
        torch::tensor({0, 1, 0}, indices),
        2,
        pyg::ScatterReduce::sum);
    require(scatter.device() == device, "device smoke scatter moved output");

    const pyg::SortedEdges sorted = pyg::sort_edge_index(
        graph.edge_index, graph.edge_attr, 3, true);
    require(sorted.edge_index.device() == device, "device smoke sort moved output");
    const pyg::LoopResult loops = pyg::add_remaining_self_loops(
        sorted.edge_index,
        sorted.edge_attr,
        pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean),
        3);
    require(loops.edge_index.device() == device, "device smoke loops moved output");

    const torch::Tensor logits = torch::tensor(
        {{1.0F, 2.0F}, {3.0F, 1.0F}, {0.0F, 4.0F}}, values);
    const torch::Tensor groups = torch::tensor({0, 0, 1}, indices);
    const torch::Tensor probabilities = pyg::segment_softmax(logits, groups, 2);
    require_finite(probabilities, "device smoke segment softmax");

    const pyg::DenseBatchResult dense = pyg::to_dense_batch(batch, -1.0);
    require(dense.values.device() == device, "device smoke dense values moved output");
    require(dense.mask.device() == device, "device smoke dense mask moved output");
    require_finite(
        pyg::global_mean_pool(batch.x, batch.batch, 2),
        "device smoke global pool");

    const pyg::GcnNormResult normalization = pyg::gcn_norm(
        graph.edge_index, std::nullopt, 3, false, true, torch::kFloat32);
    require(normalization.edge_index.device() == device, "device smoke gcn_norm moved edges");
    require_finite(normalization.edge_weight, "device smoke gcn_norm weights");

    pyg::GCNConvOptions gcn_options{};
    gcn_options.in_channels = 2;
    gcn_options.out_channels = 3;
    pyg::GCNConv gcn(gcn_options);
    gcn->to(device);

    pyg::GATConvOptions gat_options{};
    gat_options.in_channels = 3;
    gat_options.out_channels = 2;
    gat_options.heads = 2;
    gat_options.concat = true;
    gat_options.edge_dim = 1;
    pyg::GATConv gat(gat_options);
    gat->to(device);

    torch::Tensor input = graph.x.detach().clone();
    input.set_requires_grad(true);
    const torch::Tensor gcn_output = gcn->forward(input, graph.edge_index);
    const torch::Tensor gat_output = gat->forward(
        gcn_output, graph.edge_index, *graph.edge_attr);
    require(gcn_output.device() == device, "device smoke GCN moved output");
    require(gat_output.device() == device, "device smoke GAT moved output");
    require_finite(gat_output, "device smoke GAT output");
    gat_output.square().mean().backward();
    require_finite(input.grad(), "device smoke composed input gradient");
    for (const auto& item : gcn->named_parameters(true)) {
        require_finite(item.value().grad(), "device smoke GCN gradient: " + item.key());
    }
    for (const auto& item : gat->named_parameters(true)) {
        require_finite(item.value().grad(), "device smoke GAT gradient: " + item.key());
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        torch::manual_seed(0);

        test_data_and_batch();
        test_scatter_reductions();
        test_edge_sorting();
        test_self_loop_utilities();
        test_segment_softmax();
        test_dense_batch_and_pools();
        test_gcn_normalization_and_convolution();
        test_gat_convolution();
        test_validation_boundaries();
        test_device_smoke(torch::Device(torch::kCPU));

        if (torch::cuda::is_available()) {
            test_device_smoke(torch::Device(torch::kCUDA, 0));
            torch::cuda::synchronize();
        }

        std::cout << "pyg_lite_unit: PASS (cpu=true, cuda="
                  << (torch::cuda::is_available() ? "true" : "false")
                  << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pyg_lite_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
