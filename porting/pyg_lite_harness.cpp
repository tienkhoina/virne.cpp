#include "../virne/solver/learning/pyg_lite/pyg_lite.h"

#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef VIRNE_LIBTORCH_VERSION
#define VIRNE_LIBTORCH_VERSION "unknown"
#endif

namespace pyg = virne::solver::learning::pyg_lite;

namespace {

enum class Mode : std::uint8_t { parity, benchmark };

struct Arguments {
    Mode mode = Mode::parity;
    std::string device = "cpu";
    int threads = 1;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 3;
};

struct TensorRecord {
    std::string name;
    torch::Tensor value;
};

struct BenchmarkRecord {
    std::string name;
    std::vector<std::int64_t> samples_ns;
    double median_ns = 0.0;
    torch::Tensor signature;
};

[[noreturn]] void usage_error(const std::string_view message) {
    throw std::invalid_argument(
        std::string(message)
        + "\nusage: vne_pyg_lite_harness parity|benchmark "
          "[--device cpu|cuda|auto] [--threads N] [--warmups N] "
          "[--repetitions N] [--iterations N]");
}

int parse_positive(const char* value, const char* option) {
    try {
        const auto parsed = std::stoll(value);
        if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            usage_error(std::string(option) + " requires a positive integer");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " requires a positive integer");
    }
}

Arguments parse_arguments(const int argc, char** argv) {
    if (argc < 2) usage_error("a mode is required");
    Arguments result{};
    const std::string_view mode(argv[1]);
    if (mode == "parity") {
        result.mode = Mode::parity;
    } else if (mode == "benchmark") {
        result.mode = Mode::benchmark;
    } else {
        usage_error("mode must be parity or benchmark");
    }
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto read_positive = [&](const char* option) {
            if (++index >= argc) usage_error(std::string(option) + " requires a value");
            return parse_positive(argv[index], option);
        };
        if (argument == "--device") {
            if (++index >= argc) usage_error("--device requires a value");
            result.device = argv[index];
            if (result.device != "cpu" && result.device != "cuda"
                && result.device != "auto") {
                usage_error("--device requires cpu, cuda, or auto");
            }
        } else if (argument == "--threads") {
            result.threads = read_positive("--threads");
        } else if (argument == "--warmups") {
            result.warmups = read_positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = read_positive("--repetitions");
        } else if (argument == "--iterations") {
            result.iterations = read_positive("--iterations");
        } else {
            usage_error("unknown argument");
        }
    }
    return result;
}

torch::Device resolve_device(const std::string& requested) {
    const bool available = torch::cuda::is_available();
    if (requested == "cuda" && !available) {
        throw std::runtime_error("CUDA was requested but is unavailable");
    }
    return torch::Device(
        requested == "cuda" || (requested == "auto" && available)
            ? torch::kCUDA
            : torch::kCPU);
}

void synchronize(const torch::Device& device) {
    if (device.is_cuda()) torch::cuda::synchronize();
}

torch::TensorOptions values(
    const torch::Device& device,
    const torch::ScalarType dtype = torch::kFloat64) {
    return torch::TensorOptions().dtype(dtype).device(device);
}

torch::TensorOptions indices(const torch::Device& device) {
    return torch::TensorOptions().dtype(torch::kInt64).device(device);
}

void append(
    std::vector<TensorRecord>& records,
    std::string name,
    const torch::Tensor& value) {
    records.push_back({
        std::move(name),
        value.detach().to(torch::kCPU).contiguous(),
    });
}

void fill_linear(
    torch::Tensor tensor,
    const double first,
    const double step) {
    torch::NoGradGuard guard;
    const torch::Tensor sequence = torch::arange(
        tensor.numel(), tensor.options()).reshape(tensor.sizes());
    tensor.copy_(sequence * step + first);
}

torch::Tensor loss_scale(const torch::Tensor& output) {
    return torch::arange(
        output.numel(), output.options()).reshape(output.sizes()) * 0.019 - 0.17;
}

std::vector<TensorRecord> parity_records(const torch::Device& device) {
    std::vector<TensorRecord> records;
    records.reserve(64);
    const auto value_options = values(device);
    const auto index_options = indices(device);

    pyg::Data first{};
    first.x = torch::tensor(
        {{1.0, -2.0}, {3.0, 4.0}, {-5.0, 6.0}}, value_options);
    first.edge_index = torch::tensor(
        {{2, 0, 1}, {0, 1, 2}}, index_options);
    first.edge_attr = torch::tensor({{0.5}, {1.0}, {1.5}}, value_options);
    pyg::Data second{};
    second.x = torch::tensor({{7.0, 8.0}, {9.0, -10.0}}, value_options);
    second.edge_index = torch::tensor({{0, 1}, {1, 0}}, index_options);
    second.edge_attr = torch::tensor({{2.0}, {2.5}}, value_options);
    const pyg::Batch packed = pyg::Batch::from_data_list({first, second});
    append(records, "batch/x", packed.x);
    append(records, "batch/edge_index", packed.edge_index);
    append(records, "batch/edge_attr", *packed.edge_attr);
    append(records, "batch/batch", *packed.batch);
    append(records, "batch/ptr", *packed.ptr);
    const auto selected = packed.index_select(torch::tensor({1, 0}, index_options));
    append(records, "batch/select1_x", selected[0].x);
    append(records, "batch/select1_edge_index", selected[0].edge_index);
    append(records, "batch/select0_x", selected[1].x);
    append(records, "batch/select0_edge_index", selected[1].edge_index);

    const torch::Tensor scatter_source = torch::tensor(
        {{2.0, -1.0}, {4.0, 3.0}, {-6.0, 5.0}, {1.5, -2.0}, {-2.0, 4.0}},
        value_options);
    const torch::Tensor scatter_index = torch::tensor(
        {0, 1, 0, 2, 1}, index_options);
    append(records, "scatter/sum", pyg::scatter_dim0(
        scatter_source, scatter_index, 4, pyg::ScatterReduce::sum));
    append(records, "scatter/mean", pyg::scatter_dim0(
        scatter_source, scatter_index, 4, pyg::ScatterReduce::mean));
    append(records, "scatter/min", pyg::scatter_dim0(
        scatter_source, scatter_index, 4, pyg::ScatterReduce::minimum));
    append(records, "scatter/max", pyg::scatter_dim0(
        scatter_source, scatter_index, 4, pyg::ScatterReduce::maximum));
    append(records, "scatter/prod", pyg::scatter_dim0(
        scatter_source, scatter_index, 4, pyg::ScatterReduce::product));
    const torch::Tensor flags = torch::tensor(
        {0, 1, 1, 0}, values(device, torch::kBool));
    append(records, "scatter/any", pyg::scatter_dim0(
        flags, torch::tensor({0, 1, 2, 3}, index_options), 5,
        pyg::ScatterReduce::any));

    const torch::Tensor sortable = torch::tensor(
        {{2, 0, 1, 0, 2, 1}, {0, 2, 0, 1, 1, 2}}, index_options);
    const torch::Tensor sortable_attr = torch::tensor(
        {20, 2, 10, 1, 21, 12}, index_options);
    const pyg::SortedEdges by_row = pyg::sort_edge_index(
        sortable, sortable_attr, 3, true);
    append(records, "sort/row_edge_index", by_row.edge_index);
    append(records, "sort/row_edge_attr", *by_row.edge_attr);
    append(records, "sort/row_permutation", by_row.permutation);
    const pyg::SortedEdges by_column = pyg::sort_edge_index(
        sortable, sortable_attr, 3, false);
    append(records, "sort/column_edge_index", by_column.edge_index);
    append(records, "sort/column_edge_attr", *by_column.edge_attr);
    append(records, "sort/column_permutation", by_column.permutation);
    const torch::Tensor duplicate_edges = torch::zeros({2, 100}, index_options);
    const torch::Tensor duplicate_attr = torch::arange(100, index_options);
    const pyg::SortedEdges duplicate_sorted = pyg::sort_edge_index(
        duplicate_edges, duplicate_attr, 1, true);
    append(records, "sort/duplicate_edge_attr", *duplicate_sorted.edge_attr);
    append(records, "sort/duplicate_permutation", duplicate_sorted.permutation);

    const torch::Tensor loop_edges = torch::tensor(
        {{0, 1, 1, 2, 2, 3}, {0, 0, 2, 2, 3, 1}}, index_options);
    const torch::Tensor loop_attrs = torch::tensor(
        {{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0},
         {4.0, 40.0}, {5.0, 50.0}, {6.0, 60.0}}, value_options);
    const pyg::LoopResult removed = pyg::remove_self_loops(loop_edges, loop_attrs);
    append(records, "loops/remove_edge_index", removed.edge_index);
    append(records, "loops/remove_edge_attr", *removed.edge_attr);
    const pyg::LoopResult scalar_loops = pyg::add_self_loops(
        loop_edges, loop_attrs, pyg::SelfLoopFill::scalar_value(2.25), 4);
    append(records, "loops/add_scalar_edge_index", scalar_loops.edge_index);
    append(records, "loops/add_scalar_edge_attr", *scalar_loops.edge_attr);
    const pyg::LoopResult mean_loops = pyg::add_self_loops(
        loop_edges, loop_attrs,
        pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean), 4);
    append(records, "loops/add_mean_edge_attr", *mean_loops.edge_attr);
    const pyg::LoopResult remaining = pyg::add_remaining_self_loops(
        loop_edges, loop_attrs,
        pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean), 4);
    append(records, "loops/remaining_edge_index", remaining.edge_index);
    append(records, "loops/remaining_edge_attr", *remaining.edge_attr);

    torch::Tensor logits = torch::tensor(
        {{1.0, -1.0}, {2.0, 2.0}, {-0.5, 3.0}, {4.0, 0.5}, {1.5, -2.0}},
        value_options);
    logits.set_requires_grad(true);
    const torch::Tensor groups = torch::tensor({0, 0, 1, 0, 1}, index_options);
    const torch::Tensor probabilities = pyg::segment_softmax(logits, groups, 3);
    append(records, "softmax/output", probabilities);
    (probabilities * torch::tensor(
        {{0.5, -1.0}, {1.5, 0.25}, {-0.75, 2.0},
         {0.125, -0.5}, {1.0, 0.75}}, value_options)).sum().backward();
    append(records, "softmax/input_grad", logits.grad());

    const torch::Tensor dense_x = torch::tensor(
        {{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0},
         {4.0, 40.0}, {5.0, 50.0}, {6.0, 60.0}, {7.0, 70.0}},
        value_options);
    const torch::Tensor dense_batch = torch::tensor(
        {0, 0, 0, 1, 1, 2, 2}, index_options);
    const pyg::DenseBatchResult dense = pyg::to_dense_batch(
        dense_x, dense_batch, -3.0, 3, 4);
    append(records, "dense/values", dense.values);
    append(records, "dense/mask", dense.mask);
    const pyg::DenseBatchResult truncated = pyg::to_dense_batch(
        dense_x, dense_batch, 0.0, 2, 3);
    append(records, "dense/truncated_values", truncated.values);
    append(records, "dense/truncated_mask", truncated.mask);
    append(records, "pool/add", pyg::global_add_pool(dense_x, dense_batch, 4));
    append(records, "pool/mean", pyg::global_mean_pool(dense_x, dense_batch, 4));
    append(records, "pool/max", pyg::global_max_pool(dense_x, dense_batch, 4));

    const torch::Tensor norm_edges = torch::tensor(
        {{0, 1, 1, 2, 3}, {1, 0, 2, 2, 1}}, index_options);
    const pyg::GcnNormResult unit_norm = pyg::gcn_norm(
        norm_edges, std::nullopt, 4, false, true, torch::kFloat64);
    append(records, "gcn_norm/unit_edge_index", unit_norm.edge_index);
    append(records, "gcn_norm/unit_weight", unit_norm.edge_weight);
    const torch::Tensor norm_weight = torch::tensor(
        {0.5, 1.25, 2.0, 0.75, 1.5}, value_options);
    const pyg::GcnNormResult weighted_norm = pyg::gcn_norm(
        norm_edges, norm_weight, 4, true, true, torch::kFloat64);
    append(records, "gcn_norm/weighted_edge_index", weighted_norm.edge_index);
    append(records, "gcn_norm/weighted_weight", weighted_norm.edge_weight);

    pyg::GCNConvOptions gcn_options{};
    gcn_options.in_channels = 4;
    gcn_options.out_channels = 3;
    pyg::GCNConv gcn(gcn_options);
    gcn->to(device, torch::kFloat64);
    fill_linear(gcn->lin->weight, -0.28, 0.035);
    fill_linear(gcn->bias, -0.09, 0.07);
    torch::Tensor gcn_x = torch::tensor(
        {{1.0, -1.0, 0.5, 2.0}, {0.25, 1.5, -2.0, 0.75},
         {-1.5, 0.0, 2.5, -0.5}, {2.0, 1.0, -0.75, 1.25},
         {0.5, -2.0, 1.0, 3.0}}, value_options);
    gcn_x.set_requires_grad(true);
    const torch::Tensor gcn_edges = torch::tensor(
        {{0, 1, 2, 3, 4, 1, 3}, {1, 2, 3, 4, 0, 0, 1}}, index_options);
    const torch::Tensor gcn_weight = torch::tensor(
        {0.5, 1.0, 1.5, 0.75, 1.25, 2.0, 0.625}, value_options);
    const torch::Tensor gcn_output = gcn->forward(gcn_x, gcn_edges, gcn_weight);
    append(records, "gcn/output", gcn_output);
    (gcn_output * loss_scale(gcn_output)).sum().backward();
    append(records, "gcn/input_grad", gcn_x.grad());
    append(records, "gcn/lin_weight_grad", gcn->lin->weight.grad());
    append(records, "gcn/bias_grad", gcn->bias.grad());

    pyg::GATConvOptions gat_options{};
    gat_options.in_channels = 4;
    gat_options.out_channels = 2;
    gat_options.heads = 2;
    gat_options.concat = true;
    gat_options.negative_slope = 0.2;
    gat_options.dropout = 0.0;
    gat_options.add_self_loops = true;
    gat_options.edge_dim = 2;
    gat_options.fill_value = pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean);
    gat_options.bias = true;
    gat_options.residual = true;
    pyg::GATConv gat(gat_options);
    gat->to(device, torch::kFloat64);
    fill_linear(gat->lin->weight, -0.21, 0.017);
    fill_linear(gat->att_src, -0.13, 0.029);
    fill_linear(gat->att_dst, 0.07, -0.023);
    fill_linear(gat->lin_edge->weight, -0.11, 0.031);
    fill_linear(gat->att_edge, 0.19, -0.027);
    fill_linear(gat->res->weight, -0.08, 0.021);
    fill_linear(gat->bias, 0.03, -0.025);
    torch::Tensor gat_x = gcn_x.detach().clone();
    gat_x.set_requires_grad(true);
    const torch::Tensor gat_edges = torch::tensor(
        {{0, 1, 2, 3, 4, 1, 3, 4}, {1, 2, 3, 4, 0, 0, 1, 4}},
        index_options);
    torch::Tensor gat_edge_attr = torch::tensor(
        {{0.5, 1.0}, {1.5, -0.25}, {0.75, 2.0}, {-1.0, 0.5},
         {2.0, 1.25}, {-0.5, 0.75}, {1.0, -1.5}, {0.25, 0.125}},
        value_options);
    gat_edge_attr.set_requires_grad(true);
    const pyg::GATForwardResult gat_result = gat->forward_with_attention(
        gat_x, gat_edges, gat_edge_attr);
    append(records, "gat/output", gat_result.output);
    append(records, "gat/edge_index", gat_result.edge_index);
    append(records, "gat/attention", gat_result.attention);
    (gat_result.output * loss_scale(gat_result.output)).sum().backward();
    append(records, "gat/input_grad", gat_x.grad());
    append(records, "gat/edge_attr_grad", gat_edge_attr.grad());
    append(records, "gat/lin_weight_grad", gat->lin->weight.grad());
    append(records, "gat/att_src_grad", gat->att_src.grad());
    append(records, "gat/att_dst_grad", gat->att_dst.grad());
    append(records, "gat/lin_edge_weight_grad", gat->lin_edge->weight.grad());
    append(records, "gat/att_edge_grad", gat->att_edge.grad());
    append(records, "gat/res_weight_grad", gat->res->weight.grad());
    append(records, "gat/bias_grad", gat->bias.grad());
    synchronize(device);
    return records;
}

struct BenchmarkFixture {
    torch::Device device;
    torch::Tensor x;
    torch::Tensor batch;
    torch::Tensor edge_index;
    torch::Tensor edge_attr;
    pyg::Batch packed;
    pyg::GCNConv gcn{nullptr};
    pyg::GATConv gat{nullptr};
};

BenchmarkFixture make_benchmark_fixture(const torch::Device& device) {
    constexpr std::int64_t graphs = 8;
    constexpr std::int64_t nodes_per_graph = 500;
    constexpr std::int64_t nodes = graphs * nodes_per_graph;
    constexpr std::int64_t edges_per_graph = 7000;
    constexpr std::int64_t edges = graphs * edges_per_graph;
    const auto value_options = values(device, torch::kFloat32);
    const auto index_options = indices(device);
    const torch::Tensor node_ids = torch::arange(nodes, value_options).view({-1, 1});
    const torch::Tensor feature_ids = torch::arange(25, value_options).view({1, -1});
    torch::Tensor x = torch::sin(node_ids * 0.013F + feature_ids * 0.071F);
    const torch::Tensor edge_ids = torch::arange(edges, index_options);
    const torch::Tensor graph_ids = torch::floor_divide(edge_ids, edges_per_graph);
    const torch::Tensor local_ids = edge_ids.remainder(edges_per_graph);
    const torch::Tensor source = graph_ids * nodes_per_graph
        + local_ids.remainder(nodes_per_graph);
    const torch::Tensor target = graph_ids * nodes_per_graph
        + (local_ids * 37 + torch::floor_divide(local_ids, nodes_per_graph) + 1)
              .remainder(nodes_per_graph);
    const torch::Tensor edge_index = torch::stack({source, target}, 0)
        .to(index_options);
    const torch::Tensor edge_values = edge_ids.to(torch::kFloat32);
    const torch::Tensor edge_attr = torch::stack(
        {torch::sin(edge_values * 0.017F), torch::cos(edge_values * 0.011F)}, 1);
    const torch::Tensor batch = torch::floor_divide(
        torch::arange(nodes, index_options), nodes_per_graph)
        .to(index_options);
    pyg::Batch packed{};
    packed.x = x;
    packed.edge_index = edge_index;
    packed.edge_attr = edge_attr;
    packed.batch = batch;
    packed.ptr = torch::arange(graphs + 1, index_options) * nodes_per_graph;
    packed.edge_ptr = torch::arange(graphs + 1, index_options) * edges_per_graph;
    packed.num_graphs = graphs;
    packed.max_num_nodes = nodes_per_graph;

    pyg::GCNConvOptions gcn_options{};
    gcn_options.in_channels = 25;
    gcn_options.out_channels = 32;
    gcn_options.cached = true;
    pyg::GCNConv gcn(gcn_options);
    gcn->to(device, torch::kFloat32);
    fill_linear(gcn->lin->weight, -0.04, 0.00011);
    fill_linear(gcn->bias, -0.02, 0.001);
    gcn->eval();

    pyg::GATConvOptions gat_options{};
    gat_options.in_channels = 25;
    gat_options.out_channels = 8;
    gat_options.heads = 4;
    gat_options.concat = true;
    gat_options.dropout = 0.0;
    gat_options.edge_dim = 2;
    gat_options.residual = true;
    pyg::GATConv gat(gat_options);
    gat->to(device, torch::kFloat32);
    fill_linear(gat->lin->weight, -0.035, 0.00009);
    fill_linear(gat->att_src, -0.025, 0.0013);
    fill_linear(gat->att_dst, 0.019, -0.0011);
    fill_linear(gat->lin_edge->weight, -0.03, 0.0007);
    fill_linear(gat->att_edge, 0.02, -0.0012);
    fill_linear(gat->res->weight, -0.025, 0.00008);
    fill_linear(gat->bias, 0.01, -0.0005);
    gat->eval();
    return {device, x, batch, edge_index, edge_attr, packed, gcn, gat};
}

torch::Tensor signature(const torch::Tensor& value) {
    const torch::Tensor flat = value.flatten();
    const std::int64_t count = std::min<std::int64_t>(16, flat.numel());
    return torch::cat({
        flat.slice(0, 0, count).to(torch::kFloat64),
        value.to(torch::kFloat64).sum().reshape({1}),
        value.to(torch::kFloat64).square().sum().reshape({1}),
    }).to(torch::kCPU).contiguous();
}

template <typename Function>
BenchmarkRecord measure(
    std::string name,
    Function&& function,
    const Arguments& arguments,
    const torch::Device& device) {
    torch::Tensor output;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = function();
        }
    }
    synchronize(device);
    std::vector<std::int64_t> samples;
    samples.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        synchronize(device);
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = function();
        }
        synchronize(device);
        const auto stopped = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
            stopped - started).count());
    }
    std::vector<std::int64_t> ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    const std::size_t middle = ordered.size() / 2U;
    const double median = ordered.size() % 2U == 1U
        ? static_cast<double>(ordered[middle])
        : (static_cast<double>(ordered[middle - 1U])
           + static_cast<double>(ordered[middle])) / 2.0;
    return {std::move(name), std::move(samples), median, signature(output)};
}

std::vector<BenchmarkRecord> benchmark_records(
    const Arguments& arguments,
    const torch::Device& device) {
    BenchmarkFixture fixture = make_benchmark_fixture(device);
    torch::InferenceMode inference;
    std::vector<BenchmarkRecord> records;
    records.reserve(3);
    records.push_back(measure(
        "utility_pipeline",
        [&]() {
            const pyg::SortedEdges sorted = pyg::sort_edge_index(
                fixture.edge_index, fixture.edge_attr, fixture.x.size(0), true);
            const pyg::LoopResult loops = pyg::add_remaining_self_loops(
                sorted.edge_index, sorted.edge_attr,
                pyg::SelfLoopFill::reduce(pyg::SelfLoopFillKind::mean),
                fixture.x.size(0));
            const torch::Tensor probabilities = pyg::segment_softmax(
                *loops.edge_attr, loops.edge_index.select(0, 1), fixture.x.size(0));
            const pyg::DenseBatchResult dense = pyg::to_dense_batch(
                fixture.packed, 0.0);
            const torch::Tensor pooled = pyg::global_mean_pool(
                fixture.x, fixture.batch, 8);
            return torch::cat({
                probabilities.flatten().slice(0, 0, 32),
                dense.values.flatten().slice(0, 0, 32),
                dense.mask.to(torch::kFloat32).sum().reshape({1}),
                pooled.flatten(),
            });
        },
        arguments,
        device));
    records.push_back(measure(
        "gcn",
        [&]() { return fixture.gcn->forward(fixture.x, fixture.edge_index); },
        arguments,
        device));
    records.push_back(measure(
        "gat",
        [&]() {
            return fixture.gat->forward(
                fixture.x, fixture.edge_index, fixture.edge_attr);
        },
        arguments,
        device));
    return records;
}

std::string dtype_name(const torch::ScalarType dtype) {
    switch (dtype) {
    case torch::kBool: return "bool";
    case torch::kInt64: return "int64";
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    default: throw std::runtime_error("unsupported harness output dtype");
    }
}

std::string json_escape(const std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (const char value : input) {
        if (value == '\\' || value == '"') output.push_back('\\');
        output.push_back(value);
    }
    return output;
}

void emit_shape(std::ostream& output, const torch::Tensor& value) {
    output << '[';
    for (std::int64_t dimension = 0; dimension < value.dim(); ++dimension) {
        if (dimension != 0) output << ',';
        output << value.size(dimension);
    }
    output << ']';
}

void emit_values(std::ostream& output, const torch::Tensor& source) {
    const torch::Tensor value = source.contiguous().view({-1});
    output << '[';
    for (std::int64_t index = 0; index < value.numel(); ++index) {
        if (index != 0) output << ',';
        switch (value.scalar_type()) {
        case torch::kBool:
            output << (value.const_data_ptr<bool>()[index] ? "true" : "false");
            break;
        case torch::kInt64:
            output << value.const_data_ptr<std::int64_t>()[index];
            break;
        case torch::kFloat32:
            output << std::setprecision(std::numeric_limits<float>::max_digits10)
                   << value.const_data_ptr<float>()[index];
            break;
        case torch::kFloat64:
            output << std::setprecision(std::numeric_limits<double>::max_digits10)
                   << value.const_data_ptr<double>()[index];
            break;
        default:
            throw std::runtime_error("unsupported harness output dtype");
        }
    }
    output << ']';
}

void emit_tensor(std::ostream& output, const TensorRecord& record) {
    output << "{\"name\":\"" << json_escape(record.name) << "\",\"shape\":";
    emit_shape(output, record.value);
    output << ",\"dtype\":\"" << dtype_name(record.value.scalar_type())
           << "\",\"values\":";
    emit_values(output, record.value);
    output << '}';
}

void emit_parity(
    const Arguments& arguments,
    const torch::Device& device,
    const std::vector<TensorRecord>& records) {
    std::cout << "{\"component\":\"pyg_lite\",\"mode\":\"parity\""
              << ",\"libtorch_version\":\"" << json_escape(VIRNE_LIBTORCH_VERSION)
              << "\",\"device\":\"" << (device.is_cuda() ? "cuda" : "cpu")
              << "\",\"threads\":" << arguments.threads << ",\"tensors\":[";
    for (std::size_t index = 0; index < records.size(); ++index) {
        if (index != 0U) std::cout << ',';
        emit_tensor(std::cout, records[index]);
    }
    std::cout << "]}\n";
}

void emit_benchmark(
    const Arguments& arguments,
    const torch::Device& device,
    const std::vector<BenchmarkRecord>& records) {
    std::cout << "{\"component\":\"pyg_lite\",\"mode\":\"benchmark\""
              << ",\"libtorch_version\":\"" << json_escape(VIRNE_LIBTORCH_VERSION)
              << "\",\"device\":\"" << (device.is_cuda() ? "cuda" : "cpu")
              << "\",\"threads\":" << arguments.threads
              << ",\"warmups\":" << arguments.warmups
              << ",\"repetitions\":" << arguments.repetitions
              << ",\"iterations\":" << arguments.iterations
              << ",\"benchmarks\":[";
    for (std::size_t record_index = 0; record_index < records.size(); ++record_index) {
        if (record_index != 0U) std::cout << ',';
        const BenchmarkRecord& record = records[record_index];
        std::cout << "{\"name\":\"" << json_escape(record.name)
                  << "\",\"samples_ns\":[";
        for (std::size_t index = 0; index < record.samples_ns.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << record.samples_ns[index];
        }
        std::cout << "],\"median_ns\":" << std::setprecision(17)
                  << record.median_ns << ",\"signature\":";
        emit_tensor(std::cout, TensorRecord{"signature", record.signature});
        std::cout << '}';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);
        torch::manual_seed(0);
        const torch::Device device = resolve_device(arguments.device);
        if (arguments.mode == Mode::parity) {
            emit_parity(arguments, device, parity_records(device));
        } else {
            emit_benchmark(arguments, device, benchmark_records(arguments, device));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pyg_lite_harness error: " << error.what() << '\n';
        return 2;
    }
}
