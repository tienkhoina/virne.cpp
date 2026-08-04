#include "../virne/solver/learning/pyg_lite/message_passing.h"

#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    double median_ns = 0.0;
    double signature = 0.0;
};

[[noreturn]] void usage_error(const std::string_view message) {
    throw std::invalid_argument(
        std::string(message)
        + "\nusage: vne_message_passing_harness parity|benchmark "
          "[--device cpu|cuda|auto] [--threads N] [--warmups N] "
          "[--repetitions N] [--iterations N]");
}

int positive(const char* value, const char* option) {
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
    if (mode == "parity") result.mode = Mode::parity;
    else if (mode == "benchmark") result.mode = Mode::benchmark;
    else usage_error("mode must be parity or benchmark");
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        auto read_positive = [&](const char* name) {
            if (++i >= argc) usage_error(std::string(name) + " requires a value");
            return positive(argv[i], name);
        };
        if (option == "--device") {
            if (++i >= argc) usage_error("--device requires a value");
            result.device = argv[i];
            if (result.device != "cpu" && result.device != "cuda"
                && result.device != "auto") {
                usage_error("--device requires cpu, cuda, or auto");
            }
        } else if (option == "--threads") result.threads = read_positive("--threads");
        else if (option == "--warmups") result.warmups = read_positive("--warmups");
        else if (option == "--repetitions") result.repetitions = read_positive("--repetitions");
        else if (option == "--iterations") result.iterations = read_positive("--iterations");
        else usage_error("unknown option");
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
            ? torch::kCUDA : torch::kCPU);
}

void synchronize(const torch::Device& device) {
    if (device.is_cuda()) torch::cuda::synchronize();
}

torch::TensorOptions values(
    const torch::Device& device,
    const torch::ScalarType dtype) {
    return torch::TensorOptions().device(device).dtype(dtype);
}

torch::TensorOptions indices(const torch::Device& device) {
    return torch::TensorOptions().device(device).dtype(torch::kInt64);
}

void fill_linear(torch::Tensor tensor, const double first, const double step) {
    torch::NoGradGuard guard;
    tensor.copy_(torch::arange(tensor.numel(), tensor.options()).reshape(tensor.sizes())
        * step + first);
}

void append(
    std::vector<TensorRecord>& records,
    std::string name,
    const torch::Tensor& value) {
    records.push_back({
        std::move(name), value.detach().to(torch::kCPU).contiguous()});
}

void configure_graph_vine(
    pyg::GraphViNEConv& layer,
    const torch::Dtype dtype) {
    layer->to(dtype);
    fill_linear(layer->lin_rel->weight, -0.23, 0.037);
    fill_linear(layer->lin_rel->bias, 0.11, -0.021);
    fill_linear(layer->lin_root->weight, 0.31, -0.029);
}

void configure_edge_fusion(
    pyg::EdgeFusionGATConv& layer,
    const torch::Dtype dtype) {
    layer->to(dtype);
    fill_linear(layer->lin_src->weight, -0.17, 0.013);
    if (layer->lin_edge) fill_linear(layer->lin_edge->weight, 0.07, -0.019);
    auto& sequence = layer->pre_nns->at<torch::nn::SequentialImpl>(0);
    fill_linear(sequence.at<torch::nn::LinearImpl>(0).weight, -0.08, 0.017);
    fill_linear(sequence.at<torch::nn::LinearImpl>(0).bias, 0.03, -0.011);
    fill_linear(sequence.at<torch::nn::LinearImpl>(2).weight, 0.14, -0.009);
    fill_linear(sequence.at<torch::nn::LinearImpl>(2).bias, -0.02, 0.007);
    if (layer->bias.defined()) fill_linear(layer->bias, 0.05, 0.004);
}

struct SmallFixture {
    torch::Tensor x;
    torch::Tensor edge_index;
    torch::Tensor edge_weight;
    torch::Tensor edge_attr;
};

SmallFixture small_fixture(const torch::Device& device, const torch::Dtype dtype) {
    const auto index_options = indices(device);
    const auto value_options = values(device, dtype);
    SmallFixture fixture{};
    fixture.x = torch::arange(12, value_options).reshape({4, 3}) * 0.07 - 0.31;
    fixture.edge_index = torch::tensor(
        {{0, 2, 1, 3, 2, 0}, {1, 1, 2, 0, 3, 3}}, index_options);
    fixture.edge_weight = torch::arange(6, value_options) * 0.13 + 0.4;
    fixture.edge_attr = torch::arange(12, value_options).reshape({6, 2}) * 0.03 - 0.2;
    return fixture;
}

std::vector<TensorRecord> parity_records(const torch::Device& device) {
    const torch::Dtype dtype = torch::kFloat64;
    const SmallFixture fixture = small_fixture(device, dtype);
    std::vector<TensorRecord> records;

    pyg::GraphViNEConvOptions graph_options{};
    graph_options.in_channels = 3;
    graph_options.out_channels = 4;
    graph_options.normalize = false;
    pyg::GraphViNEConv graph(graph_options);
    configure_graph_vine(graph, dtype);
    append(records, "graph_vine/output", graph->forward(
        fixture.x, fixture.edge_index, fixture.edge_weight));

    graph_options.normalize = true;
    pyg::GraphViNEConv normalized(graph_options);
    configure_graph_vine(normalized, dtype);
    append(records, "graph_vine/normalized", normalized->forward(
        fixture.x, fixture.edge_index, fixture.edge_weight));

    pyg::EdgeFusionGATConvOptions edge_options{};
    edge_options.in_channels = 3;
    edge_options.out_channels = 4;
    edge_options.heads = 1;
    edge_options.edge_dim = 2;
    edge_options.add_self_loops = true;
    edge_options.fill_value = pyg::SelfLoopFill::reduce(
        pyg::SelfLoopFillKind::mean);
    pyg::EdgeFusionGATConv edge_layer(edge_options);
    configure_edge_fusion(edge_layer, dtype);
    append(records, "edge_fusion/output", edge_layer->forward(
        fixture.x, fixture.edge_index, fixture.edge_attr));
    return records;
}

SmallFixture benchmark_fixture(const torch::Device& device) {
    constexpr std::int64_t nodes = 512;
    constexpr std::int64_t edges = 8192;
    constexpr std::int64_t channels = 16;
    const auto value_options = values(device, torch::kFloat32);
    const auto index_options = indices(device);
    SmallFixture fixture{};
    fixture.x = torch::arange(nodes * channels, value_options)
        .reshape({nodes, channels}) * 0.0007 - 0.3;
    const torch::Tensor ids = torch::arange(edges, index_options);
    fixture.edge_index = torch::stack({ids.remainder(nodes),
        (ids * 17 + 3).remainder(nodes)}, 0);
    fixture.edge_weight = torch::arange(edges, value_options) * 0.00003 + 0.4;
    fixture.edge_attr = torch::arange(edges * 2, value_options)
        .reshape({edges, 2}) * 0.00002 - 0.1;
    return fixture;
}

template <typename Fn>
BenchmarkRecord measure(
    std::string name,
    Fn&& fn,
    const torch::Device& device,
    const Arguments& arguments) {
    for (int i = 0; i < arguments.warmups; ++i) {
        fn();
        synchronize(device);
    }
    std::vector<std::int64_t> samples;
    samples.reserve(static_cast<std::size_t>(arguments.repetitions));
    torch::Tensor output;
    for (int rep = 0; rep < arguments.repetitions; ++rep) {
        synchronize(device);
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < arguments.iterations; ++i) output = fn();
        synchronize(device);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin).count() / arguments.iterations);
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2U;
    const double median = samples.size() % 2U == 1U
        ? static_cast<double>(samples[middle])
        : (static_cast<double>(samples[middle - 1U])
           + static_cast<double>(samples[middle])) / 2.0;
    synchronize(device);
    return {std::move(name), median, output.sum().item<double>()};
}

std::vector<BenchmarkRecord> benchmark_records(
    const torch::Device& device,
    const Arguments& arguments) {
    const SmallFixture fixture = benchmark_fixture(device);
    pyg::GraphViNEConvOptions graph_options{};
    graph_options.in_channels = 16;
    graph_options.out_channels = 16;
    pyg::GraphViNEConv graph(graph_options);
    configure_graph_vine(graph, torch::kFloat32);
    pyg::EdgeFusionGATConvOptions edge_options{};
    edge_options.in_channels = 16;
    edge_options.out_channels = 16;
    edge_options.heads = 1;
    edge_options.edge_dim = 2;
    edge_options.add_self_loops = true;
    edge_options.fill_value = pyg::SelfLoopFill::reduce(
        pyg::SelfLoopFillKind::mean);
    pyg::EdgeFusionGATConv edge_layer(edge_options);
    configure_edge_fusion(edge_layer, torch::kFloat32);
    torch::InferenceMode inference;
    return {
        measure("graph_vine", [&] { return graph->forward(
            fixture.x, fixture.edge_index, fixture.edge_weight); },
            device, arguments),
        measure("edge_fusion", [&] { return edge_layer->forward(
            fixture.x, fixture.edge_index, fixture.edge_attr); },
            device, arguments),
    };
}

void emit_tensor(std::ostream& output, const torch::Tensor& value) {
    const torch::Tensor flat = value.reshape(-1).contiguous();
    output << "\"shape\":[";
    for (std::int64_t i = 0; i < value.dim(); ++i) {
        if (i) output << ',';
        output << value.size(i);
    }
    output << "],\"values\":[";
    for (std::int64_t i = 0; i < flat.numel(); ++i) {
        if (i) output << ',';
        output << std::setprecision(17) << flat.index({i}).item<double>();
    }
    output << ']';
}

void emit_parity(const std::vector<TensorRecord>& records) {
    std::cout << "{\"component\":\"message_passing\",\"mode\":\"parity\",\"records\":[";
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"name\":\"" << records[i].name << "\",";
        emit_tensor(std::cout, records[i].value);
        std::cout << '}';
    }
    std::cout << "]}\n";
}

void emit_benchmark(const std::vector<BenchmarkRecord>& records) {
    std::cout << "{\"component\":\"message_passing\",\"mode\":\"benchmark\",\"records\":[";
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"name\":\"" << records[i].name
                  << "\",\"median_ns\":" << std::setprecision(17)
                  << records[i].median_ns << ",\"signature\":"
                  << records[i].signature << '}';
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
        if (arguments.mode == Mode::parity) emit_parity(parity_records(device));
        else emit_benchmark(benchmark_records(device, arguments));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "message_passing_harness error: " << error.what() << '\n';
        return 2;
    }
}
