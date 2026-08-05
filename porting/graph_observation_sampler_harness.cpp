#include "../virne/solver/learning/rl_core/graph_observation_sampler.h"

#include <torch/torch.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rl = virne::solver::learning::rl_core;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    std::string device = "cpu";
    int threads = 1;
    int warmups = 1;
    int repetitions = 3;
    int iterations = 10;
};

struct TensorSummary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::string device;
    std::uint64_t checksum = fnv_offset;
    std::size_t bytes = 0U;
    std::vector<std::string> bits;
};

struct CaseResult {
    std::string name;
    TensorSummary output;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    std::vector<CaseResult> output;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: vne_graph_observation_sampler_harness [--benchmark]"
          " [--device cpu|cuda|auto] [--threads N] [--warmups N]"
          " [--repetitions N] [--iterations N]");
}

int positive_argument(const char* value, const char* option) {
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
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto read_positive = [&](const char* option) {
            if (++index >= argc) usage_error(std::string(option) + " requires a value");
            return positive_argument(argv[index], option);
        };
        auto read_string = [&](const char* option) {
            if (++index >= argc) usage_error(std::string(option) + " requires a value");
            return std::string(argv[index]);
        };
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--device") {
            result.device = read_string("--device");
            if (result.device != "cpu" && result.device != "cuda"
                && result.device != "auto") {
                usage_error("--device must be cpu, cuda, or auto");
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
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

torch::Device select_device(const std::string& requested) {
    if (requested == "cuda"
        || (requested == "auto" && torch::cuda::is_available())) {
        if (!torch::cuda::is_available()) usage_error("CUDA requested but unavailable");
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

std::string dtype_name(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    case torch::kInt64: return "int64";
    case torch::kBool: return "bool";
    default: throw std::runtime_error("unsupported graph sampler summary dtype");
    }
}

template <typename UInt>
std::string bits_hex(const UInt bits) {
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(static_cast<int>(sizeof(UInt) * 2U))
           << static_cast<std::uint64_t>(bits);
    return output.str();
}

template <typename Value, typename UInt>
void append_bits(const torch::Tensor& tensor, std::vector<std::string>& output) {
    const auto* values = tensor.const_data_ptr<Value>();
    for (std::int64_t index = 0; index < tensor.numel(); ++index) {
        UInt bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        output.push_back(bits_hex(bits));
    }
}

TensorSummary summarize(const torch::Tensor& source, const bool include_bits = true) {
    const auto tensor = source.detach().to(torch::kCPU).contiguous();
    TensorSummary result;
    result.shape = tensor.sizes().vec();
    result.dtype = dtype_name(tensor.scalar_type());
    result.device = source.device().str();
    result.bytes = static_cast<std::size_t>(tensor.nbytes());
    const auto* bytes = static_cast<const std::uint8_t*>(tensor.const_data_ptr());
    for (std::size_t index = 0U; index < result.bytes; ++index) {
        result.checksum ^= bytes[index];
        result.checksum *= fnv_prime;
    }
    if (!include_bits) return result;
    result.bits.reserve(static_cast<std::size_t>(tensor.numel()));
    if (tensor.scalar_type() == torch::kFloat32) {
        append_bits<float, std::uint32_t>(tensor, result.bits);
    } else if (tensor.scalar_type() == torch::kFloat64) {
        append_bits<double, std::uint64_t>(tensor, result.bits);
    } else if (tensor.scalar_type() == torch::kInt64) {
        append_bits<std::int64_t, std::uint64_t>(tensor, result.bits);
    } else {
        const auto* values = tensor.const_data_ptr<bool>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            result.bits.push_back(bits_hex(static_cast<std::uint8_t>(values[index])));
        }
    }
    return result;
}

torch::Tensor scalar_i64(const std::int64_t value, const torch::Device& device) {
    return torch::tensor(
        std::vector<std::int64_t>{value},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
}

pyg::Data graph(const std::int64_t seed, const std::int64_t nodes) {
    pyg::Data result;
    result.x = torch::arange(
        seed * 100, seed * 100 + nodes * 2, torch::kFloat32).reshape({nodes, 2});
    std::vector<std::int64_t> edges;
    for (std::int64_t node = 0; node + 1 < nodes; ++node) {
        edges.insert(edges.end(), {node, node + 1, node + 1, node});
    }
    result.edge_index = edges.empty()
        ? torch::empty({2, 0}, torch::kInt64)
        : torch::tensor(edges, torch::kInt64).reshape({-1, 2}).t().contiguous();
    result.edge_attr = torch::arange(
        result.edge_index.size(1), torch::kFloat32).reshape({-1, 1})
                           .add_(static_cast<float>(seed));
    result.validate();
    return result;
}

pyg::HeteroRelationData relation(
    const std::vector<std::int64_t>& edges,
    const float seed) {
    pyg::HeteroRelationData result;
    result.edge_index = edges.empty()
        ? torch::empty({2, 0}, torch::kInt64)
        : torch::tensor(edges, torch::kInt64).reshape({-1, 2}).t().contiguous();
    result.edge_attr = torch::arange(
        result.edge_index.size(1), torch::kFloat32).reshape({-1, 1}).add_(seed);
    return result;
}

pyg::HeteroData hetero_graph(
    const std::int64_t seed,
    const std::int64_t p_nodes,
    const std::int64_t v_nodes,
    const bool augment_virtual,
    const bool augment_physical) {
    pyg::HeteroData result;
    result.node(pyg::HeteroNodeType::physical) = torch::arange(
        seed * 100, seed * 100 + p_nodes * 2, torch::kFloat32).reshape({p_nodes, 2});
    result.node(pyg::HeteroNodeType::virtual_node) = torch::arange(
        seed * 10, seed * 10 + v_nodes, torch::kFloat32).reshape({v_nodes, 1});

    std::vector<std::int64_t> p_edges;
    for (std::int64_t node = 0; node + 1 < p_nodes; ++node) {
        p_edges.insert(p_edges.end(), {node, node + 1, node + 1, node});
    }
    if (augment_physical && p_nodes != 0) p_edges.insert(p_edges.end(), {0, 0});
    std::vector<std::int64_t> v_edges;
    for (std::int64_t node = 0; node + 1 < v_nodes; ++node) {
        v_edges.insert(v_edges.end(), {node, node + 1, node + 1, node});
    }
    if (augment_virtual && v_nodes != 0) v_edges.insert(v_edges.end(), {0, 0});
    std::vector<std::int64_t> mapping;
    if (p_nodes != 0 && v_nodes != 0 && seed != 2) mapping = {0, 0};
    std::vector<std::int64_t> imaginary;
    if (p_nodes != 0 && v_nodes != 0 && seed != 1) {
        imaginary = {v_nodes - 1, p_nodes - 1};
    }

    result.relation(pyg::HeteroRelationId::p_connect_p) =
        relation(p_edges, static_cast<float>(seed) + 0.1F);
    result.relation(pyg::HeteroRelationId::v_connect_v) =
        relation(v_edges, static_cast<float>(seed) + 0.2F);
    result.relation(pyg::HeteroRelationId::v_mapping_p) =
        relation(mapping, static_cast<float>(seed) + 0.3F);
    std::vector<std::int64_t> reverse_mapping;
    for (std::size_t edge = 0U; edge + 1U < mapping.size(); edge += 2U) {
        reverse_mapping.insert(reverse_mapping.end(), {mapping[edge + 1U], mapping[edge]});
    }
    result.relation(pyg::HeteroRelationId::p_mapping_v) =
        relation(reverse_mapping, static_cast<float>(seed) + 0.3F);
    result.relation(pyg::HeteroRelationId::v_imaginary_p) =
        relation(imaginary, static_cast<float>(seed) + 0.4F);
    std::vector<std::int64_t> reverse_imaginary;
    for (std::size_t edge = 0U; edge + 1U < imaginary.size(); edge += 2U) {
        reverse_imaginary.insert(
            reverse_imaginary.end(), {imaginary[edge + 1U], imaginary[edge]});
    }
    result.relation(pyg::HeteroRelationId::p_imaginary_v) =
        relation(reverse_imaginary, static_cast<float>(seed) + 0.4F);
    result.validate();
    return result;
}

void append_case(
    std::vector<CaseResult>& output,
    std::string name,
    const torch::Tensor& tensor,
    const bool include_bits = true) {
    output.push_back({std::move(name), summarize(tensor, include_bits)});
}

void append_batch_cases(
    const std::string& prefix,
    const pyg::Batch& batch,
    std::vector<CaseResult>& output,
    const bool include_bits = true) {
    append_case(output, prefix + ".x", batch.x, include_bits);
    append_case(output, prefix + ".edge_index", batch.edge_index, include_bits);
    append_case(output, prefix + ".edge_attr", *batch.edge_attr, include_bits);
    append_case(output, prefix + ".batch", *batch.batch, include_bits);
    append_case(output, prefix + ".ptr", *batch.ptr, include_bits);
    append_case(output, prefix + ".edge_ptr", batch.edge_ptr, include_bits);
    append_case(output, prefix + ".num_graphs",
                scalar_i64(batch.num_graphs, batch.x.device()), include_bits);
    append_case(output, prefix + ".max_num_nodes",
                scalar_i64(batch.max_num_nodes, batch.x.device()), include_bits);
}

constexpr std::array<const char*, pyg::kHeteroRelations> relation_names{
    "p_connect_p", "v_connect_v", "v_mapping_p",
    "p_mapping_v", "v_imaginary_p", "p_imaginary_v"};

void append_hetero_cases(
    const std::string& prefix,
    const pyg::HeteroBatch& batch,
    std::vector<CaseResult>& output) {
    constexpr std::array<const char*, pyg::kHeteroNodeTypes> node_names{"p", "v"};
    for (std::size_t type = 0U; type < pyg::kHeteroNodeTypes; ++type) {
        const auto node_prefix = prefix + "." + node_names[type];
        append_case(output, node_prefix + ".x", batch.x[type]);
        append_case(output, node_prefix + ".batch", *batch.batch[type]);
        append_case(output, node_prefix + ".ptr", *batch.ptr[type]);
        append_case(output, node_prefix + ".node_ptr", batch.node_ptr[type]);
    }
    for (std::size_t id = 0U; id < pyg::kHeteroRelations; ++id) {
        const auto relation_prefix = prefix + ".relation." + relation_names[id];
        append_case(output, relation_prefix + ".edge_index",
                    batch.relations[id].edge_index);
        append_case(output, relation_prefix + ".edge_attr",
                    *batch.relations[id].edge_attr);
    }
    append_case(output, prefix + ".num_graphs",
                scalar_i64(batch.num_graphs, batch.x[0].device()));
    append_case(output, prefix + ".max_num_nodes.p",
                scalar_i64(batch.max_num_nodes[0], batch.x[0].device()));
    append_case(output, prefix + ".max_num_nodes.v",
                scalar_i64(batch.max_num_nodes[1], batch.x[1].device()));
}

std::vector<CaseResult> run_cases(const torch::Device& device) {
    const auto indices = rl::prepare_graph_indices(
        std::vector<std::int64_t>{2, 0, 2, -2}, 3);
    const auto p_batch = pyg::Batch::from_data_list(
        {graph(1, 2), graph(2, 4), graph(3, 3)}).to(device);
    rl::PreparedBatchObservationSampler homogeneous_sampler(p_batch);
    const auto homogeneous_plan = homogeneous_sampler.prepare_selection(indices);
    const auto homogeneous = homogeneous_sampler.sample(homogeneous_plan, device);

    rl::DualGraphTensorObservation dual;
    dual.p_net = p_batch;
    dual.v_net = pyg::Batch::from_data_list(
        {graph(11, 1), graph(12, 3), graph(13, 2)}).to(device);
    dual.general.curr_v_node_id = torch::tensor({5, 6, 7}, torch::kInt64).to(device);
    dual.general.action_mask = torch::tensor(
        {{1, 0}, {0, 1}, {1, 1}}, torch::kBool).to(device);
    dual.general.v_net_size = torch::tensor({2.0F, 4.0F, 3.0F}).to(device);
    rl::PreparedDualGraphObservationSampler dual_sampler(dual);
    const auto dual_plan = dual_sampler.prepare_selection(indices);
    const auto dual_sample = dual_sampler.sample(dual_plan, device);

    const std::array<std::int64_t, 3> p_counts{2, 4, 3};
    const std::array<std::int64_t, 3> v_counts{1, 3, 2};
    std::vector<pyg::HeteroData> base;
    std::vector<pyg::HeteroData> virtual_augmented;
    std::vector<pyg::HeteroData> physical_augmented;
    for (std::int64_t graph_id = 0; graph_id < 3; ++graph_id) {
        base.push_back(hetero_graph(
            graph_id, p_counts[graph_id], v_counts[graph_id], false, false));
        virtual_augmented.push_back(hetero_graph(
            graph_id, p_counts[graph_id], v_counts[graph_id], true, false));
        physical_augmented.push_back(hetero_graph(
            graph_id, p_counts[graph_id], v_counts[graph_id], false, true));
    }
    const auto base_batch = pyg::HeteroBatch::from_data_list(base).to(device);
    const auto virtual_batch = pyg::HeteroBatch::from_data_list(virtual_augmented).to(device);
    const auto physical_batch = pyg::HeteroBatch::from_data_list(physical_augmented).to(device);
    rl::HeteroGraphTensorObservation hetero;
    hetero.base = base_batch;
    hetero.virtual_connect_augmented = virtual_batch.relation(
        pyg::HeteroRelationId::v_connect_v);
    hetero.physical_connect_augmented = physical_batch.relation(
        pyg::HeteroRelationId::p_connect_p);
    hetero.general.curr_v_node_id = torch::tensor({0, 1, 2}, torch::kInt64).to(device);
    hetero.general.action_mask = torch::ones(
        {3, 4}, torch::TensorOptions().dtype(torch::kBool).device(device));
    hetero.general.v_net_size = torch::tensor({1.0F, 3.0F, 2.0F})
                                    .reshape({3, 1}).to(device);
    rl::PreparedHeteroGraphObservationSampler hetero_sampler(hetero);
    const auto hetero_plan = hetero_sampler.prepare_selection(indices);
    const auto hetero_sample = hetero_sampler.sample(hetero_plan, device);

    std::vector<CaseResult> output;
    append_batch_cases("homogeneous", homogeneous, output);
    append_batch_cases("dual.p_net", dual_sample.p_net, output);
    append_batch_cases("dual.v_net", dual_sample.v_net, output);
    append_case(output, "dual.general.curr_v_node_id",
                *dual_sample.general.curr_v_node_id);
    append_case(output, "dual.general.action_mask", *dual_sample.general.action_mask);
    append_case(output, "dual.general.v_net_size", *dual_sample.general.v_net_size);
    append_hetero_cases(
        "hetero.base", hetero_sample.view(rl::HeteroObservationViewId::base), output);
    append_hetero_cases(
        "hetero.virtual_augmented",
        hetero_sample.view(rl::HeteroObservationViewId::virtual_augmented), output);
    append_hetero_cases(
        "hetero.physical_augmented",
        hetero_sample.view(rl::HeteroObservationViewId::physical_augmented), output);
    append_case(output, "hetero.general.curr_v_node_id",
                *hetero_sample.general.curr_v_node_id);
    append_case(output, "hetero.general.action_mask",
                *hetero_sample.general.action_mask);
    append_case(output, "hetero.general.v_net_size",
                *hetero_sample.general.v_net_size);
    return output;
}

BenchmarkResult run_benchmark(const Arguments& arguments, const torch::Device& device) {
    constexpr std::int64_t graph_count = 256;
    constexpr std::int64_t selected_count = 128;
    std::vector<pyg::Data> graphs;
    graphs.reserve(graph_count);
    for (std::int64_t graph_id = 0; graph_id < graph_count; ++graph_id) {
        graphs.push_back(graph(graph_id + 1, 24 + graph_id % 9));
    }
    const auto source = pyg::Batch::from_data_list(graphs).to(device);
    rl::PreparedBatchObservationSampler sampler(source);
    std::vector<std::int64_t> values;
    values.reserve(selected_count);
    for (std::int64_t index = 0; index < selected_count; ++index) {
        values.push_back((index * 37) % graph_count);
    }
    const auto indices = rl::prepare_graph_indices(values, graph_count);
    // The cold selection plan is deliberately outside every timed region.
    const auto selection = sampler.prepare_selection(indices);

    torch::NoGradGuard guard;
    pyg::Batch sampled;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            sampled = sampler.sample(selection, device);
        }
    }
    if (device.is_cuda()) torch::cuda::synchronize(device.index());

    BenchmarkResult result;
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        if (device.is_cuda()) torch::cuda::synchronize(device.index());
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            sampled = sampler.sample(selection, device);
        }
        if (device.is_cuda()) torch::cuda::synchronize(device.index());
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
    }
    append_batch_cases("sample", sampled, result.output, false);
    return result;
}

void emit_shape(std::ostream& output, const std::vector<std::int64_t>& shape) {
    output << '[';
    for (std::size_t index = 0U; index < shape.size(); ++index) {
        if (index != 0U) output << ',';
        output << shape[index];
    }
    output << ']';
}

void emit_summary(std::ostream& output, const TensorSummary& summary) {
    output << "{\"shape\":";
    emit_shape(output, summary.shape);
    output << ",\"dtype\":\"" << summary.dtype
           << "\",\"device\":\"" << summary.device
           << "\",\"checksum\":\"" << summary.checksum
           << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0U; index < summary.bits.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << summary.bits[index] << '"';
    }
    output << "]}";
}

void emit_cases(std::ostream& output, const std::vector<CaseResult>& cases) {
    output << '[';
    for (std::size_t index = 0U; index < cases.size(); ++index) {
        if (index != 0U) output << ',';
        output << "{\"name\":\"" << cases[index].name << "\",\"output\":";
        emit_summary(output, cases[index].output);
        output << '}';
    }
    output << ']';
}

void emit_report(
    const Arguments& arguments,
    const torch::Device& device,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult* benchmark) {
    std::cout << "{\"component\":\"graph_observation_sampler\",\"device\":\""
              << (device.is_cuda() ? "cuda" : "cpu")
              << "\",\"threads\":" << arguments.threads << ",\"cases\":";
    emit_cases(std::cout, cases);
    if (benchmark != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"prepared_homogeneous_256_to_128\""
                  << ",\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"iterations\":" << arguments.iterations
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < benchmark->samples_ns.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << benchmark->samples_ns[index];
        }
        std::cout << "],\"output\":";
        emit_cases(std::cout, benchmark->output);
        std::cout << '}';
    }
    std::cout << "}\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto device = select_device(arguments.device);
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);
        const auto cases = run_cases(device);
        BenchmarkResult benchmark;
        if (arguments.benchmark) benchmark = run_benchmark(arguments, device);
        emit_report(arguments, device, cases,
                    arguments.benchmark ? &benchmark : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph-observation-sampler harness error: "
                  << error.what() << '\n';
        return 2;
    }
}
