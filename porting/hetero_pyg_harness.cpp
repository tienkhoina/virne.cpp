#include "../virne/solver/learning/pyg_lite/hetero_conv.h"

#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pyg = virne::solver::learning::pyg_lite;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Summary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::size_t bytes = 0U;
    std::vector<std::string> bits;
};

std::string dtype_name(const torch::ScalarType dtype) {
    switch (dtype) {
    case torch::kInt64: return "int64";
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    case torch::kBool: return "bool";
    default: throw std::runtime_error("unsupported hetero harness dtype");
    }
}

template <typename UInt>
std::string hex_bits(const UInt value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(static_cast<int>(sizeof(UInt) * 2U))
           << static_cast<std::uint64_t>(value);
    return output.str();
}

Summary summarize(const torch::Tensor& source) {
    const torch::Tensor value = source.detach().to(torch::kCPU).contiguous();
    Summary result;
    result.dtype = dtype_name(value.scalar_type());
    result.bytes = static_cast<std::size_t>(value.numel()) * value.element_size();
    result.shape.reserve(static_cast<std::size_t>(value.dim()));
    for (std::int64_t dimension = 0; dimension < value.dim(); ++dimension) {
        result.shape.push_back(value.size(dimension));
    }
    const auto* bytes = static_cast<const std::uint8_t*>(value.data_ptr());
    for (std::size_t index = 0U; index < result.bytes; ++index) {
        result.checksum ^= bytes[index];
        result.checksum *= fnv_prime;
    }
    result.bits.reserve(static_cast<std::size_t>(value.numel()));
    for (std::int64_t index = 0; index < value.numel(); ++index) {
        if (value.scalar_type() == torch::kFloat32) {
            std::uint32_t bits = 0U;
            std::memcpy(&bits, value.data_ptr<float>() + index, sizeof(bits));
            result.bits.push_back(hex_bits(bits));
        } else if (value.scalar_type() == torch::kFloat64) {
            std::uint64_t bits = 0U;
            std::memcpy(&bits, value.data_ptr<double>() + index, sizeof(bits));
            result.bits.push_back(hex_bits(bits));
        } else if (value.scalar_type() == torch::kInt64) {
            std::uint64_t bits = 0U;
            std::memcpy(&bits, value.data_ptr<std::int64_t>() + index, sizeof(bits));
            result.bits.push_back(hex_bits(bits));
        } else {
            result.bits.push_back(value.data_ptr<bool>()[index] ? "01" : "00");
        }
    }
    return result;
}

void write_string(std::ostream& output, std::string_view value) {
    output << '"';
    for (const char character : value) {
        if (character == '"' || character == '\\') output << '\\';
        output << character;
    }
    output << '"';
}

void write_summary(std::ostream& output, const Summary& summary) {
    output << "{\"shape\":[";
    for (std::size_t index = 0U; index < summary.shape.size(); ++index) {
        if (index != 0U) output << ',';
        output << summary.shape[index];
    }
    output << "],\"dtype\":";
    write_string(output, summary.dtype);
    output << ",\"checksum\":\"" << summary.checksum
           << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0U; index < summary.bits.size(); ++index) {
        if (index != 0U) output << ',';
        write_string(output, summary.bits[index]);
    }
    output << "]}";
}

struct Record {
    std::string name;
    Summary value;
};

struct BenchmarkOptions {
    bool enabled = false;
    int threads = 1;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 25;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    Summary output;
};

int positive_argument(
    const std::string& value,
    const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return parsed;
}

BenchmarkOptions parse_arguments(const int argc, char** argv) {
    BenchmarkOptions result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--benchmark") {
            result.enabled = true;
        } else if (argument == "--threads" && index + 1 < argc) {
            result.threads = positive_argument(argv[++index], "--threads");
        } else if (argument == "--warmups" && index + 1 < argc) {
            result.warmups = positive_argument(argv[++index], "--warmups");
        } else if (argument == "--repetitions" && index + 1 < argc) {
            result.repetitions = positive_argument(argv[++index], "--repetitions");
        } else if (argument == "--iterations" && index + 1 < argc) {
            result.iterations = positive_argument(argv[++index], "--iterations");
        } else {
            throw std::runtime_error("unknown or incomplete harness argument: " + argument);
        }
    }
    return result;
}

torch::Tensor float_tensor(const std::initializer_list<std::initializer_list<float>>& values) {
    std::vector<float> flat;
    std::size_t columns = 0U;
    for (const auto& row : values) {
        if (columns == 0U) columns = row.size();
        if (row.size() != columns) throw std::runtime_error("ragged float fixture");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return torch::tensor(flat, torch::TensorOptions().dtype(torch::kFloat32))
        .reshape({static_cast<std::int64_t>(values.size()), static_cast<std::int64_t>(columns)});
}

torch::Tensor index_tensor(const std::initializer_list<std::initializer_list<std::int64_t>>& values) {
    std::vector<std::int64_t> flat;
    std::size_t columns = 0U;
    for (const auto& row : values) {
        if (columns == 0U) columns = row.size();
        if (row.size() != columns) throw std::runtime_error("ragged index fixture");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return torch::tensor(flat, torch::TensorOptions().dtype(torch::kInt64))
        .reshape({static_cast<std::int64_t>(values.size()), static_cast<std::int64_t>(columns)});
}

void fill_linear(torch::Tensor value, const float first, const float step) {
    torch::NoGradGuard guard;
    const auto sequence = torch::arange(value.numel(), value.options())
        .reshape(value.sizes());
    value.copy_(sequence * step + first);
}

void set_relation(
    pyg::HeteroData& data,
    const pyg::HeteroRelationId id,
    torch::Tensor edge_index,
    torch::Tensor edge_attr) {
    auto& relation = data.relation(id);
    relation.edge_index = std::move(edge_index);
    relation.edge_attr = std::move(edge_attr);
}

pyg::HeteroData graph_zero() {
    pyg::HeteroData result;
    result.node(pyg::HeteroNodeType::physical) = float_tensor({
        {1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}});
    result.node(pyg::HeteroNodeType::virtual_node) = float_tensor({
        {0.5F, 1.5F, 2.5F}, {3.5F, 4.5F, 5.5F}});
    set_relation(result, pyg::HeteroRelationId::p_connect_p,
        index_tensor({{0, 1, 2}, {1, 2, 0}}),
        torch::tensor({{0.5F}, {1.0F}, {1.5F}}));
    set_relation(result, pyg::HeteroRelationId::v_connect_v,
        index_tensor({{0, 1}, {1, 0}}),
        torch::tensor({{0.25F}, {0.75F}}));
    set_relation(result, pyg::HeteroRelationId::v_mapping_p,
        index_tensor({{0, 1}, {1, 2}}),
        torch::tensor({{1.0F}, {1.0F}}));
    set_relation(result, pyg::HeteroRelationId::p_mapping_v,
        index_tensor({{1, 2}, {0, 1}}),
        torch::tensor({{1.0F}, {1.0F}}));
    set_relation(result, pyg::HeteroRelationId::v_imaginary_p,
        index_tensor({{0, 1}, {2, 0}}),
        torch::tensor({{0.2F}, {0.4F}}));
    set_relation(result, pyg::HeteroRelationId::p_imaginary_v,
        index_tensor({{0, 2}, {1, 1}}),
        torch::tensor({{0.2F}, {0.4F}}));
    result.validate();
    return result;
}

pyg::HeteroData graph_one() {
    pyg::HeteroData result;
    result.node(pyg::HeteroNodeType::physical) = float_tensor({
        {-1.0F, -2.0F}, {-3.0F, -4.0F}});
    result.node(pyg::HeteroNodeType::virtual_node) = float_tensor({
        {-0.5F, -1.5F, -2.5F}, {-3.5F, -4.5F, -5.5F},
        {-6.5F, -7.5F, -8.5F}});
    set_relation(result, pyg::HeteroRelationId::p_connect_p,
        index_tensor({{0}, {1}}), torch::tensor({{0.6F}}));
    set_relation(result, pyg::HeteroRelationId::v_connect_v,
        index_tensor({{0, 1}, {1, 2}}),
        torch::tensor({{0.3F}, {0.9F}}));
    set_relation(result, pyg::HeteroRelationId::v_mapping_p,
        index_tensor({{0, 2}, {1, 0}}),
        torch::tensor({{1.0F}, {1.0F}}));
    set_relation(result, pyg::HeteroRelationId::p_mapping_v,
        index_tensor({{0, 1}, {2, 1}}),
        torch::tensor({{1.0F}, {1.0F}}));
    set_relation(result, pyg::HeteroRelationId::v_imaginary_p,
        index_tensor({{1}, {1}}), torch::tensor({{0.7F}}));
    set_relation(result, pyg::HeteroRelationId::p_imaginary_v,
        index_tensor({{0}, {2}}), torch::tensor({{0.7F}}));
    result.validate();
    return result;
}

pyg::BipartiteGATConv make_bipartite() {
    pyg::BipartiteGATConvOptions options;
    options.source_in_channels = 3;
    options.target_in_channels = 2;
    options.out_channels = 2;
    options.heads = 2;
    options.edge_dim = 1;
    options.add_self_loops = false;
    pyg::BipartiteGATConv result(options);
    fill_linear(result->lin_src->weight, -0.20F, 0.03F);
    fill_linear(result->lin_dst->weight, 0.10F, -0.02F);
    fill_linear(result->lin_edge->weight, 0.05F, 0.01F);
    fill_linear(result->att_src, -0.03F, 0.007F);
    fill_linear(result->att_dst, 0.02F, -0.005F);
    fill_linear(result->att_edge, 0.01F, 0.004F);
    fill_linear(result->bias, 0.04F, -0.006F);
    result->eval();
    return result;
}

pyg::HeteroConvOptions hetero_options() {
    pyg::HeteroConvOptions options;
    const std::array<std::pair<std::int64_t, std::int64_t>, pyg::kHeteroRelations> channels{
        std::pair{2, 2}, std::pair{3, 3}, std::pair{3, 2},
        std::pair{2, 3}, std::pair{3, 2}, std::pair{2, 3}};
    for (std::size_t index = 0U; index < pyg::kHeteroRelations; ++index) {
        options.enabled[index] = true;
        auto& relation = options.relation_options[index];
        relation.source_in_channels = channels[index].first;
        relation.target_in_channels = channels[index].second;
        relation.out_channels = 2;
        relation.heads = 1;
        relation.edge_dim = 1;
        relation.add_self_loops = index < 2U;
    }
    return options;
}

void fill_relation_parameters(pyg::HeteroConv& module) {
    for (std::size_t index = 0U; index < pyg::kHeteroRelations; ++index) {
        auto& relation = module->relation(static_cast<pyg::HeteroRelationId>(index));
        const float offset = static_cast<float>(index) * 0.01F;
        fill_linear(relation->lin_src->weight, -0.12F + offset, 0.011F);
        fill_linear(relation->lin_dst->weight, 0.08F - offset, -0.009F);
        fill_linear(relation->lin_edge->weight, 0.03F + offset, 0.006F);
        fill_linear(relation->att_src, -0.02F + offset, 0.004F);
        fill_linear(relation->att_dst, 0.015F - offset, -0.003F);
        fill_linear(relation->att_edge, 0.01F + offset, 0.002F);
        fill_linear(relation->bias, 0.02F - offset, -0.004F);
    }
    module->eval();
}

BenchmarkResult benchmark_hetero(
    pyg::HeteroConvImpl& module,
    const pyg::HeteroData& data,
    const BenchmarkOptions& options) {
    torch::NoGradGuard no_grad;
    for (int warmup = 0; warmup < options.warmups; ++warmup) {
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            static_cast<void>(module.forward(data));
        }
    }
    BenchmarkResult result;
    result.samples_ns.reserve(static_cast<std::size_t>(options.repetitions));
    torch::Tensor final_output;
    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            const auto output = module.forward(data);
            final_output = output.node(pyg::HeteroNodeType::physical);
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start);
        result.samples_ns.push_back(elapsed.count());
    }
    result.output = summarize(final_output);
    return result;
}

void emit_records(
    const std::vector<Record>& records,
    const BenchmarkOptions& options,
    const BenchmarkResult* benchmark) {
    std::cout << "{\"component\":\"hetero_pyg\",\"fixture_version\":1,\"outputs\":{";
    for (std::size_t index = 0U; index < records.size(); ++index) {
        if (index != 0U) std::cout << ',';
        write_string(std::cout, records[index].name);
        std::cout << ':';
        write_summary(std::cout, records[index].value);
    }
    std::cout << '}';
    if (options.enabled && benchmark != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"hetero_conv_forward\",\"threads\":"
                  << options.threads
                  << ",\"warmups\":" << options.warmups
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"iterations\":" << options.iterations
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < benchmark->samples_ns.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << benchmark->samples_ns[index];
        }
        std::cout << "],\"output\":";
        write_summary(std::cout, benchmark->output);
        std::cout << '}';
    }
    std::cout << "}\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        torch::set_num_threads(arguments.threads);
        torch::manual_seed(0);
        const auto first = graph_zero();
        const auto second = graph_one();
        const auto packed = pyg::HeteroBatch::from_data_list({first, second});
        auto bipartite = make_bipartite();
        const auto x_source = first.node(pyg::HeteroNodeType::virtual_node);
        const auto x_target = first.node(pyg::HeteroNodeType::physical);
        const auto& relation = first.relation(pyg::HeteroRelationId::v_mapping_p);
        const auto bipartite_output = bipartite->forward_with_attention(
            x_source, x_target, relation.edge_index, relation.edge_attr);

        auto self_options = pyg::BipartiteGATConvOptions{};
        self_options.source_in_channels = 2;
        self_options.target_in_channels = 2;
        self_options.out_channels = 2;
        self_options.edge_dim = 1;
        auto self_loop = pyg::BipartiteGATConv(self_options);
        fill_linear(self_loop->lin_src->weight, -0.11F, 0.02F);
        fill_linear(self_loop->lin_dst->weight, -0.11F, 0.02F);
        fill_linear(self_loop->lin_edge->weight, 0.04F, 0.01F);
        fill_linear(self_loop->att_src, -0.02F, 0.005F);
        fill_linear(self_loop->att_dst, 0.01F, -0.003F);
        fill_linear(self_loop->att_edge, 0.005F, 0.002F);
        fill_linear(self_loop->bias, 0.02F, -0.004F);
        self_loop->eval();
        const auto& p_connect = first.relation(pyg::HeteroRelationId::p_connect_p);
        const auto self_output = self_loop->forward_with_attention(
            x_target, x_target, p_connect.edge_index, p_connect.edge_attr);

        auto hetero = pyg::HeteroConv(hetero_options());
        fill_relation_parameters(hetero);
        const auto hetero_output = hetero->forward(first);

        const auto benchmark = arguments.enabled
            ? std::optional<BenchmarkResult>(benchmark_hetero(*hetero, first, arguments))
            : std::nullopt;

        emit_records({
            {"batch_p_x", summarize(packed.node(pyg::HeteroNodeType::physical))},
            {"batch_v_x", summarize(packed.node(pyg::HeteroNodeType::virtual_node))},
            {"batch_p_batch", summarize(*packed.batch[pyg::hetero_node_index(pyg::HeteroNodeType::physical)])},
            {"batch_v_batch", summarize(*packed.batch[pyg::hetero_node_index(pyg::HeteroNodeType::virtual_node)])},
            {"batch_p_ptr", summarize(packed.node_ptr[pyg::hetero_node_index(pyg::HeteroNodeType::physical)])},
            {"batch_v_ptr", summarize(packed.node_ptr[pyg::hetero_node_index(pyg::HeteroNodeType::virtual_node)])},
            {"batch_v_mapping_p_edge_index", summarize(packed.relation(pyg::HeteroRelationId::v_mapping_p).edge_index)},
            {"bipartite_output", summarize(bipartite_output.output)},
            {"bipartite_attention", summarize(bipartite_output.attention)},
            {"self_loop_output", summarize(self_output.output)},
            {"hetero_p_output", summarize(hetero_output.node(pyg::HeteroNodeType::physical))},
            {"hetero_v_output", summarize(hetero_output.node(pyg::HeteroNodeType::virtual_node))}
        }, arguments, benchmark.has_value() ? &*benchmark : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hetero-pyg harness error: " << error.what() << '\n';
        return 1;
    }
}
