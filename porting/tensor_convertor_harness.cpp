#include "../virne/solver/learning/rl_core/tensor_convertor.h"

#include <torch/torch.h>

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
#include <utility>
#include <vector>

#ifndef VIRNE_LIBTORCH_VERSION
#define VIRNE_LIBTORCH_VERSION "unknown"
#endif

namespace convertor = virne::solver::learning::rl_core;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    int threads = 1;
    int warmups = 3;
    int repetitions = 5;
    int iterations = 20;
};

struct TensorSummary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::uint64_t bytes = 0;
    std::vector<std::string> bits;
};

struct FieldResult {
    std::string name;
    TensorSummary tensor;
};

struct CaseResult {
    std::string name;
    std::vector<FieldResult> fields;
};

struct BenchmarkResult {
    int warmups = 0;
    int repetitions = 0;
    int iterations = 0;
    std::vector<std::int64_t> samples_ns;
    TensorSummary output;
};

[[noreturn]] void usage_error(const std::string_view message) {
    throw std::invalid_argument(
        std::string(message) +
        "\nusage: vne_tensor_convertor_harness [--threads N] "
        "[--warmups N] [--repetitions N] [--iterations N]");
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
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto read_value = [&](const char* option) {
            if (++index >= argc) {
                usage_error(std::string(option) + " requires a value");
            }
            return parse_positive(argv[index], option);
        };
        if (argument == "--threads") {
            result.threads = read_value("--threads");
        } else if (argument == "--warmups") {
            result.warmups = read_value("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = read_value("--repetitions");
        } else if (argument == "--iterations") {
            result.iterations = read_value("--iterations");
        } else {
            usage_error("unknown argument");
        }
    }
    return result;
}

std::string dtype_name(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat32:
        return "float32";
    case torch::kInt64:
        return "int64";
    default:
        throw std::runtime_error(
            "unsupported tensor dtype in TensorConvertor harness");
    }
}

template <typename UInt>
std::string bits_hex(const UInt bits) {
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(static_cast<int>(sizeof(UInt) * 2U)) << bits;
    return output.str();
}

TensorSummary summarize(const torch::Tensor& source, const bool include_bits) {
    const auto tensor = source.detach().to(torch::kCPU).contiguous();
    TensorSummary result{};
    result.shape = tensor.sizes().vec();
    result.dtype = dtype_name(tensor.scalar_type());
    const auto byte_count = static_cast<std::size_t>(tensor.nbytes());
    const auto* bytes = static_cast<const std::uint8_t*>(
        tensor.const_data_ptr());
    std::uint64_t checksum = fnv_offset;
    for (std::size_t index = 0; index < byte_count; ++index) {
        checksum ^= bytes[index];
        checksum *= fnv_prime;
    }
    result.checksum = checksum;
    result.bytes = byte_count;

    if (!include_bits) {
        return result;
    }
    result.bits.reserve(static_cast<std::size_t>(tensor.numel()));
    if (tensor.scalar_type() == torch::kFloat32) {
        const auto* values = tensor.const_data_ptr<float>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, values + index, sizeof(bits));
            result.bits.push_back(bits_hex(bits));
        }
    } else {
        const auto* values = tensor.const_data_ptr<std::int64_t>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            result.bits.push_back(bits_hex(
                static_cast<std::uint64_t>(values[index])));
        }
    }
    return result;
}

torch::Tensor patterned_double(
    const std::vector<std::int64_t>& shape,
    const std::int64_t multiplier,
    const std::int64_t modulus,
    const std::int64_t offset,
    const double denominator) {
    auto result = torch::empty(shape, torch::kFloat64);
    auto flat = result.view({-1});
    auto* values = flat.data_ptr<double>();
    for (std::int64_t index = 0; index < flat.numel(); ++index) {
        values[index] = static_cast<double>(
            ((index * multiplier) % modulus) - offset) / denominator;
    }
    return result;
}

convertor::DenseObservation make_single() {
    convertor::DenseObservation result{};
    result.p_net_x = patterned_double({3, 2}, 7, 29, 14, 8.0);
    result.v_node_x = patterned_double({2}, 5, 13, 6, 4.0);
    result.curr_v_node_id = 2;
    result.action_mask = torch::tensor({1, 0, 1}, torch::kInt64);
    result.v_net_size = 3.0;
    return result;
}

convertor::DenseObservationBatch make_batch(const bool include_general) {
    convertor::DenseObservationBatch result{};
    result.p_net_x = patterned_double({2, 3, 2}, 7, 29, 14, 8.0);
    result.v_node_x = patterned_double({2, 2}, 5, 13, 6, 4.0);
    if (include_general) {
        result.curr_v_node_id = torch::tensor({1, 2}, torch::kInt64);
        result.action_mask = torch::tensor(
            {{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}, torch::kFloat64);
        result.v_net_size = torch::tensor({3, 4}, torch::kInt64);
    }
    return result;
}

convertor::DenseObservationBatch make_production_batch() {
    convertor::DenseObservationBatch result{};
    result.p_net_x = patterned_double({8, 500, 14}, 13, 47, 23, 16.0);
    result.v_node_x = patterned_double({8, 11}, 17, 31, 15, 8.0);
    result.curr_v_node_id = torch::arange(0, 8, torch::kInt64);
    result.action_mask = torch::ones({8, 500}, torch::kFloat64);
    result.v_net_size = torch::arange(2, 10, torch::kInt64);
    return result;
}

void append_general(
    std::vector<FieldResult>& fields,
    const convertor::GeneralTensorObservation& general,
    const bool include_bits) {
    if (general.curr_v_node_id.has_value()) {
        fields.push_back({
            "curr_v_node_id",
            summarize(*general.curr_v_node_id, include_bits),
        });
    }
    if (general.action_mask.has_value()) {
        fields.push_back({
            "action_mask",
            summarize(*general.action_mask, include_bits),
        });
    }
    if (general.v_net_size.has_value()) {
        fields.push_back({
            "v_net_size",
            summarize(*general.v_net_size, include_bits),
        });
    }
}

CaseResult mlp_case(
    std::string name,
    const convertor::DenseObservation& input) {
    const auto output = convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"p_net_x", summarize(output.p_net_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult mlp_case(
    std::string name,
    const convertor::DenseObservationBatch& input) {
    const auto output = convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"p_net_x", summarize(output.p_net_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult cnn_case(
    std::string name,
    const convertor::DenseObservation& input) {
    const auto output = convertor::TensorConvertor::obs_as_tensor_for_cnn(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"p_net_x", summarize(output.p_net_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult cnn_case(
    std::string name,
    const convertor::DenseObservationBatch& input) {
    const auto output = convertor::TensorConvertor::obs_as_tensor_for_cnn(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"p_net_x", summarize(output.p_net_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult mlp_fc_case(
    std::string name,
    const convertor::DenseObservation& input) {
    const auto output =
        convertor::TensorConvertor::obs_as_tensor_for_mlp_fc(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"pv_nodes_x", summarize(output.pv_nodes_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult mlp_fc_case(
    std::string name,
    const convertor::DenseObservationBatch& input) {
    const auto output =
        convertor::TensorConvertor::obs_as_tensor_for_mlp_fc(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"pv_nodes_x", summarize(output.pv_nodes_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult attention_case(
    std::string name,
    const convertor::DenseObservation& input) {
    const auto output = convertor::TensorConvertor::obs_as_tensor_for_att(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"p_net_x", summarize(output.p_net_x, true)});
    result.fields.push_back({"v_node_x", summarize(output.v_node_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

CaseResult attention_case(
    std::string name,
    const convertor::DenseObservationBatch& input) {
    const auto output = convertor::TensorConvertor::obs_as_tensor_for_att(input);
    CaseResult result{std::move(name), {}};
    result.fields.push_back({"p_net_x", summarize(output.p_net_x, true)});
    result.fields.push_back({"v_node_x", summarize(output.v_node_x, true)});
    append_general(result.fields, output.general, true);
    return result;
}

std::vector<CaseResult> run_cases() {
    const auto single = make_single();
    const auto batch = make_batch(true);
    const auto batch_without_general = make_batch(false);
    std::vector<CaseResult> result;
    result.reserve(9);
    result.push_back(mlp_case("single_mlp", single));
    result.push_back(mlp_case("batch_mlp", batch));
    result.push_back(mlp_case("batch_mlp_no_general", batch_without_general));
    result.push_back(cnn_case("single_cnn", single));
    result.push_back(cnn_case("batch_cnn", batch));
    result.push_back(mlp_fc_case("single_mlp_fc", single));
    result.push_back(mlp_fc_case("batch_mlp_fc", batch));
    result.push_back(attention_case("single_att", single));
    result.push_back(attention_case("batch_att", batch));
    return result;
}

BenchmarkResult run_benchmark(const Arguments& arguments) {
    const auto input = make_production_batch();
    convertor::MlpTensorObservation output;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
        }
    }

    BenchmarkResult result{};
    result.warmups = arguments.warmups;
    result.repetitions = arguments.repetitions;
    result.iterations = arguments.iterations;
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0;
         repetition < arguments.repetitions;
         ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
        }
        const auto stopped = std::chrono::steady_clock::now();
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - started).count());
    }
    result.output = summarize(output.p_net_x, false);
    return result;
}

void emit_shape(std::ostream& output, const std::vector<std::int64_t>& shape) {
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << shape[index];
    }
    output << ']';
}

void emit_summary(std::ostream& output, const TensorSummary& summary) {
    output << "{\"shape\":";
    emit_shape(output, summary.shape);
    output << ",\"dtype\":\"" << summary.dtype
           << "\",\"checksum\":\"" << summary.checksum
           << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0; index < summary.bits.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << '"' << summary.bits[index] << '"';
    }
    output << "]}";
}

void emit_case(std::ostream& output, const CaseResult& result) {
    output << "{\"name\":\"" << result.name << "\",\"fields\":[";
    for (std::size_t index = 0; index < result.fields.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << "{\"name\":\"" << result.fields[index].name
               << "\",\"tensor\":";
        emit_summary(output, result.fields[index].tensor);
        output << '}';
    }
    output << "]}";
}

void emit_report(
    const Arguments& arguments,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult& benchmark) {
    std::cout << "{\"component\":\"tensor_convertor_dense\","
              << "\"libtorch_version\":\"" << VIRNE_LIBTORCH_VERSION
              << "\",\"threads\":" << arguments.threads << ",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        emit_case(std::cout, cases[index]);
    }
    std::cout << "],\"benchmark\":{\"warmups\":" << benchmark.warmups
              << ",\"repetitions\":" << benchmark.repetitions
              << ",\"iterations\":" << benchmark.iterations
              << ",\"samples_ns\":[";
    for (std::size_t index = 0; index < benchmark.samples_ns.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        std::cout << benchmark.samples_ns[index];
    }
    std::cout << "],\"output\":";
    emit_summary(std::cout, benchmark.output);
    std::cout << "}}\n";
}

} // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);
        const auto cases = run_cases();
        const auto benchmark = run_benchmark(arguments);
        emit_report(arguments, cases, benchmark);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor_convertor_harness error: " << error.what() << '\n';
        return 2;
    }
}
