#include "../virne/solver/learning/neural_network/mlp.h"

#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef VIRNE_LIBTORCH_VERSION
#define VIRNE_LIBTORCH_VERSION "unknown"
#endif

namespace ml = virne::solver::learning::neural_network;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    int threads = 1;
    int warmups = 3;
    int repetitions = 5;
    int iterations = 20;
};

struct Digest {
    std::uint64_t value = fnv_offset;
    std::uint64_t bytes = 0;

    void append(const std::uint8_t byte) {
        value ^= byte;
        value *= fnv_prime;
        ++bytes;
    }

    void append(const void* data, const std::size_t size) {
        const auto* bytes_ptr = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            append(bytes_ptr[index]);
        }
    }
};

struct TensorSummary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::uint64_t bytes = 0;
    std::vector<std::string> bits;
};

struct StateEntry {
    std::string name;
    TensorSummary tensor;
};

struct CaseResult {
    std::string name;
    std::string mode;
    std::optional<TensorSummary> input;
    std::optional<TensorSummary> output;
    std::optional<TensorSummary> next_rng;
    std::vector<StateEntry> state;
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
        "\nusage: vne_mlp_net_harness [--threads N] [--warmups N] "
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

std::string json_escape(const std::string_view input) {
    std::string result;
    result.reserve(input.size());
    for (const char character : input) {
        if (character == '\\' || character == '"') {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    return result;
}

std::string dtype_name(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat32:
        return "float32";
    case torch::kFloat64:
        return "float64";
    case torch::kInt64:
        return "int64";
    default:
        throw std::runtime_error("unsupported tensor dtype in MLP harness");
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
    Digest digest;
    digest.append(tensor.const_data_ptr(), byte_count);
    result.checksum = digest.value;
    result.bytes = digest.bytes;

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
    } else if (tensor.scalar_type() == torch::kFloat64) {
        const auto* values = tensor.const_data_ptr<double>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            std::uint64_t bits = 0;
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

bool ends_with(const std::string_view value, const std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.substr(value.size() - suffix.size()) == suffix;
}

void fill_floating_tensor(
    torch::Tensor tensor,
    const std::int64_t tensor_id,
    const bool running_variance) {
    auto flat = tensor.view({-1});
    if (flat.scalar_type() == torch::kFloat32) {
        auto* values = flat.data_ptr<float>();
        for (std::int64_t index = 0; index < flat.numel(); ++index) {
            if (running_variance) {
                values[index] = static_cast<float>(
                    0.5 + static_cast<double>((tensor_id + index) % 16) / 16.0);
            } else {
                const auto numerator =
                    ((tensor_id * 37 + index * 17) % 61) - 30;
                values[index] =
                    static_cast<float>(static_cast<double>(numerator) / 32.0);
            }
        }
    } else if (flat.scalar_type() == torch::kFloat64) {
        auto* values = flat.data_ptr<double>();
        for (std::int64_t index = 0; index < flat.numel(); ++index) {
            if (running_variance) {
                values[index] =
                    0.5 + static_cast<double>((tensor_id + index) % 16) / 16.0;
            } else {
                const auto numerator =
                    ((tensor_id * 37 + index * 17) % 61) - 30;
                values[index] = static_cast<double>(numerator) / 32.0;
            }
        }
    } else {
        throw std::runtime_error("unexpected floating state dtype");
    }
}

void fill_state(ml::MLPNet& model) {
    torch::NoGradGuard no_grad;
    std::int64_t tensor_id = 0;
    for (auto& item : model->named_parameters(true)) {
        fill_floating_tensor(item.value(), tensor_id, false);
        ++tensor_id;
    }
    for (auto& item : model->named_buffers(true)) {
        auto tensor = item.value();
        if (tensor.is_floating_point()) {
            fill_floating_tensor(
                tensor,
                tensor_id,
                ends_with(item.key(), "running_var"));
        } else if (tensor.scalar_type() == torch::kInt64) {
            tensor.fill_(tensor_id + 3);
        } else {
            throw std::runtime_error("unexpected MLP state buffer dtype");
        }
        ++tensor_id;
    }
}

std::vector<StateEntry> summarize_state(ml::MLPNet& model) {
    std::vector<StateEntry> result;
    const auto parameters = model->named_parameters(true);
    const auto buffers = model->named_buffers(true);
    result.reserve(parameters.size() + buffers.size());
    for (const auto& item : parameters) {
        result.push_back(StateEntry{
            item.key(),
            summarize(item.value(), false),
        });
    }
    for (const auto& item : buffers) {
        result.push_back(StateEntry{
            item.key(),
            summarize(item.value(), false),
        });
    }
    return result;
}

torch::Tensor make_input(
    const std::vector<std::int64_t>& shape,
    const torch::ScalarType dtype) {
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
    auto flat = tensor.view({-1});
    if (dtype == torch::kFloat32) {
        auto* values = flat.data_ptr<float>();
        for (std::int64_t index = 0; index < flat.numel(); ++index) {
            const auto numerator = ((index * 13) % 47) - 23;
            values[index] =
                static_cast<float>(static_cast<double>(numerator) / 16.0);
        }
    } else if (dtype == torch::kFloat64) {
        auto* values = flat.data_ptr<double>();
        for (std::int64_t index = 0; index < flat.numel(); ++index) {
            const auto numerator = ((index * 13) % 47) - 23;
            values[index] = static_cast<double>(numerator) / 16.0;
        }
    } else {
        throw std::runtime_error("unsupported MLP input dtype");
    }
    return tensor;
}

ml::MlpOptions make_options(
    const std::int64_t input_dim,
    const std::int64_t output_dim,
    const std::int64_t num_layers,
    ml::MlpHiddenDims hidden,
    const bool batch_norm,
    const double dropout_prob) {
    ml::MlpOptions result{};
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.num_layers = num_layers;
    result.embedding_dims = std::move(hidden);
    result.batch_norm = batch_norm;
    result.dropout_prob = dropout_prob;
    return result;
}

CaseResult initialization_case() {
    torch::manual_seed(0);
    ml::MLPNet model(make_options(
        4, 3, 2, std::monostate{}, false, 1.0));
    CaseResult result{};
    result.name = "init_default";
    result.mode = "init";
    result.state = summarize_state(model);
    result.next_rng = summarize(torch::rand({8}, torch::kFloat32), true);
    return result;
}

CaseResult forward_case(
    std::string name,
    ml::MlpOptions options,
    const std::vector<std::int64_t>& input_shape,
    const torch::ScalarType dtype,
    const bool training) {
    torch::manual_seed(7);
    ml::MLPNet model(std::move(options));
    if (dtype == torch::kFloat64) {
        model->to(torch::kFloat64);
    }
    fill_state(model);
    auto input = make_input(input_shape, dtype);
    if (training) {
        model->train();
    } else {
        model->eval();
    }
    torch::NoGradGuard no_grad;
    auto output = model->forward(input).contiguous();

    CaseResult result{};
    result.name = std::move(name);
    result.mode = training ? "train" : "eval";
    result.input = summarize(input, true);
    result.output = summarize(output, true);
    result.state = summarize_state(model);
    return result;
}

std::vector<CaseResult> run_cases() {
    std::vector<CaseResult> result;
    result.reserve(8);
    result.push_back(initialization_case());
    result.push_back(forward_case(
        "forward_default",
        make_options(4, 3, 2, std::monostate{}, false, 1.0),
        {7, 4},
        torch::kFloat32,
        false));
    result.push_back(forward_case(
        "forward_scalar_rank3",
        make_options(5, 1, 3, std::int64_t{7}, false, 1.0),
        {2, 4, 5},
        torch::kFloat32,
        false));
    result.push_back(forward_case(
        "forward_explicit",
        make_options(
            3,
            2,
            4,
            std::vector<std::int64_t>{5, 4, 6},
            false,
            1.0),
        {8, 3},
        torch::kFloat32,
        false));
    result.push_back(forward_case(
        "batch_norm_eval",
        make_options(
            3,
            2,
            4,
            std::vector<std::int64_t>{5, 4, 6},
            true,
            1.0),
        {8, 3},
        torch::kFloat32,
        false));
    result.push_back(forward_case(
        "batch_norm_train",
        make_options(
            3,
            2,
            4,
            std::vector<std::int64_t>{5, 4, 6},
            true,
            1.0),
        {8, 3},
        torch::kFloat32,
        true));
    result.push_back(forward_case(
        "dropout_zero",
        make_options(4, 3, 2, std::monostate{}, false, 0.0),
        {7, 4},
        torch::kFloat32,
        false));
    result.push_back(forward_case(
        "dropout_one",
        make_options(4, 3, 2, std::monostate{}, false, 1.0),
        {7, 4},
        torch::kFloat32,
        false));
    result.push_back(forward_case(
        "forward_float64",
        make_options(4, 2, 3, std::int64_t{6}, false, 1.0),
        {5, 4},
        torch::kFloat64,
        false));
    return result;
}

BenchmarkResult run_benchmark(const Arguments& arguments) {
    torch::manual_seed(19);
    ml::MLPNet model(make_options(
        25, 1, 3, std::int64_t{128}, false, 0.0));
    fill_state(model);
    model->eval();
    const auto input = make_input({8, 500, 25}, torch::kFloat32);
    torch::Tensor output;
    torch::InferenceMode inference;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = model->forward(input);
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
            output = model->forward(input);
        }
        const auto stopped = std::chrono::steady_clock::now();
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - started).count());
    }
    result.output = summarize(output, false);
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
           << "\",\"bytes\":" << summary.bytes
           << ",\"bits\":[";
    for (std::size_t index = 0; index < summary.bits.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << '"' << summary.bits[index] << '"';
    }
    output << "]}";
}

void emit_state(std::ostream& output, const std::vector<StateEntry>& state) {
    output << '[';
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << "{\"name\":\"" << json_escape(state[index].name)
               << "\",\"tensor\":";
        emit_summary(output, state[index].tensor);
        output << '}';
    }
    output << ']';
}

void emit_case(std::ostream& output, const CaseResult& result) {
    output << "{\"name\":\"" << json_escape(result.name)
           << "\",\"mode\":\"" << result.mode << "\"";
    if (result.input.has_value()) {
        output << ",\"input\":";
        emit_summary(output, *result.input);
    }
    if (result.output.has_value()) {
        output << ",\"output\":";
        emit_summary(output, *result.output);
    }
    if (result.next_rng.has_value()) {
        output << ",\"next_rng\":";
        emit_summary(output, *result.next_rng);
    }
    output << ",\"state\":";
    emit_state(output, result.state);
    output << '}';
}

void emit_report(
    const Arguments& arguments,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult& benchmark) {
    std::cout << "{\"component\":\"mlp_net\",\"libtorch_version\":\""
              << json_escape(VIRNE_LIBTORCH_VERSION)
              << "\",\"threads\":" << arguments.threads
              << ",\"cases\":[";
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
        const Arguments arguments = parse_arguments(argc, argv);
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);
        const auto cases = run_cases();
        const auto benchmark = run_benchmark(arguments);
        emit_report(arguments, cases, benchmark);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_net_harness error: " << error.what() << '\n';
        return 2;
    }
}
