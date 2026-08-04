#include "../virne/solver/learning/rl_policy/mlp_policy.h"

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

namespace policy = virne::solver::learning::rl_policy;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

enum class CaseKind : std::uint8_t {
    composite,
    actor,
    critic,
};

enum class CaseMode : std::uint8_t {
    init,
    eval,
};

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
    CaseKind kind = CaseKind::composite;
    CaseMode mode = CaseMode::eval;
    std::optional<TensorSummary> input;
    std::optional<TensorSummary> actor_output;
    std::optional<TensorSummary> critic_output;
    std::optional<TensorSummary> next_rng;
    std::vector<StateEntry> state;
};

struct BenchmarkResult {
    int warmups = 0;
    int repetitions = 0;
    int iterations = 0;
    std::vector<std::int64_t> samples_ns;
    TensorSummary actor_output;
    TensorSummary critic_output;
};

[[noreturn]] void usage_error(const std::string_view message) {
    throw std::invalid_argument(
        std::string(message) +
        "\nusage: vne_mlp_policy_harness [--threads N] [--warmups N] "
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

std::string_view case_kind_name(const CaseKind kind) {
    switch (kind) {
    case CaseKind::composite:
        return "composite";
    case CaseKind::actor:
        return "actor";
    case CaseKind::critic:
        return "critic";
    }
    throw std::logic_error("invalid MLP policy case kind");
}

std::string_view case_mode_name(const CaseMode mode) {
    switch (mode) {
    case CaseMode::init:
        return "init";
    case CaseMode::eval:
        return "eval";
    }
    throw std::logic_error("invalid MLP policy case mode");
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
        throw std::runtime_error(
            "unsupported tensor dtype in MLP policy harness");
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
                    0.5 +
                    static_cast<double>((tensor_id + index) % 16) / 16.0);
            } else {
                const auto numerator =
                    ((tensor_id * 37 + index * 17) % 61) - 30;
                values[index] = static_cast<float>(
                    static_cast<double>(numerator) / 32.0);
            }
        }
    } else if (flat.scalar_type() == torch::kFloat64) {
        auto* values = flat.data_ptr<double>();
        for (std::int64_t index = 0; index < flat.numel(); ++index) {
            if (running_variance) {
                values[index] =
                    0.5 +
                    static_cast<double>((tensor_id + index) % 16) / 16.0;
            } else {
                const auto numerator =
                    ((tensor_id * 37 + index * 17) % 61) - 30;
                values[index] = static_cast<double>(numerator) / 32.0;
            }
        }
    } else {
        throw std::runtime_error("unexpected floating policy state dtype");
    }
}

template <typename ModuleHolder>
void fill_state(ModuleHolder& model) {
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
            throw std::runtime_error("unexpected MLP policy state buffer dtype");
        }
        ++tensor_id;
    }
}

template <typename ModuleHolder>
std::vector<StateEntry> summarize_state(ModuleHolder& model) {
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

torch::Tensor make_input(const std::vector<std::int64_t>& shape) {
    auto tensor = torch::empty(
        shape,
        torch::TensorOptions().dtype(torch::kFloat32));
    auto flat = tensor.view({-1});
    auto* values = flat.data_ptr<float>();
    for (std::int64_t index = 0; index < flat.numel(); ++index) {
        const auto numerator = ((index * 13) % 47) - 23;
        values[index] = static_cast<float>(
            static_cast<double>(numerator) / 16.0);
    }
    return tensor;
}

policy::MlpActorCriticOptions make_options(
    const std::int64_t feature_dim,
    const std::int64_t action_dim,
    const std::int64_t num_layers,
    const std::int64_t embedding_dim,
    const double dropout_prob,
    const bool batch_norm) {
    policy::MlpActorCriticOptions result{};
    result.feature_dim = feature_dim;
    result.action_dim = action_dim;
    result.num_layers = num_layers;
    result.embedding_dim = embedding_dim;
    result.dropout_prob = dropout_prob;
    result.batch_norm = batch_norm;
    return result;
}

CaseResult initialization_case() {
    torch::manual_seed(0);
    policy::MlpActorCritic model(make_options(
        4, 9, 3, 6, 0.25, false));

    CaseResult result{};
    result.name = "init_composite";
    result.kind = CaseKind::composite;
    result.mode = CaseMode::init;
    result.state = summarize_state(model);
    result.next_rng = summarize(torch::rand({8}, torch::kFloat32), true);
    return result;
}

CaseResult composite_forward_case(
    std::string name,
    const std::int64_t feature_dim,
    const std::int64_t action_dim,
    const std::int64_t num_layers,
    const std::int64_t embedding_dim,
    const double dropout_prob,
    std::vector<std::int64_t> input_shape,
    const std::uint64_t seed) {
    torch::manual_seed(seed);
    policy::MlpActorCritic model(make_options(
        feature_dim,
        action_dim,
        num_layers,
        embedding_dim,
        dropout_prob,
        false));
    fill_state(model);
    model->eval();
    const auto input = make_input(input_shape);
    torch::NoGradGuard no_grad;
    const auto actor_output = model->act(input).contiguous();
    const auto critic_output = model->evaluate(input).contiguous();

    CaseResult result{};
    result.name = std::move(name);
    result.kind = CaseKind::composite;
    result.mode = CaseMode::eval;
    result.input = summarize(input, true);
    result.actor_output = summarize(actor_output, true);
    result.critic_output = summarize(critic_output, true);
    result.state = summarize_state(model);
    return result;
}

CaseResult actor_batch_norm_case() {
    torch::manual_seed(13);
    policy::Actor model(make_options(3, 17, 4, 5, 0.4, true));
    fill_state(model);
    model->eval();
    const auto input = make_input({8, 3});
    torch::NoGradGuard no_grad;
    const auto output = model->forward(input).contiguous();

    CaseResult result{};
    result.name = "actor_batch_norm_eval";
    result.kind = CaseKind::actor;
    result.mode = CaseMode::eval;
    result.input = summarize(input, true);
    result.actor_output = summarize(output, true);
    result.state = summarize_state(model);
    return result;
}

CaseResult critic_batch_norm_case() {
    torch::manual_seed(13);
    policy::Critic model(make_options(3, 17, 4, 5, 0.4, true));
    fill_state(model);
    model->eval();
    const auto input = make_input({8, 3});
    torch::NoGradGuard no_grad;
    const auto output = model->forward(input).contiguous();

    CaseResult result{};
    result.name = "critic_batch_norm_eval";
    result.kind = CaseKind::critic;
    result.mode = CaseMode::eval;
    result.input = summarize(input, true);
    result.critic_output = summarize(output, true);
    result.state = summarize_state(model);
    return result;
}

std::vector<CaseResult> run_cases() {
    std::vector<CaseResult> result;
    result.reserve(7);
    result.push_back(initialization_case());
    result.push_back(composite_forward_case(
        "forward_rank3", 5, 7, 3, 7, 0.0, {2, 4, 5}, 7));
    result.push_back(composite_forward_case(
        "production_multi_resource",
        25,
        500,
        3,
        128,
        0.0,
        {8, 500, 25},
        7));
    result.push_back(composite_forward_case(
        "action_dim_one", 4, 1, 3, 6, 0.75, {7, 4}, 11));
    result.push_back(composite_forward_case(
        "action_dim_997", 4, 997, 3, 6, 0.75, {7, 4}, 11));
    result.push_back(actor_batch_norm_case());
    result.push_back(critic_batch_norm_case());
    return result;
}

BenchmarkResult run_benchmark(const Arguments& arguments) {
    torch::manual_seed(19);
    policy::MlpActorCritic model(make_options(
        25, 500, 3, 128, 0.0, false));
    fill_state(model);
    model->eval();
    const auto input = make_input({8, 500, 25});
    torch::Tensor actor_output;
    torch::Tensor critic_output;

    torch::InferenceMode inference;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            actor_output = model->act(input);
            critic_output = model->evaluate(input);
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
            actor_output = model->act(input);
            critic_output = model->evaluate(input);
        }
        const auto stopped = std::chrono::steady_clock::now();
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - started).count());
    }
    result.actor_output = summarize(actor_output, false);
    result.critic_output = summarize(critic_output, false);
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
           << "\",\"kind\":\"" << case_kind_name(result.kind)
           << "\",\"mode\":\"" << case_mode_name(result.mode) << '"';
    if (result.input.has_value()) {
        output << ",\"input\":";
        emit_summary(output, *result.input);
    }
    if (result.actor_output.has_value()) {
        output << ",\"actor_output\":";
        emit_summary(output, *result.actor_output);
    }
    if (result.critic_output.has_value()) {
        output << ",\"critic_output\":";
        emit_summary(output, *result.critic_output);
    }
    output << ",\"state\":";
    emit_state(output, result.state);
    if (result.next_rng.has_value()) {
        output << ",\"next_rng\":";
        emit_summary(output, *result.next_rng);
    }
    output << '}';
}

void emit_report(
    const Arguments& arguments,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult& benchmark) {
    std::cout << "{\"component\":\"mlp_policy\",\"libtorch_version\":\""
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
    std::cout << "],\"actor_output\":";
    emit_summary(std::cout, benchmark.actor_output);
    std::cout << ",\"critic_output\":";
    emit_summary(std::cout, benchmark.critic_output);
    std::cout << "}}\n";
}

} // namespace

int main(const int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        // Both pools are process-global and may only be configured before the
        // first operation. The harness fixes them once at startup.
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);

        const auto cases = run_cases();
        const auto benchmark = run_benchmark(arguments);
        emit_report(arguments, cases, benchmark);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_policy_harness error: " << error.what() << '\n';
        return 2;
    }
}
