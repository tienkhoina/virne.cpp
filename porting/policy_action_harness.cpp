#include "../virne/solver/learning/rl_core/policy_action.h"

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
#include <vector>

namespace rl = virne::solver::learning::rl_core;

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
        + "\nusage: vne_policy_action_harness [--benchmark]"
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
    default: throw std::runtime_error("unsupported policy-action summary dtype");
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
    } else {
        append_bits<std::int64_t, std::uint64_t>(tensor, result.bits);
    }
    return result;
}

void append(
    std::vector<CaseResult>& output,
    std::string name,
    const torch::Tensor& value,
    const bool include_bits = true) {
    output.push_back({std::move(name), summarize(value, include_bits)});
}

torch::Tensor logits_fixture(const torch::Device& device) {
    return torch::tensor(
        {{1.0F, -0.5F, 0.25F, 2.0F},
         {-1.0F, 0.5F, 1.5F, 0.0F},
         {0.1F, 0.2F, 0.3F, 0.4F}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor mask_fixture(const torch::Device& device) {
    return torch::tensor(
        {{1, 0, 1, 1}, {0, 1, 1, 0}, {0, 0, 0, 0}},
        torch::TensorOptions().dtype(torch::kBool).device(device));
}

std::vector<CaseResult> run_cases(const torch::Device& device) {
    std::vector<CaseResult> output;
    const auto logits = logits_fixture(device);
    const auto mask = mask_fixture(device);
    rl::PolicyActionOptions options;
    options.softmax_temperature = 0.75;

    const auto distribution = rl::policy_action_distribution(logits, mask, options);
    append(output, "distribution.probabilities", distribution.probabilities);
    append(output, "distribution.candidate_logits", distribution.candidate_logits);

    const auto greedy = rl::select_policy_action(
        logits, mask, rl::PolicyActionMode::greedy, options);
    append(output, "greedy.actions", greedy.actions);
    append(output, "greedy.log_probabilities", greedy.log_probabilities);

    options.maskable_policy = false;
    const auto raw_greedy = rl::select_policy_action(
        logits, mask, rl::PolicyActionMode::greedy, options);
    append(output, "greedy_raw.actions", raw_greedy.actions);
    append(output, "greedy_raw.log_probabilities", raw_greedy.log_probabilities);

    options.maskable_policy = true;
    torch::manual_seed(90210);
    const auto sampled = rl::select_policy_action(
        logits, mask, rl::PolicyActionMode::sample, options);
    append(output, "sample.actions", sampled.actions);
    append(output, "sample.log_probabilities", sampled.log_probabilities);
    append(output, "sample.rng_continuation",
           torch::rand({4}, torch::TensorOptions().device(device)));

    auto grad_logits = logits.clone().set_requires_grad(true);
    auto values = torch::tensor(
        {{0.5F}, {1.5F}, {-0.25F}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device))
                      .set_requires_grad(true);
    auto predicted = torch::tensor(
        {{0.8F, 0.2F, 0.7F, 0.9F},
         {0.1F, 0.6F, 0.8F, 0.2F},
         {0.4F, 0.3F, 0.2F, 0.1F}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device))
                         .set_requires_grad(true);
    const auto actions = torch::tensor(
        {3, 2, 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    options.softmax_temperature = 0.5;
    const auto evaluation = rl::evaluate_policy_actions(
        grad_logits, actions, values, mask, predicted, true, options);
    append(output, "evaluation.values", *evaluation.values);
    append(output, "evaluation.log_probabilities", evaluation.log_probabilities);
    append(output, "evaluation.entropy", evaluation.entropy);
    append(output, "evaluation.masked_action_probability",
           *evaluation.masked_action_probability);
    append(output, "evaluation.prediction_loss", *evaluation.prediction_loss);
    const auto loss = evaluation.log_probabilities.mean()
        + evaluation.entropy.mean()
        + evaluation.values->mean()
        + *evaluation.masked_action_probability
        + *evaluation.prediction_loss;
    loss.backward();
    append(output, "evaluation.logits_grad", grad_logits.grad());
    append(output, "evaluation.values_grad", values.grad());
    append(output, "evaluation.predicted_mask_grad", predicted.grad());

    options.mask_actions = false;
    const auto raw_evaluation = rl::evaluate_policy_actions(
        logits, actions, std::nullopt, mask, std::nullopt, true, options);
    append(output, "evaluation_raw.log_probabilities",
           raw_evaluation.log_probabilities);
    append(output, "evaluation_raw.entropy", raw_evaluation.entropy);
    append(output, "evaluation_raw.masked_action_probability",
           *raw_evaluation.masked_action_probability);

    const auto scalar_logits = torch::tensor(
        {0.2, -0.3, 1.1},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    const auto scalar_selection = rl::select_policy_action(
        scalar_logits, std::nullopt, rl::PolicyActionMode::greedy);
    append(output, "scalar.actions", scalar_selection.actions);
    append(output, "scalar.log_probabilities",
           scalar_selection.log_probabilities);
    const auto scalar_evaluation = rl::evaluate_policy_actions(
        scalar_logits,
        torch::tensor(2, torch::TensorOptions().dtype(torch::kInt64).device(device)));
    append(output, "scalar.entropy", scalar_evaluation.entropy);
    return output;
}

struct BenchmarkFixture {
    torch::Tensor logits;
    torch::Tensor actions;
    torch::Tensor mask;
};

BenchmarkFixture benchmark_fixture(const torch::Device& device) {
    constexpr std::int64_t rows = 256;
    constexpr std::int64_t actions = 256;
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto logits = torch::arange(rows * actions, f32)
                      .reshape({rows, actions})
                      .remainder(97)
                      .mul_(0.03125F)
                      .sub_(1.5F);
    auto mask = torch::arange(rows * actions, i64)
                    .reshape({rows, actions})
                    .remainder(5)
                    .ne(0);
    auto selected = torch::arange(rows, i64).mul(37).remainder(actions);
    return {logits, selected, mask};
}

BenchmarkResult run_benchmark(
    const Arguments& arguments,
    const torch::Device& device) {
    const auto fixture = benchmark_fixture(device);
    rl::PolicyActionOptions options;
    options.softmax_temperature = 0.75;
    rl::PolicyActionEvaluation sampled;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            sampled = rl::evaluate_policy_actions(
                fixture.logits, fixture.actions, std::nullopt, fixture.mask,
                std::nullopt, true, options);
        }
    }
    if (device.is_cuda()) torch::cuda::synchronize(device.index());
    BenchmarkResult result;
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        if (device.is_cuda()) torch::cuda::synchronize(device.index());
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            sampled = rl::evaluate_policy_actions(
                fixture.logits, fixture.actions, std::nullopt, fixture.mask,
                std::nullopt, true, options);
        }
        if (device.is_cuda()) torch::cuda::synchronize(device.index());
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
    }
    append(result.output, "benchmark.log_probabilities",
           sampled.log_probabilities, false);
    append(result.output, "benchmark.entropy", sampled.entropy, false);
    append(result.output, "benchmark.masked_action_probability",
           *sampled.masked_action_probability, false);
    return result;
}

void emit_shape(std::ostream& out, const std::vector<std::int64_t>& shape) {
    out << '[';
    for (std::size_t index = 0U; index < shape.size(); ++index) {
        if (index != 0U) out << ',';
        out << shape[index];
    }
    out << ']';
}

void emit_summary(std::ostream& out, const TensorSummary& summary) {
    out << "{\"shape\":";
    emit_shape(out, summary.shape);
    out << ",\"dtype\":\"" << summary.dtype
        << "\",\"device\":\"" << summary.device
        << "\",\"checksum\":\"" << summary.checksum
        << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0U; index < summary.bits.size(); ++index) {
        if (index != 0U) out << ',';
        out << '"' << summary.bits[index] << '"';
    }
    out << "]}";
}

void emit_cases(std::ostream& out, const std::vector<CaseResult>& cases) {
    out << '[';
    for (std::size_t index = 0U; index < cases.size(); ++index) {
        if (index != 0U) out << ',';
        out << "{\"name\":\"" << cases[index].name << "\",\"output\":";
        emit_summary(out, cases[index].output);
        out << '}';
    }
    out << ']';
}

void emit_report(
    const Arguments& arguments,
    const torch::Device& device,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult* benchmark) {
    std::cout << "{\"component\":\"policy_action\",\"device\":\""
              << (device.is_cuda() ? "cuda" : "cpu")
              << "\",\"threads\":" << arguments.threads << ",\"cases\":";
    emit_cases(std::cout, cases);
    if (benchmark != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"evaluate_256x256\""
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
        std::cerr << "policy-action harness error: " << error.what() << '\n';
        return 2;
    }
}
