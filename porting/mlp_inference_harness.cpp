#include "../virne/solver/learning/rl_core/mlp_inference.h"

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

namespace core = virne::solver::learning::rl_core;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;
constexpr std::uint64_t mask64 = std::numeric_limits<std::uint64_t>::max();

struct Arguments {
    bool benchmark = false;
    int warmups = 1;
    int repetitions = 3;
};

struct TensorSummary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::vector<std::string> bits;
};

struct CaseReport {
    std::string name;
    TensorSummary logits;
    TensorSummary values;
    TensorSummary probabilities;
    TensorSummary actions;
    TensorSummary log_probabilities;
    TensorSummary entropy;
};

struct BenchmarkReport {
    std::vector<std::int64_t> samples_ns;
    std::uint64_t logits_checksum = 0U;
    std::uint64_t values_checksum = 0U;
    std::uint64_t log_prob_checksum = 0U;
    std::uint64_t entropy_checksum = 0U;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: vne_mlp_inference_harness [--benchmark]"
        " [--warmups N] [--repetitions N]");
}

Arguments parse_arguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto positive = [&](const char* option) {
            if (++index >= argc) usage_error(std::string(option) + " needs N");
            const int value = std::stoi(argv[index]);
            if (value <= 0) usage_error(std::string(option) + " must be positive");
            return value;
        };
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--warmups") {
            result.warmups = positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = positive("--repetitions");
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

void mix_bytes(std::uint64_t& checksum, const std::uint8_t* bytes,
               const std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        checksum ^= bytes[index];
        checksum = (checksum * fnv_prime) & mask64;
    }
}

TensorSummary summarize(const torch::Tensor& input, const bool include_bits = true) {
    const auto value = input.detach().to(torch::kCPU).contiguous();
    TensorSummary result;
    result.shape = value.sizes().vec();
    const auto type = value.scalar_type();
    const auto bytes = static_cast<std::size_t>(value.numel()) * value.element_size();
    mix_bytes(
        result.checksum,
        reinterpret_cast<const std::uint8_t*>(value.data_ptr()),
        bytes);
    if (type == torch::kFloat32) {
        result.dtype = "float32";
        if (include_bits) {
            const auto* bytes_value = reinterpret_cast<const std::uint8_t*>(
                value.data_ptr());
            result.bits.reserve(static_cast<std::size_t>(value.numel()));
            for (std::int64_t index = 0; index < value.numel(); ++index) {
                std::uint32_t bits = 0U;
                std::memcpy(&bits, bytes_value + index * sizeof(bits), sizeof(bits));
                std::ostringstream stream;
                stream << std::hex << std::setfill('0') << std::setw(8)
                       << bits;
                result.bits.push_back(stream.str());
            }
        }
    } else if (type == torch::kInt64) {
        result.dtype = "int64";
        if (include_bits) {
            const auto* values = value.data_ptr<std::int64_t>();
            result.bits.reserve(static_cast<std::size_t>(value.numel()));
            for (std::int64_t index = 0; index < value.numel(); ++index) {
                std::ostringstream stream;
                stream << std::hex << std::setfill('0') << std::setw(16)
                       << static_cast<std::uint64_t>(values[index]);
                result.bits.push_back(stream.str());
            }
        }
    } else {
        throw std::runtime_error("unexpected inference summary dtype");
    }
    return result;
}

core::PolicyConfiguration configuration(
    const std::int64_t physical_nodes,
    const std::int64_t physical_features,
    const std::int64_t virtual_features,
    const std::int64_t embedding = 8) {
    core::PolicyConfiguration result;
    result.feature_dimensions.p_net_x_dim = physical_features;
    result.feature_dimensions.v_node_x_dim = virtual_features;
    result.feature_dimensions.p_net_num_nodes = physical_nodes;
    result.neural_network.embedding_dim = embedding;
    result.neural_network.dropout_prob = 0.0;
    result.neural_network.batch_norm = false;
    return result;
}

core::DenseObservationBatch make_observation(
    const std::int64_t batch,
    const std::int64_t nodes,
    const std::int64_t physical_features,
    const std::int64_t virtual_features,
    const bool with_mask) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto p_count = batch * nodes * physical_features;
    const auto v_count = batch * virtual_features;
    core::DenseObservationBatch result;
    result.p_net_x = torch::arange(p_count, options)
        .reshape({batch, nodes, physical_features})
        .remainder(47.0)
        .sub(23.0)
        .div(16.0);
    result.v_node_x = torch::arange(v_count, options)
        .remainder(47.0)
        .sub(23.0)
        .div(16.0)
        .reshape({batch, virtual_features});
    if (with_mask) {
        result.action_mask = torch::remainder(
            torch::arange(batch * nodes, options), 5.0)
            .ne(0)
            .reshape({batch, nodes});
    }
    return result;
}

void fill_state(const core::MlpInferenceSession& session) {
    torch::NoGradGuard no_grad;
    std::int64_t tensor_id = 0;
    auto fill = [&tensor_id](const torch::Tensor& tensor) {
        if (tensor.scalar_type() == torch::kInt64) {
            tensor.fill_(tensor_id + 3);
            ++tensor_id;
            return;
        }
        if (tensor.scalar_type() != torch::kFloat32) {
            throw std::runtime_error("unexpected MLP state dtype");
        }
        const auto count = tensor.numel();
        auto values = torch::empty(
            {count}, tensor.options().requires_grad(false));
        auto accessor = values.accessor<float, 1>();
        for (std::int64_t index = 0; index < count; ++index) {
            accessor[index] = static_cast<float>(
                ((tensor_id * 37 + index * 17) % 61 - 30) / 32.0);
        }
        tensor.copy_(values.reshape(tensor.sizes()));
        ++tensor_id;
    };
    for (const auto& item : session.policy()->named_parameters(true)) {
        fill(item.value());
    }
    for (const auto& item : session.policy()->named_buffers(true)) {
        fill(item.value());
    }
}

CaseReport run_case(
    const std::string& name,
    const std::int64_t batch,
    const std::int64_t nodes,
    const std::int64_t physical_features,
    const std::int64_t virtual_features,
    const bool with_mask) {
    torch::manual_seed(7);
    core::MlpInferenceSession session(
        configuration(nodes, physical_features, virtual_features),
        core::PolicyActionOptions{0.75, true, true},
        torch::Device(torch::kCPU));
    fill_state(session);
    session.policy()->eval();
    const auto observation = make_observation(
        batch, nodes, physical_features, virtual_features, with_mask);
    const auto forward = session.forward(observation);
    const auto distribution = session.get_action_prob_dist(observation);
    const auto selection = session.select_action(
        observation, core::PolicyActionMode::greedy);
    const auto evaluation = session.evaluate_actions(
        observation, selection.actions, true);
    return CaseReport{
        name,
        summarize(forward.action_logits),
        summarize(forward.values),
        summarize(distribution.probabilities),
        summarize(selection.actions),
        summarize(evaluation.log_probabilities),
        summarize(evaluation.entropy),
    };
}

BenchmarkReport benchmark(const Arguments& arguments) {
    torch::manual_seed(19);
    core::MlpInferenceSession session(
        configuration(500, 14, 11, 128),
        core::PolicyActionOptions{1.0, true, true},
        torch::Device(torch::kCPU));
    fill_state(session);
    session.policy()->eval();
    const auto observation = make_observation(32, 500, 14, 11, true);
    const auto actions = torch::zeros({32}, torch::kInt64);
    BenchmarkReport result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        const auto output = session.evaluate_actions(observation, actions, true);
        result.logits_checksum = summarize(
            session.action_logits(observation), false).checksum;
        result.values_checksum = summarize(*output.values, false).checksum;
        result.log_prob_checksum = summarize(
            output.log_probabilities, false).checksum;
        result.entropy_checksum = summarize(output.entropy, false).checksum;
    }
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto output = session.evaluate_actions(observation, actions, true);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (!output.values.has_value()) {
            throw std::runtime_error("inference benchmark lost values");
        }
        result.logits_checksum = summarize(
            session.action_logits(observation), false).checksum;
        result.values_checksum = summarize(*output.values, false).checksum;
        result.log_prob_checksum = summarize(
            output.log_probabilities, false).checksum;
        result.entropy_checksum = summarize(output.entropy, false).checksum;
    }
    return result;
}

void emit_summary(const TensorSummary& value) {
    std::cout << "{\"shape\":[";
    for (std::size_t index = 0; index < value.shape.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << value.shape[index];
    }
    std::cout << "],\"dtype\":\"" << value.dtype
              << "\",\"checksum\":\"" << value.checksum << "\""
              << ",\"bits\":[";
    for (std::size_t index = 0; index < value.bits.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << value.bits[index] << '"';
    }
    std::cout << "]}";
}

void emit_case(const CaseReport& value) {
    std::cout << "{\"name\":\"" << value.name << "\",\"logits\":";
    emit_summary(value.logits);
    std::cout << ",\"values\":";
    emit_summary(value.values);
    std::cout << ",\"probabilities\":";
    emit_summary(value.probabilities);
    std::cout << ",\"actions\":";
    emit_summary(value.actions);
    std::cout << ",\"log_probabilities\":";
    emit_summary(value.log_probabilities);
    std::cout << ",\"entropy\":";
    emit_summary(value.entropy);
    std::cout << '}';
}

void emit_report(
    const Arguments& arguments,
    const std::vector<CaseReport>& cases,
    const BenchmarkReport* benchmark_report) {
    std::cout << "{\"component\":\"mlp_inference\",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) std::cout << ',';
        emit_case(cases[index]);
    }
    std::cout << ']';
    if (benchmark_report != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"batch32_nodes500\""
                  << ",\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0;
             index < benchmark_report->samples_ns.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << benchmark_report->samples_ns[index];
        }
        std::cout << "]"
                  << ",\"logits_checksum\":\""
                  << benchmark_report->logits_checksum
                  << "\",\"values_checksum\":\""
                  << benchmark_report->values_checksum
                  << "\",\"log_prob_checksum\":\""
                  << benchmark_report->log_prob_checksum
                  << "\",\"entropy_checksum\":\""
                  << benchmark_report->entropy_checksum << "\"}";
    }
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        const std::vector<CaseReport> cases{
            run_case("single_masked_greedy", 1, 7, 4, 3, true),
            run_case("batch_masked_greedy", 4, 17, 4, 3, true),
            run_case("batch_unmasked_greedy", 3, 9, 5, 2, false),
        };
        BenchmarkReport benchmark_report;
        if (arguments.benchmark) benchmark_report = benchmark(arguments);
        emit_report(
            arguments, cases, arguments.benchmark ? &benchmark_report : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_inference_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
