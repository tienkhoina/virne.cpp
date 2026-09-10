#include "tensor_bytes.h"
#include "../virne/solver/learning/rl_policy/cnn_policy.h"
#include "../virne/solver/learning/rl_policy/gnn_mlp_policy.h"

#include <torch/torch.h>

#include <chrono>
#include <stdexcept>
#include <string_view>

namespace policy = virne::solver::learning::rl_policy;
namespace nn = virne::solver::learning::neural_network;
namespace pyg = virne::solver::learning::pyg_lite;
using virne::porting::emit_module_bytes;
using virne::porting::emit_tensor_bytes;
using virne::porting::fill_fixture_parameters;

namespace {

torch::Tensor values(const std::vector<std::int64_t>& shape) {
    std::int64_t count = 1;
    for (auto dim : shape) count *= dim;
    return torch::arange(count, torch::kFloat32).reshape(shape).remainder(17).sub(8).div(8);
}

policy::GnnMlpObservation observation() {
    pyg::Data a;
    a.x = values({3, 3});
    a.edge_index = torch::tensor({{0, 1, 1, 2}, {1, 0, 2, 1}}, torch::kInt64);
    a.edge_attr = values({4, 2});
    auto b = a.clone();
    b.x = values({4, 3});
    return {pyg::Batch::from_data_list({a, b}), values({2, 2})};
}

void cnn_case(const int id, const bool backward, const bool unbatched = false) {
    torch::manual_seed(211);
    policy::CnnActorCritic model(policy::CnnActorCriticOptions{3, id == 1 ? -7 : 5, id == 1 ? 1 : 128});
    auto x = values(unbatched ? std::vector<std::int64_t>{1, 5, 3}
                              : std::vector<std::int64_t>{2, 1, 5, 3});
    if (backward) {
        model->to(torch::kFloat64);
        x = x.to(torch::kFloat64).set_requires_grad(true);
    } else {
        model->eval();
    }
    const auto logits = model->act(x);
    const auto value = model->evaluate(x);
    if (backward) {
        const auto actor_loss = logits.square().sum();
        const auto critic_loss = value.square().sum();
        (actor_loss + critic_loss).backward();
    }
    std::cout << "CASE cnn_" << id << '\n';
    emit_tensor_bytes("logits", logits);
    emit_tensor_bytes("value", value);
    if (backward) emit_tensor_bytes("input_grad", x.grad());
    emit_module_bytes(*model, backward);
    emit_tensor_bytes("rng_tail", torch::rand({4}));
}

void gnn_case(const nn::GnnKind kind, const bool backward, const bool batch_norm = false) {
    torch::manual_seed(223);
    policy::GnnMlpPolicyOptions options{4, 3, 2, backward ? 3 : 2, 8, 2, backward ? 0.25 : 0.0, batch_norm};
    policy::GnnMlpActorCritic model(kind, options);
    if (kind == nn::GnnKind::deep_edge_gat && backward) fill_fixture_parameters(*model);
    if (!backward) model->eval();
    auto obs = observation();
    if (backward) {
        obs.p_net.x.set_requires_grad(true);
        obs.p_net.edge_attr->set_requires_grad(true);
        obs.v_node_x.set_requires_grad(true);
    }
    const auto logits = model->act(obs);
    const auto value = model->evaluate(obs);
    if (backward) {
        const auto actor_loss = logits.square().sum();
        const auto critic_loss = value.square().sum();
        (actor_loss + critic_loss).backward();
    }
    const auto kind_id = kind == nn::GnnKind::gcn ? 0 : (kind == nn::GnnKind::gat ? 1 : 2);
    std::cout << "CASE gnn_" << kind_id << '_' << backward << (batch_norm ? "_bn" : "") << '\n';
    emit_tensor_bytes("logits", logits);
    emit_tensor_bytes("value", value);
    if (backward) {
        emit_tensor_bytes("p_grad", obs.p_net.x.grad());
        emit_tensor_bytes("edge_grad", obs.p_net.edge_attr->grad());
        emit_tensor_bytes("v_grad", obs.v_node_x.grad());
    }
    emit_module_bytes(*model, backward);
    emit_tensor_bytes("rng_tail", torch::rand({4}));
}

void batch_norm_error_case() {
    bool rejected = false;
    try {
        policy::GcnMlpActorCritic model(policy::GnnMlpPolicyOptions{4, 3, 2, 2, 10, 2, 0.0, true});
        model->act(observation());
    } catch (const c10::Error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("expected Python GNN/MLP BatchNorm dimension error");
    std::cout << "CASE gnn_batch_norm_boundary\nerror|batch_norm_shape\n";
}

void benchmark(const int warmups, const int repetitions, const int iterations) {
    torch::NoGradGuard no_grad;
    torch::manual_seed(227);
    policy::CnnActorCritic cnn(policy::CnnActorCriticOptions{3, 5, 128});
    policy::GatMlpActorCritic gat(policy::GnnMlpPolicyOptions{4, 3, 2, 2, 8, 2, 0.0, false});
    cnn->eval();
    gat->eval();
    const auto image = values({16, 1, 32, 3});
    const auto obs = observation();
    const auto run = [&] { return std::vector<torch::Tensor>{
        cnn->act(image), cnn->evaluate(image), gat->act(obs), gat->evaluate(obs)}; };
    const auto reference = run();
    for (int i = 0; i < warmups; ++i) run();
    std::vector<std::int64_t> samples;
    std::vector<torch::Tensor> output;
    for (int sample = 0; sample < repetitions; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) output = run();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count() / iterations);
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (!torch::equal(output[i], reference[i])) throw std::runtime_error("timed output drift");
        }
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << "]}\n";
    for (std::size_t i = 0; i < reference.size(); ++i) {
        emit_tensor_bytes("output_" + std::to_string(i), reference[i]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool timed = argc > 1 && std::string_view(argv[1]) == "--benchmark";
        const int threads = timed ? (argc > 2 ? std::stoi(argv[2]) : 1)
                                  : (argc > 1 ? std::stoi(argv[1]) : 1);
        torch::set_num_threads(threads);
        torch::set_num_interop_threads(1);
        if (timed) {
            benchmark(3, 5, 10);
        } else {
            cnn_case(0, false);
            cnn_case(1, false);
            cnn_case(2, true);
            cnn_case(3, false, true);
            for (auto kind : {nn::GnnKind::gcn, nn::GnnKind::gat, nn::GnnKind::deep_edge_gat}) {
                gnn_case(kind, false);
                gnn_case(kind, true);
            }
            gnn_case(nn::GnnKind::gcn, true, true);
            batch_norm_error_case();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
