#include "tensor_bytes.h"
#include "../virne/solver/learning/rl_core/shared_adam.h"
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <thread>

namespace rl = virne::solver::learning::rl_core;
using virne::porting::emit_tensor_bytes;
namespace {
std::unique_ptr<rl::SharedAdam> optimizer(const bool maximum = false) {
    auto a = torch::arange(6, torch::kFloat32).reshape({2,3}).div(8).requires_grad_();
    auto b = torch::arange(3, torch::kFloat64).div(8).requires_grad_();
    rl::AdamOptions first{.003,.9,.99,1e-8,.01,maximum,false}, second{.007,.85,.98,1e-7,.02,maximum,false};
    return std::make_unique<rl::SharedAdam>(std::vector<rl::AdamParameterGroup>{{{a},first},{{b},second}});
}
std::vector<torch::Tensor> packet(const int step = 0, const bool missing = false) {
    return {torch::full({2,3}, .1 + step*.03, torch::kFloat32),
        missing ? torch::Tensor{} : torch::full({3}, -.2 + step*.02, torch::kFloat64)};
}
void emit(const rl::SharedAdamSnapshot& state) {
    emit_tensor_bytes("version", torch::tensor(static_cast<std::int64_t>(state.version)));
    for (std::size_t i = 0; i < state.parameters.size(); ++i) {
        const auto name = "parameter_" + std::to_string(i);
        const auto& m = state.optimizer.moments[i];
        emit_tensor_bytes(name, state.parameters[i]); emit_tensor_bytes(name + ".step", m.step);
        emit_tensor_bytes(name + ".mean", m.exp_avg); emit_tensor_bytes(name + ".variance", m.exp_avg_sq);
        emit_tensor_bytes(name + ".maximum", m.max_exp_avg_sq);
    }
}
void sequential(const bool maximum) {
    auto shared = optimizer(maximum);
    emit(shared->snapshot());
    rl::SharedAdamSnapshot saved;
    for (int step = 0; step < 4; ++step) {
        if (step == 2) shared->restore(saved);
        const auto result = shared->apply_gradients(packet(step, step == 1), .25);
        emit_tensor_bytes("norm", result.gradient_norm); emit(shared->snapshot());
        if (step == 0) saved = shared->snapshot();
    }
    bool rejected = false;
    try { shared->apply_gradients({torch::zeros({1}), torch::zeros({3}, torch::kFloat64)}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid shared packet accepted");
    emit(shared->snapshot());
    emit_tensor_bytes("invalid_rejected", torch::tensor(rejected));
}
void concurrent() {
    auto shared = optimizer();
    const auto gradients = packet();
    std::array<std::uint64_t, 64> versions{};
    std::array<std::exception_ptr, 4> errors{};
    std::array<std::thread, 4> workers;
    for (std::size_t worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
        try {
            for (std::size_t step = 0; step < 16; ++step) {
                versions[worker*16+step] = shared->apply_gradients(gradients, .25).version;
                const auto state = shared->snapshot();
                for (const auto& moment : state.optimizer.moments)
                    if (moment.step.item<std::int64_t>() != static_cast<std::int64_t>(state.version)) throw std::runtime_error("incoherent shared snapshot");
            }
        } catch (...) { errors[worker] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    std::sort(versions.begin(), versions.end());
    for (std::size_t i = 0; i < versions.size(); ++i) if (versions[i] != i+1) throw std::runtime_error("lost shared update");
    emit(shared->snapshot());
}
void benchmark() {
    std::vector<std::int64_t> samples;
    rl::SharedAdamSnapshot final;
    for (int repetition = -3; repetition < 5; ++repetition) {
        auto shared = optimizer(); const auto gradients = packet();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 32; ++i) shared->apply_gradients(gradients, .25);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        if (repetition >= 0) samples.push_back(elapsed);
        const auto current = shared->snapshot();
        if (repetition > -3) for (std::size_t i = 0; i < final.parameters.size(); ++i)
            if (!torch::equal(current.parameters[i], final.parameters[i])) throw std::runtime_error("unstable shared benchmark");
        final = current;
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
    std::cout << "]}\n"; emit(final);
}
}
int main(int argc, char** argv) {
    try {
        torch::set_num_threads(1); torch::set_num_interop_threads(1);
        if (argc == 2 && std::string(argv[1]) == "--benchmark") { benchmark(); return 0; }
        if (argc != 1) throw std::invalid_argument("invalid shared optimizer harness arguments");
        std::cout << "CASE shared\n"; sequential(false);
        std::cout << "CASE shared_amsgrad\n"; sequential(true);
        std::cout << "CASE shared_threads\n"; concurrent();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
