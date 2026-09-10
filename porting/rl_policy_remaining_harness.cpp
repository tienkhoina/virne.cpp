#include "tensor_bytes.h"
#include "../virne/solver/learning/rl_policy/att_policy.h"
#include "../virne/solver/learning/rl_policy/dual_gnn_policy.h"
#include "../virne/solver/learning/rl_policy/seq2seq_policy.h"
#include <torch/torch.h>
#include <chrono>
#include <functional>
#include <stdexcept>

namespace p = virne::solver::learning::rl_policy;
namespace n = virne::solver::learning::neural_network;
namespace g = virne::solver::learning::pyg_lite;
using virne::porting::emit_tensor_bytes;
using virne::porting::emit_module_bytes;
using virne::porting::fill_fixture_parameters;

namespace {
torch::Tensor values(const std::vector<std::int64_t>& shape) {
    std::int64_t count = 1;
    for (auto size : shape) count *= size;
    return torch::arange(count, torch::kFloat32).reshape(shape).remainder(19).sub(9).div(16);
}
g::Batch graph(const std::int64_t features) {
    g::Data a;
    a.x = values({3, features});
    a.edge_index = torch::tensor({{0, 1, 1, 2}, {1, 0, 2, 1}}, torch::kInt64);
    a.edge_attr = values({4, 1});
    auto b = a.clone();
    b.x = values({4, features});
    return g::Batch::from_data_list({a, b});
}
void report(torch::nn::Module& model, const std::vector<torch::Tensor>& outputs,
    const std::vector<torch::Tensor>& inputs, const bool backward) {
    if (backward) {
        auto loss = outputs.front().square().sum();
        for (std::size_t i = 1; i < outputs.size(); ++i) loss = loss + outputs[i].square().sum();
        loss.backward();
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) emit_tensor_bytes("output_" + std::to_string(i), outputs[i]);
    if (backward) for (std::size_t i = 0; i < inputs.size(); ++i) emit_tensor_bytes("input_grad_" + std::to_string(i), inputs[i].grad());
    emit_module_bytes(model, backward);
    emit_tensor_bytes("rng_tail", torch::rand({4}));
}

void attention(const bool backward, const bool use_double) {
    torch::manual_seed(301);
    p::AttActorCritic model(p::AttPolicyOptions{4, 3, 2, 2, 8, 0.2, false});
    p::AttObservation obs{values({2, 4, 3}), values({2, 2})};
    if (use_double) {
        model->to(torch::kFloat64);
        obs.p_net_x = obs.p_net_x.to(torch::kFloat64);
        obs.v_node_x = obs.v_node_x.to(torch::kFloat64);
    }
    model->train(backward);
    obs.p_net_x.set_requires_grad(backward);
    obs.v_node_x.set_requires_grad(backward);
    const auto logits = model->act(obs);
    const auto value = model->evaluate(obs);
    report(*model, {logits, value}, {obs.p_net_x, obs.v_node_x}, backward);
}
void attention_encoder(const bool backward) {
    torch::manual_seed(307);
    p::AttEncoder model(p::AttPolicyOptions{4, 3, 2, 2, 8, 0.0, false}, 2);
    model->train(backward);
    p::AttObservation obs{values({2, 4, 3}), values({2, 2})};
    obs.p_net_x.set_requires_grad(backward);
    obs.v_node_x.set_requires_grad(backward);
    torch::AutoGradMode grad(backward);
    report(*model, {model->forward(obs)}, {obs.p_net_x, obs.v_node_x}, backward);
}
p::DualPolicyOptions dual_options(const int kind, const int auxiliary) {
    return {4, 3, 1, 2, 1, 4, 0.2, false, 3,
        kind == 0 ? n::GnnKind::gcn : (kind == 1 ? n::GnnKind::gat : n::GnnKind::deep_edge_gat),
        std::nullopt, std::nullopt, auxiliary >= 1, auxiliary >= 2};
}
void dual(const int kind, const int auxiliary, const bool backward, const bool mixed = false) {
    torch::manual_seed(311);
    auto options = dual_options(kind, auxiliary);
    options.batch_norm = backward;
    if (mixed) { options.p_kind = n::GnnKind::gat; options.v_kind = n::GnnKind::gcn; }
    p::DualActorCritic model(options);
    model->train(backward);
    p::DualObservation obs{graph(2), graph(3), torch::tensor({1, 2}, torch::kInt64)};
    obs.v_net.x.set_requires_grad(backward);
    obs.p_net.x.set_requires_grad(backward);
    obs.v_net.edge_attr->set_requires_grad(backward);
    obs.p_net.edge_attr->set_requires_grad(backward);
    std::vector<torch::Tensor> outputs{model->act(obs), model->evaluate(obs)};
    if (model->cost_critic) outputs.push_back(model->cost_critic->forward(obs));
    if (model->lambda_net) outputs.push_back(model->lambda_net->forward(obs));
    report(*model, outputs, {obs.v_net.x, obs.p_net.x, *obs.v_net.edge_attr, *obs.p_net.edge_attr}, backward);
}
p::Seq2SeqOptions sequence_options(const int kind) {
    p::Seq2SeqOptions o{4, 3, 1, 2, 8, 0.2, false, n::GnnKind::gcn, p::Seq2SeqFlavor::legacy_gcn};
    if (kind) {
        o.flavor = p::Seq2SeqFlavor::configurable_gnn;
        o.kind = kind == 1 ? n::GnnKind::gcn : (kind == 2 ? n::GnnKind::gat : n::GnnKind::deep_edge_gat);
    }
    return o;
}
void sequence(const int kind, const bool backward) {
    torch::manual_seed(313);
    p::GnnSeq2SeqActorCritic model(sequence_options(kind));
    if (kind == 3) fill_fixture_parameters(*model);
    model->train(backward);
    p::Seq2SeqObservation obs;
    obs.p_net = graph(3);
    obs.v_net_x = values({2, 3, 2}).set_requires_grad(backward);
    obs.p_net.x.set_requires_grad(backward);
    emit_tensor_bytes("initial_hidden", model->get_last_rnn_state());
    const auto encoded = model->encode(obs);
    obs.hidden_state = model->get_last_rnn_state().permute({1, 0, 2});
    obs.encoder_outputs = encoded.transpose(0, 1);
    obs.p_node_id = torch::tensor({0, 4}, torch::kInt64);
    obs.mask = torch::tensor({{1, 1, 0}, {1, 0, 1}}, torch::kInt64);
    const auto logits = model->act(obs);
    const auto next_hidden = model->get_last_rnn_state();
    const auto value = model->evaluate(obs);
    if (!model->get_last_rnn_state().is_same(next_hidden)) throw std::runtime_error("critic mutated last hidden");
    const auto replacement = torch::zeros_like(next_hidden);
    model->set_last_rnn_hidden(replacement);
    emit_tensor_bytes("replaced_hidden", model->get_last_rnn_state());
    report(*model, {encoded, logits, next_hidden, value}, {obs.v_net_x, obs.p_net.x}, backward);
}
void additive_attention() {
    torch::manual_seed(317);
    p::AdditiveAttention model(8);
    auto hidden = values({1, 2, 8}).set_requires_grad(true);
    auto encoder = values({2, 3, 8}).set_requires_grad(true);
    const auto mask = torch::zeros({2, 3}, torch::kInt64);
    const auto result = model->forward(hidden, encoder, mask);
    report(*model, {result.context, result.weights}, {hidden, encoder}, true);
}

void exact(const std::string& selected) {
    const auto run = [&](const std::string& name, const std::function<void()>& action) {
        if (!selected.empty() && selected != name) return;
        std::cout << "CASE " << name << '\n';
        action();
    };
    run("att_eval", [] { attention(false, false); });
    run("att_train", [] { attention(true, false); });
    run("att_double", [] { attention(true, true); });
    run("att_heads2_eval", [] { attention_encoder(false); });
    run("att_heads2_train", [] { attention_encoder(true); });
    for (int kind = 0; kind < 3; ++kind) {
        for (int auxiliary = 0; auxiliary < 3; ++auxiliary) {
            run("dual_" + std::to_string(kind) + "_" + std::to_string(auxiliary),
                [=] { dual(kind, auxiliary, false); });
        }
        run("dual_train_" + std::to_string(kind), [=] { dual(kind, 2, true); });
    }
    run("dual_mixed", [] { dual(2, 2, false, true); });
    for (int kind = 0; kind < 4; ++kind) {
        run("seq_" + std::to_string(kind), [=] { sequence(kind, false); });
        run("seq_train_" + std::to_string(kind), [=] { sequence(kind, true); });
    }
    run("additive_mask", additive_attention);
}
void benchmark() {
    torch::NoGradGuard guard;
    torch::manual_seed(331);
    p::AttActorCritic att(p::AttPolicyOptions{4, 3, 2, 2, 8, 0.0, false});
    p::DualActorCritic dual(dual_options(0, 0));
    p::GnnSeq2SeqActorCritic seq(sequence_options(0));
    att->eval(); dual->eval(); seq->eval();
    p::AttObservation dense{values({2, 4, 3}), values({2, 2})};
    p::DualObservation graphs{graph(2), graph(3), torch::tensor({1, 2}, torch::kInt64)};
    p::Seq2SeqObservation obs;
    obs.p_net = graph(3); obs.v_net_x = values({2, 3, 2});
    const auto enc = seq->encode(obs);
    obs.encoder_outputs = enc.transpose(0, 1);
    obs.hidden_state = seq->get_last_rnn_state().permute({1, 0, 2});
    obs.p_node_id = torch::tensor({0, 4}, torch::kInt64);
    const auto run = [&] { return std::vector<torch::Tensor>{
        att->act(dense), att->evaluate(dense), dual->act(graphs), dual->evaluate(graphs), seq->act(obs), seq->evaluate(obs)}; };
    const auto reference = run();
    for (int i = 0; i < 3; ++i) run();
    std::vector<std::int64_t> samples;
    for (int sample = 0; sample < 5; ++sample) {
        std::vector<torch::Tensor> output;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) output = run();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count() / 5);
        for (std::size_t i = 0; i < output.size(); ++i) if (!torch::equal(output[i], reference[i])) throw std::runtime_error("benchmark output drift");
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
    std::cout << "]}\n";
    for (std::size_t i = 0; i < reference.size(); ++i) emit_tensor_bytes("output_" + std::to_string(i), reference[i]);
}
}  // namespace
int main(int argc, char** argv) {
    try {
        std::string selected;
        bool timed = false;
        int threads = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--benchmark") timed = true;
            else if (argument == "--case" && i + 1 < argc) selected = argv[++i];
            else if (argument == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else throw std::invalid_argument("invalid harness argument");
        }
        torch::set_num_threads(threads); torch::set_num_interop_threads(1);
        if (timed) benchmark(); else exact(selected);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
