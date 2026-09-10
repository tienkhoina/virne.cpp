#include "../virne/solver/learning/neural_network/gnn.h"
#include "../virne/solver/learning/neural_network/loss.h"
#include "../virne/solver/learning/neural_network/multipler.h"
#include "../virne/solver/learning/neural_network/res.h"
#include "../virne/solver/learning/neural_network/sinkhorn.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn = virne::solver::learning::neural_network;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    int warmups = 1;
    int repetitions = 3;
    int threads = 1;
};

struct TensorRecord {
    std::string name;
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
};

struct CaseRecord {
    std::string name;
    std::vector<TensorRecord> state;
    std::vector<TensorRecord> outputs;
};

struct BenchmarkRecord {
    std::vector<std::int64_t> samples_ns;
    std::uint64_t checksum = fnv_offset;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: vne_neural_network_remaining_harness [--benchmark]"
          " [--warmups N] [--repetitions N]");
}

Arguments parse_arguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto positive = [&](const char* option) {
            if (++index >= argc) {
                usage_error(std::string(option) + " needs a value");
            }
            const int value = std::stoi(argv[index]);
            if (value <= 0) {
                usage_error(std::string(option) + " must be positive");
            }
            return value;
        };
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--warmups") {
            result.warmups = positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = positive("--repetitions");
        } else if (argument == "--threads") {
            result.threads = positive("--threads");
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

std::string dtype_name(const torch::ScalarType dtype) {
    switch (dtype) {
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    case torch::kInt64: return "int64";
    case torch::kBool: return "bool";
    default: throw std::runtime_error("unsupported tensor dtype in harness");
    }
}

std::uint64_t mix_tensor(
    std::uint64_t checksum,
    const torch::Tensor& input) {
    const auto value = input.detach().to(torch::kCPU).contiguous();
    const auto bytes = static_cast<std::size_t>(value.numel())
        * value.element_size();
    const auto* raw = reinterpret_cast<const std::uint8_t*>(value.data_ptr());
    for (std::size_t index = 0; index < bytes; ++index) {
        checksum ^= raw[index];
        checksum *= fnv_prime;
    }
    return checksum;
}

TensorRecord summarize(std::string name, const torch::Tensor& value) {
    if (!value.defined()) {
        throw std::runtime_error("cannot summarize an undefined tensor");
    }
    return TensorRecord{
        std::move(name),
        value.sizes().vec(),
        dtype_name(value.scalar_type()),
        mix_tensor(fnv_offset, value),
    };
}

std::vector<TensorRecord> state_records(torch::nn::Module& module) {
    std::vector<TensorRecord> records;
    for (const auto& item : module.named_parameters(true)) {
        records.push_back(summarize(item.key(), item.value()));
    }
    for (const auto& item : module.named_buffers(true)) {
        records.push_back(summarize(item.key(), item.value()));
    }
    std::sort(
        records.begin(),
        records.end(),
        [](const auto& left, const auto& right) {
            return left.name < right.name;
        });
    return records;
}

torch::Tensor values(
    const std::vector<std::int64_t>& shape,
    const double offset = 0.0) {
    std::int64_t count = 1;
    for (const auto extent : shape) count *= extent;
    return torch::arange(count, torch::kFloat32)
        .reshape(shape)
        .remainder(29.0)
        .sub(14.0)
        .div(9.0)
        .add(offset);
}

pyg::Batch graph_batch() {
    pyg::Data first;
    first.x = values({3, 3});
    first.edge_index = torch::tensor(
        {{0, 1, 1, 2}, {1, 0, 2, 1}}, torch::kInt64);
    first.edge_attr = values({4, 2}, 0.1);
    pyg::Data second;
    second.x = values({4, 3}, 0.25);
    second.edge_index = torch::tensor(
        {{0, 1, 2, 2, 3}, {1, 0, 1, 3, 2}}, torch::kInt64);
    second.edge_attr = values({5, 2}, -0.2);
    return pyg::Batch::from_data_list({first, second});
}

void fill_parameters(torch::nn::Module& module, const double scale = 0.01) {
    torch::NoGradGuard guard;
    std::int64_t parameter_id = 1;
    std::unordered_set<const c10::TensorImpl*> seen;
    for (auto& item : module.named_parameters(true)) {
        if (!seen.insert(item.value().unsafeGetTensorImpl()).second) continue;
        item.value().copy_(
            torch::arange(item.value().numel(), item.value().options())
                .reshape(item.value().sizes())
                .add(static_cast<double>(parameter_id))
                .mul(scale));
        ++parameter_id;
    }
}

CaseRecord multiplier_case() {
    torch::manual_seed(41);
    nn::MultiplerNet model(nn::MultiplerNetOptions{5});
    model->eval();
    CaseRecord result{"multipler", state_records(*model), {}};
    result.outputs.push_back(summarize("output", model->forward(values({7, 5}))));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord loss_case() {
    const auto h1 = values({8, 4});
    const auto h2 = values({8, 4}, 0.2).flip({0});
    nn::BarlowTwinsContrastiveLoss default_loss;
    nn::BarlowTwinsContrastiveLoss explicit_loss(
        nn::BarlowTwinsContrastiveLossOptions{0.125, false, 1e-4});
    CaseRecord result{"barlow_twins", {}, {}};
    result.outputs.push_back(summarize(
        "default", default_loss->forward(h1, h2)));
    result.outputs.push_back(summarize(
        "explicit", explicit_loss->forward(h1, h2)));
    return result;
}

CaseRecord residual_case() {
    torch::manual_seed(43);
    nn::ResNetBlock model(nn::ResNetBlockOptions{3});
    model->eval();
    CaseRecord result{"res_net_block", state_records(*model), {}};
    result.outputs.push_back(summarize(
        "output", model->forward(values({2, 3, 4, 5}))));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord sinkhorn_case() {
    const auto scores = torch::tensor({
        {0.2F, -0.1F, 0.7F},
        {0.4F, 0.3F, -0.5F},
        {-0.2F, 0.8F, 0.1F},
    });
    nn::Sinkhorn simple(nn::SinkhornOptions{7, 0.7});
    nn::SinkhornNetwork sequential(
        nn::SinkhornNetworkOptions{8, 0.8, 1e-4, true, false});
    nn::SinkhornNetwork batched(
        nn::SinkhornNetworkOptions{8, 0.8, 1e-4, true, true});
    nn::SinkhornNetworkCallOptions padded;
    padded.nrows = std::vector<std::int64_t>{3, 2};
    padded.ncols = std::vector<std::int64_t>{3, 2};
    nn::SinkhornNetworkCallOptions dummy;
    dummy.dummy = true;
    CaseRecord result{"sinkhorn", {}, {}};
    result.outputs.push_back(summarize(
        "simple", simple->forward(scores.unsqueeze(0))));
    result.outputs.push_back(summarize(
        "sequential", sequential->forward(scores)));
    result.outputs.push_back(summarize(
        "batched", batched->forward(scores)));
    result.outputs.push_back(summarize(
        "padded",
        sequential->forward(
            torch::stack({scores, scores + 0.25}), padded)));
    result.outputs.push_back(summarize(
        "dummy", sequential->forward(scores, dummy)));
    result.outputs.push_back(summarize(
        "logsumexp",
        nn::logsumexp(torch::tensor({1.0F, 2.0F, 3.0F}))));
    return result;
}

CaseRecord graph_helpers_case() {
    const auto dense = values({2, 3, 4});
    const auto mask = torch::tensor(
        {{true, false, true}, {false, true, true}}, torch::kBool);
    const auto adjacency = torch::tensor({
        {{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}},
        {{0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
    });
    nn::DenseToSparse converter;
    const auto converted = converter->forward(
        values({2, 3, 2}), adjacency, values({2, 3, 3}, 1.0));
    CaseRecord result{"graph_helpers", {}, {}};
    result.outputs.push_back(summarize(
        "to_sparse_batch", nn::to_sparse_batch(dense, mask)));
    result.outputs.push_back(summarize("x", converted.x));
    result.outputs.push_back(summarize("edge_index", converted.edge_index));
    result.outputs.push_back(summarize("edge_weight", converted.edge_weight));
    result.outputs.push_back(summarize("batch", converted.batch));
    return result;
}

CaseRecord attention_pool_case() {
    torch::manual_seed(47);
    nn::GraphAttentionPooling model(4);
    model->eval();
    const auto batch = graph_batch();
    const auto input = values({7, 4});
    CaseRecord result{"graph_attention_pool", state_records(*model), {}};
    result.outputs.push_back(summarize(
        "output", model->forward(input, *batch.batch, batch.num_graphs)));
    result.outputs.push_back(summarize(
        "coefs", model->get_coefs(input)));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord gat_case() {
    torch::manual_seed(53);
    nn::GATConvNet model(nn::GraphConvNetOptions{
        3, 4, 2, 5, 1, 2, false, 1.0, false,
        nn::GraphPoolingKind::attention});
    model->eval();
    CaseRecord result{"gat_conv_net", state_records(*model), {}};
    result.outputs.push_back(summarize("output", model->forward(graph_batch())));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord edge_fusion_case() {
    torch::manual_seed(59);
    nn::EdgeFusionGATConvNet model(nn::GraphConvNetOptions{
        3, 4, 2, 5, 1, 2, false, 1.0, true,
        nn::GraphPoolingKind::none});
    model->eval();
    CaseRecord result{"edge_fusion_gat_conv_net", state_records(*model), {}};
    result.outputs.push_back(summarize("output", model->forward(graph_batch())));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord deep_edge_case() {
    torch::manual_seed(61);
    nn::DeepEdgeFeatureGAT model(nn::DeepEdgeFeatureGATOptions{
        3, 4, 2, 3, 0.2, 0.2, 5, 1, false, 1.0, false,
        nn::GraphPoolingKind::mean});
    fill_parameters(*model);
    model->eval();
    CaseRecord result{"deep_edge_feature_gat", state_records(*model), {}};
    result.outputs.push_back(summarize("output", model->forward(graph_batch())));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord gcn_attention_case() {
    torch::manual_seed(67);
    nn::GCNConvNet model(nn::GCNConvNetOptions{
        3, 4, 5, 2, false, 1.0, false,
        nn::GraphPoolingKind::attention});
    model->eval();
    CaseRecord result{"gcn_attention_pool", state_records(*model), {}};
    result.outputs.push_back(summarize("output", model->forward(graph_batch())));
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

CaseRecord backward_case(
    const std::string& name,
    torch::nn::Module& model,
    const torch::Tensor& output,
    const std::vector<torch::Tensor>& inputs) {
    output.square().sum().backward();
    CaseRecord result{name, state_records(model), {summarize("output", output)}};
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        result.outputs.push_back(summarize(
            "input_grad_" + std::to_string(index), inputs[index].grad()));
    }
    std::vector<TensorRecord> gradients;
    std::unordered_set<const c10::TensorImpl*> seen;
    for (const auto& item : model.named_parameters(true)) {
        if (!seen.insert(item.value().unsafeGetTensorImpl()).second) continue;
        if (item.value().grad().defined()) {
            gradients.push_back(summarize("grad." + item.key(), item.value().grad()));
        }
    }
    std::sort(gradients.begin(), gradients.end(), [](const auto& a, const auto& b) {
        return a.name < b.name;
    });
    result.outputs.insert(result.outputs.end(), gradients.begin(), gradients.end());
    result.outputs.push_back(summarize("rng_tail", torch::rand({4})));
    return result;
}

std::vector<CaseRecord> lifecycle_cases() {
    torch::AutoGradMode enable_grad(true);
    std::vector<CaseRecord> results;
    {
        torch::manual_seed(71);
        nn::ResNetBlock model(nn::ResNetBlockOptions{3});
        model->to(torch::kFloat64);
        auto x = values({3, 4, 5}).to(torch::kFloat64).set_requires_grad(true);
        results.push_back(backward_case("res_unbatched_backward", *model,
                                        model->forward(x), {x}));
    }
    {
        torch::manual_seed(73);
        nn::MultiplerNet model(nn::MultiplerNetOptions{5});
        auto x = values({2, 3, 5}).set_requires_grad(true);
        results.push_back(backward_case("multipler_backward", *model,
                                        model->forward(x), {x}));
    }
    for (const bool norm : {false, true}) {
        torch::manual_seed(79);
        nn::BarlowTwinsContrastiveLoss model(
            nn::BarlowTwinsContrastiveLossOptions{0.125, norm, 1e-4});
        auto x = values({8, 4}).set_requires_grad(true);
        auto y = values({8, 4}, 0.2).flip({0}).set_requires_grad(true);
        results.push_back(backward_case(norm ? "loss_norm_backward" : "loss_raw_backward",
                                        *model, model->forward(x, y), {x, y}));
    }
    for (const bool batched : {false, true}) {
        torch::manual_seed(83);
        nn::SinkhornNetwork model(nn::SinkhornNetworkOptions{5, 0.8, 1e-4, false, batched});
        nn::SinkhornNetworkCallOptions call;
        call.r = torch::tensor({{0.5F, 1.5F}});
        call.c = torch::tensor({{0.5F, 0.75F, 0.75F}});
        call.dummy_row = true;
        auto x = values({1, 2, 3}).set_requires_grad(true);
        results.push_back(backward_case(batched ? "sinkhorn_batched_backward" : "sinkhorn_sequential_backward",
                                        *model, model->forward_log(x, call), {x}));
    }
    {
        torch::manual_seed(89);
        nn::Sinkhorn model(nn::SinkhornOptions{3, 0.7});
        auto x = values({2, 3, 3}).set_requires_grad(true);
        auto result = backward_case("sinkhorn_simple_backward", *model, model->forward(x, 0.02), {x});
        result.outputs.push_back(summarize("row_norm", model->row_norm(x)));
        result.outputs.push_back(summarize("col_norm", model->col_norm(x)));
        result.outputs.push_back(summarize("logsumexp_dim", nn::logsumexp(x, 1, true)));
        results.push_back(std::move(result));
    }
    {
        CaseRecord result{"sinkhorn_dummy_boundaries", {}, {}};
        for (const bool batched : {false, true}) {
            nn::SinkhornNetwork model(nn::SinkhornNetworkOptions{4, 0.8, 1e-4, true, batched});
            nn::SinkhornNetworkCallOptions call;
            call.nrows = std::vector<std::int64_t>{2};
            call.ncols = std::vector<std::int64_t>{3};
            call.dummy_row = true;
            const std::string prefix = batched ? "batched_" : "sequential_";
            result.outputs.push_back(summarize(prefix + "square", model->forward_log(values({1, 3, 3}), call)));
            result.outputs.push_back(summarize(prefix + "dummy", model->forward_log_dummy(values({1, 3, 3}), call)));
            call.dummy_row = false;
            call.nrows = std::vector<std::int64_t>{0};
            result.outputs.push_back(summarize(prefix + "empty", model->forward_log(values({1, 3, 3}), call)));
        }
        results.push_back(std::move(result));
    }
    for (const bool edges : {false, true}) {
        torch::manual_seed(97);
        pyg::EdgeFusionGATConvOptions options;
        options.in_channels = 3;
        options.out_channels = 4;
        if (edges) options.edge_dim = 2;
        pyg::EdgeFusionGATConv model(options);
        model->reset_parameters();
        auto batch = graph_batch();
        batch.x.set_requires_grad(true);
        std::vector<torch::Tensor> inputs{batch.x};
        if (edges) {
            batch.edge_attr->set_requires_grad(true);
            inputs.push_back(*batch.edge_attr);
        } else {
            batch.edge_attr.reset();
        }
        if (model->lin_src->weight.unsafeGetTensorImpl()
            != model->lin_dst->weight.unsafeGetTensorImpl()) {
            throw std::runtime_error("EdgeFusionGAT lost its shared projection");
        }
        results.push_back(backward_case(edges ? "edge_fusion_reset_edges_backward" : "edge_fusion_reset_no_edges_backward",
            *model, model->forward(batch.x, batch.edge_index, batch.edge_attr), inputs));
    }
    {
        torch::manual_seed(101);
        nn::GATConvNet model(nn::GraphConvNetOptions{
            3, 4, 3, 5, 7, 2, true, 0.25, false, nn::GraphPoolingKind::mean});
        auto batch = graph_batch();
        batch.x.set_requires_grad(true);
        batch.edge_attr->set_requires_grad(true);
        results.push_back(backward_case("gat_training_backward", *model,
                                        model->forward(batch), {batch.x, *batch.edge_attr}));
    }
    {
        torch::manual_seed(103);
        nn::DeepEdgeFeatureGAT model(nn::DeepEdgeFeatureGATOptions{
            3, 4, 2, 4, 0.2, 0.2, 5, 1, true, 0.25, false, nn::GraphPoolingKind::mean});
        fill_parameters(*model);
        auto batch = graph_batch();
        batch.x.set_requires_grad(true);
        batch.edge_attr->set_requires_grad(true);
        results.push_back(backward_case("deep_gat_training_backward", *model,
                                        model->forward(batch), {batch.x, *batch.edge_attr}));
    }
    {
        torch::manual_seed(107);
        nn::GraphAttentionPooling model(4);
        model->to(torch::kFloat64);
        auto x = values({7, 4}).to(torch::kFloat64).set_requires_grad(true);
        results.push_back(backward_case("attention_pool_backward", *model,
            model->forward(x, *graph_batch().batch, 3), {x}));
    }
    {
        torch::manual_seed(109);
        nn::DenseToSparse model;
        auto x = values({2, 3, 2}).set_requires_grad(true);
        // The pinned Python converter cannot backpropagate edge weights: its
        // in-place index offsets invalidate saved autograd indices.
        auto weights = values({2, 3, 3}).to(torch::kFloat64);
        const auto result = model->forward(x, values({2, 3, 3}), weights);
        results.push_back(backward_case("dense_to_sparse_x_backward", *model,
            torch::cat({result.x.flatten(), result.edge_weight}), {x}));
    }
    for (const bool scalar : {false, true}) {
        torch::manual_seed(113);
        nn::GATConvNet model(nn::GraphConvNetOptions{
            3, 4, 2, 5, 1, scalar ? 1 : 2, false, 1.0, false, nn::GraphPoolingKind::mean});
        model->eval();
        auto batch = graph_batch();
        if (scalar) batch.edge_attr = batch.edge_attr->select(1, 0).contiguous();
        else batch.edge_attr.reset();
        results.push_back(CaseRecord{scalar ? "gat_scalar_edges" : "gat_optional_edges",
            state_records(*model), {summarize("output", model->forward(batch)),
                                   summarize("rng_tail", torch::rand({4}))}});
    }
    {
        torch::manual_seed(127);
        nn::DeepEdgeFeatureGAT model(nn::DeepEdgeFeatureGATOptions{
            3, 4, std::nullopt, 3, 0.2, 0.0, 5, 1, false, 1.0, false, nn::GraphPoolingKind::mean});
        fill_parameters(*model);
        model->eval();
        auto batch = graph_batch();
        batch.edge_attr.reset();
        results.push_back(CaseRecord{"deep_no_edges_theta_zero", state_records(*model),
            {summarize("output", model->forward(batch)), summarize("rng_tail", torch::rand({4}))}});
    }
    return results;
}

class AggregateWorkload final {
public:
    AggregateWorkload()
        : batch_(graph_batch()),
          multiplier_(nn::MultiplerNetOptions{5}),
          contrastive_(),
          residual_(nn::ResNetBlockOptions{3}),
          sinkhorn_network_(
              nn::SinkhornNetworkOptions{6, 0.8, 1e-4, true, true}),
          sinkhorn_(nn::SinkhornOptions{5, 0.7}),
          attention_pool_(4),
          gat_(nn::GraphConvNetOptions{
              3, 4, 2, 5, 1, 2, false, 1.0, false,
              nn::GraphPoolingKind::attention}),
          edge_fusion_(nn::GraphConvNetOptions{
              3, 4, 2, 5, 1, 2, false, 1.0, false,
              nn::GraphPoolingKind::mean}),
          deep_(nn::DeepEdgeFeatureGATOptions{
              3, 4, 2, 3, 0.2, 0.2, 5, 1, false, 1.0, false,
              nn::GraphPoolingKind::mean}),
          gcn_(nn::GCNConvNetOptions{
              3, 4, 5, 2, false, 1.0, false,
              nn::GraphPoolingKind::attention}) {
        fill_parameters(*multiplier_);
        fill_parameters(*residual_);
        fill_parameters(*attention_pool_);
        fill_parameters(*gat_);
        fill_parameters(*edge_fusion_);
        fill_parameters(*deep_);
        fill_parameters(*gcn_);
        multiplier_->eval();
        residual_->eval();
        attention_pool_->eval();
        gat_->eval();
        edge_fusion_->eval();
        deep_->eval();
        gcn_->eval();
        const auto graph_count = batch_.num_graphs;
        attention_batch_ = *batch_.batch;
        attention_size_ = graph_count;
    }

    std::vector<torch::Tensor> run() {
        nn::DenseToSparse converter;
        const auto mask = torch::tensor(
            {{true, false, true}, {false, true, true}}, torch::kBool);
        const auto adjacency = torch::tensor({
            {{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}},
            {{0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
        });
        const auto converted = converter->forward(
            values({2, 3, 2}), adjacency, values({2, 3, 3}, 1.0));
        const auto scores = torch::stack({
            values({3, 3}),
            values({3, 3}, 0.25),
        });
        return {
            multiplier_->forward(values({32, 5})),
            contrastive_->forward(values({32, 4}), values({32, 4}, 0.2)),
            residual_->forward(values({4, 3, 8, 8})),
            sinkhorn_network_->forward(scores),
            sinkhorn_->forward(scores),
            nn::to_sparse_batch(values({2, 3, 4}), mask),
            converted.x,
            converted.edge_index,
            converted.edge_weight,
            converted.batch,
            attention_pool_->forward(
                values({7, 4}), attention_batch_, attention_size_),
            gat_->forward(batch_),
            edge_fusion_->forward(batch_),
            deep_->forward(batch_),
            gcn_->forward(batch_),
        };
    }

private:
    pyg::Batch batch_;
    nn::MultiplerNet multiplier_{nullptr};
    nn::BarlowTwinsContrastiveLoss contrastive_{nullptr};
    nn::ResNetBlock residual_{nullptr};
    nn::SinkhornNetwork sinkhorn_network_{nullptr};
    nn::Sinkhorn sinkhorn_{nullptr};
    nn::GraphAttentionPooling attention_pool_{nullptr};
    nn::GATConvNet gat_{nullptr};
    nn::EdgeFusionGATConvNet edge_fusion_{nullptr};
    nn::DeepEdgeFeatureGAT deep_{nullptr};
    nn::GCNConvNet gcn_{nullptr};
    torch::Tensor attention_batch_;
    std::int64_t attention_size_ = 0;
};

std::uint64_t workload_checksum(
    const std::vector<torch::Tensor>& outputs) {
    auto checksum = fnv_offset;
    for (const auto& output : outputs) {
        checksum = mix_tensor(checksum, output);
    }
    return checksum;
}

BenchmarkRecord benchmark(
    const Arguments& arguments,
    AggregateWorkload& workload) {
    BenchmarkRecord result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        const auto checksum = workload_checksum(workload.run());
        if (warmup == 0) result.checksum = checksum;
        if (checksum != result.checksum) {
            throw std::runtime_error("aggregate warmup checksum drift");
        }
    }
    result.samples_ns.reserve(
        static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0;
         repetition < arguments.repetitions;
         ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto outputs = workload.run();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (workload_checksum(outputs) != result.checksum) {
            throw std::runtime_error("aggregate benchmark checksum drift");
        }
    }
    return result;
}

void emit_tensor(const TensorRecord& value) {
    std::cout << "{\"name\":\"" << value.name << "\",\"shape\":[";
    for (std::size_t index = 0; index < value.shape.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << value.shape[index];
    }
    std::cout << "],\"dtype\":\"" << value.dtype
              << "\",\"checksum\":\"" << value.checksum << "\"}";
}

void emit_case(const CaseRecord& value) {
    std::cout << "{\"name\":\"" << value.name << "\",\"state\":[";
    for (std::size_t index = 0; index < value.state.size(); ++index) {
        if (index != 0) std::cout << ',';
        emit_tensor(value.state[index]);
    }
    std::cout << "],\"outputs\":[";
    for (std::size_t index = 0; index < value.outputs.size(); ++index) {
        if (index != 0) std::cout << ',';
        emit_tensor(value.outputs[index]);
    }
    std::cout << "]}";
}

void emit_report(
    const Arguments& arguments,
    const std::vector<CaseRecord>& cases,
    const BenchmarkRecord* timed) {
    std::cout << "{\"component\":\"neural_network_remaining\",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) std::cout << ',';
        emit_case(cases[index]);
    }
    std::cout << "],\"boundaries\":{"
                 "\"res_linear_net\":\"python_constructor_error\","
                 "\"nn_conv_net\":\"python_constructor_error\","
                 "\"pna_conv_net\":\"python_constructor_error\"}";
    if (timed != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"all_valid_leaves\""
                  << ",\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0; index < timed->samples_ns.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << timed->samples_ns[index];
        }
        std::cout << "],\"checksum\":\"" << timed->checksum << "\"}";
    }
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(1);
        torch::NoGradGuard no_grad;
        std::vector<CaseRecord> cases{
            multiplier_case(),
            loss_case(),
            residual_case(),
            sinkhorn_case(),
            graph_helpers_case(),
            attention_pool_case(),
            gat_case(),
            edge_fusion_case(),
            deep_edge_case(),
            gcn_attention_case(),
        };
        const auto lifecycle = lifecycle_cases();
        cases.insert(cases.end(), lifecycle.begin(), lifecycle.end());
        AggregateWorkload workload;
        BenchmarkRecord timed;
        if (arguments.benchmark) timed = benchmark(arguments, workload);
        emit_report(
            arguments,
            cases,
            arguments.benchmark ? &timed : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neural_network_remaining_harness: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
