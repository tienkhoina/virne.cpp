#include "tensor_bytes.h"
#include "../virne/solver/learning/neural_network/mlp.h"
#include "../virne/solver/learning/neural_network/gnn.h"
#include "../virne/solver/learning/rl_policy/seq2seq_policy.h"
#include <torch/torch.h>

namespace n = virne::solver::learning::neural_network;
namespace p = virne::solver::learning::rl_policy;
namespace g = virne::solver::learning::pyg_lite;
using virne::porting::emit_tensor_bytes;
using virne::porting::emit_module_bytes;
namespace {
torch::Tensor values(const std::vector<std::int64_t>& shape) {
    std::int64_t count = 1; for (const auto size : shape) count *= size;
    return torch::arange(count,torch::kFloat32).reshape(shape).remainder(19).sub(9).div(16);
}
g::Batch graph() {
    g::Data a; a.x = values({3,3});
    a.edge_index = torch::tensor({{0,1,1,2},{1,0,2,1}},torch::kInt64); a.edge_attr = values({4,1});
    auto b = a.clone(); b.x = values({4,3}); return g::Batch::from_data_list({a,b});
}
void report(torch::nn::Module& model,const std::vector<torch::Tensor>& outputs,const std::vector<torch::Tensor>& inputs) {
    auto loss = outputs.front().square().sum();
    for (std::size_t i = 1; i < outputs.size(); ++i) loss = loss + outputs[i].square().sum();
    loss.backward();
    for (std::size_t i = 0; i < outputs.size(); ++i) emit_tensor_bytes("output_"+std::to_string(i),outputs[i]);
    for (std::size_t i = 0; i < inputs.size(); ++i) emit_tensor_bytes("input_grad_"+std::to_string(i),inputs[i].grad());
    emit_module_bytes(model,true); emit_tensor_bytes("rng_tail",torch::rand({4}));
}
void single_layer() {
    torch::manual_seed(739);
    n::MLPNet model(n::MlpOptions{4,3,1,std::monostate{},true,.25});
    auto input = values({2,4}).requires_grad_();
    report(*model,{model->forward(input)},{input});
}
void residual_initialization() {
    torch::manual_seed(743);
    n::DeepEdgeFeatureGATOptions options;
    options.input_dim = 3; options.output_dim = 2; options.edge_dim = 1;
    options.num_layers = 3; options.embedding_dim = 4; options.dropout_prob = 0.;
    n::DeepEdgeFeatureGAT model(options);
    auto data = graph(); data.x.requires_grad_(); data.edge_attr->requires_grad_();
    report(*model,{model->forward(data)},{data.x,*data.edge_attr});
}
void recurrent_clamp() {
    torch::manual_seed(751);
    p::GnnSeq2SeqActorCritic model(p::Seq2SeqOptions{4,3,1,2,8,0.,false,n::GnnKind::gcn,p::Seq2SeqFlavor::configurable_gnn});
    p::Seq2SeqObservation o; o.p_net = graph(); o.p_net.x.requires_grad_(); o.v_net_x = values({2,3,2}).requires_grad_();
    const auto encoded = model->encode(o);
    o.hidden_state = model->get_last_rnn_state().permute({1,0,2}); o.encoder_outputs = encoded.transpose(0,1);
    o.p_node_id = torch::tensor({77,101},torch::kInt64);
    const auto logits = model->act(o); const auto hidden = model->get_last_rnn_state(); const auto value = model->evaluate(o);
    report(*model,{encoded,logits,hidden,value},{o.v_net_x,o.p_net.x});
}
}
int main() {
    try {
        torch::set_num_threads(1); torch::set_num_interop_threads(1);
        std::cout << "CASE single_layer\n"; single_layer();
        std::cout << "CASE residual_initialization\n"; residual_initialization();
        std::cout << "CASE recurrent_clamp\n"; recurrent_clamp();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
