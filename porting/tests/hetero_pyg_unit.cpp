#include "../../virne/solver/learning/pyg_lite/hetero_conv.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor indices(const std::initializer_list<std::initializer_list<std::int64_t>>& values) {
    std::vector<std::int64_t> flat;
    std::size_t columns = 0U;
    for (const auto& row : values) {
        if (columns == 0U) columns = row.size();
        if (row.size() != columns) throw std::runtime_error("ragged index fixture");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return torch::tensor(flat, torch::TensorOptions().dtype(torch::kInt64))
        .reshape({static_cast<std::int64_t>(values.size()), static_cast<std::int64_t>(columns)});
}

pyg::HeteroData make_graph(const float offset, const std::int64_t p_nodes, const std::int64_t v_nodes) {
    pyg::HeteroData data;
    data.node(pyg::HeteroNodeType::physical) =
        torch::arange(p_nodes * 2, torch::TensorOptions().dtype(torch::kFloat32))
            .reshape({p_nodes, 2}) + offset;
    data.node(pyg::HeteroNodeType::virtual_node) =
        torch::arange(v_nodes * 3, torch::TensorOptions().dtype(torch::kFloat32))
            .reshape({v_nodes, 3}) + offset;
    data.relation(pyg::HeteroRelationId::p_connect_p).edge_index =
        indices({{0}, {p_nodes > 1 ? 1 : 0}});
    data.relation(pyg::HeteroRelationId::v_connect_v).edge_index =
        indices({{0}, {v_nodes > 1 ? 1 : 0}});
    data.relation(pyg::HeteroRelationId::v_mapping_p).edge_index =
        indices({{0}, {0}});
    data.relation(pyg::HeteroRelationId::p_mapping_v).edge_index =
        indices({{0}, {0}});
    data.relation(pyg::HeteroRelationId::v_imaginary_p).edge_index =
        indices({{0}, {0}});
    data.relation(pyg::HeteroRelationId::p_imaginary_v).edge_index =
        indices({{0}, {0}});
    data.validate();
    return data;
}

void test_fixed_relation_ids() {
    require(
        pyg::hetero_relation_id(
            pyg::HeteroNodeType::physical,
            pyg::HeteroRelationKind::connect,
            pyg::HeteroNodeType::physical)
            == pyg::HeteroRelationId::p_connect_p,
        "physical connect relation id");
    require(
        pyg::hetero_relation_id(
            pyg::HeteroNodeType::virtual_node,
            pyg::HeteroRelationKind::mapping,
            pyg::HeteroNodeType::physical)
            == pyg::HeteroRelationId::v_mapping_p,
        "virtual mapping relation id");
    require(
        !pyg::hetero_relation_id(
             pyg::HeteroNodeType::physical,
             pyg::HeteroRelationKind::connect,
             pyg::HeteroNodeType::virtual_node)
             .has_value(),
        "cross-type connect must be rejected");
}

void test_batch_offsets() {
    const auto first = make_graph(0.0F, 2, 3);
    const auto second = make_graph(10.0F, 3, 2);
    const auto batch = pyg::HeteroBatch::from_data_list({first, second});
    require(batch.num_graphs == 2, "batch graph count");
    require(batch.node(pyg::HeteroNodeType::physical).size(0) == 5, "batched p count");
    require(batch.node(pyg::HeteroNodeType::virtual_node).size(0) == 5, "batched v count");
    require(batch.node_ptr[0].equal(torch::tensor({0, 2, 5}, torch::kInt64)), "p ptr");
    require(batch.node_ptr[1].equal(torch::tensor({0, 3, 5}, torch::kInt64)), "v ptr");
    const auto& mapping = batch.relation(pyg::HeteroRelationId::v_mapping_p).edge_index;
    require(mapping.equal(indices({{0, 3}, {0, 2}})), "mapping edge offsets");
    require(batch.batch[0].has_value() && batch.batch[1].has_value(), "batch vectors");
}

void test_bipartite_and_hetero_conv() {
    pyg::BipartiteGATConvOptions options{};
    options.source_in_channels = 3;
    options.target_in_channels = 2;
    options.out_channels = 2;
    options.heads = 2;
    options.edge_dim = 1;
    options.add_self_loops = false;
    auto layer = pyg::BipartiteGATConv(options);
    const auto graph = make_graph(0.0F, 2, 3);
    const auto& relation = graph.relation(pyg::HeteroRelationId::v_mapping_p);
    const auto output = layer->forward(
        graph.node(pyg::HeteroNodeType::virtual_node),
        graph.node(pyg::HeteroNodeType::physical),
        relation.edge_index,
        torch::ones({relation.edge_index.size(1), 1}));
    require(output.sizes() == torch::IntArrayRef({2, 4}), "bipartite output shape");

    pyg::HeteroConvOptions hetero_options{};
    const std::array<std::pair<std::int64_t, std::int64_t>, pyg::kHeteroRelations> channels{
        std::pair{2, 2}, std::pair{3, 3}, std::pair{3, 2},
        std::pair{2, 3}, std::pair{3, 2}, std::pair{2, 3}};
    for (std::size_t index = 0; index < pyg::kHeteroRelations; ++index) {
        hetero_options.enabled[index] = true;
        hetero_options.relation_options[index].source_in_channels = channels[index].first;
        hetero_options.relation_options[index].target_in_channels = channels[index].second;
        hetero_options.relation_options[index].out_channels = 2;
        hetero_options.relation_options[index].edge_dim = 1;
        hetero_options.relation_options[index].add_self_loops = index < 2;
    }
    auto hetero = pyg::HeteroConv(hetero_options);
    auto data = graph.clone();
    for (std::size_t index = 0; index < pyg::kHeteroRelations; ++index) {
        data.relations[index].edge_attr = torch::ones({data.relations[index].edge_index.size(1), 1});
    }
    const auto result = hetero->forward(data);
    require(result.node(pyg::HeteroNodeType::physical).sizes() == torch::IntArrayRef({2, 2}), "hetero p shape");
    require(result.node(pyg::HeteroNodeType::virtual_node).sizes() == torch::IntArrayRef({3, 2}), "hetero v shape");
}

void test_invalid_edge_index() {
    auto data = make_graph(0.0F, 2, 3);
    data.relation(pyg::HeteroRelationId::v_mapping_p).edge_index = indices({{0}, {3}});
    bool threw = false;
    try {
        data.validate();
    } catch (const pyg::PygLiteException&) {
        threw = true;
    }
    require(threw, "out-of-range hetero endpoint must fail validation");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::manual_seed(0);
        test_fixed_relation_ids();
        test_batch_offsets();
        test_bipartite_and_hetero_conv();
        test_invalid_edge_index();
        std::cout << "hetero_pyg_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hetero_pyg_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
