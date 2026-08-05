#include "../../virne/solver/learning/rl_core/graph_observation_sampler.h"

#include <torch/torch.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rl = virne::solver::learning::rl_core;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message) {
    require(actual.defined() && expected.defined(), message + " defined");
    require(actual.scalar_type() == expected.scalar_type(), message + " dtype");
    require(actual.sizes() == expected.sizes(), message + " shape");
    require(torch::equal(actual.to(torch::kCPU), expected.to(torch::kCPU)),
            message + " values");
}

void require_optional_equal(
    const std::optional<torch::Tensor>& actual,
    const std::optional<torch::Tensor>& expected,
    const std::string& message) {
    require(actual.has_value() == expected.has_value(), message + " presence");
    if (actual.has_value()) require_equal(*actual, *expected, message);
}

pyg::Data graph(const std::int64_t seed, const std::int64_t nodes) {
    pyg::Data result;
    result.x = torch::arange(
        seed * 100, seed * 100 + nodes * 2, torch::kFloat32).reshape({nodes, 2});
    std::vector<std::int64_t> edges;
    for (std::int64_t node = 0; node + 1 < nodes; ++node) {
        edges.push_back(node);
        edges.push_back(node + 1);
        edges.push_back(node + 1);
        edges.push_back(node);
    }
    result.edge_index = edges.empty()
        ? torch::empty({2, 0}, torch::kInt64)
        : torch::tensor(edges, torch::kInt64).reshape({-1, 2}).t().contiguous();
    result.edge_attr = torch::arange(
        result.edge_index.size(1), torch::kFloat32).reshape({-1, 1})
                           .add_(static_cast<float>(seed));
    result.validate();
    return result;
}

void require_batch_equal(
    const pyg::Batch& actual,
    const pyg::Batch& expected,
    const std::string& message) {
    require_equal(actual.x, expected.x, message + " x");
    require_equal(actual.edge_index, expected.edge_index,
                  message + " edge_index");
    require_optional_equal(actual.edge_attr, expected.edge_attr,
                           message + " edge_attr");
    require_optional_equal(actual.batch, expected.batch, message + " batch");
    require_optional_equal(actual.ptr, expected.ptr, message + " ptr");
    require_equal(actual.edge_ptr, expected.edge_ptr, message + " edge_ptr");
    require(actual.num_graphs == expected.num_graphs,
            message + " graph count");
    require(actual.max_num_nodes == expected.max_num_nodes,
            message + " max nodes");
}

pyg::HeteroRelationData relation(
    const std::vector<std::int64_t>& edges,
    const float seed) {
    pyg::HeteroRelationData result;
    result.edge_index = edges.empty()
        ? torch::empty({2, 0}, torch::kInt64)
        : torch::tensor(edges, torch::kInt64).reshape({-1, 2}).t().contiguous();
    result.edge_attr = torch::arange(
        result.edge_index.size(1), torch::kFloat32).reshape({-1, 1}).add_(seed);
    return result;
}

pyg::HeteroData hetero_graph(
    const std::int64_t seed,
    const std::int64_t p_nodes,
    const std::int64_t v_nodes,
    const bool augment_virtual,
    const bool augment_physical) {
    pyg::HeteroData result;
    result.node(pyg::HeteroNodeType::physical) = torch::arange(
        seed * 100, seed * 100 + p_nodes * 2, torch::kFloat32)
                                                     .reshape({p_nodes, 2});
    result.node(pyg::HeteroNodeType::virtual_node) = torch::arange(
        seed * 10, seed * 10 + v_nodes, torch::kFloat32).reshape({v_nodes, 1});

    std::vector<std::int64_t> p_edges;
    for (std::int64_t node = 0; node + 1 < p_nodes; ++node) {
        p_edges.insert(p_edges.end(), {node, node + 1, node + 1, node});
    }
    if (augment_physical && p_nodes != 0) p_edges.insert(p_edges.end(), {0, 0});
    std::vector<std::int64_t> v_edges;
    for (std::int64_t node = 0; node + 1 < v_nodes; ++node) {
        v_edges.insert(v_edges.end(), {node, node + 1, node + 1, node});
    }
    if (augment_virtual && v_nodes != 0) v_edges.insert(v_edges.end(), {0, 0});
    std::vector<std::int64_t> mapping;
    if (p_nodes != 0 && v_nodes != 0 && seed != 2) mapping = {0, 0};
    std::vector<std::int64_t> imaginary;
    if (p_nodes != 0 && v_nodes != 0 && seed != 1) {
        imaginary = {v_nodes - 1, p_nodes - 1};
    }

    result.relation(pyg::HeteroRelationId::p_connect_p) =
        relation(p_edges, static_cast<float>(seed) + 0.1F);
    result.relation(pyg::HeteroRelationId::v_connect_v) =
        relation(v_edges, static_cast<float>(seed) + 0.2F);
    result.relation(pyg::HeteroRelationId::v_mapping_p) =
        relation(mapping, static_cast<float>(seed) + 0.3F);
    std::vector<std::int64_t> reverse_mapping;
    for (std::size_t edge = 0U; edge + 1U < mapping.size(); edge += 2U) {
        reverse_mapping.insert(reverse_mapping.end(),
                               {mapping[edge + 1U], mapping[edge]});
    }
    result.relation(pyg::HeteroRelationId::p_mapping_v) =
        relation(reverse_mapping, static_cast<float>(seed) + 0.3F);
    result.relation(pyg::HeteroRelationId::v_imaginary_p) =
        relation(imaginary, static_cast<float>(seed) + 0.4F);
    std::vector<std::int64_t> reverse_imaginary;
    for (std::size_t edge = 0U; edge + 1U < imaginary.size(); edge += 2U) {
        reverse_imaginary.insert(reverse_imaginary.end(),
                                 {imaginary[edge + 1U], imaginary[edge]});
    }
    result.relation(pyg::HeteroRelationId::p_imaginary_v) =
        relation(reverse_imaginary, static_cast<float>(seed) + 0.4F);
    result.validate();
    return result;
}

void require_hetero_equal(
    const pyg::HeteroBatch& actual,
    const pyg::HeteroBatch& expected,
    const std::string& message) {
    require(actual.num_graphs == expected.num_graphs,
            message + " graph count");
    for (std::size_t type = 0U; type < pyg::kHeteroNodeTypes; ++type) {
        require_equal(actual.x[type], expected.x[type], message + " node x");
        require_optional_equal(actual.batch[type], expected.batch[type],
                               message + " node batch");
        require_optional_equal(actual.ptr[type], expected.ptr[type],
                               message + " node ptr");
        require_equal(actual.node_ptr[type], expected.node_ptr[type],
                      message + " node_ptr");
        require(actual.max_num_nodes[type] == expected.max_num_nodes[type],
                message + " max nodes");
    }
    for (std::size_t id = 0U; id < pyg::kHeteroRelations; ++id) {
        require(actual.relations[id].present() == expected.relations[id].present(),
                message + " relation presence");
        if (!actual.relations[id].present()) continue;
        require_equal(actual.relations[id].edge_index,
                      expected.relations[id].edge_index,
                      message + " relation edges");
        require_optional_equal(actual.relations[id].edge_attr,
                               expected.relations[id].edge_attr,
                               message + " relation attrs");
    }
}

void test_homogeneous_and_dual() {
    const auto first = graph(1, 2);
    const auto second = graph(2, 4);
    const auto third = graph(3, 3);
    const auto batch = pyg::Batch::from_data_list({first, second, third});
    rl::PreparedBatchObservationSampler sampler(batch);
    const auto indices = rl::prepare_graph_indices(
        std::vector<std::int64_t>{2, 0, 2, -2}, 3);
    const auto selection = sampler.prepare_selection(indices);
    const auto actual = sampler.sample(selection);
    const auto expected = pyg::Batch::from_data_list(
        {third, first, third, second});
    require_batch_equal(actual, expected, "prepared homogeneous sample");
    require_batch_equal(sampler.sample(
                            torch::tensor({2, 0, 2, -2}, torch::kInt64)),
                        expected,
                        "homogeneous convenience sample");

    const auto v_first = graph(11, 1);
    const auto v_second = graph(12, 3);
    const auto v_third = graph(13, 2);
    rl::DualGraphTensorObservation dual;
    dual.p_net = batch;
    dual.v_net = pyg::Batch::from_data_list({v_first, v_second, v_third});
    dual.general.curr_v_node_id = torch::tensor({5, 6, 7}, torch::kInt64);
    dual.general.action_mask = torch::tensor(
        {{1, 0}, {0, 1}, {1, 1}}, torch::kBool);
    dual.general.v_net_size = torch::tensor({2.0F, 4.0F, 3.0F});
    rl::PreparedDualGraphObservationSampler dual_sampler(dual);
    const auto dual_plan = dual_sampler.prepare_selection(indices);
    const auto sampled = dual_sampler.sample(dual_plan);
    require_batch_equal(sampled.p_net, expected, "dual physical sample");
    require_batch_equal(sampled.v_net, pyg::Batch::from_data_list(
        {v_third, v_first, v_third, v_second}), "dual virtual sample");
    require_equal(*sampled.general.curr_v_node_id,
                  torch::tensor({7, 5, 7, 6}, torch::kInt64), "dual curr ids");
    require_equal(*sampled.general.action_mask,
                  torch::tensor({{1, 1}, {1, 0}, {1, 1}, {0, 1}}, torch::kBool),
                  "dual masks");
    require_equal(*sampled.general.v_net_size,
                  torch::tensor({3.0F, 2.0F, 3.0F, 4.0F}),
                  "dual virtual sizes");

    pyg::Data empty;
    empty.x = torch::empty({0, 2}, torch::kFloat32);
    empty.edge_index = torch::empty({2, 0}, torch::kInt64);
    auto without_attr = graph(21, 2);
    without_attr.edge_attr.reset();
    const auto sparse_batch = pyg::Batch::from_data_list({empty, without_attr});
    rl::PreparedBatchObservationSampler sparse_sampler(sparse_batch);
    const auto sparse_sample = sparse_sampler.sample(rl::prepare_graph_indices(
        std::vector<std::int64_t>{0, 1, 0}, 2));
    require_batch_equal(
        sparse_sample,
        pyg::Batch::from_data_list({empty, without_attr, empty}),
        "zero-node sample without edge attributes");
}

void test_heterogeneous_views() {
    const std::array<std::int64_t, 3> p_counts{2, 4, 3};
    const std::array<std::int64_t, 3> v_counts{1, 3, 2};
    std::vector<pyg::HeteroData> base;
    std::vector<pyg::HeteroData> virtual_augmented;
    std::vector<pyg::HeteroData> physical_augmented;
    for (std::int64_t graph_id = 0; graph_id < 3; ++graph_id) {
        base.push_back(hetero_graph(
            graph_id, p_counts[graph_id], v_counts[graph_id], false, false));
        virtual_augmented.push_back(hetero_graph(
            graph_id, p_counts[graph_id], v_counts[graph_id], true, false));
        physical_augmented.push_back(hetero_graph(
            graph_id, p_counts[graph_id], v_counts[graph_id], false, true));
    }
    const auto base_batch = pyg::HeteroBatch::from_data_list(base);
    const auto virtual_batch = pyg::HeteroBatch::from_data_list(virtual_augmented);
    const auto physical_batch = pyg::HeteroBatch::from_data_list(physical_augmented);
    rl::PreparedHeteroBatchObservationSampler relation_sampler(base_batch);
    const auto relation_indices = rl::prepare_graph_indices(
        std::vector<std::int64_t>{2, 0}, 3);
    const auto relation_plan = relation_sampler.prepare_relation_selection(
        base_batch.relation(pyg::HeteroRelationId::v_connect_v),
        pyg::HeteroRelationId::v_connect_v, relation_indices);
    rl::PreparedHeteroBatchObservationSampler foreign_relation_sampler(
        pyg::HeteroBatch::from_data_list(base));
    bool foreign_relation_plan = false;
    try {
        static_cast<void>(foreign_relation_sampler.sample_relation(
            base_batch.relation(pyg::HeteroRelationId::v_connect_v),
            pyg::HeteroRelationId::v_connect_v, relation_plan));
    } catch (const rl::GraphObservationSampleException& error) {
        foreign_relation_plan = error.code()
            == rl::GraphObservationSampleErrorCode::foreign_selection;
    }
    require(foreign_relation_plan,
            "foreign heterogeneous relation plan rejected");

    rl::HeteroGraphTensorObservation source;
    source.base = base_batch;
    source.virtual_connect_augmented = virtual_batch.relation(
        pyg::HeteroRelationId::v_connect_v);
    source.physical_connect_augmented = physical_batch.relation(
        pyg::HeteroRelationId::p_connect_p);
    source.general.curr_v_node_id = torch::tensor({0, 1, 2}, torch::kInt64);
    source.general.action_mask = torch::ones({3, 4}, torch::kBool);
    source.general.v_net_size = torch::tensor({1.0F, 3.0F, 2.0F}).reshape({3, 1});

    rl::PreparedHeteroGraphObservationSampler sampler(source);
    const auto indices = rl::prepare_graph_indices(
        std::vector<std::int64_t>{2, 0, 2, -2}, 3);
    const auto selection = sampler.prepare_selection(indices);
    const auto actual = sampler.sample(selection);
    const std::vector<pyg::HeteroData> expected_base{
        base[2], base[0], base[2], base[1]};
    const std::vector<pyg::HeteroData> expected_virtual{
        virtual_augmented[2], virtual_augmented[0],
        virtual_augmented[2], virtual_augmented[1]};
    const std::vector<pyg::HeteroData> expected_physical{
        physical_augmented[2], physical_augmented[0],
        physical_augmented[2], physical_augmented[1]};
    require_hetero_equal(actual.view(rl::HeteroObservationViewId::base),
                         pyg::HeteroBatch::from_data_list(expected_base),
                         "heterogeneous base sample");
    require_hetero_equal(
        actual.view(rl::HeteroObservationViewId::virtual_augmented),
        pyg::HeteroBatch::from_data_list(expected_virtual),
        "heterogeneous virtual augmentation");
    require_hetero_equal(
        actual.view(rl::HeteroObservationViewId::physical_augmented),
        pyg::HeteroBatch::from_data_list(expected_physical),
        "heterogeneous physical augmentation");
    require_equal(*actual.general.curr_v_node_id,
                  torch::tensor({2, 0, 2, 1}, torch::kInt64),
                  "heterogeneous general IDs");
    require_equal(*actual.general.action_mask,
                  torch::ones({4, 4}, torch::kBool),
                  "heterogeneous action masks");
    require_equal(*actual.general.v_net_size,
                  torch::tensor({2.0F, 1.0F, 2.0F, 3.0F}).reshape({4, 1}),
                  "heterogeneous virtual sizes");

    rl::HeteroGraphTensorObservation foreign_source;
    foreign_source.base = pyg::HeteroBatch::from_data_list(base);
    foreign_source.virtual_connect_augmented =
        pyg::HeteroBatch::from_data_list(virtual_augmented).relation(
            pyg::HeteroRelationId::v_connect_v);
    foreign_source.physical_connect_augmented =
        pyg::HeteroBatch::from_data_list(physical_augmented).relation(
            pyg::HeteroRelationId::p_connect_p);
    foreign_source.general = source.general;
    rl::PreparedHeteroGraphObservationSampler foreign_sampler(
        std::move(foreign_source));
    bool foreign_selection = false;
    try {
        static_cast<void>(foreign_sampler.sample(selection));
    } catch (const rl::GraphObservationSampleException& error) {
        foreign_selection = error.code()
            == rl::GraphObservationSampleErrorCode::foreign_selection;
    }
    require(foreign_selection,
            "same-count foreign heterogeneous selection rejected");
}

void test_errors_and_cuda() {
    bool invalid = false;
    try {
        static_cast<void>(rl::prepare_graph_indices(
            torch::tensor({0.0F}), 2));
    } catch (const rl::GraphObservationSampleException& error) {
        invalid = error.code()
            == rl::GraphObservationSampleErrorCode::invalid_indices;
    }
    require(invalid, "float graph indices rejected");

    const auto batch = pyg::Batch::from_data_list({graph(1, 2), graph(2, 3)});
    rl::PreparedBatchObservationSampler sampler(batch);
    bool wrong_source = false;
    try {
        const auto indices = rl::prepare_graph_indices(
            std::vector<std::int64_t>{0}, 3);
        static_cast<void>(sampler.sample(indices));
    } catch (const rl::GraphObservationSampleException& error) {
        wrong_source = error.code()
            == rl::GraphObservationSampleErrorCode::graph_count_mismatch;
    }
    require(wrong_source, "foreign prepared indices rejected");

    const auto other_batch = pyg::Batch::from_data_list(
        {graph(10, 3), graph(20, 2)});
    rl::PreparedBatchObservationSampler other_sampler(other_batch);
    const auto foreign_plan = sampler.prepare_selection(
        rl::prepare_graph_indices(std::vector<std::int64_t>{0}, 2));
    bool foreign_selection = false;
    try {
        static_cast<void>(other_sampler.sample(foreign_plan));
    } catch (const rl::GraphObservationSampleException& error) {
        foreign_selection = error.code()
            == rl::GraphObservationSampleErrorCode::foreign_selection;
    }
    require(foreign_selection, "same-count foreign selection rejected");

    pyg::Data isolated_a;
    isolated_a.x = torch::arange(4, torch::kFloat32).reshape({2, 2});
    isolated_a.edge_index = torch::empty({2, 0}, torch::kInt64);
    pyg::Data isolated_b = isolated_a;
    isolated_b.x = torch::arange(4, 8, torch::kFloat32).reshape({2, 2});
    const auto shared_storage_a = pyg::Batch::from_data_list(
        {isolated_a, isolated_b});
    auto shared_storage_b = shared_storage_a;
    shared_storage_b.ptr = torch::tensor({0, 1, 4}, torch::kInt64);
    shared_storage_b.batch = torch::tensor({0, 1, 1, 1}, torch::kInt64);
    shared_storage_b.max_num_nodes = 3;
    rl::PreparedBatchObservationSampler partition_a(shared_storage_a);
    rl::PreparedBatchObservationSampler partition_b(shared_storage_b);
    const auto partition_plan = partition_a.prepare_selection(
        rl::prepare_graph_indices(std::vector<std::int64_t>{0}, 2));
    bool foreign_partition = false;
    try {
        static_cast<void>(partition_b.sample(partition_plan));
    } catch (const rl::GraphObservationSampleException& error) {
        foreign_partition = error.code()
            == rl::GraphObservationSampleErrorCode::foreign_selection;
    }
    require(foreign_partition,
            "shared tensors with a foreign graph partition are rejected");

    bool malformed_ptr = false;
    try {
        auto malformed = batch;
        malformed.ptr = torch::tensor({0, 6, 5}, torch::kInt64);
        static_cast<void>(rl::PreparedBatchObservationSampler(
            std::move(malformed)));
    } catch (const rl::GraphObservationSampleException& error) {
        malformed_ptr = error.code()
            == rl::GraphObservationSampleErrorCode::malformed_batch;
    }
    require(malformed_ptr, "non-monotonic node ptr rejected");

    bool cross_graph_edge = false;
    try {
        auto malformed = batch;
        malformed.edge_index = batch.edge_index.clone();
        malformed.edge_index.index_put_({0, 0}, 4);
        static_cast<void>(rl::PreparedBatchObservationSampler(
            std::move(malformed)));
    } catch (const rl::GraphObservationSampleException& error) {
        cross_graph_edge = error.code()
            == rl::GraphObservationSampleErrorCode::malformed_batch;
    }
    require(cross_graph_edge, "cross-graph homogeneous edge rejected");

    bool mismatched_dual = false;
    try {
        rl::DualGraphTensorObservation source;
        source.p_net = batch;
        source.v_net = pyg::Batch::from_data_list({graph(3, 2)});
        static_cast<void>(rl::PreparedDualGraphObservationSampler(
            std::move(source)));
    } catch (const rl::GraphObservationSampleException& error) {
        mismatched_dual = error.code()
            == rl::GraphObservationSampleErrorCode::graph_count_mismatch;
    }
    require(mismatched_dual, "dual graph count mismatch rejected");

    bool mismatched_general = false;
    try {
        rl::DualGraphTensorObservation source;
        source.p_net = batch;
        source.v_net = batch;
        source.general.curr_v_node_id = torch::tensor({0}, torch::kInt64);
        static_cast<void>(rl::PreparedDualGraphObservationSampler(
            std::move(source)));
    } catch (const rl::GraphObservationSampleException& error) {
        mismatched_general = error.code()
            == rl::GraphObservationSampleErrorCode::graph_count_mismatch;
    }
    require(mismatched_general, "general graph count mismatch rejected");

    if (torch::cuda::is_available()) {
        const torch::Device cuda(torch::kCUDA);
        rl::PreparedBatchObservationSampler cuda_sampler(batch.to(cuda));
        const auto selection = cuda_sampler.prepare_selection(
            rl::prepare_graph_indices(std::vector<std::int64_t>{1, 0}, 2));
        const auto output = cuda_sampler.sample(selection, cuda);
        require(output.x.device() == cuda && output.edge_index.device() == cuda,
                "CUDA prepared sample device");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        test_homogeneous_and_dual();
        test_heterogeneous_views();
        test_errors_and_cuda();
        std::cout << "graph_observation_sampler_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph_observation_sampler_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
