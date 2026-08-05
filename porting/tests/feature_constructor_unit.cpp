#include "../feature_constructor_fixture.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace fixture = feature_constructor_fixture;
namespace feature = virne::solver::learning::rl_core;
namespace attribute = virne::network::attribute;

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function>
void require_error(
    feature::FeatureConstructorErrorCode expected,
    Function&& function,
    std::string_view message) {
    try {
        function();
    } catch (const feature::FeatureConstructorException& error) {
        require(error.code() == expected, message);
        return;
    }
    throw std::runtime_error(std::string(message));
}

void test_registry_boundary() {
    require(
        feature::feature_constructor_kind_from_string("p_net") ==
            feature::FeatureConstructorKind::p_net,
        "p_net registry boundary mismatch");
    require(
        feature::feature_constructor_kind_from_string("p_net_v_node") ==
            feature::FeatureConstructorKind::p_net_v_node,
        "p_net_v_node registry boundary mismatch");
    require(
        feature::feature_constructor_kind_from_string("p_net_v_net") ==
            feature::FeatureConstructorKind::p_net_v_net,
        "p_net_v_net registry boundary mismatch");
    require_error(
        feature::FeatureConstructorErrorCode::invalid_kind,
        [] { (void)feature::feature_constructor_kind_from_string("unknown"); },
        "invalid feature kind did not fail deterministically");
}

void test_observation_shapes_and_values() {
    const auto p_net = fixture::physical_network();
    const auto v_net = fixture::virtual_network();
    const auto benchmarks = fixture::benchmarks(p_net);
    const auto solution = fixture::solution_with_slots(v_net);
    const auto kinds = feature::ObservationHandler::AttributeKinds{
        attribute::AttributeKind::resource};

    const auto node = feature::ObservationHandler::get_node_attrs_obs(
        p_net, kinds, &*benchmarks.node_attr_benchmarks);
    require(node.rows == 6U && node.columns == 3U,
        "physical node feature shape mismatch");
    require(node.values[0] == 1.0F, "cpu normalization mismatch");
    require(node.values[1] == 0.5F, "gpu normalization mismatch");
    require(node.values[2] == 1.0F, "ram normalization mismatch");

    const auto links = feature::ObservationHandler::get_link_attrs_obs(
        p_net, kinds, &*benchmarks.link_attr_benchmarks);
    require(links.rows == 16U && links.columns == 2U,
        "bidirectional link feature shape mismatch");
    for (std::size_t row = 0U; row < 8U; ++row) {
        require(
            links.values[row * 2U] == links.values[(row + 8U) * 2U] &&
            links.values[row * 2U + 1U] ==
                links.values[(row + 8U) * 2U + 1U],
            "reverse link attributes are not bit-stable duplicates");
    }

    const auto p_status = feature::ObservationHandler::get_p_net_nodes_status(
        p_net, v_net, solution.node_slots, 2U);
    require(p_status.rows == 6U && p_status.columns == 2U,
        "physical status shape mismatch");
    require(p_status.values[2U * 2U] == 1.0F,
        "selected physical node flag mismatch");
    require(p_status.values[4U * 2U + 1U] == 1.0F,
        "placed-neighbor physical flag mismatch");

    const auto v_status = feature::ObservationHandler::get_v_net_nodes_status(
        v_net, solution.node_slots, 2U, true, true);
    require(v_status.rows == 4U && v_status.columns == 3U,
        "virtual status shape mismatch");
    require(v_status.values[2U * 3U + 1U] == 1.0F,
        "current virtual node decision flag mismatch");
    require(v_status.values[0U * 3U + 2U] == 1.0F &&
            v_status.values[1U * 3U + 2U] == 1.0F,
        "placed virtual-neighbor flags mismatch");

    const auto distance = feature::ObservationHandler::get_average_distance(
        p_net, solution.node_slots, true);
    require(
        distance.values == std::vector<float>{
            1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
        "average-distance normalization changed after its minimum row");
}

void test_constructor_variants() {
    const auto p_net = fixture::physical_network();
    const auto v_net = fixture::virtual_network();
    const auto populated = fixture::solution_with_slots(v_net, true);
    const auto empty = fixture::solution_with_slots(v_net, false);

    feature::FeatureConstructor p_only(
        p_net,
        v_net,
        fixture::options(feature::FeatureConstructorKind::p_net),
        fixture::prepared(p_net, v_net));
    const auto p_observation = p_only.construct(populated, 2U);
    require(p_observation.p_net.x.rows == 6U &&
            p_observation.p_net.x.columns == 18U,
        "all-flags physical feature shape mismatch");
    require(!p_observation.v_node_x.has_value() &&
            !p_observation.v_net.has_value(),
        "p_net variant exposed an unrelated feature field");

    feature::FeatureConstructor p_v_node(
        p_net,
        v_net,
        fixture::options(feature::FeatureConstructorKind::p_net_v_node),
        fixture::prepared(p_net, v_net));
    const auto node_observation = p_v_node.construct(populated, 2U);
    require(node_observation.v_node_x.has_value() &&
            node_observation.v_node_x->values.size() == 15U,
        "virtual-node feature shape mismatch");
    require(p_v_node.construct_v_node_features(
                populated, v_net.num_nodes()).values.empty(),
        "terminal virtual-node feature must be empty");

    feature::FeatureConstructor p_v_net(
        p_net,
        v_net,
        fixture::options(feature::FeatureConstructorKind::p_net_v_net),
        fixture::prepared(p_net, v_net));
    const auto network_observation = p_v_net.construct(populated, 2U);
    require(network_observation.v_net.has_value(),
        "virtual-network variant omitted v_net");
    require(network_observation.v_net->x.rows == 4U &&
            network_observation.v_net->x.columns == 19U,
        "virtual-network feature shape mismatch");

    feature::FeatureConstructor minimal(
        p_net,
        v_net,
        fixture::options(
            feature::FeatureConstructorKind::p_net_v_node,
            false, false, false, false),
        fixture::prepared(p_net, v_net));
    const auto minimal_observation = minimal.construct(empty, 0U);
    require(minimal_observation.p_net.x.columns == 4U,
        "minimal physical feature shape mismatch");
    require(minimal_observation.v_node_x->values.size() == 4U,
        "minimal virtual-node feature shape mismatch");
}

void test_worker_determinism_and_errors() {
    const auto p_net = fixture::physical_network();
    const auto v_net = fixture::virtual_network();
    const auto solution = fixture::solution_with_slots(v_net);

    feature::FeatureConstructor sequential(
        p_net,
        v_net,
        fixture::options(
            feature::FeatureConstructorKind::p_net_v_node,
            true, true, true, true, 1U),
        fixture::prepared(p_net, v_net, 1U));
    feature::FeatureConstructor parallel(
        p_net,
        v_net,
        fixture::options(
            feature::FeatureConstructorKind::p_net_v_node,
            true, true, true, true, 4U),
        fixture::prepared(p_net, v_net, 4U));
    const auto one = sequential.construct(solution, 2U);
    const auto four = parallel.construct(solution, 2U);
    require(one.p_net.x.values == four.p_net.x.values &&
            one.p_net.edge_attr.values == four.p_net.edge_attr.values &&
            one.v_node_x->values == four.v_node_x->values,
        "worker count changed feature output");

    require_error(
        feature::FeatureConstructorErrorCode::invalid_worker_count,
        [&] {
            feature::FeatureConstructor invalid(
                p_net,
                v_net,
                fixture::options(
                    feature::FeatureConstructorKind::p_net_v_node,
                    true, true, true, true, 0U),
                fixture::prepared(p_net, v_net));
        },
        "zero worker count did not fail deterministically");

    require_error(
        feature::FeatureConstructorErrorCode::invalid_current_virtual_node,
        [&] {
            feature::FeatureConstructor network_constructor(
                p_net,
                v_net,
                fixture::options(feature::FeatureConstructorKind::p_net_v_net),
                fixture::prepared(p_net, v_net));
            (void)network_constructor.construct(solution, v_net.num_nodes());
        },
        "out-of-range virtual-network current node did not fail");
}

}  // namespace

int main() {
    try {
        test_registry_boundary();
        test_observation_shapes_and_values();
        test_constructor_variants();
        test_worker_determinism_and_errors();
        std::cout << "feature_constructor_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "feature_constructor_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
