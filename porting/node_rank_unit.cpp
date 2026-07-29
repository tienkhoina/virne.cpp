#include "../virne/solver/rank/node_rank.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace net = virne::network;
namespace attr = virne::network::attribute;
namespace rank = virne::solver::rank;

using rank::NodeRankEntry;
using rank::NodeRankErrorCode;
using rank::NodeRankException;
using rank::NodeRankMethod;
using rank::NodeRankOptions;
using rank::NodeRankParameters;
using rank::NodeRankResourceId;
using rank::NodeRankSelection;
using rank::NodeRankValueKind;
using rank::NodeRanker;
using rank::NodeRanking;

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t double_bits(double value) noexcept
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

double double_from_bits(std::uint64_t bits) noexcept
{
    double value = 0.0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::int64_t wrapped_multiply(std::int64_t value, std::uint64_t factor)
{
    const std::uint64_t product =
        static_cast<std::uint64_t>(value) * factor;
    std::int64_t result = 0;
    static_assert(sizeof(product) == sizeof(result));
    std::memcpy(&result, &product, sizeof(result));
    return result;
}

NodeRankOptions options(
    bool sort,
    std::size_t workers,
    std::optional<std::size_t> max_iterations = std::nullopt)
{
    NodeRankOptions result{};
    result.sort = sort;
    result.workers = workers;
    result.max_iterations = max_iterations;
    return result;
}

attr::AttributeFactorySpec make_spec(
    std::string name,
    attr::AttributeOwner owner,
    attr::AttributeKind kind = attr::AttributeKind::resource)
{
    attr::AttributeFactorySpec spec{};
    spec.name = std::move(name);
    spec.owner = owner;
    spec.kind = kind;
    spec.generative = false;
    if (kind == attr::AttributeKind::resource) {
        spec.restriction = attr::ConstraintRestriction::hard;
        spec.checking_level = owner == attr::AttributeOwner::node
            ? attr::CheckingLevel::node
            : attr::CheckingLevel::link;
    }
    return spec;
}

struct Fixture {
    net::BaseNetwork network;
    std::vector<NodeRankResourceId> node_ids;
    std::vector<NodeRankResourceId> link_ids;
};

Fixture make_fixture(
    std::size_t node_count,
    std::vector<EdgeEndpoints> edges,
    const std::vector<std::vector<AttrValue>>& node_rows,
    const std::vector<std::vector<AttrValue>>& link_rows,
    bool include_non_resource = false)
{
    net::BaseNetworkConstruction construction{};
    construction.incoming_graph.emplace(node_count, std::move(edges));
    for (std::size_t index = 0U; index < node_rows.size(); ++index) {
        construction.config.node_attribute_specs.push_back(make_spec(
            "node_resource_" + std::to_string(index),
            attr::AttributeOwner::node));
    }
    for (std::size_t index = 0U; index < link_rows.size(); ++index) {
        construction.config.link_attribute_specs.push_back(make_spec(
            "link_resource_" + std::to_string(index),
            attr::AttributeOwner::link));
    }
    if (include_non_resource) {
        construction.config.node_attribute_specs.push_back(make_spec(
            "node_status",
            attr::AttributeOwner::node,
            attr::AttributeKind::status));
        construction.config.link_attribute_specs.push_back(make_spec(
            "link_status",
            attr::AttributeOwner::link,
            attr::AttributeKind::status));
    }

    net::BaseNetwork network(std::move(construction));
    std::vector<NodeRankResourceId> node_ids;
    std::vector<net::NodeAttributeDataUpdate> node_updates;
    for (std::size_t index = 0U; index < node_rows.size(); ++index) {
        const auto binding = network.bind_node_attribute(
            "node_resource_" + std::to_string(index));
        require(binding.has_value(), "fixture node resource binding failed");
        node_ids.push_back(binding->registry_id);
        net::NodeAttributeDataUpdate update{};
        update.registry_id = binding->registry_id;
        update.layout = net::AttributeDataLayout::dense;
        update.dense_values = node_rows[index];
        node_updates.push_back(std::move(update));
    }
    if (!node_updates.empty()) {
        network.set_node_attrs_data(node_updates, 1U);
    }

    std::vector<NodeRankResourceId> link_ids;
    std::vector<net::LinkAttributeDataUpdate> link_updates;
    for (std::size_t index = 0U; index < link_rows.size(); ++index) {
        const auto binding = network.bind_link_attribute(
            "link_resource_" + std::to_string(index));
        require(binding.has_value(), "fixture link resource binding failed");
        link_ids.push_back(binding->registry_id);
        net::LinkAttributeDataUpdate update{};
        update.registry_id = binding->registry_id;
        update.layout = net::AttributeDataLayout::dense;
        update.dense_values = link_rows[index];
        link_updates.push_back(std::move(update));
    }
    if (!link_updates.empty()) {
        network.set_link_attrs_data(link_updates, 1U);
    }
    return Fixture{std::move(network), std::move(node_ids), std::move(link_ids)};
}

bool same_ranking_bits(
    const NodeRanking& lhs,
    const NodeRanking& rhs) noexcept
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < lhs.size(); ++index) {
        if (lhs[index].node_id != rhs[index].node_id ||
            lhs[index].kind != rhs[index].kind ||
            double_bits(lhs[index].value) != double_bits(rhs[index].value) ||
            double_bits(lhs[index].distance) !=
                double_bits(rhs[index].distance)) {
            return false;
        }
    }
    return true;
}

void require_scalar(
    const NodeRanking& actual,
    const std::vector<Vertex>& nodes,
    const std::vector<double>& values,
    const std::string& context)
{
    require(
        actual.size() == nodes.size() && actual.size() == values.size(),
        context + ": ranking length mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require(
            actual[index].node_id == nodes[index],
            context + ": node order mismatch");
        require(
            actual[index].kind == NodeRankValueKind::scalar,
            context + ": value kind mismatch");
        require(
            double_bits(actual[index].value) == double_bits(values[index]),
            [&] {
                std::ostringstream message;
                message << context << ": score bits mismatch at " << index
                        << " actual=0x" << std::hex
                        << double_bits(actual[index].value)
                        << " expected=0x" << double_bits(values[index]);
                return message.str();
            }());
    }
}

template <typename Action>
NodeRankException capture_error(const Action& action)
{
    try {
        action();
    } catch (const NodeRankException& error) {
        return error;
    }
    throw std::runtime_error("expected NodeRankException");
}

void test_method_surface()
{
    const std::vector<std::pair<std::string_view, NodeRankMethod>> methods{
        {"order", NodeRankMethod::order},
        {"random", NodeRankMethod::random},
        {"ffd", NodeRankMethod::ffd},
        {"nrm", NodeRankMethod::nrm},
        {"nea", NodeRankMethod::nea},
        {"grc", NodeRankMethod::grc},
        {"rw", NodeRankMethod::rw},
        {"nps", NodeRankMethod::nps},
    };
    for (const auto& item : methods) {
        require(
            rank::node_rank_method_from_string(item.first) == item.second,
            "method parser mismatch");
        require(
            rank::node_rank_method_name(item.second) == item.first,
            "method name mismatch");
    }
    const NodeRankException error = capture_error([] {
        static_cast<void>(rank::node_rank_method_from_string("unknown"));
    });
    require(
        error.code() == NodeRankErrorCode::unsupported_method,
        "unknown method error mismatch");
}

void test_order_and_random()
{
    Fixture fixture = make_fixture(
        6U,
        {{0U, 1U}, {1U, 2U}},
        {},
        {});
    const auto prepared = NodeRanker{}.prepare(fixture.network);
    require_scalar(
        prepared.rank_order(options(false, 8U)),
        {0U, 1U, 2U, 3U, 4U, 5U},
        std::vector<double>(6U, 1.0 / 6.0),
        "order unsorted");
    require_scalar(
        prepared.rank_order(options(true, 1U)),
        {0U, 1U, 2U, 3U, 4U, 5U},
        std::vector<double>(6U, 1.0 / 6.0),
        "order ignores sort");

    NumpyRandomState random(42U);
    require_scalar(
        prepared.rank_random(random, options(false, 8U)),
        {0U, 1U, 2U, 3U, 4U, 5U},
        {0.0, 1.0, 5.0, 2.0, 4.0, 3.0},
        "random first shuffle");
    require_scalar(
        prepared.rank_random(random, options(false, 1U)),
        {0U, 1U, 2U, 3U, 4U, 5U},
        {3.0, 0.0, 1.0, 2.0, 5.0, 4.0},
        "random second shuffle");
    require(
        random.randint(std::int64_t{0}, std::int64_t{2147483648LL}) ==
            std::int64_t{429389014},
        "random continuation mismatch");

    const NodeRankException dispatch_error = capture_error([&] {
        static_cast<void>(prepared.rank(NodeRankMethod::random));
    });
    require(
        dispatch_error.code() == NodeRankErrorCode::random_stream_required,
        "random dispatch must require explicit stream");

    Fixture empty = make_fixture(0U, {}, {}, {});
    const auto empty_prepared = NodeRanker{}.prepare(empty.network);
    NumpyRandomState empty_random(7U);
    require(
        empty_prepared.rank_random(empty_random).empty(),
        "empty random ranking must succeed");
    require(
        capture_error([&] {
            static_cast<void>(empty_prepared.rank_order());
        }).operation() == rank::NodeRankOperation::reduce,
        "empty order must fail at reduction");
}

void test_ffd_and_workers()
{
    Fixture fixture = make_fixture(
        4U,
        {{0U, 1U}, {1U, 2U}, {2U, 3U}},
        {
            {
                AttrValue{std::int64_t{5}},
                AttrValue{std::numeric_limits<std::int64_t>::max()},
                AttrValue{std::int64_t{-3}},
                AttrValue{std::int64_t{7}},
            },
            {
                AttrValue{true},
                AttrValue{std::int64_t{1}},
                AttrValue{false},
                AttrValue{std::int64_t{-2}},
            },
        },
        {});
    const auto prepared = NodeRanker{}.prepare(fixture.network);
    const std::vector<double> expected{
        6.0,
        static_cast<double>(std::numeric_limits<std::int64_t>::min()),
        -3.0,
        5.0,
    };
    const NodeRanking canonical = prepared.rank_ffd(options(false, 1U));
    require_scalar(
        canonical,
        {0U, 1U, 2U, 3U},
        expected,
        "ffd integer lane");
    for (const std::size_t workers : {0U, 2U, 8U}) {
        require(
            same_ranking_bits(
                canonical,
                prepared.rank_ffd(options(false, workers))),
            "ffd worker output drift");
    }
    require_scalar(
        prepared.rank_ffd(options(true, 8U)),
        {0U, 3U, 2U, 1U},
        {6.0, 5.0, -3.0, expected[1]},
        "ffd sorted");

    Fixture mixed = make_fixture(
        3U,
        {{0U, 1U}, {1U, 2U}},
        {
            {
                AttrValue{std::int64_t{2}},
                AttrValue{std::int64_t{-2}},
                AttrValue{std::int64_t{1}},
            },
            {
                AttrValue{0.5},
                AttrValue{-0.0},
                AttrValue{std::numeric_limits<double>::quiet_NaN()},
            },
        },
        {});
    require_scalar(
        NodeRanker{}.prepare(mixed.network).rank_ffd(options(true, 2U)),
        {0U, 1U, 2U},
        {2.5, -2.0, std::numeric_limits<double>::quiet_NaN()},
        "ffd mixed NaN schedule");
}

void test_nrm_and_nea()
{
    Fixture fixture = make_fixture(
        4U,
        {{0U, 1U}, {1U, 1U}, {1U, 2U}},
        {{
            AttrValue{std::int64_t{2}},
            AttrValue{std::int64_t{3}},
            AttrValue{std::int64_t{-4}},
            AttrValue{std::int64_t{5}},
        }},
        {{
            AttrValue{1.0},
            AttrValue{2.0},
            AttrValue{3.0},
        }});
    const auto prepared = NodeRanker{}.prepare(fixture.network);
    require_scalar(
        prepared.rank_nrm(options(false, 8U)),
        {0U, 1U, 2U, 3U},
        {2.0, 18.0, -12.0, 0.0},
        "nrm incident sums");
    require_scalar(
        prepared.rank_nea(options(false, 2U)),
        {0U, 1U, 2U, 3U},
        {
            2.0,
            static_cast<double>(wrapped_multiply(3, 4U)),
            -4.0,
            0.0,
        },
        "nea degree and self-loop");

    Fixture no_links = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{-2}}, AttrValue{std::int64_t{3}}}},
        {});
    const auto no_link_prepared = NodeRanker{}.prepare(no_links.network);
    require_scalar(
        no_link_prepared.rank_nrm(options(false, 1U)),
        {0U, 1U},
        {-0.0, 0.0},
        "nrm empty links signed zero");
}

void test_iterative_anchors()
{
    Fixture fixture = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{3}}}},
        {{AttrValue{5.0}}});
    const auto prepared = NodeRanker{}.prepare(fixture.network);

    const NodeRanking grc = prepared.rank_grc(options(false, 1U));
    require_scalar(
        grc,
        {0U, 1U},
        {
            double_from_bits(UINT64_C(0x3fdf7b1c6a210810)),
            double_from_bits(UINT64_C(0x3fe04271caef7bf5)),
        },
        "grc default anchor");
    for (const std::size_t workers : {2U, 8U}) {
        require(
            same_ranking_bits(
                grc,
                prepared.rank_grc(options(false, workers))),
            "grc worker output drift");
    }

    NodeRankParameters one_step_parameters{};
    one_step_parameters.grc.sigma =
        std::numeric_limits<double>::infinity();
    const auto one_step = NodeRanker({}, one_step_parameters)
                              .prepare(fixture.network)
                              .rank_grc(options(false, 1U));
    require_scalar(
        one_step,
        {0U, 1U},
        {
            double_from_bits(UINT64_C(0x3fe23d70a3d70a3e)),
            double_from_bits(UINT64_C(0x3fdb851eb851eb86)),
        },
        "grc one-step anchor");

    const NodeRanking rw = prepared.rank_rw(options(false, 1U));
    require_scalar(
        rw,
        {0U, 1U},
        {
            double_from_bits(UINT64_C(0x3fdf7bae3d748862)),
            double_from_bits(UINT64_C(0x3fe04228e117d41c)),
        },
        "rw default anchor");
    for (const std::size_t workers : {2U, 8U}) {
        require(
            same_ranking_bits(rw, prepared.rank_rw(options(false, workers))),
            "rw worker output drift");
    }

    NodeRankParameters capped_parameters{};
    capped_parameters.grc.sigma = 0.0;
    const auto capped =
        NodeRanker({}, capped_parameters).prepare(fixture.network);
    require(
        capture_error([&] {
            static_cast<void>(capped.rank_grc(options(false, 1U, 2U)));
        }).code() == NodeRankErrorCode::iteration_limit_reached,
        "GRC caller iteration cap mismatch");
}

void test_nps()
{
    Fixture fixture = make_fixture(
        4U,
        {{0U, 2U}, {0U, 1U}},
        {{
            AttrValue{std::int64_t{1}},
            AttrValue{std::int64_t{5}},
            AttrValue{std::int64_t{2}},
            AttrValue{std::int64_t{9}},
        }},
        {{AttrValue{1.0}, AttrValue{1.0}}});
    const auto prepared = NodeRanker{}.prepare(fixture.network);
    const NodeRanking unsorted = prepared.rank_nps(options(false, 1U));
    require(unsorted.size() == 3U, "NPS must omit disconnected isolate");
    require(
        unsorted[0U].node_id == 0U && unsorted[1U].node_id == 2U &&
            unsorted[2U].node_id == 1U,
        "NPS Dijkstra settlement order mismatch");
    for (const NodeRankEntry& entry : unsorted) {
        require(
            entry.kind == NodeRankValueKind::proximity,
            "NPS result kind mismatch");
    }
    const NodeRanking sorted = prepared.rank_nps(options(true, 8U));
    require(
        sorted[0U].node_id == 0U && sorted[1U].node_id == 1U &&
            sorted[2U].node_id == 2U,
        "NPS composite sort mismatch");
    require(
        double_bits(sorted[0U].distance) == double_bits(0.0) &&
            double_bits(sorted[1U].distance) == double_bits(1.0) &&
            double_bits(sorted[2U].distance) == double_bits(1.0),
        "NPS distance mismatch");
    require(
        double_bits(sorted[0U].value) == double_bits(2.0) &&
            double_bits(sorted[1U].value) == double_bits(5.0) &&
            double_bits(sorted[2U].value) == double_bits(2.0),
        "NPS NRM score mismatch");

    const AttrId weight = fixture.network.graph().attr_id("weight");
    fixture.network.graph()
        .edge_attrs(fixture.network.graph().edge(0U, 2U))
        .set(weight, AttrValue{3.0});
    fixture.network.graph()
        .edge_attrs(fixture.network.graph().edge(0U, 1U))
        .set(weight, AttrValue{1.0});
    const NodeRanking weighted = prepared.rank_nps(options(true, 2U));
    require(
        weighted[0U].node_id == 0U && weighted[1U].node_id == 1U &&
            weighted[2U].node_id == 2U,
        "NPS weighted order mismatch");
    require(
        double_bits(weighted[1U].distance) == double_bits(1.0) &&
            double_bits(weighted[2U].distance) == double_bits(3.0),
        "NPS weighted distance mismatch");
}

void test_boundaries_and_prepared_reuse()
{
    Fixture fixture = make_fixture(
        3U,
        {{0U, 1U}, {1U, 2U}},
        {
            {
                AttrValue{std::int64_t{1}},
                AttrValue{std::int64_t{2}},
                AttrValue{std::int64_t{3}},
            },
            {
                AttrValue{std::int64_t{10}},
                AttrValue{std::int64_t{20}},
                AttrValue{std::int64_t{30}},
            },
        },
        {{AttrValue{1.0}, AttrValue{1.0}}},
        true);
    const auto prepared = NodeRanker{}.prepare(fixture.network);
    require_scalar(
        prepared.rank_ffd(options(false, 1U)),
        {0U, 1U, 2U},
        {11.0, 22.0, 33.0},
        "prepared initial");

    const auto binding = fixture.network.bind_node_attribute("node_resource_0");
    require(binding.has_value(), "prepared mutation binding missing");
    fixture.network.graph().node_attrs(1U).set(
        binding->value_id, AttrValue{std::int64_t{100}});
    require_scalar(
        prepared.rank_ffd(options(false, 8U)),
        {0U, 1U, 2U},
        {11.0, 120.0, 33.0},
        "prepared value mutation");

    NodeRankSelection empty_selection{};
    empty_selection.node_resources = std::vector<NodeRankResourceId>{};
    empty_selection.link_resources = fixture.link_ids;
    const auto empty_nodes =
        NodeRanker(empty_selection).prepare(fixture.network);
    require(
        capture_error([&] {
            static_cast<void>(empty_nodes.rank_ffd());
        }).code() == NodeRankErrorCode::empty_node_resource_selection,
        "empty node selection error mismatch");

    const auto node_status = fixture.network.bind_node_attribute("node_status");
    require(node_status.has_value(), "status fixture binding missing");
    NodeRankSelection invalid_selection{};
    invalid_selection.node_resources =
        std::vector<NodeRankResourceId>{node_status->registry_id};
    invalid_selection.link_resources = fixture.link_ids;
    require(
        capture_error([&] {
            static_cast<void>(
                NodeRanker(invalid_selection).prepare(fixture.network));
        }).code() == NodeRankErrorCode::invalid_node_resource_selection,
        "invalid node resource selection error mismatch");

    Fixture ragged = make_fixture(
        3U,
        {{0U, 1U}, {1U, 2U}},
        {
            {
                AttrValue{std::int64_t{1}},
                AttrValue{std::int64_t{2}},
                AttrValue{std::int64_t{3}},
            },
            {
                AttrValue{std::int64_t{10}},
                AttrValue{std::int64_t{20}},
                AttrValue{std::int64_t{30}},
            },
        },
        {});
    const auto ragged_binding =
        ragged.network.bind_node_attribute("node_resource_1");
    require(ragged_binding.has_value(), "ragged binding missing");
    require(
        ragged.network.graph().node_attrs(1U).erase(ragged_binding->value_id),
        "ragged erase failed");
    require(
        capture_error([&] {
            static_cast<void>(
                NodeRanker{}.prepare(ragged.network).rank_ffd());
        }).code() == NodeRankErrorCode::ragged_node_resource_matrix,
        "ragged resource matrix error mismatch");

    Fixture nonnumeric = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{1}}, AttrValue{std::string{"bad"}}}},
        {});
    require(
        capture_error([&] {
            static_cast<void>(
                NodeRanker{}.prepare(nonnumeric.network).rank_ffd());
        }).code() == NodeRankErrorCode::non_numeric_node_resource_value,
        "nonnumeric resource error mismatch");

    const NodeRanking expected = prepared.rank_ffd(options(true, 1U));
    std::atomic<bool> failed{false};
    std::vector<std::thread> callers;
    for (std::size_t index = 0U; index < 8U; ++index) {
        callers.emplace_back([&, index] {
            try {
                if (!same_ranking_bits(
                        expected,
                        prepared.rank_ffd(options(true, index % 3U + 1U)))) {
                    failed.store(true, std::memory_order_relaxed);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }
    require(
        !failed.load(std::memory_order_relaxed),
        "concurrent read-only prepared calls drifted");
}

void test_additional_boundaries()
{
    Fixture equal_short = make_fixture(
        3U,
        {{0U, 1U}, {1U, 2U}},
        {
            {
                AttrValue{std::int64_t{1}},
                AttrValue{std::int64_t{2}},
                AttrValue{std::int64_t{3}},
            },
            {
                AttrValue{std::int64_t{10}},
                AttrValue{std::int64_t{20}},
                AttrValue{std::int64_t{30}},
            },
        },
        {});
    const auto short_0 =
        equal_short.network.bind_node_attribute("node_resource_0");
    const auto short_1 =
        equal_short.network.bind_node_attribute("node_resource_1");
    require(
        short_0.has_value() && short_1.has_value(),
        "equal-short bindings missing");
    require(
        equal_short.network.graph().node_attrs(0U).erase(short_0->value_id),
        "equal-short first erase failed");
    require(
        equal_short.network.graph().node_attrs(2U).erase(short_1->value_id),
        "equal-short second erase failed");
    require(
        capture_error([&] {
            static_cast<void>(
                NodeRanker{}.prepare(equal_short.network).rank_ffd());
        }).code() == NodeRankErrorCode::ranking_length_mismatch,
        "equal-short rows must fail after numeric reduction");

    Fixture no_links = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{3}}}},
        {});
    const auto no_link_prepared = NodeRanker{}.prepare(no_links.network);
    require(
        capture_error([&] {
            static_cast<void>(no_link_prepared.rank_grc());
        }).code() == NodeRankErrorCode::empty_link_resource_selection,
        "GRC empty-link error mismatch");
    require(
        capture_error([&] {
            static_cast<void>(no_link_prepared.rank_rw());
        }).code() == NodeRankErrorCode::empty_link_resource_selection,
        "RW empty-link error mismatch");

    Fixture zero = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{0}}, AttrValue{std::int64_t{0}}}},
        {{AttrValue{5.0}}});
    const auto zero_prepared = NodeRanker{}.prepare(zero.network);
    require_scalar(
        zero_prepared.rank_rw(options(false, 8U)),
        {0U, 1U},
        {0.0, 0.0},
        "RW zero capacity");
    const AttrId zero_weight = zero.network.graph().attr_id("weight");
    zero.network.graph()
        .edge_attrs(zero.network.graph().edge(0U, 1U))
        .set(zero_weight, AttrValue{0.0});
    require(
        capture_error([&] {
            static_cast<void>(zero_prepared.rank_rw());
        }).code() == NodeRankErrorCode::sparse_assignment_mismatch,
        "RW explicit-zero sparse mismatch error");

    Fixture empty = make_fixture(0U, {}, {{}}, {});
    require(
        capture_error([&] {
            static_cast<void>(
                NodeRanker{}.prepare(empty.network).rank_nps());
        }).operation() == rank::NodeRankOperation::traverse,
        "empty NPS must fail during root selection");

    Fixture stale = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{1}}, AttrValue{std::int64_t{2}}}},
        {});
    static_cast<void>(stale.network.num_nodes());
    stale.network.graph().add_node();
    const auto stale_prepared = NodeRanker{}.prepare(stale.network);
    require(
        capture_error([&] {
            static_cast<void>(stale_prepared.rank_ffd(options(false, 1U)));
        }).code() == NodeRankErrorCode::stale_cardinality,
        "stale cardinality boundary mismatch");

    Fixture singleton = make_fixture(1U, {}, {}, {});
    const auto singleton_prepared = NodeRanker{}.prepare(singleton.network);
    NumpyRandomState singleton_random(123U);
    NumpyRandomState singleton_control(123U);
    require(
        singleton_prepared.rank_random(
            singleton_random, options(false, 8U)).size() == 1U,
        "singleton random ranking length mismatch");
    require(
        singleton_random.next_uint32() == singleton_control.next_uint32(),
        "singleton shuffle consumed RNG state");

    Fixture iterative = make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{3}}}},
        {{AttrValue{5.0}}});
    NodeRankParameters nan_parameters{};
    nan_parameters.grc.sigma =
        std::numeric_limits<double>::quiet_NaN();
    nan_parameters.rw.sigma =
        std::numeric_limits<double>::quiet_NaN();
    const auto nan_prepared =
        NodeRanker({}, nan_parameters).prepare(iterative.network);
    require_scalar(
        nan_prepared.rank_grc(options(false, 2U)),
        {0U, 1U},
        {0.4, 0.6},
        "GRC NaN sigma initial state");
    const NodeRanking rw_initial = nan_prepared.rank_rw(options(false, 2U));
    require(
        rw_initial.size() == 2U &&
            double_bits(rw_initial[0U].value) ==
                UINT64_C(0x3fd99999999533b3) &&
            double_bits(rw_initial[1U].value) ==
                UINT64_C(0x3fe33333332fe6c6),
        "RW NaN sigma initial state mismatch");
}

}  // namespace

int main()
{
    try {
        test_method_surface();
        test_order_and_random();
        test_ffd_and_workers();
        test_nrm_and_nea();
        test_iterative_anchors();
        test_nps();
        test_boundaries_and_prepared_reuse();
        test_additional_boundaries();
        std::cout << "node rank unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "node rank unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
