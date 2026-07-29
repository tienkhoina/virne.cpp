#include "../virne/solver/rank/link_rank.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace net = virne::network;
namespace attr = virne::network::attribute;
namespace rank = virne::solver::rank;

using rank::LinkRankEdgeId;
using rank::LinkRankEntry;
using rank::LinkRankErrorCode;
using rank::LinkRankException;
using rank::LinkRankMethod;
using rank::LinkRankOperation;
using rank::LinkRankOptions;
using rank::LinkRankResourceId;
using rank::LinkRankSelection;
using rank::LinkRanker;
using rank::LinkRanking;

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

double wrapped_int64_sum(std::initializer_list<std::int64_t> values)
{
    std::uint64_t sum = 0U;
    for (const std::int64_t value : values) {
        sum += static_cast<std::uint64_t>(value);
    }

    std::int64_t signed_sum = 0;
    static_assert(sizeof(sum) == sizeof(signed_sum));
    std::memcpy(&signed_sum, &sum, sizeof(sum));
    return static_cast<double>(signed_sum);
}

LinkRankOptions options(bool sort, std::size_t workers)
{
    LinkRankOptions result{};
    result.sort = sort;
    result.workers = workers;
    return result;
}

attr::AttributeFactorySpec make_link_spec(
    std::string name,
    attr::AttributeKind kind = attr::AttributeKind::resource)
{
    attr::AttributeFactorySpec spec{};
    spec.name = std::move(name);
    spec.owner = attr::AttributeOwner::link;
    spec.kind = kind;
    spec.generative = false;
    if (kind == attr::AttributeKind::resource) {
        spec.restriction = attr::ConstraintRestriction::hard;
        spec.checking_level = attr::CheckingLevel::link;
    }
    return spec;
}

Graph make_path_graph(std::size_t edge_count)
{
    std::vector<EdgeEndpoints> edges;
    edges.reserve(edge_count);
    for (std::size_t index = 0U; index < edge_count; ++index) {
        edges.emplace_back(
            static_cast<Vertex>(index),
            static_cast<Vertex>(index + 1U));
    }
    return Graph(edge_count + 1U, edges);
}

Graph make_hole_graph()
{
    Graph graph(
        std::size_t{5},
        std::vector<EdgeEndpoints>{
            {3U, 1U},
            {0U, 4U},
            {3U, 4U},
            {1U, 2U},
        });

    require(
        graph.edge_id(graph.edge(0U, 4U)) == 1U,
        "fixture removed edge must initially have stable ID 1");
    require(
        graph.remove_edge(0U, 4U),
        "fixture edge removal failed");
    const Edge appended = graph.add_edge(0U, 2U);
    require(
        graph.edge_id(appended) == 4U,
        "fixture appended edge must have stable ID 4");
    require(graph.num_edges() == 4U, "fixture must have four live edges");
    require(
        graph.edge_id_capacity() == 5U,
        "fixture must retain the stable edge-ID hole");
    return graph;
}

net::BaseNetwork make_network(
    Graph graph,
    const std::vector<std::string>& resource_names,
    const std::vector<std::vector<AttrValue>>& dense_rows = {},
    bool include_non_resource = false,
    std::size_t workers = 1U)
{
    require(
        dense_rows.size() <= resource_names.size(),
        "fixture has more dense rows than resource definitions");

    net::BaseNetworkConstruction construction{};
    construction.incoming_graph.emplace(std::move(graph));
    construction.config.factory_workers = workers;
    construction.config.link_attribute_specs.reserve(
        resource_names.size() + (include_non_resource ? 1U : 0U));
    for (const std::string& name : resource_names) {
        construction.config.link_attribute_specs.push_back(
            make_link_spec(name));
    }
    if (include_non_resource) {
        construction.config.link_attribute_specs.push_back(
            make_link_spec("status_only", attr::AttributeKind::status));
    }

    net::BaseNetwork network(std::move(construction));
    std::vector<net::LinkAttributeDataUpdate> updates;
    updates.reserve(dense_rows.size());
    for (std::size_t index = 0U; index < dense_rows.size(); ++index) {
        const auto binding =
            network.bind_link_attribute(resource_names[index]);
        require(binding.has_value(), "fixture resource binding failed");

        net::LinkAttributeDataUpdate update{};
        update.registry_id = binding->registry_id;
        update.layout = net::AttributeDataLayout::dense;
        update.dense_values = dense_rows[index];
        updates.push_back(std::move(update));
    }
    if (!updates.empty()) {
        network.set_link_attrs_data(updates, workers);
    }
    return network;
}

net::LinkNetworkAttributeBinding binding_of(
    const net::BaseNetwork& network,
    std::string_view name)
{
    const auto binding = network.bind_link_attribute(name);
    require(binding.has_value(), "required link attribute is not bound");
    return *binding;
}

LinkRankSelection select_resources(std::vector<LinkRankResourceId> ids)
{
    LinkRankSelection selection{};
    selection.resources = std::move(ids);
    return selection;
}

bool same_ranking_bits(
    const LinkRanking& lhs,
    const LinkRanking& rhs) noexcept
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < lhs.size(); ++index) {
        if (lhs[index].edge_id != rhs[index].edge_id ||
            lhs[index].source != rhs[index].source ||
            lhs[index].target != rhs[index].target ||
            double_bits(lhs[index].value) != double_bits(rhs[index].value)) {
            return false;
        }
    }
    return true;
}

void require_entry_identity(
    const net::BaseNetwork& network,
    const LinkRanking& ranking)
{
    std::vector<bool> seen(network.graph().edge_id_capacity(), false);
    for (const LinkRankEntry& entry : ranking) {
        require(
            entry.edge_id < seen.size(),
            "ranking returned an edge ID outside edge_id_capacity");
        require(!seen[entry.edge_id], "ranking returned a duplicate edge ID");
        seen[entry.edge_id] = true;
        require(
            network.graph().has_edge(entry.source, entry.target),
            "ranking returned endpoints that are not live");
        require(
            network.graph().edge_id(
                network.graph().edge(entry.source, entry.target)) ==
                entry.edge_id,
            "ranking endpoint and stable edge ID disagree");
    }
}

void require_ids_and_values(
    const net::BaseNetwork& network,
    const LinkRanking& actual,
    const std::vector<LinkRankEdgeId>& expected_ids,
    const std::vector<double>& expected_values,
    const std::string& context)
{
    require(
        actual.size() == expected_ids.size() &&
            actual.size() == expected_values.size(),
        context + ": ranking length mismatch");
    require_entry_identity(network, actual);
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require(
            actual[index].edge_id == expected_ids[index],
            context + ": stable edge-ID mismatch");
        require(
            double_bits(actual[index].value) ==
                double_bits(expected_values[index]),
            context + ": raw score bits mismatch");
    }
}

struct ErrorSnapshot {
    LinkRankErrorCode code;
    LinkRankOperation operation;
    std::size_t input_index;
    std::optional<LinkRankResourceId> resource_id;
    std::optional<LinkRankEdgeId> edge_id;
    std::string message;
};

template <typename Function>
ErrorSnapshot capture_rank_error(Function&& function)
{
    try {
        function();
    } catch (const LinkRankException& error) {
        return {
            error.code(),
            error.operation(),
            error.input_index(),
            error.resource_id(),
            error.edge_id(),
            error.what(),
        };
    }
    throw std::runtime_error("expected LinkRankException was not thrown");
}

void test_method_strings_and_errors()
{
    require(
        rank::link_rank_method_from_string("order") == LinkRankMethod::order,
        "order method string did not resolve");
    require(
        rank::link_rank_method_from_string("ffd") == LinkRankMethod::ffd,
        "ffd method string did not resolve");
    require(
        rank::link_rank_method_name(LinkRankMethod::order) == "order",
        "order method name mismatch");
    require(
        rank::link_rank_method_name(LinkRankMethod::ffd) == "ffd",
        "ffd method name mismatch");

    const ErrorSnapshot string_error = capture_rank_error([] {
        static_cast<void>(rank::link_rank_method_from_string("FFD"));
    });
    require(
        string_error.code == LinkRankErrorCode::unsupported_method,
        "unsupported method string error code mismatch");
    require(
        string_error.operation == LinkRankOperation::resolve_method,
        "unsupported method string operation mismatch");
    require(
        string_error.input_index == rank::invalid_link_rank_input_index &&
            !string_error.resource_id.has_value() &&
            !string_error.edge_id.has_value() &&
            !string_error.message.empty(),
        "unsupported method string metadata mismatch");

    auto network = make_network(make_path_graph(1U), {});
    const auto prepared = LinkRanker{}.prepare(network);
    const ErrorSnapshot enum_error = capture_rank_error([&] {
        static_cast<void>(prepared.rank(
            static_cast<LinkRankMethod>(0xffU),
            options(false, 1U)));
    });
    require(
        enum_error.code == LinkRankErrorCode::unsupported_method,
        "unsupported method enum error code mismatch");
}

void test_order_sort_and_edge_id_holes()
{
    auto network = make_network(make_hole_graph(), {});
    const auto prepared = LinkRanker{}.prepare(network);

    const LinkRanking unsorted = prepared.rank_order(options(false, 8U));
    require_ids_and_values(
        network,
        unsorted,
        {4U, 0U, 3U, 2U},
        {0.0, 1.0, 2.0, 3.0},
        "order unsorted");

    const std::array<EdgeEndpoints, 4U> expected_endpoints{{
        {0U, 2U},
        {1U, 3U},
        {1U, 2U},
        {3U, 4U},
    }};
    for (std::size_t index = 0U; index < unsorted.size(); ++index) {
        require(
            EdgeEndpoints{unsorted[index].source, unsorted[index].target} ==
                expected_endpoints[index],
            "order unsorted endpoint order mismatch");
    }

    const LinkRanking sorted = prepared.rank_order(options(true, 0U));
    require_ids_and_values(
        network,
        sorted,
        {2U, 3U, 0U, 4U},
        {3.0, 2.0, 1.0, 0.0},
        "order sorted");
    require(
        same_ranking_bits(
            unsorted,
            prepared.rank(LinkRankMethod::order, options(false, 2U))),
        "generic order dispatch changed the ranking");
    require(
        same_ranking_bits(sorted, prepared.rank_order()),
        "default order options must sort descending");
}

void test_ffd_default_explicit_and_duplicate_resources()
{
    auto network = make_network(
        make_hole_graph(),
        {"r0", "r1"},
        {
            {
                AttrValue{std::int64_t{1}},
                AttrValue{std::int64_t{2}},
                AttrValue{std::int64_t{3}},
                AttrValue{std::int64_t{4}},
            },
            {
                AttrValue{std::int64_t{10}},
                AttrValue{std::int64_t{20}},
                AttrValue{std::int64_t{30}},
                AttrValue{std::int64_t{40}},
            },
        },
        true);

    const auto r0 = binding_of(network, "r0");
    const auto r1 = binding_of(network, "r1");
    const auto default_prepared = LinkRanker{}.prepare(network);
    require(
        default_prepared.resource_ids() ==
            std::vector<LinkRankResourceId>{r0.registry_id, r1.registry_id},
        "default FFD selection must contain only resources in registry order");

    const LinkRanking default_unsorted =
        default_prepared.rank_ffd(options(false, 1U));
    require_ids_and_values(
        network,
        default_unsorted,
        {4U, 0U, 3U, 2U},
        {11.0, 22.0, 33.0, 44.0},
        "default FFD");
    require_ids_and_values(
        network,
        default_prepared.rank_ffd(),
        {2U, 3U, 0U, 4U},
        {44.0, 33.0, 22.0, 11.0},
        "default sorted FFD");
    require(
        same_ranking_bits(
            default_prepared.rank(LinkRankMethod::ffd),
            default_prepared.rank_ffd()),
        "generic FFD dispatch changed the default ranking");

    const auto explicit_prepared =
        LinkRanker(select_resources({r1.registry_id})).prepare(network);
    require(
        explicit_prepared.resource_ids() ==
            std::vector<LinkRankResourceId>{r1.registry_id},
        "explicit resource selection mismatch");
    require_ids_and_values(
        network,
        explicit_prepared.rank_ffd(options(false, 2U)),
        {4U, 0U, 3U, 2U},
        {10.0, 20.0, 30.0, 40.0},
        "explicit FFD");

    const auto duplicate_prepared = LinkRanker(select_resources(
        {r0.registry_id, r0.registry_id, r1.registry_id}))
                                        .prepare(network);
    require(
        duplicate_prepared.resource_ids() ==
            std::vector<LinkRankResourceId>{
                r0.registry_id, r0.registry_id, r1.registry_id},
        "duplicate resource selection was not retained");
    require_ids_and_values(
        network,
        duplicate_prepared.rank_ffd(options(false, 8U)),
        {4U, 0U, 3U, 2U},
        {12.0, 24.0, 36.0, 48.0},
        "duplicate-resource FFD");
}

void test_integer_modular_lane_and_bool()
{
    const std::int64_t maximum =
        std::numeric_limits<std::int64_t>::max();
    const std::int64_t minimum =
        std::numeric_limits<std::int64_t>::min();

    auto modular_network = make_network(
        make_path_graph(3U),
        {"a", "b"},
        {
            {
                AttrValue{maximum},
                AttrValue{minimum},
                AttrValue{maximum},
            },
            {
                AttrValue{std::int64_t{1}},
                AttrValue{std::int64_t{-1}},
                AttrValue{maximum},
            },
        });
    const auto modular_prepared = LinkRanker{}.prepare(modular_network);
    require_ids_and_values(
        modular_network,
        modular_prepared.rank_ffd(options(false, 2U)),
        {0U, 1U, 2U},
        {
            wrapped_int64_sum({maximum, 1}),
            wrapped_int64_sum({minimum, -1}),
            wrapped_int64_sum({maximum, maximum}),
        },
        "modulo-2^64 integer FFD");

    auto bool_network = make_network(
        make_path_graph(3U),
        {"flag", "integer"},
        {
            {AttrValue{true}, AttrValue{false}, AttrValue{true}},
            {
                AttrValue{std::int64_t{5}},
                AttrValue{std::int64_t{-2}},
                AttrValue{std::int64_t{-1}},
            },
        });
    const auto bool_prepared = LinkRanker{}.prepare(bool_network);
    require_ids_and_values(
        bool_network,
        bool_prepared.rank_ffd(options(false, 8U)),
        {0U, 1U, 2U},
        {6.0, -2.0, 0.0},
        "bool plus integer FFD");
}

void test_mixed_double_special_values_and_workers()
{
    const double nan_payload = double_from_bits(0x7ff8000000000042ULL);
    auto network = make_network(
        make_path_graph(8U),
        {"floating", "integer"},
        {
            {
                AttrValue{0.0},
                AttrValue{-0.0},
                AttrValue{std::numeric_limits<double>::infinity()},
                AttrValue{-std::numeric_limits<double>::infinity()},
                AttrValue{nan_payload},
                AttrValue{-5.0},
                AttrValue{3.0},
                AttrValue{3.0},
            },
            {
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
                AttrValue{std::int64_t{0}},
            },
        });

    const auto floating = binding_of(network, "floating");
    const AttrValue* negative_zero = network.graph()
                                         .edge_attrs(network.graph().edge(1U, 2U))
                                         .find(floating.value_id);
    require(
        negative_zero != nullptr &&
            std::holds_alternative<double>(*negative_zero) &&
            double_bits(std::get<double>(*negative_zero)) ==
                double_bits(-0.0),
        "signed-zero fixture was not retained by dense storage");

    const auto prepared = LinkRanker{}.prepare(network);
    const LinkRanking raw_reference =
        prepared.rank_ffd(options(false, 0U));
    require_entry_identity(network, raw_reference);
    require(raw_reference.size() == 8U, "mixed FFD length mismatch");
    require(
        raw_reference[0U].value == 0.0 &&
            raw_reference[1U].value == 0.0,
        "signed-zero inputs must remain equal-score ties");
    require(
        std::isinf(raw_reference[2U].value) &&
            raw_reference[2U].value > 0.0,
        "positive infinity score mismatch");
    require(
        std::isinf(raw_reference[3U].value) &&
            raw_reference[3U].value < 0.0,
        "negative infinity score mismatch");
    require(std::isnan(raw_reference[4U].value), "NaN score was lost");
    require(
        raw_reference[5U].value == -5.0 &&
            raw_reference[6U].value == 3.0 &&
            raw_reference[7U].value == 3.0,
        "finite mixed-double scores mismatch");

    const LinkRanking sorted_reference =
        prepared.rank_ffd(options(true, 0U));
    require_entry_identity(network, sorted_reference);
    std::vector<LinkRankEdgeId> non_nan_ids;
    std::size_t nan_count = 0U;
    for (const LinkRankEntry& entry : sorted_reference) {
        if (std::isnan(entry.value)) {
            ++nan_count;
        } else {
            non_nan_ids.push_back(entry.edge_id);
        }
    }
    require(nan_count == 1U, "sorted FFD must retain exactly one NaN");
    require(
        non_nan_ids ==
            std::vector<LinkRankEdgeId>{2U, 6U, 7U, 0U, 1U, 5U, 3U},
        "NaN-safe sort changed descending order or stable ties");

    for (const std::size_t workers :
         std::array<std::size_t, 4U>{0U, 1U, 2U, 8U}) {
        require(
            same_ranking_bits(
                raw_reference,
                prepared.rank_ffd(options(false, workers))),
            "worker width changed unsorted raw score bits");
        require(
            same_ranking_bits(
                sorted_reference,
                prepared.rank_ffd(options(true, workers))),
            "worker width changed sorted order or raw score bits");
    }
}

void test_zero_edges_and_empty_resources()
{
    auto zero_edge_network = make_network(
        Graph(std::size_t{3}, std::vector<EdgeEndpoints>{}),
        {"resource"},
        {std::vector<AttrValue>{}});
    const auto zero_prepared = LinkRanker{}.prepare(zero_edge_network);
    require(
        zero_prepared.rank_order().empty(),
        "order on a zero-edge network must be empty");
    require(
        zero_prepared.rank_ffd(options(false, 8U)).empty(),
        "FFD on zero edges with one resource must be empty");

    auto no_resource_network = make_network(make_path_graph(2U), {});
    const auto no_resource_prepared =
        LinkRanker{}.prepare(no_resource_network);
    require(
        no_resource_prepared.resource_ids().empty(),
        "default selection on a resource-free network must be empty");
    require(
        no_resource_prepared.rank_order(options(false, 1U)).size() == 2U,
        "order must ignore an empty resource selection");
    const ErrorSnapshot empty_error = capture_rank_error([&] {
        static_cast<void>(
            no_resource_prepared.rank_ffd(options(false, 1U)));
    });
    require(
        empty_error.code == LinkRankErrorCode::empty_resource_selection,
        "resource-free FFD error code mismatch");

    auto explicit_empty_network = make_network(
        make_path_graph(1U),
        {"resource"},
        {{AttrValue{std::int64_t{1}}}});
    const auto explicit_empty =
        LinkRanker(select_resources({})).prepare(explicit_empty_network);
    require(
        explicit_empty.rank_order(options(false, 1U)).size() == 1U,
        "order must accept a present empty resource vector");
    require(
        capture_rank_error([&] {
            static_cast<void>(explicit_empty.rank_ffd());
        }).code == LinkRankErrorCode::empty_resource_selection,
        "explicit empty FFD selection error code mismatch");
}

void test_selection_and_resource_matrix_errors()
{
    auto invalid_network = make_network(
        make_path_graph(1U),
        {"valid"},
        {{AttrValue{std::int64_t{1}}}});
    const auto valid = binding_of(invalid_network, "valid");
    const LinkRankResourceId invalid_id =
        std::numeric_limits<LinkRankResourceId>::max();
    const ErrorSnapshot invalid_error = capture_rank_error([&] {
        static_cast<void>(LinkRanker(select_resources(
            {valid.registry_id, invalid_id}))
                              .prepare(invalid_network));
    });
    require(
        invalid_error.code ==
            LinkRankErrorCode::invalid_resource_selection,
        "invalid resource selection error code mismatch");
    require(
        invalid_error.operation == LinkRankOperation::prepare &&
            invalid_error.input_index == 1U &&
            invalid_error.resource_id == invalid_id,
        "invalid resource selection metadata mismatch");

    auto ragged_network = make_network(
        make_path_graph(3U),
        {"r0", "r1"},
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
        });
    const auto ragged_r1 = binding_of(ragged_network, "r1");
    require(
        ragged_network.graph()
            .edge_attrs(ragged_network.graph().edge(1U, 2U))
            .erase(ragged_r1.value_id),
        "ragged fixture erase failed");
    const auto ragged_prepared = LinkRanker{}.prepare(ragged_network);
    require(
        capture_rank_error([&] {
            static_cast<void>(
                ragged_prepared.rank_ffd(options(false, 8U)));
        }).code == LinkRankErrorCode::ragged_resource_matrix,
        "ragged resource matrix error code mismatch");

    auto short_network = make_network(
        make_path_graph(3U),
        {"r0", "r1"},
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
        });
    const auto short_r0 = binding_of(short_network, "r0");
    const auto short_r1 = binding_of(short_network, "r1");
    require(
        short_network.graph()
            .edge_attrs(short_network.graph().edge(0U, 1U))
            .erase(short_r0.value_id),
        "equal-short first erase failed");
    require(
        short_network.graph()
            .edge_attrs(short_network.graph().edge(2U, 3U))
            .erase(short_r1.value_id),
        "equal-short second erase failed");
    const auto short_prepared = LinkRanker{}.prepare(short_network);
    require(
        capture_rank_error([&] {
            static_cast<void>(
                short_prepared.rank_ffd(options(false, 2U)));
        }).code == LinkRankErrorCode::ranking_length_mismatch,
        "equal-short rows must fail at final ranking length validation");

    auto nonnumeric_network = make_network(
        make_path_graph(3U),
        {"resource"},
        {{
            AttrValue{std::int64_t{1}},
            AttrValue{std::string{"not numeric"}},
            AttrValue{std::int64_t{3}},
        }});
    const auto nonnumeric = binding_of(nonnumeric_network, "resource");
    const auto nonnumeric_prepared =
        LinkRanker{}.prepare(nonnumeric_network);
    const ErrorSnapshot nonnumeric_error = capture_rank_error([&] {
        static_cast<void>(
            nonnumeric_prepared.rank_ffd(options(false, 8U)));
    });
    require(
        nonnumeric_error.code ==
            LinkRankErrorCode::non_numeric_resource_value,
        "nonnumeric resource error code mismatch");
    require(
        nonnumeric_error.operation == LinkRankOperation::reduce &&
            nonnumeric_error.input_index == 0U &&
            nonnumeric_error.resource_id == nonnumeric.registry_id &&
            nonnumeric_error.edge_id == LinkRankEdgeId{1U},
        "nonnumeric resource error metadata mismatch");
}

void test_prepared_reuse_and_invalidation()
{
    auto network = make_network(
        make_hole_graph(),
        {"resource"},
        {{
            AttrValue{std::int64_t{1}},
            AttrValue{std::int64_t{2}},
            AttrValue{std::int64_t{3}},
            AttrValue{std::int64_t{4}},
        }});
    const auto resource = binding_of(network, "resource");
    const auto prepared = LinkRanker{}.prepare(network);

    require_ids_and_values(
        network,
        prepared.rank_ffd(options(false, 1U)),
        {4U, 0U, 3U, 2U},
        {1.0, 2.0, 3.0, 4.0},
        "prepared initial ranking");

    network.graph()
        .edge_attrs(network.graph().edge(0U, 2U))
        .set(resource.value_id, AttrValue{std::int64_t{100}});
    require_ids_and_values(
        network,
        prepared.rank_ffd(options(false, 2U)),
        {4U, 0U, 3U, 2U},
        {100.0, 2.0, 3.0, 4.0},
        "prepared value mutation");

    const Edge added = network.graph().add_edge(0U, 4U);
    require(
        network.graph().edge_id(added) == 5U,
        "prepared edge mutation must not reuse the removed ID");
    network.graph()
        .edge_attrs(added)
        .set(resource.value_id, AttrValue{std::int64_t{7}});
    network.invalidate_cached_cardinalities();
    require_ids_and_values(
        network,
        prepared.rank_ffd(options(false, 8U)),
        {4U, 5U, 0U, 3U, 2U},
        {100.0, 7.0, 2.0, 3.0, 4.0},
        "prepared edge mutation");

    auto invalidated_network = make_network(
        make_path_graph(2U),
        {"resource"},
        {{
            AttrValue{std::int64_t{1}},
            AttrValue{std::int64_t{2}},
        }});
    const auto invalidated_prepared =
        LinkRanker{}.prepare(invalidated_network);
    const ::AttributeRegistry* old_registry =
        &invalidated_network.graph().attribute_registry();
    Graph replacement(invalidated_network.graph());
    invalidated_network.graph() = std::move(replacement);
    invalidated_network.rebind_attribute_values();
    invalidated_network.invalidate_cached_cardinalities();
    require(
        &invalidated_network.graph().attribute_registry() != old_registry,
        "invalidation fixture did not replace graph registry identity");

    const ErrorSnapshot invalidated_error = capture_rank_error([&] {
        static_cast<void>(invalidated_prepared.rank_order());
    });
    require(
        invalidated_error.code == LinkRankErrorCode::invalid_prepared_state &&
            invalidated_error.operation ==
                LinkRankOperation::validate_prepared,
        "prepared registry replacement was not rejected");
}

void test_concurrent_read_only_ranking()
{
    auto network = make_network(
        make_hole_graph(),
        {"r0", "r1"},
        {
            {
                AttrValue{std::int64_t{1}},
                AttrValue{std::int64_t{2}},
                AttrValue{std::int64_t{3}},
                AttrValue{std::int64_t{4}},
            },
            {
                AttrValue{0.5},
                AttrValue{-2.5},
                AttrValue{4.5},
                AttrValue{8.5},
            },
        });
    const auto prepared = LinkRanker{}.prepare(network);
    const LinkRanking expected = prepared.rank_ffd(options(true, 2U));

    constexpr std::size_t thread_count = 4U;
    constexpr std::size_t repetitions = 4U;
    constexpr std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    std::atomic<bool> okay{true};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (std::size_t thread_index = 0U;
         thread_index < thread_count;
         ++thread_index) {
        threads.emplace_back([&, thread_index] {
            try {
                for (std::size_t repetition = 0U;
                     repetition < repetitions;
                     ++repetition) {
                    const std::size_t workers =
                        widths[(thread_index + repetition) % widths.size()];
                    if (!same_ranking_bits(
                            expected,
                            prepared.rank_ffd(options(true, workers)))) {
                        okay.store(false, std::memory_order_relaxed);
                        return;
                    }
                }
            } catch (...) {
                okay.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    require(
        okay.load(std::memory_order_relaxed),
        "concurrent read-only prepared ranking was not deterministic");
}

}  // namespace

int main()
{
    try {
        test_method_strings_and_errors();
        test_order_sort_and_edge_id_holes();
        test_ffd_default_explicit_and_duplicate_resources();
        test_integer_modular_lane_and_bool();
        test_mixed_double_special_values_and_workers();
        test_zero_edges_and_empty_resources();
        test_selection_and_resource_matrix_errors();
        test_prepared_reuse_and_invalidation();
        test_concurrent_read_only_ranking();
        std::cout << "link_rank_unit: all cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "link_rank_unit: " << error.what() << '\n';
        return 1;
    }
}
