#include "../virne/solver/rank/link_rank.h"

#include <cstddef>
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
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;
namespace rank = virne::solver::rank;

struct Fixture {
    network::BaseNetwork network;
    std::vector<rank::LinkRankResourceId> resource_ids;
};

AttrValue integer(const std::int64_t value) {
    return AttrValue(value);
}

AttrValue real(const double value) {
    return AttrValue(value);
}

attribute::AttributeFactorySpec resource_spec(std::string name) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = attribute::AttributeOwner::link;
    result.kind = attribute::AttributeKind::resource;
    return result;
}

Fixture make_fixture(
    std::vector<EdgeEndpoints> public_links,
    std::vector<std::vector<AttrValue>> resource_rows,
    std::optional<std::vector<EdgeEndpoints>> insertion_links = std::nullopt) {
    const auto& graph_input = insertion_links
        ? *insertion_links
        : public_links;
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(graph_input);
    std::vector<std::string> resource_names;
    resource_names.reserve(resource_rows.size());
    construction.config.link_attribute_specs.reserve(resource_rows.size());
    for (std::size_t index = 0U; index < resource_rows.size(); ++index) {
        resource_names.push_back("resource_" + std::to_string(index));
        construction.config.link_attribute_specs.push_back(
            resource_spec(resource_names.back()));
    }

    network::BaseNetwork result(std::move(construction));
    if (result.graph().num_edges() != public_links.size()) {
        throw std::runtime_error("LinkRank harness edge-count drift");
    }

    std::vector<rank::LinkRankResourceId> resource_ids;
    resource_ids.reserve(resource_names.size());
    std::vector<network::LinkAttributeDataUpdate> updates;
    updates.reserve(resource_rows.size());
    for (std::size_t row = 0U; row < resource_rows.size(); ++row) {
        if (resource_rows[row].size() > public_links.size()) {
            throw std::runtime_error("LinkRank harness resource row is too long");
        }
        const auto binding = result.bind_link_attribute(resource_names[row]);
        if (!binding) {
            throw std::runtime_error("LinkRank harness resource binding failed");
        }
        resource_ids.push_back(binding->registry_id);

        network::LinkAttributeDataUpdate update;
        update.registry_id = binding->registry_id;
        update.layout = network::AttributeDataLayout::sparse;
        update.sparse_values.reserve(resource_rows[row].size());
        for (std::size_t column = 0U;
             column < resource_rows[row].size();
             ++column) {
            update.sparse_values.push_back({
                public_links[column].first,
                public_links[column].second,
                std::move(resource_rows[row][column]),
            });
        }
        updates.push_back(std::move(update));
    }
    result.set_link_attrs_data(updates);

    std::size_t ordered_index = 0U;
    const auto [edge_begin, edge_end] = result.graph().edges();
    for (auto iterator = edge_begin; iterator != edge_end; ++iterator) {
        if (ordered_index >= public_links.size()) {
            throw std::runtime_error("LinkRank harness edge-order overflow");
        }
        if (result.graph().source(*iterator) != public_links[ordered_index].first ||
            result.graph().target(*iterator) != public_links[ordered_index].second) {
            throw std::runtime_error("LinkRank harness endpoint order drift");
        }
        ++ordered_index;
    }
    if (ordered_index != public_links.size()) {
        throw std::runtime_error("LinkRank harness edge-order underflow");
    }

    return Fixture{std::move(result), std::move(resource_ids)};
}

rank::PreparedLinkRanker prepare(const Fixture& fixture) {
    rank::LinkRankSelection selection;
    selection.resources = fixture.resource_ids;
    return rank::LinkRanker(std::move(selection)).prepare(fixture.network);
}

rank::LinkRankOptions options(
    const bool sort,
    const std::size_t workers) {
    rank::LinkRankOptions result;
    result.sort = sort;
    result.workers = workers;
    return result;
}

std::string double_bits(const double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << std::hex << std::nouppercase << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

std::string ranking_payload(
    const rank::LinkRanking& ranking,
    const Fixture& fixture) {
    std::ostringstream stream;
    stream << "{\"error\":\"none\",\"ranking\":[";
    for (std::size_t index = 0U; index < ranking.size(); ++index) {
        const auto& entry = ranking[index];
        const auto native_endpoints =
            fixture.network.graph().edge_endpoints(entry.edge_id);
        if (entry.source != native_endpoints.first ||
            entry.target != native_endpoints.second) {
            throw std::runtime_error("LinkRank result endpoint/ID mismatch");
        }
        if (index != 0U) {
            stream << ',';
        }
        stream << '[' << entry.source << ',' << entry.target
               << ",\"" << double_bits(entry.value) << "\"]";
    }
    stream << "]}";
    return stream.str();
}

std::string error_payload(const std::string_view token) {
    return "{\"error\":\"" + std::string(token) + "\",\"ranking\":[]}";
}

template <typename Function>
std::string mapped_error(
    Function&& function,
    const rank::LinkRankErrorCode expected,
    const std::string_view token) {
    try {
        static_cast<void>(function());
    } catch (const rank::LinkRankException& error) {
        if (error.code() != expected) {
            throw std::runtime_error("LinkRank harness error-code mismatch");
        }
        return error_payload(token);
    }
    throw std::runtime_error("LinkRank harness expected an error");
}

void emit_case(const std::string_view name, const std::string& payload) {
    std::cout << name << '\t' << payload << '\n';
}

void differential() {
    auto order_fixture = make_fixture(
        {{0U, 3U}, {0U, 1U}, {1U, 4U}, {2U, 4U}},
        {},
        std::vector<EdgeEndpoints>{
            {2U, 4U}, {0U, 3U}, {1U, 4U}, {0U, 1U}});
    const auto order_ranker = prepare(order_fixture);
    emit_case(
        "order_unsorted",
        ranking_payload(
            order_ranker.rank_order(options(false, 1U)), order_fixture));
    emit_case(
        "order_sorted",
        ranking_payload(
            order_ranker.rank_order(options(true, 1U)), order_fixture));

    auto integer_fixture = make_fixture(
        {{0U, 1U}, {1U, 2U}, {2U, 3U}},
        {
            {integer(7), integer(-3), integer(10)},
            {integer(2), integer(8), integer(-4)},
            {integer(1), integer(0), integer(2)},
        });
    const auto integer_ranker = prepare(integer_fixture);
    emit_case(
        "ffd_int",
        ranking_payload(
            integer_ranker.rank_ffd(options(false, 1U)), integer_fixture));

    auto mixed_fixture = make_fixture(
        {{10U, 11U}, {11U, 12U}, {12U, 13U}, {13U, 14U}},
        {
            {real(0.5), real(-2.0), integer(8), real(-0.0)},
            {real(0.25), integer(4), real(-1.5), real(0.0)},
            {integer(1), real(-0.5), real(0.25), real(-2.0)},
        });
    const auto mixed_ranker = prepare(mixed_fixture);
    for (const std::size_t workers : {1U, 2U, 8U}) {
        emit_case(
            "ffd_mixed_workers_" + std::to_string(workers),
            ranking_payload(
                mixed_ranker.rank_ffd(options(true, workers)), mixed_fixture));
    }

    auto ties_fixture = make_fixture(
        {{20U, 21U}, {21U, 22U}, {22U, 23U}, {23U, 24U}},
        {
            {integer(4), integer(2), integer(4), integer(3)},
            {integer(1), integer(3), integer(0), integer(2)},
        });
    const auto ties_ranker = prepare(ties_fixture);
    emit_case(
        "ffd_stable_ties",
        ranking_payload(
            ties_ranker.rank_ffd(options(true, 1U)), ties_fixture));

    auto zero_fixture = make_fixture({}, {{}, {}});
    const auto zero_ranker = prepare(zero_fixture);
    emit_case(
        "ffd_zero_edge",
        ranking_payload(
            zero_ranker.rank_ffd(options(true, 1U)), zero_fixture));

    auto wrap_fixture = make_fixture(
        {{30U, 31U}, {31U, 32U}},
        {
            {
                integer(std::numeric_limits<std::int64_t>::max()),
                integer(std::numeric_limits<std::int64_t>::min()),
            },
            {integer(1), integer(-1)},
        });
    const auto wrap_ranker = prepare(wrap_fixture);
    emit_case(
        "ffd_int64_wrap",
        ranking_payload(
            wrap_ranker.rank_ffd(options(false, 1U)), wrap_fixture));

    auto empty_fixture = make_fixture({{40U, 41U}}, {});
    const auto empty_ranker = prepare(empty_fixture);
    emit_case(
        "ffd_empty_resources_error",
        mapped_error(
            [&empty_ranker]() {
                return empty_ranker.rank_ffd(options(true, 1U));
            },
            rank::LinkRankErrorCode::empty_resource_selection,
            "empty_resources"));

    auto ragged_fixture = make_fixture(
        {{50U, 51U}, {51U, 52U}, {52U, 53U}},
        {
            {integer(1), integer(2), integer(3)},
            {integer(4), integer(5)},
        });
    const auto ragged_ranker = prepare(ragged_fixture);
    emit_case(
        "ffd_ragged_error",
        mapped_error(
            [&ragged_ranker]() {
                return ragged_ranker.rank_ffd(options(true, 1U));
            },
            rank::LinkRankErrorCode::ragged_resource_matrix,
            "ragged_resources"));

    // This is an explicitly recorded Python-only boundary. The native typed
    // API intentionally implements FFD via resolved link-resource IDs.
    emit_case(
        "ffd_base_network_typo", error_payload("missing_get_attrs"));
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        if (argc != 2 || std::string_view(argv[1]) != "differential") {
            std::cerr << "usage: link_rank_harness differential\n";
            return 2;
        }
        differential();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
