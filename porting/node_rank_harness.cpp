#include "../virne/solver/rank/node_rank.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace net = virne::network;
namespace attr = virne::network::attribute;
namespace rank = virne::solver::rank;

struct Fixture {
    net::BaseNetwork network;
};

attr::AttributeFactorySpec resource_spec(
    std::string name,
    attr::AttributeOwner owner)
{
    attr::AttributeFactorySpec spec{};
    spec.name = std::move(name);
    spec.owner = owner;
    spec.kind = attr::AttributeKind::resource;
    spec.generative = false;
    spec.restriction = attr::ConstraintRestriction::hard;
    spec.checking_level = owner == attr::AttributeOwner::node
        ? attr::CheckingLevel::node
        : attr::CheckingLevel::link;
    return spec;
}

Fixture make_fixture(
    std::size_t node_count,
    std::vector<EdgeEndpoints> edges,
    const std::vector<std::vector<AttrValue>>& node_rows,
    const std::vector<std::vector<AttrValue>>& link_rows)
{
    net::BaseNetworkConstruction construction{};
    construction.incoming_graph.emplace(node_count, std::move(edges));
    for (std::size_t index = 0U; index < node_rows.size(); ++index) {
        construction.config.node_attribute_specs.push_back(resource_spec(
            "node_resource_" + std::to_string(index),
            attr::AttributeOwner::node));
    }
    for (std::size_t index = 0U; index < link_rows.size(); ++index) {
        construction.config.link_attribute_specs.push_back(resource_spec(
            "link_resource_" + std::to_string(index),
            attr::AttributeOwner::link));
    }

    net::BaseNetwork network(std::move(construction));
    std::vector<net::NodeAttributeDataUpdate> node_updates;
    for (std::size_t index = 0U; index < node_rows.size(); ++index) {
        const auto binding = network.bind_node_attribute(
            "node_resource_" + std::to_string(index));
        if (!binding.has_value()) {
            throw std::runtime_error("NodeRank harness node binding failed");
        }
        net::NodeAttributeDataUpdate update{};
        update.registry_id = binding->registry_id;
        update.layout = net::AttributeDataLayout::dense;
        update.dense_values = node_rows[index];
        node_updates.push_back(std::move(update));
    }
    if (!node_updates.empty()) {
        network.set_node_attrs_data(node_updates, 1U);
    }

    std::vector<net::LinkAttributeDataUpdate> link_updates;
    for (std::size_t index = 0U; index < link_rows.size(); ++index) {
        const auto binding = network.bind_link_attribute(
            "link_resource_" + std::to_string(index));
        if (!binding.has_value()) {
            throw std::runtime_error("NodeRank harness link binding failed");
        }
        net::LinkAttributeDataUpdate update{};
        update.registry_id = binding->registry_id;
        update.layout = net::AttributeDataLayout::dense;
        update.dense_values = link_rows[index];
        link_updates.push_back(std::move(update));
    }
    if (!link_updates.empty()) {
        network.set_link_attrs_data(link_updates, 1U);
    }
    return Fixture{std::move(network)};
}

std::string bits_hex(double value)
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << bits;
    return output.str();
}

void emit_ranking(std::ostream& output, const rank::NodeRanking& ranking)
{
    output << '[';
    for (std::size_t index = 0U; index < ranking.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        const rank::NodeRankEntry& entry = ranking[index];
        output << '[' << entry.node_id << ','
               << (entry.kind == rank::NodeRankValueKind::scalar
                       ? "\"s\""
                       : "\"p\"")
               << ",\"" << bits_hex(entry.value) << "\",\""
               << bits_hex(entry.distance) << "\"]";
    }
    output << ']';
}

rank::NodeRankOptions make_options(
    bool sort,
    std::size_t workers)
{
    rank::NodeRankOptions result{};
    result.sort = sort;
    result.workers = workers;
    return result;
}

Fixture order_fixture()
{
    return make_fixture(4U, {{2U, 3U}, {0U, 1U}}, {}, {});
}

Fixture integer_fixture()
{
    return make_fixture(
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
}

Fixture mixed_fixture()
{
    return make_fixture(
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
}

Fixture metric_fixture()
{
    return make_fixture(
        4U,
        {{0U, 1U}, {1U, 1U}, {1U, 2U}},
        {{
            AttrValue{std::int64_t{2}},
            AttrValue{std::int64_t{3}},
            AttrValue{std::int64_t{-4}},
            AttrValue{std::int64_t{5}},
        }},
        {{AttrValue{1.0}, AttrValue{2.0}, AttrValue{3.0}}});
}

Fixture no_link_fixture()
{
    return make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{-2}}, AttrValue{std::int64_t{3}}}},
        {});
}

Fixture iterative_fixture()
{
    return make_fixture(
        2U,
        {{0U, 1U}},
        {{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{3}}}},
        {{AttrValue{5.0}}});
}

Fixture nps_fixture(bool weighted)
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
    if (weighted) {
        const AttrId weight = fixture.network.graph().attr_id("weight");
        fixture.network.graph()
            .edge_attrs(fixture.network.graph().edge(0U, 2U))
            .set(weight, AttrValue{3.0});
        fixture.network.graph()
            .edge_attrs(fixture.network.graph().edge(0U, 1U))
            .set(weight, AttrValue{1.0});
    }
    return fixture;
}

rank::NodeRanking run_case(
    std::string_view name,
    std::size_t workers)
{
    if (name == "order") {
        Fixture fixture = order_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_order(make_options(true, workers));
    }
    if (name == "ffd_int") {
        Fixture fixture = integer_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_ffd(make_options(true, workers));
    }
    if (name == "ffd_mixed_nan") {
        Fixture fixture = mixed_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_ffd(make_options(true, workers));
    }
    if (name == "nrm") {
        Fixture fixture = metric_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_nrm(make_options(false, workers));
    }
    if (name == "nea") {
        Fixture fixture = metric_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_nea(make_options(false, workers));
    }
    if (name == "nrm_no_links") {
        Fixture fixture = no_link_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_nrm(make_options(false, workers));
    }
    if (name == "grc") {
        Fixture fixture = iterative_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_grc(make_options(false, workers));
    }
    if (name == "grc_inf") {
        Fixture fixture = iterative_fixture();
        rank::NodeRankParameters parameters{};
        parameters.grc.sigma = std::numeric_limits<double>::infinity();
        return rank::NodeRanker({}, parameters)
            .prepare(fixture.network)
            .rank_grc(make_options(false, workers));
    }
    if (name == "rw") {
        Fixture fixture = iterative_fixture();
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_rw(make_options(false, workers));
    }
    if (name == "nps_unsorted" || name == "nps_sorted" ||
        name == "nps_weighted") {
        Fixture fixture = nps_fixture(name == "nps_weighted");
        return rank::NodeRanker{}
            .prepare(fixture.network)
            .rank_nps(make_options(name != "nps_unsorted", workers));
    }
    throw std::invalid_argument("Unknown NodeRank harness case");
}

void run_random_sequence(std::size_t workers)
{
    Fixture fixture = make_fixture(
        6U,
        {{0U, 1U}, {1U, 2U}},
        {},
        {});
    const auto prepared = rank::NodeRanker{}.prepare(fixture.network);
    NumpyRandomState random(42U);
    const rank::NodeRanking first =
        prepared.rank_random(random, make_options(false, workers));
    const rank::NodeRanking second =
        prepared.rank_random(random, make_options(false, workers));
    const std::int64_t next = random.randint(
        std::int64_t{0}, std::int64_t{2147483648LL});
    std::cout << "{\"first\":";
    emit_ranking(std::cout, first);
    std::cout << ",\"second\":";
    emit_ranking(std::cout, second);
    std::cout << ",\"next\":" << next << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            std::cerr << "usage: vne_node_rank_harness CASE WORKERS\n";
            return 64;
        }
        const std::string name = argv[1];
        const std::size_t workers =
            static_cast<std::size_t>(std::stoull(argv[2]));
        if (name == "random_sequence") {
            run_random_sequence(workers);
            return 0;
        }
        const rank::NodeRanking ranking = run_case(name, workers);
        std::cout << "{\"ranking\":";
        emit_ranking(std::cout, ranking);
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "node rank harness error: " << error.what() << '\n';
        return 2;
    }
}
