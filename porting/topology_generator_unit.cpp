#include "topology/topology_generator.h"

#include "generators/topology_generators.h"
#include "py_random.h"
#include "random_context.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using virne::network::TopologyGenerator;
using virne::network::TopologyOptions;
using virne::network::TopologyRequest;
using virne::network::TopologyType;

constexpr std::size_t default_parity_guard = 100000;

void require(
    bool condition,
    const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::uint64_t double_bits(
    double value)
{
    static_assert(
        sizeof(double) == sizeof(std::uint64_t),
        "unit test requires an IEEE-754-sized double");

    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool attr_values_bit_equal(
    const AttrValue& lhs,
    const AttrValue& rhs)
{
    if (lhs.index() != rhs.index())
    {
        return false;
    }

    if (const auto* value = std::get_if<std::int64_t>(&lhs))
    {
        return *value == std::get<std::int64_t>(rhs);
    }
    if (const auto* value = std::get_if<double>(&lhs))
    {
        return double_bits(*value) ==
            double_bits(std::get<double>(rhs));
    }
    if (const auto* value = std::get_if<bool>(&lhs))
    {
        return *value == std::get<bool>(rhs);
    }
    if (const auto* value = std::get_if<std::string>(&lhs))
    {
        return *value == std::get<std::string>(rhs);
    }
    if (std::holds_alternative<AttrListPtr>(lhs))
    {
        const AttrList* left = attr_list(lhs);
        const AttrList* right = attr_list(rhs);
        if (left == nullptr || right == nullptr)
        {
            return left == right;
        }
        if (left->values.size() != right->values.size())
        {
            return false;
        }
        for (std::size_t index = 0;
             index < left->values.size();
             ++index)
        {
            if (!attr_values_bit_equal(
                    left->values[index],
                    right->values[index]))
            {
                return false;
            }
        }
        return true;
    }

    const AttrObject* left = attr_object(lhs);
    const AttrObject* right = attr_object(rhs);
    if (left == nullptr || right == nullptr)
    {
        return left == right;
    }
    if (left->entries.size() != right->entries.size())
    {
        return false;
    }
    for (std::size_t index = 0;
         index < left->entries.size();
         ++index)
    {
        if (left->entries[index].first !=
                right->entries[index].first ||
            !attr_values_bit_equal(
                left->entries[index].second,
                right->entries[index].second))
        {
            return false;
        }
    }
    return true;
}

void require_attrs_equal(
    const Graph& lhs_owner,
    const AttrMap& lhs,
    const Graph& rhs_owner,
    const AttrMap& rhs,
    const std::string& context)
{
    require(
        lhs.size() == rhs.size(),
        context + ": attribute count mismatch");

    const auto& left_ids = lhs.attribute_ids();
    const auto& right_ids = rhs.attribute_ids();
    for (std::size_t index = 0;
         index < left_ids.size();
         ++index)
    {
        const std::string_view left_name =
            lhs_owner.attr_name(left_ids[index]);
        const std::string_view right_name =
            rhs_owner.attr_name(right_ids[index]);
        require(
            left_name == right_name,
            context + ": attribute name/order mismatch");
        require(
            attr_values_bit_equal(
                lhs.at(left_ids[index]),
                rhs.at(right_ids[index])),
            context + ": attribute value mismatch for " +
                std::string(left_name));
    }
}

std::pair<std::size_t, std::size_t> attribute_counts(
    const Graph& graph)
{
    std::size_t node_attributes = 0;
    for (Vertex node = 0;
         node < graph.num_nodes();
         ++node)
    {
        node_attributes += graph.node_attrs(node).size();
    }

    std::size_t edge_attributes = 0;
    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        edge_attributes += graph.edge_attrs(*edge).size();
    }
    return {node_attributes, edge_attributes};
}

std::vector<Vertex> neighbor_order(
    const Graph& graph,
    Vertex node)
{
    std::vector<Vertex> result;
    const auto [neighbor, neighbor_end] = graph.neighbors(node);
    for (auto current = neighbor; current != neighbor_end; ++current)
    {
        result.push_back(*current);
    }
    return result;
}

void require_graph_equal(
    const Graph& lhs,
    const Graph& rhs,
    const std::string& context)
{
    require(
        lhs.num_nodes() == rhs.num_nodes(),
        context + ": node count mismatch");
    require(
        lhs.num_edges() == rhs.num_edges(),
        context + ": edge count mismatch");
    require(
        lhs.attribute_registry().size() ==
            rhs.attribute_registry().size(),
        context + ": attribute registry size mismatch");
    for (std::size_t id = 0;
         id < lhs.attribute_registry().size();
         ++id)
    {
        require(
            lhs.attr_name(static_cast<AttrId>(id)) ==
                rhs.attr_name(static_cast<AttrId>(id)),
            context + ": attribute registry order mismatch");
    }

    require_attrs_equal(
        lhs,
        lhs.graph_attrs(),
        rhs,
        rhs.graph_attrs(),
        context + ": graph");

    auto [left_node, left_node_end] = lhs.nodes();
    auto [right_node, right_node_end] = rhs.nodes();
    std::size_t node_index = 0;
    for (;
         left_node != left_node_end &&
             right_node != right_node_end;
         ++left_node, ++right_node, ++node_index)
    {
        require(
            *left_node == *right_node,
            context + ": node order mismatch at index " +
                std::to_string(node_index));
        require_attrs_equal(
            lhs,
            lhs.node_attrs(*left_node),
            rhs,
            rhs.node_attrs(*right_node),
            context + ": node " + std::to_string(node_index));
    }
    require(
        left_node == left_node_end &&
            right_node == right_node_end,
        context + ": node iterator length mismatch");

    auto [left_edge, left_edge_end] = lhs.edges();
    auto [right_edge, right_edge_end] = rhs.edges();
    std::size_t edge_index = 0;
    for (;
         left_edge != left_edge_end &&
             right_edge != right_edge_end;
         ++left_edge, ++right_edge, ++edge_index)
    {
        require(
            lhs.source(*left_edge) == rhs.source(*right_edge) &&
                lhs.target(*left_edge) == rhs.target(*right_edge),
            context + ": edge order/endpoints mismatch at index " +
                std::to_string(edge_index));
        require_attrs_equal(
            lhs,
            lhs.edge_attrs(*left_edge),
            rhs,
            rhs.edge_attrs(*right_edge),
            context + ": edge " + std::to_string(edge_index));
    }
    require(
        left_edge == left_edge_end &&
            right_edge == right_edge_end,
        context + ": edge iterator length mismatch");
    require(
        attribute_counts(lhs) == attribute_counts(rhs),
        context + ": aggregate attribute counts mismatch");
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    std::string_view expected_text,
    const std::string& context)
{
    try
    {
        function();
    }
    catch (const Exception& error)
    {
        require(
            std::string_view(error.what()).find(expected_text) !=
                std::string_view::npos,
            context + ": exception text mismatch: " + error.what());
        return;
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            context + ": wrong exception type: " + error.what());
    }

    throw std::runtime_error(context + ": expected an exception");
}

template <typename Function>
std::string exception_signature(
    Function&& function)
{
    try
    {
        function();
    }
    catch (const std::invalid_argument& error)
    {
        return "invalid_argument:" + std::string(error.what());
    }
    catch (const std::overflow_error& error)
    {
        return "overflow_error:" + std::string(error.what());
    }
    catch (const std::runtime_error& error)
    {
        return "runtime_error:" + std::string(error.what());
    }
    catch (const std::exception& error)
    {
        return "exception:" + std::string(error.what());
    }

    throw std::runtime_error("expected an exception while capturing signature");
}

void require_stream_continuation_equal(
    PyRandom& lhs,
    PyRandom& rhs,
    const std::string& context)
{
    require(
        double_bits(lhs.random()) == double_bits(rhs.random()),
        context + ": next RNG value mismatch");
}

void require_waxman_shape(
    const Graph& graph,
    const std::string& context)
{
    require(
        graph.attribute_registry().size() == 1 &&
            graph.attr_name(0) == "pos",
        context + ": Waxman attribute registry mismatch");
    const auto [node_attributes, edge_attributes] =
        attribute_counts(graph);
    require(
        node_attributes == graph.num_nodes(),
        context + ": every Waxman node must have exactly one attribute");
    require(
        edge_attributes == 0,
        context + ": Waxman edges must not have attributes");

    const std::optional<AttrId> pos_id =
        graph.attribute_registry().find("pos");
    require(
        pos_id.has_value(),
        context + ": missing pos field ID");

    for (Vertex node = 0;
         node < graph.num_nodes();
         ++node)
    {
        const AttrValue* position =
            graph.node_attrs(node).find(*pos_id);
        require(position != nullptr, context + ": missing pos attribute");
        const AttrList* coordinates = attr_list(*position);
        require(
            coordinates != nullptr && coordinates->values.size() == 2 &&
                std::holds_alternative<double>(coordinates->values[0]) &&
                std::holds_alternative<double>(coordinates->values[1]),
            context + ": invalid pos attribute shape");
    }
}

TopologyRequest make_request(
    TopologyType type,
    std::int64_t num_nodes,
    TopologyOptions options,
    std::uint64_t seed)
{
    TopologyRequest request;
    request.type = std::move(type);
    request.num_nodes = num_nodes;
    request.options = std::move(options);
    request.seed = seed;
    return request;
}

} // namespace

int main()
{
    try
    {
        const TopologyOptions defaults;

        for (const std::int64_t node_count : {1, 2, 7})
        {
            PyRandom random(11);
            const Graph actual = TopologyGenerator::generate(
                "path", node_count, defaults, random);
            require_graph_equal(
                actual,
                nx::path_graph(static_cast<std::size_t>(node_count)),
                "path wrapper");
        }

        for (const std::int64_t node_count : {1, 2, 8})
        {
            PyRandom random(12);
            const Graph actual = TopologyGenerator::generate(
                "star", node_count, defaults, random);
            require_graph_equal(
                actual,
                nx::star_graph(
                    static_cast<std::size_t>(node_count - 1)),
                "star wrapper");
        }

        struct GridCase
        {
            std::int64_t rows;
            std::int64_t columns;
            std::size_t expected_rows;
            std::size_t expected_columns;
        };
        const std::vector<GridCase> grid_cases{
            {3, 4, 3, 4},
            {1, 1, 1, 1},
            {0, 4, 0, 4},
            {3, 0, 3, 0}};
        for (const GridCase& test : grid_cases)
        {
            TopologyOptions options;
            options.m = test.rows;
            options.n = test.columns;
            PyRandom random(13);
            const Graph actual = TopologyGenerator::generate(
                "grid_2d", 999, options, random);
            require_graph_equal(
                actual,
                nx::grid_2d_graph(
                    test.expected_rows,
                    test.expected_columns,
                    false),
                "grid wrapper");
        }

        {
            TopologyOptions options;
            options.m = 3;
            options.n = 4;
            PyRandom random(13);
            const Graph grid = TopologyGenerator::generate(
                "grid_2d", 1, options, random);
            require(
                neighbor_order(grid, 1) ==
                    std::vector<Vertex>{5, 0, 2},
                "grid node 1 neighbor order must match NetworkX");
            require(
                neighbor_order(grid, 5) ==
                    std::vector<Vertex>{1, 9, 4, 6},
                "grid node 5 neighbor order must match NetworkX");
        }

        TopologyOptions only_rows;
        only_rows.m = 3;
        TopologyOptions only_columns;
        only_columns.n = 4;
        PyRandom validation_random(14);
        for (const TopologyOptions& options :
             {defaults, only_rows, only_columns})
        {
            require_throws<std::invalid_argument>(
                [&] {
                    (void)TopologyGenerator::generate(
                        "grid_2d", 1, options, validation_random);
                },
                "requires 'm' and 'n'",
                "grid missing dimension");
        }

        TopologyOptions negative_rows;
        negative_rows.m = -3;
        negative_rows.n = 4;
        TopologyOptions negative_columns;
        negative_columns.m = 3;
        negative_columns.n = -4;
        for (const TopologyOptions& options :
             {negative_rows, negative_columns})
        {
            require_throws<std::invalid_argument>(
                [&] {
                    (void)TopologyGenerator::generate(
                        "grid_2d", 1, options, validation_random);
                },
                "Negative number of nodes not valid",
                "negative grid dimension");
        }

        for (const std::string_view type :
             {"path", "star", "grid_2d", "random", "waxman", "unknown"})
        {
            require_throws<std::invalid_argument>(
                [&] {
                    (void)TopologyGenerator::generate(
                        type, 0, defaults, validation_random);
                },
                "num_nodes must be >= 1",
                "zero num_nodes validation");
        }
        require_throws<std::invalid_argument>(
            [&] {
                (void)TopologyGenerator::generate(
                    "path", -7, defaults, validation_random);
            },
            "num_nodes must be >= 1",
            "negative num_nodes validation");
        require_throws<std::invalid_argument>(
            [&] {
                (void)TopologyGenerator::generate(
                    "unsupported", 5, defaults, validation_random);
            },
            "Graph type 'unsupported' is not implemented",
            "unsupported topology validation");

        PyRandom deterministic_stream(19);
        PyRandom deterministic_control(19);
        (void)TopologyGenerator::generate(
            "path", 20, defaults, deterministic_stream);
        require_stream_continuation_equal(
            deterministic_stream,
            deterministic_control,
            "deterministic topology must not consume RNG");

        const std::vector<std::uint64_t> seeds{
            0,
            1,
            42,
            UINT64_C(0x100000001)};
        for (const std::uint64_t seed : seeds)
        {
            TopologyOptions options;
            options.random_prob = 0.43;
            options.max_attempts = 2000;
            PyRandom actual_random(seed);
            PyRandom expected_random(seed);
            const Graph actual = TopologyGenerator::generate(
                "random", 12, options, actual_random);
            const Graph expected = nx::connected_erdos_renyi_graph(
                12,
                options.random_prob,
                expected_random,
                *options.max_attempts);
            require_graph_equal(
                actual,
                expected,
                "random wrapper seed " + std::to_string(seed));
            require(
                attribute_counts(actual) ==
                    std::pair<std::size_t, std::size_t>{0, 0},
                "random wrapper added attributes");
            require_stream_continuation_equal(
                actual_random,
                expected_random,
                "random wrapper continuation");
        }

        {
            PyRandom actual_random(73);
            PyRandom expected_random(73);
            const Graph actual = TopologyGenerator::generate(
                "random", 9, defaults, actual_random);
            const Graph expected = nx::connected_erdos_renyi_graph(
                9,
                0.5,
                expected_random,
                default_parity_guard);
            require_graph_equal(actual, expected, "random defaults");
            require_stream_continuation_equal(
                actual_random,
                expected_random,
                "random defaults continuation");
        }

        {
            TopologyOptions options;
            options.random_prob = 1.0;
            PyRandom actual_random(81);
            PyRandom expected_random(81);
            const Graph actual = TopologyGenerator::generate(
                "random", 7, options, actual_random);
            const Graph expected = nx::connected_erdos_renyi_graph(
                7,
                1.0,
                expected_random,
                default_parity_guard);
            require_graph_equal(actual, expected, "complete random graph");
            require_stream_continuation_equal(
                actual_random,
                expected_random,
                "complete random graph continuation");
        }

        for (const std::uint64_t seed : seeds)
        {
            TopologyOptions options;
            options.wm_alpha = 0.88;
            options.wm_beta = 0.57;
            options.max_attempts = 2000;
            PyRandom actual_random(seed);
            PyRandom expected_random(seed);

            WaxmanConfig expected_config;
            expected_config.num_nodes = 9;
            expected_config.beta = options.wm_alpha;
            expected_config.alpha = options.wm_beta;

            const Graph actual = TopologyGenerator::generate(
                "waxman", 9, options, actual_random);
            const Graph expected = nx::connected_waxman_graph(
                expected_config,
                expected_random,
                *options.max_attempts);
            require_graph_equal(
                actual,
                expected,
                "Waxman positional mapping seed " +
                    std::to_string(seed));
            require_waxman_shape(actual, "Waxman positional mapping");
            require_stream_continuation_equal(
                actual_random,
                expected_random,
                "Waxman positional mapping continuation");
        }

        {
            PyRandom actual_random(91);
            PyRandom expected_random(91);
            WaxmanConfig expected_config;
            expected_config.num_nodes = 5;
            expected_config.beta = 0.5;
            expected_config.alpha = 0.2;
            const Graph actual = TopologyGenerator::generate(
                "waxman", 5, defaults, actual_random);
            const Graph expected = nx::connected_waxman_graph(
                expected_config,
                expected_random,
                default_parity_guard);
            require_graph_equal(actual, expected, "Waxman defaults");
            require_waxman_shape(actual, "Waxman defaults");
            require_stream_continuation_equal(
                actual_random,
                expected_random,
                "Waxman defaults continuation");
        }

        {
            TopologyOptions options;
            options.random_prob = 0.41;
            options.max_attempts = 2000;
            constexpr std::uint64_t seed = UINT64_C(0x123456789);

            global_py_random().seed(seed);
            const Graph actual = TopologyGenerator::generate(
                "random", 11, options);
            const std::uint64_t actual_next =
                double_bits(global_py_random().random());

            global_py_random().seed(seed);
            const Graph expected = nx::connected_erdos_renyi_graph(
                11,
                options.random_prob,
                global_py_random(),
                *options.max_attempts);
            const std::uint64_t expected_next =
                double_bits(global_py_random().random());

            require_graph_equal(actual, expected, "global random stream");
            require(
                actual_next == expected_next,
                "global random stream continuation mismatch");
        }

        {
            TopologyOptions options;
            options.wm_alpha = 0.9;
            options.wm_beta = 0.6;
            options.max_attempts = 2000;
            constexpr std::uint64_t seed = UINT64_C(0x200000007);

            global_py_random().seed(seed);
            const Graph actual = TopologyGenerator::generate(
                "waxman", 8, options);
            const std::uint64_t actual_next =
                double_bits(global_py_random().random());

            WaxmanConfig expected_config;
            expected_config.num_nodes = 8;
            expected_config.beta = options.wm_alpha;
            expected_config.alpha = options.wm_beta;
            global_py_random().seed(seed);
            const Graph expected = nx::connected_waxman_graph(
                expected_config,
                global_py_random(),
                *options.max_attempts);
            const std::uint64_t expected_next =
                double_bits(global_py_random().random());

            require_graph_equal(actual, expected, "global Waxman stream");
            require(
                actual_next == expected_next,
                "global Waxman stream continuation mismatch");
        }

        {
            TopologyOptions zero_attempts;
            zero_attempts.random_prob = 1.0;
            zero_attempts.max_attempts = 0;
            PyRandom random(101);
            require_throws<std::invalid_argument>(
                [&] {
                    (void)TopologyGenerator::generate(
                        "random", 3, zero_attempts, random);
                },
                "max_attempts > 0",
                "zero max_attempts safety guard");
        }

        {
            TopologyOptions impossible;
            impossible.random_prob = 0.0;
            impossible.max_attempts = 3;
            PyRandom random(102);
            require_throws<std::runtime_error>(
                [&] {
                    (void)TopologyGenerator::generate(
                        "random", 3, impossible, random);
                },
                "Unable to generate a connected graph",
                "impossible random graph safety guard");
        }

        {
            TopologyOptions exhausted;
            exhausted.wm_alpha = 0.0;
            exhausted.wm_beta = 0.2;
            exhausted.max_attempts = 2;
            PyRandom actual_random(103);
            PyRandom expected_random(103);
            WaxmanConfig expected_config;
            expected_config.num_nodes = 4;
            expected_config.beta = 0.0;
            expected_config.alpha = 0.2;

            const std::string actual_error = exception_signature(
                [&] {
                    (void)TopologyGenerator::generate(
                        "waxman", 4, exhausted, actual_random);
                });
            const std::string expected_error = exception_signature(
                [&] {
                    (void)nx::connected_waxman_graph(
                        expected_config,
                        expected_random,
                        *exhausted.max_attempts);
                });
            require(
                actual_error == expected_error &&
                    actual_error.find("Unable to generate") !=
                        std::string::npos,
                "Waxman retry exhaustion mismatch");
            require_stream_continuation_equal(
                actual_random,
                expected_random,
                "Waxman exhausted retry continuation");
        }

        const std::vector<TopologyRequest> empty_requests;
        for (const std::size_t workers : {0U, 1U, 2U, 4U, 8U})
        {
            require(
                TopologyGenerator::generate_batch(
                    empty_requests, workers).empty(),
                "empty batch must remain empty");
        }

        std::vector<TopologyRequest> requests;
        TopologyOptions grid_options;
        grid_options.m = 3;
        grid_options.n = 5;
        TopologyOptions empty_grid_options;
        empty_grid_options.m = 0;
        empty_grid_options.n = 7;
        TopologyOptions random_options;
        random_options.random_prob = 0.48;
        random_options.max_attempts = 1000;
        TopologyOptions waxman_options;
        waxman_options.wm_alpha = 0.92;
        waxman_options.wm_beta = 0.64;
        waxman_options.max_attempts = 1000;

        requests.push_back(make_request(
            TopologyType::Path, 17, defaults, 201));
        requests.push_back(make_request(
            TopologyType::Random, 10, random_options, 202));
        requests.push_back(make_request(
            TopologyType::Star, 11, defaults, 203));
        requests.push_back(make_request(
            TopologyType::Waxman, 8, waxman_options, 204));
        requests.push_back(make_request(
            TopologyType::Grid2D, 1, grid_options, 205));
        requests.push_back(make_request(
            TopologyType::Path, 3, defaults, 206));
        requests.push_back(make_request(
            TopologyType::Random, 13, random_options, 207));
        requests.push_back(make_request(
            TopologyType::Star, 6, defaults, 208));
        requests.push_back(make_request(
            TopologyType::Waxman, 9, waxman_options, 209));
        requests.push_back(make_request(
            TopologyType::Grid2D, 77, empty_grid_options, 210));

        std::vector<Graph> expected_batch;
        expected_batch.reserve(requests.size());
        for (const TopologyRequest& request : requests)
        {
            PyRandom random(request.seed);
            expected_batch.push_back(TopologyGenerator::generate(
                request.type,
                request.num_nodes,
                request.options,
                random));
        }

        for (const std::size_t workers : {1U, 2U, 4U, 8U, 0U})
        {
            const std::vector<Graph> actual_batch =
                TopologyGenerator::generate_batch(requests, workers);
            require(
                actual_batch.size() == expected_batch.size(),
                "batch result count mismatch");
            for (std::size_t index = 0;
                 index < actual_batch.size();
                 ++index)
            {
                require_graph_equal(
                    actual_batch[index],
                    expected_batch[index],
                    "batch order/workers=" + std::to_string(workers) +
                        "/index=" + std::to_string(index));
            }
        }

        std::vector<TopologyRequest> failing_requests;
        TopologyOptions impossible_batch;
        impossible_batch.random_prob = 0.0;
        impossible_batch.max_attempts = 1;
        failing_requests.push_back(
            make_request(TopologyType::Path, 4, defaults, 301));
        failing_requests.push_back(
            make_request(
                TopologyType::Random, 4, impossible_batch, 302));
        failing_requests.push_back(
            make_request(
                static_cast<TopologyType>(255), 4, defaults, 303));
        failing_requests.push_back(
            make_request(TopologyType::Path, -1, defaults, 304));

        std::optional<std::string> failure_reference;
        for (const std::size_t workers : {1U, 2U, 4U, 8U, 0U})
        {
            const std::string failure = exception_signature(
                [&] {
                    (void)TopologyGenerator::generate_batch(
                        failing_requests, workers);
                });
            require(
                failure.find(
                    "runtime_error:Unable to generate a connected graph") == 0,
                "batch did not select earliest input exception");
            if (!failure_reference.has_value())
            {
                failure_reference = failure;
            }
            require(
                failure == *failure_reference,
                "batch exception changed across worker counts");
        }

        std::cout << "vne_topology_generator_unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vne_topology_generator_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
