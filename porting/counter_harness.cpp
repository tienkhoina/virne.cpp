#include "counter.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace core = virne::core;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using core::Counter;
using core::CounterErrorCode;
using core::CounterException;
using core::CounterNumber;
using core::CounterOptions;
using core::CounterResourceId;
using core::CounterSelection;
using core::LinkPathInfoKey;
using core::PreparedCounter;
using core::Solution;
using core::SolutionAttributeValues;
using core::SolutionLink;
using core::SolutionMetadata;

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = owner == AttributeOwner::node
        ? CheckingLevel::node
        : CheckingLevel::link;
    return result;
}

network::NodeAttributeDataUpdate dense_node_update(
    CounterResourceId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    CounterResourceId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing Counter node binding");
    }
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing Counter link binding");
    }
    return *binding;
}

network::VirtualNetwork make_main_network(bool with_lifetime)
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node),
        resource_spec("memory", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bw", AttributeOwner::link)};
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto memory = require_node_binding(result, "memory");
    const auto bandwidth = require_link_binding(result, "bw");
    result.set_node_attrs_data({
        dense_node_update(
            cpu.registry_id,
            {std::int64_t{2}, std::int64_t{3}, std::int64_t{4}}),
        dense_node_update(memory.registry_id, {1.5, 2.5, 3.5})});
    result.set_link_attrs_data({
        sparse_link_update(
            bandwidth.registry_id,
            {{0U, 1U, std::int64_t{5}},
             {1U, 2U, std::int64_t{7}}})});
    if (with_lifetime)
    {
        result.set_lifetime(3.0);
    }
    return result;
}

network::VirtualNetwork make_single_resource_network(
    std::vector<AttrValue> values)
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        values.size(), std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("value", AttributeOwner::node)};
    network::VirtualNetwork result(std::move(construction));
    const auto binding = require_node_binding(result, "value");
    result.set_node_attrs_data({dense_node_update(
        binding.registry_id, std::move(values))});
    return result;
}

struct Fixture
{
    network::VirtualNetwork network;
    network::NodeNetworkAttributeBinding cpu;
    network::NodeNetworkAttributeBinding memory;
    network::LinkNetworkAttributeBinding bandwidth;
    PreparedCounter counter;

    explicit Fixture(bool with_lifetime = true)
        : network(make_main_network(with_lifetime)),
          cpu(require_node_binding(network, "cpu")),
          memory(require_node_binding(network, "memory")),
          bandwidth(require_link_binding(network, "bw")),
          counter(Counter().prepare(network))
    {
    }
};

Solution make_solution()
{
    SolutionMetadata metadata;
    metadata.v_net_id = 31;
    metadata.v_net_num_nodes = 3U;
    metadata.v_net_num_edges = 2U;
    Solution result(metadata);
    result.node_slots.insert_or_assign(0, 10);
    result.node_slots.insert_or_assign(2, 12);
    result.link_paths.insert_or_assign(
        SolutionLink{0, 1},
        {SolutionLink{0, 1}, SolutionLink{1, 2}});
    result.link_paths.insert_or_assign(SolutionLink{1, 2}, {});

    SolutionAttributeValues first;
    first.set(0U, std::int64_t{5});
    SolutionAttributeValues second;
    second.set(0U, std::int64_t{5});
    result.link_paths_info.insert_or_assign(
        LinkPathInfoKey{SolutionLink{0, 1}, SolutionLink{0, 1}},
        std::move(first));
    result.link_paths_info.insert_or_assign(
        LinkPathInfoKey{SolutionLink{0, 1}, SolutionLink{1, 2}},
        std::move(second));
    return result;
}

void seed_metrics(Solution& value)
{
    value.v_net_demand = 4.0;
    value.v_net_node_demand = 41.0;
    value.v_net_link_demand = 42.0;
    value.v_net_node_revenue = 43.0;
    value.v_net_link_revenue = 44.0;
    value.v_net_revenue = 45.0;
    value.v_net_node_cost = 46.0;
    value.v_net_link_cost = 47.0;
    value.v_net_path_cost = 48.0;
    value.v_net_cost = 49.0;
    value.v_net_r2c_ratio = 50.0;
    value.v_net_time_revenue = 51.0;
    value.v_net_time_cost = 52.0;
    value.v_net_time_rc_ratio = 53.0;
}

std::string double_token(double value)
{
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

std::string number_token(const CounterNumber& value)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    return double_token(std::get<double>(value));
}

std::string optional_size_token(
    const std::optional<std::size_t>& value)
{
    return value ? std::to_string(*value) : "none";
}

std::string solution_payload(const Solution& value)
{
    std::ostringstream stream;
    stream << "counts=" << optional_size_token(value.num_placed_nodes)
           << ',' << optional_size_token(value.num_routed_links)
           << ";demand=" << double_token(value.v_net_node_demand)
           << ',' << double_token(value.v_net_link_demand)
           << ',' << double_token(value.v_net_demand)
           << ";revenue=" << double_token(value.v_net_node_revenue)
           << ',' << double_token(value.v_net_link_revenue)
           << ',' << double_token(value.v_net_revenue)
           << ";cost=" << double_token(value.v_net_node_cost)
           << ',' << double_token(value.v_net_link_cost)
           << ',' << double_token(value.v_net_path_cost)
           << ',' << double_token(value.v_net_cost)
           << ";ratio=" << double_token(value.v_net_r2c_ratio)
           << ";time=" << double_token(value.v_net_time_revenue)
           << ',' << double_token(value.v_net_time_cost)
           << ',' << double_token(value.v_net_time_rc_ratio)
           << ";flags=" << (value.result ? '1' : '0')
           << ',' << (value.place_result ? '1' : '0')
           << ',' << (value.route_result ? '1' : '0')
           << ',' << (value.early_rejection ? '1' : '0');
    return stream.str();
}

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

std::string run_count_success(std::size_t workers)
{
    Fixture fixture;
    Solution solution = make_solution();
    seed_metrics(solution);
    solution.result = true;
    solution.place_result = false;
    solution.route_result = false;
    solution.early_rejection = true;
    fixture.counter.count_solution(solution, CounterOptions{workers});
    return solution_payload(solution);
}

void differential()
{
    Fixture fixture;
    emit(
        "sum_node_mixed",
        number_token(fixture.counter.calculate_sum_node_resource()));
    emit(
        "sum_link_integer",
        number_token(fixture.counter.calculate_sum_link_resource()));
    emit(
        "sum_network_both",
        number_token(
            fixture.counter.calculate_sum_network_resource(true, true)));
    emit(
        "sum_network_node_only",
        number_token(
            fixture.counter.calculate_sum_network_resource(true, false)));
    emit(
        "sum_network_link_only",
        number_token(
            fixture.counter.calculate_sum_network_resource(false, true)));
    emit(
        "sum_network_disabled",
        number_token(
            fixture.counter.calculate_sum_network_resource(false, false)));

    {
        auto network = make_single_resource_network(
            {true, std::int64_t{2}});
        const PreparedCounter counter = Counter().prepare(network);
        emit(
            "sum_bool_int_promotion",
            number_token(counter.calculate_sum_node_resource()));
    }
    {
        auto network = make_single_resource_network(
            {std::numeric_limits<std::int64_t>::max(), std::int64_t{1}});
        const PreparedCounter counter = Counter().prepare(network);
        emit(
            "sum_int64_wrap",
            number_token(counter.calculate_sum_node_resource()));
    }

    {
        Solution solution = make_solution();
        emit(
            "helper_link_cost",
            number_token(
                fixture.counter.calculate_v_net_link_cost(solution)));
        emit(
            "helper_total_cost",
            number_token(
                fixture.counter.calculate_v_net_cost(solution)));
        emit(
            "helper_revenue",
            number_token(fixture.counter.calculate_v_net_revenue()));
    }
    {
        Solution solution = make_solution();
        fixture.counter.count_partial_solution(solution);
        emit("partial_success", solution_payload(solution));
    }
    {
        Solution solution = make_solution();
        seed_metrics(solution);
        solution.link_paths_info.erase(
            LinkPathInfoKey{
                SolutionLink{0, 1}, SolutionLink{1, 2}});
        std::string error = "none";
        try
        {
            fixture.counter.count_partial_solution(solution);
        }
        catch (const CounterException& exception)
        {
            if (exception.code() != CounterErrorCode::missing_route_info)
            {
                throw;
            }
            error = "missing_info";
        }
        emit(
            "partial_missing_info",
            "error=" + error + ";" + solution_payload(solution));
    }
    {
        CounterSelection selection;
        selection.node_resources = std::vector<CounterResourceId>{};
        selection.link_resources =
            std::vector<CounterResourceId>{fixture.bandwidth.registry_id};
        const PreparedCounter empty_node =
            Counter(std::move(selection)).prepare(fixture.network);
        Solution solution = make_solution();
        seed_metrics(solution);
        std::string error = "none";
        try
        {
            empty_node.count_partial_solution(solution);
        }
        catch (const CounterException& exception)
        {
            if (exception.code() !=
                CounterErrorCode::empty_node_resource_selection)
            {
                throw;
            }
            error = "empty_node";
        }
        emit(
            "partial_empty_node_selection",
            "error=" + error + ";" + solution_payload(solution));
    }

    emit("count_success_demand_bug", run_count_success(1U));

    {
        Fixture local;
        Solution solution = make_solution();
        seed_metrics(solution);
        solution.result = false;
        solution.place_result = false;
        solution.route_result = true;
        solution.early_rejection = true;
        local.counter.count_solution(solution);
        emit("count_failure_stale_fields", solution_payload(solution));
    }
    {
        Fixture local(false);
        Solution solution = make_solution();
        seed_metrics(solution);
        solution.result = true;
        solution.place_result = false;
        solution.route_result = false;
        solution.early_rejection = true;
        std::string error = "none";
        try
        {
            local.counter.count_solution(solution);
        }
        catch (const CounterException& exception)
        {
            if (exception.code() != CounterErrorCode::missing_virtual_lifetime)
            {
                throw;
            }
            error = "missing_lifetime";
        }
        emit(
            "count_missing_lifetime_partial",
            "error=" + error + ";" + solution_payload(solution));
    }
    {
        Fixture local;
        Solution solution = make_solution();
        seed_metrics(solution);
        solution.result = true;
        solution.place_result = false;
        solution.route_result = false;
        solution.early_rejection = true;
        solution.link_paths_info.erase(
            LinkPathInfoKey{
                SolutionLink{0, 1}, SolutionLink{1, 2}});
        std::string error = "none";
        try
        {
            local.counter.count_solution(solution);
        }
        catch (const CounterException& exception)
        {
            if (exception.code() != CounterErrorCode::missing_route_info)
            {
                throw;
            }
            error = "missing_info";
        }
        emit(
            "count_missing_info_partial",
            "error=" + error + ";" + solution_payload(solution));
    }

    std::string sum_workers;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        if (!sum_workers.empty())
        {
            sum_workers.push_back('|');
        }
        sum_workers += number_token(
            fixture.counter.calculate_sum_network_resource(
                true, true, CounterOptions{workers}));
    }
    emit("sum_workers_0_1_2_8", sum_workers);

    std::string count_workers;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        if (!count_workers.empty())
        {
            count_workers.push_back('|');
        }
        count_workers += run_count_success(workers);
    }
    emit("count_workers_0_1_2_8", count_workers);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 ||
            (argc == 2 && std::string_view(argv[1]) == "differential"))
        {
            differential();
            return 0;
        }
        throw std::invalid_argument("usage: counter_harness [differential]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "counter harness: " << error.what() << '\n';
        return 1;
    }
}
