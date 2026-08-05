#include "feature_constructor_fixture.h"

#include <chrono>
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
#include <vector>

namespace fixture = feature_constructor_fixture;
namespace feature = virne::solver::learning::rl_core;
namespace attribute = virne::network::attribute;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    std::size_t threads = 1U;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 25;
};

struct ArraySummary {
    std::vector<std::size_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::size_t bytes = 0U;
    std::vector<std::string> bits;
};

struct Output {
    std::string name;
    ArraySummary summary;
};

struct Case {
    std::string name;
    std::vector<Output> outputs;
};

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(
        std::string(message) +
        "\nusage: vne_feature_constructor_harness [--benchmark] "
        "[--threads N] [--warmups N] [--repetitions N] [--iterations N]");
}

int positive_int(const char* text, const char* option) {
    try {
        const auto value = std::stoll(text);
        if (value <= 0 || value > std::numeric_limits<int>::max()) {
            usage_error(std::string(option) + " requires a positive integer");
        }
        return static_cast<int>(value);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " requires a positive integer");
    }
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto read = [&](const char* option) {
            if (++index >= argc) {
                usage_error(std::string(option) + " requires a value");
            }
            return positive_int(argv[index], option);
        };
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--threads") {
            result.threads = static_cast<std::size_t>(read("--threads"));
        } else if (argument == "--warmups") {
            result.warmups = read("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = read("--repetitions");
        } else if (argument == "--iterations") {
            result.iterations = read("--iterations");
        } else {
            usage_error("unknown argument");
        }
    }
    return result;
}

template <typename UInt>
std::string hex_bits(UInt value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(static_cast<int>(sizeof(UInt) * 2U))
           // uint8_t is an unsigned-char typedef on the supported toolchains;
           // stream it as an integer or JSON receives raw control characters.
           << static_cast<std::uint64_t>(value);
    return output.str();
}

template <typename Value, typename UInt>
ArraySummary summarize_raw(
    std::vector<std::size_t> shape,
    std::string dtype,
    const std::vector<Value>& values) {
    ArraySummary result;
    result.shape = std::move(shape);
    result.dtype = std::move(dtype);
    result.bytes = values.size() * sizeof(Value);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    for (std::size_t index = 0U; index < result.bytes; ++index) {
        result.checksum ^= bytes[index];
        result.checksum *= fnv_prime;
    }
    result.bits.reserve(values.size());
    for (const Value value : values) {
        UInt bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        result.bits.push_back(hex_bits(bits));
    }
    return result;
}

ArraySummary summarize(const feature::DenseMatrixF32& value) {
    return summarize_raw<float, std::uint32_t>(
        {value.rows, value.columns}, "float32", value.values);
}

ArraySummary summarize(const feature::DenseMatrixI64& value) {
    return summarize_raw<std::int64_t, std::uint64_t>(
        {value.rows, value.columns}, "int64", value.values);
}

ArraySummary summarize(const feature::DenseVectorF32& value) {
    return summarize_raw<float, std::uint32_t>(
        {value.values.size()}, "float32", value.values);
}

ArraySummary summarize(const feature::DenseVectorI32& value) {
    return summarize_raw<std::int32_t, std::uint32_t>(
        {value.values.size()}, "int32", value.values);
}

ArraySummary summarize(const feature::DenseVectorBool& value) {
    return summarize_raw<std::uint8_t, std::uint8_t>(
        {value.values.size()}, "bool", value.values);
}

template <typename Array>
void add(std::vector<Output>& outputs, std::string name, const Array& value) {
    outputs.push_back({std::move(name), summarize(value)});
}

Case physical_observation_case(
    const virne::network::PhysicalNetwork& p_net,
    const virne::network::VirtualNetwork& v_net,
    const attribute::AttributeBenchmarks& benchmarks,
    const virne::core::Solution& populated,
    const virne::core::Solution& empty) {
    Case result{"obs_physical_multi_resource", {}};
    const auto kinds = feature::ObservationHandler::AttributeKinds{
        attribute::AttributeKind::resource};
    const auto metrics = fixture::fixed_metrics(p_net.num_nodes());
    add(result.outputs, "node_order",
        feature::ObservationHandler::get_node_order_obs(p_net));
    add(result.outputs, "node_degree",
        feature::ObservationHandler::get_node_degree_obs(p_net));
    add(result.outputs, "node_attrs_norm",
        feature::ObservationHandler::get_node_attrs_obs(
            p_net, kinds, &*benchmarks.node_attr_benchmarks));
    add(result.outputs, "link_pair",
        feature::ObservationHandler::get_link_pair_obs(p_net));
    add(result.outputs, "link_index",
        feature::ObservationHandler::get_link_index_obs(p_net));
    add(result.outputs, "link_attrs_norm",
        feature::ObservationHandler::get_link_attrs_obs(
            p_net, kinds, &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "link_aggr_min",
        feature::ObservationHandler::get_link_aggr_attrs_obs(
            p_net, kinds, attribute::LinkAggregation::minimum,
            &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "link_aggr_mean",
        feature::ObservationHandler::get_link_aggr_attrs_obs(
            p_net, kinds, attribute::LinkAggregation::mean,
            &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "link_aggr_max",
        feature::ObservationHandler::get_link_aggr_attrs_obs(
            p_net, kinds, attribute::LinkAggregation::maximum,
            &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "link_aggr_sum",
        feature::ObservationHandler::get_link_aggr_attrs_obs(
            p_net, kinds, attribute::LinkAggregation::sum,
            &*benchmarks.link_sum_attr_benchmarks));
    add(result.outputs, "link_filters",
        feature::ObservationHandler::get_link_filters(
            p_net, {{"bw", 70.0}, {"latency", 10.0}}));
    add(result.outputs, "average_distance_slots",
        feature::ObservationHandler::get_average_distance(
            p_net, populated.node_slots, true));
    add(result.outputs, "average_distance_empty",
        feature::ObservationHandler::get_average_distance(
            p_net, empty.node_slots, true));
    add(result.outputs, "p_status_none",
        feature::ObservationHandler::get_p_net_nodes_status(
            p_net, v_net, populated.node_slots));
    add(result.outputs, "p_status_curr2",
        feature::ObservationHandler::get_p_net_nodes_status(
            p_net, v_net, populated.node_slots, 2U));
    add(result.outputs, "topology_all",
        feature::ObservationHandler::get_node_topological_metrics(
            p_net, metrics));
    return result;
}

Case virtual_observation_case(
    const virne::network::PhysicalNetwork& p_net,
    const virne::network::VirtualNetwork& v_net,
    const attribute::AttributeBenchmarks& benchmarks,
    const virne::core::Solution& populated) {
    Case result{"obs_virtual_multi_resource", {}};
    const auto kinds = feature::ObservationHandler::AttributeKinds{
        attribute::AttributeKind::resource};
    const auto metrics = fixture::fixed_metrics(v_net.num_nodes());
    add(result.outputs, "node_attrs_norm",
        feature::ObservationHandler::get_node_attrs_obs(
            v_net, kinds, &*benchmarks.node_attr_benchmarks));
    add(result.outputs, "link_pair",
        feature::ObservationHandler::get_link_pair_obs(v_net));
    add(result.outputs, "link_attrs_norm",
        feature::ObservationHandler::get_link_attrs_obs(
            v_net, kinds, &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "v_node_status_curr2",
        feature::ObservationHandler::get_v_node_status(
            v_net, 2U, p_net.num_nodes()));
    add(result.outputs, "v_node_demand_curr2",
        feature::ObservationHandler::get_v_node_demand(
            v_net, 2U, kinds, &*benchmarks.node_attr_benchmarks));
    add(result.outputs, "v_link_aggr_mean",
        feature::ObservationHandler::get_v_node_aggr_link_demands(
            v_net, 2U, attribute::LinkAggregation::mean,
            kinds, &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "v_link_aggr_max",
        feature::ObservationHandler::get_v_node_aggr_link_demands(
            v_net, 2U, attribute::LinkAggregation::maximum,
            kinds, &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "v_link_aggr_min",
        feature::ObservationHandler::get_v_node_aggr_link_demands(
            v_net, 2U, attribute::LinkAggregation::minimum,
            kinds, &*benchmarks.link_attr_benchmarks));
    add(result.outputs, "v_link_aggr_sum",
        feature::ObservationHandler::get_v_node_aggr_link_demands(
            v_net, 2U, attribute::LinkAggregation::sum,
            kinds, &*benchmarks.link_sum_attr_benchmarks));
    add(result.outputs, "v_status_none",
        feature::ObservationHandler::get_v_net_nodes_status(
            v_net, populated.node_slots));
    add(result.outputs, "v_status_neighbor",
        feature::ObservationHandler::get_v_net_nodes_status(
            v_net, populated.node_slots, 2U, true, true));
    add(result.outputs, "v_status_end",
        feature::ObservationHandler::get_v_net_nodes_status(
            v_net, populated.node_slots, v_net.num_nodes(), true, false));
    add(result.outputs, "v_neighbor_flags",
        feature::ObservationHandler::get_v_node_neighbor_flags(
            v_net, populated.node_slots, 2U));
    add(result.outputs, "topology_all",
        feature::ObservationHandler::get_node_topological_metrics(
            v_net, metrics));
    return result;
}

Case feature_case(
    std::string name,
    const virne::network::PhysicalNetwork& p_net,
    const virne::network::VirtualNetwork& v_net,
    feature::FeatureConstructorOptions options,
    const virne::core::Solution& solution,
    std::size_t current) {
    const auto kind = options.kind;
    feature::FeatureConstructor constructor(
        p_net, v_net, std::move(options), fixture::prepared(p_net, v_net));
    const auto observation = constructor.construct(solution, current);
    Case result{std::move(name), {}};
    add(result.outputs, "p_net_x", observation.p_net.x);
    add(result.outputs, "p_net_edge_index", observation.p_net.edge_index);
    add(result.outputs, "p_net_edge_attr", observation.p_net.edge_attr);
    if (kind == feature::FeatureConstructorKind::p_net_v_node) {
        add(result.outputs, "v_node_x", *observation.v_node_x);
    } else if (kind == feature::FeatureConstructorKind::p_net_v_net) {
        add(result.outputs, "v_net_x", observation.v_net->x);
        add(result.outputs, "v_net_edge_index", observation.v_net->edge_index);
        add(result.outputs, "v_net_edge_attr", observation.v_net->edge_attr);
    }
    return result;
}

std::vector<Case> exact_cases(
    const virne::network::PhysicalNetwork& p_net,
    const virne::network::VirtualNetwork& v_net,
    const attribute::AttributeBenchmarks& benchmarks,
    const virne::core::Solution& populated,
    const virne::core::Solution& empty) {
    std::vector<Case> result;
    result.push_back(physical_observation_case(
        p_net, v_net, benchmarks, populated, empty));
    result.push_back(virtual_observation_case(
        p_net, v_net, benchmarks, populated));
    result.push_back(feature_case(
        "feature_p_net_all_flags",
        p_net, v_net,
        fixture::options(feature::FeatureConstructorKind::p_net),
        populated, 2U));
    result.push_back(feature_case(
        "feature_p_net_v_node_all_flags",
        p_net, v_net,
        fixture::options(feature::FeatureConstructorKind::p_net_v_node),
        populated, 2U));
    result.push_back(feature_case(
        "feature_p_net_v_net_all_flags",
        p_net, v_net,
        fixture::options(feature::FeatureConstructorKind::p_net_v_net),
        populated, 2U));
    result.push_back(feature_case(
        "feature_no_optional_empty_slots",
        p_net, v_net,
        fixture::options(
            feature::FeatureConstructorKind::p_net_v_node,
            false, false, false, false),
        empty, 0U));
    result.push_back(feature_case(
        "feature_last_valid_empty_slots",
        p_net, v_net,
        fixture::options(feature::FeatureConstructorKind::p_net_v_node),
        empty, v_net.num_nodes() - 1U));
    feature::FeatureConstructor terminal(
        p_net,
        v_net,
        fixture::options(feature::FeatureConstructorKind::p_net_v_node),
        fixture::prepared(p_net, v_net));
    Case terminal_case{"feature_v_node_curr_end", {}};
    add(terminal_case.outputs, "v_node_x",
        terminal.construct_v_node_features(
            populated, v_net.num_nodes()));
    result.push_back(std::move(terminal_case));
    return result;
}

std::string json_escape(std::string_view input) {
    std::string result;
    result.reserve(input.size() + 2U);
    for (const char character : input) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    return result;
}

void write_string(std::ostream& output, std::string_view value) {
    output << '"' << json_escape(value) << '"';
}

void write_summary(std::ostream& output, const ArraySummary& summary) {
    output << "{\"shape\":[";
    for (std::size_t index = 0U; index < summary.shape.size(); ++index) {
        if (index != 0U) output << ',';
        output << summary.shape[index];
    }
    output << "],\"dtype\":";
    write_string(output, summary.dtype);
    output << ",\"checksum\":\"" << summary.checksum
           << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0U; index < summary.bits.size(); ++index) {
        if (index != 0U) output << ',';
        write_string(output, summary.bits[index]);
    }
    output << "]}";
}

void write_cases(std::ostream& output, const std::vector<Case>& cases) {
    output << '[';
    for (std::size_t case_index = 0U; case_index < cases.size(); ++case_index) {
        if (case_index != 0U) output << ',';
        const Case& item = cases[case_index];
        output << "{\"name\":";
        write_string(output, item.name);
        output << ",\"status\":\"ok\",\"outputs\":{";
        for (std::size_t index = 0U; index < item.outputs.size(); ++index) {
            if (index != 0U) output << ',';
            write_string(output, item.outputs[index].name);
            output << ':';
            write_summary(output, item.outputs[index].summary);
        }
        output << "}}";
    }
    output << ']';
}

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    ArraySummary output;
};

BenchmarkResult run_benchmark(
    const Arguments& arguments,
    const virne::network::PhysicalNetwork& p_net,
    const virne::network::VirtualNetwork& v_net,
    const virne::core::Solution& solution) {
    feature::FeatureConstructor constructor(
        p_net,
        v_net,
        fixture::options(
            feature::FeatureConstructorKind::p_net_v_node,
            true, true, true, true, arguments.threads),
        fixture::prepared(p_net, v_net, arguments.threads));
    feature::FeatureObservation last;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            last = constructor.construct(solution, 2U);
        }
    }
    BenchmarkResult result;
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            last = constructor.construct(solution, 2U);
        }
        const auto end = std::chrono::steady_clock::now();
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count());
    }
    result.output = summarize(last.p_net.x);
    return result;
}

void write_benchmark(
    std::ostream& output,
    const Arguments& arguments,
    const BenchmarkResult& benchmark) {
    output << ",\"benchmark\":{\"name\":\"construct_p_net_v_node_all_flags\""
           << ",\"threads\":" << arguments.threads
           << ",\"warmups\":" << arguments.warmups
           << ",\"repetitions\":" << arguments.repetitions
           << ",\"iterations\":" << arguments.iterations
           << ",\"samples_ns\":[";
    for (std::size_t index = 0U; index < benchmark.samples_ns.size(); ++index) {
        if (index != 0U) output << ',';
        output << benchmark.samples_ns[index];
    }
    output << "],\"output\":";
    write_summary(output, benchmark.output);
    output << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const auto p_net = fixture::physical_network();
        const auto v_net = fixture::virtual_network();
        const auto benchmarks = fixture::benchmarks(p_net);
        const auto populated = fixture::solution_with_slots(v_net, true);
        const auto empty = fixture::solution_with_slots(v_net, false);
        const auto cases = exact_cases(
            p_net, v_net, benchmarks, populated, empty);

        std::cout << "{\"component\":\"feature_constructor\","
                     "\"fixture_version\":1,\"cases\":";
        write_cases(std::cout, cases);
        if (arguments.benchmark) {
            const auto benchmark = run_benchmark(
                arguments, p_net, v_net, populated);
            write_benchmark(std::cout, arguments, benchmark);
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "feature-constructor harness error: "
                  << error.what() << '\n';
        return 1;
    }
}
