#include "controller/node_mapper.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace core = virne::core;
namespace controller = virne::core::controller;
namespace network = virne::network;
namespace attribute = virne::network::attribute;

struct Fixture {
    std::unique_ptr<network::VirtualNetwork> virtual_network;
    std::unique_ptr<network::PhysicalNetwork> physical_network;
    controller::PreparedNodeMapper mapper;
    core::Solution solution;
    std::vector<Vertex> virtual_nodes;
    std::vector<Vertex> physical_nodes;
    AttrId physical_cpu_value_id = 0U;
    attribute::AttributeRegistryId cpu_resource_id = 0U;
};

struct BenchmarkResult {
    std::uint64_t elapsed_ns = 0U;
    std::uint64_t checksum = 0U;
    std::size_t output_bytes = 0U;
    std::size_t entry_count = 0U;
};

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

std::string number_token(const attribute::AttributeNumber& value) {
    return std::visit(
        [](const auto item) -> std::string {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, bool>) {
                return item ? "b:1" : "b:0";
            } else if constexpr (std::is_same_v<Item, std::int64_t>) {
                return "i:" + std::to_string(item);
            } else {
                return double_token(item);
            }
        },
        value);
}

std::string attr_value_token(const AttrValue& value) {
    return std::visit(
        [](const auto& item) -> std::string {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, bool>) {
                return item ? "b:1" : "b:0";
            } else if constexpr (std::is_same_v<Item, std::int64_t>) {
                return "i:" + std::to_string(item);
            } else if constexpr (std::is_same_v<Item, double>) {
                return double_token(item);
            } else {
                throw std::runtime_error(
                    "NodeMapper benchmark entered a non-numeric AttrValue lane");
            }
        },
        value);
}

std::uint64_t fingerprint(const std::string_view value) {
    std::uint64_t result = 14695981039346656037ULL;
    for (const char raw_byte : value) {
        result ^= static_cast<unsigned char>(raw_byte);
        result *= 1099511628211ULL;
    }
    return result;
}

attribute::AttributeFactorySpec cpu_spec() {
    attribute::AttributeFactorySpec result;
    result.name = "cpu";
    result.owner = attribute::AttributeOwner::node;
    result.kind = attribute::AttributeKind::resource;
    return result;
}

network::VirtualNetwork make_virtual_network(const std::size_t node_count) {
    Graph graph;
    for (std::size_t index = 0U; index < node_count; ++index) {
        graph.add_node();
    }
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(std::move(graph));
    construction.config.node_attribute_specs.push_back(cpu_spec());
    network::VirtualNetwork result(std::move(construction));
    const auto cpu_id = result.node_attributes().bind("cpu");
    if (!cpu_id) {
        throw std::runtime_error("virtual cpu definition was not bound");
    }
    std::vector<AttrValue> demands(node_count, AttrValue(std::int64_t{2}));
    result.set_node_attrs_data({
        {*cpu_id, network::AttributeDataLayout::dense, {}, std::move(demands)},
    });
    return result;
}

network::PhysicalNetwork make_physical_network(
    const std::size_t virtual_nodes,
    const std::size_t low_candidates) {
    const std::size_t node_count = low_candidates + virtual_nodes;
    Graph graph;
    for (std::size_t index = 0U; index < node_count; ++index) {
        graph.add_node();
    }
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(std::move(graph));
    construction.config.node_attribute_specs.push_back(cpu_spec());
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu_id = result.node_attributes().bind("cpu");
    if (!cpu_id) {
        throw std::runtime_error("physical cpu definition was not bound");
    }
    std::vector<AttrValue> capacities;
    capacities.reserve(node_count);
    capacities.insert(
        capacities.end(), low_candidates, AttrValue(std::int64_t{1}));
    capacities.insert(
        capacities.end(), virtual_nodes, AttrValue(std::int64_t{2}));
    result.set_node_attrs_data({
        {*cpu_id, network::AttributeDataLayout::dense, {}, std::move(capacities)},
    });
    return result;
}

Fixture make_fixture(
    const std::size_t virtual_node_count,
    const std::size_t low_candidate_count) {
    auto virtual_network = std::make_unique<network::VirtualNetwork>(
        make_virtual_network(virtual_node_count));
    auto physical_network = std::make_unique<network::PhysicalNetwork>(
        make_physical_network(virtual_node_count, low_candidate_count));
    const auto cpu_id = virtual_network->node_attributes().bind("cpu");
    const auto cpu_binding = physical_network->bind_node_attribute("cpu");
    if (!cpu_id || !cpu_binding) {
        throw std::runtime_error("cpu benchmark binding is missing");
    }

    controller::NodeMapper mapper(controller::NodeMapperSelection{
        {*cpu_id}, {*cpu_id}, {*cpu_id}});
    auto prepared = mapper.prepare(*virtual_network, *physical_network);
    core::Solution solution(core::SolutionMetadata{
        0,
        0.0,
        0.0,
        virtual_node_count,
        0U,
    });
    // The differential fixture begins with both result flags true. Successful
    // mapping preserves them, so this is part of the exact output gate.
    solution.result = true;

    std::vector<Vertex> virtual_nodes(virtual_node_count);
    for (std::size_t index = 0U; index < virtual_node_count; ++index) {
        virtual_nodes[index] = static_cast<Vertex>(index);
    }
    const std::size_t physical_node_count =
        low_candidate_count + virtual_node_count;
    std::vector<Vertex> physical_nodes(physical_node_count);
    for (std::size_t index = 0U; index < physical_node_count; ++index) {
        physical_nodes[index] = static_cast<Vertex>(index);
    }

    return Fixture{
        std::move(virtual_network),
        std::move(physical_network),
        std::move(prepared),
        std::move(solution),
        std::move(virtual_nodes),
        std::move(physical_nodes),
        cpu_binding->value_id,
        *cpu_id,
    };
}

std::string values_payload(
    const core::SolutionAttributeValues& values,
    const attribute::AttributeRegistryId cpu_id) {
    const auto* cpu = values.find(cpu_id);
    if (cpu == nullptr) {
        throw std::runtime_error("solution cpu value is missing");
    }
    return "{" + std::to_string(cpu_id) + "=" + number_token(*cpu) + "}";
}

std::string solution_payload(const Fixture& fixture) {
    const auto& solution = fixture.solution;
    std::ostringstream stream;
    stream << "phys=[";
    for (Vertex node = 0U;
         node < fixture.physical_network->graph().num_nodes();
         ++node) {
        if (node != 0U) {
            stream << ',';
        }
        stream << attr_value_token(
            fixture.physical_network->graph().node_attrs(node).at(
                fixture.physical_cpu_value_id));
    }
    stream << "];slots=[";
    for (std::size_t index = 0U; index < solution.node_slots.entries().size();
         ++index) {
        if (index != 0U) {
            stream << ',';
        }
        const auto& entry = solution.node_slots.entries()[index];
        stream << entry.key << ':' << entry.value;
    }
    stream << "];info=[";
    for (std::size_t index = 0U;
         index < solution.node_slots_info.entries().size(); ++index) {
        if (index != 0U) {
            stream << ',';
        }
        const auto& entry = solution.node_slots_info.entries()[index];
        stream << entry.key.virtual_node << ':' << entry.key.physical_node
               << values_payload(entry.value, fixture.cpu_resource_id);
    }
    stream << "];offsets=[";
    for (std::size_t index = 0U;
         index < solution.v_net_constraint_offsets.node_level.entries().size();
         ++index) {
        if (index != 0U) {
            stream << ',';
        }
        const auto& entry =
            solution.v_net_constraint_offsets.node_level.entries()[index];
        stream << entry.key
               << values_payload(entry.value, fixture.cpu_resource_id);
    }
    stream << "];violations=[";
    for (std::size_t index = 0U;
         index < solution.v_net_constraint_violations.node_level.entries().size();
         ++index) {
        if (index != 0U) {
            stream << ',';
        }
        const auto& entry =
            solution.v_net_constraint_violations.node_level.entries()[index];
        stream << entry.key
               << values_payload(entry.value, fixture.cpu_resource_id);
    }
    stream << "];total="
           << double_token(solution.v_net_total_hard_constraint_violation)
           << ";flags=" << (solution.place_result ? '1' : '0') << ','
           << (solution.result ? '1' : '0');
    return stream.str();
}

BenchmarkResult run_benchmark(
    const std::size_t virtual_node_count,
    const std::size_t low_candidate_count,
    const std::size_t candidate_workers) {
    Fixture fixture = make_fixture(virtual_node_count, low_candidate_count);
    controller::NodeMappingOptions options;
    options.reusable = false;
    options.inplace = true;
    options.method = controller::NodeMatchingMethod::greedy;
    options.allow_constraint_violation = false;
    options.candidate_workers = candidate_workers;

    const auto begin = std::chrono::steady_clock::now();
    const bool mapped = fixture.mapper.node_mapping(
        fixture.virtual_nodes,
        fixture.physical_nodes,
        fixture.solution,
        options);
    const auto end = std::chrono::steady_clock::now();
    if (!mapped) {
        throw std::runtime_error("valid greedy mapping unexpectedly failed");
    }

    const auto elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    const std::string output = "mapped=1;" + solution_payload(fixture);
    const std::size_t physical_nodes =
        low_candidate_count + virtual_node_count;
    const std::size_t entries =
        physical_nodes + 4U * virtual_node_count;
    return BenchmarkResult{
        elapsed_ns,
        fingerprint(output),
        output.size(),
        entries,
    };
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: node_mapper_benchmark <virtual_nodes> "
                "<low_candidates> <candidate_workers>");
        }
        const auto virtual_nodes =
            static_cast<std::size_t>(std::stoull(argv[1]));
        const auto low_candidates =
            static_cast<std::size_t>(std::stoull(argv[2]));
        const auto candidate_workers =
            static_cast<std::size_t>(std::stoull(argv[3]));
        if (virtual_nodes == 0U || low_candidates == 0U) {
            throw std::invalid_argument(
                "virtual_nodes and low_candidates must be positive");
        }

        const BenchmarkResult result = run_benchmark(
            virtual_nodes, low_candidates, candidate_workers);
        const std::size_t physical_nodes = low_candidates + virtual_nodes;
        const std::size_t minimum_candidate_checks =
            virtual_nodes * (low_candidates + 1U);
        std::cout << "protocol=1\n"
                  << "kind=node_mapper_greedy\n"
                  << "virtual_nodes=" << virtual_nodes << '\n'
                  << "low_candidates=" << low_candidates << '\n'
                  << "physical_nodes=" << physical_nodes << '\n'
                  << "minimum_candidate_checks=" << minimum_candidate_checks
                  << '\n'
                  << "candidate_workers=" << candidate_workers << '\n'
                  << "type_tag=ordered_numeric_mapping_v1\n"
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "entry_count=" << result.entry_count << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "node_mapper_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
