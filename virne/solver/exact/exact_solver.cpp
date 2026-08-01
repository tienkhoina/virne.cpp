#include "exact_solver.h"

#include "../../core/controller/controller.h"
#include "../../network/physical_network.h"
#include "../../network/virtual_network.h"
#include "py_random.h"

#include <ortools/linear_solver/linear_solver.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace virne::solver::exact {
namespace {

using ResourceId = network::attribute::AttributeRegistryId;
using AttributeNumber = network::attribute::AttributeNumber;
using VertexPair = std::pair<Vertex, Vertex>;

constexpr double kEpsilon = 1.0e-9;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kMetaBandwidth = 9'999.0;
constexpr std::int64_t kLargestExactlyRepresentableInteger =
    std::int64_t{1} << 53U;

struct NodeResourceBinding {
    ResourceId solution_id = 0U;
    AttrId virtual_value_id = 0U;
    AttrId physical_value_id = 0U;
};

struct LinkResourceBinding {
    ResourceId solution_id = 0U;
    AttrId virtual_value_id = 0U;
    AttrId physical_value_id = 0U;
};

struct PhysicalEdge {
    Vertex source = 0U;
    Vertex target = 0U;
    std::uint32_t edge_id = 0U;
};

struct PhysicalArc {
    Vertex source = 0U;
    Vertex target = 0U;
    std::size_t edge_index = 0U;
};

struct ExactModelData {
    const network::VirtualNetwork& virtual_network;
    const network::PhysicalNetwork& physical_network;
    std::vector<NodeResourceBinding> node_resources;
    std::vector<LinkResourceBinding> link_resources;
    std::vector<AttributeNumber> node_demand_values;
    std::vector<AttributeNumber> node_capacity_values;
    std::vector<AttributeNumber> link_demand_values;
    std::vector<AttributeNumber> link_capacity_values;
    std::vector<double> node_demands;
    std::vector<double> node_capacities;
    std::vector<double> link_demands;
    std::vector<double> link_capacities;
    std::vector<double> link_flow_units;
    std::vector<std::uint8_t> link_flow_integral;
    std::vector<double> link_resource_per_flow_unit;
    std::vector<VertexPair> virtual_edges;
    std::vector<PhysicalEdge> physical_edges;
    std::vector<PhysicalArc> physical_arcs;
    std::vector<std::vector<std::size_t>> outgoing_arcs;
    std::vector<std::vector<std::size_t>> incoming_arcs;

    std::size_t virtual_node_count = 0U;
    std::size_t physical_node_count = 0U;
    std::size_t virtual_edge_count = 0U;
    std::size_t physical_edge_count = 0U;
    std::size_t node_resource_count = 0U;
    std::size_t link_resource_count = 0U;
    bool coefficients_exact_in_double = true;

    double node_demand(std::size_t v, std::size_t r) const noexcept {
        return node_demands[v * node_resource_count + r];
    }

    double node_capacity(std::size_t p, std::size_t r) const noexcept {
        return node_capacities[p * node_resource_count + r];
    }

    double link_demand(std::size_t e, std::size_t r) const noexcept {
        return link_demands[e * link_resource_count + r];
    }

    double link_capacity(std::size_t e, std::size_t r) const noexcept {
        return link_capacities[e * link_resource_count + r];
    }

    double link_flow_unit(std::size_t e) const noexcept {
        return link_flow_units[e];
    }

    bool has_integral_link_flow(std::size_t e) const noexcept {
        return link_flow_integral[e] != 0U;
    }

    double link_resource_ratio(std::size_t e, std::size_t r) const noexcept {
        return link_resource_per_flow_unit[e * link_resource_count + r];
    }
};

AttributeNumber attribute_number(const AttrValue& value) {
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return *item;
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return *item;
    }
    throw std::invalid_argument("exact solver resource is not numeric");
}

bool coefficient_is_exact_in_double(const AttributeNumber& value) noexcept {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer >= -kLargestExactlyRepresentableInteger &&
               *integer <= kLargestExactlyRepresentableInteger;
    }
    return true;
}

double required_numeric(
    const AttrValue& value,
    const char* diagnostic) {
    const double result = attr_to_double(value);
    if (!std::isfinite(result) || result < 0.0) {
        throw std::invalid_argument(
            std::string("exact solver received invalid ") + diagnostic +
            " resource");
    }
    return result;
}

double required_numeric(
    const AttrMap& values,
    AttrId id,
    const char* diagnostic) {
    const AttrValue* value = values.find(id);
    if (value == nullptr) {
        throw std::invalid_argument(
            std::string("exact solver missing ") + diagnostic + " resource");
    }
    return required_numeric(*value, diagnostic);
}

AttributeNumber scaled_resource_amount(
    const AttributeNumber& original,
    const double original_demand,
    const double used_flow,
    const double flow_units) {
    if (flow_units <= 0.0 || used_flow <= 0.0 ||
        original_demand <= 0.0) {
        return std::holds_alternative<double>(original)
            ? AttributeNumber{0.0}
            : AttributeNumber{std::int64_t{0}};
    }
    if (!std::holds_alternative<double>(original)) {
        const std::int64_t original_integer =
            std::holds_alternative<std::int64_t>(original)
            ? std::get<std::int64_t>(original)
            : (std::get<bool>(original) ? 1 : 0);
        const double rounded_flow = std::round(used_flow);
        const double rounded_units = std::round(flow_units);
        if (std::fabs(used_flow - rounded_flow) > 1.0e-7 ||
            std::fabs(flow_units - rounded_units) > 1.0e-7 ||
            rounded_units <= 0.0 ||
            rounded_flow < 0.0 ||
            rounded_units > static_cast<double>(
                std::numeric_limits<std::int64_t>::max()) ||
            rounded_flow > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument(
                "exact solver cannot extract an integral resource journal");
        }
        const auto integral_units = static_cast<std::int64_t>(rounded_units);
        const auto integral_flow = static_cast<std::int64_t>(rounded_flow);
        if (original_integer % integral_units != 0) {
            throw std::invalid_argument(
                "exact solver integral resource ratio is not exact");
        }
        const std::int64_t ratio = original_integer / integral_units;
        if (ratio != 0 && integral_flow >
                std::numeric_limits<std::int64_t>::max() / ratio) {
            throw std::overflow_error(
                "exact solver integral resource journal overflow");
        }
        return ratio * integral_flow;
    }
    const double amount = used_flow * original_demand / flow_units;
    return amount;
}

bool subtract_resource_exactly(
    const AttributeNumber& demand,
    AttributeNumber& residual) noexcept {
    try {
        return network::attribute::update_resource_value(
            demand,
            residual,
            network::attribute::ResourceUpdateOperation::subtract,
            true);
    } catch (const network::attribute::AttributeMethodException&) {
        return false;
    }
}

std::vector<NodeResourceBinding> bind_node_resources(
    const core::controller::ControllerSelection& selection,
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network) {
    std::vector<NodeResourceBinding> result;
    result.reserve(selection.node_resources.size());
    const auto& virtual_registry = virtual_network.node_attributes();
    const auto& physical_registry = physical_network.node_attributes();
    for (const ResourceId id : selection.node_resources) {
        if (static_cast<std::size_t>(id) >= virtual_registry.entries().size()) {
            throw std::invalid_argument("exact solver node resource ID is out of range");
        }
        // Names are resolved once at the model boundary. The returned AttrIds
        // are the only fields used by the MIP construction and extraction.
        const std::string& name = virtual_registry.entries()[id].name;
        const auto physical_id = physical_registry.bind(name);
        if (!physical_id.has_value()) {
            throw std::invalid_argument(
                "physical network has no node resource named " + name);
        }
        result.push_back(NodeResourceBinding{
            id,
            virtual_registry.at(id).bind(virtual_network.graph()).value_id,
            physical_registry.at(*physical_id)
                .bind(physical_network.graph())
                .value_id});
    }
    return result;
}

std::vector<LinkResourceBinding> bind_link_resources(
    const core::controller::ControllerSelection& selection,
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network) {
    std::vector<LinkResourceBinding> result;
    result.reserve(selection.link_resources.size());
    const auto& virtual_registry = virtual_network.link_attributes();
    const auto& physical_registry = physical_network.link_attributes();
    for (const ResourceId id : selection.link_resources) {
        if (static_cast<std::size_t>(id) >= virtual_registry.entries().size()) {
            throw std::invalid_argument("exact solver link resource ID is out of range");
        }
        const std::string& name = virtual_registry.entries()[id].name;
        const auto physical_id = physical_registry.bind(name);
        if (!physical_id.has_value()) {
            throw std::invalid_argument(
                "physical network has no link resource named " + name);
        }
        result.push_back(LinkResourceBinding{
            id,
            virtual_registry.at(id).bind(virtual_network.graph()).value_id,
            physical_registry.at(*physical_id)
                .bind(physical_network.graph())
                .value_id});
    }
    return result;
}

std::vector<VertexPair> ordered_edges(const Graph& graph) {
    std::vector<VertexPair> result;
    result.reserve(graph.num_edges());
    const auto edges = graph.edges();
    for (auto iterator = edges.first; iterator != edges.second; ++iterator) {
        const Edge edge = *iterator;
        result.emplace_back(graph.source(edge), graph.target(edge));
    }
    return result;
}

ExactModelData prepare_model(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const core::controller::ControllerSelection& selection) {
    ExactModelData data{
        virtual_network,
        physical_network,
        bind_node_resources(selection, virtual_network, physical_network),
        bind_link_resources(selection, virtual_network, physical_network)};
    data.virtual_node_count = virtual_network.num_nodes();
    data.physical_node_count = physical_network.num_nodes();
    data.virtual_edges = ordered_edges(virtual_network.graph());
    data.virtual_edge_count = data.virtual_edges.size();
    data.node_resource_count = data.node_resources.size();
    data.link_resource_count = data.link_resources.size();

    data.node_demands.resize(
        data.virtual_node_count * data.node_resource_count);
    data.node_capacities.resize(
        data.physical_node_count * data.node_resource_count);
    data.node_demand_values.resize(
        data.virtual_node_count * data.node_resource_count);
    data.node_capacity_values.resize(
        data.physical_node_count * data.node_resource_count);
    for (std::size_t v = 0U; v < data.virtual_node_count; ++v) {
        for (std::size_t r = 0U; r < data.node_resource_count; ++r) {
            const auto& binding = data.node_resources[r];
            const AttrValue* value = virtual_network.graph().node_attrs(
                static_cast<Vertex>(v)).find(binding.virtual_value_id);
            if (value == nullptr) {
                throw std::invalid_argument("virtual node resource is missing");
            }
            const AttributeNumber number = attribute_number(*value);
            data.node_demand_values[v * data.node_resource_count + r] =
                number;
            data.coefficients_exact_in_double =
                data.coefficients_exact_in_double &&
                coefficient_is_exact_in_double(number);
            data.node_demands[v * data.node_resource_count + r] =
                required_numeric(*value, "virtual node");
        }
    }
    for (std::size_t p = 0U; p < data.physical_node_count; ++p) {
        for (std::size_t r = 0U; r < data.node_resource_count; ++r) {
            const AttrMap& values = physical_network.graph().node_attrs(
                static_cast<Vertex>(p));
            const AttrValue* value = values.find(
                data.node_resources[r].physical_value_id);
            if (value == nullptr) {
                throw std::invalid_argument(
                    "exact solver missing physical node resource");
            }
            const AttributeNumber number = attribute_number(*value);
            data.node_capacity_values[p * data.node_resource_count + r] =
                number;
            data.coefficients_exact_in_double =
                data.coefficients_exact_in_double &&
                coefficient_is_exact_in_double(number);
            data.node_capacities[p * data.node_resource_count + r] =
                required_numeric(*value, "physical node");
        }
    }

    data.link_demands.resize(data.virtual_edge_count * data.link_resource_count);
    data.link_demand_values.resize(
        data.virtual_edge_count * data.link_resource_count);
    data.link_flow_units.resize(data.virtual_edge_count, 0.0);
    data.link_flow_integral.assign(data.virtual_edge_count, 1U);
    data.link_resource_per_flow_unit.resize(
        data.virtual_edge_count * data.link_resource_count, 0.0);
    for (std::size_t e = 0U; e < data.virtual_edge_count; ++e) {
        const auto [source, target] = data.virtual_edges[e];
        const Edge edge = virtual_network.graph().edge(source, target);
        for (std::size_t r = 0U; r < data.link_resource_count; ++r) {
            const auto& binding = data.link_resources[r];
            const AttrValue* value = virtual_network.graph().edge_attrs(edge).find(
                binding.virtual_value_id);
            if (value == nullptr) {
                throw std::invalid_argument("virtual link resource is missing");
            }
            const AttributeNumber number = attribute_number(*value);
            data.link_demand_values[e * data.link_resource_count + r] =
                number;
            data.coefficients_exact_in_double =
                data.coefficients_exact_in_double &&
                coefficient_is_exact_in_double(number);
            data.link_demands[e * data.link_resource_count + r] =
                required_numeric(*value, "virtual link");
        }
        if (data.link_resource_count == 0U) {
            data.link_flow_units[e] = 1.0;
        } else {
            std::int64_t integral_gcd = 0;
            for (std::size_t r = 0U;
                 r < data.link_resource_count;
                 ++r) {
                const AttributeNumber& number = data.link_demand_values[
                    e * data.link_resource_count + r];
                std::int64_t integral = 0;
                if (const auto* value = std::get_if<std::int64_t>(&number)) {
                    integral = *value;
                } else if (const auto* value = std::get_if<bool>(&number)) {
                    integral = *value ? 1 : 0;
                }
                if (integral > 0) {
                    integral_gcd = std::gcd(integral_gcd, integral);
                }
            }
            if (integral_gcd > 0) {
                // Preserve integral journals in every integer resource lane.
                // With one lane this is exactly Python's bandwidth flow.
                data.link_flow_units[e] =
                    static_cast<double>(integral_gcd);
            } else {
                for (std::size_t r = 0U;
                     r < data.link_resource_count;
                     ++r) {
                    const double demand = data.link_demand(e, r);
                    if (demand > kEpsilon) {
                        data.link_flow_units[e] = demand;
                        data.link_flow_integral[e] = 0U;
                        break;
                    }
                }
            }
            const double units = data.link_flow_unit(e);
            if (units > kEpsilon) {
                for (std::size_t r = 0U;
                     r < data.link_resource_count;
                     ++r) {
                    data.link_resource_per_flow_unit[
                        e * data.link_resource_count + r] =
                        data.link_demand(e, r) / units;
                }
            }
        }
    }

    const auto physical_graph_edges = physical_network.graph().edges();
    data.physical_edges.reserve(physical_network.num_edges());
    for (auto iterator = physical_graph_edges.first;
         iterator != physical_graph_edges.second;
         ++iterator) {
        const Edge edge = *iterator;
        data.physical_edges.push_back(PhysicalEdge{
            physical_network.graph().source(edge),
            physical_network.graph().target(edge),
            physical_network.graph().edge_id(edge)});
    }
    data.physical_edge_count = data.physical_edges.size();
    data.link_capacities.resize(
        data.physical_edge_count * data.link_resource_count);
    data.link_capacity_values.resize(
        data.physical_edge_count * data.link_resource_count);
    for (std::size_t e = 0U; e < data.physical_edge_count; ++e) {
        const auto& physical_edge = data.physical_edges[e];
        const Edge edge = physical_network.graph().edge(
            physical_edge.source, physical_edge.target);
        for (std::size_t r = 0U; r < data.link_resource_count; ++r) {
            const AttrMap& values = physical_network.graph().edge_attrs(edge);
            const AttrValue* value = values.find(
                data.link_resources[r].physical_value_id);
            if (value == nullptr) {
                throw std::invalid_argument(
                    "exact solver missing physical link resource");
            }
            const AttributeNumber number = attribute_number(*value);
            data.link_capacity_values[e * data.link_resource_count + r] =
                number;
            data.coefficients_exact_in_double =
                data.coefficients_exact_in_double &&
                coefficient_is_exact_in_double(number);
            data.link_capacities[e * data.link_resource_count + r] =
                required_numeric(*value, "physical link");
        }
    }

    data.physical_arcs.reserve(data.physical_edge_count * 2U);
    data.outgoing_arcs.resize(data.physical_node_count);
    data.incoming_arcs.resize(data.physical_node_count);
    for (std::size_t edge_index = 0U;
         edge_index < data.physical_edge_count;
         ++edge_index) {
        const auto& edge = data.physical_edges[edge_index];
        const std::size_t first = data.physical_arcs.size();
        data.physical_arcs.push_back(
            PhysicalArc{edge.source, edge.target, edge_index});
        data.physical_arcs.push_back(
            PhysicalArc{edge.target, edge.source, edge_index});
        data.outgoing_arcs[edge.source].push_back(first);
        data.outgoing_arcs[edge.target].push_back(first + 1U);
        data.incoming_arcs[edge.target].push_back(first);
        data.incoming_arcs[edge.source].push_back(first + 1U);
    }
    for (auto& arcs : data.outgoing_arcs) {
        std::sort(
            arcs.begin(),
            arcs.end(),
            [&data](std::size_t lhs, std::size_t rhs) {
                const auto& left = data.physical_arcs[lhs];
                const auto& right = data.physical_arcs[rhs];
                if (left.target != right.target) {
                    return left.target < right.target;
                }
                return lhs < rhs;
            });
    }
    return data;
}

struct SolvedModel {
    core::Solution solution;
    bool feasible = false;
};

bool supported_constraints(
    const core::controller::ControllerSelection& selection) {
    const auto is_node_resource = [&selection](
        const core::controller::ConstraintId id) {
        return std::find(
                   selection.node_resources.begin(),
                   selection.node_resources.end(),
                   id) != selection.node_resources.end();
    };
    const auto is_link_resource = [&selection](
        const core::controller::ConstraintId id) {
        return std::find(
                   selection.link_resources.begin(),
                   selection.link_resources.end(),
                   id) != selection.link_resources.end();
    };
    for (const core::controller::ConstraintId id :
         selection.constraints.node_at_node) {
        if (!is_node_resource(id)) {
            return false;
        }
    }
    for (const core::controller::ConstraintId id :
         selection.hard_node_constraints) {
        if (!is_node_resource(id)) {
            return false;
        }
    }
    for (const core::controller::ConstraintId id :
         selection.constraints.link_at_link) {
        if (!is_link_resource(id)) {
            return false;
        }
    }
    // Resource rows are edge-capacity rows. A resource selected at path level
    // has different Controller semantics and cannot be represented by this
    // formulation without silently changing the constraint domain.
    if (!selection.constraints.link_at_path.empty()) {
        return false;
    }
    for (const core::controller::ConstraintId id :
         selection.hard_link_constraints) {
        if (!is_link_resource(id)) {
            return false;
        }
    }
    return selection.constraints.graph.empty();
}

struct SolverBackend {
    std::unique_ptr<operations_research::MPSolver> solver;
    bool scip = false;
};

SolverBackend create_solver(ExactAlgorithm algorithm) {
    if (algorithm != ExactAlgorithm::mixed_integer) {
        return SolverBackend{
            std::unique_ptr<operations_research::MPSolver>(
                operations_research::MPSolver::CreateSolver("GLOP")),
            false};
    }

    SolverBackend result;
    result.solver.reset(
        operations_research::MPSolver::CreateSolver("SCIP"));
    result.scip = result.solver != nullptr;
    if (!result.solver) {
        result.solver.reset(
            operations_research::MPSolver::CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING"));
    }
    if (!result.solver) {
        result.solver.reset(
            operations_research::MPSolver::CreateSolver("SAT_INTEGER_PROGRAMMING"));
    }
    return result;
}

SolvedModel solve_model(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const core::controller::Controller& controller,
    const SolverConfig& config,
    ExactAlgorithm algorithm,
    const ExactSolverParameters& parameters,
    PyRandom* random) {
    core::Solution solution = core::Solution::from_v_net(virtual_network);
    const auto& selection = controller.selection();
    if (!supported_constraints(selection)) {
        // Exact rows are resource-only. Fail closed rather than silently
        // accepting an embedding that omitted position, latency, path or
        // graph constraints from the optimization model.
        solution.description =
            "exact solver supports resource constraints only";
        return SolvedModel{std::move(solution), false};
    }
    ExactModelData data = prepare_model(
        virtual_network, physical_network, selection);
    if (std::any_of(
            data.virtual_edges.begin(), data.virtual_edges.end(),
            [](const VertexPair& edge) { return edge.first == edge.second; }) ||
        std::any_of(
            data.physical_edges.begin(), data.physical_edges.end(),
            [](const PhysicalEdge& edge) {
                return edge.source == edge.target;
            })) {
        solution.description = "exact solver does not support self-loops";
        return SolvedModel{std::move(solution), false};
    }
    for (const double demand : data.link_demands) {
        if (demand > 0.0 && demand <= kEpsilon) {
            solution.description =
                "exact solver link demand is below numeric tolerance";
            return SolvedModel{std::move(solution), false};
        }
    }
    if (std::any_of(
            data.link_resource_per_flow_unit.begin(),
            data.link_resource_per_flow_unit.end(),
            [](const double value) { return !std::isfinite(value); })) {
        solution.description =
            "exact solver link resource ratio is outside numeric range";
        return SolvedModel{std::move(solution), false};
    }
    if (!data.coefficients_exact_in_double) {
        // MPSolver exposes double coefficients even for integer backends.
        // Refuse values that cannot be represented exactly instead of
        // accepting 2^53+1 as if it were 2^53.
        solution.description =
            "exact solver resource exceeds exact MPSolver coefficient range";
        return SolvedModel{std::move(solution), false};
    }
    std::vector<double> incident_flow_units(
        data.virtual_node_count, 0.0);
    for (std::size_t e = 0U; e < data.virtual_edge_count; ++e) {
        const auto [source, target] = data.virtual_edges[e];
        incident_flow_units[source] += data.link_flow_unit(e);
        incident_flow_units[target] += data.link_flow_unit(e);
    }
    if (std::any_of(
            incident_flow_units.begin(),
            incident_flow_units.end(),
            [](const double value) {
                return value > kMetaBandwidth + kEpsilon;
            })) {
        solution.description =
            "exact solver virtual attachment exceeds META_BW";
        return SolvedModel{std::move(solution), false};
    }
    SolverBackend backend = create_solver(algorithm);
    auto& solver = backend.solver;
    if (!solver) {
        return SolvedModel{std::move(solution), false};
    }
    if (parameters.workers == 0U ||
        parameters.workers > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        solution.description = "exact solver worker count is out of range";
        return SolvedModel{std::move(solution), false};
    }
    if (parameters.workers > 1U &&
        algorithm == ExactAlgorithm::mixed_integer &&
        !solver->SetNumThreads(
                    static_cast<int>(parameters.workers))
              .ok()) {
        solution.description =
            "exact solver backend rejected the worker count";
        return SolvedModel{std::move(solution), false};
    }
    solver->set_time_limit(static_cast<std::int64_t>(std::min<std::uint64_t>(
        parameters.time_limit_ms,
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()))));
    if (backend.scip && parameters.search_node_limit != 0U) {
        solver->SetSolverSpecificParametersAsString(
            "limits/nodes = " +
            std::to_string(parameters.search_node_limit));
    }
    const bool integral = algorithm == ExactAlgorithm::mixed_integer;

    using operations_research::MPConstraint;
    using operations_research::MPVariable;
    const std::size_t virtual_nodes = data.virtual_node_count;
    const std::size_t physical_nodes = data.physical_node_count;
    const std::size_t virtual_edges = data.virtual_edge_count;
    const std::size_t physical_arcs = data.physical_arcs.size();

    const auto placement_index = [physical_nodes](
        const std::size_t v,
        const std::size_t p) noexcept {
        return v * physical_nodes + p;
    };
    const auto path_index = [physical_arcs](
        const std::size_t e,
        const std::size_t arc) noexcept {
        return e * physical_arcs + arc;
    };
    const auto attachment_index = [physical_nodes](
        const std::size_t e,
        const std::size_t p) noexcept {
        return e * physical_nodes + p;
    };
    const auto path_upper_index = [physical_edge_count =
                                       data.physical_edge_count](
                                      const std::size_t e,
                                      const std::size_t edge) noexcept {
        return e * physical_edge_count + edge;
    };

    std::vector<double> path_upper_bounds(
        virtual_edges * data.physical_edge_count, 0.0);
    for (std::size_t e = 0U; e < virtual_edges; ++e) {
        const bool path_integral =
            integral && data.has_integral_link_flow(e);
        for (std::size_t edge = 0U;
             edge < data.physical_edge_count;
             ++edge) {
            double upper = std::min(
                kMetaBandwidth, data.link_flow_unit(e));
            for (std::size_t r = 0U;
                 r < data.link_resource_count;
                 ++r) {
                const double ratio = data.link_resource_ratio(e, r);
                if (ratio > 0.0) {
                    upper = std::min(
                        upper, data.link_capacity(edge, r) / ratio);
                }
            }
            upper = std::max(0.0, upper);
            if (path_integral) {
                upper = std::floor(upper + kEpsilon);
            }
            path_upper_bounds[path_upper_index(e, edge)] = upper;
        }
    }

    std::vector<MPVariable*> placement(
        virtual_nodes * physical_nodes, nullptr);
    for (std::size_t v = 0U; v < virtual_nodes; ++v) {
        for (std::size_t p = 0U; p < physical_nodes; ++p) {
            bool candidate = true;
            for (std::size_t r = 0U; r < data.node_resource_count; ++r) {
                if (data.node_demand(v, r) > data.node_capacity(p, r) + kEpsilon) {
                    candidate = false;
                    break;
                }
            }
            placement[placement_index(v, p)] = integral ? solver->MakeIntVar(
                0.0,
                candidate ? 1.0 : 0.0,
                "")
                : solver->MakeNumVar(
                    0.0,
                    candidate ? 1.0 : 0.0,
                    "");
        }
    }

    std::vector<MPVariable*> path_use(
        virtual_edges * physical_arcs, nullptr);
    for (std::size_t e = 0U; e < virtual_edges; ++e) {
        const bool path_integral =
            integral && data.has_integral_link_flow(e);
        for (std::size_t arc = 0U; arc < physical_arcs; ++arc) {
            const std::size_t edge = data.physical_arcs[arc].edge_index;
            const double upper =
                path_upper_bounds[path_upper_index(e, edge)];
            path_use[path_index(e, arc)] = path_integral
                ? solver->MakeIntVar(
                    0.0, upper, "")
                : solver->MakeNumVar(
                    0.0, upper, "");
        }
    }

    // Python's LP rounding score uses the flow on each virtual meta-node to
    // physical-node attachment. Keep those variables only for GLOP; integral
    // placement makes them exactly flow_units * placement and can eliminate
    // them without changing MIP semantics.
    std::vector<MPVariable*> source_attachment;
    std::vector<MPVariable*> target_attachment;
    if (!integral) {
        source_attachment.resize(virtual_edges * physical_nodes, nullptr);
        target_attachment.resize(virtual_edges * physical_nodes, nullptr);
        for (std::size_t e = 0U; e < virtual_edges; ++e) {
            const double upper = std::min(
                kMetaBandwidth, data.link_flow_unit(e));
            for (std::size_t p = 0U; p < physical_nodes; ++p) {
                const std::size_t index = attachment_index(e, p);
                source_attachment[index] = solver->MakeNumVar(
                    0.0, upper, "");
                target_attachment[index] = solver->MakeNumVar(
                    0.0, upper, "");
            }
        }
    }

    for (std::size_t v = 0U; v < virtual_nodes; ++v) {
        MPConstraint* assignment = solver->MakeRowConstraint(1.0, 1.0);
        for (std::size_t p = 0U; p < physical_nodes; ++p) {
            assignment->SetCoefficient(
                placement[placement_index(v, p)], 1.0);
        }
    }
    // Python's rounding solvers always enforce one virtual node per physical
    // node. Integral MIP alone honors the native reusable extension.
    if (!selection.reusable || !integral) {
        for (std::size_t p = 0U; p < physical_nodes; ++p) {
            MPConstraint* exclusive = solver->MakeRowConstraint(
                -kInfinity, 1.0);
            for (std::size_t v = 0U; v < virtual_nodes; ++v) {
                exclusive->SetCoefficient(
                    placement[placement_index(v, p)], 1.0);
            }
        }
    }
    for (std::size_t p = 0U; p < physical_nodes; ++p) {
        for (std::size_t r = 0U; r < data.node_resource_count; ++r) {
            MPConstraint* capacity = solver->MakeRowConstraint(
                -kInfinity, data.node_capacity(p, r));
            for (std::size_t v = 0U; v < virtual_nodes; ++v) {
                capacity->SetCoefficient(
                    placement[placement_index(v, p)],
                    data.node_demand(v, r));
            }
        }
    }

    for (std::size_t e = 0U; e < virtual_edges; ++e) {
        const auto [source, target] = data.virtual_edges[e];
        const double flow_units = data.link_flow_unit(e);
        if (!integral) {
            MPConstraint* source_total = solver->MakeRowConstraint(
                flow_units, flow_units);
            MPConstraint* target_total = solver->MakeRowConstraint(
                flow_units, flow_units);
            for (std::size_t p = 0U; p < physical_nodes; ++p) {
                const std::size_t index = attachment_index(e, p);
                source_total->SetCoefficient(source_attachment[index], 1.0);
                target_total->SetCoefficient(target_attachment[index], 1.0);
            }
        }
        for (std::size_t p = 0U; p < physical_nodes; ++p) {
            MPConstraint* conservation = solver->MakeRowConstraint(0.0, 0.0);
            for (const std::size_t arc : data.outgoing_arcs[p]) {
                conservation->SetCoefficient(
                    path_use[path_index(e, arc)], 1.0);
            }
            for (const std::size_t arc : data.incoming_arcs[p]) {
                conservation->SetCoefficient(
                    path_use[path_index(e, arc)], -1.0);
            }
            if (integral) {
                conservation->SetCoefficient(
                    placement[placement_index(source, p)], -flow_units);
                conservation->SetCoefficient(
                    placement[placement_index(target, p)], flow_units);
            } else {
                const std::size_t index = attachment_index(e, p);
                conservation->SetCoefficient(
                    source_attachment[index], -1.0);
                conservation->SetCoefficient(
                    target_attachment[index], 1.0);
            }
        }
    }
    if (!integral) {
        for (std::size_t v = 0U; v < virtual_nodes; ++v) {
            for (std::size_t p = 0U; p < physical_nodes; ++p) {
                MPConstraint* attachment_capacity =
                    solver->MakeRowConstraint(-kInfinity, 0.0);
                attachment_capacity->SetCoefficient(
                    placement[placement_index(v, p)],
                    -kMetaBandwidth);
                for (std::size_t e = 0U; e < virtual_edges; ++e) {
                    const auto [source, target] = data.virtual_edges[e];
                    const std::size_t index = attachment_index(e, p);
                    if (source == v) {
                        attachment_capacity->SetCoefficient(
                            source_attachment[index], 1.0);
                    }
                    if (target == v) {
                        attachment_capacity->SetCoefficient(
                            target_attachment[index], 1.0);
                    }
                }
            }
        }
    }
    for (std::size_t edge = 0U;
         edge < data.physical_edge_count;
         ++edge) {
        const std::size_t forward = edge * 2U;
        const std::size_t reverse = forward + 1U;
        for (std::size_t r = 0U; r < data.link_resource_count; ++r) {
            MPConstraint* capacity = solver->MakeRowConstraint(
                -kInfinity, data.link_capacity(edge, r));
            for (std::size_t e = 0U; e < virtual_edges; ++e) {
                const double ratio = data.link_resource_ratio(e, r);
                capacity->SetCoefficient(
                    path_use[path_index(e, forward)], ratio);
                capacity->SetCoefficient(
                    path_use[path_index(e, reverse)], ratio);
            }
        }
    }

    auto* objective = solver->MutableObjective();
    objective->SetMinimization();
    for (std::size_t v = 0U; v < virtual_nodes; ++v) {
        for (std::size_t p = 0U; p < physical_nodes; ++p) {
            double node_weight = 0.0;
            for (std::size_t r = 0U;
                 r < data.node_resource_count;
                 ++r) {
                node_weight += integral
                    ? data.node_demand(v, r)
                    : data.node_demand(v, r) /
                          (data.node_capacity(p, r) + 1.0e-6);
            }
            objective->SetCoefficient(
                placement[placement_index(v, p)], node_weight);
        }
    }
    for (std::size_t e = 0U; e < virtual_edges; ++e) {
        for (std::size_t arc = 0U; arc < physical_arcs; ++arc) {
            const std::size_t edge = data.physical_arcs[arc].edge_index;
            double link_weight = 1.0;
            if (!integral && data.link_resource_count != 0U) {
                link_weight = 0.0;
                for (std::size_t r = 0U;
                     r < data.link_resource_count;
                     ++r) {
                    link_weight += data.link_resource_ratio(e, r) /
                        (data.link_capacity(edge, r) + 1.0e-6);
                }
            }
            objective->SetCoefficient(
                path_use[path_index(e, arc)], link_weight);
        }
    }

    const auto status = solver->Solve();
    if (status != operations_research::MPSolver::OPTIMAL &&
        status != operations_research::MPSolver::FEASIBLE) {
        return SolvedModel{std::move(solution), false};
    }

    if (!integral) {
        core::NodeSlots rounded_slots;
        std::vector<double> residual = data.node_capacities;
        std::vector<std::uint8_t> used(physical_nodes, 0U);
        std::vector<Vertex> candidates;
        std::vector<double> weights;
        candidates.reserve(physical_nodes);
        weights.reserve(physical_nodes);

        for (std::size_t v = 0U; v < virtual_nodes; ++v) {
            candidates.clear();
            weights.clear();
            for (std::size_t p = 0U; p < physical_nodes; ++p) {
                // Python's rounding solvers never reuse a physical node.
                // Preserve that public behavior even if a caller supplied a
                // reusable controller, and validate every resource lane.
                if (used[p] != 0U) {
                    continue;
                }
                bool fits = true;
                for (std::size_t r = 0U;
                     r < data.node_resource_count;
                     ++r) {
                    if (data.node_demand(v, r) >
                        residual[p * data.node_resource_count + r] +
                            kEpsilon) {
                        fits = false;
                        break;
                    }
                }
                if (!fits) {
                    continue;
                }
                candidates.push_back(static_cast<Vertex>(p));
                double attached_flow = 0.0;
                for (std::size_t e = 0U; e < virtual_edges; ++e) {
                    const auto [source, target] = data.virtual_edges[e];
                    const std::size_t index = attachment_index(e, p);
                    if (source == v) {
                        attached_flow +=
                            source_attachment[index]->solution_value();
                    }
                    if (target == v) {
                        attached_flow +=
                            target_attachment[index]->solution_value();
                    }
                }
                weights.push_back(std::max(
                    0.0,
                    placement[placement_index(v, p)]->solution_value() *
                        attached_flow));
            }
            if (candidates.empty()) {
                solution.place_result = false;
                return SolvedModel{std::move(solution), false};
            }

            std::size_t selected_index = 0U;
            if (algorithm == ExactAlgorithm::deterministic_rounding) {
                for (std::size_t index = 1U;
                     index < weights.size();
                     ++index) {
                    if (weights[index] > weights[selected_index]) {
                        selected_index = index;
                    }
                }
            } else {
                if (random == nullptr) {
                    throw std::invalid_argument(
                        "randomized exact rounding requires PyRandom");
                }
                const double total = std::accumulate(
                    weights.begin(), weights.end(), 0.0);
                const Vertex selected = total == 0.0
                    ? random->choices(candidates, 1U).front()
                    : random->choices_weights(
                          candidates, weights, 1U).front();
                selected_index = static_cast<std::size_t>(
                    std::find(candidates.begin(), candidates.end(), selected) -
                    candidates.begin());
            }

            const std::size_t selected = candidates[selected_index];
            used[selected] = 1U;
            for (std::size_t r = 0U;
                 r < data.node_resource_count;
                 ++r) {
                residual[selected * data.node_resource_count + r] -=
                    data.node_demand(v, r);
            }
            rounded_slots.insert_or_assign(
                static_cast<core::SolutionNodeId>(v),
                static_cast<core::SolutionNodeId>(selected));
        }

        network::PhysicalNetwork physical_copy = physical_network.clone();
        auto prepared = controller.prepare(virtual_network, physical_copy);
        core::controller::DeployWithNodeSlotsOptions options;
        options.shortest_method = config.shortest_method;
        options.k = config.k_shortest;
        if (!prepared.deploy_with_node_slots(
                rounded_slots, solution, options)) {
            return SolvedModel{std::move(solution), false};
        }
        return SolvedModel{std::move(solution), true};
    }

    std::vector<Vertex> selected_physical_nodes(virtual_nodes, 0U);
    std::vector<AttributeNumber> exact_node_residual =
        data.node_capacity_values;
    for (std::size_t v = 0U; v < virtual_nodes; ++v) {
        bool found = false;
        for (std::size_t p = 0U; p < physical_nodes; ++p) {
            if (placement[placement_index(v, p)]->solution_value() > 0.5) {
                selected_physical_nodes[v] = static_cast<Vertex>(p);
                found = true;
                break;
            }
        }
        if (!found) {
            solution.place_result = false;
            return SolvedModel{std::move(solution), false};
        }
        solution.node_slots.insert_or_assign(
            static_cast<core::SolutionNodeId>(v),
            static_cast<core::SolutionNodeId>(selected_physical_nodes[v]));
        core::SolutionAttributeValues values;
        for (std::size_t r = 0U; r < data.node_resource_count; ++r) {
            const AttributeNumber& demand = data.node_demand_values[
                v * data.node_resource_count + r];
            AttributeNumber& residual = exact_node_residual[
                static_cast<std::size_t>(selected_physical_nodes[v]) *
                    data.node_resource_count + r];
            if (!subtract_resource_exactly(demand, residual)) {
                solution.reset();
                solution.place_result = false;
                solution.description =
                    "exact solver placement failed exact resource validation";
                return SolvedModel{std::move(solution), false};
            }
            values.set(
                data.node_resources[r].solution_id,
                demand);
        }
        solution.node_slots_info.insert_or_assign(
            core::NodeSlotInfoKey{
                static_cast<core::SolutionNodeId>(v),
                static_cast<core::SolutionNodeId>(selected_physical_nodes[v])},
            std::move(values));
    }

    std::vector<AttributeNumber> exact_link_residual =
        data.link_capacity_values;
    for (std::size_t e = 0U; e < virtual_edges; ++e) {
        const auto [source, target] = data.virtual_edges[e];
        std::vector<core::SolutionLink> path;
        path.reserve(std::min(physical_arcs, physical_nodes));
        for (std::size_t arc = 0U; arc < physical_arcs; ++arc) {
            const double used_flow = std::round(
                path_use[path_index(e, arc)]->solution_value());
            if (used_flow <= kEpsilon) {
                continue;
            }
            const auto& physical_arc = data.physical_arcs[arc];
            const core::SolutionLink physical_link{
                static_cast<core::SolutionNodeId>(physical_arc.source),
                static_cast<core::SolutionNodeId>(physical_arc.target)};
            path.push_back(physical_link);
            core::SolutionAttributeValues values;
            for (std::size_t r = 0U; r < data.link_resource_count; ++r) {
                const AttributeNumber amount = scaled_resource_amount(
                    data.link_demand_values[
                        e * data.link_resource_count + r],
                    data.link_demand(e, r),
                    used_flow,
                    data.link_flow_unit(e));
                AttributeNumber& residual = exact_link_residual[
                    physical_arc.edge_index * data.link_resource_count + r];
                if (!subtract_resource_exactly(amount, residual)) {
                    solution.reset();
                    solution.route_result = false;
                    solution.description =
                        "exact solver route failed exact resource validation";
                    return SolvedModel{std::move(solution), false};
                }
                values.set(
                    data.link_resources[r].solution_id,
                    amount);
            }
            solution.link_paths_info.insert_or_assign(
                core::LinkPathInfoKey{
                    core::SolutionLink{
                        static_cast<core::SolutionNodeId>(source),
                        static_cast<core::SolutionNodeId>(target)},
                    physical_link},
                std::move(values));
        }
        solution.link_paths.insert_or_assign(
            core::SolutionLink{
                static_cast<core::SolutionNodeId>(source),
                static_cast<core::SolutionNodeId>(target)},
            std::move(path));
    }
    solution.place_result = true;
    solution.route_result = true;
    solution.result = true;
    return SolvedModel{std::move(solution), true};
}

}  // namespace

ExactSolver::ExactSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    ExactAlgorithm algorithm,
    PyRandom& random,
    ExactSolverParameters parameters)
    : Solver(std::move(dependencies), std::move(config)),
      algorithm_(algorithm),
      parameters_(parameters),
      random_(&random) {}

ExactAlgorithm ExactSolver::algorithm() const noexcept {
    return algorithm_;
}

const ExactSolverParameters& ExactSolver::parameters() const noexcept {
    return parameters_;
}

core::Solution ExactSolver::solve(const SolverInstance& instance) {
    return solve_model(
               instance.virtual_network,
               instance.physical_network,
               controller(),
               config(),
               algorithm_,
               parameters_,
               random_)
        .solution;
}

MutableSolverResult ExactSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    SolvedModel solved = solve_model(
        instance.virtual_network,
        instance.physical_network,
        controller(),
        config(),
        algorithm_,
        parameters_,
        random_);
    if (!solved.feasible) {
        return MutableSolverResult{
            std::move(solved.solution), SolverMutationState::detached};
    }
    const bool owns_transaction = !instance.mutation.transaction_active();
    if (owns_transaction) {
        instance.mutation.begin_transaction();
    }
    try {
        if (!instance.mutation.deploy(solved.solution)) {
            if (owns_transaction && instance.mutation.transaction_active()) {
                instance.mutation.rollback_transaction();
            }
            return MutableSolverResult{
                std::move(solved.solution), SolverMutationState::detached};
        }
        if (owns_transaction) {
            instance.mutation.commit_transaction();
        }
    } catch (...) {
        // Only close a checkpoint created by this solver. A caller-owned
        // transaction may contain earlier work and remains active so its
        // owner can choose the transaction-wide recovery policy.
        if (owns_transaction && instance.mutation.transaction_active()) {
            instance.mutation.rollback_transaction();
        }
        throw;
    }
    return MutableSolverResult{
        std::move(solved.solution), SolverMutationState::committed};
}

ExactSolverIds register_exact_solvers(
    SolverRegistry& registry,
    PyRandom& random,
    ExactSolverParameters parameters) {
    PyRandom* const random_ptr = &random;
    ExactSolverIds ids{};
    ids.mip = registry.register_solver(
        "mip",
        SolverCategory::exact,
        [parameters, random_ptr](SolverDependencies dependencies,
                     SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<ExactSolver>(
                std::move(dependencies),
                std::move(config),
                ExactAlgorithm::mixed_integer,
                *random_ptr,
                parameters);
        });
    ids.d_round = registry.register_solver(
        "d_round",
        SolverCategory::rounding,
        [parameters, random_ptr](SolverDependencies dependencies,
                     SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<ExactSolver>(
                std::move(dependencies),
                std::move(config),
                ExactAlgorithm::deterministic_rounding,
                *random_ptr,
                parameters);
        });
    ids.r_round = registry.register_solver(
        "r_round",
        SolverCategory::rounding,
        [parameters, random_ptr](SolverDependencies dependencies,
                     SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<ExactSolver>(
                std::move(dependencies),
                std::move(config),
                ExactAlgorithm::randomized_rounding,
                *random_ptr,
                parameters);
        });
    return ids;
}

}  // namespace virne::solver::exact
