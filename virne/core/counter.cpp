#include "counter.h"

#include "../../csv/csv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

namespace virne::core {
namespace {

constexpr std::size_t numpy_pairwise_block_size = 128U;

std::int64_t signed_bits(std::uint64_t value) noexcept {
    std::int64_t result = 0;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double numpy_pairwise_sum(const double* values, std::size_t count) noexcept {
    if (count == 0U) {
        return 0.0;
    }

    if (count < 8U) {
        double result = -0.0;
        for (std::size_t index = 0U; index < count; ++index) {
            result += values[index];
        }
        return result;
    }

    if (count <= numpy_pairwise_block_size) {
        double r0 = values[0U];
        double r1 = values[1U];
        double r2 = values[2U];
        double r3 = values[3U];
        double r4 = values[4U];
        double r5 = values[5U];
        double r6 = values[6U];
        double r7 = values[7U];

        std::size_t index = 8U;
        for (; index + 7U < count; index += 8U) {
            r0 += values[index];
            r1 += values[index + 1U];
            r2 += values[index + 2U];
            r3 += values[index + 3U];
            r4 += values[index + 4U];
            r5 += values[index + 5U];
            r6 += values[index + 6U];
            r7 += values[index + 7U];
        }

        double result =
            ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; index < count; ++index) {
            result += values[index];
        }
        return result;
    }

    std::size_t left_count = count / 2U;
    left_count -= left_count % 8U;
    return numpy_pairwise_sum(values, left_count) +
        numpy_pairwise_sum(values + left_count, count - left_count);
}

double counter_number_to_double(const CounterNumber& value) noexcept {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return static_cast<double>(*integer);
    }
    return std::get<double>(value);
}

CounterNumber add_numpy_numbers(
    const CounterNumber& left,
    const CounterNumber& right) noexcept {
    const auto* left_integer = std::get_if<std::int64_t>(&left);
    const auto* right_integer = std::get_if<std::int64_t>(&right);
    if (left_integer != nullptr && right_integer != nullptr) {
        const std::uint64_t bits =
            static_cast<std::uint64_t>(*left_integer) +
            static_cast<std::uint64_t>(*right_integer);
        return signed_bits(bits);
    }
    return counter_number_to_double(left) + counter_number_to_double(right);
}

[[noreturn]] void throw_numeric_overflow(
    CounterOperation operation,
    std::optional<CounterResourceId> resource_id = std::nullopt,
    std::optional<SolutionNodeId> virtual_node = std::nullopt,
    std::optional<SolutionLink> virtual_link = std::nullopt,
    std::optional<SolutionLink> physical_link = std::nullopt) {
    throw CounterException(
        CounterErrorCode::numeric_overflow,
        operation,
        "Counter integer accumulation exceeds the int64 domain",
        resource_id,
        virtual_node,
        virtual_link,
        physical_link);
}

CounterNumber add_checked_numbers(
    const CounterNumber& left,
    const CounterNumber& right,
    CounterOperation operation,
    std::optional<CounterResourceId> resource_id = std::nullopt,
    std::optional<SolutionNodeId> virtual_node = std::nullopt,
    std::optional<SolutionLink> virtual_link = std::nullopt,
    std::optional<SolutionLink> physical_link = std::nullopt) {
    const auto* left_integer = std::get_if<std::int64_t>(&left);
    const auto* right_integer = std::get_if<std::int64_t>(&right);
    if (left_integer == nullptr || right_integer == nullptr) {
        return counter_number_to_double(left) + counter_number_to_double(right);
    }

    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((*right_integer > 0 && *left_integer > maximum - *right_integer) ||
        (*right_integer < 0 && *left_integer < minimum - *right_integer)) {
        throw_numeric_overflow(
            operation,
            resource_id,
            virtual_node,
            virtual_link,
            physical_link);
    }
    return *left_integer + *right_integer;
}

CounterNumber graph_numeric_value(
    const AttrValue& value,
    CounterOperation operation,
    CounterResourceId resource_id,
    std::optional<SolutionNodeId> virtual_node = std::nullopt,
    std::optional<SolutionLink> virtual_link = std::nullopt) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return static_cast<std::int64_t>(*boolean);
    }
    throw CounterException(
        CounterErrorCode::non_numeric_resource_value,
        operation,
        "Counter resource value is not bool, int64, or double",
        resource_id,
        virtual_node,
        virtual_link);
}

CounterNumber solution_numeric_value(
    const network::attribute::AttributeNumber& value) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    return static_cast<std::int64_t>(std::get<bool>(value));
}

Vertex checked_virtual_node(
    const Graph& graph,
    SolutionNodeId node,
    CounterOperation operation) {
    if (node < 0) {
        throw CounterException(
            CounterErrorCode::invalid_solution_node,
            operation,
            "Solution virtual node is negative",
            std::nullopt,
            node);
    }

    using UnsignedNode = std::make_unsigned_t<SolutionNodeId>;
    const auto unsigned_node = static_cast<UnsignedNode>(node);
    if (unsigned_node >
            static_cast<UnsignedNode>(std::numeric_limits<Vertex>::max()) ||
        static_cast<std::size_t>(unsigned_node) >= graph.num_nodes()) {
        throw CounterException(
            CounterErrorCode::invalid_solution_node,
            operation,
            "Solution virtual node is outside the prepared graph",
            std::nullopt,
            node);
    }
    return static_cast<Vertex>(unsigned_node);
}

std::pair<Vertex, Vertex> checked_virtual_link(
    const Graph& graph,
    const SolutionLink& link,
    CounterOperation operation) {
    if (link.source < 0 || link.target < 0) {
        throw CounterException(
            CounterErrorCode::invalid_solution_link,
            operation,
            "Solution virtual link has a negative endpoint",
            std::nullopt,
            std::nullopt,
            link);
    }

    using UnsignedNode = std::make_unsigned_t<SolutionNodeId>;
    const auto source = static_cast<UnsignedNode>(link.source);
    const auto target = static_cast<UnsignedNode>(link.target);
    const auto maximum_vertex =
        static_cast<UnsignedNode>(std::numeric_limits<Vertex>::max());
    if (source > maximum_vertex || target > maximum_vertex ||
        static_cast<std::size_t>(source) >= graph.num_nodes() ||
        static_cast<std::size_t>(target) >= graph.num_nodes()) {
        throw CounterException(
            CounterErrorCode::invalid_solution_link,
            operation,
            "Solution virtual link endpoint is outside the prepared graph",
            std::nullopt,
            std::nullopt,
            link);
    }

    const Vertex source_vertex = static_cast<Vertex>(source);
    const Vertex target_vertex = static_cast<Vertex>(target);
    if (!graph.has_edge(source_vertex, target_vertex)) {
        throw CounterException(
            CounterErrorCode::invalid_solution_link,
            operation,
            "Solution virtual link is absent from the prepared graph",
            std::nullopt,
            std::nullopt,
            link);
    }
    return {source_vertex, target_vertex};
}

struct CounterResourceScan {
    std::uint64_t integer_bits = 0U;
    bool missing = false;
    bool nonnumeric = false;
    bool has_double = false;
};

template <typename Function>
void run_counter_tasks(
    std::size_t task_count,
    std::size_t requested_workers,
    Function&& function) {
    if (task_count <= 1U || requested_workers <= 1U) {
        function(0U, task_count);
        return;
    }

    const std::size_t worker_count =
        std::min(task_count, requested_workers);
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(worker_count);
    threads.reserve(worker_count - 1U);

    const auto bounds = [task_count, worker_count](std::size_t worker) {
        const std::size_t block = task_count / worker_count;
        const std::size_t remainder = task_count % worker_count;
        const std::size_t begin =
            worker * block + std::min(worker, remainder);
        const std::size_t end =
            begin + block + (worker < remainder ? 1U : 0U);
        return std::pair<std::size_t, std::size_t>{begin, end};
    };

    try {
        for (std::size_t worker = 1U; worker < worker_count; ++worker) {
            const auto range = bounds(worker);
            const std::size_t begin = range.first;
            const std::size_t end = range.second;
            threads.emplace_back([&, worker, begin, end]() noexcept {
                try {
                    function(begin, end);
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (std::thread& thread : threads) {
            thread.join();
        }
        throw;
    }

    const auto [begin, end] = bounds(0U);
    try {
        function(begin, end);
    } catch (...) {
        errors[0U] = std::current_exception();
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
}

double mean_skip_nan(const std::vector<double>& values) {
    std::vector<double> valid;
    valid.reserve(values.size());
    for (const double value : values) {
        if (!std::isnan(value)) {
            valid.push_back(value);
        }
    }
    if (valid.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return numpy_pairwise_sum(valid.data(), valid.size()) /
        static_cast<double>(valid.size());
}

double sum_skip_nan(const std::vector<double>& values) {
    std::vector<double> valid;
    valid.reserve(values.size());
    for (const double value : values) {
        if (!std::isnan(value)) {
            valid.push_back(value);
        }
    }
    return valid.empty()
        ? 0.0
        : numpy_pairwise_sum(valid.data(), valid.size());
}

double minimum_skip_nan(const std::vector<double>& values) {
    bool present = false;
    double result = std::numeric_limits<double>::quiet_NaN();
    for (const double value : values) {
        if (std::isnan(value)) {
            continue;
        }
        result = present ? std::fmin(result, value) : value;
        present = true;
    }
    return present ? result : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

CounterException::CounterException(
    CounterErrorCode code,
    CounterOperation operation,
    std::string message,
    std::optional<CounterResourceId> resource_id,
    std::optional<SolutionNodeId> virtual_node,
    std::optional<SolutionLink> virtual_link,
    std::optional<SolutionLink> physical_link,
    std::optional<std::size_t> row_index)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      resource_id_(resource_id),
      virtual_node_(virtual_node),
      virtual_link_(virtual_link),
      physical_link_(physical_link),
      row_index_(row_index) {}

CounterErrorCode CounterException::code() const noexcept { return code_; }

CounterOperation CounterException::operation() const noexcept {
    return operation_;
}

const std::optional<CounterResourceId>& CounterException::resource_id()
    const noexcept {
    return resource_id_;
}

const std::optional<SolutionNodeId>& CounterException::virtual_node()
    const noexcept {
    return virtual_node_;
}

const std::optional<SolutionLink>& CounterException::virtual_link()
    const noexcept {
    return virtual_link_;
}

const std::optional<SolutionLink>& CounterException::physical_link()
    const noexcept {
    return physical_link_;
}

const std::optional<std::size_t>& CounterException::row_index()
    const noexcept {
    return row_index_;
}

Counter::Counter(CounterSelection selection)
    : selection_(std::move(selection)) {}

const CounterSelection& Counter::selection() const noexcept {
    return selection_;
}

PreparedCounter Counter::prepare(const network::BaseNetwork& network) const {
    return prepare_impl(network, nullptr);
}

PreparedCounter Counter::prepare(
    const network::VirtualNetwork& virtual_network) const {
    return prepare_impl(virtual_network, &virtual_network);
}

PreparedCounter Counter::prepare_impl(
    const network::BaseNetwork& network,
    const network::VirtualNetwork* virtual_network) const {
    network::AttributeSelection all_resources;
    all_resources.kinds = std::vector<network::attribute::AttributeKind>{
        network::attribute::AttributeKind::resource};

    const std::vector<CounterResourceId> node_ids =
        selection_.node_resources.has_value()
        ? *selection_.node_resources
        : network.select_node_attributes(all_resources);
    const std::vector<CounterResourceId> link_ids =
        selection_.link_resources.has_value()
        ? *selection_.link_resources
        : network.select_link_attributes(all_resources);

    const auto& node_registry = network.node_attributes();
    const auto& link_registry = network.link_attributes();
    const Graph& graph = network.graph();

    std::vector<std::optional<network::NodeNetworkAttributeBinding>>
        node_cache(node_registry.size());
    std::vector<PreparedCounter::ResourceBinding> node_resources;
    node_resources.reserve(node_ids.size());
    for (const CounterResourceId id : node_ids) {
        if (static_cast<std::size_t>(id) >= node_registry.size()) {
            throw CounterException(
                CounterErrorCode::invalid_node_resource_selection,
                CounterOperation::prepare,
                "Counter node resource ID is outside the node registry",
                id);
        }
        const auto& definition = node_registry.at(id);
        if (definition.spec().kind !=
            network::attribute::AttributeKind::resource) {
            throw CounterException(
                CounterErrorCode::invalid_node_resource_selection,
                CounterOperation::prepare,
                "Counter node selection contains a non-resource definition",
                id);
        }

        if (!node_cache[id].has_value()) {
            try {
                const auto binding =
                    network.bind_node_attribute(definition.spec().name);
                if (!binding.has_value()) {
                    throw CounterException(
                        CounterErrorCode::graph_binding_mismatch,
                        CounterOperation::prepare,
                        "Counter node resource cannot bind to the graph",
                        id);
                }
                node_cache[id] = *binding;
            } catch (const CounterException&) {
                throw;
            } catch (...) {
                std::throw_with_nested(CounterException(
                    CounterErrorCode::graph_binding_mismatch,
                    CounterOperation::prepare,
                    "Counter node resource graph binding failed",
                    id));
            }
        }

        const auto& binding = *node_cache[id];
        if (binding.registry_id != id ||
            binding.registry_identity != &node_registry) {
            throw CounterException(
                CounterErrorCode::attribute_registry_mismatch,
                CounterOperation::prepare,
                "Counter node binding belongs to another registry",
                id);
        }
        if (binding.graph_identity != &graph) {
            throw CounterException(
                CounterErrorCode::graph_binding_mismatch,
                CounterOperation::prepare,
                "Counter node binding belongs to another graph",
                id);
        }
        node_resources.push_back({id, binding.value_id});
    }

    std::vector<std::optional<network::LinkNetworkAttributeBinding>>
        link_cache(link_registry.size());
    std::vector<PreparedCounter::ResourceBinding> link_resources;
    link_resources.reserve(link_ids.size());
    for (const CounterResourceId id : link_ids) {
        if (static_cast<std::size_t>(id) >= link_registry.size()) {
            throw CounterException(
                CounterErrorCode::invalid_link_resource_selection,
                CounterOperation::prepare,
                "Counter link resource ID is outside the link registry",
                id);
        }
        const auto& definition = link_registry.at(id);
        if (definition.spec().kind !=
            network::attribute::AttributeKind::resource) {
            throw CounterException(
                CounterErrorCode::invalid_link_resource_selection,
                CounterOperation::prepare,
                "Counter link selection contains a non-resource definition",
                id);
        }

        if (!link_cache[id].has_value()) {
            try {
                const auto binding =
                    network.bind_link_attribute(definition.spec().name);
                if (!binding.has_value()) {
                    throw CounterException(
                        CounterErrorCode::graph_binding_mismatch,
                        CounterOperation::prepare,
                        "Counter link resource cannot bind to the graph",
                        id);
                }
                link_cache[id] = *binding;
            } catch (const CounterException&) {
                throw;
            } catch (...) {
                std::throw_with_nested(CounterException(
                    CounterErrorCode::graph_binding_mismatch,
                    CounterOperation::prepare,
                    "Counter link resource graph binding failed",
                    id));
            }
        }

        const auto& binding = *link_cache[id];
        if (binding.registry_id != id ||
            binding.registry_identity != &link_registry) {
            throw CounterException(
                CounterErrorCode::attribute_registry_mismatch,
                CounterOperation::prepare,
                "Counter link binding belongs to another registry",
                id);
        }
        if (binding.graph_identity != &graph) {
            throw CounterException(
                CounterErrorCode::graph_binding_mismatch,
                CounterOperation::prepare,
                "Counter link binding belongs to another graph",
                id);
        }
        link_resources.push_back({id, binding.value_id});
    }

    return PreparedCounter(
        network,
        virtual_network,
        std::move(node_resources),
        std::move(link_resources),
        &node_registry,
        &link_registry,
        &graph,
        &graph.attribute_registry());
}

PreparedCounter::PreparedCounter(
    const network::BaseNetwork& network,
    const network::VirtualNetwork* virtual_network,
    std::vector<ResourceBinding> node_resources,
    std::vector<ResourceBinding> link_resources,
    const network::attribute::NodeAttributeRegistry* node_registry,
    const network::attribute::LinkAttributeRegistry* link_registry,
    const Graph* graph,
    const ::AttributeRegistry* graph_attribute_registry)
    : network_(&network),
      virtual_network_(virtual_network),
      node_resources_(std::move(node_resources)),
      link_resources_(std::move(link_resources)),
      node_registry_(node_registry),
      link_registry_(link_registry),
      graph_(graph),
      graph_attribute_registry_(graph_attribute_registry) {}

void PreparedCounter::validate_network_binding(
    CounterOperation operation) const {
    if (network_ == nullptr || graph_ == nullptr ||
        &network_->graph() != graph_ ||
        &network_->graph().attribute_registry() !=
            graph_attribute_registry_) {
        throw CounterException(
            CounterErrorCode::graph_binding_mismatch,
            operation,
            "Prepared Counter graph binding is no longer valid");
    }
    if (&network_->node_attributes() != node_registry_ ||
        &network_->link_attributes() != link_registry_) {
        throw CounterException(
            CounterErrorCode::attribute_registry_mismatch,
            operation,
            "Prepared Counter attribute registry is no longer valid");
    }
}

CounterNumber PreparedCounter::calculate_resource_sum(
    bool node,
    CounterOptions options) const {
    const auto& resources = node ? node_resources_ : link_resources_;
    const CounterOperation operation = node
        ? CounterOperation::sum_node_resources
        : CounterOperation::sum_link_resources;
    if (resources.empty()) {
        throw CounterException(
            node ? CounterErrorCode::empty_node_resource_selection
                 : CounterErrorCode::empty_link_resource_selection,
            operation,
            node ? "Counter node resource selection is empty"
                 : "Counter link resource selection is empty");
    }

    validate_network_binding(operation);
    const std::size_t width =
        node ? graph_->num_nodes() : graph_->num_edges();
    if (width == 0U) {
        // NumPy infers float64 for a rectangular nested array with zero
        // columns and returns its floating additive identity.
        return 0.0;
    }
    if (resources.size() >
        std::numeric_limits<std::size_t>::max() / width) {
        throw_numeric_overflow(operation);
    }

    const std::size_t value_count = resources.size() * width;
    const std::size_t worker_count = options.workers <= 1U
        ? 1U
        : std::min(options.workers, width);
    std::unique_ptr<double[]> flattened{new double[value_count]};
    std::vector<CounterResourceScan> scans(
        worker_count * resources.size());

    std::vector<Edge> ordered_edges;
    if (!node) {
        ordered_edges.reserve(width);
        const auto [edge, edge_end] = graph_->edges();
        for (auto current = edge; current != edge_end; ++current) {
            ordered_edges.push_back(*current);
        }
        if (ordered_edges.size() != width) {
            throw CounterException(
                CounterErrorCode::graph_binding_mismatch,
                operation,
                "Counter graph edge inventory changed during calculation");
        }
    }

    const auto consume_value = [&flattened, width](
                                   CounterResourceScan& scan,
                                   std::size_t row,
                                   std::size_t column,
                                   const AttrValue* value) noexcept {
        if (value == nullptr) {
            scan.missing = true;
            return;
        }
        const std::size_t output_index = row * width + column;
        if (const auto* floating = std::get_if<double>(value)) {
            flattened[output_index] = *floating;
            scan.has_double = true;
            return;
        }
        if (const auto* integer = std::get_if<std::int64_t>(value)) {
            flattened[output_index] = static_cast<double>(*integer);
            scan.integer_bits += static_cast<std::uint64_t>(*integer);
            return;
        }
        if (const auto* boolean = std::get_if<bool>(value)) {
            flattened[output_index] = *boolean ? 1.0 : 0.0;
            scan.integer_bits += static_cast<std::uint64_t>(*boolean);
            return;
        }
        scan.nonnumeric = true;
    };

    const auto column_bounds = [width, worker_count](std::size_t worker) {
        const std::size_t block = width / worker_count;
        const std::size_t remainder = width % worker_count;
        const std::size_t begin =
            worker * block + std::min(worker, remainder);
        const std::size_t end =
            begin + block + (worker < remainder ? 1U : 0U);
        return std::pair<std::size_t, std::size_t>{begin, end};
    };

    if (node) {
        run_counter_tasks(
            worker_count,
            worker_count,
            [&](std::size_t begin, std::size_t end) noexcept {
                for (std::size_t worker = begin; worker < end; ++worker) {
                    const auto columns = column_bounds(worker);
                    for (std::size_t row = 0U;
                         row < resources.size();
                         ++row) {
                        CounterResourceScan& scan =
                            scans[worker * resources.size() + row];
                        const AttrId value_id = resources[row].value_id;
                        for (std::size_t index = columns.first;
                             index < columns.second;
                             ++index) {
                            consume_value(
                                scan,
                                row,
                                index,
                                graph_->node_attrs(index).find(value_id));
                        }
                    }
                }
            });
    } else {
        run_counter_tasks(
            worker_count,
            worker_count,
            [&](std::size_t begin, std::size_t end) noexcept {
                for (std::size_t worker = begin; worker < end; ++worker) {
                    const auto columns = column_bounds(worker);
                    for (std::size_t row = 0U;
                         row < resources.size();
                         ++row) {
                        CounterResourceScan& scan =
                            scans[worker * resources.size() + row];
                        const AttrId value_id = resources[row].value_id;
                        for (std::size_t index = columns.first;
                             index < columns.second;
                             ++index) {
                            consume_value(
                                scan,
                                row,
                                index,
                                graph_->edge_attrs(ordered_edges[index])
                                    .find(value_id));
                        }
                    }
                }
            });
    }

    const CounterErrorCode missing_code = node
        ? CounterErrorCode::missing_node_resource_value
        : CounterErrorCode::missing_link_resource_value;
    for (std::size_t row = 0U; row < resources.size(); ++row) {
        bool missing = false;
        for (std::size_t worker = 0U; worker < worker_count; ++worker) {
            missing = missing ||
                scans[worker * resources.size() + row].missing;
        }
        if (missing) {
            throw CounterException(
                missing_code,
                operation,
                "Counter resource row is missing one or more graph values",
            resources[row].registry_id);
        }
    }
    for (std::size_t row = 0U; row < resources.size(); ++row) {
        bool nonnumeric = false;
        for (std::size_t worker = 0U; worker < worker_count; ++worker) {
            nonnumeric = nonnumeric ||
                scans[worker * resources.size() + row].nonnumeric;
        }
        if (nonnumeric) {
            throw CounterException(
                CounterErrorCode::non_numeric_resource_value,
                operation,
                "Counter resource matrix contains a nonnumeric value",
                resources[row].registry_id);
        }
    }

    bool has_double = false;
    std::uint64_t integer_bits = 0U;
    for (const CounterResourceScan& scan : scans) {
        has_double = has_double || scan.has_double;
        integer_bits += scan.integer_bits;
    }
    if (!has_double) {
        return signed_bits(integer_bits);
    }
    return numpy_pairwise_sum(flattened.get(), value_count);
}

CounterNumber PreparedCounter::calculate_sum_network_resource(
    bool node,
    bool link,
    CounterOptions options) const {
    CounterNumber result = std::int64_t{0};
    if (node) {
        result = calculate_sum_node_resource(options);
    }
    if (link) {
        result = add_numpy_numbers(result, calculate_sum_link_resource(options));
    }
    return result;
}

CounterNumber PreparedCounter::calculate_sum_node_resource(
    CounterOptions options) const {
    return calculate_resource_sum(true, options);
}

CounterNumber PreparedCounter::calculate_sum_link_resource(
    CounterOptions options) const {
    return calculate_resource_sum(false, options);
}

CounterNumber PreparedCounter::calculate_link_cost(
    const Solution& solution,
    CounterOperation operation) const {
    CounterNumber total = std::int64_t{0};
    for (const auto& route : solution.link_paths.entries()) {
        for (const SolutionLink& physical_link : route.value) {
            const LinkPathInfoKey key{route.key, physical_link};
            const auto info_id = solution.link_paths_info.find_id(key);
            if (!info_id.has_value()) {
                throw CounterException(
                    CounterErrorCode::missing_route_info,
                    operation,
                    "Solution physical route has no link_paths_info entry",
                    std::nullopt,
                    std::nullopt,
                    route.key,
                    physical_link);
            }
            const SolutionAttributeValues& info =
                solution.link_paths_info.at(*info_id);
            for (const ResourceBinding& resource : link_resources_) {
                const auto* value = info.find(resource.registry_id);
                if (value == nullptr) {
                    throw CounterException(
                        CounterErrorCode::missing_link_resource_value,
                        operation,
                        "Solution route info is missing a link resource",
                        resource.registry_id,
                        std::nullopt,
                        route.key,
                        physical_link);
                }
                total = add_checked_numbers(
                    total,
                    solution_numeric_value(
                        *value),
                    operation,
                    resource.registry_id,
                    std::nullopt,
                    route.key,
                    physical_link);
            }
        }
    }
    return total;
}

CounterNumber PreparedCounter::calculate_v_net_link_cost(
    const Solution& solution) const {
    validate_network_binding(CounterOperation::count_link_cost);
    return calculate_link_cost(solution, CounterOperation::count_link_cost);
}

CounterNumber PreparedCounter::calculate_v_net_cost(
    const Solution& solution,
    CounterOptions options) const {
    const CounterNumber node_cost = calculate_sum_node_resource(options);
    const CounterNumber link_cost = calculate_link_cost(
        solution,
        CounterOperation::count_link_cost);
    return add_checked_numbers(
        node_cost,
        link_cost,
        CounterOperation::count_link_cost);
}

CounterNumber PreparedCounter::calculate_v_net_revenue(
    CounterOptions options) const {
    return calculate_sum_network_resource(true, true, options);
}

void PreparedCounter::count_partial_solution(
    Solution& solution,
    CounterOptions options) const {
    (void)options;
    constexpr CounterOperation operation =
        CounterOperation::count_partial_solution;
    if (virtual_network_ == nullptr) {
        throw CounterException(
            CounterErrorCode::virtual_network_required,
            operation,
            "Solution mutation requires a Counter prepared from VirtualNetwork");
    }
    validate_network_binding(operation);

    CounterNumber raw_node_revenue = std::int64_t{0};
    for (const auto& slot : solution.node_slots.entries()) {
        const Vertex node = checked_virtual_node(*graph_, slot.key, operation);
        const AttrMap& attributes = graph_->node_attrs(node);
        for (const ResourceBinding& resource : node_resources_) {
            const AttrValue* value = attributes.find(resource.value_id);
            if (value == nullptr) {
                throw CounterException(
                    CounterErrorCode::missing_node_resource_value,
                    operation,
                    "Placed virtual node is missing a selected resource",
                    resource.registry_id,
                    slot.key);
            }
            raw_node_revenue = add_checked_numbers(
                raw_node_revenue,
                graph_numeric_value(
                    *value,
                    operation,
                    resource.registry_id,
                    slot.key),
                operation,
                resource.registry_id,
                slot.key);
        }
    }

    CounterNumber link_revenue = std::int64_t{0};
    for (const auto& route : solution.link_paths.entries()) {
        const auto endpoints =
            checked_virtual_link(*graph_, route.key, operation);
        const AttrMap& attributes =
            graph_->edge_attrs(graph_->edge(endpoints.first, endpoints.second));
        for (const ResourceBinding& resource : link_resources_) {
            const AttrValue* value = attributes.find(resource.value_id);
            if (value == nullptr) {
                throw CounterException(
                    CounterErrorCode::missing_link_resource_value,
                    operation,
                    "Routed virtual link is missing a selected resource",
                    resource.registry_id,
                    std::nullopt,
                    route.key);
            }
            link_revenue = add_checked_numbers(
                link_revenue,
                graph_numeric_value(
                    *value,
                    operation,
                    resource.registry_id,
                    std::nullopt,
                    route.key),
                operation,
                resource.registry_id,
                std::nullopt,
                route.key);
        }
    }

    const CounterNumber link_cost = calculate_link_cost(solution, operation);
    if (node_resources_.empty()) {
        throw CounterException(
            CounterErrorCode::empty_node_resource_selection,
            operation,
            "Partial Solution node normalization has zero resources");
    }

    const double normalized_node_revenue =
        counter_number_to_double(raw_node_revenue) /
        static_cast<double>(node_resources_.size());
    solution.v_net_node_revenue = normalized_node_revenue;
    solution.v_net_link_revenue = counter_number_to_double(link_revenue);
    solution.v_net_revenue = counter_number_to_double(add_checked_numbers(
        raw_node_revenue,
        link_revenue,
        operation));
    solution.v_net_link_cost = counter_number_to_double(link_cost);
    solution.v_net_path_cost =
        solution.v_net_link_cost - solution.v_net_link_revenue;
    solution.v_net_node_cost = normalized_node_revenue;
    solution.v_net_cost =
        solution.v_net_node_cost + solution.v_net_link_cost;
    solution.v_net_r2c_ratio = solution.v_net_cost == 0.0
        ? 0.0
        : solution.v_net_revenue / solution.v_net_cost;
}

void PreparedCounter::count_solution(
    Solution& solution,
    CounterOptions options) const {
    constexpr CounterOperation operation = CounterOperation::count_solution;
    if (virtual_network_ == nullptr) {
        throw CounterException(
            CounterErrorCode::virtual_network_required,
            operation,
            "Solution mutation requires a Counter prepared from VirtualNetwork");
    }
    validate_network_binding(operation);

    solution.num_placed_nodes = solution.node_slots.size();
    solution.num_routed_links = solution.link_paths.size();
    solution.v_net_node_demand =
        counter_number_to_double(calculate_sum_node_resource(options)) /
        static_cast<double>(node_resources_.size());
    solution.v_net_link_demand =
        counter_number_to_double(calculate_sum_link_resource(options));
    solution.v_net_demand =
        solution.v_net_node_demand + solution.v_net_demand;

    if (solution.result) {
        solution.place_result = true;
        solution.route_result = true;
        solution.early_rejection = false;
        solution.v_net_node_revenue = solution.v_net_node_demand;
        solution.v_net_link_revenue = solution.v_net_link_demand;
        solution.v_net_node_cost = solution.v_net_node_revenue;
        solution.v_net_link_cost = counter_number_to_double(
            calculate_link_cost(solution, operation));
        solution.v_net_path_cost =
            solution.v_net_link_cost - solution.v_net_link_revenue;
        solution.v_net_revenue =
            solution.v_net_node_revenue + solution.v_net_link_revenue;
        solution.v_net_cost =
            solution.v_net_revenue + solution.v_net_path_cost;
        solution.v_net_r2c_ratio = solution.v_net_cost == 0.0
            ? 0.0
            : solution.v_net_revenue / solution.v_net_cost;
    } else {
        solution.v_net_node_revenue = 0.0;
        solution.v_net_link_revenue = 0.0;
        solution.v_net_revenue = 0.0;
        solution.v_net_path_cost = 0.0;
        solution.v_net_cost = 0.0;
        solution.v_net_r2c_ratio = 0.0;
    }

    const auto& lifetime = virtual_network_->lifetime();
    if (!lifetime.has_value()) {
        throw CounterException(
            CounterErrorCode::missing_virtual_lifetime,
            operation,
            "VirtualNetwork lifetime is missing");
    }
    solution.v_net_time_revenue = solution.v_net_revenue * *lifetime;
    solution.v_net_time_cost = solution.v_net_cost * *lifetime;
    solution.v_net_time_rc_ratio = solution.v_net_r2c_ratio * *lifetime;
}

CounterSummary summary_records(const CounterRecords& records) {
    constexpr CounterOperation operation =
        CounterOperation::summarize_records;
    if (records.rows.empty()) {
        throw CounterException(
            CounterErrorCode::empty_records,
            operation,
            "Counter summary requires at least one record");
    }

    std::vector<double> arrival_ratios;
    std::vector<double> arrival_rewards;
    std::vector<double> physical_resources;
    std::vector<double> physical_node_resources;
    std::vector<double> physical_link_resources;
    std::vector<double> violations;
    std::vector<double> maximum_step_violations;
    arrival_ratios.reserve(records.rows.size());
    arrival_rewards.reserve(records.rows.size());
    physical_resources.reserve(records.rows.size());
    physical_node_resources.reserve(records.rows.size());
    physical_link_resources.reserve(records.rows.size());
    violations.reserve(records.rows.size());
    maximum_step_violations.reserve(records.rows.size());

    std::size_t early_rejection_count = 0U;
    std::size_t place_failure_count = 0U;
    std::size_t route_failure_count = 0U;
    std::int64_t maximum_inservice_count =
        std::numeric_limits<std::int64_t>::min();

    for (std::size_t row_index = 0U;
         row_index < records.rows.size();
         ++row_index) {
        const CounterRecord& row = records.rows[row_index];
        if (row.event_type != network::VirtualEventType::leave &&
            row.event_type != network::VirtualEventType::arrival) {
            throw CounterException(
                CounterErrorCode::invalid_record_value,
                operation,
                "Counter record contains an invalid event type",
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                row_index);
        }

        physical_resources.push_back(row.physical_available_resource);
        physical_node_resources.push_back(
            row.physical_node_available_resource);
        physical_link_resources.push_back(
            row.physical_link_available_resource);
        violations.push_back(row.hard_constraint_violation);
        maximum_step_violations.push_back(
            row.max_single_step_hard_constraint_violation);
        maximum_inservice_count =
            std::max(maximum_inservice_count, row.inservice_count);

        if (row.event_type == network::VirtualEventType::arrival) {
            arrival_ratios.push_back(row.v_net_r2c_ratio);
            early_rejection_count += row.early_rejection ? 1U : 0U;
            place_failure_count += row.place_result ? 0U : 1U;
            route_failure_count += row.route_result ? 0U : 1U;
            if (records.reward_column_present && row.reward.has_value()) {
                arrival_rewards.push_back(*row.reward);
            }
        }
    }

    const CounterRecord& final = records.rows.back();
    CounterSummary result;
    result.acceptance_rate =
        static_cast<double>(final.success_count) /
        static_cast<double>(final.virtual_network_count);
    result.average_r2c_ratio = mean_skip_nan(arrival_ratios);
    result.long_term_time_r2c_ratio =
        final.total_time_revenue / final.total_time_cost;
    result.long_term_average_time_revenue =
        final.total_time_revenue / final.virtual_network_arrival_time;
    result.success_count = final.success_count;
    result.early_rejection_count = early_rejection_count;
    result.place_failure_count = place_failure_count;
    result.route_failure_count = route_failure_count;
    result.total_cost = final.total_cost;
    result.total_revenue = final.total_revenue;
    result.total_time_revenue = final.total_time_revenue;
    result.total_time_cost = final.total_time_cost;
    result.long_term_r2c_ratio =
        final.total_revenue / final.total_cost;
    result.total_simulation_time = final.virtual_network_arrival_time;
    result.long_term_average_revenue =
        final.total_revenue / final.virtual_network_arrival_time;
    result.long_term_average_cost =
        final.total_cost / final.virtual_network_arrival_time;
    result.minimum_physical_available_resource =
        minimum_skip_nan(physical_resources);
    result.minimum_physical_node_available_resource =
        minimum_skip_nan(physical_node_resources);
    result.minimum_physical_link_available_resource =
        minimum_skip_nan(physical_link_resources);
    result.maximum_inservice_count = maximum_inservice_count;
    result.total_violation = sum_skip_nan(violations);
    result.total_max_single_step_violation =
        sum_skip_nan(maximum_step_violations);
    result.average_reward = records.reward_column_present
        ? mean_skip_nan(arrival_rewards)
        : 0.0;
    return result;
}

CounterSummary summary_csv(const std::string& path) {
    (void)csvio::read_csv(path);
    throw CounterException(
        CounterErrorCode::legacy_summary_csv_binding,
        CounterOperation::summarize_csv,
        "Legacy Counter.summary_csv reaches the unbound instance method");
}

}  // namespace virne::core
