#include "base_attribute.h"

#include "numpy_random_state.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>

namespace virne::network::attribute {
namespace {

constexpr std::string_view kDistributionHint =
    "You may initialize the attribute with a distribution key, e.g., "
    "'uniform', 'normal', 'exponential', 'poisson', or 'customized'.";

std::string_view distribution_name(virne::utils::DistributionKind kind) noexcept {
    using virne::utils::DistributionKind;
    switch (kind) {
        case DistributionKind::none:
            return "None";
        case DistributionKind::uniform:
            return "uniform";
        case DistributionKind::normal:
            return "normal";
        case DistributionKind::exponential:
            return "exponential";
        case DistributionKind::poisson:
            return "poisson";
        case DistributionKind::customized:
            return "customized";
    }
    return "unknown";
}

std::string_view value_kind_name(virne::utils::DatasetValueKind kind) noexcept {
    using virne::utils::DatasetValueKind;
    switch (kind) {
        case DatasetValueKind::integer:
            return "int";
        case DatasetValueKind::floating:
            return "float";
        case DatasetValueKind::boolean:
            return "bool";
    }
    return "unknown";
}

bool scalar_number(
    const virne::utils::DatasetScalar& value,
    double& result) noexcept {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        result = static_cast<double>(*integer);
        return true;
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        result = *floating;
        return true;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        result = *boolean ? 1.0 : 0.0;
        return true;
    }
    return false;
}

bool scalar_integral(const virne::utils::DatasetScalar& value) noexcept {
    return std::holds_alternative<std::int64_t>(value)
        || std::holds_alternative<bool>(value);
}

std::int64_t scalar_integer(
    const virne::utils::DatasetScalar& value) noexcept {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    return std::get<bool>(value) ? std::int64_t{1} : std::int64_t{0};
}

bool scalar_less(
    const virne::utils::DatasetScalar& lhs,
    const virne::utils::DatasetScalar& rhs) noexcept {
    if (scalar_integral(lhs) && scalar_integral(rhs)) {
        return scalar_integer(lhs) < scalar_integer(rhs);
    }
    double left = 0.0;
    double right = 0.0;
    if (!scalar_number(lhs, left) || !scalar_number(rhs, right)) {
        return false;
    }
    if (!scalar_integral(lhs) && !scalar_integral(rhs)) {
        return left < right;
    }

    constexpr double two_to_63 = 9223372036854775808.0;
    const auto integer_less_double = [&](std::int64_t integer, double floating) {
        if (std::isnan(floating)) {
            return false;
        }
        if (floating >= two_to_63) {
            return true;
        }
        if (floating < -two_to_63) {
            return false;
        }
        const std::int64_t truncated = static_cast<std::int64_t>(floating);
        if (integer != truncated) {
            return integer < truncated;
        }
        return floating > static_cast<double>(truncated);
    };
    if (scalar_integral(lhs)) {
        return integer_less_double(scalar_integer(lhs), right);
    }
    const std::int64_t integer = scalar_integer(rhs);
    if (std::isnan(left)) {
        return false;
    }
    if (left >= two_to_63) {
        return false;
    }
    if (left < -two_to_63) {
        return true;
    }
    const std::int64_t truncated = static_cast<std::int64_t>(left);
    if (truncated != integer) {
        return truncated < integer;
    }
    return left < static_cast<double>(truncated);
}

double customized_span(
    const virne::utils::DatasetScalar& minimum,
    const virne::utils::DatasetScalar& maximum,
    double minimum_value,
    double maximum_value) noexcept {
    if (scalar_integral(minimum) && scalar_integral(maximum)) {
        const std::uint64_t low =
            static_cast<std::uint64_t>(scalar_integer(minimum));
        const std::uint64_t high =
            static_cast<std::uint64_t>(scalar_integer(maximum));
        return static_cast<double>(high - low);
    }
    return maximum_value - minimum_value;
}

std::size_t configured_worker_count(
    std::size_t count,
    std::size_t configured_workers) noexcept {
    if (count == 0U || configured_workers <= 1U) {
        return 1U;
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    const std::size_t available = hardware == 0U
        ? configured_workers
        : static_cast<std::size_t>(hardware);
    return std::max(
        std::size_t{1U},
        std::min({configured_workers, count, available}));
}

void transform_customized(
    std::vector<double>& values,
    double span,
    double minimum,
    std::size_t configured_workers) {
    const std::size_t width =
        configured_worker_count(values.size(), configured_workers);
    const auto block = [&](std::size_t worker) noexcept {
        const std::size_t base = values.size() / width;
        const std::size_t remainder = values.size() % width;
        const std::size_t begin = worker * base + std::min(worker, remainder);
        const std::size_t end = begin + base + (worker < remainder ? 1U : 0U);
        for (std::size_t index = begin; index < end; ++index) {
            const double scaled = values[index] * span;
            values[index] = scaled + minimum;
        }
    };
    if (width == 1U) {
        block(0U);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(width - 1U);
    std::size_t launched = 1U;
    try {
        for (; launched < width; ++launched) {
            threads.emplace_back(block, launched);
        }
    } catch (...) {
        block(0U);
        for (std::size_t worker = launched; worker < width; ++worker) {
            block(worker);
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        return;
    }
    block(0U);
    for (std::thread& thread : threads) {
        thread.join();
    }
}

void append_optional(
    BaseAttributeSnapshot& snapshot,
    std::string name,
    const std::optional<virne::utils::DatasetScalar>& value) {
    if (value.has_value()) {
        snapshot.push_back({std::move(name), *value});
    }
}

BaseAttributeException unsupported_distribution(
    virne::utils::DistributionKind kind) {
    std::string message = "Distribution '";
    message.append(distribution_name(kind));
    message.append("' is not implemented.\n");
    message.append(kDistributionHint);
    return BaseAttributeException(
        BaseAttributeErrorCode::unsupported_distribution,
        BaseAttributeOperation::generate_configured_data,
        std::move(message));
}

}  // namespace

BaseAttributeException::BaseAttributeException(
    BaseAttributeErrorCode code,
    BaseAttributeOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

BaseAttributeErrorCode BaseAttributeException::code() const noexcept {
    return code_;
}

BaseAttributeOperation BaseAttributeException::operation() const noexcept {
    return operation_;
}

AttributeOwner attribute_owner_from_string(std::string_view value) {
    if (value == "node") {
        return AttributeOwner::node;
    }
    if (value == "link") {
        return AttributeOwner::link;
    }
    if (value == "graph") {
        return AttributeOwner::graph;
    }
    throw BaseAttributeException(
        BaseAttributeErrorCode::invalid_owner,
        BaseAttributeOperation::resolve_owner,
        "invalid attribute owner: " + std::string(value));
}

std::string_view attribute_owner_name(AttributeOwner value) noexcept {
    switch (value) {
        case AttributeOwner::node:
            return "node";
        case AttributeOwner::link:
            return "link";
        case AttributeOwner::graph:
            return "graph";
    }
    return "unknown";
}

AttributeKind attribute_kind_from_string(std::string_view value) {
    if (value == "resource") {
        return AttributeKind::resource;
    }
    if (value == "extrema") {
        return AttributeKind::extrema;
    }
    if (value == "status") {
        return AttributeKind::status;
    }
    if (value == "position") {
        return AttributeKind::position;
    }
    if (value == "latency") {
        return AttributeKind::latency;
    }
    throw BaseAttributeException(
        BaseAttributeErrorCode::invalid_kind,
        BaseAttributeOperation::resolve_kind,
        "invalid attribute kind: " + std::string(value));
}

std::string_view attribute_kind_name(AttributeKind value) noexcept {
    switch (value) {
        case AttributeKind::resource:
            return "resource";
        case AttributeKind::extrema:
            return "extrema";
        case AttributeKind::status:
            return "status";
        case AttributeKind::position:
            return "position";
        case AttributeKind::latency:
            return "latency";
    }
    return "unknown";
}

BaseAttribute::BaseAttribute(BaseAttributeSpec spec)
    : spec_(std::move(spec)) {}

const BaseAttributeSpec& BaseAttribute::spec() const noexcept {
    return spec_;
}

virne::utils::GeneratedData BaseAttribute::generate_data() const {
    throw BaseAttributeException(
        BaseAttributeErrorCode::not_implemented,
        BaseAttributeOperation::generate_data,
        "Subclasses must implement this method.");
}

void BaseAttribute::update_data() {
    throw BaseAttributeException(
        BaseAttributeErrorCode::not_implemented,
        BaseAttributeOperation::update_data,
        "Subclasses must implement this method.");
}

virne::utils::GeneratedData BaseAttribute::generate_configured_data(
    const NetworkCardinality& network,
    NumpyRandomState& rng,
    std::size_t workers) const {
    using virne::utils::DatasetValueKind;
    using virne::utils::DistributionKind;
    using virne::utils::GeneratedData;

    if (!spec_.generative) {
        throw BaseAttributeException(
            BaseAttributeErrorCode::not_generative,
            BaseAttributeOperation::generate_configured_data,
            "Attribute is not generative.");
    }
    const std::size_t count = spec_.owner == AttributeOwner::node
        ? network.num_nodes
        : network.num_links;

    if (spec_.distribution.kind == DistributionKind::customized) {
        const auto& minimum = spec_.distribution.minimum;
        const auto& maximum = spec_.distribution.maximum;
        double minimum_value = 0.0;
        double maximum_value = 0.0;
        if (!minimum.has_value() || !maximum.has_value()
            || !scalar_number(*minimum, minimum_value)
            || !scalar_number(*maximum, maximum_value)) {
            throw BaseAttributeException(
                BaseAttributeErrorCode::invalid_custom_range,
                BaseAttributeOperation::generate_configured_data,
                "Min and max must be numeric.");
        }
        if (!scalar_less(*minimum, *maximum)) {
            throw BaseAttributeException(
                BaseAttributeErrorCode::invalid_custom_range,
                BaseAttributeOperation::generate_configured_data,
                "Min must be less than max.");
        }
        std::vector<double> values = rng.uniform(0.0, 1.0, count);
        const double span = customized_span(
            *minimum, *maximum, minimum_value, maximum_value);
        transform_customized(values, span, minimum_value, workers);
        return GeneratedData{DatasetValueKind::floating, std::move(values)};
    }

    switch (spec_.distribution.kind) {
        case DistributionKind::uniform:
        case DistributionKind::normal:
        case DistributionKind::exponential:
        case DistributionKind::poisson:
            break;
        case DistributionKind::none:
        case DistributionKind::customized:
        default:
            throw unsupported_distribution(spec_.distribution.kind);
    }

    virne::utils::DistributionRequest request;
    request.count = count;
    request.value_kind = spec_.dtype.value_or(DatasetValueKind::floating);
    request.distribution = spec_.distribution;
    request.distribution.reciprocal = false;
    if (request.distribution.kind == DistributionKind::normal) {
        if (!request.distribution.loc.has_value()) {
            request.distribution.loc = virne::utils::DatasetScalar{std::monostate{}};
        }
        if (!request.distribution.scale.has_value()) {
            request.distribution.scale =
                virne::utils::DatasetScalar{std::monostate{}};
        }
    }
    return virne::utils::generate_data_with_distribution(request, rng, workers);
}

BaseAttributeSnapshot BaseAttribute::to_dict() const {
    BaseAttributeSnapshot snapshot;
    snapshot.reserve(15U);
    snapshot.push_back({"name", spec_.name});
    snapshot.push_back({"owner", std::string(attribute_owner_name(spec_.owner))});
    snapshot.push_back({"type", std::string(attribute_kind_name(spec_.kind))});
    snapshot.push_back({"generative", spec_.generative});

    if (spec_.distribution.kind != virne::utils::DistributionKind::none) {
        snapshot.push_back(
            {"distribution", std::string(distribution_name(spec_.distribution.kind))});
    }
    if (spec_.dtype.has_value()) {
        snapshot.push_back({"dtype", std::string(value_kind_name(*spec_.dtype))});
    }
    append_optional(snapshot, "low", spec_.distribution.low);
    append_optional(snapshot, "high", spec_.distribution.high);
    append_optional(snapshot, "loc", spec_.distribution.loc);
    append_optional(snapshot, "scale", spec_.distribution.scale);
    append_optional(snapshot, "lam", spec_.distribution.lambda);
    append_optional(snapshot, "min", spec_.distribution.minimum);
    append_optional(snapshot, "max", spec_.distribution.maximum);
    if (spec_.originator.has_value()) {
        snapshot.push_back({"originator", *spec_.originator});
    }
    if (spec_.is_constraint.has_value()) {
        snapshot.push_back({"is_constraint", *spec_.is_constraint});
    }
    return snapshot;
}

std::string BaseAttribute::repr(std::string_view class_name) const {
    const BaseAttributeSnapshot snapshot = to_dict();
    std::string result(class_name);
    result.push_back('(');
    for (std::size_t index = 0U; index < snapshot.size(); ++index) {
        if (index != 0U) {
            result.append(", ");
        }
        result.append(snapshot[index].name);
        result.push_back('=');
        result.append(virne::utils::format_dataset_scalar(snapshot[index].value));
    }
    result.push_back(')');
    return result;
}

}  // namespace virne::network::attribute
