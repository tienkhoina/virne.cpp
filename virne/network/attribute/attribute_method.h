#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace virne::network::attribute {

using AttributeDefinitionId = std::uint32_t;

enum class AttributeOwner : std::uint8_t {
    node,
    link,
    graph,
};

enum class AttributeKind : std::uint8_t {
    resource,
    extrema,
    status,
    position,
    latency,
};

enum class ResourceUpdateOperation : std::uint8_t {
    add,
    subtract,
};

enum class ComparisonOperation : std::uint8_t {
    greater_equal,
    less_equal,
    equal,
};

enum class ConstraintRestriction : std::uint8_t {
    hard,
    soft,
};

enum class AttributeNumberKind : std::uint8_t {
    boolean,
    integer,
    floating,
};

using AttributeNumber = std::variant<bool, std::int64_t, double>;

struct ResourceMethodSpec {
    AttributeDefinitionId definition_id = 0;
    bool generative = false;
};

enum class ExtremaOriginRegistry : std::uint8_t {
    node,
    link,
};

struct ExtremaMethodSpec {
    AttributeOwner declared_owner = AttributeOwner::node;
    ExtremaOriginRegistry origin_registry = ExtremaOriginRegistry::node;
    AttributeDefinitionId originator_id = 0;
};

struct InformationMethodSpec {
    static constexpr bool is_constraint = false;
};

struct ConstraintMethodSpec {
    static constexpr bool is_constraint = true;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
};

struct SatisfiabilityResult {
    bool flag = false;
    AttributeNumber offset = std::int64_t{0};
};

enum class AttributeMethodErrorCode : std::uint8_t {
    invalid_resource_state,
    unsupported_update_operation,
    insufficient_resource,
    non_generative_resource,
    missing_generator,
    missing_extrema_field,
    missing_originator,
    unsupported_comparison,
    invalid_restriction,
    invalid_numeric_type,
    numeric_range,
    invalid_batch_shape,
};

enum class AttributeMethodOperation : std::uint8_t {
    resolve_update,
    resolve_comparison,
    resolve_restriction,
    update_resource,
    generate_resource,
    resolve_extrema,
    generate_extrema,
    initialize_constraint,
    calculate_satisfiability,
    calculate_satisfiability_batch,
};

class AttributeMethodException : public std::runtime_error {
public:
    AttributeMethodException(
        AttributeMethodErrorCode code,
        AttributeMethodOperation operation,
        std::string message);

    AttributeMethodErrorCode code() const noexcept;
    AttributeMethodOperation operation() const noexcept;

private:
    AttributeMethodErrorCode code_;
    AttributeMethodOperation operation_;
};

ResourceUpdateOperation resource_update_operation_from_string(
    std::string_view value);

ComparisonOperation comparison_operation_from_string(std::string_view value);

ConstraintRestriction constraint_restriction_from_string(
    std::string_view value);

AttributeNumberKind attribute_number_kind(const AttributeNumber& value) noexcept;

bool update_resource_value(
    const AttributeNumber& virtual_value,
    AttributeNumber& physical_value,
    ResourceUpdateOperation operation,
    bool safe = true,
    std::string_view diagnostic_name = {});

SatisfiabilityResult calculate_satisfiability_values(
    const AttributeNumber& virtual_value,
    const AttributeNumber& physical_value,
    ComparisonOperation operation,
    ConstraintRestriction restriction);

std::size_t double_satisfiability_batch_worker_count(
    std::size_t count,
    std::size_t configured_workers = 1) noexcept;

void calculate_satisfiability_values_double_batch(
    const std::vector<double>& virtual_values,
    const std::vector<double>& physical_values,
    ComparisonOperation operation,
    ConstraintRestriction restriction,
    std::vector<std::uint8_t>& flags,
    std::vector<double>& offsets,
    std::size_t workers = 1);

}  // namespace virne::network::attribute
