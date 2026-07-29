#pragma once

#include "attribute_method.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace virne::network::attribute {

using AttributeBenchmarkId = std::uint32_t;

struct AttributeBenchmarkDescriptor {
    AttributeDefinitionId definition_id = 0U;
    AttributeKind kind = AttributeKind::status;
    std::string name;
    std::optional<std::string> originator_name;
};

struct AttributeBenchmarkMatrix {
    std::size_t rows = 0U;
    std::size_t columns = 0U;
    std::vector<float> values;
};

struct PreparedAttributeBenchmarkData {
    std::vector<AttributeBenchmarkDescriptor> attributes;
    AttributeBenchmarkMatrix matrix;
    bool extrema_requested = false;
    std::size_t column_repetitions = 1U;
};

struct AttributeBenchmarkEntry {
    std::string name;
    double value = 0.0;
};

enum class AttributeBenchmarkErrorCode : std::uint8_t {
    invalid_matrix_shape,
    invalid_column_repetitions,
    empty_retained_row,
    benchmark_id_range,
    invalid_benchmark_id,
};

enum class AttributeBenchmarkOperation : std::uint8_t {
    validate_prepared_data,
    reduce_rows,
    build_map,
    access_map,
};

class AttributeBenchmarkException : public std::runtime_error {
public:
    AttributeBenchmarkException(
        AttributeBenchmarkErrorCode code,
        AttributeBenchmarkOperation operation,
        std::string message);

    AttributeBenchmarkErrorCode code() const noexcept;
    AttributeBenchmarkOperation operation() const noexcept;

private:
    AttributeBenchmarkErrorCode code_;
    AttributeBenchmarkOperation operation_;
};

class AttributeBenchmarkMap {
public:
    AttributeBenchmarkMap() = default;
    AttributeBenchmarkMap(const AttributeBenchmarkMap& other);
    AttributeBenchmarkMap& operator=(const AttributeBenchmarkMap& other);
    AttributeBenchmarkMap(AttributeBenchmarkMap&&) noexcept = default;
    AttributeBenchmarkMap& operator=(AttributeBenchmarkMap&&) noexcept = default;

    std::optional<AttributeBenchmarkId> bind(std::string_view name) const;
    const AttributeBenchmarkEntry& at(AttributeBenchmarkId id) const;
    const double* find(std::string_view name) const;
    const std::vector<AttributeBenchmarkEntry>& entries() const noexcept;

private:
    void reserve(std::size_t size);
    void insert_or_assign(std::string_view name, double value);
    void rebuild_index();

    std::vector<AttributeBenchmarkEntry> entries_;
    std::unordered_map<std::string_view, AttributeBenchmarkId> ids_;

    friend AttributeBenchmarkMap get_attr_benchmarks(
        const PreparedAttributeBenchmarkData& data,
        std::size_t workers);
};

struct AttributeBenchmarks {
    std::optional<AttributeBenchmarkMap> node_attr_benchmarks;
    std::optional<AttributeBenchmarkMap> link_attr_benchmarks;
    std::optional<AttributeBenchmarkMap> link_sum_attr_benchmarks;
};

struct AttributeBenchmarkRequest {
    std::optional<PreparedAttributeBenchmarkData> node;
    std::optional<PreparedAttributeBenchmarkData> link;
    std::optional<PreparedAttributeBenchmarkData> link_sum;
    std::size_t workers = 1U;
};

AttributeBenchmarkMap get_attr_benchmarks(
    const PreparedAttributeBenchmarkData& data,
    std::size_t workers = 1U);

class AttributeBenchmarkManager {
public:
    explicit AttributeBenchmarkManager(const AttributeBenchmarkRequest& request);

    const AttributeBenchmarks& benchmarks() const noexcept;

    static AttributeBenchmarks get_benchmarks(
        const AttributeBenchmarkRequest& request);

    static void add_to_cache(
        std::string key,
        std::shared_ptr<AttributeBenchmarks> value);

    static std::shared_ptr<AttributeBenchmarks> get_from_cache(
        std::string_view key);

    static void clear_cache();
    static std::size_t cache_size();

private:
    AttributeBenchmarks benchmarks_;
};

}  // namespace virne::network::attribute
