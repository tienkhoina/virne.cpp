#include "attribute/attribute_benchmark_manager.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;

constexpr std::uint64_t fnv_offset = UINT64_C(14695981039346656037);
constexpr std::uint64_t fnv_prime = UINT64_C(1099511628211);

struct Fingerprint {
    std::uint64_t checksum = fnv_offset;
    std::size_t output_bytes = 0U;
};

struct Result {
    std::uint64_t elapsed_ns = 0U;
    std::uint64_t checksum = 0U;
    std::size_t output_bytes = 0U;
    std::size_t entry_count = 0U;
};

void fnv_byte(std::uint64_t& checksum, unsigned char byte) noexcept {
    checksum ^= byte;
    checksum *= fnv_prime;
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

Fingerprint fingerprint(const attribute::AttributeBenchmarkMap& values) {
    Fingerprint result;
    for (const auto& entry : values.entries()) {
        for (const char character : entry.name) {
            fnv_byte(
                result.checksum,
                static_cast<unsigned char>(character));
        }
        fnv_byte(result.checksum, 0U);
        const std::uint64_t bits = double_bits(entry.value);
        for (std::size_t shift = 56U;; shift -= 8U) {
            fnv_byte(
                result.checksum,
                static_cast<unsigned char>((bits >> shift) & UINT64_C(0xff)));
            if (shift == 0U) {
                break;
            }
        }
        result.output_bytes += entry.name.size() + 1U + sizeof(double);
    }
    return result;
}

float input_value(std::size_t row, std::size_t column) noexcept {
    const std::size_t residue = (row * 131U + column * 17U) % 2048U;
    const auto centered = static_cast<std::int64_t>(residue) - 1024;
    return static_cast<float>(centered) * 0.125F;
}

attribute::PreparedAttributeBenchmarkData make_fixture(
    std::size_t rows,
    std::size_t columns) {
    if (rows != 0U &&
        columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::invalid_argument("benchmark matrix size overflow");
    }
    attribute::PreparedAttributeBenchmarkData data;
    data.attributes.reserve(rows);
    data.matrix.rows = rows;
    data.matrix.columns = columns;
    data.matrix.values.resize(rows * columns);
    data.extrema_requested = false;
    data.column_repetitions = 2U;
    for (std::size_t row = 0U; row < rows; ++row) {
        data.attributes.push_back(attribute::AttributeBenchmarkDescriptor{
            static_cast<attribute::AttributeDefinitionId>(row),
            attribute::AttributeKind::status,
            "metric_" + std::to_string(row),
            std::nullopt});
        for (std::size_t column = 0U; column < columns; ++column) {
            data.matrix.values[row * columns + column] =
                input_value(row, column);
        }
    }
    return data;
}

Result prepared_row_maxima(
    std::size_t rows,
    std::size_t columns,
    std::size_t workers) {
    const auto data = make_fixture(rows, columns);
    const auto begin = std::chrono::steady_clock::now();
    const auto output = attribute::get_attr_benchmarks(data, workers);
    const auto end = std::chrono::steady_clock::now();
    const Fingerprint encoded = fingerprint(output);
    return Result{
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count()),
        encoded.checksum,
        encoded.output_bytes,
        output.entries().size()};
}

std::size_t parse_size(const char* value, const char* name) {
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (value[consumed] != '\0' ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<std::size_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: vne_attribute_benchmark_manager_benchmark "
                "ROWS COLUMNS WORKERS");
        }
        const std::size_t rows = parse_size(argv[1], "rows");
        const std::size_t columns = parse_size(argv[2], "columns");
        const std::size_t workers = parse_size(argv[3], "workers");
        if (rows == 0U || columns == 0U) {
            throw std::invalid_argument("rows and columns must be positive");
        }
        const Result result = prepared_row_maxima(rows, columns, workers);
        std::cout << "protocol=1\n"
                  << "kind=prepared_row_maxima\n"
                  << "rows=" << rows << '\n'
                  << "columns=" << columns << '\n'
                  << "column_repetitions=2\n"
                  << "workers=" << workers << '\n'
                  << "type_tag=ordered_utf8_raw64\n"
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "entry_count=" << result.entry_count << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attribute_benchmark_manager_benchmark: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
