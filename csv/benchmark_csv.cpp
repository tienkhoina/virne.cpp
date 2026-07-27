#include "csv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

template <typename Function>
double median_ms(Function&& function)
{
    std::array<double, 5> values{};
    for (auto& value : values)
    {
        const auto start = std::chrono::steady_clock::now();
        function();
        value = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count();
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main()
{
    using namespace csvio;
    constexpr std::size_t row_count = 200'000;
    const auto file = std::filesystem::temp_directory_path() /
                      "virne-csv-benchmark.csv";

    DataFrame data;
    data.columns = {"id", "accepted", "score", "node_slots", "link_paths"};
    data.rows.reserve(row_count);
    for (std::size_t i = 0; i < row_count; ++i)
    {
        data.rows.push_back({
            std::to_string(i),
            i % 3 == 0 ? "true" : "false",
            std::to_string(static_cast<double>(i) / 17.0),
            "{0: 4, 1: 9, 2: 12}",
            "[(0, 1), (1, 7), (7, 9)]"});
    }

    const double write_ms = median_ms([&] { write_csv(file.string(), data); });
    const auto byte_count = std::filesystem::file_size(file);
    std::size_t checksum = 0;
    const double read_ms = median_ms([&] {
        const auto loaded = read_csv(file.string());
        checksum += loaded.nrows();
    });

    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const double mib = static_cast<double>(byte_count) / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(2)
              << "rows=" << row_count << " bytes=" << byte_count
              << " samples=5 statistic=median checksum=" << checksum << '\n'
              << "operation,elapsed_ms,rows_per_second,mib_per_second\n"
              << "write," << write_ms << ',' << row_count * 1000.0 / write_ms
              << ',' << mib * 1000.0 / write_ms << '\n'
              << "read," << read_ms << ',' << row_count * 1000.0 / read_ms
              << ',' << mib * 1000.0 / read_ms << '\n';
    return 0;
}
