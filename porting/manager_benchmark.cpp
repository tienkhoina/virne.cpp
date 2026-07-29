#include "manager.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using virne::utils::EmptyDirectoryConfig;
using virne::utils::ManagerErrorCode;
using virne::utils::ManagerException;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Metrics
{
    std::uint64_t elapsed_ns = 0;
    std::uint64_t semantic_checksum = fnv_offset;
    std::uint64_t output_checksum = fnv_offset;
    std::uint64_t output_bytes = 0;
    std::size_t operations = 0;
};

std::uint64_t fnv_update(
    std::uint64_t hash,
    std::string_view bytes) noexcept
{
    for (const char character : bytes)
    {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
    return hash;
}

std::uint64_t fnv_update_u64(
    std::uint64_t hash,
    std::uint64_t value) noexcept
{
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
    {
        hash ^= static_cast<unsigned char>(value >> shift);
        hash *= fnv_prime;
    }
    return hash;
}

fs::path checked_sandbox(std::string_view text)
{
    const fs::path sandbox =
        fs::absolute(fs::path(std::string(text))).lexically_normal();
    const fs::path temporary =
        fs::absolute(fs::temp_directory_path()).lexically_normal();
    const std::string name = sandbox.filename().string();
    if (sandbox.parent_path() != temporary ||
        name.rfind("virne_manager_bench_", 0) != 0 ||
        !fs::is_directory(sandbox) || !fs::is_empty(sandbox))
    {
        throw std::invalid_argument(
            "benchmark sandbox must be an empty direct temporary child");
    }
    return sandbox;
}

void write_file(const fs::path& path, std::string_view bytes = "x")
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
        throw std::runtime_error("unable to create manager benchmark fixture");
    }
}

std::string normalized_output(
    std::string output,
    const fs::path& sandbox)
{
    const std::string root = sandbox.string();
    std::size_t position = 0;
    while ((position = output.find(root, position)) != std::string::npos)
    {
        output.replace(position, root.size(), "<ROOT>");
        position += 6U;
    }
    return output;
}

std::uint64_t semantic_checksum(
    std::size_t operations,
    std::uint64_t marker) noexcept
{
    std::uint64_t hash = fnv_offset;
    for (std::size_t index = 0; index < operations; ++index)
    {
        hash = fnv_update_u64(hash, static_cast<std::uint64_t>(index));
        hash = fnv_update_u64(hash, marker);
    }
    return hash;
}

template <typename Callable>
std::uint64_t timed_nanoseconds(Callable&& callable)
{
    const auto started = std::chrono::steady_clock::now();
    callable();
    const auto stopped = std::chrono::steady_clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            stopped - started).count());
}

Metrics benchmark_temp_empty(
    const fs::path& sandbox,
    std::size_t operations)
{
    const fs::path target = sandbox / "target";
    fs::create_directory(target);
    Metrics metrics;
    metrics.elapsed_ns = timed_nanoseconds([&]
    {
        for (std::size_t index = 0; index < operations; ++index)
        {
            virne::utils::delete_temp_files(target);
        }
    });
    if (!fs::is_directory(target) || !fs::is_empty(target))
    {
        throw std::runtime_error("temp-empty benchmark changed its fixture");
    }
    metrics.semantic_checksum = semantic_checksum(operations, 1U);
    metrics.operations = operations;
    return metrics;
}

Metrics benchmark_temp_legacy(
    const fs::path& sandbox,
    std::size_t operations)
{
    const fs::path target = sandbox / "target";
    write_file(target / "file_temp", "payload");
    std::size_t caught = 0;
    Metrics metrics;
    metrics.elapsed_ns = timed_nanoseconds([&]
    {
        for (std::size_t index = 0; index < operations; ++index)
        {
            try
            {
                virne::utils::delete_temp_files(target);
            }
            catch (const ManagerException& error)
            {
                if (error.code() !=
                    ManagerErrorCode::legacy_temp_join_type_error)
                {
                    throw;
                }
                ++caught;
            }
        }
    });
    if (caught != operations || !fs::exists(target / "file_temp"))
    {
        throw std::runtime_error("temp legacy benchmark invariant failed");
    }
    metrics.semantic_checksum = semantic_checksum(operations, 2U);
    metrics.operations = operations;
    return metrics;
}

Metrics benchmark_clean_retained(
    const fs::path& sandbox,
    std::size_t operations)
{
    const fs::path save = sandbox / "save";
    constexpr std::size_t algorithm_count = 8;
    constexpr std::size_t runs_per_algorithm = 16;
    for (std::size_t algorithm = 0;
         algorithm < algorithm_count;
         ++algorithm)
    {
        for (std::size_t run = 0; run < runs_per_algorithm; ++run)
        {
            write_file(
                save / ("algorithm_" + std::to_string(algorithm)) /
                    ("run_" + std::to_string(run)) / "records" / "result");
        }
    }
    std::ostringstream output;
    Metrics metrics;
    metrics.elapsed_ns = timed_nanoseconds([&]
    {
        for (std::size_t index = 0; index < operations; ++index)
        {
            virne::utils::clean_save_dir(save, output);
        }
    });
    if (!output.str().empty())
    {
        throw std::runtime_error("retained scan emitted deletion output");
    }
    const std::size_t retained = algorithm_count * runs_per_algorithm;
    metrics.semantic_checksum = semantic_checksum(
        operations, static_cast<std::uint64_t>(retained));
    metrics.operations = operations;
    return metrics;
}

Metrics benchmark_clean_delete(
    const fs::path& sandbox,
    std::size_t operations,
    bool create_empty_records)
{
    std::vector<fs::path> saves;
    saves.reserve(operations);
    for (std::size_t index = 0; index < operations; ++index)
    {
        const fs::path save =
            sandbox / ("case_" + std::to_string(index)) / "save";
        const fs::path run = save / "algorithm" / "run";
        if (create_empty_records)
        {
            fs::create_directories(run / "records");
        }
        else
        {
            write_file(run / "models" / "weights");
        }
        saves.push_back(save);
    }

    std::ostringstream output;
    Metrics metrics;
    metrics.elapsed_ns = timed_nanoseconds([&]
    {
        for (const fs::path& save : saves)
        {
            virne::utils::clean_save_dir(save, output);
        }
    });
    for (const fs::path& save : saves)
    {
        if (fs::exists(save / "algorithm" / "run"))
        {
            throw std::runtime_error("destructive clean benchmark retained run");
        }
    }
    const std::string normalized = normalized_output(output.str(), sandbox);
    metrics.semantic_checksum = semantic_checksum(
        operations, create_empty_records ? 4U : 3U);
    metrics.output_checksum = fnv_update(fnv_offset, normalized);
    metrics.output_bytes = static_cast<std::uint64_t>(normalized.size());
    metrics.operations = operations;
    return metrics;
}

Metrics benchmark_empty_retained(
    const fs::path& sandbox,
    std::size_t operations)
{
    const fs::path record = sandbox / "record";
    const fs::path log = sandbox / "log";
    const fs::path save = sandbox / "save";
    write_file(record / "keep");
    write_file(log / "keep");
    write_file(save / "keep");
    const EmptyDirectoryConfig config{record, log, save};
    Metrics metrics;
    metrics.elapsed_ns = timed_nanoseconds([&]
    {
        for (std::size_t index = 0; index < operations; ++index)
        {
            virne::utils::delete_empty_dir(config);
        }
    });
    if (!fs::exists(record / "keep") ||
        !fs::exists(log / "keep") ||
        !fs::exists(save / "keep"))
    {
        throw std::runtime_error("retained empty-dir benchmark changed fixture");
    }
    metrics.semantic_checksum = semantic_checksum(operations, 3U);
    metrics.operations = operations;
    return metrics;
}

Metrics benchmark_empty_remove(
    const fs::path& sandbox,
    std::size_t operations)
{
    std::vector<EmptyDirectoryConfig> configs;
    configs.reserve(operations);
    for (std::size_t index = 0; index < operations; ++index)
    {
        const fs::path root = sandbox / ("case_" + std::to_string(index));
        const fs::path record = root / "record";
        const fs::path log = root / "log";
        const fs::path save = root / "save";
        fs::create_directories(record);
        fs::create_directory(log);
        fs::create_directory(save);
        configs.push_back({record, log, save});
    }
    Metrics metrics;
    metrics.elapsed_ns = timed_nanoseconds([&]
    {
        for (const EmptyDirectoryConfig& config : configs)
        {
            virne::utils::delete_empty_dir(config);
        }
    });
    for (const EmptyDirectoryConfig& config : configs)
    {
        if (fs::exists(config.record_dir) ||
            fs::exists(config.log_dir) ||
            fs::exists(config.save_dir))
        {
            throw std::runtime_error("empty removal benchmark retained directory");
        }
    }
    metrics.semantic_checksum = semantic_checksum(operations, 3U);
    metrics.operations = operations;
    return metrics;
}

Metrics run_kind(
    std::string_view kind,
    const fs::path& sandbox,
    std::size_t operations)
{
    if (kind == "temp_empty")
    {
        return benchmark_temp_empty(sandbox, operations);
    }
    if (kind == "temp_legacy")
    {
        return benchmark_temp_legacy(sandbox, operations);
    }
    if (kind == "clean_retained")
    {
        return benchmark_clean_retained(sandbox, operations);
    }
    if (kind == "clean_delete_missing")
    {
        return benchmark_clean_delete(sandbox, operations, false);
    }
    if (kind == "clean_delete_empty")
    {
        return benchmark_clean_delete(sandbox, operations, true);
    }
    if (kind == "empty_retained")
    {
        return benchmark_empty_retained(sandbox, operations);
    }
    if (kind == "empty_remove")
    {
        return benchmark_empty_remove(sandbox, operations);
    }
    throw std::invalid_argument("unknown manager benchmark kind");
}

std::size_t parse_operations(std::string_view text)
{
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(std::string(text), &consumed);
    if (consumed != text.size() || parsed == 0 ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("invalid operation count");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 4)
        {
            throw std::invalid_argument(
                "usage: vne_manager_benchmark <sandbox> <kind> <operations>");
        }
        const fs::path sandbox = checked_sandbox(argv[1]);
        const std::string_view kind(argv[2]);
        const std::size_t operations = parse_operations(argv[3]);
        const Metrics metrics = run_kind(kind, sandbox, operations);
        std::cout << "benchmark_version=1\n"
                  << "kind=" << kind << '\n'
                  << "operations=" << metrics.operations << '\n'
                  << "elapsed_ns=" << metrics.elapsed_ns << '\n'
                  << "semantic_checksum=" << metrics.semantic_checksum << '\n'
                  << "output_checksum=" << metrics.output_checksum << '\n'
                  << "output_bytes=" << metrics.output_bytes << '\n'
                  << "status=PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "manager benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
